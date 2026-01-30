# MIT License
#
# Copyright (c) 2026
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "`pyannote.audio.models.embedding.wespeaker.flax_nnx` requires `jax` and `flax`.\n"
        "Install them with e.g. `pip install jax flax`."
    ) from exc


def _pad2d(padding: int):
    # nnx.Conv accepts padding="SAME"/"VALID" or explicit per-axis pairs.
    return ((padding, padding), (padding, padding))


class StatsPool(nnx.Module):
    """Statistics pooling (mean + unbiased std) with optional weights.

    Port of `pyannote.audio.models.blocks.pooling.StatsPool` for JAX.

    Notes
    -----
    - `sequences` must be shaped as (batch, features, frames)
    - `weights` can be (batch, frames) or (batch, speakers, frames)
    """

    def __call__(
        self, sequences: jax.Array, weights: Optional[jax.Array] = None
    ) -> jax.Array:
        if weights is None:
            mean = sequences.mean(axis=-1)
            std = sequences.std(axis=-1, ddof=1)
            return jnp.concatenate([mean, std], axis=1)

        squeeze_speaker_dim = False
        if weights.ndim == 2:
            squeeze_speaker_dim = True
            weights = weights[:, None, :]

        # (B, S, T_w)
        weights = weights.astype(sequences.dtype)

        b, f, t = sequences.shape
        _, s, t_w = weights.shape

        if t_w != t:
            # Nearest-neighbor resize to match `sequences` frames.
            weights = jax.image.resize(weights, (b, s, t), method="nearest")

        # Compute weighted mean and unbiased std per speaker.
        # sequences: (B, 1, F, T)
        seq = sequences[:, None, :, :]
        w = weights[:, :, None, :]  # (B, S, 1, T)

        v1 = w.sum(axis=-1) + 1e-8  # (B, S, 1)
        mean = (seq * w).sum(axis=-1) / v1  # (B, S, F)

        dx2 = jnp.square(seq - mean[..., None])  # (B, S, F, T)
        v2 = jnp.square(w).sum(axis=-1)  # (B, S, 1)

        denom = v1 - v2 / v1 + 1e-8
        var = (dx2 * w).sum(axis=-1) / denom  # (B, S, F)
        std = jnp.sqrt(var)

        out = jnp.concatenate([mean, std], axis=-1)  # (B, S, 2F)
        if squeeze_speaker_dim:
            out = out[:, 0, :]
        return out


class BasicBlock(nnx.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            in_features=in_planes,
            out_features=planes,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding=_pad2d(1),
            use_bias=False,
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(
            num_features=planes,
            axis=-1,
            epsilon=1e-5,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=planes,
            out_features=planes,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=_pad2d(1),
            use_bias=False,
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(
            num_features=planes,
            axis=-1,
            epsilon=1e-5,
            rngs=rngs,
        )

        # Optional modules must be registered as pytree data upfront so we can
        # conditionally overwrite them with Conv/BatchNorm modules.
        self.shortcut_conv = nnx.data(None)
        self.shortcut_bn = nnx.data(None)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut_conv = nnx.data(
                nnx.Conv(
                    in_features=in_planes,
                    out_features=self.expansion * planes,
                    kernel_size=(1, 1),
                    strides=(stride, stride),
                    padding="VALID",
                    use_bias=False,
                    rngs=rngs,
                )
            )
            self.shortcut_bn = nnx.data(
                nnx.BatchNorm(
                    num_features=self.expansion * planes,
                    axis=-1,
                    epsilon=1e-5,
                    rngs=rngs,
                )
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        out = nnx.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        residual = x
        if self.shortcut_conv is not None and self.shortcut_bn is not None:
            residual = self.shortcut_bn(self.shortcut_conv(x))

        out = nnx.relu(out + residual)
        return out


class ResNet(nnx.Module):
    """WeSpeaker ResNet backbone (JAX/Flax NNX).

    This is a minimal port of `pyannote.audio.models.embedding.wespeaker.resnet.ResNet`
    with support for ResNet34 (BasicBlock).
    """

    def __init__(
        self,
        num_blocks: list[int],
        *,
        m_channels: int = 32,
        feat_dim: int = 80,
        embed_dim: int = 256,
        two_emb_layer: bool = False,
        rngs: nnx.Rngs,
    ):
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.two_emb_layer = two_emb_layer

        self.stats_dim = int(feat_dim / 8) * m_channels * 8

        self.conv1 = nnx.Conv(
            in_features=1,
            out_features=m_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=_pad2d(1),
            use_bias=False,
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(
            num_features=m_channels,
            axis=-1,
            epsilon=1e-5,
            rngs=rngs,
        )

        self.layer1 = self._make_layer(m_channels, num_blocks[0], stride=1, rngs=rngs)
        self.layer2 = self._make_layer(
            m_channels * 2, num_blocks[1], stride=2, rngs=rngs
        )
        self.layer3 = self._make_layer(
            m_channels * 4, num_blocks[2], stride=2, rngs=rngs
        )
        self.layer4 = self._make_layer(
            m_channels * 8, num_blocks[3], stride=2, rngs=rngs
        )

        self.pool = StatsPool()
        self.seg_1 = nnx.Linear(
            in_features=self.stats_dim * 2,
            out_features=embed_dim,
            rngs=rngs,
        )

        if two_emb_layer:
            self.seg_bn_1 = nnx.BatchNorm(
                num_features=embed_dim,
                axis=-1,
                epsilon=1e-5,
                use_scale=False,
                use_bias=False,
                rngs=rngs,
            )
            self.seg_2 = nnx.Linear(
                in_features=embed_dim,
                out_features=embed_dim,
                rngs=rngs,
            )
        else:
            self.seg_bn_1 = None
            self.seg_2 = None

    def _make_layer(
        self, planes: int, num_blocks: int, stride: int, *, rngs: nnx.Rngs
    ) -> nnx.List:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks: list[BasicBlock] = []
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, s, rngs=rngs))
            self.in_planes = planes * BasicBlock.expansion
        return nnx.List(blocks)

    def __call__(
        self, fbank: jax.Array, weights: Optional[jax.Array] = None
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Forward pass.

        Parameters
        ----------
        fbank:
            (batch, frames, features) fbank features (same as PyTorch).
        weights:
            Optional weights passed to stats pooling.
        """

        # Match the PyTorch implementation:
        # (B, T, F) -> (B, F, T, 1)
        x = jnp.transpose(fbank, (0, 2, 1))[..., None]

        out = nnx.relu(self.bn1(self.conv1(x)))
        for block in self.layer1:
            out = block(out)
        for block in self.layer2:
            out = block(out)
        for block in self.layer3:
            out = block(out)
        for block in self.layer4:
            out = block(out)

        # Pool over time (width) and flatten frequency+channel as features.
        b, h, w, c = out.shape
        seq = jnp.transpose(out, (0, 3, 1, 2)).reshape(b, c * h, w)
        stats = self.pool(seq, weights=weights)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer and self.seg_bn_1 is not None and self.seg_2 is not None:
            y = nnx.relu(embed_a)
            y = self.seg_bn_1(y)
            embed_b = self.seg_2(y)
            return embed_a, embed_b

        return embed_a


def ResNet34(*, feat_dim: int, embed_dim: int, rngs: nnx.Rngs) -> ResNet:
    return ResNet([3, 4, 6, 3], feat_dim=feat_dim, embed_dim=embed_dim, rngs=rngs)


def _torch_tensor_to_jax(x) -> jax.Array:
    return jnp.asarray(np.asarray(x.detach().cpu()))


def _copy_conv2d(torch_conv, flax_conv: nnx.Conv):
    w = _torch_tensor_to_jax(torch_conv.weight)
    # PyTorch conv2d: (out, in, kh, kw) -> NNX Conv: (kh, kw, in, out)
    flax_conv.kernel[...] = jnp.transpose(w, (2, 3, 1, 0))


def _copy_linear(torch_linear, flax_linear: nnx.Linear):
    w = _torch_tensor_to_jax(torch_linear.weight)
    b = _torch_tensor_to_jax(torch_linear.bias)
    # PyTorch Linear: (out, in) -> NNX Linear: (in, out)
    flax_linear.kernel[...] = jnp.transpose(w, (1, 0))
    flax_linear.bias[...] = b


def _copy_batchnorm(torch_bn, flax_bn: nnx.BatchNorm):
    # Running stats
    flax_bn.mean[...] = _torch_tensor_to_jax(torch_bn.running_mean)
    flax_bn.var[...] = _torch_tensor_to_jax(torch_bn.running_var)

    # Affine params (optional in NNX)
    if hasattr(flax_bn, "scale") and torch_bn.weight is not None:
        flax_bn.scale[...] = _torch_tensor_to_jax(torch_bn.weight)
    if hasattr(flax_bn, "bias") and torch_bn.bias is not None:
        flax_bn.bias[...] = _torch_tensor_to_jax(torch_bn.bias)


def resnet34_from_torch(torch_resnet, *, rngs: Optional[nnx.Rngs] = None) -> ResNet:
    """Convert a PyTorch WeSpeaker ResNet34 to a Flax NNX ResNet34.

    Parameters
    ----------
    torch_resnet:
        Instance of `pyannote.audio.models.embedding.wespeaker.resnet.ResNet` (PyTorch).
    rngs:
        Optional NNX RNG container used to init the module before overwriting weights.
    """

    rngs = rngs or nnx.Rngs(0)
    model = ResNet34(feat_dim=torch_resnet.feat_dim, embed_dim=torch_resnet.embed_dim, rngs=rngs)
    model.eval()

    _copy_conv2d(torch_resnet.conv1, model.conv1)
    _copy_batchnorm(torch_resnet.bn1, model.bn1)

    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        torch_layer = getattr(torch_resnet, layer_name)
        flax_layer = getattr(model, layer_name)
        for torch_block, flax_block in zip(torch_layer, flax_layer, strict=True):
            _copy_conv2d(torch_block.conv1, flax_block.conv1)
            _copy_batchnorm(torch_block.bn1, flax_block.bn1)
            _copy_conv2d(torch_block.conv2, flax_block.conv2)
            _copy_batchnorm(torch_block.bn2, flax_block.bn2)

            if len(torch_block.shortcut) > 0:
                assert flax_block.shortcut_conv is not None
                assert flax_block.shortcut_bn is not None
                _copy_conv2d(torch_block.shortcut[0], flax_block.shortcut_conv)
                _copy_batchnorm(torch_block.shortcut[1], flax_block.shortcut_bn)

    _copy_linear(torch_resnet.seg_1, model.seg_1)

    if getattr(torch_resnet, "two_emb_layer", False):
        if model.seg_bn_1 is None or model.seg_2 is None:
            raise ValueError("Expected a two-layer embedding NNX model.")
        _copy_batchnorm(torch_resnet.seg_bn_1, model.seg_bn_1)
        _copy_linear(torch_resnet.seg_2, model.seg_2)

    return model


def resnet34_from_pretrained(
    checkpoint: str = "pyannote/wespeaker-voxceleb-resnet34-LM",
    *,
    token: str | bool | None = None,
    cache_dir: str | None = None,
    rngs: Optional[nnx.Rngs] = None,
) -> ResNet:
    """Download a WeSpeaker ResNet34 (PyTorch) checkpoint and convert to Flax NNX.

    This is a convenience wrapper around `pyannote.audio.Model.from_pretrained`.
    """

    from pyannote.audio import Model  # local import: optional deps

    torch_model = Model.from_pretrained(checkpoint, token=token, cache_dir=cache_dir)
    if torch_model is None:
        raise ValueError(f"Could not load checkpoint: {checkpoint!r}")
    return resnet34_from_torch(torch_model.resnet.eval(), rngs=rngs)
