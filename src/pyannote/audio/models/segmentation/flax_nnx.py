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
        "`pyannote.audio.models.segmentation.flax_nnx` requires `jax` and `flax`.\n"
        "Install them with e.g. `pip install jax flax`."
    ) from exc


class InstanceNorm1d(nnx.Module):
    """InstanceNorm1d (per-example, per-channel) with affine parameters.

    This matches PyTorch's `torch.nn.InstanceNorm1d(..., track_running_stats=False)`
    behavior used by `pyannote.audio.models.blocks.sincnet.SincNet`.

    Notes
    -----
    - Expects input shaped as (batch, time, channels) (channels last).
    """

    def __init__(self, num_features: int, *, eps: float = 1e-5):
        self.num_features = num_features
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((num_features,), dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros((num_features,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.ndim != 3:
            raise ValueError(f"Expected (batch, time, channels) input, got {x.shape}.")

        mean = x.mean(axis=1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=1, keepdims=True)  # biased var
        y = (x - mean) / jnp.sqrt(var + self.eps)
        return y * self.weight[None, None, :] + self.bias[None, None, :]


def _torch_tensor_to_jax(x) -> jax.Array:
    return jnp.asarray(np.asarray(x.detach().cpu()))


def _copy_conv1d(torch_conv, flax_conv: nnx.Conv):
    w = _torch_tensor_to_jax(torch_conv.weight)
    # PyTorch conv1d: (out, in, k) -> NNX Conv1d: (k, in, out)
    flax_conv.kernel[...] = jnp.transpose(w, (2, 1, 0))
    if getattr(torch_conv, "bias", None) is not None and hasattr(flax_conv, "bias"):
        flax_conv.bias[...] = _torch_tensor_to_jax(torch_conv.bias)


def _copy_linear(torch_linear, flax_linear: nnx.Linear):
    w = _torch_tensor_to_jax(torch_linear.weight)
    b = _torch_tensor_to_jax(torch_linear.bias)
    # PyTorch Linear: (out, in) -> NNX Linear: (in, out)
    flax_linear.kernel[...] = jnp.transpose(w, (1, 0))
    flax_linear.bias[...] = b


def _copy_instance_norm1d(torch_norm, flax_norm: InstanceNorm1d):
    flax_norm.weight[...] = _torch_tensor_to_jax(torch_norm.weight)
    flax_norm.bias[...] = _torch_tensor_to_jax(torch_norm.bias)


def _copy_lstm_cell_from_torch(
    torch_lstm, *, layer_index: int, reverse: bool, flax_cell: nnx.LSTMCell
):
    suffix = "_reverse" if reverse else ""
    w_ih = getattr(torch_lstm, f"weight_ih_l{layer_index}{suffix}")
    w_hh = getattr(torch_lstm, f"weight_hh_l{layer_index}{suffix}")
    b_ih = getattr(torch_lstm, f"bias_ih_l{layer_index}{suffix}")
    b_hh = getattr(torch_lstm, f"bias_hh_l{layer_index}{suffix}")

    hidden = torch_lstm.hidden_size
    w_ih = w_ih.detach().cpu().numpy()
    w_hh = w_hh.detach().cpu().numpy()
    b = (b_ih + b_hh).detach().cpu().numpy()

    # Gate order matches PyTorch: i, f, g, o.
    wi_i, wi_f, wi_g, wi_o = np.split(w_ih, 4, axis=0)
    wh_i, wh_f, wh_g, wh_o = np.split(w_hh, 4, axis=0)
    b_i, b_f, b_g, b_o = np.split(b, 4, axis=0)

    flax_cell.ii.kernel[...] = jnp.asarray(wi_i.T)
    flax_cell.if_.kernel[...] = jnp.asarray(wi_f.T)
    flax_cell.ig.kernel[...] = jnp.asarray(wi_g.T)
    flax_cell.io.kernel[...] = jnp.asarray(wi_o.T)

    flax_cell.hi.kernel[...] = jnp.asarray(wh_i.T)
    flax_cell.hf.kernel[...] = jnp.asarray(wh_f.T)
    flax_cell.hg.kernel[...] = jnp.asarray(wh_g.T)
    flax_cell.ho.kernel[...] = jnp.asarray(wh_o.T)

    flax_cell.hi.bias[...] = jnp.asarray(b_i)
    flax_cell.hf.bias[...] = jnp.asarray(b_f)
    flax_cell.hg.bias[...] = jnp.asarray(b_g)
    flax_cell.ho.bias[...] = jnp.asarray(b_o)


class SincNet(nnx.Module):
    """SincNet feature extractor (JAX/Flax NNX).

    This is a minimal inference-focused port of
    `pyannote.audio.models.blocks.sincnet.SincNet`.
    """

    def __init__(self, *, stride: int, rngs: nnx.Rngs):
        self.stride = stride
        self.wav_norm1d = InstanceNorm1d(1)

        # Layer 0: ParamSincFB encoder weights are copied from PyTorch at conversion time.
        self.conv0 = nnx.Conv(
            in_features=1,
            out_features=80,
            kernel_size=(251,),
            strides=(stride,),
            padding="VALID",
            use_bias=False,
            rngs=rngs,
        )
        self.norm0 = InstanceNorm1d(80)

        self.conv1 = nnx.Conv(
            in_features=80,
            out_features=60,
            kernel_size=(5,),
            strides=(1,),
            padding="VALID",
            rngs=rngs,
        )
        self.norm1 = InstanceNorm1d(60)

        self.conv2 = nnx.Conv(
            in_features=60,
            out_features=60,
            kernel_size=(5,),
            strides=(1,),
            padding="VALID",
            rngs=rngs,
        )
        self.norm2 = InstanceNorm1d(60)

    def __call__(self, waveforms: jax.Array) -> jax.Array:
        # waveforms: (batch, channel, sample) -> (batch, sample, channel)
        if waveforms.ndim != 3:
            raise ValueError(f"Expected (batch, channel, sample) input, got {waveforms.shape}.")
        x = jnp.transpose(waveforms, (0, 2, 1))

        x = self.wav_norm1d(x)

        x = self.conv0(x)
        x = jnp.abs(x)
        x = nnx.max_pool(x, window_shape=(3,), strides=(3,), padding="VALID")
        x = nnx.leaky_relu(self.norm0(x))

        x = self.conv1(x)
        x = nnx.max_pool(x, window_shape=(3,), strides=(3,), padding="VALID")
        x = nnx.leaky_relu(self.norm1(x))

        x = self.conv2(x)
        x = nnx.max_pool(x, window_shape=(3,), strides=(3,), padding="VALID")
        x = nnx.leaky_relu(self.norm2(x))

        # Return in PyTorch layout: (batch, features, frames)
        return jnp.transpose(x, (0, 2, 1))


class BiLSTMLayer(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, *, rngs: nnx.Rngs):
        self.fwd = nnx.RNN(
            nnx.LSTMCell(in_features=in_features, hidden_features=hidden_features, rngs=rngs),
            reverse=False,
            keep_order=False,
            rngs=rngs,
        )
        self.bwd = nnx.RNN(
            nnx.LSTMCell(in_features=in_features, hidden_features=hidden_features, rngs=rngs),
            reverse=True,
            keep_order=True,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        y_fwd = self.fwd(x)
        y_bwd = self.bwd(x)
        return jnp.concatenate([y_fwd, y_bwd], axis=-1)


class PyanNet(nnx.Module):
    """PyanNet segmentation model (JAX/Flax NNX).

    Inference-focused port of `pyannote.audio.models.segmentation.PyanNet.PyanNet`.
    """

    def __init__(
        self,
        *,
        sincnet_stride: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        linear_layer_sizes: list[int],
        num_classes: int,
        rngs: nnx.Rngs,
    ):
        self.sincnet = SincNet(stride=sincnet_stride, rngs=rngs)

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        bilstm_layers: list[BiLSTMLayer] = []
        in_features = 60
        for _ in range(lstm_num_layers):
            bilstm_layers.append(
                BiLSTMLayer(in_features=in_features, hidden_features=lstm_hidden_size, rngs=rngs)
            )
            in_features = 2 * lstm_hidden_size
        self.lstm = nnx.List(bilstm_layers)

        ff_layers: list[nnx.Linear] = []
        for out_features in linear_layer_sizes:
            ff_layers.append(nnx.Linear(in_features=in_features, out_features=out_features, rngs=rngs))
            in_features = out_features
        self.linear = nnx.List(ff_layers)

        self.classifier = nnx.Linear(in_features=in_features, out_features=num_classes, rngs=rngs)

    def __call__(self, waveforms: jax.Array) -> jax.Array:
        # SincNet: (B, F, T)
        x = self.sincnet(waveforms)
        # LSTM expects (B, T, F)
        x = jnp.transpose(x, (0, 2, 1))

        for layer in self.lstm:
            x = layer(x)

        for linear in self.linear:
            x = nnx.leaky_relu(linear(x))

        logits = self.classifier(x)
        return jax.nn.log_softmax(logits, axis=-1)


def pyannet_from_torch(torch_pyannet, *, rngs: Optional[nnx.Rngs] = None) -> PyanNet:
    """Convert a PyTorch PyanNet model to Flax NNX."""
    rngs = rngs or nnx.Rngs(0)

    torch_pyannet = torch_pyannet.eval()
    torch_sincnet = torch_pyannet.sincnet
    torch_lstm = torch_pyannet.lstm

    # Feed-forward layer sizes
    linear_layer_sizes: list[int] = []
    if hasattr(torch_pyannet, "linear"):
        for layer in torch_pyannet.linear:
            linear_layer_sizes.append(layer.out_features)

    model = PyanNet(
        sincnet_stride=torch_sincnet.stride,
        lstm_hidden_size=torch_lstm.hidden_size,
        lstm_num_layers=torch_lstm.num_layers,
        linear_layer_sizes=linear_layer_sizes,
        num_classes=torch_pyannet.dimension,
        rngs=rngs,
    )
    model.eval()

    # ---- SincNet weights ----
    _copy_instance_norm1d(torch_sincnet.wav_norm1d, model.sincnet.wav_norm1d)

    # ParamSincFB filters: (out, in, k) -> (k, in, out)
    filters = torch_sincnet.conv1d[0].filterbank.filters()
    model.sincnet.conv0.kernel[...] = jnp.transpose(_torch_tensor_to_jax(filters), (2, 1, 0))

    _copy_instance_norm1d(torch_sincnet.norm1d[0], model.sincnet.norm0)
    _copy_conv1d(torch_sincnet.conv1d[1], model.sincnet.conv1)
    _copy_instance_norm1d(torch_sincnet.norm1d[1], model.sincnet.norm1)
    _copy_conv1d(torch_sincnet.conv1d[2], model.sincnet.conv2)
    _copy_instance_norm1d(torch_sincnet.norm1d[2], model.sincnet.norm2)

    # ---- LSTM weights ----
    for layer_index, bilstm in enumerate(model.lstm):
        _copy_lstm_cell_from_torch(
            torch_lstm, layer_index=layer_index, reverse=False, flax_cell=bilstm.fwd.cell
        )
        _copy_lstm_cell_from_torch(
            torch_lstm, layer_index=layer_index, reverse=True, flax_cell=bilstm.bwd.cell
        )

    # ---- Feed-forward + classifier ----
    for torch_layer, flax_layer in zip(getattr(torch_pyannet, "linear", []), model.linear, strict=False):
        _copy_linear(torch_layer, flax_layer)

    _copy_linear(torch_pyannet.classifier, model.classifier)
    return model


def pyannet_from_pretrained(
    checkpoint: str = "pyannote/speaker-diarization-community-1",
    *,
    subfolder: str = "segmentation",
    token: str | bool | None = None,
    cache_dir: str | None = None,
    rngs: Optional[nnx.Rngs] = None,
) -> PyanNet:
    """Download a PyTorch PyanNet checkpoint and convert to Flax NNX."""
    from pyannote.audio import Model  # local import: optional deps

    torch_model = Model.from_pretrained(
        checkpoint, subfolder=subfolder, token=token, cache_dir=cache_dir
    )
    if torch_model is None:
        raise ValueError(f"Could not load checkpoint: {checkpoint!r}")
    return pyannet_from_torch(torch_model, rngs=rngs)

