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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from functools import cached_property
import math
from pathlib import Path
from typing import Optional, Text, Union

import numpy as np
import torch

from pyannote.audio.core.inference import BaseInference
from pyannote.audio.core.inference_flax_nnx import parse_jax_device, parse_jax_devices
from pyannote.audio.pipelines.utils import PipelineModel, get_model

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "`pyannote.audio.pipelines.speaker_verification_flax_nnx` requires `jax` and `flax`.\n"
        "Install them with e.g. `pip install jax flax` (or `pip install -e '.[flax]'`)."
    ) from exc


def _strip_backend(model: PipelineModel) -> PipelineModel:
    """Remove `backend` key from dict model spec (if any)."""
    if isinstance(model, dict):
        stripped = dict(model)
        stripped.pop("backend", None)
        stripped.pop("jax_device", None)
        stripped.pop("jit", None)
        stripped.pop("data_parallel", None)
        return stripped
    return model


class FlaxNNXWeSpeakerPretrainedSpeakerEmbedding(BaseInference):
    """Flax NNX speaker embedding extractor for WeSpeaker ResNet34 checkpoints.

    This loads a PyTorch `pyannote.audio` embedding model to reuse its fbank
    extraction hyperparameters, converts the ResNet backbone to Flax NNX, and
    runs inference in JAX.
    """

    def __init__(
        self,
        embedding: PipelineModel = "pyannote/wespeaker-voxceleb-resnet34-LM",
        device: Optional[torch.device] = None,
        token: Union[Text, None] = None,
        cache_dir: Union[Path, Text, None] = None,
        *,
        jax_device: str | jax.Device | torch.device | None = None,
        jit: bool = True,
        data_parallel: bool = False,
    ):
        super().__init__()

        self.embedding = embedding
        self.device = device or torch.device("cpu")
        self.jax_device = parse_jax_device(jax_device)
        self.data_parallel = bool(data_parallel)
        self._dp_devices: list[jax.Device] | None = None
        if self.data_parallel:
            dp_devices = parse_jax_devices(jax_device)
            if len(dp_devices) > 1:
                self._dp_devices = dp_devices

        # Load the PyTorch model (for fbank extraction + metadata).
        self.torch_model_ = get_model(_strip_backend(embedding), token=token, cache_dir=cache_dir)
        self.torch_model_.eval()
        # Keep it on CPU: JAX backend is responsible for accelerator work.
        self.torch_model_.to(torch.device("cpu"))

        if not hasattr(self.torch_model_, "resnet"):
            raise ValueError(
                "Flax NNX WeSpeaker backend expects a model with a `.resnet` attribute "
                "(e.g. WeSpeakerResNet34)."
            )

        from pyannote.audio.models.embedding.wespeaker.flax_nnx import resnet34_from_torch

        self.flax_resnet_ = resnet34_from_torch(self.torch_model_.resnet.eval())
        self.flax_resnet_.eval()

        def _forward_no_weights(model: nnx.Module, fbank: jax.Array) -> jax.Array:
            return model(fbank)

        def _forward_with_weights(
            model: nnx.Module, fbank: jax.Array, weights: jax.Array
        ) -> jax.Array:
            return model(fbank, weights=weights)

        self._forward_no_weights_fn = _forward_no_weights
        self._forward_with_weights_fn = _forward_with_weights
        self._jit = bool(jit)
        self._rebuild_forward()

    def _rebuild_forward(self) -> None:
        if self._dp_devices is not None:
            self._forward_no_weights = nnx.pmap(
                self._forward_no_weights_fn,
                axis_name="data",
                in_axes=(None, 0),
                out_axes=0,
                devices=self._dp_devices,
            )
            self._forward_with_weights = nnx.pmap(
                self._forward_with_weights_fn,
                axis_name="data",
                in_axes=(None, 0, 0),
                out_axes=0,
                devices=self._dp_devices,
            )
        else:
            self._forward_no_weights = (
                nnx.jit(self._forward_no_weights_fn)
                if self._jit
                else self._forward_no_weights_fn
            )
            self._forward_with_weights = (
                nnx.jit(self._forward_with_weights_fn)
                if self._jit
                else self._forward_with_weights_fn
            )

    @cached_property
    def supports_speaker_weights(self) -> bool:
        return True

    def to(self, device: torch.device | str | None):
        # Keep torch-like `.to(...)` so `Pipeline.to(...)` can propagate device
        # selections. We interpret this as a JAX device hint.
        if isinstance(device, str):
            self.jax_device = parse_jax_device(device)
            if self.data_parallel:
                dp_devices = parse_jax_devices(device)
                self._dp_devices = dp_devices if len(dp_devices) > 1 else None
                self._rebuild_forward()
            return self

        if device is None:
            self.jax_device = None
            if self.data_parallel:
                dp_devices = parse_jax_devices(None)
                self._dp_devices = dp_devices if len(dp_devices) > 1 else None
                self._rebuild_forward()
            return self

        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        self.device = device
        self.jax_device = parse_jax_device(device)
        if self.data_parallel:
            dp_devices = parse_jax_devices(device)
            self._dp_devices = dp_devices if len(dp_devices) > 1 else None
            self._rebuild_forward()
        return self

    @cached_property
    def sample_rate(self) -> int:
        return self.torch_model_.audio.sample_rate

    @cached_property
    def dimension(self) -> int:
        return self.torch_model_.dimension

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        # Conservative probe: reuse PyTorch model forward constraints.
        with torch.inference_mode():
            lower, upper = 2, round(0.5 * self.sample_rate)
            middle = (lower + upper) // 2
            while lower + 1 < upper:
                try:
                    _ = self.torch_model_(torch.randn(1, 1, middle))
                    upper = middle
                except Exception:
                    lower = middle
                middle = (lower + upper) // 2
        return upper

    def __call__(
        self, waveforms: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        batch_size, num_channels, _ = waveforms.shape
        if num_channels != 1:
            raise ValueError("Only mono waveforms (num_channels == 1) are supported.")

        with torch.inference_mode():
            # Always compute fbank on CPU for now.
            fbank = self.torch_model_.compute_fbank(waveforms.detach().cpu())
            fbank_np = fbank.cpu().numpy()

        fbank_jax = jnp.asarray(fbank_np, dtype=jnp.float32)
        if self._dp_devices is None and self.jax_device is not None:
            fbank_jax = jax.device_put(fbank_jax, self.jax_device)

        if masks is None:
            if self._dp_devices is None:
                y = self._forward_no_weights(self.flax_resnet_, fbank_jax)
                return np.asarray(jax.device_get(y))

            dp_count = len(self._dp_devices)
            padded_bs = int(math.ceil(batch_size / dp_count) * dp_count)
            if padded_bs != batch_size:
                pad = jnp.zeros((padded_bs - batch_size,) + fbank_jax.shape[1:], dtype=fbank_jax.dtype)
                fbank_jax = jnp.concatenate([fbank_jax, pad], axis=0)

            fbank_jax = fbank_jax.reshape(
                dp_count, padded_bs // dp_count, fbank_jax.shape[1], fbank_jax.shape[2]
            )
            y = self._forward_no_weights(self.flax_resnet_, fbank_jax)
            y_np = np.asarray(jax.device_get(y)).reshape((padded_bs,) + y.shape[2:])
            return y_np[:batch_size]

        weights_np = masks.detach().cpu().numpy().astype(np.float32, copy=False)
        weights_jax = jnp.asarray(weights_np, dtype=jnp.float32)
        if self._dp_devices is None and self.jax_device is not None:
            weights_jax = jax.device_put(weights_jax, self.jax_device)

        if self._dp_devices is None:
            y = self._forward_with_weights(self.flax_resnet_, fbank_jax, weights_jax)
            out = np.asarray(jax.device_get(y))
            if out.shape[0] != batch_size:
                out = out[:batch_size]
            return out

        dp_count = len(self._dp_devices)
        padded_bs = int(math.ceil(batch_size / dp_count) * dp_count)
        if padded_bs != batch_size:
            pad = jnp.zeros((padded_bs - batch_size,) + fbank_jax.shape[1:], dtype=fbank_jax.dtype)
            fbank_jax = jnp.concatenate([fbank_jax, pad], axis=0)

            pad_w = jnp.zeros((padded_bs - batch_size,) + weights_jax.shape[1:], dtype=weights_jax.dtype)
            weights_jax = jnp.concatenate([weights_jax, pad_w], axis=0)

        fbank_jax = fbank_jax.reshape(
            dp_count, padded_bs // dp_count, fbank_jax.shape[1], fbank_jax.shape[2]
        )
        if weights_jax.ndim == 2:
            weights_jax = weights_jax.reshape(dp_count, padded_bs // dp_count, weights_jax.shape[1])
        else:
            weights_jax = weights_jax.reshape(
                dp_count, padded_bs // dp_count, weights_jax.shape[1], weights_jax.shape[2]
            )

        y = self._forward_with_weights(self.flax_resnet_, fbank_jax, weights_jax)
        out = np.asarray(jax.device_get(y)).reshape((padded_bs,) + y.shape[2:])
        return out[:batch_size]
