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
from pathlib import Path
from typing import Optional, Text, Union

import numpy as np
import torch

from pyannote.audio.core.inference import BaseInference
from pyannote.audio.core.inference_flax_nnx import parse_jax_device
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
    if isinstance(model, dict) and "backend" in model:
        stripped = dict(model)
        stripped.pop("backend", None)
        stripped.pop("jax_device", None)
        stripped.pop("jit", None)
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
    ):
        super().__init__()

        self.embedding = embedding
        self.device = device or torch.device("cpu")
        self.jax_device = parse_jax_device(jax_device)

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

        self._forward_no_weights = nnx.jit(_forward_no_weights) if jit else _forward_no_weights
        self._forward_with_weights = (
            nnx.jit(_forward_with_weights) if jit else _forward_with_weights
        )

    def to(self, device: torch.device | str | None):
        # Keep torch-like `.to(...)` so `Pipeline.to(...)` can propagate device
        # selections. We interpret this as a JAX device hint.
        if isinstance(device, str):
            self.jax_device = parse_jax_device(device)
            return self

        if device is None:
            self.jax_device = None
            return self

        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        self.device = device
        self.jax_device = parse_jax_device(device)
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
        if self.jax_device is not None:
            fbank_jax = jax.device_put(fbank_jax, self.jax_device)

        if masks is None:
            y = self._forward_no_weights(self.flax_resnet_, fbank_jax)
            return np.asarray(jax.device_get(y))

        weights_np = masks.detach().cpu().numpy().astype(np.float32, copy=False)
        weights_jax = jnp.asarray(weights_np, dtype=jnp.float32)
        if self.jax_device is not None:
            weights_jax = jax.device_put(weights_jax, self.jax_device)

        y = self._forward_with_weights(self.flax_resnet_, fbank_jax, weights_jax)
        out = np.asarray(jax.device_get(y))
        if out.shape[0] != batch_size:
            out = out[:batch_size]
        return out

