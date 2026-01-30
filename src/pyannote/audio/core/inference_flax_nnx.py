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

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pyannote.audio.core.inference import BaseInference
from pyannote.audio.core.io import Audio
from pyannote.audio.core.io import AudioFile
from pyannote.core import SlidingWindow, SlidingWindowFeature

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "`pyannote.audio.core.inference_flax_nnx` requires `jax` and `flax`.\n"
        "Install them with e.g. `pip install jax flax` (or `pip install -e '.[flax]'`)."
    ) from exc


def build_powerset_mapping(num_classes: int, max_set_size: int) -> np.ndarray:
    """Build powerset-to-multilabel mapping (NumPy).

    This matches the order used by `pyannote.audio.utils.powerset.Powerset`:
    increasing set size, then lexicographic combinations.
    """

    if max_set_size < 0:
        raise ValueError("`max_set_size` must be non-negative.")

    powerset_classes = []
    for set_size in range(0, max_set_size + 1):
        for current_set in combinations(range(num_classes), set_size):
            powerset_classes.append(current_set)

    mapping = np.zeros((len(powerset_classes), num_classes), dtype=np.float32)
    for k, current_set in enumerate(powerset_classes):
        mapping[k, list(current_set)] = 1.0
    return mapping


def _parse_backend_index(spec: str) -> Tuple[str, Optional[int]]:
    spec = spec.strip().lower()
    if ":" not in spec:
        return spec, None
    backend, index_str = spec.split(":", 1)
    backend = backend.strip()
    index_str = index_str.strip()
    if index_str == "":
        return backend, None
    return backend, int(index_str)


def parse_jax_device(device: Any | None) -> jax.Device | None:
    """Best-effort parsing of a device spec into a `jax.Device`.

    Supported values
    ----------------
    - `None`: use JAX default placement
    - `jax.Device`: returned as-is
    - `torch.device`: maps cpu/cuda/xla/tpu to cpu/gpu/tpu when available
    - `str`: "auto", "cpu", "gpu", "tpu", optionally with ":N" index
    """

    if device is None:
        return None

    if isinstance(device, jax.Device):
        return device

    if isinstance(device, torch.device):
        if device.type == "cpu":
            backend = "cpu"
            index = 0
        elif device.type == "cuda":
            backend = "gpu"
            index = device.index or 0
        elif device.type in ("xla", "tpu"):
            backend = "tpu"
            index = device.index or 0
        else:
            return None

        try:
            return jax.devices(backend)[index]
        except Exception:
            return None

    if not isinstance(device, str):
        raise TypeError(
            f"Unsupported JAX device specification type: {type(device).__name__}. "
            "Expected `jax.Device`, `torch.device`, `str`, or `None`."
        )

    backend, index = _parse_backend_index(device)
    if backend in ("", "auto", "default"):
        return None

    if backend == "cuda":
        backend = "gpu"

    if backend not in ("cpu", "gpu", "tpu"):
        raise ValueError(
            f"Unsupported JAX backend {backend!r}. Expected one of 'cpu', 'gpu', 'tpu', or 'auto'."
        )

    index = index or 0
    try:
        return jax.devices(backend)[index]
    except Exception:
        return None


def parse_jax_devices(device: Any | None) -> list[jax.Device]:
    """Best-effort parsing of a device spec into a list of *local* `jax.Device`.

    This is mostly meant for data parallel inference (`pmap`) where we want all
    available devices for a given backend.
    """

    if device is None:
        return list(jax.local_devices())

    if isinstance(device, jax.Device):
        return [device]

    if isinstance(device, torch.device):
        if device.type == "cpu":
            backend = "cpu"
        elif device.type == "cuda":
            backend = "gpu"
        elif device.type in ("xla", "tpu"):
            backend = "tpu"
        else:
            return list(jax.local_devices())

        try:
            devs = list(jax.local_devices(backend=backend))
        except Exception:
            return list(jax.local_devices())

        if device.index is None:
            return devs

        try:
            return [devs[device.index]]
        except Exception:
            return devs

    if not isinstance(device, str):
        raise TypeError(
            f"Unsupported JAX device specification type: {type(device).__name__}. "
            "Expected `jax.Device`, `torch.device`, `str`, or `None`."
        )

    backend, index = _parse_backend_index(device)
    if backend in ("", "auto", "default"):
        return list(jax.local_devices())

    if backend == "cuda":
        backend = "gpu"

    if backend not in ("cpu", "gpu", "tpu"):
        raise ValueError(
            f"Unsupported JAX backend {backend!r}. Expected one of 'cpu', 'gpu', 'tpu', or 'auto'."
        )

    try:
        devs = list(jax.local_devices(backend=backend))
    except Exception:
        return list(jax.local_devices())

    if index is None:
        return devs

    try:
        return [devs[index]]
    except Exception:
        return devs


@dataclass(frozen=True)
class FlaxNNXSlidingConfig:
    """Configuration for Flax NNX sliding-window inference."""

    duration: float
    step: float
    batch_size: int = 32


class FlaxNNXSlidingInference(BaseInference):
    """Sliding-window inference for Flax NNX models (waveform -> frame scores).

    This is a minimal inference utility meant to integrate Flax NNX ports into
    existing pyannote.audio pipelines.

    Notes
    -----
    - Uses `Audio` for waveform loading/resampling (PyTorch backend).
    - Uses `nnx.jit` and batch padding to keep static shapes and avoid extra
      recompilations (important for TPU/GPU).
    """

    def __init__(
        self,
        model: nnx.Module,
        *,
        audio: Audio,
        config: FlaxNNXSlidingConfig,
        powerset_mapping: np.ndarray | None = None,
        jax_device: Any | None = None,
        jit: bool = True,
        data_parallel: bool = False,
    ):
        super().__init__()
        self.model = model
        self.audio = audio
        self.duration = float(config.duration)
        self.step = float(config.step)
        self.batch_size = int(config.batch_size)
        if self.batch_size < 1:
            raise ValueError("`batch_size` must be >= 1.")

        self.jax_device = parse_jax_device(jax_device)
        self.data_parallel = bool(data_parallel)
        self._dp_devices: list[jax.Device] | None = None
        self._dp_devices_count = 1
        if self.data_parallel:
            dp_devices = parse_jax_devices(jax_device)
            if len(dp_devices) > 1:
                self._dp_devices = dp_devices
                self._dp_devices_count = len(dp_devices)
                if self.batch_size % self._dp_devices_count != 0:
                    self.batch_size = int(
                        math.ceil(self.batch_size / self._dp_devices_count)
                        * self._dp_devices_count
                    )

        mapping = None
        if powerset_mapping is not None:
            mapping = jnp.asarray(powerset_mapping, dtype=jnp.float32)

        def _forward(model: nnx.Module, chunks: jax.Array) -> jax.Array:
            y = model(chunks)
            if mapping is None:
                return y

            # Hard conversion: argmax in powerset space, then map to multilabel.
            idx = jnp.argmax(y, axis=-1)
            return mapping[idx]

        self._forward_fn = _forward
        self._jit = bool(jit)
        self._rebuild_forward()

    def _rebuild_forward(self) -> None:
        if self._dp_devices is not None:
            self._forward = nnx.pmap(
                self._forward_fn,
                axis_name="data",
                in_axes=(None, 0),
                out_axes=0,
                devices=self._dp_devices,
            )
        else:
            self._forward = nnx.jit(self._forward_fn) if self._jit else self._forward_fn

    def to(self, device: torch.device | str | None) -> "FlaxNNXSlidingInference":
        # Keep a torch-like `.to(...)` so `Pipeline.to(...)` can propagate device
        # selections. When the requested backend is not available, fall back to
        # JAX default device placement.
        if isinstance(device, str):
            self.jax_device = parse_jax_device(device)
            if self.data_parallel:
                dp_devices = parse_jax_devices(device)
                if len(dp_devices) > 1:
                    self._dp_devices = dp_devices
                    self._dp_devices_count = len(dp_devices)
                    if self.batch_size % self._dp_devices_count != 0:
                        self.batch_size = int(
                            math.ceil(self.batch_size / self._dp_devices_count)
                            * self._dp_devices_count
                        )
                else:
                    self._dp_devices = None
                    self._dp_devices_count = 1
                self._rebuild_forward()
            return self

        if device is None:
            self.jax_device = None
            if self.data_parallel:
                dp_devices = parse_jax_devices(None)
                if len(dp_devices) > 1:
                    self._dp_devices = dp_devices
                    self._dp_devices_count = len(dp_devices)
                    if self.batch_size % self._dp_devices_count != 0:
                        self.batch_size = int(
                            math.ceil(self.batch_size / self._dp_devices_count)
                            * self._dp_devices_count
                        )
                else:
                    self._dp_devices = None
                    self._dp_devices_count = 1
                self._rebuild_forward()
            return self

        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be a `torch.device`, `str`, or `None`, got `{type(device).__name__}`."
            )

        self.jax_device = parse_jax_device(device)
        if self.data_parallel:
            dp_devices = parse_jax_devices(device)
            if len(dp_devices) > 1:
                self._dp_devices = dp_devices
                self._dp_devices_count = len(dp_devices)
                if self.batch_size % self._dp_devices_count != 0:
                    self.batch_size = int(
                        math.ceil(self.batch_size / self._dp_devices_count)
                        * self._dp_devices_count
                    )
            else:
                self._dp_devices = None
                self._dp_devices_count = 1
            self._rebuild_forward()
        return self

    def __call__(
        self, file: AudioFile, hook: Optional[Callable] = None
    ) -> SlidingWindowFeature:
        waveform, sample_rate = self.audio(file)
        return self.slide(waveform, sample_rate, hook=hook)

    def slide(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        hook: Optional[Callable] = None,
    ) -> SlidingWindowFeature:
        if waveform.ndim != 2:
            raise ValueError(f"Expected (channel, sample) waveform, got {waveform.shape}.")

        waveform = waveform.detach()
        if waveform.device.type != "cpu":
            waveform = waveform.cpu()
        waveform = waveform.to(dtype=torch.float32)

        window_size = self.audio.get_num_samples(self.duration, sample_rate=sample_rate)
        step_size = round(self.step * sample_rate)
        _, num_samples = waveform.shape

        if step_size <= 0:
            raise ValueError("`step` must be > 0.")

        # Full chunks
        if num_samples >= window_size:
            chunks = waveform.unfold(1, window_size, step_size).permute(1, 0, 2)
            num_chunks = chunks.shape[0]
        else:
            chunks = waveform.new_zeros((0, waveform.shape[0], window_size))
            num_chunks = 0

        # Last partial chunk (zero-padded)
        has_last_chunk = (num_samples < window_size) or (
            (num_samples - window_size) % step_size > 0
        )
        if has_last_chunk:
            last_chunk = waveform[:, num_chunks * step_size :]
            last_pad = window_size - last_chunk.shape[1]
            last_chunk = F.pad(last_chunk, (0, last_pad))
            chunks = torch.cat([chunks, last_chunk[None]], dim=0)

        total = chunks.shape[0]
        if hook is not None:
            hook(completed=0, total=total)

        outputs: list[np.ndarray] = []
        bs = self.batch_size
        dp_count = self._dp_devices_count

        for start in range(0, total, bs):
            batch = chunks[start : start + bs]
            actual_bs = batch.shape[0]

            # Pad to keep a constant batch shape (helps JIT caching).
            if actual_bs < bs:
                pad = batch.new_zeros((bs - actual_bs, batch.shape[1], batch.shape[2]))
                batch = torch.cat([batch, pad], dim=0)

            x = jnp.asarray(np.asarray(batch.numpy()))
            if self._dp_devices is not None:
                # (B, C, T) -> (D, B/D, C, T)
                x = x.reshape(dp_count, bs // dp_count, x.shape[1], x.shape[2])
            elif self.jax_device is not None:
                x = jax.device_put(x, self.jax_device)

            y = self._forward(self.model, x)
            y_np_full = np.asarray(jax.device_get(y))
            if self._dp_devices is not None:
                y_np_full = y_np_full.reshape((bs,) + y_np_full.shape[2:])
            y_np = y_np_full[:actual_bs]
            outputs.append(y_np)

            if hook is not None:
                hook(completed=min(start + bs, total), total=total)

        data = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0, 0), dtype=np.float32)
        frames = SlidingWindow(start=0.0, duration=self.duration, step=self.step)
        return SlidingWindowFeature(data, frames)
