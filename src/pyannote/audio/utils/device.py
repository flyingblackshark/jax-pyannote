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

from typing import Any, Optional, Tuple

import torch


def _try_import_torch_xla():
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
    except Exception:  # pragma: no cover - exact exception depends on installation/runtime
        return None
    return xm


def _xla_device(index: Optional[int] = None) -> torch.device:
    xm = _try_import_torch_xla()
    if xm is None:
        raise RuntimeError(
            "XLA/TPU device requested but `torch_xla` is not available. "
            "Install `torch_xla` matching your PyTorch version (and run on a TPU runtime)."
        )

    # torch_xla APIs accept an optional ordinal for multi-core setups.
    try:
        return xm.xla_device(index) if index is not None else xm.xla_device()
    except TypeError:
        # Backward compatibility with older torch_xla versions where xla_device()
        # might not accept an argument.
        if index is not None:
            raise
        return xm.xla_device()


def _xla_hardware(device: torch.device) -> Optional[str]:
    xm = _try_import_torch_xla()
    if xm is None:
        return None
    try:
        return xm.xla_device_hw(device)  # type: ignore[attr-defined]
    except Exception:
        return None


def _parse_xla_spec(spec: str) -> Tuple[str, Optional[int]]:
    spec = spec.strip().lower()
    if ":" not in spec:
        return spec, None

    base, index_str = spec.split(":", 1)
    base = base.strip()
    index_str = index_str.strip()
    if index_str == "":
        return base, None
    return base, int(index_str)


def parse_device(device: Any) -> torch.device:
    """Parse a device specification into a ``torch.device``.

    Supported inputs
    ----------------
    - ``torch.device``: returned as-is
    - strings: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``, ``"xla"``, ``"xla:N"``,
      ``"tpu"``, ``"tpu:N"``, and any other string accepted by ``torch.device``.
    """

    if isinstance(device, torch.device):
        return device

    if device is None:
        return torch.device("cpu")

    if not isinstance(device, str):
        raise TypeError(
            f"Unsupported device specification type: {type(device).__name__}. "
            "Expected `torch.device`, `str`, or `None`."
        )

    spec = device.strip().lower()

    if spec in ("", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")

        # Prefer TPU when torch_xla is available and actually backed by TPU.
        xm = _try_import_torch_xla()
        if xm is not None:
            try:
                xla_dev = _xla_device()
                if _xla_hardware(xla_dev) == "TPU":
                    return xla_dev
            except Exception:
                pass

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")

        return torch.device("cpu")

    base, index = _parse_xla_spec(spec)
    if base in ("xla", "tpu"):
        xla_dev = _xla_device(index=index)
        if base == "tpu":
            hw = _xla_hardware(xla_dev)
            if hw is not None and hw != "TPU":
                raise RuntimeError(
                    f"Requested TPU device but XLA backend reports {hw!r}. "
                    "Make sure you are running on a TPU runtime (e.g. set `PJRT_DEVICE=TPU`)."
                )
        return xla_dev

    # Let PyTorch handle the rest (cpu/cuda/mps/cuda:N/...).
    try:
        return torch.device(spec)
    except Exception as exc:
        raise ValueError(f"Invalid device specification: {device!r}.") from exc

