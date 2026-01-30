import os

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.getenv("PYANNOTE_E2E") != "1",
    reason="Set PYANNOTE_E2E=1 to run HF gated + Flax NNX conversion tests.",
)


def test_pyannet_torch_to_flax_nnx():
    token = os.getenv("HF_TOKEN")
    offline = os.getenv("HF_HUB_OFFLINE") == "1"
    if not token and not offline:
        pytest.skip("HF_TOKEN is not set (set HF_HUB_OFFLINE=1 to use cached artifacts)")

    try:
        import jax.numpy as jnp
    except Exception:
        pytest.skip("jax is not installed")

    from pyannote.audio import Model
    from pyannote.audio.models.segmentation.flax_nnx import pyannet_from_torch

    torch_model = Model.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        subfolder="segmentation",
        token=token,
    ).eval()

    # (batch, channel, sample)
    waveforms = torch.randn(1, 1, 16000, dtype=torch.float32)

    with torch.inference_mode():
        torch_out = torch_model(waveforms).cpu().numpy()

    flax_model = pyannet_from_torch(torch_model)
    flax_out = flax_model(jnp.asarray(waveforms.cpu().numpy()))

    assert flax_out.shape == torch_out.shape
    np.testing.assert_allclose(np.asarray(flax_out), torch_out, rtol=1e-3, atol=1e-3)
