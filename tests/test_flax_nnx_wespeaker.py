import os

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.getenv("PYANNOTE_E2E") != "1",
    reason="Set PYANNOTE_E2E=1 to run HF download + Flax NNX conversion tests.",
)


def test_wespeaker_resnet34_torch_to_flax_nnx():
    token = os.getenv("HF_TOKEN")

    try:
        import jax.numpy as jnp
    except Exception:
        pytest.skip("jax is not installed")

    try:
        from pyannote.audio import Model
        from pyannote.audio.models.embedding.wespeaker.flax_nnx import resnet34_from_torch
    except Exception as e:
        pytest.skip(str(e))

    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", token=token)
    torch_resnet = model.resnet.eval()

    # Random fbank input (batch, frames, features)
    fbank = torch.randn(2, 200, model.hparams.num_mel_bins, dtype=torch.float32)

    with torch.inference_mode():
        torch_out = torch_resnet(fbank)[1].cpu().numpy()

    flax_resnet = resnet34_from_torch(torch_resnet)
    flax_out = flax_resnet(jnp.asarray(fbank.cpu().numpy()))

    assert flax_out.shape == torch_out.shape
    np.testing.assert_allclose(np.asarray(flax_out), torch_out, rtol=2e-4, atol=2e-4)
