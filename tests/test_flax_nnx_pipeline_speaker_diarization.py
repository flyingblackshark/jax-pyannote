import os

import pytest
import torchaudio


pytestmark = pytest.mark.skipif(
    os.getenv("PYANNOTE_E2E") != "1",
    reason="Set PYANNOTE_E2E=1 to run HF gated + Flax NNX pipeline tests.",
)


def test_pipeline_speaker_diarization_flax_nnx_runs():
    token = os.getenv("HF_TOKEN")
    offline = os.getenv("HF_HUB_OFFLINE") == "1"
    if not token and not offline:
        pytest.skip("HF_TOKEN is not set (set HF_HUB_OFFLINE=1 to use cached artifacts)")

    try:
        import jax  # noqa: F401
    except Exception:
        pytest.skip("jax is not installed")

    from pyannote.audio.pipelines import SpeakerDiarization

    pipeline = SpeakerDiarization(
        backend="flax_nnx",
        token=token,
    )

    waveform, sample_rate = torchaudio.load("tests/data/dev00.wav")
    seconds = 5
    waveform = waveform[:, : seconds * sample_rate]

    output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    assert hasattr(output, "speaker_diarization")
    assert hasattr(output, "exclusive_speaker_diarization")

