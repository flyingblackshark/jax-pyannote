import os

import pytest
import torchaudio


pytestmark = pytest.mark.skipif(
    os.getenv("PYANNOTE_E2E") != "1",
    reason="Set PYANNOTE_E2E=1 to run HF gated pipeline test.",
)


def test_pipeline_speaker_diarization_community1_runs():
    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN is not set")

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )

    waveform, sample_rate = torchaudio.load("tests/data/dev00.wav")
    seconds = 12
    waveform = waveform[:, : seconds * sample_rate]

    output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # community-1 returns a rich output object
    assert hasattr(output, "speaker_diarization")
    assert hasattr(output, "exclusive_speaker_diarization")

