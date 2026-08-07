import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def make_wav():
    """Write a synthetic PCM-16 wav and return its path."""

    def _make(path, sr=8000, seconds=0.5, freq=440.0, amplitude=0.5, channels=1):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = int(sr * seconds)
        t = np.linspace(0, seconds, n, endpoint=False)
        data = np.sin(2 * np.pi * freq * t) * amplitude
        if channels == 2:
            data = np.stack([data, data], axis=1)
        sf.write(path, data, sr, subtype="PCM_16")
        return path

    return _make


@pytest.fixture
def make_empty_wav():
    """Write a valid but sample-less wav file."""

    def _make(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(b"")
        return path

    return _make


@pytest.fixture
def make_config():
    """Build a valid benchmark config dict pointing at fresh tmp dirs."""

    def _make(base):
        base = Path(base)
        speech = base / "speech"
        noise = base / "noise"
        degraded = base / "degraded"
        noisy = base / "noisy"
        for d in (speech, noise, degraded, noisy):
            d.mkdir(parents=True, exist_ok=True)
        return {
            "dataset": {
                "speech": {"output_dir": str(base / "dataset_speech")},
                "noise": {"output_dir": str(base / "dataset_noise")},
            },
            "preprocess": {
                "input_dir": str(speech),
                "output_dir": str(degraded),
            },
            "noise_mixing": {
                "snr_db_min": 5,
                "snr_db_max": 15,
                "input_dir": str(degraded),
                "noise_dir": str(noise),
                "output_dir": str(noisy),
            },
            "evaluation": {
                "original_dir": str(speech),
                "degraded_dir": str(noisy),
                "output_path": str(base / "evaluation.json"),
                "metrics": ["wer", "cer"],
                "endpoints": [
                    {
                        "name": "test-endpoint",
                        "base_url": "http://localhost:1/v1",
                        "model": "fake-model",
                    }
                ],
            },
        }

    return _make


@pytest.fixture
def write_config(tmp_path):
    def _write(config, name="config.yaml"):
        path = tmp_path / name
        with open(path, "w") as f:
            yaml.safe_dump(config, f)
        return path

    return _write