import numpy as np
import pytest
import soundfile as sf

from telebench.core.process import Process


@pytest.fixture
def proc(speech_wav):
    proc = Process(speech_wav)
    proc.process()
    return proc


@pytest.fixture
def speech_wav(tmp_path, make_wav):
    return make_wav(tmp_path / "speech.wav")


@pytest.fixture
def noise_wav(tmp_path, make_wav):
    return make_wav(tmp_path / "noise.wav", amplitude=0.1)


class TestLoad:
    def test_process_loads_audio(self, proc):
        assert proc.data is not None
        assert proc.sample_rate == 8000
        assert len(proc.data) == 4000

    def test_process_converts_stereo_to_mono(self, tmp_path, make_wav):
        path = make_wav(tmp_path / "stereo.wav", channels=2)
        proc = Process(path)
        proc.process()
        assert proc.data.ndim == 1

    def test_get_data_returns_loaded_audio(self, proc):
        assert proc.get_data() is proc.data


class TestFourierTransform:
    def test_halves_sample_rate_and_length(self, proc):
        original_len = len(proc.data)
        proc.fourier_transform()
        assert proc.sample_rate == 4000
        assert len(proc.data) == original_len // 2
        assert proc.data.dtype == np.float32

    def test_zeroes_frequencies_above_quarter_rate(self, tmp_path, make_wav):
        path = make_wav(tmp_path / "high.wav", freq=3000.0)
        proc = Process(path)
        proc.process()
        original_energy = np.mean(proc.data.astype(np.float32) ** 2)
        proc.fourier_transform()
        filtered_energy = np.mean(proc.data.astype(np.float32) ** 2)
        assert filtered_energy < original_energy * 0.1

    def test_keeps_low_frequencies(self, tmp_path, make_wav):
        path = make_wav(tmp_path / "low.wav", freq=500.0)
        proc = Process(path)
        proc.process()
        original_energy = np.mean(proc.data.astype(np.float32) ** 2)
        proc.fourier_transform()
        filtered_energy = np.mean(proc.data.astype(np.float32) ** 2)
        assert filtered_energy > original_energy * 0.5


class TestMuLaw:
    def test_returns_int16_samples(self, proc):
        proc.fourier_transform()
        proc.mu_law()
        assert proc.data.dtype == np.int16
        assert proc.data.min() >= -32768
        assert proc.data.max() <= 32767


class TestAddNoise:
    def test_requires_noise_file(self, proc):
        with pytest.raises(ValueError, match="Noise file missing"):
            proc.add_noise()

    def test_requires_loaded_data(self, noise_wav):
        proc = Process("unloaded.wav", noise_file=str(noise_wav))
        with pytest.raises(ValueError, match="Data not loaded"):
            proc.add_noise()

    def test_sample_rate_mismatch_raises(self, tmp_path, make_wav, speech_wav):
        mismatch = make_wav(tmp_path / "noise16k.wav", sr=16000)
        proc = Process(str(speech_wav), noise_file=str(mismatch))
        proc.process()
        with pytest.raises(ValueError, match="Sample rate mismatch"):
            proc.add_noise(output_path=str(tmp_path / "out.wav"))

    def test_empty_noise_raises(self, tmp_path, make_empty_wav, speech_wav):
        empty = make_empty_wav(tmp_path / "empty.wav")
        proc = Process(str(speech_wav), noise_file=str(empty))
        proc.process()
        with pytest.raises(ValueError, match="Noise file is empty"):
            proc.add_noise(output_path=str(tmp_path / "out.wav"))

    def test_writes_output_file(self, tmp_path, speech_wav, noise_wav):
        out = tmp_path / "noisy.wav"
        proc = Process(str(speech_wav), noise_file=str(noise_wav))
        proc.process()
        proc.add_noise(output_path=str(out))
        data, sr = sf.read(out, dtype="int16")
        assert sr == 8000
        assert data.dtype == np.int16
        assert len(data) == 4000

    def test_defaults_to_filename(self, speech_wav, noise_wav):
        proc = Process(str(speech_wav), noise_file=str(noise_wav))
        proc.process()
        proc.add_noise()
        data, sr = sf.read(speech_wav, dtype="int16")
        assert sr == 8000
        assert data.dtype == np.int16

    def test_mixing_increases_signal_energy(self, tmp_path, speech_wav, noise_wav):
        proc = Process(str(speech_wav), noise_file=str(noise_wav))
        proc.process()
        speech_energy = np.mean(proc.data.astype(np.float32) ** 2)
        out = tmp_path / "noisy.wav"
        proc.add_noise(output_path=str(out))
        noisy = sf.read(out, dtype="int16")[0]
        noisy_energy = np.mean(noisy.astype(np.float32) ** 2)
        assert noisy_energy > speech_energy