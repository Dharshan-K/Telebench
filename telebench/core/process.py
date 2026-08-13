import os
import tempfile

import numpy as np
import soundfile as sf


class Process:
    def __init__(
        self,
        filename,
        noise_file=None,
        snr_db_min=5,
        snr_db_max=15,
        normalize_scale=32768,
    ):
        self.filename = filename
        self.data = None
        self.sample_rate = None
        self.noise_file = noise_file
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max
        self.normalize_scale = normalize_scale

    def process(self):
        self.data, self.sample_rate = sf.read(self.filename, dtype="int16")
        if self.data.ndim > 1:
            self.data = self.data.mean(axis=1)

    def fourier_transform(self):
        fft_spectrum = np.fft.rfft(self.data)
        frequencies = np.fft.rfftfreq(len(self.data), d=1 / self.sample_rate)
        mask = frequencies > (self.sample_rate // 4)
        fft_spectrum[mask] = 0
        filtered_data = np.fft.irfft(fft_spectrum)
        downsampled_data = filtered_data[::2]
        self.sample_rate //= 2
        normalized = (downsampled_data / self.normalize_scale).astype(np.float32)
        self.data = normalized

    def mu_law(self):
        mu = 255

        mulaw = np.sign(self.data) * (
            np.log1p(mu * np.abs(self.data))
            / np.log1p(mu)
        )
        mulaw_pcm = (mulaw * 32767).astype(np.int16)
        self.data = mulaw_pcm

    def add_noise(self, output_path=None):
        if self.noise_file is None:
            raise ValueError("Noise file missing.")
        speech = self.data
        if speech is None:
            raise ValueError("Data not loaded. Call process() first.")
        speech_sample_rate = self.sample_rate
        noise, noise_sample_rate = sf.read(self.noise_file, dtype="int16")
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        if noise.ndim > 1:
            noise = noise.mean(axis=1)

        if len(noise) == 0:
            raise ValueError("Noise file is empty.")

        if noise_sample_rate != speech_sample_rate:
            raise ValueError(
                f"Sample rate mismatch: speech is {speech_sample_rate} Hz but "
                f"noise is {noise_sample_rate} Hz. Resample the noise to match "
                f"before mixing."
            )

        repeat = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, repeat)
        noise = noise[: len(speech)]

        speech_power = np.mean(speech.astype(np.float32) ** 2)
        noise_power = np.mean(noise.astype(np.float32) ** 2)
        snr_db = np.random.uniform(self.snr_db_min, self.snr_db_max)

        desired_noise_power = speech_power / (10 ** (snr_db / 10))
        if noise_power == 0:
            scale = 0.0
        else:
            scale = np.sqrt(desired_noise_power / noise_power)
        noise = noise * scale
        noisy = speech.astype(np.float32) + noise
        noisy = np.clip(noisy, -32768, 32767)
        noisy = noisy.astype(np.int16)
        dest = output_path or self.filename
        dest_dir = os.path.dirname(dest) or "."
        ext = os.path.splitext(dest)[1]
        fd, tmp_path = tempfile.mkstemp(dir=dest_dir, suffix=ext)
        os.close(fd)
        os.unlink(tmp_path)
        sf.write(tmp_path, noisy, speech_sample_rate)
        os.replace(tmp_path, dest)

    def get_data(self):
        return self.data
