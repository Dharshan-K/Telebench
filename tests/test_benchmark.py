import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from telebench.benchmark import Benchmark
from telebench.core import evaluate
from fakes import FakeEvaluator


@pytest.fixture
def benchmark(tmp_path, make_config, make_wav, write_config):
    cfg = make_config(tmp_path)
    make_wav(Path(cfg["preprocess"]["input_dir"]) / "s1.wav")
    make_wav(Path(cfg["noise_mixing"]["noise_dir"]) / "n1.wav")
    return Benchmark(str(write_config(cfg)))


class TestValidate:
    def test_missing_section(self, tmp_path, make_config, write_config):
        cfg = make_config(tmp_path)
        cfg.pop("evaluation")
        with pytest.raises(ValueError, match="missing required sections"):
            Benchmark(str(write_config(cfg, "case1.yaml")))

    def test_missing_keys(self, tmp_path, make_config, write_config):
        cfg = make_config(tmp_path)
        cfg["preprocess"].pop("input_dir")
        with pytest.raises(ValueError, match="missing required keys"):
            Benchmark(str(write_config(cfg, "case2.yaml")))

    def test_snr_min_must_be_below_max(self, tmp_path, make_config, write_config):
        cfg = make_config(tmp_path)
        cfg["noise_mixing"]["snr_db_min"] = 15
        cfg["noise_mixing"]["snr_db_max"] = 5
        with pytest.raises(ValueError, match="snr_db_min must be < snr_db_max"):
            Benchmark(str(write_config(cfg, "case3.yaml")))

    def test_missing_noise_dir(self, tmp_path, make_config, write_config):
        cfg = make_config(tmp_path)
        shutil.rmtree(cfg["noise_mixing"]["noise_dir"])
        with pytest.raises(ValueError, match="noise_dir does not exist"):
            Benchmark(str(write_config(cfg, "case4.yaml")))

    def test_noise_dir_without_audio(self, tmp_path, make_config, write_config):
        cfg = make_config(tmp_path)
        shutil.rmtree(cfg["noise_mixing"]["noise_dir"])
        Path(cfg["noise_mixing"]["noise_dir"]).mkdir()
        with pytest.raises(ValueError, match="No audio files found in noise_mixing.noise_dir"):
            Benchmark(str(write_config(cfg, "case5.yaml")))

    def test_endpoint_requires_base_url_and_model(self, tmp_path, make_config, write_config):
        cfg = make_config(tmp_path)
        cfg["evaluation"]["endpoints"] = [{"name": "broken"}]
        with pytest.raises(ValueError, match="base_url' and 'model"):
            Benchmark(str(write_config(cfg, "case6.yaml")))

    def test_unknown_metrics(self, tmp_path, make_config, write_config):
        cfg = make_config(tmp_path)
        cfg["evaluation"]["metrics"] = ["bogus"]
        with pytest.raises(ValueError, match="unknown metrics"):
            Benchmark(str(write_config(cfg, "case7.yaml")))


class TestPipeline:
    def test_load_lists_audio_files_only(self, tmp_path, make_config, make_wav, write_config):
        cfg = make_config(tmp_path)
        speech = Path(cfg["preprocess"]["input_dir"])
        make_wav(speech / "a.wav")
        make_wav(speech / "b.mp3")
        (speech / "c.txt").write_text("not audio")
        bm = Benchmark(str(write_config(cfg, "load.yaml")))
        assert [f.name for f in bm.load()] == ["a.wav", "b.mp3"]

    def test_unknown_process_mode(self, benchmark):
        with pytest.raises(ValueError, match="Unknown mode"):
            benchmark.process("bogus")

    def test_evaluate_requires_staged_dirs(self, tmp_path, make_config, make_wav, write_config):
        cfg = make_config(tmp_path)
        make_wav(Path(cfg["preprocess"]["input_dir"]) / "s1.wav")
        make_wav(Path(cfg["noise_mixing"]["noise_dir"]) / "n1.wav")
        cfg_path = write_config(cfg, "stage.yaml")
        bm = Benchmark(str(cfg_path))
        shutil.rmtree(cfg["evaluation"]["degraded_dir"])
        with pytest.raises(ValueError, match="evaluation.degraded_dir does not exist"):
            bm.process("evaluate")

    def test_full_pipeline(self, tmp_path, make_config, make_wav, write_config, monkeypatch):
        cfg = make_config(tmp_path)
        speech = Path(cfg["preprocess"]["input_dir"])
        noise_dir = Path(cfg["noise_mixing"]["noise_dir"])
        make_wav(speech / "s1.wav")
        make_wav(speech / "s2.wav")
        make_wav(noise_dir / "n1.wav", sr=4000)
        monkeypatch.setattr(evaluate, "Evaluator", FakeEvaluator)

        bm = Benchmark(str(write_config(cfg, "pipe.yaml")))

        bm.process("degrade")
        degraded_dir = Path(cfg["preprocess"]["output_dir"])
        assert sorted(f.name for f in degraded_dir.iterdir()) == ["s1.wav", "s2.wav"]
        data, sr = sf.read(degraded_dir / "s1.wav", dtype="int16")
        assert sr == 4000
        assert data.dtype == np.int16

        bm.process("noise")
        noisy_dir = Path(cfg["noise_mixing"]["output_dir"])
        data, sr = sf.read(noisy_dir / "s1.wav", dtype="int16")
        assert sr == 4000
        assert data.dtype == np.int16

        bm.process("evaluate")
        report = json.loads(Path(cfg["evaluation"]["output_path"]).read_text())
        assert report[0]["n"] == 2
        assert report[0]["endpoint"] == "test-endpoint"