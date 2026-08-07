import json
from pathlib import Path

import pytest

from cli.cli import main
from core import evaluate
from fakes import FakeEvaluator


@pytest.fixture
def project(tmp_path, make_config, make_wav, write_config):
    """A valid config on disk with speech + noise clips ready for degrade."""
    cfg = make_config(tmp_path)
    make_wav(Path(cfg["preprocess"]["input_dir"]) / "s1.wav")
    make_wav(Path(cfg["noise_mixing"]["noise_dir"]) / "n1.wav")
    return cfg, write_config(cfg)


def test_info_prints_paths(project, capsys):
    _, cfg_path = project
    assert main(["info", "-c", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "preprocess input:" in out
    assert "preprocess output:" in out
    assert "noise output:" in out


def test_endpoints_lists_configured_endpoints(project, capsys):
    _, cfg_path = project
    assert main(["endpoints", "-c", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "endpoints:" in out
    assert "test-endpoint" in out
    assert "fake-model" in out


def test_unknown_endpoint_fails(project, capsys):
    _, cfg_path = project
    with pytest.raises(SystemExit, match="Unknown endpoint"):
        main(["evaluate", "-e", "nope", "-c", str(cfg_path)])


def test_degrate_writes_degraded_files(project, capsys):
    cfg, cfg_path = project
    assert main(["degrade", "-c", str(cfg_path)]) == 0
    degraded = Path(cfg["preprocess"]["output_dir"])
    assert (degraded / "s1.wav").exists()
    assert "Running degrade..." in capsys.readouterr().out


def test_eval_runs_full_pipeline(project, tmp_path, make_wav, capsys, monkeypatch):
    cfg, cfg_path = project
    noise_dir = Path(cfg["noise_mixing"]["noise_dir"])
    (noise_dir / "n1.wav").unlink()
    make_wav(noise_dir / "n1.wav", sr=4000)
    monkeypatch.setattr(evaluate, "Evaluator", FakeEvaluator)

    assert main(["eval", "-c", str(cfg_path)]) == 0
    report = json.loads(Path(cfg["evaluation"]["output_path"]).read_text())
    assert report[0]["n"] == 1
    assert report[0]["metrics"]["wer"] == pytest.approx(0.0)
    assert "Report written to" in capsys.readouterr().out