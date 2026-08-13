import json
from pathlib import Path

import pytest

from telebench.core import evaluate
from telebench.core.evaluate import (
    Evaluator,
    EvaluationRunner,
    OpenAITranscriber,
    SarvamTranscriber,
    VALID_METRICS,
    _resolve_api_key,
)
from fakes import FakeEvaluator


@pytest.fixture
def evaluation_cfg(tmp_path, make_config):
    return make_config(tmp_path)["evaluation"]


@pytest.fixture
def paired_files(tmp_path, make_wav, make_config):
    cfg = make_config(tmp_path)["evaluation"]
    make_wav(Path(cfg["original_dir"]) / "clip1.wav")
    make_wav(Path(cfg["degraded_dir"]) / "clip1.wav")
    return cfg


class TestResolveApiKey:
    def test_uses_named_env_var(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "secret")
        assert _resolve_api_key({"api_key_env": "GROQ_API_KEY"}) == "secret"

    def test_falls_back_to_openai_api_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        assert _resolve_api_key({"api_key_env": "GROQ_API_KEY"}) == "openai-key"

    def test_uses_openai_api_key_without_env_name(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        assert _resolve_api_key({}) == "openai-key"

    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert _resolve_api_key({"api_key_env": "GROQ_API_KEY"}) is None


class TestEvaluatorSelectsAdapter:
    def test_defaults_to_openai_adapter(self, monkeypatch):
        calls = {}

        def fake_openai(endpoint, defaults):
            calls["adapter"] = "openai"
            return FakeEvaluator(endpoint, defaults)

        monkeypatch.setattr(evaluate, "OpenAITranscriber", fake_openai)
        ev = Evaluator({"base_url": "http://groq.example/v1", "model": "m"}, {})
        assert ev.transcriber is not None
        assert calls["adapter"] == "openai"

    def test_selects_sarvam_adapter_by_provider(self, monkeypatch):
        calls = {}

        def fake_sarvam(endpoint, defaults):
            calls["adapter"] = "sarvam"
            return FakeEvaluator(endpoint, defaults)

        monkeypatch.setattr(evaluate, "SarvamTranscriber", fake_sarvam)
        ev = Evaluator(
            {"provider": "sarvam", "base_url": "https://api.sarvam.ai/", "model": "m"},
            {},
        )
        assert ev.transcriber is not None
        assert calls["adapter"] == "sarvam"

    def test_auto_detects_sarvam_by_url(self, monkeypatch):
        calls = {}

        def fake_sarvam(endpoint, defaults):
            calls["adapter"] = "sarvam"
            return FakeEvaluator(endpoint, defaults)

        monkeypatch.setattr(evaluate, "SarvamTranscriber", fake_sarvam)
        ev = Evaluator({"base_url": "https://api.sarvam.ai/stt", "model": "m"}, {})
        assert ev.transcriber is not None
        assert calls["adapter"] == "sarvam"

    def test_raises_on_unknown_provider(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            Evaluator({"provider": "wat", "base_url": "x", "model": "m"}, {})


class TestSarvamTranscriber:
    def test_maps_iso_language_to_bcp47(self, monkeypatch):
        monkeypatch.setenv("SARVAM_API_KEY", "secret")
        tr = SarvamTranscriber({"language": "ta"}, {})
        assert tr.language_code == "ta-IN"

    def test_keeps_full_language_code(self, monkeypatch):
        monkeypatch.setenv("SARVAM_API_KEY", "secret")
        tr = SarvamTranscriber({"language": "ta-IN"}, {})
        assert tr.language_code == "ta-IN"

    def test_transcribe_sends_form_and_parses_transcript(self, make_wav, tmp_path, monkeypatch):
        monkeypatch.setenv("SARVAM_API_KEY", "secret")
        audio = make_wav(tmp_path / "clip.wav")
        tr = SarvamTranscriber({"language": "ta", "api_key_env": "SARVAM_API_KEY"}, {})

        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"transcript": "வணக்கம்"}

        def fake_post(url, headers, files, data):
            captured["url"] = url
            captured["headers"] = headers
            captured["data"] = data
            captured["files"] = files
            return FakeResponse()

        monkeypatch.setattr(evaluate.requests, "post", fake_post)
        text, _ = tr.transcribe(str(audio))
        assert text == "வணக்கம்"
        assert captured["url"] == "https://api.sarvam.ai/speech-to-text"
        assert captured["headers"]["api-subscription-key"] == "secret"
        assert captured["data"]["model"] == "saaras:v3"
        assert captured["data"]["language_code"] == "ta-IN"

    def test_transcribe_uses_endpoint_base_url(self, make_wav, tmp_path, monkeypatch):
        monkeypatch.setenv("SARVAM_API_KEY", "secret")
        audio = make_wav(tmp_path / "clip.wav")
        tr = SarvamTranscriber(
            {"base_url": "https://proxy.example/custom", "language": "en"}, {}
        )
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"transcript": "hello"}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return FakeResponse()

        monkeypatch.setattr(evaluate.requests, "post", fake_post)
        text, _ = tr.transcribe(str(audio))
        assert captured["url"] == "https://proxy.example/custom"
        assert text == "hello"


class FakeOpenAI:
    pass


class TestConfiguration:
    def test_valid_metrics_are_exact(self):
        assert VALID_METRICS == {"wer", "cer", "mer", "wil"}

    def test_metrics_default_to_wer_cer(self, evaluation_cfg):
        evaluation_cfg.pop("metrics")
        assert EvaluationRunner(evaluation_cfg).metrics == ["wer", "cer"]

    def test_metrics_filter_unknown(self, evaluation_cfg):
        evaluation_cfg["metrics"] = ["wer", "bogus"]
        assert EvaluationRunner(evaluation_cfg).metrics == ["wer"]


class TestPairFiles:
    def test_pairs_files_by_stem(self, paired_files):
        pairs = EvaluationRunner(paired_files)._pair_files()
        assert [(o.name, d.name) for o, d in pairs] == [("clip1.wav", "clip1.wav")]

    def test_warns_on_unmatched_files(self, tmp_path, make_wav, make_config, capsys):
        cfg = make_config(tmp_path)["evaluation"]
        make_wav(Path(cfg["original_dir"]) / "clip1.wav")
        make_wav(Path(cfg["original_dir"]) / "orphan.wav")
        make_wav(Path(cfg["degraded_dir"]) / "clip1.wav")
        pairs = EvaluationRunner(cfg)._pair_files()
        assert len(pairs) == 1
        assert "orphan.wav" in capsys.readouterr().out


class TestRun:
    def test_run_evaluates_and_writes_report(self, paired_files, monkeypatch):
        monkeypatch.setattr(evaluate, "Evaluator", FakeEvaluator)
        rows = EvaluationRunner(paired_files).run()
        assert len(rows) == 1
        row = rows[0]
        assert row["endpoint"] == "test-endpoint"
        assert row["model"] == "fake-model"
        assert row["n"] == 1
        assert row["skipped"] == 0
        assert row["metrics"]["wer"] == pytest.approx(0.0)
        assert row["metrics"]["cer"] == pytest.approx(0.0)
        assert row["latency_mean_s"] == pytest.approx(0.1)
        report = json.loads(Path(paired_files["output_path"]).read_text())
        assert report[0]["n"] == 1

    def test_run_reports_mismatched_transcripts(self, paired_files, monkeypatch):
        paired_files["endpoints"][0]["hyp"] = "hello world test"
        monkeypatch.setattr(evaluate, "Evaluator", FakeEvaluator)
        row = EvaluationRunner(paired_files).run()[0]
        assert row["metrics"]["wer"] == pytest.approx(0.5)
        assert row["metrics"]["cer"] > 0.0

    def test_run_skips_empty_transcripts(self, paired_files, monkeypatch):
        paired_files["endpoints"][0]["hyp_empty"] = True
        monkeypatch.setattr(evaluate, "Evaluator", FakeEvaluator)
        row = EvaluationRunner(paired_files).run()[0]
        assert row["n"] == 0
        assert row["skipped"] == 1
        assert row["metrics"]["wer"] is None

    def test_run_without_output_path(self, paired_files, monkeypatch):
        paired_files["output_path"] = None
        monkeypatch.setattr(evaluate, "Evaluator", FakeEvaluator)
        rows = EvaluationRunner(paired_files).run()
        assert rows[0]["n"] == 1