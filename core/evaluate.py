import json
import os
import time
from abc import ABC, abstractmethod

import jiwer
import requests
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from dataloader.localloader import LocalLoader

VALID_METRICS = {"wer", "cer", "mer", "wil"}


def _resolve_api_key(endpoint):
    env_name = endpoint.get("api_key_env")
    if env_name:
        key = os.environ.get(env_name)
        if key:
            return key
        print(f"WARNING: env var {env_name!r} not set, falling back to OPENAI_API_KEY.")
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    print("WARNING: no API key found, using placeholder (local servers only).")
    return "not-needed"


class BaseTranscriber(ABC):
    @abstractmethod
    def transcribe(self, path: str) -> tuple[str, float]:
        """Transcribe audio file at path and return tuple (transcript_text, latency_seconds)."""
        pass


class OpenAITranscriber(BaseTranscriber):
    """Adapter for OpenAI-compatible speech-to-text APIs (Groq, OpenAI, etc.)."""

    def __init__(self, endpoint, defaults):
        self.model = endpoint["model"]
        self.temperature = endpoint.get("temperature", defaults.get("temperature"))
        self.response_format = endpoint.get(
            "response_format", defaults.get("response_format", "verbose_json")
        )
        self.language = endpoint.get("language", defaults.get("language"))
        self.client = OpenAI(
            base_url=endpoint["base_url"],
            api_key=_resolve_api_key(endpoint),
        )

    def transcribe(self, path: str) -> tuple[str, float]:
        with open(path, "rb") as f:
            start = time.perf_counter()
            result = self.client.audio.transcriptions.create(
                file=f,
                model=self.model,
                temperature=self.temperature,
                response_format=self.response_format,
                language=self.language,
            )
            elapsed = time.perf_counter() - start
        return result.text, elapsed


class SarvamTranscriber(BaseTranscriber):
    """Adapter for Sarvam AI Speech-to-Text API."""

    LANGUAGE_MAP = {
        "ta": "ta-IN",
        "hi": "hi-IN",
        "en": "en-IN",
        "te": "te-IN",
        "kn": "kn-IN",
        "ml": "ml-IN",
        "mr": "mr-IN",
        "bn": "bn-IN",
        "gu": "gu-IN",
        "pa": "pa-IN",
        "od": "od-IN",
    }

    def __init__(self, endpoint, defaults):
        self.base_url = endpoint.get("base_url", "https://api.sarvam.ai/speech-to-text")
        self.api_key = _resolve_api_key(endpoint)
        self.model = endpoint.get("model", "saaras:v3")
        lang = endpoint.get("language", defaults.get("language", "unknown"))
        self.language_code = self.LANGUAGE_MAP.get(lang, lang)

    def transcribe(self, path: str) -> tuple[str, float]:
        headers = {
            "api-subscription-key": self.api_key,
        }
        with open(path, "rb") as f:
            files = {
                "file": (os.path.basename(path), f, "audio/wav")
            }
            data = {
                "model": self.model,
            }
            if self.language_code:
                data["language_code"] = self.language_code

            start = time.perf_counter()
            response = requests.post(self.base_url, headers=headers, files=files, data=data)
            elapsed = time.perf_counter() - start

            response.raise_for_status()
            res_json = response.json()
            text = res_json.get("transcript", "")
            return text, elapsed


class Evaluator:
    def __init__(self, endpoint, defaults):
        self.name = endpoint.get("name") or f"{endpoint['base_url']}:{endpoint['model']}"
        self.model = endpoint["model"]

        provider = endpoint.get("provider")
        if not provider:
            base_url = endpoint.get("base_url", "")
            if "sarvam.ai" in base_url:
                provider = "sarvam"
            else:
                provider = "openai"

        if provider == "sarvam":
            self.transcriber = SarvamTranscriber(endpoint, defaults)
        elif provider == "openai":
            self.transcriber = OpenAITranscriber(endpoint, defaults)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def transcribe(self, path: str) -> tuple[str, float]:
        return self.transcriber.transcribe(path)


class EvaluationRunner:
    def __init__(self, evaluation_cfg):
        self.cfg = evaluation_cfg
        self.metrics = [m for m in evaluation_cfg.get("metrics", ["wer", "cer"]) if m in VALID_METRICS]

    def run(self):
        rows = [self._evaluate_endpoint(ep) for ep in self.cfg["endpoints"]]
        self._print(rows)
        self._write_json(rows)
        return rows

    def _pair_files(self):
        original_dir = self.cfg["original_dir"]
        degraded_dir = self.cfg["degraded_dir"]
        originals = {f.stem: f for f in LocalLoader(original_dir)}
        degraded = {f.stem: f for f in LocalLoader(degraded_dir)}
        pairs = [(originals[stem], degraded[stem]) for stem in sorted(originals.keys() & degraded.keys())]
        unmatched = set(originals) ^ set(degraded)
        if unmatched:
            print(f"WARNING: {len(unmatched)} files without a match skipped: {sorted(unmatched)[:5]}")
        return pairs

    def _evaluate_endpoint(self, endpoint):
        evaluator = Evaluator(endpoint, self.cfg)
        pairs = self._pair_files()

        refs, hyps = [], []
        latencies = []
        skipped = 0
        for original_path, degraded_path in pairs:
            ref, lat_orig = evaluator.transcribe(str(original_path))
            hyp, lat_degraded = evaluator.transcribe(str(degraded_path))
            latencies.append((lat_orig + lat_degraded) / 2)
            if not ref.strip() or not hyp.strip():
                skipped += 1
                continue
            refs.append(ref)
            hyps.append(hyp)

        measures = {}
        if refs:
            words = jiwer.process_words(refs, hyps)
            chars = jiwer.process_characters(refs, hyps)
            measures = {
                "wer": words.wer,
                "mer": words.mer,
                "wil": words.wil,
                "cer": chars.cer,
            }
        return {
            "endpoint": evaluator.name,
            "model": evaluator.model,
            "n": len(refs),
            "skipped": skipped,
            "metrics": {m: measures.get(m) for m in self.metrics},
            "latency_mean_s": sum(latencies) / len(latencies) if latencies else None,
        }

    def _print(self, rows):
        table = Table(title="Evaluation Results")
        table.add_column("Endpoint", justify="left", style="cyan", no_wrap=True)
        table.add_column("n", justify="right")
        for m in self.metrics:
            table.add_column(m.upper(), justify="right")
        table.add_column("Latency (s/file)", justify="right")

        for row in rows:
            values = [row["endpoint"], str(row["n"])]
            values += [
                f"{row['metrics'][m]:.4f}" if row["metrics"][m] is not None else "n/a"
                for m in self.metrics
            ]
            values.append(
                f"{row['latency_mean_s']:.2f}"
                if row["latency_mean_s"] is not None
                else "n/a"
            )
            table.add_row(*values)

        Console().print(table)

    def _write_json(self, rows):
        output_path = self.cfg.get("output_path")
        if not output_path:
            return
        with open(output_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Report written to {output_path}")
