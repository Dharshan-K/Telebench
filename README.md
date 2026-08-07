# Telebench

A benchmark toolkit for evaluating the robustness of Automatic Speech Recognition (ASR) models against telephony-style signal degradations.

Telebench simulates the kind of audio quality loss you get in real phone calls — bandwidth limiting, downsampling, µ-law companding, and background noise mixing — and then measures how well a range of ASR models still transcribe the degraded audio compared to the clean originals.

[![PyPI version](https://img.shields.io/pypi/v/telebench.svg)](https://pypi.org/project/telebench/)
[![Python versions](https://img.shields.io/pypi/pyversions/telebench.svg)](https://pypi.org/project/telebench/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

### From PyPI (recommended for users)

```bash
pip install telebench
```

### From source (recommended for contributors)

```bash
git clone <repo-url> telebench
cd telebench

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e .                     # installs telebench + runtime deps
pip install -e ".[dev]"              # also installs pytest, ruff, mypy, pre-commit
```

The editable install (`-e`) makes the `telebench` command available in your environment and wires your source directory into `sys.path`, so code edits take effect without reinstalling.

Dependencies (runtime): `numpy`, `soundfile`, `pyyaml`, `jiwer`, `openai`, `requests`, `rich`.
The same set is listed in `requirements.txt` for users that prefer `pip install -r requirements.txt`.

## Quickstart

1. Put some clean speech clips (`.wav`/`.mp3`/`.flac`) in `example/dataset/` and a few noise clips in `example/noise/`. (The repo ships empty `example/` directories as a starting point.)
2. Set your API key(s) as environment variables (see [Configuration](#configuration)).
3. Run the whole pipeline:

```bash
telebench eval                       # degrade → noise → evaluate
```

Or step by step:

```bash
telebench degrade                    # FFT filter → downsample → µ-law
telebench noise                      # add background noise
telebench evaluate                   # transcribe + compute metrics
```

The evaluation report is written to `example/output/evaluation.json` and a `rich` table is printed to the console.

## How it works

The pipeline is a three-stage process, each stage writing audio files to a directory referenced by the config. The degraded/normalized audio pairs are then fed to ASR APIs for transcription, and the transcripts are compared using standard text-similarity metrics.

```
speech corpus ──▸ degrade      ──▸ noise mixing ──▸ ASR evaluation ──▸ report
(clean audio)     (filter/       (add noise at       (transcribe         (WER/CER/
                  downsample/    random SNR)         both versions        MER/WIL,
                  µ-law codec)                       via APIs)            latency)
```

### 1. Degrade (`telebench degrade`)

The `degrade` stage simulates an 8 kHz telephone-channel speech codec. For each input audio file:

1. **Load** the audio as mono PCM-16 (`core/process.py:Process.process`).
2. **FFT low-pass filter** — frequencies above the quarter-rate (4000 Hz for 8 kHz audio) are zeroed, simulating band-limiting (`fourier_transform`).
3. **Downsample** by 2 — cutting the sample rate in half.
4. **µ-law compand** (`mu_law`) — a non-linear companding transform typical of G.711 telephony codecs; the sample is normalized, mapped through the µ-law curve with `µ = 255`, and stored back as PCM16.

The output is the "clean" version of the signal *after* channel degradation.

### 2. Noise mixing — `telebench noise`

Random background noise is mixed into each degraded file at a randomly chosen signal-to-noise ratio (SNR):

- A noise file is chosen at random from `noise_mixing.noise_dir`.
- The noise is tiled/truncated to match the speech length.
- An SNR is sampled uniformly from `[snr_db_min, snr_db_max]`.
- The noise amplitude is scaled so the mixed signal hits the target SNR, then both are summed and clipped to PCM16 range (`core/process.py:add_noise`).

Noise and speech must share the same sample rate; the process will fail loudly if they don't (`add_noise` raises on mismatch).

### 3. Evaluation — `telebench evaluate`

Every file in `degraded_dir` is paired with its same-stem counterpart in `original_dir` (unmatched files are skipped with a warning, and each pair is transcribed via one or more configured ASR endpoints):

- The **original** file provides the *reference* transcript.
- The **degraded-noisy** file provides the *hypothesis* transcript.

For each endpoint the runner reports:

- **WER** — word error rate
- **CER** — character error rate
- **MER** — match error rate
- **WIL** — word information lost
- **Latency** — mean round-trip time in seconds per file (averaged across the two transcription calls per pair)

Metrics are computed with [jiwer](https://github.com/jitsi/jiwer); the supported set is fixed to `{wer, cer, mer, wil}` (`core/evaluate.py:VALID_METRICS`).

Transcription pairs with an empty transcript on either side are counted as *skipped* rather than scored.

## Project layout

```
telebench/
├── pyproject.toml           # Packaging: name, deps, entry points, tool configs
├── requirements.txt         # Runtime dependency list (pip install -r)
├── benchmark.py             # Benchmark orchestrator: YAML parsing, validation, stage execution
├── run.py                   # Simple scripted driver (runs degrade → noise → evaluate)
├── example.py               # Original Colab experiment: dataset download, offline repro of pipeline
├── example.yml              # Pipeline config driven by a Hugging Face dataset
├── config.yaml              # Default config (runs against ./example)
├── cli/
│   └── cli.py               # argparse CLI (telebench)
├── core/
│   ├── process.py           # Process: audio loading, FFT filter, µ-law, noise mixing
│   └── evaluate.py          # Transcribers, Evaluator adapters, EvaluationRunner, metrics
├── dataloader/
│   ├── loader.py            # DatasetLoader base class (file iteration)
│   └── localloader.py       # LocalLoader: yields .wav/.mp3/.flac files from a directory
├── example/
│   ├── dataset/             # Clean speech clips
│   ├── noise/               # Background noise clips
│   └── output/              # degraded/, noisy/, evaluation.json
└── tests/                   # pytest suite
```

## Configuration

Telebench is driven entirely by a YAML config. The default config file is `config.yaml`; point at a different one with `-c/--config`.

The config requires four top-level sections — `dataset`, `preprocess`, `noise_mixing` and `evaluation` — and is validated eagerly on startup (missing keys, non-existent input directories, unknown metrics, bad SNR bounds, etc. fail fast with a `ValueError`).

### `dataset`

Declarative description of where the speech and noise corpora come from. The `Benchmark` class does not download anything — the directories listed here are simply created as output locations (see `example.yml` before deciding to use a Hugging Face dataset).

```yaml
dataset:
  speech:
    output_dir: example/dataset
  noise:
    output_dir: example/noise
```

### `preprocess`

Controls the `degrade` stage.

| Key               | Meaning                                         |
|-------------------|-------------------------------------------------|
| `input_dir`       | Directory of clean speech to degrade.           |
| `output_dir`      | Where the degraded files are written.           |
| `normalize_scale` | Optional amplitude used in normalization (default `32768`). |

### `noise_mixing`

Controls the `noise` stage.

| Key          | Meaning                                                            |
|--------------|--------------------------------------------------------------------|
| `input_dir`  | Degraded speech input (normally the preprocess output).            |
| `noise_dir`  | Directory of background noise clips. Must contain audio.           |
| `output_dir` | Where noisy files are written.                                      |
| `snr_db_min` | Lower bound of uniform SNR range (must be < `snr_db_max`).         |
| `snr_db_max` | Upper bound of uniform SNR range.                                   |

### `evaluation`

Configures ASR endpoints and the comparison.

| Key              | Meaning                                                     |
|------------------|-------------------------------------------------------------|
| `original_dir`   | Clean speech reference files.                                  |
| `degraded_dir`   | Noisy audio to be scored (normally `noise_mixing.output_dir`). |
| `output_path`    | Where the JSON report is written. Omit for console-only output. |
| `metrics`        | List of metrics to report; default `[wer, cer]`; valid: `{wer, cer, mer, wil}`. |
| `temperature`    | Default transcription temperature (applied to OpenAI-compatible endpoints). |
| `response_format`| Default `verbose_json`.                                         |
| `language`       | Spoken language of the clips (2-letter ISO code for Sravam mapping). |
| `endpoints`      | List of ASR endpoints to evaluate (see below).                    |

#### Endpoint entries

Each entry describes one ASR model:

| Key            | Meaning                                                      |
|----------------|--------------------------------------------------------------|
| `name`         | Display name. Auto-set from `base_url:model` if omitted.     |
| `provider`     | `openai` or `sarvam`. Auto-detected from the URL if omitted (`sarvam.ai` → Sarvam, anything else → OpenAI-compatible). |
| `base_url`     | API base URL (OpenAI-compatible) or Sravam speech-to-text endpoint. |
| `model`        | Model identifier for the endpoint.                           |
| `api_key_env`  | Env var holding the API key. Falls back to `OPENAI_API_KEY`, then to a `not-needed` placeholder (allows the test fixtures and local servers to run without keys). |

### Example config

```yaml
evaluation:
  endpoints:
    - name: groq-whisper-large-v3
      provider: openai
      base_url: https://api.groq.com/openai/v1
      api_key_env: GROQ_API_KEY
      model: whisper-large-v3
    - name: sarvam-saaras-v3
      provider: sarvam
      base_url: https://api.sarvam.ai/speech-to-text
      api_key_env: SARVAM_API_KEY
      model: saaras:v3
  temperature: 0
  response_format: verbose_json
  language: ta
  metrics: [wer, cer, mer, wil]
  original_dir: example/dataset
  degraded_dir: example/output/noisy
  output_path: example/output/evaluation.json
```

## Usage

### Command line

After the editable install the `telebench` command is on your `PATH`. (Before installing — or to run without an install — use `python -m cli` with the exact same arguments.)

The `-c/--config` flag is a global option and must come **before** the subcommand:

```bash
telebench -c example.yml eval      # run the pipeline with a different config
```

Run the whole pipeline (degrade → noise → evaluate) with the default config:

```bash
telebench eval
```

Run a single stage:

```bash
telebench degrade          # 1. FFT-filter → downsample → µ-law
telebench noise            # 2. add background noise
telebench evaluate         # 3. transcribe + compute metrics
```

Inspect the config without executing anything:

```bash
telebench info             # print resolved paths + endpoints
telebench endpoints        # list configured ASR endpoints
```

Analyze only selected endpoints:

```bash
telebench evaluate -e groq-whisper-large-v3
```

Show CLI help:

```bash
telebench --help           # or: telebench -h
```

### The `Benchmark` API (e.g. from Jupyter, inside modules)

```python
from benchmark import Benchmark

bench = Benchmark("config.yaml")   # loads + validates config
bench.load()                       # list input audio files
bench.process("degrade")           # stage 1
bench.process("noise")             # stage 2
bench.process("evaluate")          # stage 3
```

## Output report

`evaluation.json` (configured by `evaluation.output_path`) holds a JSON array with one object per endpoint:

```json
[
  {
    "endpoint": "groq-whisper-large-v3",
    "model": "whisper-large-v3",
    "n": 2,
    "skipped": 0,
    "metrics": { "wer": 0.83, "cer": 0.59, "mer": 0.83, "wil": 0.96 },
    "latency_mean_s": 1.30
  }
]
```

`n` is the number of scored pairs, `skipped` counts pairs dropped due to empty transcripts, and `latency_mean_s` is the mean seconds per file. A `rich` table of the same data is printed to the console.

## Adding your own datasets / endpoints

- **Datasets**: drop `.wav`/`.mp3`/`.flac` files into the directories referenced by the config. The `LocalLoader` in `dataloader/` iterates one directory (non-recursively) as filters by extension, expanding `~`.
- **Endpoints**: add an entry under `evaluation.endpoints`. OpenAI-compatible providers are supported out of the box (Groq, OpenAI, vLLM, local servers, ...). A dedicated adapter — `SarvamTranscriber` — exists for Sarvam's Speech-to-Text Mosaix API (`api.sarvam.ai/speech-to-text`) and speaks `language_code` as a BCP-47 tag (e.g. `ta-IN`).

## Testing

The repo has a pytest suite exercising the pipeline stages, config validation and CLI behaviors in `tests/`:

```
pytest
```

The `tests/` use synthetic WAV fixtures (see `tests/conftest.py`) and a `FakeEvaluator` that avoids hitting external APIs.

Development tooling is wired into `pyproject.toml` and installed via `pip install -e ".[dev]"`:

```
ruff check .        # lint
ruff format .       # format
mypy .              # type check
pytest              # run tests
```

## Example experiment

`example.py` is a self-contained Colab notebook script that mirrors the exact pipeline against a Hugging Face dataset (`ai4bharat/Kathbath`, Tamil speech) plus Musan noise, then transcribes originals vs noisy clips with Groq's `whisper-large-v3`. `example.yml` is the corresponding config for the reproduction.

## Notes & warnings

- Degraded and noisy audio must share sample rates — the cross-mixer raises `ValueError` on a mismatch rather than resampling.
- The code does **not** resample noise; it assumes noise is served at the same rate as the degraded speech.
- API keys are read from env variables (`GROQ_API_KEY`, `SARVAM_API_KEY`, ... or `OPENAI_API_KEY` as a fallback). Never don't commit real keys; the ones that appear in `example.py` are placeholders from an old notebook.
```