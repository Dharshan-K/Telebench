# Telebench

Telebench is a tool to test how well speech-to-text (ASR) models handle phone-call-quality audio.

It takes clean speech, applies typical telephony degradations (bandwidth filtering, downsampling, mu-law companding), mixes in background noise, and measures how well ASR models still transcribe the degraded audio. The report shows word/character error rates and latency per model.

## Install

```bash
pip install telebench
```

## Quick start

Run the whole pipeline with the config and example audio that ship inside the package — no setup needed:

```bash
telebench default
```

This runs all three stages (degrade -> noise -> evaluate) against the bundled example clips and prints a results table:

```
                                 Evaluation Results
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Endpoint              ┃ n ┃    WER ┃    CER ┃    MER ┃    WIL ┃ Latency (s/file) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ groq-whisper-large-v3 │ 2 │ 1.0000 │ 0.5941 │ 0.8571 │ 0.9667 │             1.90 │
│ sarvam-saaras-v3      │ 2 │ 0.3810 │ 0.2632 │ 0.3810 │ 0.5976 │             1.90 │
└───────────────────────┴───┴────────┴────────┴────────┴────────┴──────────────────┘
Report written to example/output/evaluation.json
Done.
```

The report is also saved to `evaluation.json` in the config's `output_path`.

## API keys

Each endpoint in the config reads its key from an environment variable (set with `api_key_env` in the config, e.g. `GROQ_API_KEY`, `SARVAM_API_KEY`). `OPENAI_API_KEY` is used as a fallback.

```bash
export GROQ_API_KEY=your-key
export SARVAM_API_KEY=your-key
```

If a key is missing, telebench prints a warning and skips that endpoint. It never fails.

## Commands

| Command | What it does |
|---|---|
| `telebench default` | Run the full pipeline using the config and audio bundled in the package |
| `telebench eval` | Run the full pipeline using `config.yaml` in the current folder |
| `telebench degrade` | Stage 1: apply telephone degradation (FFT filter, downsample, mu-law) |
| `telebench noise` | Stage 2: mix background noise into the degraded files |
| `telebench evaluate` | Stage 3: transcribe audio via the configured ASR endpoints and compute metrics |
| `telebench endpoints` | List the configured ASR endpoints |
| `telebench info` | Show the resolved paths and endpoints from the config |
| `telebench -h` | Show help |

Use `-c <file>` to point at your own config (must come before the command):

```bash
telebench -c /path/to/my-config.yaml eval
```

## Using your own audio

1. Copy the config and example data that came with the package, or write your own YAML config (see below).
2. Put your clean speech clips in `preprocess.input_dir` and noise clips in `noise_mixing.noise_dir`.
3. Run:

```bash
telebench -c config.yaml eval
```

## What the pipeline does

```
speech corpus -> degrade -> noise mixing -> ASR evaluation -> report
(clean audio)    (band-limit,   (mix noise at    (transcribe both     (WER, CER,
                 downsample,    random SNR)      original + noisy)    MER, WIL,
                 mu-law)                                             latency)
```

- **Degrade** simulates an 8 kHz telephone channel: low-pass filter above 4 kHz, downsample by 2, then mu-law companding.
- **Noise** mixes a random noise clip into each degraded file at a random SNR (default range 5-15 dB).
- **Evaluate** transcribes each original and noisy pair through every configured endpoint and compares them with [jiwer](https://github.com/jitsi/jiwer) metrics: WER, CER, MER, WIL, plus mean latency per file.

## Config

Telebench is driven by a YAML config with four sections: `dataset`, `preprocess`, `noise_mixing`, `evaluation`. A minimal example:

```yaml
preprocess:
  input_dir: example/dataset
  output_dir: example/output/degraded

noise_mixing:
  input_dir: example/output/degraded
  noise_dir: example/noise
  output_dir: example/output/noisy
  snr_db_min: 5
  snr_db_max: 15

evaluation:
  metrics: [wer, cer, mer, wil]
  original_dir: example/dataset
  degraded_dir: example/output/noisy
  output_path: example/output/evaluation.json
  endpoints:
    - name: groq-whisper-large-v3
      provider: openai
      base_url: https://api.groq.com/openai/v1
      api_key_env: GROQ_API_KEY
      model: whisper-large-v3
```

Add as many endpoints as you like: any OpenAI-compatible provider works (Groq, OpenAI, vLLM, local servers), and Sarvam's Speech-to-Text API is supported with `provider: sarvam`.

## Development

```bash
git clone <repo-url> && cd telebench
pip install -e ".[dev]"
pytest
ruff check .
mypy .
```