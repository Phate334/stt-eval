# Breeze-ASR-26 Quantization Evaluation Plan

## Goal

Evaluate the CER impact of CT2 and GGML quantization for
`MediaTek-Research/Breeze-ASR-26`.

## Artifact Names

- Baseline HF snapshot: `artifacts/models/Breeze-ASR-26-f32-HF/`
- CT2 artifacts:
  - `artifacts/models/Breeze-ASR-26-float16-CT2/`
  - `artifacts/models/Breeze-ASR-26-int8_float16-CT2/`
  - `artifacts/models/Breeze-ASR-26-int8-CT2/`
- GGML artifacts:
  - `artifacts/models/Breeze-ASR-26-GGML/`

Each generated artifact directory contains model files plus `metadata.json`,
`README.md`, and `QUANTIZATION.md`. The GGML artifact directory keeps all
quantized files for the same model in one repository-style directory, for
example `ggml-model-q8_0.bin` and `ggml-model-q4_0.bin`.

## Model Preparation

Run all conversions:

```bash
uv run --extra hf --extra ct2 stt-eval prepare-models
```

Run a single variant:

```bash
uv run --extra hf --extra ct2 stt-eval prepare-models \
  --variant Breeze-ASR-26-int8-CT2
```

The preparation flow resolves the Hugging Face revision, downloads the source
snapshot, converts CT2 artifacts with `ct2-transformers-converter`, and uses
whisper.cpp tools to create GGML quantized artifacts.

## Evaluation

Manifest JSONL fields:

- `id`
- `audio_path`
- `reference`
- optional `duration_sec`

Run evaluation against any OpenAI-compatible transcription endpoint:

```bash
uv run stt-eval run-eval \
  --manifest data/manifest.jsonl \
  --output artifacts/results/Breeze-ASR-26-int8-CT2.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model Breeze-ASR-26-int8-CT2 \
  --backend ct2 \
  --quantization int8 \
  --model-artifact-dir artifacts/models/Breeze-ASR-26-int8-CT2
```

Summarize results:

```bash
uv run stt-eval summarize --results artifacts/results/Breeze-ASR-26-int8-CT2.jsonl
```

## Metric

CER is the primary metric. Normalization applies Unicode normalization,
full-width/half-width normalization, ASCII lowercasing, whitespace removal, and
common Chinese/ASCII punctuation removal.
