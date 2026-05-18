# Quantization

This project prepares local artifacts for evaluating the quantization impact of
`MediaTek-Research/Breeze-ASR-26`.

The goal is not to train a new model. Each artifact is derived from the same
source model revision and is intended for comparing runtime behavior and CER
under different inference backends and quantization types.

## Targets

- **HF baseline**: an unquantized Transformers snapshot used as the source of
  all converted artifacts.
- **CTranslate2**: converted model directories for faster-whisper /
  CTranslate2-style serving. The current target quantization types are
  `float16`, `int8_float16`, and `int8`.
- **GGML / whisper.cpp**: one whisper.cpp-style model directory containing
  multiple GGML files for the same source model. The current target
  quantization types are `q8_0`, `q5_0`, `q4_0`, and `q4_1`.

## Artifact Layout

- Baseline artifact: `artifacts/models/Breeze-ASR-26-f32-HF/`
- CT2 artifacts:
  - `artifacts/models/Breeze-ASR-26-float16-CT2/`
  - `artifacts/models/Breeze-ASR-26-int8_float16-CT2/`
  - `artifacts/models/Breeze-ASR-26-int8-CT2/`
- GGML artifact:
  - `artifacts/models/Breeze-ASR-26-GGML/`

The GGML directory follows the common GGML convention of keeping multiple
quantized files for the same model in one repository-style directory, for
example `ggml-model-q8_0.bin` and `ggml-model-q4_0.bin`.

Each artifact directory contains model files plus generated metadata files:

- `README.md`: Hugging Face model card metadata and source model attribution.
- `metadata.json`: local preparation metadata.
- `QUANTIZATION.md`: exact conversion or quantization command records.

## Preparation

Run all artifact preparation through `uv`:

```bash
uv run --extra hf --extra ct2 stt-eval prepare-models
```

Prepare only selected variants by passing `--variant` multiple times:

```bash
uv run --extra hf --extra ct2 stt-eval prepare-models \
  --variant Breeze-ASR-26-int8-CT2 \
  --variant Breeze-ASR-26-q4_0-GGML
```

The preparation flow downloads the HF source snapshot if needed, ensures the HF
tokenizer assets required by CT2 exist, converts CT2 artifacts, and uses
whisper.cpp tools to produce GGML files.

## Validation

Check the generated artifact layout and metadata files:

```bash
uv run stt-eval validate-artifacts
```
