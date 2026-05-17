# Quantization

Model download, conversion, and quantization are local development tasks managed
by `uv`.

Run all artifact preparation:

```bash
uv run --extra hf --extra ct2 stt-eval prepare-models
```

## Artifact Layout

- Baseline artifact: `artifacts/models/Breeze-ASR-26-f32-HF/`
- CT2 artifacts:
  - `artifacts/models/Breeze-ASR-26-float16-CT2/`
  - `artifacts/models/Breeze-ASR-26-int8_float16-CT2/`
  - `artifacts/models/Breeze-ASR-26-int8-CT2/`
- GGML artifacts:
  - `artifacts/models/Breeze-ASR-26-q8_0-GGML/`
  - `artifacts/models/Breeze-ASR-26-q5_0-GGML/`
  - `artifacts/models/Breeze-ASR-26-q4_0-GGML/`
  - `artifacts/models/Breeze-ASR-26-q4_1-GGML/`

## CT2

The CT2 variants are produced from the local HF snapshot with
`ct2-transformers-converter`.

```bash
uv run --extra ct2 ct2-transformers-converter \
  --model artifacts/models/Breeze-ASR-26-f32-HF \
  --output_dir artifacts/models/Breeze-ASR-26-float16-CT2 \
  --quantization float16 --force

uv run --extra ct2 ct2-transformers-converter \
  --model artifacts/models/Breeze-ASR-26-f32-HF \
  --output_dir artifacts/models/Breeze-ASR-26-int8_float16-CT2 \
  --quantization int8_float16 --force

uv run --extra ct2 ct2-transformers-converter \
  --model artifacts/models/Breeze-ASR-26-f32-HF \
  --output_dir artifacts/models/Breeze-ASR-26-int8-CT2 \
  --quantization int8 --force
```

## GGML

GGML artifacts are produced with whisper.cpp conversion and quantization tools.

```bash
python artifacts/tools/whisper.cpp/models/convert-h5-to-ggml.py \
  artifacts/models/Breeze-ASR-26-f32-HF \
  artifacts/tools/openai-whisper \
  artifacts/tools/whispercpp-base

artifacts/tools/whisper.cpp/build/bin/quantize \
  artifacts/tools/whispercpp-base/ggml-model.bin \
  artifacts/models/Breeze-ASR-26-q8_0-GGML/ggml-model-q8_0.bin \
  q8_0

artifacts/tools/whisper.cpp/build/bin/quantize \
  artifacts/tools/whispercpp-base/ggml-model.bin \
  artifacts/models/Breeze-ASR-26-q5_0-GGML/ggml-model-q5_0.bin \
  q5_0

artifacts/tools/whisper.cpp/build/bin/quantize \
  artifacts/tools/whispercpp-base/ggml-model.bin \
  artifacts/models/Breeze-ASR-26-q4_0-GGML/ggml-model-q4_0.bin \
  q4_0

artifacts/tools/whisper.cpp/build/bin/quantize \
  artifacts/tools/whispercpp-base/ggml-model.bin \
  artifacts/models/Breeze-ASR-26-q4_1-GGML/ggml-model-q4_1.bin \
  q4_1
```
