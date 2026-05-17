# stt-eval

`stt-eval` 是用來評估 `MediaTek-Research/Breeze-ASR-26` 量化版本的工具。

目前保留兩類功能：

- 模型準備與量化：下載 HF 原始模型，轉成 CT2 與 GGML 量化 artifacts。
- 評估與彙總：呼叫 OpenAI-compatible STT API，計算 CER / RTF，輸出與彙總 JSONL 結果。

## 安裝

專案使用 `uv` 管理環境：

```bash
uv sync
```

需要跑模型轉換時安裝 extras：

```bash
uv sync --extra hf --extra ct2
```

## 準備模型

下載 HF 原始模型並產生所有量化版本：

```bash
uv run --extra hf --extra ct2 stt-eval prepare-models
```

只產生單一版本：

```bash
uv run --extra hf --extra ct2 stt-eval prepare-models \
  --variant Breeze-ASR-26-int8-CT2
```

Artifacts 會放在 `artifacts/models/`，命名格式如下：

- `Breeze-ASR-26-<量化版本>-CT2`
- `Breeze-ASR-26-<量化版本>-GGML`

例如：

- `Breeze-ASR-26-int8-CT2`
- `Breeze-ASR-26-q8_0-GGML`

## 評估

準備 manifest JSONL：

```jsonl
{"id":"sample-001","audio_path":"audio/sample.wav","reference":"參考逐字稿","duration_sec":3.2}
```

執行評估：

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

`base-url` 需指向相容 OpenAI audio transcriptions API 的服務。

## 彙總結果

```bash
uv run stt-eval summarize \
  --results artifacts/results/Breeze-ASR-26-int8-CT2.jsonl
```

會輸出 sample 數、平均 CER、平均 RTF。

## 檢查 artifacts

```bash
uv run stt-eval validate-artifacts
```

會確認每個 artifact 目錄是否包含：

- `metadata.json`
- `README.md`
- `QUANTIZATION.md`
