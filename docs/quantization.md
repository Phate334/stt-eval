# 量化流程

本專案會在本機準備 `MediaTek-Research/Breeze-ASR-26` 的模型產物（artifact），用來評估不同模型格式與量化設定對輸出結果的影響。

這裡的目標不是訓練新模型，而是從同一個來源模型版本產生可重現的模型產物，後續用相同音檔比較推論輸出與 CER-like drift。

## 產出目標

- **HF baseline**：未量化的 Transformers snapshot，作為所有轉換流程的來源。
- **CTranslate2（CT2）**：提供 faster-whisper / CTranslate2 類型服務使用的模型目錄，目前準備 `float16`、`int8_float16`、`int8`。
- **whisper.cpp / GGML**：提供 whisper.cpp 使用的 GGML 模型檔，目前準備 `q8_0`、`q5_0`、`q4_0`、`q4_1`。

## 產物目錄

HF baseline：

```text
artifacts/models/Breeze-ASR-26-f32-HF/
```

CT2 產物：

```text
artifacts/models/Breeze-ASR-26-float16-CT2/
artifacts/models/Breeze-ASR-26-int8_float16-CT2/
artifacts/models/Breeze-ASR-26-int8-CT2/
```

GGML 產物：

```text
artifacts/models/Breeze-ASR-26-GGML/
```

GGML 目錄會依照常見慣例，把同一個模型的多個量化檔放在同一個目錄，例如：

```text
ggml-model-q8_0.bin
ggml-model-q4_0.bin
```

每個產物目錄都會包含：

- `README.md`：模型卡與來源模型資訊。
- `metadata.json`：本機準備流程的中繼資料。
- `QUANTIZATION.md`：轉換與量化指令紀錄。

## 準備產物

產生全部模型產物：

```bash
uv run --extra hf --extra ct2 stt-eval prepare-models
```

只產生指定 variant：

```bash
uv run --extra hf --extra ct2 stt-eval prepare-models \
  --variant Breeze-ASR-26-int8-CT2 \
  --variant Breeze-ASR-26-q4_0-GGML
```

流程會視需要下載 HF 來源模型、補齊 CT2 需要的 tokenizer 檔案、產生 CT2 產物，並透過 whisper.cpp 工具產生 GGML 量化檔。

## 檢查產物

檢查模型產物目錄與中繼資料是否完整：

```bash
uv run stt-eval validate-artifacts
```
