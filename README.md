# stt-eval

`stt-eval` 是用來評估 `MediaTek-Research/Breeze-ASR-26` 量化版本的工具。

目前保留的功能：

- 模型準備與量化：下載 HF 原始模型，轉成 CT2 與 GGML 量化 artifacts。

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

## 檢查 artifacts

```bash
uv run stt-eval validate-artifacts
```

會確認每個 artifact 目錄是否包含：

- `metadata.json`
- `README.md`
- `QUANTIZATION.md`
