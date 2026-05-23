# stt-eval

`stt-eval` 是用來評估 `MediaTek-Research/Breeze-ASR-26` 量化版本的本機工具。專案目前聚焦在準備可重現的模型產物（artifact）與評估資料集樣本，後續用固定音檔比較原始模型與不同量化版本的輸出偏移。

## 功能

- **模型量化準備**：下載 HF 原始模型，產生 CTranslate2 與 whisper.cpp / GGML 量化產物。
- **資料集準備**：下載 Common Voice `nan-tw` 或 Hugging Face 台語資料集，整理成 `data/samples/<dataset name>`。
- **產物檢查**：確認模型產物目錄是否包含中繼資料、README 與量化紀錄。

## 文件

- [量化流程](docs/quantization.md)
- [資料集準備](docs/datasets.md)
- [評測決策紀錄](docs/benchmark.md)

## 開發

專案使用 `uv` 管理 Python 環境：

```bash
uv sync
```

CLI 入口：

```bash
uv run stt-eval --help
```
