# stt-eval

`stt-eval` 是用來評估 `MediaTek-Research/Breeze-ASR-26` 量化版本的本機工具。專案目前聚焦在準備可重現的模型產物（artifact）與評估資料集樣本，後續用固定音檔比較原始模型與不同量化版本的輸出偏移。

## 功能

- **模型量化準備**：下載 HF 原始模型，產生 CTranslate2 與 whisper.cpp / GGML 量化產物。
- **資料集準備**：下載教育部臺灣台語常用詞辭典例句資料，整理成 `data/samples/<dataset name>`。
- **產物檢查**：確認模型產物目錄是否包含中繼資料、README 與量化紀錄。

## 文件

- [量化流程](docs/quantization.md)
- [資料集準備](docs/datasets.md)
- [評測決策紀錄](docs/benchmark.md)

## 資料來源與授權提醒

目前預設資料集為教育部「臺灣台語常用詞辭典」相關資源頁提供的例句資料：

- 來源頁面：[教育部臺灣台語常用詞辭典相關資源](https://sutian.moe.edu.tw/und-hani/siongkuantsuguan/)
- 文字來源：`kautian.ods`
- 音檔來源：`leku-wav.zip`

本專案只提供下載、解壓縮與整理腳本，不重新散布教育部原始檔、音檔或整理後的完整資料副本。使用資料前請自行確認教育部網站公告的授權與使用限制，並避免將 `data/raw/`、`data/samples/` 內的資料提交到版控或公開發布。

## 開發

專案使用 `uv` 管理 Python 環境：

```bash
uv sync
```

CLI 入口：

```bash
uv run stt-eval --help
```

比較 `results/` 內各版本相對於 baseline 的 CER：

```bash
uv run stt-eval compare-results --baseline vllm-hf-float16.jsonl
```

預設 `--normalization strip-whitespace`，會先移除 transcript 內所有空白，再用 baseline transcription 當 reference，對其他 `jsonl` 計算聚合 CER。

若要用較接近 Breeze-ASR-26 公開評測描述的口徑，可以改用：

```bash
uv run stt-eval compare-results --baseline vllm-hf-float16.jsonl --normalization breeze-compatible
```

`breeze-compatible` 目前定義為移除空白、去除 Unicode 標點、並將英文轉小寫；這是依公開 model card 的評測描述近似，並不是官方釋出的逐步實作。
