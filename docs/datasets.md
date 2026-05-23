# 資料集準備

本文件記錄 `stt-eval` 目前支援的資料集下載與樣本產生流程。資料集音檔與完整中繼資料只保留在本機，不應提交到 repo。

## 環境設定

在專案根目錄建立 `.env`，依需要放入權杖：

```bash
MDC_API_KEY=你的 Mozilla Data Collective API 金鑰
HF_TOKEN=你的 Hugging Face 權杖
```

## 支援資料集

- `nan-tw`：Common Voice Taiwanese / Minnan，透過 Mozilla Data Collective API 下載。
- `example-sentences`：`sarahwei/Taiwanese-Minnan-Example-Sentences`，透過 Hugging Face 資料集下載。

預設資料集是 `nan-tw`。

## 下載資料集

下載預設資料集：

```bash
uv run stt-eval download-dataset
```

指定 Hugging Face 資料集：

```bash
uv run stt-eval download-dataset --dataset example-sentences
```

指定資料集 revision：

```bash
uv run stt-eval download-dataset \
  --dataset example-sentences \
  --revision <revision>
```

Hugging Face 資料集下載後會保留 Hub 上的 parquet shard，例如：

```text
data/raw/taiwanese_minnan_example_sentences/data/train-00000-of-00013.parquet
```

## 產生樣本

產生預設資料集樣本：

```bash
uv run stt-eval prepare-dataset-samples
```

指定資料集與筆數：

```bash
uv run stt-eval prepare-dataset-samples \
  --dataset example-sentences \
  --count 100
```

樣本會輸出到：

```text
data/samples/<dataset name>/
```

例如：

- `data/samples/nan-tw/`
- `data/samples/example-sentences/`

### Hugging Face parquet 展開

`example-sentences` 的 raw 資料是 Hugging Face 常見的 parquet 格式。執行樣本產生時，`stt-eval` 會優先讀取本機已下載的 parquet shard，從 `audio.bytes` 欄位寫出 WAV 檔，並產生對應 manifest：

```text
data/samples/example-sentences/00001.wav
data/samples/example-sentences/first_100_transcripts.tsv
```

raw parquet 已經在本機時，這一步不需要重新連線到 Hugging Face。如果本機 raw 目錄沒有 parquet shard，流程才會改用 Hugging Face Datasets API 讀取資料集。
