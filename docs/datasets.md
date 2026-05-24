# 資料集準備

本專案目前只支援教育部「臺灣台語常用詞辭典」相關資源頁提供的例句資料。資料來源為官方原始檔：

- 文字：`kautian.ods`
- 音檔：`leku-wav.zip`

來源頁面：https://sutian.moe.edu.tw/und-hani/siongkuantsuguan/

資料集音檔與完整中繼資料只保留在本機，不應提交到 repo。

## 支援資料集

- `moe-example-sentences`：教育部臺灣台語常用詞辭典例句 WAV 音檔與 `kautian.ods` 內的「例句」工作表。

預設資料集是 `moe-example-sentences`。

## 下載與解壓縮

下載官方原始檔並解壓縮例句 WAV：

```bash
uv run stt-eval download-dataset
```

輸出位置：

```text
data/raw/moe_sutian_example_sentences/
├── kautian.ods
├── leku-wav.zip
├── leku-wav/
└── leku.tsv
```

`leku.tsv` 是整理後的 manifest，欄位如下：

- `rank`：依 `kautian.ods`「例句」工作表原始順序產生的序號。
- `audio_id`：ODS 的「音檔檔名」欄位，例如 `1-1-1`。
- `audio_path`：解壓後相對於 raw dataset 目錄的 WAV 路徑。
- `hanzi`：台語漢字例句。
- `tailo`：羅馬字例句。
- `mandarin`：華語對譯。

若只想下載原始檔、不解壓縮 9GB 以上的音檔壓縮檔：

```bash
uv run stt-eval download-dataset --no-extract
```

重新下載或重新解壓：

```bash
uv run stt-eval download-dataset --force
```

## 產生樣本

產生預設前 100 筆例句音檔樣本：

```bash
uv run stt-eval prepare-dataset-samples
```

指定筆數：

```bash
uv run stt-eval prepare-dataset-samples --count 1000
```

樣本會輸出到：

```text
data/samples/moe-example-sentences/
├── 00001.wav
├── 00002.wav
└── first_<count>_transcripts.tsv
```

樣本 manifest 保留原始例句文字與來源音檔路徑，後續評測可以只使用 `path` 欄位的固定音檔集合。
