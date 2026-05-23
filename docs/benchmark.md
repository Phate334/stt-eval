# Benchmark 紀錄

本文件記錄 benchmark 的決策與資料處理方式。目標是評估 `MediaTek-Research/Breeze-ASR-26` 在不同模型格式與量化設定下，台語 ASR 的 CER 變化。

## 評估目標

本次比較：

- HF baseline。
- CTranslate2（CT2）：`float16`、`int8_float16`、`int8`。
- whisper.cpp / GGML 量化版本：`q8`、`q5_0`、`q4_0`、`q4_1`。

主要指標：

- CER（Character Error Rate）。
- 每次結果都要記錄 normalization 規則版本或 git commit，避免不同批次分數無法比較。

## 主資料集：Common Voice `nan-tw`

目前主 benchmark 使用 Mozilla Common Voice Scripted Speech 25.0 的台語資料。

- 來源：https://mozilladatacollective.com/datasets/cmn2cyd8901jemm0738nubysq
- Locale：`nan-tw`
- 語言：Taiwanese（Minnan），也就是台語 / 台灣閩南語。
- Release：`cv-corpus-25.0-2026-03-09`
- 格式：MP3 archive。
- License：CC0-1.0。

資料規模：

- 299 位 speaker。
- 總錄音 23.87 小時。
- validated 21.78 小時。
- validated clips：29,587。

官方 split：

- Train：11,507 clips。
- Dev：5,999 clips。
- Test：6,423 clips。

本次預設使用官方 `test` split。

### 下載與本機路徑紀錄

下載日期：2026-05-19（Asia/Taipei）。

官方下載流程：

1. 先登入 Mozilla Data Collective，在 dataset 頁面接受下載條款。
2. 在專案根目錄的 `.env` 放入 `MDC_API_KEY`。
3. 使用官方 API 取得一次性的 presigned download URL：

```bash
curl -sS -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MDC_API_KEY" \
  https://mozilladatacollective.com/api/datasets/cmn2cyd8901jemm0738nubysq/download
```

下載與資料路徑：

- Dataset ID：`cmn2cyd8901jemm0738nubysq`

本機解壓：

```bash
tar -xzf \
  data/raw/common_voice_nan_tw_25_0/common-voice-scripted-speech-25-0-taiwan-c14db9f7.tar.gz \
  -C data/raw/common_voice_nan_tw_25_0
```


### 使用限制

Common Voice 頁面明確禁止 re-host / re-share dataset。因此本專案：

- 不上傳音檔。
- 不上傳原始 archive。
- 不上傳完整 TSV。
- 不上傳轉換後的 dataset 副本到 Hugging Face 或其他公開 host。
- 只公開準備流程、normalization code、checksum、操作說明。
- 不嘗試識別 speaker。

## Reference 與文字處理

Common Voice `nan-tw` 文本以台語漢字為主，括號內常附台羅或白話字參考發音，例如：

```text
皇帝菜（hông-tè-tshài）
```


### 與 Breeze-ASR-26 官方 benchmark 的差異

Breeze-ASR-26 對台語語音的輸出目標偏華語漢字；Common Voice `nan-tw` 是台語漢字 reference。兩者文字目標不同，因此：

- Common Voice `taigi_hanzi` CER 只適合比較同一設定下的相對差異。
- 不可直接對比 Breeze-ASR-26 官方 30.13% CER。

## 其他候選資料集

本節僅列尚未採用為主 benchmark 的資料來源。

### `sarahwei/Taiwanese-Minnan-Example-Sentences`

狀態：research-only 候選。

- HF：https://huggingface.co/datasets/sarahwei/Taiwanese-Minnan-Example-Sentences
- 欄位：`hanzi`、`chinese`、`minnan roman`、`audio`
- 15,708 筆
- License：CC BY-NC-SA 4.0

限制：NC-SA 不適合作為通用預設 benchmark，且只有 train split。

### `sarahwei/Taiwanese-Minnan-Sutiau`

狀態：低優先 research-only 輔助資料。

- HF：https://huggingface.co/datasets/sarahwei/Taiwanese-Minnan-Sutiau
- 欄位：`hanzi`、`minnan roman`、`audio`
- 21,011 筆
- License：CC BY-NC-SA 4.0

限制：多為詞條/短語，不適合句子級 CER 主 benchmark。

### `TaigiSpeech/TaigiSpeech`

暫不作為主 ASR benchmark，原因是其主要任務偏 intent classification / SLU，非句子級 ASR CER 評估。

## 暫不採用資料集

### Common Voice mirror / derived copies

- `hydedada/nan_tw`
- `jiyuntu/common_voice_minnan`
- `lazy-worm/preprocessed_cv-nan-tw-validate-split-2`

原因：疑似 re-host 或衍生重發，與 Mozilla Data Collective 限制衝突。

### 授權或來源不明

- `thomas0104/nan_tw_soap_opera`
- `gacky1601/Taiwanese_ASR`
- `Curiousfox/NRP_NIE04B_Hokkien_dataset`

原因：license 與來源資訊不足，暫不納入公開可重現 benchmark。

- 行政院 PSA 月包
  - https://www.ey.gov.tw/Page/AA7FD03FF4A55EF8

- `formospeech/yttd_taigi_trs`
  - HF：https://huggingface.co/datasets/formospeech/yttd_taigi_trs
  - Gated manual access
  - 欄位含 `audio`、`duration`、`text`、`text_mandarin`
  - 約 train 50,984 / test 4,859
