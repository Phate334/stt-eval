# Benchmark 紀錄

本文件記錄 benchmark 的決策與資料處理方式。目標是評估 `MediaTek-Research/Breeze-ASR-26` 在不同模型格式與量化設定下，量化模型相對原始模型輸出的文字偏移程度。

## 評估目標

本次比較同一批音檔在下列模型格式的輸出：

- HF baseline。
- CTranslate2（CT2）：`float16`、`int8_float16`、`int8`。
- whisper.cpp / GGML 量化版本：`q8`、`q5_0`、`q4_0`、`q4_1`。

主要指標：

- 以原始模型輸出作為 pseudo-reference，計算量化模型輸出相對原始模型輸出的 CER / character edit distance。
- 補充統計：完全相同率、空輸出率、長度差、插入 / 刪除 / 替換比例。
- 每次結果都要記錄原始模型格式、量化格式、解碼參數、normalization 規則版本或 git commit，避免不同批次分數無法比較。

資料集選擇原則：

- 目前的任務重點不是驗證資料集文字 reference 是否正確，也不是評估模型真實 ASR 品質，而是用固定音檔量測原始模型與量化模型之間的相對偏移。
- 資料集只需要提供可重現、授權可接受、音質與長度分布合理的音檔；文字欄位可以忽略。
- 同一批比較必須使用完全相同的音檔、排序、解碼參數與 normalization 規則。
- 即使資料集沒有逐字轉錄，也可以納入量化偏移評估；只是結果不能解讀為對人工 reference 的 ASR 正確率。

## Pseudo-reference 與文字處理

傳統 ASR CER 會使用資料集提供的人工 reference；本專案目前改採原始模型輸出作為 pseudo-reference。資料集文字只保留為抽查與除錯輔助，不參與主分數。

### 與傳統 ASR benchmark 的差異

Breeze-ASR-26 對台語語音的輸出目標偏華語漢字；部分資料集 reference 可能是台語漢字、台羅或華語對譯。由於主指標不使用資料集文字，因此：

- 主分數代表量化模型相對原始模型輸出的漂移程度，不代表人工 reference CER。
- 不可直接對比 Breeze-ASR-26 官方 30.13% CER。
- 若要另外報告資料集 reference CER，必須獨立標示為輔助分析，不能與 pseudo-reference 指標混用。

## 其他候選資料集

本節記錄 2026-05-23 重新調查的候選資料來源。

### Common Voice `nan-tw`

狀態：候選資料集。因文字品質與授權 / re-host 限制，目前不作為主資料集。

- 來源：https://mozilladatacollective.com/datasets/cmn2cyd8901jemm0738nubysq
- Dataset ID：`cmn2cyd8901jemm0738nubysq`
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

限制與建議：

- `nan-tw` 文本以台語漢字為主，括號內常附台羅或白話字參考發音，例如 `皇帝菜（hông-tè-tshài）`；文字品質與 Breeze-ASR-26 預期輸出有落差。
- 本任務主分數不使用資料集文字，因此文字問題不是計分阻礙，但仍會增加人工檢查與除錯成本。
- Common Voice 頁面明確禁止 re-host / re-share dataset；本專案不可上傳音檔、原始 archive、完整 TSV 或轉換後資料副本。
- 若使用，只能保留準備流程、normalization code、checksum、抽樣清單與操作說明，不提供資料 mirror。
- 可作為備用量化偏移評估集；不建議作為第一優先資料集。

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

本機解壓：

```bash
tar -xzf \
  data/raw/common_voice_nan_tw_25_0/common-voice-scripted-speech-25-0-taiwan-c14db9f7.tar.gz \
  -C data/raw/common_voice_nan_tw_25_0
```

### `sarahwei/Taiwanese-Minnan-Example-Sentences`

狀態：高優先候選，可作為主要量化偏移評估集。

- HF：https://huggingface.co/datasets/sarahwei/Taiwanese-Minnan-Example-Sentences
- 來源：教育部臺灣閩南語常用詞辭典 / Sutian Resource Center，dataset card 標示文字來源 `kautian.ods`、音檔來源 `leku-wav.zip`。
- 欄位：`hanzi`、`chinese`、`minnan roman`、`audio`
- Split：只有 `train`。
- 筆數：15,708。
- 音檔長度：HF viewer 顯示約 0.86 到 14.3 秒。
- License：CC BY-NC-SA 4.0。
- HF usage：截至調查時 dataset card 顯示約 193 downloads last month；HF 自動列出 3 個 fine-tuned model 使用此資料集，皆為 `Curiousfox/helsinki_new_*`，看起來偏翻譯/文字模型用途，未看到公開 ASR benchmark 採用紀錄。

與 `Taiwanese-Minnan-Sutiau` 的差異：

- `Example-Sentences` 是例句音檔，長度分布較接近日常句子，較能測到量化後在短句、長句上的輸出偏移。
- 欄位雖含台語漢字、華語對譯、台羅/羅馬字，但主評估不依賴這些文字欄位。
- 相較 `Sutiau`，較適合做主要音訊集合。

限制與建議：

- NC-SA 不適合作為通用公開預設 benchmark；若只做內部研究可以納入。
- 只有 train split，需固定抽樣規則產生 eval subset，例如 deterministic sample 500 或 1,000 筆，並記錄 dataset revision。
- 主分數只使用音檔與模型輸出；`hanzi`、`chinese`、`minnan roman` 不需清理。

### `sarahwei/Taiwanese-Minnan-Sutiau`

狀態：低優先候選，適合做詞條/短語 smoke test，不建議作為主量化偏移 benchmark。

- HF：https://huggingface.co/datasets/sarahwei/Taiwanese-Minnan-Sutiau
- 來源：教育部臺灣閩南語常用詞辭典 / Sutian Resource Center。
- 欄位：`hanzi`、`minnan roman`、`audio`
- Split：只有 `train`。
- 筆數：21,011。
- 音檔長度：HF viewer 顯示約 0.39 到 3.28 秒。
- License：CC BY-NC-SA 4.0。
- HF usage：截至調查時 dataset card 顯示約 188 downloads last month；未看到 HF 自動列出模型使用此資料集，也未查到公開 ASR benchmark 採用紀錄。

與 `Example-Sentences` 的差異：

- `Sutiau` 是詞條、短語、成語或很短的固定搭配，音檔明顯短於 `Example-Sentences`。
- 文字欄位較少，但主評估不依賴文字欄位，所以這不是阻礙。
- 音檔很短，character edit distance 容易被單一字差異放大；對量化後的小幅退化不一定穩定。

限制與建議：

- 可用來測量極短 utterance 下量化是否造成漏字、幻覺、空輸出。
- 不建議用作主排名；若使用，應獨立報告為 `sutiau_short_phrase` subset。

### `TaigiSpeech/TaigiSpeech`

狀態：可納入量化偏移評估候選；若做 intent 任務，則另開 SLU 評估。

- HF：https://huggingface.co/datasets/TaigiSpeech/TaigiSpeech
- Paper：https://arxiv.org/abs/2603.21478
- 任務：Spoken Language Understanding（SLU）/ intent classification。
- 場景：長照、健康緊急狀況、智慧家庭語音命令。
- License：CC BY 4.0。
- 筆數：3,079。
- Split：
  - Train：1,600 samples，10 speakers，每個 intent 200 筆。
  - Val：519 samples，5 speakers。
  - Test：960 samples，6 speakers，每個 intent 120 筆。
- Speaker split：train / val / test speaker 不重疊。
- 欄位：HF dataset card 標示 `audio`、`speaker_id`、`intent`；沒有逐字轉錄欄位。
- 音訊：WAV、48 kHz、mono。
- Speaker：21 位，年齡 20 到 78 歲，且多數 54 歲以上。
- Intent 類別：`SOS_CALL`、`FALL_HELP`、`BREATHING_CHEST_EMERG`、`PAIN_GENERAL`、`CALL_CONTACT`、`LIGHT_ON`、`LIGHT_OFF`、`CANCEL_ALERT`。
- HF usage：截至調查時 dataset card 顯示約 327 downloads last month；資料集本身搭配 2026 arXiv paper 發表，外部報導多聚焦低資源台語 intent detection，未看到逐字 ASR CER benchmark 採用紀錄。

限制與建議：

- 因無逐字 transcript，不能計算傳統人工 reference CER。
- 可直接用同一批音檔比較原始模型與量化模型輸出的 character edit distance，作為真實場景語音命令的量化偏移評估。
- 若後續要做任務層評估，可把原始/量化模型輸出接到同一個 intent classifier 或關鍵詞規則，觀察量化對 intent accuracy 的影響；這是 SLU 評估，不應與輸出偏移指標混在同一張表。

## 目前結論

- `sarahwei/Taiwanese-Minnan-Example-Sentences`：最值得下一步試跑。雖然授權為 NC-SA 且只有 train split，但例句音檔長度與數量適合作為主要偏移評估集合。
- `TaigiSpeech/TaigiSpeech`：缺 transcript 已不是阻礙；可納入輸出偏移評估。若要看任務效果，再另開 intent/SLU 評估。
- `nan-tw`：因文字品質與 re-host / re-share 授權限制，降為候選資料集；可保留現有準備流程，但不作為第一優先。
- `sarahwei/Taiwanese-Minnan-Sutiau`：可作短詞條壓力測試，觀察量化後短音檔空輸出或錯字率，不適合主 benchmark。

下一步建議：

- 先固定 `Example-Sentences` 的 dataset revision 與抽樣規則，建立 `example_sentences_eval` audio subset，不處理文字欄位。
- 同一批音檔跑原始模型、CT2 與 GGML 量化模型，以原始模型輸出作 pseudo-reference，報告每個格式的 character edit distance / CER-like drift。
- 可加跑 `TaigiSpeech` test split，補一組真實語音命令場景的輸出偏移結果。
- 另外抽少量 `Sutiau` 做短語 smoke test，檢查極短音訊下量化模型是否更容易空輸出或 hallucination。

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
