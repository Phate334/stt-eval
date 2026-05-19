# Benchmark 紀錄

本文件記錄本次 benchmark 的決策與資料處理方式。目標是評估 `MediaTek-Research/Breeze-ASR-26` 在不同模型格式與量化設定下，台語 ASR 的 CER 變化。

## 評估目標

本次比較：

- HF baseline。
- CTranslate2（CT2）：`float16`、`int8_float16`、`int8`。
- whisper.cpp / GGML 量化版本。

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

### 使用限制

Common Voice 頁面明確禁止 re-host / re-share dataset。因此本專案：

- 不上傳音檔。
- 不上傳原始 archive。
- 不上傳完整 TSV。
- 不上傳轉換後的 dataset 副本到 Hugging Face 或其他公開 host。
- 只公開準備流程、normalization code、checksum、操作說明。
- 不嘗試識別 speaker。

## Reference 與文字處理

Common Voice `nan-tw` 的文字多數是台語漢字，括號內常附台羅或白話字發音參考，例如：

```text
漢字本文（romanization）
```

Mozilla Data Collective 頁面對 `nan-tw` 的描述是「漢字——語音資料集」，文本語料以漢字為主，同時括號標注台羅或白話字參考發音。頁面也明確建議使用前先移除 `（）` 包住的參考發音，僅取漢字部分。

實際文字形式例子：

```text
皇帝菜（hông-tè-tshài）
敦化和平路口（Tun-huà Hô-pîng Lōo-kháu | Tun-hòa Hô-pêng Lō͘-kháu）
我欲去食晝（guá beh khì tsia̍h-tàu）
```

移除括號羅馬字後會得到：

```text
皇帝菜
敦化和平路口
我欲去食晝
```

這些 reference 仍是台語漢字 / 台語正字傾向的文字，不是華語翻譯。例如 `我欲去食晝` 和華語語意上的 `我要去吃午餐` 是不同文字目標；只做 Unicode、標點、空白 normalization 無法把兩者變成同一個 CER reference。

本次 Common Voice benchmark 的 reference target 是：

- `taigi_hanzi`
- 也就是移除括號羅馬字後的台語漢字。

### 與 Breeze-ASR-26 輸出形式的差異

Breeze-ASR-26 model card 和論文都說明此模型接受台語語音，但輸出目標是華語漢字，而不是 native Taigi orthography / 台語正字。官方 Breeze Taigi ASR benchmark 使用行政院廣播公共服務音檔的華語 / 台語平行音檔，對台語音檔建立的是華語漢字 reference。

因此 Common Voice `nan-tw` 與 Breeze-ASR-26 官方 benchmark 的核心差異不是 locale 名稱，而是文字形式：

- Common Voice `nan-tw`：台語語音對台語漢字 reference。
- Breeze-ASR-26 官方設定：台語語音對華語漢字 reference。

這代表 Common Voice `taigi_hanzi` CER 可以用來比較同一批 Breeze-ASR-26 量化版本之間的相對變化，但不能解讀成重現官方 Breeze-ASR-26 CER，也不能和官方 30.13% CER 直接比較。

若模型輸出接近華語漢字，而 reference 是台語漢字，CER 會同時混入兩種誤差：

- ASR 辨識錯誤。
- 台語漢字與華語漢字之間的書寫 / 翻譯目標不一致。

第二種不是 normalization 可以處理的問題。若要處理它，已經是在做台語轉華語或語意改寫，必須另開 `mandarin_hanzi` benchmark track。

### 預設處理流程

1. 使用官方 `test` split。
2. 從原始 TSV 讀取 `path` 與 `text`。
3. 將 `path` 對應到本機 `clips/` 音檔。
4. 移除 `text` 中括號內的羅馬字發音參考。
5. 套用 deterministic normalization。
6. 產生本機 manifest。

若移除羅馬字後 reference 為空，或音檔不存在，該筆才丟棄。

### Normalization 範圍

允許：

- Unicode normalization。
- 全形 / 半形正規化。
- ASCII lowercasing。
- 移除空白。
- 移除常見中文與 ASCII 標點。
- 移除括號內羅馬字。

不允許在主分數中做：

- 台語轉華語。
- 華語轉台語。
- 同義詞替換。
- LLM 修正。
- 字典式大範圍改寫。

如果要做台語語音對華語 reference 的評估，需另開 benchmark track，例如 `mandarin_hanzi`，不可和 Common Voice `taigi_hanzi` 分數混在一起。

## Manifest 格式

所有資料集進 pipeline 前都要轉成同一種 JSONL manifest。

必要欄位：

- `id`
- `audio_path`
- `reference`
- `reference_target`，例如 `taigi_hanzi` 或 `mandarin_hanzi`
- `dataset`
- `split`

可選欄位：

- `duration_sec`

資料準備 script 必須是 deterministic，並記錄：

- source dataset version。
- source URL。
- split。
- filtering rules。
- normalization version。

## 結果輸出

單筆 result 至少包含：

- sample id。
- audio path。
- reference。
- prediction。
- CER。
- backend。
- quantization。
- model artifact path。

Summary 至少包含：

- 模型名稱。
- backend。
- quantization。
- sample 數。
- 平均 CER。
- 評估時間。
- reference target。
- normalization version 或 git commit。
- dropped sample 數與原因。

## 其他候選資料集

本節只列尚未採用為主 benchmark 的資料來源。Mozilla Common Voice `nan-tw` 已在「主資料集」章節說明，不在候選清單重複列出。

### 1. `formospeech/yttd_taigi_trs`

狀態：高優先候選，但尚未採用。

- HF：https://huggingface.co/datasets/formospeech/yttd_taigi_trs
- Gated manual access。
- metadata 顯示欄位含 `audio`、`duration`、`text`、`text_mandarin`。
- train config 約 50,984 筆。
- test config 約 4,859 筆。

優點：

- 有 `text_mandarin`，可能比 Common Voice 更接近 Breeze-ASR-26 的官方評估型態。
- 規模足夠。

待確認：

- license。
- source provenance。
- 是否允許公開重現。
- `test` config 是否就是正式 evaluation split。
- `text` 與 `text_mandarin` 的實際內容與品質。

若採用：

- Breeze-style 評估優先用 `text_mandarin`。
- `reference_target = mandarin_hanzi`。
- 不和 Common Voice 的 `taigi_hanzi` 結果混算。

### 2. `sarahwei/Taiwanese-Minnan-Example-Sentences`

狀態：research-only 候選。

- HF：https://huggingface.co/datasets/sarahwei/Taiwanese-Minnan-Example-Sentences
- 來源：教育部臺灣閩南語常用詞辭典相關資源。
- 欄位：`hanzi`、`chinese`、`minnan roman`、`audio`。
- 15,708 筆。
- License：CC BY-NC-SA 4.0。

優點：

- 有 audio 與文字。
- 有 `hanzi` 與 `chinese`，可分別做台語漢字與華語文字 track。
- 來源有描述。

限制：

- NC-SA 不適合作為通用預設 benchmark。
- 只有 train split，需要自己定義 deterministic split。

### 3. `sarahwei/Taiwanese-Minnan-Sutiau`

狀態：低優先 research-only 輔助資料。

- HF：https://huggingface.co/datasets/sarahwei/Taiwanese-Minnan-Sutiau
- 欄位：`hanzi`、`minnan roman`、`audio`。
- 21,011 筆。
- License：CC BY-NC-SA 4.0。

限制：

- 多為詞條或短語，不是句子級 ASR。
- 很短的 reference 會讓 CER 波動較大。
- 若使用，需和句子級 benchmark 分開報告。

### 4. 行政院 PSA 月包

狀態：Breeze-style reconstructed benchmark 候選。

- 來源：https://www.ey.gov.tw/Page/AA7FD03FF4A55EF8
- Breeze-ASR-26 論文表示官方 benchmark 取自行政院華語 / 台語公益廣播音檔配對。

優點：

- 最接近 Breeze-ASR-26 官方 benchmark 的資料來源。
- 可設計成台語音訊對華語漢字 reference。

限制：

- 官方 30 筆清單沒有公開。
- 官方 normalized references 沒有公開。
- 音檔與衍生 reference 的公開散布權限需另外確認。
- 自建版本只能稱為 reconstructed / Breeze-style，不能宣稱重現官方 CER。

若採用：

- 固定檔案清單、URL、checksum。
- 建立華語 reference 流程。
- `reference_target = mandarin_hanzi`。
- 和 Common Voice 分開報告。

## 暫不採用資料集

### Common Voice mirror / derived copies

暫不採用：

- `hydedada/nan_tw`
- `jiyuntu/common_voice_minnan`
- `lazy-worm/preprocessed_cv-nan-tw-validate-split-2`

原因：

- 疑似 re-host 或處理後重發 Common Voice。
- 與 Mozilla Data Collective 禁止 re-host / re-share 的限制衝突。
- 本專案應使用 Mozilla 官方來源搭配本機處理 script。

### 授權或來源不明

暫不採用：

- `thomas0104/nan_tw_soap_opera`
- `gacky1601/Taiwanese_ASR`
- `Curiousfox/NRP_NIE04B_Hokkien_dataset`

原因：

- license 不清楚。
- source provenance 不足。
- 不確定是否有權公開重現或作為 benchmark 使用。

重新考慮前需要補齊：

- 原始資料來源。
- license。
- 是否可用於公開評估。
- 是否可重新散布。
- reference 是否人工標註、轉換或機器產生。

### `TaigiSpeech/TaigiSpeech`

暫不作為主 ASR benchmark。

原因：

- 主要是 intent classification / SLU 資料。
- 不適合直接做句子級 CER ASR 評估。
