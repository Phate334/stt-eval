# 成功推論結果

本目錄的主要產出是每個模型版本各一份 `jsonl` 推論結果。正式比較請優先使用根目錄下的 `*.jsonl`。

音檔來源：`data/samples/moe-example-sentences-longest-hanzi-100`

| 版本 | 推論結果 | 成功/總數 | VRAM MiB |
|---|---|---:|---:|
| vLLM HF `float16` | `vllm-hf-float16.jsonl` | 100/100 | 21267-21267 |
| CT2 `float16` | `ct2-float16.jsonl` | 100/100 | 3991-3991 |
| CT2 `int8_float16` | `ct2-int8_float16.jsonl` | 100/100 | 2103-2135 |
| CT2 `int8` | `ct2-int8.jsonl` | 100/100 | 2097-2129 |
| whisper.cpp / GGML `q4_0` | `whisper-cpp-ggml-q4_0.jsonl` | 100/100 | 1843-1843 |
| whisper.cpp / GGML `q4_1` | `whisper-cpp-ggml-q4_1.jsonl` | 100/100 | 1935-1935 |
| whisper.cpp / GGML `q5_0` | `whisper-cpp-ggml-q5_0.jsonl` | 100/100 | 2027-2027 |
| whisper.cpp / GGML `q8_0` | `whisper-cpp-ggml-q8_0.jsonl` | 100/100 | 2575-2575 |

## llama.cpp / GGML 觀察

JSONL 內可以看到各版本的第一筆樣本（`sample_index=1`）普遍比後續樣本慢，尤其是 whisper.cpp / GGML 系列特別明顯。這比較像啟動/暖機成本，不是單筆音檔本身的異常失敗。

另外，幾個量化版本也有輸出品質上的離群值：

- `q4_1` 有明顯短輸出，例如 `sample_index=25` 只有「（宏述）」、`sample_index=70` 只有「（農夫之家常言）」，長度跟其他樣本相比很突兀。
- `q4_0` 的 `sample_index=1` 混入英文碎片，例如「Rose through the…」，屬於明顯雜訊型輸出。
- `q8_0` 雖然沒有空輸出，但有多筆長尾慢樣本，例如 `sample_index=89`、`30`、`55`、`73`、`99`，都明顯慢於同版本中位數。

整體來說，CT2 和 vLLM 的結果比較穩，JSONL 沒看到明顯執行中斷；真正比較異常的是 whisper.cpp / GGML 的量化版本，尤其是 `q4_1` 的首筆超慢，以及少數疑似截斷或雜訊輸出。


## vLLM 記憶體用量

vLLM 成功 run 來源：`artifacts/compose-benchmark-20260524T080918Z/`。

啟動 log 重點：

```text
Checkpoint size: 5.75 GiB
Model loading took 2.88 GiB memory
Estimated CUDA graph memory: 0.18 GiB total
Available KV cache memory: 16.9 GiB
GPU KV cache size: 19,881 tokens
Maximum concurrency for 448 tokens per request: 44.38x
```

目前 compose 使用 vLLM 預設 `--gpu-memory-utilization=0.92`，因此 vLLM 會盡量把 GPU 記憶體配置給 KV cache。這次觀測到的 `21267 MiB` 主要是預留高併發容量，不代表單一請求真的需要完整 21 GiB。

粗估單請求最低需求：

- 固定成本約 `20.8 GiB - 16.9 GiB = 3.9 GiB`。
- 單一 448 token request 的 KV cache 約 `16.9 GiB / 19881 * 448 = 0.38 GiB`。
- 理論低標約 `4.3 GiB`。
