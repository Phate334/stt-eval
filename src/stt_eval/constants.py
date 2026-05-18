from pathlib import Path

MODEL_ID = "MediaTek-Research/Breeze-ASR-26"

DEFAULT_ARTIFACT_ROOT = Path("artifacts")
DEFAULT_MODEL_ROOT = DEFAULT_ARTIFACT_ROOT / "models"
DEFAULT_TOOL_ROOT = DEFAULT_ARTIFACT_ROOT / "tools"

HF_VARIANT = "Breeze-ASR-26-f32-HF"
CT2_VARIANTS = (
    "Breeze-ASR-26-float16-CT2",
    "Breeze-ASR-26-int8_float16-CT2",
    "Breeze-ASR-26-int8-CT2",
)
WHISPERCPP_VARIANTS = (
    "Breeze-ASR-26-q8_0-GGML",
    "Breeze-ASR-26-q5_0-GGML",
    "Breeze-ASR-26-q4_0-GGML",
    "Breeze-ASR-26-q4_1-GGML",
)
WHISPERCPP_ARTIFACT_DIR = "Breeze-ASR-26-GGML"

ALL_VARIANTS = (HF_VARIANT, *CT2_VARIANTS, *WHISPERCPP_VARIANTS)
ARTIFACT_DIR_BY_VARIANT = {
    HF_VARIANT: HF_VARIANT,
    **{variant: variant for variant in CT2_VARIANTS},
    **dict.fromkeys(WHISPERCPP_VARIANTS, WHISPERCPP_ARTIFACT_DIR),
}
ALL_ARTIFACT_DIRS = tuple(dict.fromkeys(ARTIFACT_DIR_BY_VARIANT.values()))

CT2_QUANTIZATION_BY_VARIANT = {
    "Breeze-ASR-26-float16-CT2": "float16",
    "Breeze-ASR-26-int8_float16-CT2": "int8_float16",
    "Breeze-ASR-26-int8-CT2": "int8",
}

WHISPERCPP_QUANTIZATION_BY_VARIANT = {
    "Breeze-ASR-26-q8_0-GGML": "q8_0",
    "Breeze-ASR-26-q5_0-GGML": "q5_0",
    "Breeze-ASR-26-q4_0-GGML": "q4_0",
    "Breeze-ASR-26-q4_1-GGML": "q4_1",
}

COMMON_HF_COPY_FILES = (
    "added_tokens.json",
    "generation_config.json",
    "merges.txt",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
