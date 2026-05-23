from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SttEvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mdc_api_key: str = Field(default="", alias="MDC_API_KEY")
    hf_token: str = Field(default="", alias="HF_TOKEN")
    default_dataset: str = "nan-tw"
    raw_root: Path = Path("data/raw")
    sample_root: Path = Path("data/samples")
