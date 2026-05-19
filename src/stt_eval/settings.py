from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = Field(
        default="http://127.0.0.1:8080/v1",
        validation_alias=AliasChoices(
            "STT_EVAL_OPENAI_BASE_URL",
            "OPENAI_BASE_URL",
        ),
    )
    api_key: SecretStr = Field(
        default=SecretStr("not-needed"),
        validation_alias=AliasChoices(
            "STT_EVAL_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        ),
    )
    model: str = Field(
        default="whisper-1",
        validation_alias=AliasChoices(
            "STT_EVAL_OPENAI_MODEL",
            "OPENAI_MODEL",
        ),
    )
    timeout_sec: float = Field(
        default=180.0,
        validation_alias=AliasChoices(
            "STT_EVAL_OPENAI_TIMEOUT_SEC",
            "OPENAI_TIMEOUT_SEC",
        ),
    )

    def api_key_value(self) -> str:
        return self.api_key.get_secret_value()


class TranscriptionSettings(OpenAISettings):
    backend: str = Field(
        default="openai-compatible",
        validation_alias=AliasChoices(
            "STT_EVAL_BACKEND",
            "BACKEND",
        ),
    )


@lru_cache
def get_openai_settings() -> OpenAISettings:
    return OpenAISettings()


@lru_cache
def get_transcription_settings() -> TranscriptionSettings:
    return TranscriptionSettings()
