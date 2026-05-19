import json
from pathlib import Path
from time import perf_counter
from typing import Literal, TypedDict

from openai import OpenAI
from openai.types.audio.transcription_create_params import FileTypes
from pydantic import BaseModel, Field

from stt_eval.settings import get_transcription_settings


class RequiredTranscriptionParams(TypedDict):
    file: FileTypes
    model: str
    response_format: Literal["json"]
    temperature: float


class TranscriptionParams(RequiredTranscriptionParams, total=False):
    language: str
    prompt: str


class TranscriptionOptions(BaseModel):
    audio_paths: tuple[Path, ...]
    output: Path | None = None
    backend: str = Field(default_factory=lambda: get_transcription_settings().backend)
    base_url: str = Field(default_factory=lambda: get_transcription_settings().base_url)
    api_key: str = Field(
        default_factory=lambda: get_transcription_settings().api_key_value()
    )
    model: str = Field(default_factory=lambda: get_transcription_settings().model)
    timeout_sec: float = Field(
        default_factory=lambda: get_transcription_settings().timeout_sec
    )


class TranscriptionResult(BaseModel):
    audio_path: str
    text: str
    elapsed_sec: float


class TranscriptionPayload(BaseModel):
    backend: str
    base_url: str
    model: str
    results: list[TranscriptionResult]


def make_openai_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def transcribe_file(
    client: OpenAI,
    model: str,
    audio_path: Path,
    timeout_sec: float,
    temperature: float = 0.0,
    language: str | None = None,
    prompt: str | None = None,
) -> TranscriptionResult:
    started = perf_counter()
    with audio_path.open("rb") as audio_file:
        params: TranscriptionParams = {
            "file": audio_file,
            "model": model,
            "response_format": "json",
            "temperature": temperature,
        }
        if language:
            params["language"] = language
        if prompt:
            params["prompt"] = prompt
        response = client.audio.transcriptions.create(**params, timeout=timeout_sec)
    elapsed_sec = perf_counter() - started
    return TranscriptionResult(
        audio_path=str(audio_path),
        text=response.text,
        elapsed_sec=elapsed_sec,
    )


def transcribe_files(options: TranscriptionOptions) -> TranscriptionPayload:
    client = make_openai_client(options.base_url, options.api_key)
    results = [
        transcribe_file(
            client=client,
            model=options.model,
            audio_path=audio_path,
            timeout_sec=options.timeout_sec,
        )
        for audio_path in options.audio_paths
    ]
    payload = TranscriptionPayload(
        backend=options.backend,
        base_url=options.base_url,
        model=options.model,
        results=results,
    )
    if options.output:
        write_transcription_payload(payload, options.output)
    return payload


def write_transcription_payload(
    payload: TranscriptionPayload,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def payload_to_json(payload: TranscriptionPayload) -> str:
    return json.dumps(payload.model_dump(), ensure_ascii=False, indent=2)
