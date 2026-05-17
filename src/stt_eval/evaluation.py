import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from stt_eval.cer import calculate_cer
from stt_eval.manifest import ManifestItem, iter_manifest
from stt_eval.normalization import normalize_for_cer
from stt_eval.runtime import collect_runtime_metadata


class EvalOptions(BaseModel):
    manifest: Path
    output: Path
    base_url: str
    model: str
    backend: str
    quantization: str
    model_artifact_dir: Path | None = None
    language: str | None = None
    prompt: str | None = None
    temperature: float = 0.0
    api_key: str = "local"


class EvalResultRow(BaseModel):
    sample_id: str
    backend: str
    quantization: str
    model: str
    model_artifact_dir: str | None
    raw_hypothesis: str
    normalized_hypothesis: str
    normalized_reference: str
    cer_substitutions: int
    cer_insertions: int
    cer_deletions: int
    cer_reference_length: int
    cer_errors: int
    cer: float
    wall_time_sec: float
    audio_duration_sec: float | None
    rtf: float | None
    request: dict[str, Any]
    runtime: dict[str, Any] = Field(default_factory=dict)


def run_eval(options: EvalOptions) -> None:
    client = OpenAI(base_url=options.base_url, api_key=options.api_key)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("w", encoding="utf-8") as output:
        for item in iter_manifest(options.manifest):
            row = transcribe_and_score(client, item, options)
            output.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")


def transcribe_and_score(
    client: OpenAI,
    item: ManifestItem,
    options: EvalOptions,
) -> EvalResultRow:
    started = time.perf_counter()
    request = _request_payload(options)
    with item.audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            **request,
        )
    wall_time = time.perf_counter() - started
    raw_hypothesis = extract_text(response)
    normalized_reference = normalize_for_cer(item.reference)
    normalized_hypothesis = normalize_for_cer(raw_hypothesis)
    cer = calculate_cer(normalized_reference, normalized_hypothesis)
    rtf = (
        wall_time / item.duration_sec
        if item.duration_sec and item.duration_sec > 0
        else None
    )
    return EvalResultRow(
        sample_id=item.id,
        backend=options.backend,
        quantization=options.quantization,
        model=options.model,
        model_artifact_dir=(
            str(options.model_artifact_dir) if options.model_artifact_dir else None
        ),
        raw_hypothesis=raw_hypothesis,
        normalized_hypothesis=normalized_hypothesis,
        normalized_reference=normalized_reference,
        cer_substitutions=cer.substitutions,
        cer_insertions=cer.insertions,
        cer_deletions=cer.deletions,
        cer_reference_length=cer.reference_length,
        cer_errors=cer.errors,
        cer=cer.ratio,
        wall_time_sec=wall_time,
        audio_duration_sec=item.duration_sec,
        rtf=rtf,
        request={key: value for key, value in request.items() if key != "file"},
        runtime=collect_runtime_metadata(),
    )


def _request_payload(options: EvalOptions) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": options.model,
        "response_format": "json",
        "temperature": options.temperature,
    }
    if options.language:
        payload["language"] = options.language
    if options.prompt:
        payload["prompt"] = options.prompt
    return payload


def extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        value = response.get("text")
        if isinstance(value, str):
            return value
    value = getattr(response, "text", None)
    if isinstance(value, str):
        return value
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        value = dumped.get("text")
        if isinstance(value, str):
            return value
    raise ValueError(f"Could not extract text from transcription response: {response!r}")


def summarize_results(rows: Iterable[EvalResultRow]) -> dict[tuple[str, str], dict[str, float]]:
    groups: dict[tuple[str, str], list[EvalResultRow]] = {}
    for row in rows:
        groups.setdefault((row.backend, row.quantization), []).append(row)
    summary = {}
    for key, values in groups.items():
        rtf_values = [row.rtf for row in values if row.rtf is not None]
        summary[key] = {
            "samples": float(len(values)),
            "mean_cer": sum(row.cer for row in values) / len(values),
            "mean_rtf": (
                sum(rtf_values) / len(rtf_values) if rtf_values else float("nan")
            ),
        }
    return summary
