import json
from collections.abc import Iterable
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from stt_eval.cer import calculate_cer
from stt_eval.manifest import ManifestItem, iter_manifest
from stt_eval.normalization import normalize_for_cer
from stt_eval.openai_transcription import transcribe_file
from stt_eval.runtime import collect_runtime_metadata
from stt_eval.settings import get_openai_settings


class EvalOptions(BaseModel):
    manifest: Path
    output: Path
    base_url: str = Field(default_factory=lambda: get_openai_settings().base_url)
    model: str = Field(default_factory=lambda: get_openai_settings().model)
    backend: str
    quantization: str
    model_artifact_dir: Path | None = None
    language: str | None = None
    prompt: str | None = None
    temperature: float = 0.0
    timeout_sec: float = Field(
        default_factory=lambda: get_openai_settings().timeout_sec
    )
    api_key: str = Field(default_factory=lambda: get_openai_settings().api_key_value())


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
    request: dict[str, str | float]
    runtime: dict[str, str | None] = Field(default_factory=dict)


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
    transcription = transcribe_file(
        client=client,
        model=options.model,
        audio_path=item.audio_path,
        timeout_sec=options.timeout_sec,
        temperature=options.temperature,
        language=options.language,
        prompt=options.prompt,
    )
    raw_hypothesis = transcription.text
    normalized_reference = normalize_for_cer(item.reference)
    normalized_hypothesis = normalize_for_cer(raw_hypothesis)
    cer = calculate_cer(normalized_reference, normalized_hypothesis)
    rtf = (
        transcription.elapsed_sec / item.duration_sec
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
        wall_time_sec=transcription.elapsed_sec,
        audio_duration_sec=item.duration_sec,
        rtf=rtf,
        request=_request_record(options),
        runtime=collect_runtime_metadata(),
    )


def _request_record(options: EvalOptions) -> dict[str, str | float]:
    request: dict[str, str | float] = {
        "model": options.model,
        "response_format": "json",
        "temperature": options.temperature,
    }
    if options.language:
        request["language"] = options.language
    if options.prompt:
        request["prompt"] = options.prompt
    return request


def summarize_results(
    rows: Iterable[EvalResultRow],
) -> dict[tuple[str, str], dict[str, float]]:
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
