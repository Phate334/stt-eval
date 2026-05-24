from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

JsonObject = dict[str, object]
ResultKey = tuple[int, str]
NormalizationMode = Literal["raw", "strip-whitespace", "breeze-compatible"]
NORMALIZATION_MODES: tuple[NormalizationMode, ...] = (
    "raw",
    "strip-whitespace",
    "breeze-compatible",
)


@dataclass(frozen=True)
class ResultRecord:
    sample_index: int
    audio_path: str
    ok: bool
    transcript: str | None

    @property
    def key(self) -> ResultKey:
        return (self.sample_index, self.audio_path)


@dataclass(frozen=True)
class CompareResultsOptions:
    baseline_path: Path
    candidate_paths: tuple[Path, ...]
    normalization: NormalizationMode = "strip-whitespace"


@dataclass(frozen=True)
class WorstSample:
    sample_index: int
    audio_path: str
    cer: float


@dataclass(frozen=True)
class ResultComparison:
    candidate_path: Path
    compared_samples: int
    total_samples: int
    missing_samples: int
    failed_samples: int
    exact_matches: int
    char_errors: int
    reference_chars: int
    cer: float
    worst_samples: tuple[WorstSample, ...]


def compare_results(options: CompareResultsOptions) -> list[ResultComparison]:
    baseline_records = _load_result_records(options.baseline_path)
    baseline_keys = tuple(sorted(baseline_records))
    total_samples = len(baseline_keys)

    baseline_failures = [
        record.key for record in baseline_records.values() if not record.ok
    ]
    if baseline_failures:
        raise RuntimeError(
            f"baseline 檔案包含失敗樣本，無法作為 reference：{options.baseline_path}"
        )

    comparisons = []
    for candidate_path in options.candidate_paths:
        candidate_records = _load_result_records(candidate_path)
        missing_samples = 0
        failed_samples = 0
        exact_matches = 0
        char_errors = 0
        reference_chars = 0
        per_sample: list[WorstSample] = []

        for key in baseline_keys:
            baseline_record = baseline_records[key]
            candidate_record = candidate_records.get(key)
            if candidate_record is None:
                missing_samples += 1
                continue
            if not candidate_record.ok or candidate_record.transcript is None:
                failed_samples += 1
                continue

            baseline_text = _normalize_text(
                baseline_record.transcript or "",
                normalization=options.normalization,
            )
            candidate_text = _normalize_text(
                candidate_record.transcript,
                normalization=options.normalization,
            )
            if not baseline_text:
                raise RuntimeError(
                    "baseline transcript 正規化後為空字串，無法計算 CER："
                    f"sample_index={baseline_record.sample_index}"
                )

            distance = levenshtein_distance(baseline_text, candidate_text)
            reference_chars += len(baseline_text)
            char_errors += distance
            if baseline_text == candidate_text:
                exact_matches += 1
            per_sample.append(
                WorstSample(
                    sample_index=baseline_record.sample_index,
                    audio_path=baseline_record.audio_path,
                    cer=distance / len(baseline_text),
                )
            )

        compared_samples = len(per_sample)
        if compared_samples == 0 or reference_chars == 0:
            raise RuntimeError(f"找不到可比較樣本：{candidate_path}")
        comparisons.append(
            ResultComparison(
                candidate_path=candidate_path,
                compared_samples=compared_samples,
                total_samples=total_samples,
                missing_samples=missing_samples,
                failed_samples=failed_samples,
                exact_matches=exact_matches,
                char_errors=char_errors,
                reference_chars=reference_chars,
                cer=char_errors / reference_chars,
                worst_samples=tuple(
                    sorted(
                        per_sample,
                        key=lambda sample: (sample.cer, sample.sample_index),
                        reverse=True,
                    )[:3]
                ),
            )
        )

    return sorted(comparisons, key=lambda item: (item.cer, item.candidate_path.name))


def render_markdown_report(
    baseline_path: Path,
    comparisons: list[ResultComparison],
    *,
    normalization: NormalizationMode,
) -> str:
    lines = [
        "# CER baseline comparison",
        "",
        f"- baseline: `{baseline_path.name}`",
        f"- normalization: {_describe_normalization(normalization)}",
        "",
        "| 檔案 | CER | 字元錯誤/參考字元 | 完全一致 | 可比較樣本 | 缺漏 | 失敗 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in comparisons:
        lines.append(
            "| "
            f"{comparison.candidate_path.name} | "
            f"{comparison.cer:.4f} | "
            f"{comparison.char_errors}/{comparison.reference_chars} | "
            f"{comparison.exact_matches} | "
            f"{comparison.compared_samples}/{comparison.total_samples} | "
            f"{comparison.missing_samples} | "
            f"{comparison.failed_samples} |"
        )
    lines.append("")
    for comparison in comparisons:
        lines.append(f"## {comparison.candidate_path.name}")
        lines.append("")
        for sample in comparison.worst_samples:
            lines.append(
                "- "
                f"sample_index={sample.sample_index}, CER={sample.cer:.4f}, audio={sample.audio_path}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_json_report(
    baseline_path: Path,
    comparisons: list[ResultComparison],
    *,
    normalization: NormalizationMode,
) -> str:
    payload = {
        "baseline": baseline_path.name,
        "normalization": normalization,
        "comparisons": [
            {
                "file": comparison.candidate_path.name,
                "cer": round(comparison.cer, 8),
                "char_errors": comparison.char_errors,
                "reference_chars": comparison.reference_chars,
                "exact_matches": comparison.exact_matches,
                "compared_samples": comparison.compared_samples,
                "total_samples": comparison.total_samples,
                "missing_samples": comparison.missing_samples,
                "failed_samples": comparison.failed_samples,
                "worst_samples": [
                    {
                        "sample_index": sample.sample_index,
                        "audio_path": sample.audio_path,
                        "cer": round(sample.cer, 8),
                    }
                    for sample in comparison.worst_samples
                ],
            }
            for comparison in comparisons
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _load_result_records(path: Path) -> dict[ResultKey, ResultRecord]:
    records: dict[ResultKey, ResultRecord] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            data = json.loads(line)
            record = _parse_result_record(data, path=path, line_number=line_number)
            if record.key in records:
                raise RuntimeError(
                    f"重複的 sample key：{path}:{line_number} -> {record.key}"
                )
            records[record.key] = record
    if not records:
        raise RuntimeError(f"空的 results 檔案：{path}")
    return records


def _parse_result_record(
    data: JsonObject,
    *,
    path: Path,
    line_number: int,
) -> ResultRecord:
    sample_index = data.get("sample_index")
    audio_path = data.get("audio_path")
    ok = data.get("ok")
    response = data.get("response")
    if not isinstance(sample_index, int):
        raise RuntimeError(f"sample_index 格式錯誤：{path}:{line_number}")
    if not isinstance(audio_path, str):
        raise RuntimeError(f"audio_path 格式錯誤：{path}:{line_number}")
    if not isinstance(ok, bool):
        raise RuntimeError(f"ok 格式錯誤：{path}:{line_number}")

    transcript: str | None = None
    if ok:
        if not isinstance(response, dict):
            raise RuntimeError(f"response 格式錯誤：{path}:{line_number}")
        text = response.get("text")
        if not isinstance(text, str):
            raise RuntimeError(f"response.text 格式錯誤：{path}:{line_number}")
        transcript = text

    return ResultRecord(
        sample_index=sample_index,
        audio_path=audio_path,
        ok=ok,
        transcript=transcript,
    )


def _normalize_text(text: str, *, normalization: NormalizationMode) -> str:
    normalized = text.strip()
    if normalization == "raw":
        return normalized
    normalized = "".join(normalized.split())
    if normalization == "strip-whitespace":
        return normalized
    if normalization == "breeze-compatible":
        lowered = normalized.lower()
        return "".join(
            character
            for character in lowered
            if not unicodedata.category(character).startswith("P")
        )
    raise AssertionError(f"Unhandled normalization mode: {normalization}")


def _describe_normalization(normalization: NormalizationMode) -> str:
    if normalization == "raw":
        return "raw transcript"
    if normalization == "strip-whitespace":
        return "remove all whitespace"
    if normalization == "breeze-compatible":
        return "remove whitespace, strip punctuation, lowercase ASCII/English"
    raise AssertionError(f"Unhandled normalization mode: {normalization}")


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_char in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hyp_char in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_char == hyp_char else 1
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]
