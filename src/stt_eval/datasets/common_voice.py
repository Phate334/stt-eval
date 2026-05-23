import csv
import json
import shutil
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stt_eval.config import SttEvalSettings
from stt_eval.datasets.base import (
    DatasetDownloadOptions,
    DatasetSampleOptions,
)

MOZILLA_DOWNLOAD_ENDPOINT = (
    "https://mozilladatacollective.com/api/datasets/{dataset_id}/download"
)


@dataclass(frozen=True)
class CommonVoiceDataset:
    name: str
    dataset_id: str
    archive_name: str
    extracted_dir: Path
    raw_dir_name: str


COMMON_VOICE_DATASETS = {
    "nan-tw": CommonVoiceDataset(
        name="nan-tw",
        dataset_id="cmn2cyd8901jemm0738nubysq",
        archive_name="common-voice-scripted-speech-25-0-taiwan-c14db9f7.tar.gz",
        extracted_dir=Path("cv-corpus-25.0-2026-03-09") / "nan-tw",
        raw_dir_name="common_voice_nan_tw_25_0",
    )
}
SAMPLE_SPLITS = (
    "train",
    "dev",
    "test",
    "validated",
    "other",
    "invalidated",
)


def download_common_voice_dataset(
    dataset: CommonVoiceDataset,
    options: DatasetDownloadOptions,
    settings: SttEvalSettings,
) -> None:
    raw_root = options.raw_root or settings.raw_root
    target_dir = raw_root / dataset.raw_dir_name
    archive_path = target_dir / dataset.archive_name

    if archive_path.exists() and not options.force:
        if options.extract:
            _extract_archive(archive_path, target_dir, force=False)
        print(
            json.dumps(
                {
                    "ok": True,
                    "dataset": dataset.name,
                    "archive": str(archive_path),
                    "skipped": True,
                },
                ensure_ascii=False,
            )
        )
        return

    if not settings.mdc_api_key:
        raise RuntimeError("請在專案根目錄 .env 設定 MDC_API_KEY。")

    target_dir.mkdir(parents=True, exist_ok=True)
    download_url = _request_download_url(dataset, settings.mdc_api_key)
    _download_file(download_url, archive_path)
    if options.extract:
        _extract_archive(archive_path, target_dir, force=True)

    print(
        json.dumps(
            {
                "ok": True,
                "dataset": dataset.name,
                "archive": str(archive_path),
                "extracted_root": str(target_dir / dataset.extracted_dir),
            },
            ensure_ascii=False,
        )
    )


def prepare_common_voice_samples(
    dataset: CommonVoiceDataset,
    options: DatasetSampleOptions,
    settings: SttEvalSettings,
) -> None:
    raw_root = options.raw_root or settings.raw_root
    sample_root = options.sample_root or settings.sample_root
    source_dir = raw_root / dataset.raw_dir_name / dataset.extracted_dir
    sample_dir = sample_root / dataset.name

    _ensure_extracted_dataset(source_dir)
    clip_durations = _read_clip_durations(source_dir / "clip_durations.tsv")
    rows = _collect_rows(source_dir, clip_durations)
    selected_rows = rows[: options.count]

    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sample_dir / f"largest_{options.count}_transcripts.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["rank", "path", "duration_ms", "source_split", "sentence"],
            delimiter="\t",
        )
        writer.writeheader()
        for rank, row in enumerate(selected_rows, start=1):
            source_clip = source_dir / "clips" / row["path"]
            target_clip = sample_dir / row["path"]
            if options.force or not target_clip.exists():
                shutil.copy2(source_clip, target_clip)
            writer.writerow(
                {
                    "rank": rank,
                    "path": row["path"],
                    "duration_ms": row["duration_ms"],
                    "source_split": row["source_split"],
                    "sentence": row["sentence"],
                }
            )

    print(
        json.dumps(
            {
                "ok": True,
                "dataset": dataset.name,
                "samples": len(selected_rows),
                "sample_dir": str(sample_dir),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )


def _request_download_url(dataset: CommonVoiceDataset, api_key: str) -> str:
    request = urllib.request.Request(
        MOZILLA_DOWNLOAD_ENDPOINT.format(dataset_id=dataset.dataset_id),
        data=b"{}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Mozilla Data Collective 下載授權失敗：{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"無法連線到 Mozilla Data Collective：{exc}") from exc

    download_url = _find_download_url(payload)
    if not download_url:
        raise RuntimeError(f"下載 API 回應沒有 presigned URL：{payload}")
    return download_url


def _find_download_url(payload: Any) -> str | None:
    if isinstance(payload, str) and payload.startswith("http"):
        return payload
    if isinstance(payload, dict):
        for key in ("url", "download_url", "downloadUrl", "href"):
            value = payload.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        for value in payload.values():
            found = _find_download_url(value)
            if found:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _find_download_url(item)
            if found:
                return found
    return None


def _download_file(url: str, target_path: Path) -> None:
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "stt-eval/0.1.0"})
    try:
        with urllib.request.urlopen(request) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
    except urllib.error.URLError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"下載資料集失敗：{exc}") from exc
    tmp_path.replace(target_path)


def _extract_archive(archive_path: Path, target_dir: Path, force: bool) -> None:
    marker = target_dir / ".extract-complete"
    if marker.exists() and not force:
        return
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(target_dir, filter="data")
    marker.write_text(archive_path.name + "\n", encoding="utf-8")


def _ensure_extracted_dataset(source_dir: Path) -> None:
    required_paths = [source_dir / "clips", source_dir / "clip_durations.tsv"]
    required_paths.extend(source_dir / f"{split}.tsv" for split in SAMPLE_SPLITS)
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(
            "找不到解壓後的 Common Voice 資料。請先執行 "
            "`uv run stt-eval download-dataset --dataset nan-tw`。缺少："
            + ", ".join(str(path) for path in missing)
        )


def _read_clip_durations(path: Path) -> dict[str, int]:
    durations = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            durations[row["clip"]] = int(row["duration[ms]"])
    return durations


def _collect_rows(
    source_dir: Path, clip_durations: dict[str, int]
) -> list[dict[str, str]]:
    rows_by_path: dict[str, dict[str, str]] = {}
    for split in SAMPLE_SPLITS:
        with (source_dir / f"{split}.tsv").open(
            "r", encoding="utf-8", newline=""
        ) as file:
            reader = csv.DictReader(file, delimiter="\t")
            for row in reader:
                clip_name = row["path"]
                if clip_name in rows_by_path or clip_name not in clip_durations:
                    continue
                rows_by_path[clip_name] = {
                    "path": clip_name,
                    "duration_ms": str(clip_durations[clip_name]),
                    "source_split": split,
                    "sentence": row["sentence"],
                }
    rows = list(rows_by_path.values())
    rows.sort(key=lambda row: (-int(row["duration_ms"]), row["path"]))
    return rows
