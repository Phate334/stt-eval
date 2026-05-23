import csv
import json
import shutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from stt_eval.config import SttEvalSettings
from stt_eval.datasets.base import DatasetDownloadOptions, DatasetSampleOptions


@dataclass(frozen=True)
class HuggingFaceDataset:
    name: str
    repo_id: str
    raw_dir_name: str
    split: str = "train"
    audio_column: str = "audio"
    text_columns: tuple[str, ...] = ("hanzi", "chinese", "minnan roman")


HUGGINGFACE_DATASETS = {
    "example-sentences": HuggingFaceDataset(
        name="example-sentences",
        repo_id="sarahwei/Taiwanese-Minnan-Example-Sentences",
        raw_dir_name="taiwanese_minnan_example_sentences",
    )
}


def download_huggingface_dataset(
    dataset: HuggingFaceDataset,
    options: DatasetDownloadOptions,
    settings: SttEvalSettings,
) -> None:
    if not settings.hf_token:
        raise RuntimeError("請在專案根目錄 .env 設定 HF_TOKEN。")

    raw_root = options.raw_root or settings.raw_root
    target_dir = raw_root / dataset.raw_dir_name
    if target_dir.exists() and any(target_dir.iterdir()) and not options.force:
        print(
            json.dumps(
                {
                    "ok": True,
                    "dataset": dataset.name,
                    "repo_id": dataset.repo_id,
                    "path": str(target_dir),
                    "skipped": True,
                },
                ensure_ascii=False,
            )
        )
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=dataset.repo_id,
        repo_type="dataset",
        revision=options.revision,
        token=settings.hf_token,
        local_dir=target_dir,
        force_download=options.force,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "dataset": dataset.name,
                "repo_id": dataset.repo_id,
                "path": snapshot_path,
            },
            ensure_ascii=False,
        )
    )


def prepare_huggingface_samples(
    dataset: HuggingFaceDataset,
    options: DatasetSampleOptions,
    settings: SttEvalSettings,
) -> None:
    if not settings.hf_token:
        raise RuntimeError("請在專案根目錄 .env 設定 HF_TOKEN。")

    datasets_module = import_module("datasets")
    raw_root = options.raw_root or settings.raw_root
    sample_root = options.sample_root or settings.sample_root
    cache_dir = raw_root / dataset.raw_dir_name / ".datasets-cache"
    sample_dir = sample_root / dataset.name
    sample_dir.mkdir(parents=True, exist_ok=True)

    loaded_dataset = datasets_module.load_dataset(
        dataset.repo_id,
        split=dataset.split,
        revision=options.revision,
        token=settings.hf_token,
        cache_dir=str(cache_dir),
    )
    if dataset.audio_column in loaded_dataset.column_names:
        loaded_dataset = loaded_dataset.cast_column(
            dataset.audio_column,
            datasets_module.Audio(decode=False),
        )

    manifest_path = sample_dir / f"first_{options.count}_transcripts.tsv"
    written_rows = 0
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["rank", "path", "source_split", *dataset.text_columns],
            delimiter="\t",
        )
        writer.writeheader()
        for row in loaded_dataset:
            if written_rows >= options.count:
                break
            audio = row.get(dataset.audio_column)
            if not isinstance(audio, dict):
                continue
            audio_path = _write_audio_sample(
                sample_dir=sample_dir,
                rank=written_rows + 1,
                audio=audio,
                force=options.force,
            )
            writer.writerow(
                {
                    "rank": written_rows + 1,
                    "path": audio_path.name,
                    "source_split": dataset.split,
                    **{
                        column: str(row.get(column, ""))
                        for column in dataset.text_columns
                    },
                }
            )
            written_rows += 1

    print(
        json.dumps(
            {
                "ok": True,
                "dataset": dataset.name,
                "repo_id": dataset.repo_id,
                "samples": written_rows,
                "sample_dir": str(sample_dir),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )


def _write_audio_sample(
    sample_dir: Path,
    rank: int,
    audio: dict[str, Any],
    force: bool,
) -> Path:
    source_path = audio.get("path")
    source_bytes = audio.get("bytes")
    suffix = _audio_suffix(source_path)
    target_path = sample_dir / f"{rank:05d}{suffix}"
    if target_path.exists() and not force:
        return target_path

    if isinstance(source_bytes, bytes):
        target_path.write_bytes(source_bytes)
        return target_path
    if isinstance(source_path, str) and Path(source_path).exists():
        shutil.copy2(source_path, target_path)
        return target_path
    raise RuntimeError(f"HF audio row 沒有可寫出的音檔內容：{audio}")


def _audio_suffix(source_path: Any) -> str:
    if isinstance(source_path, str):
        suffix = Path(source_path).suffix
        if suffix:
            return suffix
    return ".wav"
