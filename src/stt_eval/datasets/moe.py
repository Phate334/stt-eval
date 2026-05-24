import csv
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import iterparse

from stt_eval.config import SttEvalSettings
from stt_eval.datasets.base import DatasetDownloadOptions, DatasetSampleOptions

ODS_TABLE_NS = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
ODS_TEXT_NS = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
USER_AGENT = "stt-eval/0.1.0"


@dataclass(frozen=True)
class MoeDataset:
    name: str
    raw_dir_name: str
    text_url: str
    audio_url: str
    text_archive_name: str
    audio_archive_name: str
    audio_extract_dir_name: str
    manifest_name: str


MOE_DATASETS = {
    "moe-example-sentences": MoeDataset(
        name="moe-example-sentences",
        raw_dir_name="moe_sutian_example_sentences",
        text_url="https://sutian.moe.edu.tw/media/senn/ods/kautian.ods",
        audio_url="https://sutian.moe.edu.tw/media/senn/leku-wav.zip",
        text_archive_name="kautian.ods",
        audio_archive_name="leku-wav.zip",
        audio_extract_dir_name="leku-wav",
        manifest_name="leku.tsv",
    )
}


def download_moe_dataset(
    dataset: MoeDataset,
    options: DatasetDownloadOptions,
    settings: SttEvalSettings,
) -> None:
    raw_root = options.raw_root or settings.raw_root
    target_dir = raw_root / dataset.raw_dir_name
    text_path = target_dir / dataset.text_archive_name
    audio_zip_path = target_dir / dataset.audio_archive_name
    audio_dir = target_dir / dataset.audio_extract_dir_name
    manifest_path = target_dir / dataset.manifest_name

    target_dir.mkdir(parents=True, exist_ok=True)
    _download_file(dataset.text_url, text_path, force=options.force)
    _download_file(dataset.audio_url, audio_zip_path, force=options.force)

    if options.extract:
        _extract_zip(audio_zip_path, audio_dir, force=options.force)
        written_rows, missing_audio = _write_example_manifest(
            ods_path=text_path,
            audio_dir=audio_dir,
            manifest_path=manifest_path,
        )
    else:
        written_rows = _write_text_only_manifest(
            ods_path=text_path,
            manifest_path=manifest_path,
        )
        missing_audio = written_rows

    print(
        json.dumps(
            {
                "ok": True,
                "dataset": dataset.name,
                "raw_dir": str(target_dir),
                "text": str(text_path),
                "audio_archive": str(audio_zip_path),
                "audio_dir": str(audio_dir) if options.extract else None,
                "manifest": str(manifest_path),
                "rows": written_rows,
                "missing_audio": missing_audio,
            },
            ensure_ascii=False,
        )
    )


def prepare_moe_samples(
    dataset: MoeDataset,
    options: DatasetSampleOptions,
    settings: SttEvalSettings,
) -> None:
    raw_root = options.raw_root or settings.raw_root
    sample_root = options.sample_root or settings.sample_root
    raw_dataset_dir = raw_root / dataset.raw_dir_name
    text_path = raw_dataset_dir / dataset.text_archive_name
    audio_dir = raw_dataset_dir / dataset.audio_extract_dir_name
    manifest_path = raw_dataset_dir / dataset.manifest_name
    sample_dir = sample_root / dataset.name
    sample_manifest_path = sample_dir / f"first_{options.count}_transcripts.tsv"

    if not manifest_path.exists():
        _ensure_downloaded(text_path, audio_dir)
        _write_example_manifest(text_path, audio_dir, manifest_path)

    selected_rows = _read_manifest_rows(manifest_path, options.count)
    sample_dir.mkdir(parents=True, exist_ok=True)
    with sample_manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["rank", "path", "source_path", "hanzi", "tailo", "mandarin"],
            delimiter="\t",
        )
        writer.writeheader()
        for rank, row in enumerate(selected_rows, start=1):
            source_path = raw_dataset_dir / row["audio_path"]
            if not source_path.exists():
                raise RuntimeError(f"找不到例句音檔：{source_path}")
            target_path = sample_dir / f"{rank:05d}{source_path.suffix}"
            if options.force or not target_path.exists():
                shutil.copy2(source_path, target_path)
            writer.writerow(
                {
                    "rank": rank,
                    "path": target_path.name,
                    "source_path": row["audio_path"],
                    "hanzi": row["hanzi"],
                    "tailo": row["tailo"],
                    "mandarin": row["mandarin"],
                }
            )

    print(
        json.dumps(
            {
                "ok": True,
                "dataset": dataset.name,
                "samples": len(selected_rows),
                "sample_dir": str(sample_dir),
                "manifest": str(sample_manifest_path),
            },
            ensure_ascii=False,
        )
    )


def _download_file(url: str, target_path: Path, force: bool) -> None:
    if target_path.exists() and not force:
        return

    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    resume_from = tmp_path.stat().st_size if tmp_path.exists() and not force else 0
    headers = {"User-Agent": USER_AGENT}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            append = resume_from > 0 and response.status == 206
            mode = "ab" if append else "wb"
            if resume_from and not append:
                print(f"伺服器不支援續傳，重新下載：{url}", file=sys.stderr)
            else:
                action = "續傳" if append else "下載"
                print(f"{action}：{url}", file=sys.stderr)
            with tmp_path.open(mode) as out:
                shutil.copyfileobj(response, out)
    except urllib.error.URLError as exc:
        if _curl_available():
            _download_file_with_curl(url, target_path, tmp_path, force=force)
            return
        raise RuntimeError(f"下載資料集失敗：{url}：{exc}") from exc
    tmp_path.replace(target_path)


def _download_file_with_curl(
    url: str,
    target_path: Path,
    tmp_path: Path,
    force: bool,
) -> None:
    if force:
        tmp_path.unlink(missing_ok=True)
    command = [
        "curl",
        "--location",
        "--fail",
        "--show-error",
        "--continue-at",
        "-",
        "--output",
        str(tmp_path),
        url,
    ]
    print(f"改用 curl 下載：{url}", file=sys.stderr)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"curl 下載資料集失敗：{url}") from exc
    tmp_path.replace(target_path)


def _curl_available() -> bool:
    return shutil.which("curl") is not None


def _extract_zip(zip_path: Path, target_dir: Path, force: bool) -> None:
    marker = target_dir / ".extract-complete"
    if marker.exists() and not force:
        return
    if force and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target_path = target_dir / member.filename
            if not _is_relative_to(target_path.resolve(), target_dir.resolve()):
                raise RuntimeError(f"ZIP 內含不安全路徑：{member.filename}")
            archive.extract(member, target_dir)

    marker.write_text(zip_path.name + "\n", encoding="utf-8")


def _write_example_manifest(
    ods_path: Path,
    audio_dir: Path,
    manifest_path: Path,
) -> tuple[int, int]:
    _ensure_downloaded(ods_path, audio_dir)
    audio_by_stem = _collect_audio_files(audio_dir)
    missing_audio = 0

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = _manifest_writer(file)
        written_rows = 0
        for row in _iter_example_rows(ods_path):
            audio_id = row["音檔檔名"]
            audio_path = audio_by_stem.get(audio_id)
            if audio_path is None:
                missing_audio += 1
                continue
            written_rows += 1
            writer.writerow(
                {
                    "rank": written_rows,
                    "audio_id": audio_id,
                    "audio_path": audio_path.relative_to(manifest_path.parent),
                    "hanzi": row["漢字"],
                    "tailo": row["羅馬字"],
                    "mandarin": row["華語"],
                }
            )
    return written_rows, missing_audio


def _write_text_only_manifest(ods_path: Path, manifest_path: Path) -> int:
    if not ods_path.exists():
        raise RuntimeError(f"找不到教育部辭典文字檔：{ods_path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = _manifest_writer(file)
        written_rows = 0
        for row in _iter_example_rows(ods_path):
            written_rows += 1
            writer.writerow(
                {
                    "rank": written_rows,
                    "audio_id": row["音檔檔名"],
                    "audio_path": "",
                    "hanzi": row["漢字"],
                    "tailo": row["羅馬字"],
                    "mandarin": row["華語"],
                }
            )
    return written_rows


def _manifest_writer(file: Any) -> csv.DictWriter:
    writer = csv.DictWriter(
        file,
        fieldnames=["rank", "audio_id", "audio_path", "hanzi", "tailo", "mandarin"],
        delimiter="\t",
    )
    writer.writeheader()
    return writer


def _iter_example_rows(ods_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    headers: list[str] | None = None
    for table_name, cells in _iter_ods_rows(ods_path):
        if table_name != "例句":
            continue
        if headers is None:
            headers = cells
            continue
        row = {header: cells[index] if index < len(cells) else "" for index, header in enumerate(headers)}
        if row.get("音檔檔名"):
            rows.append(row)
    return rows


def _iter_ods_rows(ods_path: Path) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    table_name: str | None = None
    with zipfile.ZipFile(ods_path) as ods, ods.open("content.xml") as content:
        for event, elem in iterparse(content, events=("start", "end")):
            if event == "start" and elem.tag == ODS_TABLE_NS + "table":
                table_name = elem.attrib.get(ODS_TABLE_NS + "name")
            elif event == "end" and elem.tag == ODS_TABLE_NS + "table-row" and table_name:
                row_repeat = int(elem.attrib.get(ODS_TABLE_NS + "number-rows-repeated", "1"))
                cells = _ods_row_cells(elem)
                rows.extend((table_name, cells) for _ in range(row_repeat))
                elem.clear()
            elif event == "end" and elem.tag == ODS_TABLE_NS + "table":
                table_name = None
                elem.clear()
    return rows


def _ods_row_cells(row: Any) -> list[str]:
    cells = []
    for cell in row.findall(ODS_TABLE_NS + "table-cell"):
        column_repeat = int(cell.attrib.get(ODS_TABLE_NS + "number-columns-repeated", "1"))
        value = "\n".join(
            text
            for text in ("".join(paragraph.itertext()) for paragraph in cell.findall(".//" + ODS_TEXT_NS + "p"))
            if text
        )
        cells.extend([value] * column_repeat)
    return cells


def _collect_audio_files(audio_dir: Path) -> dict[str, Path]:
    audio_files = sorted(audio_dir.rglob("*.wav"))
    if not audio_files:
        raise RuntimeError(f"找不到解壓後的例句 WAV 音檔：{audio_dir}")
    return {path.stem: path for path in audio_files}


def _read_manifest_rows(manifest_path: Path, count: int) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        rows: list[dict[str, str]] = []
        for row in reader:
            if len(rows) >= count:
                break
            if row.get("audio_path"):
                rows.append(row)
    return rows


def _ensure_downloaded(text_path: Path, audio_dir: Path) -> None:
    missing = [path for path in (text_path, audio_dir) if not path.exists()]
    if missing:
        raise RuntimeError(
            "找不到教育部例句資料。請先執行 "
            "`uv run stt-eval download-dataset`。缺少："
            + ", ".join(str(path) for path in missing)
        )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
