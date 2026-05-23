from dataclasses import dataclass
from pathlib import Path

DEFAULT_SAMPLE_COUNT = 100


@dataclass(frozen=True)
class DatasetDownloadOptions:
    dataset: str | None = None
    raw_root: Path | None = None
    revision: str | None = None
    force: bool = False
    extract: bool = True


@dataclass(frozen=True)
class DatasetSampleOptions:
    dataset: str | None = None
    raw_root: Path | None = None
    sample_root: Path | None = None
    revision: str | None = None
    count: int = DEFAULT_SAMPLE_COUNT
    force: bool = False
