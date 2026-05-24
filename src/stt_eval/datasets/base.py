from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_SAMPLE_COUNT = 100
type SampleSelection = Literal["first", "longest-hanzi"]
DEFAULT_SAMPLE_SELECTION: SampleSelection = "longest-hanzi"
SAMPLE_SELECTIONS: tuple[SampleSelection, ...] = ("first", "longest-hanzi")


@dataclass(frozen=True)
class DatasetDownloadOptions:
    dataset: str | None = None
    raw_root: Path | None = None
    force: bool = False
    extract: bool = True


@dataclass(frozen=True)
class DatasetSampleOptions:
    dataset: str | None = None
    raw_root: Path | None = None
    sample_root: Path | None = None
    count: int = DEFAULT_SAMPLE_COUNT
    selection: SampleSelection = DEFAULT_SAMPLE_SELECTION
    force: bool = False
