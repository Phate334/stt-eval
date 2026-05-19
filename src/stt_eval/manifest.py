import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator


class ManifestItem(BaseModel):
    id: str = Field(min_length=1)
    audio_path: Path
    reference: str
    duration_sec: float | None = Field(default=None, ge=0)

    @field_validator("audio_path")
    @classmethod
    def audio_path_must_exist(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"audio_path does not exist: {value}")
        return value


def iter_manifest(path: Path) -> Iterator[ManifestItem]:
    base_dir = path.parent
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if "audio_path" in payload:
                audio_path = Path(payload["audio_path"])
                if not audio_path.is_absolute():
                    payload["audio_path"] = str(base_dir / audio_path)
            try:
                yield ManifestItem.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(f"Invalid manifest row {line_number}: {exc}") from exc
