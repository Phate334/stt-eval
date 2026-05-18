import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from stt_eval.constants import MODEL_ID
from stt_eval.runtime import directory_size_bytes


class ModelArtifactMetadata(BaseModel):
    variant: str
    backend: str
    quantization: str
    source_model_id: str = MODEL_ID
    source_revision: str
    artifact_path: str
    conversion_command: list[str] = Field(default_factory=list)
    runtime_versions: dict[str, str | None] = Field(default_factory=dict)
    platform_notes: list[str] = Field(default_factory=list)
    model_size_bytes: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    license_reference: str = f"https://huggingface.co/{MODEL_ID}"


def write_metadata(path: Path, metadata: ModelArtifactMetadata) -> None:
    path.mkdir(parents=True, exist_ok=True)
    refreshed = metadata.model_copy(
        update={
            "artifact_path": str(path),
            "model_size_bytes": directory_size_bytes(path),
        }
    )
    (path / "metadata.json").write_text(
        json.dumps(refreshed.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_readme(path: Path, metadata: ModelArtifactMetadata) -> None:
    path.mkdir(parents=True, exist_ok=True)
    readme = f"""{_format_model_card_metadata(metadata)}
# {metadata.variant}

{_format_model_summary(metadata)}

## Source Model

- Model: [{metadata.source_model_id}]({metadata.license_reference})
- Revision: `{metadata.source_revision}`
- License: Apache-2.0, as declared by the source model card

## Quantization

{_format_quantization_summary(metadata)}
"""
    (path / "README.md").write_text(readme, encoding="utf-8")


def write_quantization_record(path: Path, metadata: ModelArtifactMetadata) -> None:
    path.mkdir(parents=True, exist_ok=True)
    content = f"""# Quantization Record: {metadata.variant}

## Source

- Model: `{metadata.source_model_id}`
- Revision: `{metadata.source_revision}`
- Backend: `{metadata.backend}`
- Quantization: `{metadata.quantization}`

## Command

```bash
{" ".join(metadata.conversion_command)}
```

## Runtime Versions

{_format_key_values(metadata.runtime_versions)}

## Platform Notes

{_format_notes(metadata.platform_notes)}
"""
    (path / "QUANTIZATION.md").write_text(content, encoding="utf-8")


def _format_notes(notes: list[str]) -> str:
    if not notes:
        return "- No special platform notes recorded."
    return "\n".join(f"- {note}" for note in notes)


def _format_model_summary(metadata: ModelArtifactMetadata) -> str:
    if metadata.quantization == "f32":
        return f"This model is an unquantized snapshot of `{metadata.source_model_id}`."
    return (
        f"This model is a quantized `{metadata.backend}` version of "
        f"`{metadata.source_model_id}`. No additional training is documented for this "
        "artifact."
    )


def _format_quantization_summary(metadata: ModelArtifactMetadata) -> str:
    if metadata.quantization == "f32":
        return "- Quantization: none (`f32`)"
    values = ", ".join(f"`{value}`" for value in metadata.quantization.split(","))
    return f"- Backend: `{metadata.backend}`\n- Quantization: {values}"


def _format_model_card_metadata(metadata: ModelArtifactMetadata) -> str:
    tags = [
        "automatic-speech-recognition",
        "whisper",
        "taiwanese-hokkien",
        "taigi",
        "low-resource-language",
        _tag_value(metadata.backend),
    ]
    if metadata.quantization != "f32":
        tags.append("quantized")
        tags.extend(_tag_value(value) for value in metadata.quantization.split(","))

    lines = [
        "---",
        "base_model: " + metadata.source_model_id,
    ]
    if metadata.quantization != "f32":
        lines.append("base_model_relation: quantized")
    lines.extend(
        [
            "language:",
            "- nan",
            "- zh",
            f"library_name: {_library_name(metadata.backend)}",
            "license: apache-2.0",
            "metrics:",
            "- cer",
            "pipeline_tag: automatic-speech-recognition",
            "tags:",
        ]
    )
    lines.extend(f"- {tag}" for tag in dict.fromkeys(tags))
    lines.extend(["---", ""])
    return "\n".join(lines)


def _library_name(backend: str) -> str:
    if backend == "ctranslate2":
        return "ctranslate2"
    if backend == "whisper.cpp":
        return "whisper.cpp"
    return "transformers"


def _tag_value(value: str) -> str:
    return value.lower().replace(" ", "-").replace(".", "-")


def _format_key_values(values: dict[str, str | None]) -> str:
    if not values:
        return "- No runtime versions recorded."
    return "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(values.items()))


def validate_artifact_dir(path: Path) -> list[str]:
    errors = []
    if not path.exists():
        return [f"missing artifact directory: {path}"]
    for filename in ("metadata.json", "README.md", "QUANTIZATION.md"):
        if not (path / filename).exists():
            errors.append(f"missing {filename} in {path}")
    return errors
