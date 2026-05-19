import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from huggingface_hub import HfApi

from stt_eval.artifacts import (
    ModelArtifactMetadata,
    write_metadata,
    write_quantization_record,
    write_readme,
)
from stt_eval.constants import (
    ARTIFACT_DIR_BY_VARIANT,
    COMMON_HF_COPY_FILES,
    CT2_QUANTIZATION_BY_VARIANT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_TOOL_ROOT,
    HF_VARIANT,
    MODEL_ID,
    WHISPERCPP_QUANTIZATION_BY_VARIANT,
)
from stt_eval.runtime import collect_runtime_metadata, command_output


@dataclass(frozen=True)
class PrepareOptions:
    model_id: str = MODEL_ID
    revision: str | None = None
    model_root: Path = DEFAULT_MODEL_ROOT
    tool_root: Path = DEFAULT_TOOL_ROOT
    variants: tuple[str, ...] = (
        HF_VARIANT,
        *CT2_QUANTIZATION_BY_VARIANT.keys(),
        *WHISPERCPP_QUANTIZATION_BY_VARIANT.keys(),
    )
    skip_existing: bool = True
    whispercpp_dir: Path | None = None


def prepare_models(options: PrepareOptions) -> None:
    revision = options.revision or _resolve_revision(options.model_id)
    hf_dir = options.model_root / HF_VARIANT
    if HF_VARIANT in options.variants:
        _download_hf_model(options.model_id, revision, hf_dir, options.skip_existing)
        _write_hf_metadata(hf_dir, options.model_id, revision)

    for variant, quantization in CT2_QUANTIZATION_BY_VARIANT.items():
        if variant not in options.variants:
            continue
        _convert_ct2(
            hf_dir,
            options.model_root / ARTIFACT_DIR_BY_VARIANT[variant],
            quantization,
            revision,
        )

    whisper_variants = [
        variant
        for variant in WHISPERCPP_QUANTIZATION_BY_VARIANT
        if variant in options.variants
    ]
    if whisper_variants:
        whispercpp_dir = _ensure_whispercpp(options.tool_root, options.whispercpp_dir)
        f16_model = _convert_whispercpp_base(hf_dir, whispercpp_dir, options.tool_root)
        target_dir = options.model_root / ARTIFACT_DIR_BY_VARIANT[whisper_variants[0]]
        commands = []
        quantizations = []
        for variant in whisper_variants:
            quantization = WHISPERCPP_QUANTIZATION_BY_VARIANT[variant]
            command = _quantize_whispercpp(
                whispercpp_dir=whispercpp_dir,
                source_model=f16_model,
                target_dir=target_dir,
                quantization=quantization,
            )
            commands.append(command)
            quantizations.append(quantization)
        _write_whispercpp_metadata(
            whispercpp_dir=whispercpp_dir,
            target_dir=target_dir,
            quantizations=quantizations,
            revision=revision,
            commands=commands,
        )


def _resolve_revision(model_id: str) -> str:
    info = HfApi().model_info(model_id)
    if not info.sha:
        raise RuntimeError(f"Could not resolve Hugging Face revision for {model_id}")
    return str(info.sha)


def _run(command: list[str], cwd: Path | None = None) -> None:
    printable = " ".join(command)
    try:
        completed = subprocess.run(command, cwd=cwd, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Command not found while running model preparation: {command[0]}. "
            "Run model preparation through uv with the required extras, for example: "
            "uv run --extra hf --extra ct2 stt-eval prepare-models"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {printable}")


def _download_hf_model(
    model_id: str,
    revision: str,
    target_dir: Path,
    skip_existing: bool,
) -> None:
    if skip_existing and (target_dir / "config.json").exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "hf",
        "download",
        model_id,
        "--revision",
        revision,
        "--local-dir",
        str(target_dir),
    ]
    _run(command)


def _write_hf_metadata(path: Path, model_id: str, revision: str) -> None:
    metadata = ModelArtifactMetadata(
        variant=HF_VARIANT,
        backend="transformers",
        quantization="f32",
        source_model_id=model_id,
        source_revision=revision,
        artifact_path=str(path),
        conversion_command=[
            "hf",
            "download",
            model_id,
            "--revision",
            revision,
            "--local-dir",
            str(path),
        ],
        runtime_versions=collect_runtime_metadata(),
    )
    write_metadata(path, metadata)
    write_readme(path, metadata)
    write_quantization_record(path, metadata)


def _convert_ct2(
    hf_dir: Path,
    target_dir: Path,
    quantization: str,
    revision: str,
) -> None:
    _ensure_tokenizer_json(hf_dir)
    copy_files = [name for name in COMMON_HF_COPY_FILES if (hf_dir / name).exists()]
    command = [
        "ct2-transformers-converter",
        "--model",
        str(hf_dir),
        "--output_dir",
        str(target_dir),
        "--quantization",
        quantization,
        "--force",
    ]
    if copy_files:
        command.extend(["--copy_files", *copy_files])
    target_dir.mkdir(parents=True, exist_ok=True)
    if not (target_dir / "model.bin").exists():
        _run(command)
    _copy_auxiliary_files(hf_dir, target_dir, copy_files)
    _write_ct2_metadata(target_dir, quantization, revision, command)


def _copy_auxiliary_files(
    source_dir: Path, target_dir: Path, filenames: list[str]
) -> None:
    for filename in filenames:
        source = source_dir / filename
        target = target_dir / filename
        if source.exists() and not target.exists():
            shutil.copy2(source, target)


def _ensure_tokenizer_json(hf_dir: Path) -> None:
    tokenizer_json = hf_dir / "tokenizer.json"
    if tokenizer_json.exists():
        return
    try:
        transformers = import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "Generating tokenizer.json requires the hf extra. "
            "Run model preparation through uv with both extras, for example: "
            "uv run --extra hf --extra ct2 stt-eval prepare-models"
        ) from exc

    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)


def _write_ct2_metadata(
    target_dir: Path,
    quantization: str,
    revision: str,
    command: list[str],
) -> None:
    variant = f"Breeze-ASR-26-{quantization}-CT2"
    metadata = ModelArtifactMetadata(
        variant=variant,
        backend="ctranslate2",
        quantization=quantization,
        source_revision=revision,
        artifact_path=str(target_dir),
        conversion_command=command,
        runtime_versions=collect_runtime_metadata(),
        platform_notes=[
            "Converted with CTranslate2 for evaluation and model artifact comparison."
        ],
    )
    write_metadata(target_dir, metadata)
    write_readme(target_dir, metadata)
    write_quantization_record(target_dir, metadata)


def _ensure_whispercpp(tool_root: Path, provided_dir: Path | None) -> Path:
    if provided_dir:
        return provided_dir
    target = tool_root / "whisper.cpp"
    if (target / ".git").exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "https://github.com/ggml-org/whisper.cpp.git", str(target)])
    _run(["cmake", "-B", "build"], cwd=target)
    _run(["cmake", "--build", "build", "--config", "Release"], cwd=target)
    return target


def _convert_whispercpp_base(
    hf_dir: Path,
    whispercpp_dir: Path,
    tool_root: Path,
) -> Path:
    output_dir = tool_root / "whispercpp-base"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_model = output_dir / "ggml-model.bin"
    if output_model.exists():
        return output_model

    script = _find_whispercpp_convert_script(whispercpp_dir)
    whisper_repo = _ensure_openai_whisper_repo(tool_root)
    command = [
        sys.executable,
        str(script),
        str(hf_dir),
        str(whisper_repo),
        str(output_dir),
    ]
    _run(command)
    if not output_model.exists():
        candidates = list(output_dir.glob("ggml*.bin"))
        if not candidates:
            raise RuntimeError(f"whisper.cpp conversion did not create {output_model}")
        return candidates[0]
    return output_model


def _find_whispercpp_convert_script(whispercpp_dir: Path) -> Path:
    candidates = (
        whispercpp_dir / "models" / "convert-h5-to-ggml.py",
        whispercpp_dir / "models" / "convert-whisper-to-ggml.py",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"Could not find whisper.cpp Hugging Face conversion script under {whispercpp_dir}"
    )


def _ensure_openai_whisper_repo(tool_root: Path) -> Path:
    target = tool_root / "openai-whisper"
    if (target / ".git").exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "https://github.com/openai/whisper.git", str(target)])
    return target


def _quantize_whispercpp(
    whispercpp_dir: Path,
    source_model: Path,
    target_dir: Path,
    quantization: str,
) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    output_model = target_dir / f"ggml-model-{quantization}.bin"
    quantize = _find_executable(
        (
            whispercpp_dir / "build" / "bin" / "quantize",
            whispercpp_dir / "build" / "bin" / "whisper-quantize",
            whispercpp_dir / "quantize",
        ),
        required=not output_model.exists(),
    )
    command = [str(quantize), str(source_model), str(output_model), quantization]
    if not output_model.exists():
        _run(command)
    return command


def _write_whispercpp_metadata(
    whispercpp_dir: Path,
    target_dir: Path,
    quantizations: list[str],
    revision: str,
    commands: list[list[str]],
) -> None:
    recorded_quantizations = [
        quantization
        for quantization in WHISPERCPP_QUANTIZATION_BY_VARIANT.values()
        if quantization in quantizations
        or (target_dir / f"ggml-model-{quantization}.bin").exists()
    ]
    metadata = ModelArtifactMetadata(
        variant=target_dir.name,
        backend="whisper.cpp",
        quantization=",".join(recorded_quantizations),
        source_revision=revision,
        artifact_path=str(target_dir),
        conversion_command=[" && ".join(" ".join(command) for command in commands)],
        runtime_versions={
            **collect_runtime_metadata(),
            "whispercpp_git_revision": command_output(
                ["git", "-C", str(whispercpp_dir), "rev-parse", "HEAD"]
            ),
        },
        platform_notes=[
            "Converted to GGML for whisper.cpp quantized artifact evaluation.",
            "GGML convention keeps multiple quantized files for the same model in one repository.",
        ],
    )
    write_metadata(target_dir, metadata)
    write_readme(target_dir, metadata)
    write_quantization_record(target_dir, metadata)


def _find_executable(candidates: tuple[Path, ...], required: bool = True) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    binary = shutil.which("quantize") or shutil.which("whisper-quantize")
    if binary:
        return Path(binary)
    if not required:
        return candidates[0]
    raise RuntimeError("Could not find whisper.cpp quantize executable")
