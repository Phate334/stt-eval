import importlib.metadata
import platform
import subprocess
from pathlib import Path


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output or None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def directory_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def collect_runtime_metadata() -> dict[str, str | None]:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "huggingface_hub_version": package_version("huggingface-hub"),
        "transformers_version": package_version("transformers"),
        "ctranslate2_version": package_version("ctranslate2"),
    }
