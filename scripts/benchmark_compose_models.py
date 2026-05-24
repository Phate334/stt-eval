from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from huggingface_hub import HfApi
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data/samples/moe-example-sentences-longest-hanzi-100"
ERROR_PATTERNS = (
    re.compile(r"\b(error|exception|traceback|failed|fatal|cuda out of memory|oom)\b", re.I),
)


@dataclass(frozen=True)
class Target:
    slug: str
    family: str
    variant: str
    compose_file: str
    client_model: str
    env: dict[str, str]
    hf_repo: str | None = None
    required_file: str | None = None


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def run(command: list[str], *, env: dict[str, str] | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        timeout=timeout,
        check=False,
        capture_output=True,
        text=True,
    )


def docker_compose(target: Target, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return run(["docker", "compose", "-p", project_name(target), "-f", target.compose_file, *args], env=env)


def project_name(target: Target) -> str:
    return f"stteval-{target.slug}".replace("_", "-").lower()


def build_targets() -> list[Target]:
    targets = [
        Target(
            slug="vllm-hf-float16",
            family="vllm",
            variant="hf-float16",
            compose_file="compose.vllm-whisper.cuda.yml",
            client_model="breeze-asr-26",
            hf_repo="MediaTek-Research/Breeze-ASR-26",
            env={
                "MODEL_ID": "MediaTek-Research/Breeze-ASR-26",
                "SERVED_MODEL_NAME": "breeze-asr-26",
                "DTYPE": "float16",
            },
        )
    ]
    for variant in ("float16", "int8_float16", "int8"):
        repo = f"phate334/Breeze-ASR-26-{variant}-CT2"
        targets.append(
            Target(
                slug=f"speaches-ct2-{variant}",
                family="ct2",
                variant=variant,
                compose_file="compose.speaches-ct2.cuda.yml",
                client_model=repo,
                hf_repo=repo,
                env={"MODEL_ID": repo, "MODEL_NAME": "whisper-1"},
            )
        )
    for variant in ("q4_0", "q4_1", "q5_0", "q8_0"):
        filename = f"ggml-model-{variant}.bin"
        targets.append(
            Target(
                slug=f"whisper-cpp-ggml-{variant}",
                family="ggml",
                variant=variant,
                compose_file="compose.whisper-cpp-ggml.cuda.yml",
                client_model="whisper-1",
                hf_repo="phate334/Breeze-ASR-26-GGML",
                required_file=filename,
                env={
                    "HF_MODEL_REPO": "phate334/Breeze-ASR-26-GGML",
                    "MODEL_FILE": filename,
                    "LANGUAGE": "auto",
                },
            )
        )
    return targets


def verify_hf_access(target: Target, token: str | None) -> dict[str, Any]:
    if not target.hf_repo:
        return {"ok": True}
    api = HfApi(token=token)
    info = api.model_info(target.hf_repo, files_metadata=True)
    files = {s.rfilename: getattr(s, "size", None) for s in info.siblings}
    if target.required_file and target.required_file not in files:
        raise RuntimeError(f"{target.hf_repo} 缺少 {target.required_file}")
    return {
        "ok": True,
        "repo": target.hf_repo,
        "sha": info.sha,
        "required_file": target.required_file,
        "required_file_size": files.get(target.required_file) if target.required_file else None,
        "file_count": len(files),
    }


def wait_ready(client: OpenAI, model: str, sample: Path, deadline: float) -> None:
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with sample.open("rb") as audio:
                client.audio.transcriptions.create(model=model, file=audio)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            if (
                "Invalid or unsupported audio file" in last_error
                or "Please install vllm[audio]" in last_error
                or "BadRequestError" in last_error
                or "NotFoundError" in last_error
            ):
                raise RuntimeError(f"服務已啟動，但 warmup transcription 失敗：{last_error}") from exc
            time.sleep(5)
    raise TimeoutError(f"服務未在期限內可推論，最後錯誤：{last_error}")


def http_request(method: str, url: str, timeout: int = 10) -> tuple[int, str]:
    request = Request(url, method=method)
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.status, response.read().decode("utf-8", errors="replace")


def wait_http_ok(url: str, deadline: float) -> None:
    last_error = ""
    while time.monotonic() < deadline:
        try:
            status, _ = http_request("GET", url, timeout=5)
            if 200 <= status < 300:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"HTTP endpoint 未 ready：{url}；最後錯誤：{last_error}")


def prepare_service(target: Target) -> dict[str, Any]:
    if target.family != "ct2":
        return {"ok": True, "action": "none"}
    wait_http_ok("http://127.0.0.1:8080/v1/models", time.monotonic() + 300)
    model_id = quote(target.client_model, safe="")
    status, body = http_request("POST", f"http://127.0.0.1:8080/v1/models/{model_id}", timeout=900)
    return {"ok": 200 <= status < 300, "action": "download_model", "status": status, "body": body[:1000]}


def nvidia_memory() -> dict[str, Any]:
    completed = run(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 4:
                rows.append(
                    {
                        "timestamp": parts[0],
                        "name": parts[1],
                        "memory_used_mib": int(parts[2]),
                        "memory_total_mib": int(parts[3]),
                    }
                )
    return {"ok": completed.returncode == 0, "gpus": rows, "stderr": completed.stderr.strip()}


def collect_logs(target: Target, env: dict[str, str], out_dir: Path) -> tuple[str, list[str]]:
    completed = docker_compose(target, ["logs", "--no-color"], env)
    text = (completed.stdout or "") + (completed.stderr or "")
    (out_dir / "docker.log").write_text(text)
    errors = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in ERROR_PATTERNS):
            errors.append(line[-500:])
    return text, errors


def transcribe_all(target: Target, out_dir: Path) -> dict[str, Any]:
    client = OpenAI(api_key="local", base_url="http://127.0.0.1:8080/v1")
    wavs = sorted(SAMPLE_DIR.glob("*.wav"))
    result_path = out_dir / f"{target.slug}.jsonl"
    memory_samples = []
    successes = 0
    failures = 0
    started_at = datetime.now(UTC).isoformat()
    with result_path.open("w") as output:
        wait_ready(client, target.client_model, wavs[0], time.monotonic() + 900)
        memory_samples.append({"phase": "ready_after_warmup", **nvidia_memory()})
        for index, wav in enumerate(wavs, start=1):
            before = time.monotonic()
            record: dict[str, Any] = {
                "sample_index": index,
                "audio_path": str(wav.relative_to(ROOT)),
                "model": target.client_model,
                "target": target.slug,
            }
            try:
                with wav.open("rb") as audio:
                    response = client.audio.transcriptions.create(
                        model=target.client_model,
                        file=audio,
                    )
                record["ok"] = True
                record["duration_seconds"] = round(time.monotonic() - before, 3)
                record["response"] = response.model_dump(mode="json")
                successes += 1
            except Exception as exc:  # noqa: BLE001
                record["ok"] = False
                record["duration_seconds"] = round(time.monotonic() - before, 3)
                record["error"] = repr(exc)
                failures += 1
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if index == 1 or index % 10 == 0 or index == len(wavs):
                memory_samples.append({"phase": f"after_{index}", **nvidia_memory()})
    return {
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "jsonl": str(result_path.relative_to(ROOT)),
        "samples": len(wavs),
        "successes": successes,
        "failures": failures,
        "memory_samples": memory_samples,
    }


def write_markdown(run_dir: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Compose 模型逐版推論紀錄",
        "",
        f"- 執行時間：{datetime.now(UTC).isoformat()}",
        f"- 音檔目錄：`{SAMPLE_DIR.relative_to(ROOT)}`",
        f"- 輸出目錄：`{run_dir.relative_to(ROOT)}`",
        "",
        "| 服務 | 版本 | 結果 | 成功/總數 | VRAM MiB | JSONL | 錯誤摘要 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for item in summaries:
        transcribe = item.get("transcribe") or {}
        memory_values = [
            gpu["memory_used_mib"]
            for sample in transcribe.get("memory_samples", [])
            for gpu in sample.get("gpus", [])
        ]
        vram = f"{min(memory_values)}-{max(memory_values)}" if memory_values else "n/a"
        total = transcribe.get("samples", 0)
        successes = transcribe.get("successes", 0)
        jsonl = transcribe.get("jsonl", "")
        errors = item.get("errors") or []
        if not errors and transcribe.get("failures", 0):
            errors = [f"{transcribe['failures']} 筆推論失敗，詳見 jsonl"]
        error_text = "<br>".join(str(error).replace("|", "\\|")[:160] for error in errors[:3]) or ""
        lines.append(
            f"| {item['family']} | {item['variant']} | {item['status']} | "
            f"{successes}/{total} | {vram} | `{jsonl}` | {error_text} |"
        )
    (run_dir / "benchmark-report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", help="只跑指定 slug，可重複指定")
    parser.add_argument("--skip-vllm", action="store_true")
    args = parser.parse_args()

    dotenv = load_dotenv(ROOT / ".env")
    env_base = os.environ.copy()
    env_base.update(dotenv)
    token = env_base.get("HF_TOKEN")
    if not token:
        print("缺少 .env 的 HF_TOKEN", file=sys.stderr)
        return 2

    run_dir = ROOT / "runs" / f"compose-benchmark-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    targets = build_targets()
    if args.skip_vllm:
        targets = [target for target in targets if target.family != "vllm"]
    if args.only:
        selected = set(args.only)
        targets = [target for target in targets if target.slug in selected]

    summaries: list[dict[str, Any]] = []
    for target in targets:
        target_dir = run_dir / target.slug
        target_dir.mkdir()
        env = env_base | target.env
        summary: dict[str, Any] = {
            "slug": target.slug,
            "family": target.family,
            "variant": target.variant,
            "compose_file": target.compose_file,
            "status": "failed",
            "errors": [],
        }
        print(f"==> {target.slug}", flush=True)
        try:
            summary["hf"] = verify_hf_access(target, token)
            up = docker_compose(target, ["up", "-d", "--remove-orphans"], env)
            (target_dir / "compose-up.stdout").write_text(up.stdout)
            (target_dir / "compose-up.stderr").write_text(up.stderr)
            if up.returncode != 0:
                raise RuntimeError(f"docker compose up 失敗：{up.stderr.strip() or up.stdout.strip()}")
            summary["service_prepare"] = prepare_service(target)
            summary["transcribe"] = transcribe_all(target, target_dir)
            _, log_errors = collect_logs(target, env, target_dir)
            summary["log_error_lines"] = log_errors
            if log_errors:
                summary["errors"].extend(log_errors[:5])
            if summary["transcribe"]["failures"]:
                summary["status"] = "partial"
            else:
                summary["status"] = "ok" if not log_errors else "ok_with_log_warnings"
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(repr(exc))
            try:
                collect_logs(target, env, target_dir)
            except Exception as log_exc:  # noqa: BLE001
                summary["errors"].append(f"讀取 log 失敗：{log_exc!r}")
        finally:
            down = docker_compose(target, ["down"], env)
            (target_dir / "compose-down.stdout").write_text(down.stdout)
            (target_dir / "compose-down.stderr").write_text(down.stderr)
            (target_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            summaries.append(summary)
            write_markdown(run_dir, summaries)
    return 1 if any(item["status"] not in {"ok", "ok_with_log_warnings"} for item in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
