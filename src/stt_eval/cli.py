import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from stt_eval.artifacts import validate_artifact_dir
from stt_eval.constants import (
    ALL_ARTIFACT_DIRS,
    ALL_VARIANTS,
    DEFAULT_MODEL_ROOT,
    DEFAULT_TOOL_ROOT,
)
from stt_eval.evaluation import EvalOptions, EvalResultRow, run_eval, summarize_results
from stt_eval.openai_transcription import (
    TranscriptionOptions,
    payload_to_json,
    transcribe_files,
)
from stt_eval.prepare import PrepareOptions, prepare_models


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stt-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_prepare_parser(subparsers)
    _add_run_eval_parser(subparsers)
    _add_transcribe_openai_parser(subparsers)
    _add_summarize_parser(subparsers)
    _add_validate_parser(subparsers)
    args = parser.parse_args(argv)

    if args.command == "prepare-models":
        prepare_models(
            PrepareOptions(
                revision=args.revision,
                model_root=args.model_root,
                tool_root=args.tool_root,
                variants=tuple(args.variant or ALL_VARIANTS),
                skip_existing=not args.force,
                whispercpp_dir=args.whispercpp_dir,
            )
        )
        return
    if args.command == "run-eval":
        _run_eval(args)
        return
    if args.command == "transcribe-openai":
        _transcribe_openai(args)
        return
    if args.command == "summarize":
        _summarize(args.results)
        return
    if args.command == "validate-artifacts":
        _validate(args.model_root)
        return
    raise AssertionError(f"Unhandled command: {args.command}")


def _add_prepare_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("prepare-models")
    parser.add_argument("--revision")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--tool-root", type=Path, default=DEFAULT_TOOL_ROOT)
    parser.add_argument("--whispercpp-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--variant",
        action="append",
        choices=ALL_VARIANTS,
        default=None,
        help="Variant to prepare. May be passed multiple times. Defaults to all.",
    )


def _add_run_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run-eval")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--model-artifact-dir", type=Path)
    parser.add_argument("--language")
    parser.add_argument("--prompt")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=float)
    parser.add_argument("--api-key")


def _add_transcribe_openai_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("transcribe-openai")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--backend")
    parser.add_argument("--timeout-sec", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("audio_paths", nargs="+", type=Path)


def _add_summarize_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("summarize")
    parser.add_argument("--results", type=Path, required=True)


def _add_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("validate-artifacts")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)


def _run_eval(args: argparse.Namespace) -> None:
    values = {
        "manifest": args.manifest,
        "output": args.output,
        "backend": args.backend,
        "quantization": args.quantization,
        "model_artifact_dir": args.model_artifact_dir,
        "language": args.language,
        "prompt": args.prompt,
        "temperature": args.temperature,
    }
    values.update(_present_args(args, "base_url", "model", "api_key", "timeout_sec"))
    try:
        options = EvalOptions(**values)
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc
    run_eval(options)


def _transcribe_openai(args: argparse.Namespace) -> None:
    values = {"audio_paths": tuple(args.audio_paths), "output": args.output}
    values.update(
        _present_args(args, "base_url", "model", "api_key", "backend", "timeout_sec")
    )
    try:
        options = TranscriptionOptions(**values)
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc
    payload = transcribe_files(options)
    print(payload_to_json(payload))


def _present_args(args: argparse.Namespace, *keys: str) -> dict[str, object]:
    values = {}
    for key in keys:
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    return values


def _summarize(results: Path) -> None:
    rows = []
    with results.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(EvalResultRow.model_validate_json(line))
    summary = summarize_results(rows)
    for (backend, quantization), values in sorted(summary.items()):
        print(
            f"{backend}\t{quantization}\t"
            f"samples={int(values['samples'])}\t"
            f"mean_cer={values['mean_cer']:.6f}\t"
            f"mean_rtf={values['mean_rtf']:.6f}"
        )


def _validate(model_root: Path) -> None:
    errors = []
    for artifact_dir in ALL_ARTIFACT_DIRS:
        errors.extend(validate_artifact_dir(model_root / artifact_dir))
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps({"ok": True}, ensure_ascii=False))
