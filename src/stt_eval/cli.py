import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from stt_eval.artifacts import validate_artifact_dir
from stt_eval.constants import ALL_VARIANTS, DEFAULT_MODEL_ROOT, DEFAULT_TOOL_ROOT
from stt_eval.evaluation import EvalOptions, EvalResultRow, run_eval, summarize_results
from stt_eval.prepare import PrepareOptions, prepare_models


def main() -> None:
    parser = argparse.ArgumentParser(prog="stt-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_prepare_parser(subparsers)
    _add_run_eval_parser(subparsers)
    _add_summarize_parser(subparsers)
    _add_validate_parser(subparsers)
    args = parser.parse_args()

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
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--model-artifact-dir", type=Path)
    parser.add_argument("--language")
    parser.add_argument("--prompt")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key", default="local")


def _add_summarize_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("summarize")
    parser.add_argument("--results", type=Path, required=True)


def _add_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("validate-artifacts")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)


def _run_eval(args: argparse.Namespace) -> None:
    try:
        options = EvalOptions(
            manifest=args.manifest,
            output=args.output,
            base_url=args.base_url,
            model=args.model,
            backend=args.backend,
            quantization=args.quantization,
            model_artifact_dir=args.model_artifact_dir,
            language=args.language,
            prompt=args.prompt,
            temperature=args.temperature,
            api_key=args.api_key,
        )
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc
    run_eval(options)


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
    for variant in ALL_VARIANTS:
        errors.extend(validate_artifact_dir(model_root / variant))
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps({"ok": True}, ensure_ascii=False))
