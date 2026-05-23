import argparse
import json
from pathlib import Path

from stt_eval.artifacts import validate_artifact_dir
from stt_eval.constants import (
    ALL_ARTIFACT_DIRS,
    ALL_VARIANTS,
    DEFAULT_MODEL_ROOT,
    DEFAULT_TOOL_ROOT,
)
from stt_eval.prepare import PrepareOptions, prepare_models


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stt-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_prepare_parser(subparsers)
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


def _add_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("validate-artifacts")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)


def _validate(model_root: Path) -> None:
    errors = []
    for artifact_dir in ALL_ARTIFACT_DIRS:
        errors.extend(validate_artifact_dir(model_root / artifact_dir))
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps({"ok": True}, ensure_ascii=False))
