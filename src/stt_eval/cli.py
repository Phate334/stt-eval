import argparse
import json
from pathlib import Path

from stt_eval.datasets import (
    SAMPLE_SELECTIONS,
    DatasetDownloadOptions,
    DatasetSampleOptions,
    dataset_choices,
    download_dataset,
    prepare_dataset_samples,
)
from stt_eval.metrics import (
    NORMALIZATION_MODES,
    CompareResultsOptions,
    compare_results,
    render_json_report,
    render_markdown_report,
)
from stt_eval.quantization import (
    ALL_ARTIFACT_DIRS,
    ALL_VARIANTS,
    DEFAULT_MODEL_ROOT,
    DEFAULT_TOOL_ROOT,
    PrepareOptions,
    prepare_models,
    validate_artifact_dir,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stt-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_prepare_parser(subparsers)
    _add_download_dataset_parser(subparsers)
    _add_prepare_dataset_samples_parser(subparsers)
    _add_compare_results_parser(subparsers)
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
    if args.command == "download-dataset":
        download_dataset(
            DatasetDownloadOptions(
                dataset=args.dataset,
                raw_root=args.raw_root,
                force=args.force,
                extract=not args.no_extract,
            )
        )
        return
    if args.command == "prepare-dataset-samples":
        prepare_dataset_samples(
            DatasetSampleOptions(
                dataset=args.dataset,
                raw_root=args.raw_root,
                sample_root=args.sample_root,
                count=args.count,
                selection=args.selection,
                force=args.force,
            )
        )
        return
    if args.command == "compare-results":
        candidate_paths = _resolve_candidate_paths(
            results_dir=args.results_dir,
            baseline_filename=args.baseline,
            include=tuple(args.include or ()),
        )
        baseline_path = args.results_dir / args.baseline
        comparisons = compare_results(
            CompareResultsOptions(
                baseline_path=baseline_path,
                candidate_paths=candidate_paths,
                normalization=args.normalization,
            )
        )
        if args.format == "json":
            output = render_json_report(
                baseline_path,
                comparisons,
                normalization=args.normalization,
            )
        else:
            output = render_markdown_report(
                baseline_path,
                comparisons,
                normalization=args.normalization,
            )
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        print(output, end="")
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


def _add_download_dataset_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("download-dataset")
    parser.add_argument(
        "--dataset",
        default=None,
        choices=dataset_choices(),
        help="Dataset 名稱。預設讀 .env 或設定中的 default_dataset，目前預設 moe-example-sentences。",
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-extract", action="store_true")


def _add_prepare_dataset_samples_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser("prepare-dataset-samples")
    parser.add_argument(
        "--dataset",
        default=None,
        choices=dataset_choices(),
        help="Dataset 名稱。預設讀 .env 或設定中的 default_dataset，目前預設 moe-example-sentences。",
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--selection",
        choices=SAMPLE_SELECTIONS,
        default="longest-hanzi",
        help="樣本挑選方式。預設挑 hanzi 字數最長的樣本。",
    )
    parser.add_argument("--force", action="store_true")


def _add_compare_results_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("compare-results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--baseline", default="vllm-hf-float16.jsonl")
    parser.add_argument(
        "--include",
        action="append",
        help="只比較指定檔名，可重複指定。預設比較 results 目錄下除了 baseline 以外的所有 jsonl。",
    )
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATION_MODES,
        default="strip-whitespace",
        help=(
            "CER 前處理規則。"
            "raw=不做正規化；strip-whitespace=移除所有空白；"
            "breeze-compatible=移除空白、去標點、英文轉小寫。"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="可選，將報表另外寫到指定路徑。",
    )


def _add_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("validate-artifacts")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)


def _resolve_candidate_paths(
    *,
    results_dir: Path,
    baseline_filename: str,
    include: tuple[str, ...],
) -> tuple[Path, ...]:
    baseline_path = results_dir / baseline_filename
    if not baseline_path.exists():
        raise RuntimeError(f"找不到 baseline 檔案：{baseline_path}")

    if include:
        candidate_paths = tuple(results_dir / filename for filename in include)
    else:
        candidate_paths = tuple(
            path
            for path in sorted(results_dir.glob("*.jsonl"))
            if path.name != baseline_filename
        )

    missing_paths = [path for path in candidate_paths if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise RuntimeError(f"找不到 results 檔案：{missing_text}")
    if not candidate_paths:
        raise RuntimeError("沒有可比較的 results 檔案")
    return candidate_paths


def _validate(model_root: Path) -> None:
    errors = []
    for artifact_dir in ALL_ARTIFACT_DIRS:
        errors.extend(validate_artifact_dir(model_root / artifact_dir))
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps({"ok": True}, ensure_ascii=False))
