from stt_eval.config import SttEvalSettings
from stt_eval.datasets.base import (
    SAMPLE_SELECTIONS,
    DatasetDownloadOptions,
    DatasetSampleOptions,
)
from stt_eval.datasets.moe import (
    MOE_DATASETS,
    MoeDataset,
    download_moe_dataset,
    prepare_moe_samples,
)

DATASETS = MOE_DATASETS


def download_dataset(options: DatasetDownloadOptions) -> None:
    settings = SttEvalSettings()
    dataset_name = options.dataset or settings.default_dataset
    dataset = _resolve_dataset(dataset_name)
    download_moe_dataset(dataset, options, settings)


def prepare_dataset_samples(options: DatasetSampleOptions) -> None:
    settings = SttEvalSettings()
    dataset_name = options.dataset or settings.default_dataset
    dataset = _resolve_dataset(dataset_name)
    prepare_moe_samples(dataset, options, settings)


def dataset_choices() -> tuple[str, ...]:
    return tuple(sorted(DATASETS))


def _resolve_dataset(dataset_name: str) -> MoeDataset:
    try:
        return DATASETS[dataset_name]
    except KeyError as exc:
        choices = ", ".join(dataset_choices())
        raise RuntimeError(
            f"不支援的 dataset：{dataset_name}。目前可用：{choices}"
        ) from exc


__all__ = [
    "DatasetDownloadOptions",
    "DatasetSampleOptions",
    "SAMPLE_SELECTIONS",
    "dataset_choices",
    "download_dataset",
    "prepare_dataset_samples",
]
