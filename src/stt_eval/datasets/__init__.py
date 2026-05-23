from typing import Literal

from stt_eval.config import SttEvalSettings
from stt_eval.datasets.base import DatasetDownloadOptions, DatasetSampleOptions
from stt_eval.datasets.common_voice import (
    COMMON_VOICE_DATASETS,
    download_common_voice_dataset,
    prepare_common_voice_samples,
)
from stt_eval.datasets.huggingface import (
    HUGGINGFACE_DATASETS,
    download_huggingface_dataset,
    prepare_huggingface_samples,
)

DatasetBackend = Literal["common_voice", "huggingface"]
DATASET_BACKENDS: dict[str, DatasetBackend] = {
    **dict.fromkeys(COMMON_VOICE_DATASETS, "common_voice"),
    **dict.fromkeys(HUGGINGFACE_DATASETS, "huggingface"),
}


def download_dataset(options: DatasetDownloadOptions) -> None:
    settings = SttEvalSettings()
    dataset_name = options.dataset or settings.default_dataset
    backend = _resolve_backend(dataset_name)
    if backend == "common_voice":
        download_common_voice_dataset(
            COMMON_VOICE_DATASETS[dataset_name],
            options,
            settings,
        )
        return
    download_huggingface_dataset(HUGGINGFACE_DATASETS[dataset_name], options, settings)


def prepare_dataset_samples(options: DatasetSampleOptions) -> None:
    settings = SttEvalSettings()
    dataset_name = options.dataset or settings.default_dataset
    backend = _resolve_backend(dataset_name)
    if backend == "common_voice":
        prepare_common_voice_samples(
            COMMON_VOICE_DATASETS[dataset_name],
            options,
            settings,
        )
        return
    prepare_huggingface_samples(HUGGINGFACE_DATASETS[dataset_name], options, settings)


def dataset_choices() -> tuple[str, ...]:
    return tuple(sorted(DATASET_BACKENDS))


def _resolve_backend(dataset_name: str) -> DatasetBackend:
    try:
        return DATASET_BACKENDS[dataset_name]
    except KeyError as exc:
        choices = ", ".join(dataset_choices())
        raise RuntimeError(
            f"不支援的 dataset：{dataset_name}。目前可用：{choices}"
        ) from exc


__all__ = [
    "DatasetDownloadOptions",
    "DatasetSampleOptions",
    "dataset_choices",
    "download_dataset",
    "prepare_dataset_samples",
]
