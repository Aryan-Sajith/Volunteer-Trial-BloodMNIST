from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitSpec:
    source: str
    indices: list[int]


@dataclass(frozen=True)
class DatasetSplits:
    dataset: str
    seed: int
    train: SplitSpec
    val: SplitSpec
    test: SplitSpec


def load_splits(path: Path) -> DatasetSplits:
    raw = json.loads(path.read_text())
    return DatasetSplits(
        dataset=raw["dataset"],
        seed=int(raw["seed"]),
        train=SplitSpec(source=raw["train"]["source"], indices=list(map(int, raw["train"]["indices"]))),
        val=SplitSpec(source=raw["val"]["source"], indices=list(map(int, raw["val"]["indices"]))),
        test=SplitSpec(source=raw["test"]["source"], indices=list(map(int, raw["test"]["indices"]))),
    )


def save_splits(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class _NumpyClassificationDataset:
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self._images = images
        self._labels = labels
        self._transform = transform

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, idx: int):
        image = self._images[idx]
        label = int(self._labels[idx])

        # medmnist returns HWC uint8
        pil = Image.fromarray(image)
        if self._transform is not None:
            pil = self._transform(pil)
        return pil, label


def load_bloodmnist_trainval(root: Path, transform=None) -> _NumpyClassificationDataset:
    import medmnist

    ds_train = medmnist.BloodMNIST(root=str(root), split="train", download=True)
    ds_val = medmnist.BloodMNIST(root=str(root), split="val", download=True)

    x = np.concatenate([ds_train.imgs, ds_val.imgs], axis=0)
    y = np.concatenate([ds_train.labels, ds_val.labels], axis=0).squeeze()
    return _NumpyClassificationDataset(x, y, transform=transform)


def load_bloodmnist_test(root: Path, transform=None) -> _NumpyClassificationDataset:
    import medmnist

    ds_test = medmnist.BloodMNIST(root=str(root), split="test", download=True)
    x = ds_test.imgs
    y = ds_test.labels.squeeze()
    return _NumpyClassificationDataset(x, y, transform=transform)
