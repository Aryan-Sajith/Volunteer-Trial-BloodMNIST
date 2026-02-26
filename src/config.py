from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 1337
    deterministic: bool = True

    # Data
    data_dir: Path = Path("data")
    splits_path: Path = Path("dataset_splits.json")
    image_size: int = 224

    # Outputs (checkpoints, temporary artifacts)
    output_dir: Path = Path("outputs")

    # Training
    model: str = "resnet18"
    pretrained: bool = True
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # Logging
    wandb_project: str = "Volunteer-Trial-BloodMNIST"
    wandb_entity: str | None = None
    wandb_mode: str = "online"  # online|offline|disabled

    def asdict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}
        d["data_dir"] = str(d["data_dir"])
        d["splits_path"] = str(d["splits_path"])
        d["output_dir"] = str(d["output_dir"])
        return d


def with_overrides(cfg: Config, **overrides: Any) -> Config:
    normalized: dict[str, Any] = dict(overrides)
    if "data_dir" in normalized and normalized["data_dir"] is not None:
        normalized["data_dir"] = Path(normalized["data_dir"])
    if "splits_path" in normalized and normalized["splits_path"] is not None:
        normalized["splits_path"] = Path(normalized["splits_path"])
    if "output_dir" in normalized and normalized["output_dir"] is not None:
        normalized["output_dir"] = Path(normalized["output_dir"])
    return replace(cfg, **{k: v for k, v in normalized.items() if v is not None})
