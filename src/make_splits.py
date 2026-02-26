from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import Config
from src.data import save_splits

_DEFAULTS = Config()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate frozen train/val/test splits for BloodMNIST.")
    p.add_argument("--data-dir", type=Path, default=_DEFAULTS.data_dir)
    p.add_argument("--out", type=Path, default=_DEFAULTS.splits_path)
    p.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--force", action="store_true", help="Overwrite existing split file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} already exists. Refusing to overwrite (use --force).")

    args.data_dir.mkdir(parents=True, exist_ok=True)

    import medmnist

    ds_train = medmnist.BloodMNIST(root=str(args.data_dir), split="train", download=True)
    ds_val = medmnist.BloodMNIST(root=str(args.data_dir), split="val", download=True)
    ds_test = medmnist.BloodMNIST(root=str(args.data_dir), split="test", download=True)

    y_train = ds_train.labels.squeeze()
    y_val = ds_val.labels.squeeze()
    y_pool = np.concatenate([y_train, y_val], axis=0)

    pool_indices = np.arange(len(y_pool))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(pool_indices, y_pool))

    train_idx = sorted(map(int, train_idx))
    val_idx = sorted(map(int, val_idx))
    test_idx = list(range(len(ds_test)))

    payload = {
        "dataset": "BloodMNIST",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "pool": {
            "train_len": int(len(ds_train)),
            "val_len": int(len(ds_val)),
            "test_len": int(len(ds_test)),
            "trainval_len": int(len(y_pool)),
        },
        "train": {"source": "train+val", "indices": train_idx},
        "val": {"source": "train+val", "indices": val_idx},
        "test": {"source": "test", "indices": test_idx},
    }

    save_splits(args.out, payload)
    print(f"Wrote frozen splits to: {args.out}")


if __name__ == "__main__":
    main()
