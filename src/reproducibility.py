from __future__ import annotations

import os
import random
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int
    deterministic: bool = True


def configure_reproducibility(cfg: ReproducibilityConfig) -> None:
    """Best-effort reproducibility setup.

    Notes:
    - Environment variables like CUBLAS_WORKSPACE_CONFIG must be set before CUDA context
      is initialized to take effect.
    - Deterministic algorithms may reduce performance and can error if an op has no
      deterministic implementation.
    """
    
    if cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)


def seed_torch_everything(seed: int, deterministic: bool) -> None:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def _dataloader_worker_init(worker_id: int, *, base_seed: int) -> None:
    """Seed each DataLoader worker deterministically.

    Must be a top-level function so it can be pickled when using the
    multiprocessing "spawn" start method (default on macOS).
    """

    worker_seed = (int(base_seed) + int(worker_id)) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    try:
        import torch

        torch.manual_seed(worker_seed)
    except Exception:
        # Torch may not be available in some minimal contexts.
        pass


def dataloader_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    return partial(_dataloader_worker_init, base_seed=base_seed)
