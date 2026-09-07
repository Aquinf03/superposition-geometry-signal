"""Deterministic seeding for all experiment entrypoints."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


DEFAULT_SEED = 0


def set_seed(seed: int = DEFAULT_SEED, deterministic_cudnn: bool = True) -> int:
    """Set Python, NumPy, and PyTorch seeds. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn and torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def seed_from_config(cfg: Optional[dict] = None) -> int:
    """Read ``seed`` from a config dict (default 0) and apply it."""
    seed = DEFAULT_SEED if cfg is None else int(cfg.get("seed", DEFAULT_SEED))
    return set_seed(seed)
