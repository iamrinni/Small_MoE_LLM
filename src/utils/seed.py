"""Determinism helpers — central to the project's reproducibility requirement.

Call :func:`set_seed` once at the start of every entry point (train / eval / data prep)
so that results are replicable from a clean environment.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs and (optionally) enable deterministic kernels.

    Args:
        seed: Global seed value.
        deterministic: If True, request deterministic cuDNN/cuBLAS algorithms. This can
            slow training slightly but makes runs bit-for-bit comparable across machines.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # torch optional until installed
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS workspace config required for deterministic matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    """`worker_init_fn` for DataLoader so each worker is seeded reproducibly."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
