"""Runtime helpers shared by training and prediction scripts."""

from __future__ import annotations

import torch

from murakami_lab_modules.utils import initialize_random_seed


def auto_device_name() -> str:
    """Use CUDA when available and otherwise use the CPU."""

    return "cuda" if torch.cuda.is_available() else "cpu"


def initialize_seed(seed: int) -> None:
    """Initialize Python, NumPy, and PyTorch random seeds."""

    initialize_random_seed(int(seed))
