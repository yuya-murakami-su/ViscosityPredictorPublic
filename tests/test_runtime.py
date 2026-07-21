from __future__ import annotations

import random

import numpy as np
import torch

from viscosity_predictor.runtime import auto_device_name, initialize_seed


def test_auto_device_name_uses_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert auto_device_name() == "cuda"


def test_auto_device_name_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert auto_device_name() == "cpu"


def test_initialize_seed_reproduces_random_values() -> None:
    initialize_seed(17)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    initialize_seed(17)
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert first == second
