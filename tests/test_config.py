from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from viscosity_predictor.config import load_training_config, validate_training_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_training_config_is_valid() -> None:
    config = load_training_config(PROJECT_ROOT / "config" / "training.toml")

    assert config["cross_validation"]["n_folds"] == 5
    assert config["temperature"]["high_temperature_validation_quantile"] == 0.90
    assert config["temperature"]["collocation_min_K"] == 200.0
    assert config["temperature"]["collocation_max_K"] == 450.0
    assert len(config["descriptors"]["candidates"]) == 5
    assert config["collocation"]["hessian_descriptor_std_multiplier"] == 2.0
    assert config["collocation"]["soft_descriptor_std_multiplier"] == 1.0
    assert config["training"]["batch_size"] == 256
    assert config["training"]["max_epochs"] == 3000


def test_training_config_requires_at_least_two_folds() -> None:
    config = load_training_config(PROJECT_ROOT / "config" / "training.toml")
    invalid = deepcopy(config)
    invalid["cross_validation"]["n_folds"] = 1

    with pytest.raises(ValueError, match="at least 2"):
        validate_training_config(invalid)


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.1])
def test_training_config_rejects_invalid_temperature_quantile(quantile: float) -> None:
    config = load_training_config(PROJECT_ROOT / "config" / "training.toml")
    invalid = deepcopy(config)
    invalid["temperature"]["high_temperature_validation_quantile"] = quantile

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_training_config(invalid)
