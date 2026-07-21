"""Load the public training workflow configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = (
    "paths",
    "cross_validation",
    "temperature",
    "collocation",
    "training",
    "descriptors",
    "nn_search",
    "regularization_search",
    "seeds",
)


def load_training_config(path: str | Path = "config/training.toml") -> dict[str, Any]:
    """Read and validate a training TOML file."""

    with Path(path).open("rb") as file:
        config = tomllib.load(file)
    validate_training_config(config)
    return config


def validate_training_config(config: dict[str, Any]) -> None:
    """Validate the small set of values required by the public workflow."""

    missing = [name for name in REQUIRED_SECTIONS if name not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")

    paths = config["paths"]
    if not str(paths.get("training_csv", "")).strip():
        raise ValueError("paths.training_csv must not be empty.")
    if not str(paths.get("output_dir", "")).strip():
        raise ValueError("paths.output_dir must not be empty.")

    cross_validation = config["cross_validation"]
    if int(cross_validation["n_folds"]) < 2:
        raise ValueError("cross_validation.n_folds must be at least 2.")
    int(cross_validation["split_seed"])

    temperature = config["temperature"]
    if float(temperature["cluster_tolerance_K"]) < 0.0:
        raise ValueError("temperature.cluster_tolerance_K must be non-negative.")
    validation_quantile = float(temperature["high_temperature_validation_quantile"])
    if not 0.0 < validation_quantile < 1.0:
        raise ValueError(
            "temperature.high_temperature_validation_quantile must be between 0 and 1."
        )
    collocation_min = float(temperature["collocation_min_K"])
    collocation_max = float(temperature["collocation_max_K"])
    if collocation_min <= 0.0 or collocation_max <= collocation_min:
        raise ValueError("The collocation temperature range is invalid.")

    collocation = config["collocation"]
    _require_positive(
        collocation["hessian_descriptor_std_multiplier"],
        "collocation.hessian_descriptor_std_multiplier",
    )
    _require_positive(
        collocation["soft_descriptor_std_multiplier"],
        "collocation.soft_descriptor_std_multiplier",
    )

    training = config["training"]
    for name in ("batch_size", "max_epochs", "collocation_samples"):
        value = training[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"training.{name} must be a positive integer.")
    patience = training["early_stopping_patience"]
    if not isinstance(patience, int) or isinstance(patience, bool) or patience < 0:
        raise ValueError("training.early_stopping_patience must be a non-negative integer.")

    descriptor_candidates = config["descriptors"]["candidates"]
    _require_unique_nonempty_list(descriptor_candidates, "descriptors.candidates")

    nn_search = config["nn_search"]
    _require_positive_integer_list(nn_search["hidden_layers"], "nn_search.hidden_layers")
    _require_positive_integer_list(nn_search["hidden_units"], "nn_search.hidden_units")
    _require_positive_list(nn_search["learning_rates"], "nn_search.learning_rates")
    _require_nonnegative_list(nn_search["weight_decays"], "nn_search.weight_decays")

    regularization = config["regularization_search"]
    _require_nonnegative_list(
        regularization["hessian_lambdas"],
        "regularization_search.hessian_lambdas",
    )
    _require_nonnegative_list(
        regularization["soft_lambdas"],
        "regularization_search.soft_lambdas",
    )

    seeds = config["seeds"]
    _require_unique_integer_list(seeds["search"], "seeds.search")
    _require_unique_integer_list(seeds["ensemble"], "seeds.ensemble")


def _require_positive(value: Any, name: str) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _require_unique_nonempty_list(values: Any, name: str) -> None:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates.")


def _require_positive_integer_list(values: Any, name: str) -> None:
    _require_unique_nonempty_list(values, name)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers.")


def _require_unique_integer_list(values: Any, name: str) -> None:
    _require_unique_nonempty_list(values, name)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError(f"{name} must contain integers.")


def _require_positive_list(values: Any, name: str) -> None:
    _require_unique_nonempty_list(values, name)
    if any(float(value) <= 0.0 for value in values):
        raise ValueError(f"{name} must contain positive values.")


def _require_nonnegative_list(values: Any, name: str) -> None:
    _require_unique_nonempty_list(values, name)
    if any(float(value) < 0.0 for value in values):
        raise ValueError(f"{name} must contain non-negative values.")
