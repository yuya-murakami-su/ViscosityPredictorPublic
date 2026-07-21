"""Train and save the selected full-data ensemble."""

from __future__ import annotations

import importlib.metadata
import platform
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .artifacts import atomic_write_json, json_fingerprint
from .data import CANONICAL_SMILES
from .descriptors import descriptor_block_label
from .preprocessing import (
    CORRELATION_THRESHOLD,
    SPREAD_QUANTILE_HIGH,
    SPREAD_QUANTILE_LOW,
    VARIANCE_THRESHOLD,
    prepare_full_data,
)
from .regularization import COLLOCATION_BATCH_SIZE, build_curvature_regularization
from .training import NN_BATCH_SIZE, NativeCandidate, train_fixed_candidate


MODEL_FORMAT_VERSION = "viscosity_joint_nn_ensemble_v1"


def train_final_ensemble(
    data: pd.DataFrame,
    descriptor_table: pd.DataFrame,
    config: dict[str, Any],
    base_candidate: NativeCandidate,
    selected_joint: Mapping[str, Any],
    *,
    output_dir: str | Path,
    device_name: str,
    batch_size: int = NN_BATCH_SIZE,
    collocation_samples: int = COLLOCATION_BATCH_SIZE,
    run_fingerprint: str = "standalone",
    resume: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Train every ensemble seed on all rows and save model states plus metadata."""

    destination = Path(output_dir)
    model_dir = destination / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    fixed_epochs = int(selected_joint["fixed_epochs"])
    lambda_hessian = float(selected_joint["lambda_hessian"])
    lambda_soft = float(selected_joint["lambda_soft"])
    if fixed_epochs <= 0:
        raise ValueError("fixed_epochs must be positive.")
    training_fingerprint = json_fingerprint(
        {
            "run_fingerprint": run_fingerprint,
            "base_candidate_id": base_candidate.candidate_id,
            "blocks": list(base_candidate.blocks),
            "network": {
                "hidden_layers": base_candidate.hyperparameters.hidden_layers,
                "hidden_units": base_candidate.hyperparameters.hidden_units,
                "learning_rate": base_candidate.hyperparameters.learning_rate,
                "weight_decay": base_candidate.hyperparameters.weight_decay,
            },
            "lambda_hessian": lambda_hessian,
            "lambda_soft": lambda_soft,
            "fixed_epochs": fixed_epochs,
        }
    )

    model_records = []
    reference_prepared = None
    seeds = list(map(int, config["seeds"]["ensemble"]))
    for completed, seed in enumerate(seeds, start=1):
        prepared = prepare_full_data(
            data,
            descriptor_table,
            blocks=base_candidate.blocks,
            batch_size=int(batch_size),
            device_name=device_name,
            random_seed=seed,
        )
        if reference_prepared is None:
            reference_prepared = prepared
        elif prepared.selection != reference_prepared.selection:
            raise RuntimeError("Full-data descriptor selection changed between ensemble seeds.")
        filename = f"seed_{seed}.pt"
        model_path = model_dir / filename
        if resume and model_path.is_file():
            _validate_saved_model(model_path, seed, training_fingerprint)
            model_records.append({"seed": seed, "path": f"models/{filename}"})
            if progress:
                print(
                    f"final ensemble {completed}/{len(seeds)}: reused seed={seed}",
                    flush=True,
                )
            continue
        regularization = build_curvature_regularization(
            prepared.data_handler,
            config,
            lambda_hessian=lambda_hessian,
            lambda_soft=lambda_soft,
            seed=seed,
            n_samples=collocation_samples,
        )
        handler = train_fixed_candidate(
            prepared,
            base_candidate.hyperparameters,
            seed=seed,
            regularization=regularization,
            fixed_epochs=fixed_epochs,
        )
        state = {
            key: value.detach().cpu()
            for key, value in handler.nn.state_dict().items()
        }
        temporary = model_path.with_name(f"{model_path.name}.tmp")
        torch.save(
            {
                "format_version": MODEL_FORMAT_VERSION,
                "seed": seed,
                "training_fingerprint": training_fingerprint,
                "model_state_dict": state,
            },
            temporary,
        )
        temporary.replace(model_path)
        model_records.append({"seed": seed, "path": f"models/{filename}"})
        if progress:
            print(
                f"final ensemble {completed}/{len(seeds)}: trained seed={seed}",
                flush=True,
            )

    if reference_prepared is None:
        raise ValueError("At least one ensemble seed is required.")
    handler_data = reference_prepared.data_handler
    metadata = {
        "format_version": MODEL_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_fingerprint": run_fingerprint,
        "training_fingerprint": training_fingerprint,
        "n_training_rows": int(len(data)),
        "n_training_compounds": int(data[CANONICAL_SMILES].nunique()),
        "input_schema": ["compound_id", "smiles", "temperature_K"],
        "training_target": "ln(viscosity_Pa_s)",
        "canonicalization": "RDKit canonical isomeric SMILES",
        "workflow_configuration": {
            section: values
            for section, values in config.items()
            if section != "paths"
        },
        "base_candidate_id": base_candidate.candidate_id,
        "selected_blocks": list(base_candidate.blocks),
        "selected_block_labels": [
            descriptor_block_label(block)
            for block in base_candidate.blocks
        ],
        "descriptor_filter": {
            "input_columns": list(reference_prepared.selection.input_columns),
            "selected_columns": list(reference_prepared.selection.selected_columns),
            "low_variance_columns": list(reference_prepared.selection.low_variance_columns),
            "high_correlation_columns": list(reference_prepared.selection.high_correlation_columns),
            "variance_threshold": VARIANCE_THRESHOLD,
            "correlation_threshold": CORRELATION_THRESHOLD,
            "spread_quantile_low": SPREAD_QUANTILE_LOW,
            "spread_quantile_high": SPREAD_QUANTILE_HIGH,
        },
        "feature_names": list(reference_prepared.feature_names),
        "input_normalizer": _normalizer_metadata(handler_data.input_normalizer),
        "output_normalizer": _normalizer_metadata(handler_data.output_normalizer),
        "network": {
            "input_dim": len(reference_prepared.feature_names),
            "output_dim": 1,
            "hidden_layers": base_candidate.hyperparameters.hidden_layers,
            "hidden_units": base_candidate.hyperparameters.hidden_units,
            "activation": "tanh",
            "learning_rate": base_candidate.hyperparameters.learning_rate,
            "weight_decay": base_candidate.hyperparameters.weight_decay,
        },
        "regularization": {
            "lambda_hessian": lambda_hessian,
            "lambda_soft": lambda_soft,
            "collocation_samples": int(collocation_samples),
            "temperature_min_K": float(config["temperature"]["collocation_min_K"]),
            "temperature_max_K": float(config["temperature"]["collocation_max_K"]),
            "hessian_descriptor_std_multiplier": float(
                config["collocation"]["hessian_descriptor_std_multiplier"]
            ),
            "soft_descriptor_std_multiplier": float(
                config["collocation"]["soft_descriptor_std_multiplier"]
            ),
        },
        "fixed_epochs": fixed_epochs,
        "batch_size": int(batch_size),
        "model_files": model_records,
        "dependencies": _dependency_versions(),
    }
    atomic_write_json(metadata, destination / "metadata.json")
    return metadata


def _validate_saved_model(path: Path, seed: int, training_fingerprint: str) -> None:
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Could not read saved ensemble model: {path}") from error
    if (
        bundle.get("format_version") != MODEL_FORMAT_VERSION
        or int(bundle.get("seed", -1)) != seed
        or bundle.get("training_fingerprint") != training_fingerprint
        or not bundle.get("model_state_dict")
    ):
        raise ValueError(f"Saved ensemble model does not match the resumed run: {path}")


def _normalizer_metadata(normalizer) -> dict[str, Any]:
    return {
        "class": normalizer.__class__.__name__,
        "epsilon": float(normalizer.epsilon),
        "mean": normalizer.mean.detach().cpu().tolist(),
        "std": normalizer.std.detach().cpu().tolist(),
    }


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in (
        "numpy",
        "pandas",
        "torch",
        "murakami-lab-modules",
        "rdkit",
        "matminer",
        "pymetis",
    ):
        versions[distribution] = importlib.metadata.version(distribution)
    return versions
