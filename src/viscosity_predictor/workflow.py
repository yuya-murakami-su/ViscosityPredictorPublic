"""End-to-end model selection and final ensemble training."""

from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import atomic_write_csv, atomic_write_json, initialize_run_manifest
from .data import CANONICAL_SMILES, prepare_training_data
from .descriptors import calculate_descriptor_table
from .final_model import train_final_ensemble
from .regularization import joint_search_fit_count, rank_joint_results, run_joint_search
from .runtime import auto_device_name
from .splits import (
    INTEGRATED_SPLIT_STRATEGY,
    TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY,
    CrossValidationFold,
    create_cv_splits,
    validate_split_strategy,
)
from .training import (
    native_candidate_from_ranking_row,
    native_search_fit_count,
    rank_native_results,
    run_native_search,
)


MIN_RECOMMENDED_TRAINING_ROWS = 500
MIN_RECOMMENDED_TRAINING_COMPOUNDS = 50


def run_training_workflow(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    progress: bool = True,
    resume: bool = False,
    split_strategy: str = INTEGRATED_SPLIT_STRATEGY,
) -> dict[str, Any]:
    """Run native selection, Joint selection, and full-data ensemble training."""

    strategy = validate_split_strategy(split_strategy)
    if strategy != INTEGRATED_SPLIT_STRATEGY:
        config = deepcopy(config)
        config["workflow_options"] = {"validation_split_strategy": strategy}
    root = Path(project_root).resolve()
    training_csv = _resolve(root, config["paths"]["training_csv"])
    output_dir = _resolve(root, config["paths"]["output_dir"])
    device_name = auto_device_name()
    training = config["training"]
    batch_size = int(training["batch_size"])
    max_epochs = int(training["max_epochs"])
    patience = int(training["early_stopping_patience"])
    collocation_samples = int(training["collocation_samples"])
    fit_counts = {
        "native_search": native_search_fit_count(config),
        "joint_search": joint_search_fit_count(config),
        "final_ensemble": len(config["seeds"]["ensemble"]),
    }
    fit_counts["total"] = sum(fit_counts.values())
    strategy_message = (
        "" if strategy == INTEGRATED_SPLIT_STRATEGY else f"; split={strategy}"
    )
    print(
        f"device={device_name}{strategy_message}; planned model fits={fit_counts['total']} "
        f"(native={fit_counts['native_search']}, joint={fit_counts['joint_search']}, "
        f"final={fit_counts['final_ensemble']}); output={output_dir}",
        flush=True,
    )

    data = prepare_training_data(
        training_csv,
        cluster_tolerance_k=float(config["temperature"]["cluster_tolerance_K"]),
    )
    temperature_ranges = _validate_collocation_temperature_range(data, config)
    manifest = initialize_run_manifest(
        config,
        training_csv=training_csv,
        output_dir=output_dir,
        device_name=device_name,
        resume=resume,
    )
    _warn_if_small_training_data(data)
    if progress:
        print(
            f"effective dataset: rows={len(data)}; "
            f"compounds={data[CANONICAL_SMILES].nunique()}",
            flush=True,
        )
        print(
            "effective dataset temperature range: "
            f"{temperature_ranges['dataset_min_K']:.2f}-"
            f"{temperature_ranges['dataset_max_K']:.2f} K",
            flush=True,
        )
        print(
            "collocation temperature range: "
            f"{temperature_ranges['collocation_min_K']:.2f}-"
            f"{temperature_ranges['collocation_max_K']:.2f} K",
            flush=True,
        )
    descriptor_table = calculate_descriptor_table(
        data["canonical_smiles"],
        blocks=config["descriptors"]["candidates"],
    )
    splits = create_cv_splits(
        data,
        split_strategy=strategy,
        n_folds=int(config["cross_validation"]["n_folds"]),
        split_seed=int(config["cross_validation"]["split_seed"]),
        high_temperature_validation_quantile=float(
            config["temperature"]["high_temperature_validation_quantile"]
        ),
    )
    fold_summaries = [_fold_summary(data, split) for split in splits]
    if progress:
        for values in fold_summaries:
            if strategy == TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY:
                print(
                    f"fold={values['fold']}: train={values['train_rows']}; "
                    f"interpolation_valid={values['validation_rows']}",
                    flush=True,
                )
            else:
                print(
                    f"fold={values['fold']}: "
                    f"cutoff={values['temperature_cutoff_K']:.2f} K; "
                    f"train={values['train_rows']}; "
                    f"structure_valid={values['structure_validation_rows']}; "
                    f"temperature_valid={values['temperature_validation_rows']} "
                    f"({values['temperature_validation_fraction']:.1%}); "
                    f"temperature_compounds={values['temperature_validation_compounds']}",
                    flush=True,
                )

    native_results_path = output_dir / "native_search_results.csv"
    native_results = run_native_search(
        data,
        descriptor_table,
        splits,
        config,
        device_name=device_name,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        progress=progress,
        existing_results=_read_checkpoint(native_results_path) if resume else None,
        checkpoint_path=native_results_path,
    )
    atomic_write_csv(native_results, native_results_path)
    native_ranking = rank_native_results(
        native_results,
        expected_folds=range(len(splits)),
        expected_seeds=config["seeds"]["search"],
    )
    atomic_write_csv(native_ranking, output_dir / "native_search_ranking.csv")
    base_candidate = native_candidate_from_ranking_row(native_ranking.iloc[0])

    joint_results_path = output_dir / "joint_search_results.csv"
    joint_results = run_joint_search(
        data,
        descriptor_table,
        splits,
        config,
        base_candidate,
        device_name=device_name,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        n_samples=collocation_samples,
        progress=progress,
        existing_results=_read_checkpoint(joint_results_path) if resume else None,
        checkpoint_path=joint_results_path,
    )
    atomic_write_csv(joint_results, joint_results_path)
    joint_ranking = rank_joint_results(
        joint_results,
        expected_folds=range(len(splits)),
        expected_seeds=config["seeds"]["search"],
    )
    atomic_write_csv(joint_ranking, output_dir / "joint_search_ranking.csv")
    selected_joint = joint_ranking.iloc[0]
    metadata = train_final_ensemble(
        data,
        descriptor_table,
        config,
        base_candidate,
        selected_joint,
        output_dir=output_dir / "final_model",
        device_name=device_name,
        batch_size=batch_size,
        collocation_samples=collocation_samples,
        run_fingerprint=str(manifest["fingerprint"]),
        resume=resume,
        progress=progress,
    )

    summary = {
        "device": device_name,
        "run_fingerprint": manifest["fingerprint"],
        "fit_counts": fit_counts,
        "n_training_rows": len(data),
        "n_training_compounds": int(data["canonical_smiles"].nunique()),
        "temperature_ranges": temperature_ranges,
        "validation_folds": fold_summaries,
        "selected_native": _json_safe(native_ranking.iloc[0].to_dict()),
        "selected_joint": _json_safe(selected_joint.to_dict()),
        "final_model_metadata": str(output_dir / "final_model" / "metadata.json"),
        "ensemble_members": len(metadata["model_files"]),
    }
    atomic_write_json(summary, output_dir / "workflow_summary.json")
    return summary


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_checkpoint(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, float_precision="round_trip")
    except Exception as error:
        raise ValueError(f"Could not read training checkpoint: {path}") from error


def _warn_if_small_training_data(data: pd.DataFrame) -> None:
    n_rows = len(data)
    n_compounds = int(data[CANONICAL_SMILES].nunique())
    if (
        n_rows <= MIN_RECOMMENDED_TRAINING_ROWS
        or n_compounds <= MIN_RECOMMENDED_TRAINING_COMPOUNDS
    ):
        warnings.warn(
            f"The effective training dataset contains {n_rows} rows and "
            f"{n_compounds} unique compounds. Hyperparameter selection may be unstable "
            f"for datasets with {MIN_RECOMMENDED_TRAINING_ROWS} or fewer rows or "
            f"{MIN_RECOMMENDED_TRAINING_COMPOUNDS} or fewer compounds.",
            UserWarning,
            stacklevel=2,
        )


def _validate_collocation_temperature_range(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, float]:
    dataset_min = float(data["temperature_K"].min())
    dataset_max = float(data["temperature_K"].max())
    temperature = config["temperature"]
    collocation_min = float(temperature["collocation_min_K"])
    collocation_max = float(temperature["collocation_max_K"])
    if collocation_min > dataset_min or collocation_max < dataset_max:
        raise ValueError(
            f"The configured collocation range {collocation_min:.2f}-"
            f"{collocation_max:.2f} K does not cover the effective dataset range "
            f"{dataset_min:.2f}-{dataset_max:.2f} K. Set collocation_min_K to "
            f"{dataset_min:.2f} K or lower and collocation_max_K to "
            f"{dataset_max:.2f} K or higher."
        )
    return {
        "dataset_min_K": dataset_min,
        "dataset_max_K": dataset_max,
        "collocation_min_K": collocation_min,
        "collocation_max_K": collocation_max,
    }


def _fold_summary(data: pd.DataFrame, split: CrossValidationFold) -> dict[str, Any]:
    if split.split_strategy == TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY:
        return {
            "fold": int(split.fold),
            "train_rows": len(split.train_indices),
            "validation_rows": len(split.validation_indices),
        }
    n_train = len(split.train_indices)
    n_temperature = len(split.temperature_validation_indices)
    development_rows = n_train + n_temperature
    temperature_compounds = int(
        data.iloc[split.temperature_validation_indices][CANONICAL_SMILES].nunique()
    )
    return {
        "fold": int(split.fold),
        "temperature_cutoff_K": float(split.temperature_validation_cutoff_k),
        "train_rows": n_train,
        "structure_validation_rows": len(split.structure_validation_indices),
        "temperature_validation_rows": n_temperature,
        "temperature_validation_fraction": n_temperature / development_rows,
        "temperature_validation_compounds": temperature_compounds,
    }


def _json_safe(values: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in values.items():
        if isinstance(value, np.integer):
            output[key] = int(value)
        elif isinstance(value, np.floating):
            output[key] = float(value)
        elif isinstance(value, np.bool_):
            output[key] = bool(value)
        else:
            output[key] = value
    return output
