from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from viscosity_predictor.config import load_training_config
from viscosity_predictor.data import CANONICAL_SMILES, canonicalize_smiles
from viscosity_predictor.prediction import predict_ensemble
from viscosity_predictor.splits import (
    TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY,
    assign_similarity_graph_folds,
)
from viscosity_predictor.workflow import (
    _validate_collocation_temperature_range,
    _warn_if_small_training_data,
    run_training_workflow,
)


def test_end_to_end_training_and_prediction_workflow(tmp_path, monkeypatch) -> None:
    training_csv = _small_training_csv(tmp_path)
    config = deepcopy(load_training_config("config/training.toml"))
    config["paths"] = {
        "training_csv": str(training_csv),
        "output_dir": str(tmp_path / "output"),
    }
    config["cross_validation"]["n_folds"] = 2
    config["descriptors"]["candidates"] = ["physicochemical_descriptors"]
    config["nn_search"] = {
        "hidden_layers": [1],
        "hidden_units": [8],
        "learning_rates": [1.0e-2],
        "weight_decays": [0.0],
    }
    config["regularization_search"] = {
        "hessian_lambdas": [0.0, 0.1],
        "soft_lambdas": [0.0, 0.01],
    }
    config["seeds"] = {"search": [1], "ensemble": [1]}
    config["training"] = {
        "batch_size": 8,
        "max_epochs": 2,
        "early_stopping_patience": 1,
        "collocation_samples": 4,
    }

    with pytest.warns(UserWarning, match="500 or fewer rows"):
        summary = run_training_workflow(
            config,
            project_root=tmp_path,
            progress=False,
        )

    assert summary["fit_counts"] == {
        "native_search": 2,
        "joint_search": 8,
        "final_ensemble": 1,
        "total": 11,
    }
    assert len(summary["validation_folds"]) == 2
    assert summary["temperature_ranges"] == {
        "dataset_min_K": 300.0,
        "dataset_max_K": 400.0,
        "collocation_min_K": 200.0,
        "collocation_max_K": 450.0,
    }
    assert all(
        values["temperature_validation_fraction"] >= 0.10
        for values in summary["validation_folds"]
    )
    output = tmp_path / "output"
    assert (output / "native_search_ranking.csv").is_file()
    assert (output / "joint_search_ranking.csv").is_file()
    assert (output / "run_manifest.json").is_file()
    assert (output / "workflow_summary.json").is_file()
    native_results = pd.read_csv(output / "native_search_results.csv")
    assert "temperature_validation_cutoff_K" in native_results.columns
    prediction = predict_ensemble(
        pd.DataFrame(
            {
                "compound_id": ["query"],
                "smiles": ["CCO"],
                "temperature_K": [350.0],
            }
        ),
        output / "final_model",
        device_name="cpu",
    )
    assert prediction["pred_viscosity_cP"].notna().all()

    with pytest.raises(FileExistsError, match="--resume"):
        run_training_workflow(config, project_root=tmp_path, progress=False)

    changed = deepcopy(config)
    changed["training"]["max_epochs"] += 1
    with pytest.raises(ValueError, match="conditions differ"):
        run_training_workflow(
            changed,
            project_root=tmp_path,
            progress=False,
            resume=True,
        )

    with pytest.raises(ValueError, match="conditions differ"):
        run_training_workflow(
            config,
            project_root=tmp_path,
            progress=False,
            resume=True,
            split_strategy=TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY,
        )

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("A completed fit was trained again during resume.")

    monkeypatch.setattr("viscosity_predictor.training.train_native_candidate", unexpected_fit)
    monkeypatch.setattr("viscosity_predictor.regularization.train_candidate", unexpected_fit)
    monkeypatch.setattr("viscosity_predictor.final_model.train_fixed_candidate", unexpected_fit)
    with pytest.warns(UserWarning, match="500 or fewer rows"):
        resumed = run_training_workflow(
            config,
            project_root=tmp_path,
            progress=False,
            resume=True,
        )
    assert resumed["run_fingerprint"] == summary["run_fingerprint"]


def test_interpolation_training_uses_point_mae_and_saves_model(
    tmp_path,
) -> None:
    training_csv = _small_interpolation_training_csv(tmp_path)
    config = deepcopy(load_training_config("config/training.toml"))
    config["paths"] = {
        "training_csv": str(training_csv),
        "output_dir": str(tmp_path / "interpolation_output"),
    }
    config["cross_validation"]["n_folds"] = 2
    config["descriptors"]["candidates"] = ["physicochemical_descriptors"]
    config["nn_search"] = {
        "hidden_layers": [1],
        "hidden_units": [8],
        "learning_rates": [1.0e-2],
        "weight_decays": [0.0],
    }
    config["regularization_search"] = {
        "hessian_lambdas": [0.0],
        "soft_lambdas": [0.0],
    }
    config["seeds"] = {"search": [1], "ensemble": [1]}
    config["training"] = {
        "batch_size": 8,
        "max_epochs": 1,
        "early_stopping_patience": 0,
        "collocation_samples": 4,
    }

    with pytest.warns(UserWarning, match="500 or fewer rows"):
        summary = run_training_workflow(
            config,
            project_root=tmp_path,
            progress=False,
            split_strategy=TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY,
        )

    assert summary["fit_counts"]["total"] == 5
    assert [fold["validation_rows"] for fold in summary["validation_folds"]] == [2, 2]
    native = pd.read_csv(
        tmp_path / "interpolation_output" / "native_search_results.csv"
    )
    assert native["integrated_score"].notna().all()
    assert native["structure_mae"].isna().all()
    assert native["temperature_mae"].isna().all()
    metadata_path = Path(summary["final_model_metadata"])
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["workflow_configuration"]["workflow_options"] == {
        "validation_split_strategy": TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY
    }


@pytest.mark.parametrize("n_rows,n_compounds", [(500, 100), (600, 50)])
def test_small_effective_dataset_warning_uses_rows_or_compounds(
    n_rows: int,
    n_compounds: int,
) -> None:
    data = pd.DataFrame(
        {CANONICAL_SMILES: [f"compound_{index % n_compounds}" for index in range(n_rows)]}
    )

    with pytest.warns(UserWarning, match="Hyperparameter selection may be unstable"):
        _warn_if_small_training_data(data)


def test_large_effective_dataset_does_not_warn() -> None:
    data = pd.DataFrame(
        {CANONICAL_SMILES: [f"compound_{index % 51}" for index in range(501)]}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_small_training_data(data)


@pytest.mark.parametrize(
    "collocation_min,collocation_max",
    [(251.0, 450.0), (200.0, 399.0)],
)
def test_collocation_range_must_cover_effective_dataset(
    collocation_min: float,
    collocation_max: float,
) -> None:
    config = deepcopy(load_training_config("config/training.toml"))
    config["temperature"]["collocation_min_K"] = collocation_min
    config["temperature"]["collocation_max_K"] = collocation_max
    data = pd.DataFrame({"temperature_K": [250.0, 400.0]})

    with pytest.raises(ValueError, match="does not cover the effective dataset range"):
        _validate_collocation_temperature_range(data, config)


def test_collocation_range_accepts_exact_dataset_boundaries() -> None:
    config = deepcopy(load_training_config("config/training.toml"))
    config["temperature"]["collocation_min_K"] = 250.0
    config["temperature"]["collocation_max_K"] = 400.0
    data = pd.DataFrame({"temperature_K": [250.0, 400.0]})

    assert _validate_collocation_temperature_range(data, config) == {
        "dataset_min_K": 250.0,
        "dataset_max_K": 400.0,
        "collocation_min_K": 250.0,
        "collocation_max_K": 400.0,
    }


def test_workflow_rejects_narrow_collocation_range_before_writing_artifacts(
    tmp_path,
) -> None:
    training_csv = _small_training_csv(tmp_path)
    output_dir = tmp_path / "invalid_output"
    config = deepcopy(load_training_config("config/training.toml"))
    config["paths"] = {
        "training_csv": str(training_csv),
        "output_dir": str(output_dir),
    }
    config["temperature"]["collocation_max_K"] = 350.0

    with pytest.raises(ValueError, match="does not cover the effective dataset range"):
        run_training_workflow(config, project_root=tmp_path, progress=False)

    assert not output_dir.exists()


def test_train_and_predict_scripts_run_end_to_end(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    training_csv = _small_training_csv(tmp_path)
    output_dir = tmp_path / "script_output"
    config_path = tmp_path / "training.toml"
    config_path.write_text(
        f'''[paths]
training_csv = "{training_csv.as_posix()}"
output_dir = "{output_dir.as_posix()}"

[cross_validation]
n_folds = 2
split_seed = 0

[temperature]
cluster_tolerance_K = 0.03
high_temperature_validation_quantile = 0.90
collocation_min_K = 250.0
collocation_max_K = 450.0

[collocation]
hessian_descriptor_std_multiplier = 2.0
soft_descriptor_std_multiplier = 1.0

[training]
batch_size = 8
max_epochs = 1
early_stopping_patience = 0
collocation_samples = 4

[descriptors]
candidates = ["physicochemical_descriptors"]

[nn_search]
hidden_layers = [1]
hidden_units = [8]
learning_rates = [1.0e-2]
weight_decays = [0.0]

[regularization_search]
hessian_lambdas = [0.0]
soft_lambdas = [0.0]

[seeds]
search = [1]
ensemble = [1]
''',
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    subprocess.run(
        [sys.executable, str(project_root / "scripts" / "train.py"), "--config", str(config_path)],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "train.py"),
            "--config",
            str(config_path),
            "--resume",
        ],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    prediction_input = tmp_path / "prediction.csv"
    prediction_output = tmp_path / "prediction_output.csv"
    pd.DataFrame(
        {"compound_id": ["query"], "smiles": ["CCO"], "temperature_K": [350.0]}
    ).to_csv(prediction_input, index=False)
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "predict.py"),
            "--input",
            str(prediction_input),
            "--model",
            str(output_dir / "final_model"),
            "--output",
            str(prediction_output),
        ],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert pd.read_csv(prediction_output)["pred_viscosity_cP"].notna().all()


def _small_training_csv(tmp_path: Path) -> Path:
    smiles = ["C", "CC", "CCC", "CCCC", "O", "CO", "CCO", "CCCO"]
    structures = pd.DataFrame({CANONICAL_SMILES: [canonicalize_smiles(value) for value in smiles]})
    folds = assign_similarity_graph_folds(structures, n_folds=2, split_seed=0)
    temperatures = [320.0] * len(smiles)
    for fold in (0, 1):
        positions = folds.index[folds.eq(fold)].tolist()
        temperatures[positions[0]] = 400.0
        temperatures[positions[1]] = 300.0
    training_csv = tmp_path / "training.csv"
    pd.DataFrame(
        {
            "compound_id": [f"compound_{index}" for index in range(len(smiles))],
            "smiles": smiles,
            "temperature_K": temperatures,
            "viscosity_cP": [float(index + 1) for index in range(len(smiles))],
        }
    ).to_csv(training_csv, index=False)
    return training_csv


def _small_interpolation_training_csv(tmp_path: Path) -> Path:
    smiles = ["C", "CC", "CCC", "O"]
    rows = []
    for compound_index, smiles_value in enumerate(smiles):
        for temperature in (300.0, 350.0, 400.0):
            rows.append(
                {
                    "compound_id": f"compound_{compound_index}",
                    "smiles": smiles_value,
                    "temperature_K": temperature,
                    "viscosity_cP": float(compound_index + 1) * 300.0 / temperature,
                }
            )
    training_csv = tmp_path / "interpolation_training.csv"
    pd.DataFrame(rows).to_csv(training_csv, index=False)
    return training_csv
