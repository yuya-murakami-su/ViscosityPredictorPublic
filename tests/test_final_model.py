from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from viscosity_predictor.config import load_training_config
from viscosity_predictor.data import CANONICAL_SMILES, INVERSE_TEMPERATURE, LN_VISCOSITY_PA_S
from viscosity_predictor.final_model import MODEL_FORMAT_VERSION, train_final_ensemble
from viscosity_predictor.training import NNHyperparameters, NativeCandidate


def test_final_ensemble_saves_each_seed_and_reproducibility_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    data = pd.DataFrame(
        {
            CANONICAL_SMILES: ["C", "CC", "CCC", "CCCC", "O", "CO", "CCO", "CCCO"],
            INVERSE_TEMPERATURE: np.linspace(0.0040, 0.0020, 8),
            LN_VISCOSITY_PA_S: np.linspace(-2.0, 2.0, 8),
        }
    )
    descriptors = pd.DataFrame(
        {
            CANONICAL_SMILES: data[CANONICAL_SMILES],
            "physicochemical_descriptors__test": np.linspace(-1.0, 1.0, 8),
        }
    )
    config = deepcopy(load_training_config("config/training.toml"))
    config["seeds"]["ensemble"] = [1, 2]
    base = NativeCandidate(
        candidate_id="test",
        blocks=("physicochemical_descriptors",),
        hyperparameters=NNHyperparameters(1, 8, 1.0e-2, 0.0),
    )

    metadata = train_final_ensemble(
        data,
        descriptors,
        config,
        base,
        {"lambda_hessian": 0.1, "lambda_soft": 0.01, "fixed_epochs": 2},
        output_dir=tmp_path,
        device_name="cpu",
        collocation_samples=8,
    )

    assert metadata["format_version"] == MODEL_FORMAT_VERSION
    assert metadata["fixed_epochs"] == 2
    assert metadata["feature_names"] == [
        "physicochemical_descriptors__test",
        INVERSE_TEMPERATURE,
    ]
    assert metadata["selected_block_labels"] == ["physicochemical descriptors"]
    assert "paths" not in metadata["workflow_configuration"]
    assert metadata["workflow_configuration"]["cross_validation"] == {
        "n_folds": 5,
        "split_seed": 0,
    }
    assert (
        metadata["workflow_configuration"]["temperature"]
        ["high_temperature_validation_quantile"]
        == 0.90
    )
    assert metadata["workflow_configuration"]["regularization_search"] == {
        "hessian_lambdas": [0.0, 0.1, 1.0, 10.0, 100.0],
        "soft_lambdas": [0.0, 0.001, 0.01, 0.1, 1.0],
    }
    assert len(metadata["model_files"]) == 2
    assert (tmp_path / "metadata.json").is_file()
    for record in metadata["model_files"]:
        bundle = torch.load(tmp_path / record["path"], weights_only=True)
        assert bundle["format_version"] == MODEL_FORMAT_VERSION
        assert bundle["training_fingerprint"] == metadata["training_fingerprint"]
        assert bundle["model_state_dict"]

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("A completed ensemble member was trained again.")

    monkeypatch.setattr("viscosity_predictor.final_model.train_fixed_candidate", unexpected_fit)
    resumed = train_final_ensemble(
        data,
        descriptors,
        config,
        base,
        {"lambda_hessian": 0.1, "lambda_soft": 0.01, "fixed_epochs": 2},
        output_dir=tmp_path,
        device_name="cpu",
        collocation_samples=8,
        resume=True,
        progress=False,
    )
    assert resumed["training_fingerprint"] == metadata["training_fingerprint"]
