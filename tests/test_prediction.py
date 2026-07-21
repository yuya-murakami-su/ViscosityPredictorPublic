from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from viscosity_predictor.config import load_training_config
from viscosity_predictor.data import CANONICAL_SMILES, INVERSE_TEMPERATURE, LN_VISCOSITY_PA_S
from viscosity_predictor.descriptors import calculate_descriptor_table
from viscosity_predictor.final_model import train_final_ensemble
from viscosity_predictor.prediction import predict_csv
from viscosity_predictor.training import NNHyperparameters, NativeCandidate

def test_saved_ensemble_can_be_reloaded_for_csv_prediction(tmp_path) -> None:
    training = pd.DataFrame(
        {
            CANONICAL_SMILES: ["C", "CC", "CCC", "CCCC", "O", "CO", "CCO", "CCCO"],
            INVERSE_TEMPERATURE: np.linspace(0.0040, 0.0020, 8),
            LN_VISCOSITY_PA_S: np.linspace(-2.0, 2.0, 8),
        }
    )
    descriptor_table = calculate_descriptor_table(
        training[CANONICAL_SMILES],
        blocks=["physicochemical_descriptors"],
    )
    config = deepcopy(load_training_config("config/training.toml"))
    config["seeds"]["ensemble"] = [1, 2]
    base = NativeCandidate(
        candidate_id="prediction_test",
        blocks=("physicochemical_descriptors",),
        hyperparameters=NNHyperparameters(1, 8, 1.0e-2, 0.0),
    )
    model_dir = tmp_path / "model"
    train_final_ensemble(
        training,
        descriptor_table,
        config,
        base,
        {"lambda_hessian": 0.0, "lambda_soft": 0.0, "fixed_epochs": 2},
        output_dir=model_dir,
        device_name="cpu",
        collocation_samples=8,
    )
    input_csv = tmp_path / "prediction_input.csv"
    pd.DataFrame(
        {
            "compound_id": ["methane", "ethanol"],
            "smiles": ["C", "CCO"],
            "temperature_K": [300.0, 350.0],
        }
    ).to_csv(input_csv, index=False)

    predictions = predict_csv(
        input_csv,
        model_dir,
        tmp_path / "predictions.csv",
        device_name="cpu",
    )

    member_columns = [
        "pred_ln_viscosity_Pa_s_seed_1",
        "pred_ln_viscosity_Pa_s_seed_2",
    ]
    assert np.isfinite(predictions[member_columns].to_numpy()).all()
    assert np.allclose(
        predictions["pred_ln_viscosity_Pa_s"],
        predictions[member_columns].mean(axis=1),
    )
    assert np.allclose(
        predictions["pred_viscosity_cP"],
        np.exp(predictions["pred_ln_viscosity_Pa_s"]) * 1.0e3,
    )
    assert (tmp_path / "predictions.csv").is_file()
