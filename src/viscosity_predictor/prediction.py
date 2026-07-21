"""Load a saved ensemble and predict pure-component viscosity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import CANONICAL_SMILES, canonicalize_smiles
from .descriptors import calculate_descriptor_table
from .final_model import MODEL_FORMAT_VERSION
from .training import NNHyperparameters, create_native_nn


REQUIRED_PREDICTION_COLUMNS = ("compound_id", "smiles", "temperature_K")


def load_prediction_csv(path: str | Path) -> pd.DataFrame:
    """Read and minimally validate a prediction input CSV."""

    data = pd.read_csv(path, low_memory=False)
    data.columns = [str(column).strip() for column in data.columns]
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")
    data = data.loc[:, list(REQUIRED_PREDICTION_COLUMNS)].copy()
    for column in ("compound_id", "smiles"):
        values = data[column].astype("string").str.strip()
        if (values.isna() | (values == "")).any():
            raise ValueError(f"Prediction column {column!r} contains blank values.")
        data[column] = values.astype(str)
    temperature = pd.to_numeric(data["temperature_K"], errors="coerce")
    if not np.isfinite(temperature).all() or (temperature <= 0.0).any():
        raise ValueError("Prediction temperatures must be finite and positive.")
    data["temperature_K"] = temperature.astype(float)
    return data


def predict_ensemble(
    data: pd.DataFrame,
    model_dir: str | Path,
    *,
    device_name: str,
) -> pd.DataFrame:
    """Return member predictions and the arithmetic ensemble mean in log space."""

    root = Path(model_dir)
    with (root / "metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported model artifact format.")

    output = data.loc[:, list(REQUIRED_PREDICTION_COLUMNS)].copy()
    canonical_by_smiles = {
        smiles: canonicalize_smiles(smiles)
        for smiles in output["smiles"].drop_duplicates()
    }
    output[CANONICAL_SMILES] = output["smiles"].map(canonical_by_smiles)
    descriptors = calculate_descriptor_table(
        output[CANONICAL_SMILES],
        blocks=metadata["selected_blocks"],
    ).set_index(CANONICAL_SMILES, verify_integrity=True)
    selected_columns = metadata["descriptor_filter"]["selected_columns"]
    descriptor_matrix = descriptors.loc[output[CANONICAL_SMILES], selected_columns].to_numpy(dtype=float)
    inverse_temperature = (1.0 / output["temperature_K"].to_numpy(dtype=float)).reshape(-1, 1)
    raw_inputs = np.hstack([descriptor_matrix, inverse_temperature]).astype(np.float32)

    input_mean = np.asarray(metadata["input_normalizer"]["mean"], dtype=np.float32)
    input_std = np.asarray(metadata["input_normalizer"]["std"], dtype=np.float32)
    normalized_inputs = (raw_inputs - input_mean) / input_std
    output_mean = float(metadata["output_normalizer"]["mean"][0][0])
    output_std = float(metadata["output_normalizer"]["std"][0][0])
    hp = NNHyperparameters(
        hidden_layers=int(metadata["network"]["hidden_layers"]),
        hidden_units=int(metadata["network"]["hidden_units"]),
        learning_rate=float(metadata["network"]["learning_rate"]),
        weight_decay=float(metadata["network"]["weight_decay"]),
    )

    member_predictions = []
    tensor_inputs = torch.as_tensor(normalized_inputs, dtype=torch.float32, device=device_name)
    for record in metadata["model_files"]:
        seed = int(record["seed"])
        bundle = torch.load(root / record["path"], map_location=device_name, weights_only=True)
        if bundle.get("format_version") != MODEL_FORMAT_VERSION or int(bundle["seed"]) != seed:
            raise ValueError(f"Model bundle metadata mismatch: {record['path']}")
        network = create_native_nn(raw_inputs.shape[1], hp, seed).to(device_name)
        network.load_state_dict(bundle["model_state_dict"])
        network.eval()
        with torch.no_grad():
            normalized_prediction = network(tensor_inputs).detach().cpu().numpy().reshape(-1)
        prediction = normalized_prediction * output_std + output_mean
        member_predictions.append(prediction.astype(float))
        output[f"pred_ln_viscosity_Pa_s_seed_{seed}"] = prediction
        output[f"pred_viscosity_cP_seed_{seed}"] = np.exp(prediction) * 1.0e3

    members = np.column_stack(member_predictions)
    output["pred_ln_viscosity_Pa_s"] = members.mean(axis=1)
    output["pred_ln_viscosity_Pa_s_seed_std"] = (
        members.std(axis=1, ddof=1) if members.shape[1] > 1 else np.nan
    )
    output["n_ensemble_members"] = members.shape[1]
    output["pred_viscosity_Pa_s"] = np.exp(output["pred_ln_viscosity_Pa_s"])
    output["pred_viscosity_cP"] = output["pred_viscosity_Pa_s"] * 1.0e3
    return output


def predict_csv(
    input_csv: str | Path,
    model_dir: str | Path,
    output_csv: str | Path,
    *,
    device_name: str,
) -> pd.DataFrame:
    """Predict one CSV and save the resulting table."""

    predictions = predict_ensemble(
        load_prediction_csv(input_csv),
        model_dir,
        device_name=device_name,
    )
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    return predictions
