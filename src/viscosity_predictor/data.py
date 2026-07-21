"""Load and minimally prepare user-provided viscosity measurements."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


REQUIRED_TRAINING_COLUMNS = (
    "compound_id",
    "smiles",
    "temperature_K",
    "viscosity_cP",
)

CANONICAL_SMILES = "canonical_smiles"
TEMPERATURE_CLUSTER_ID = "temperature_cluster_id"
VISCOSITY_PA_S = "viscosity_Pa_s"
LN_VISCOSITY_PA_S = "ln_viscosity_Pa_s"
INVERSE_TEMPERATURE = "inverse_temperature"
N_MEASUREMENTS = "n_measurements"


def load_training_csv(path: str | Path) -> pd.DataFrame:
    """Read the required four-column training CSV and validate its values."""

    data = pd.read_csv(path, low_memory=False)
    data.columns = [str(column).strip() for column in data.columns]
    missing = [column for column in REQUIRED_TRAINING_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required training columns: {missing}")
    if data.empty:
        raise ValueError("The training CSV contains no data rows.")

    data = data.loc[:, list(REQUIRED_TRAINING_COLUMNS)].copy()
    _validate_text_column(data, "compound_id")
    _validate_text_column(data, "smiles")
    _validate_positive_numeric_column(data, "temperature_K")
    _validate_positive_numeric_column(data, "viscosity_cP")
    return data


def canonicalize_smiles(smiles: str) -> str:
    """Return RDKit's canonical isomeric SMILES without changing salt or charge state."""

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def prepare_training_data(
    path: str | Path,
    *,
    cluster_tolerance_k: float,
) -> pd.DataFrame:
    """Load, canonicalize, cluster, and aggregate viscosity measurements."""

    data = load_training_csv(path)
    canonical_by_input = {
        smiles: canonicalize_smiles(smiles)
        for smiles in data["smiles"].drop_duplicates().tolist()
    }
    data[CANONICAL_SMILES] = data["smiles"].map(canonical_by_input)
    data[VISCOSITY_PA_S] = data["viscosity_cP"] * 1.0e-3
    data[LN_VISCOSITY_PA_S] = np.log(data[VISCOSITY_PA_S])

    clustered = assign_temperature_clusters(data, cluster_tolerance_k=cluster_tolerance_k)
    return aggregate_temperature_clusters(clustered)


def assign_temperature_clusters(
    data: pd.DataFrame,
    *,
    cluster_tolerance_k: float,
) -> pd.DataFrame:
    """Assign per-compound clusters whose total temperature width is bounded."""

    if float(cluster_tolerance_k) < 0.0:
        raise ValueError("cluster_tolerance_k must be non-negative.")

    work = data.copy()
    work["_input_order"] = np.arange(len(work), dtype=int)
    work = work.sort_values(
        [CANONICAL_SMILES, "temperature_K", "_input_order"],
        kind="stable",
    ).copy()
    cluster_ids = pd.Series(index=work.index, dtype="int64")

    for _, compound in work.groupby(CANONICAL_SMILES, sort=False):
        cluster_id = 0
        cluster_minimum: float | None = None
        for index, temperature in compound["temperature_K"].items():
            value = float(temperature)
            if cluster_minimum is None:
                cluster_minimum = value
            elif value - cluster_minimum > float(cluster_tolerance_k):
                cluster_id += 1
                cluster_minimum = value
            cluster_ids.loc[index] = cluster_id

    work[TEMPERATURE_CLUSTER_ID] = cluster_ids.astype(int)
    return work.drop(columns="_input_order").reset_index(drop=True)


def aggregate_temperature_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    """Create one median representative for every compound-temperature cluster."""

    group_columns = [CANONICAL_SMILES, TEMPERATURE_CLUSTER_ID]
    representatives = (
        clustered.groupby(group_columns, sort=False, as_index=False)
        .agg(
            compound_id=("compound_id", _join_unique_text),
            temperature_K=("temperature_K", "median"),
            ln_viscosity_Pa_s=(LN_VISCOSITY_PA_S, "median"),
            n_measurements=("temperature_K", "size"),
        )
        .reset_index(drop=True)
    )
    representatives[VISCOSITY_PA_S] = np.exp(representatives[LN_VISCOSITY_PA_S])
    representatives["viscosity_cP"] = representatives[VISCOSITY_PA_S] * 1.0e3
    representatives[INVERSE_TEMPERATURE] = 1.0 / representatives["temperature_K"]
    return representatives[
        [
            "compound_id",
            CANONICAL_SMILES,
            "temperature_K",
            "viscosity_cP",
            VISCOSITY_PA_S,
            LN_VISCOSITY_PA_S,
            INVERSE_TEMPERATURE,
            TEMPERATURE_CLUSTER_ID,
            N_MEASUREMENTS,
        ]
    ]


def _validate_text_column(data: pd.DataFrame, column: str) -> None:
    values = data[column].astype("string").str.strip()
    invalid = values.isna() | (values == "") | (values.str.lower() == "nan")
    if invalid.any():
        raise ValueError(f"Column {column!r} contains blank values at CSV rows: {_csv_rows(invalid)}")
    data[column] = values.astype(str)


def _validate_positive_numeric_column(data: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(data[column], errors="coerce")
    invalid = ~np.isfinite(values) | (values <= 0.0)
    if invalid.any():
        raise ValueError(
            f"Column {column!r} contains non-positive or non-numeric values at CSV rows: "
            f"{_csv_rows(invalid)}"
        )
    data[column] = values.astype(float)


def _csv_rows(mask: pd.Series) -> list[int]:
    return (np.flatnonzero(mask.to_numpy())[:10] + 2).tolist()


def _join_unique_text(values: pd.Series) -> str:
    return "; ".join(sorted(values.astype(str).unique()))
