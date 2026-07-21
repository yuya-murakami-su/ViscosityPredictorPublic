from __future__ import annotations

import math

import pandas as pd
import pytest

from viscosity_predictor.data import (
    CANONICAL_SMILES,
    INVERSE_TEMPERATURE,
    LN_VISCOSITY_PA_S,
    N_MEASUREMENTS,
    TEMPERATURE_CLUSTER_ID,
    VISCOSITY_PA_S,
    canonicalize_smiles,
    load_training_csv,
    prepare_training_data,
)


def _write_training_csv(tmp_path, rows: list[dict[str, object]]):
    path = tmp_path / "training.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_training_csv_requires_all_columns(tmp_path) -> None:
    path = _write_training_csv(
        tmp_path,
        [{"compound_id": "ethanol", "smiles": "CCO", "temperature_K": 298.15}],
    )

    with pytest.raises(ValueError, match="viscosity_cP"):
        load_training_csv(path)


def test_canonicalize_smiles_preserves_isomeric_identity() -> None:
    assert canonicalize_smiles("C(C)O") == "CCO"
    assert canonicalize_smiles("F/C=C/F") != canonicalize_smiles("F/C=C\\F")


def test_prepare_training_data_uses_bounded_temperature_clusters(tmp_path) -> None:
    path = _write_training_csv(
        tmp_path,
        [
            {"compound_id": "ethanol-a", "smiles": "CCO", "temperature_K": 300.00, "viscosity_cP": 1.0},
            {"compound_id": "ethanol-b", "smiles": "C(C)O", "temperature_K": 300.02, "viscosity_cP": 4.0},
            {"compound_id": "ethanol-c", "smiles": "CCO", "temperature_K": 300.04, "viscosity_cP": 9.0},
        ],
    )

    prepared = prepare_training_data(path, cluster_tolerance_k=0.03)

    assert prepared[CANONICAL_SMILES].tolist() == ["CCO", "CCO"]
    assert prepared[TEMPERATURE_CLUSTER_ID].tolist() == [0, 1]
    assert prepared[N_MEASUREMENTS].tolist() == [2, 1]
    assert prepared.loc[0, "compound_id"] == "ethanol-a; ethanol-b"
    assert prepared.loc[0, "temperature_K"] == pytest.approx(300.01)
    assert prepared.loc[0, "viscosity_cP"] == pytest.approx(2.0)


def test_prepare_training_data_adds_research_unit_transforms(tmp_path) -> None:
    path = _write_training_csv(
        tmp_path,
        [{"compound_id": "water", "smiles": "O", "temperature_K": 400.0, "viscosity_cP": 2.0}],
    )

    prepared = prepare_training_data(path, cluster_tolerance_k=0.03)
    row = prepared.iloc[0]

    assert row[VISCOSITY_PA_S] == pytest.approx(0.002)
    assert row[LN_VISCOSITY_PA_S] == pytest.approx(math.log(0.002))
    assert row[INVERSE_TEMPERATURE] == pytest.approx(1.0 / 400.0)
