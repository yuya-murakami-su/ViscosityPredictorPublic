from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from viscosity_predictor.data import CANONICAL_SMILES
from viscosity_predictor.splits import (
    STRUCTURE_FOLD,
    TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY,
    assign_similarity_graph_folds,
    create_integrated_cv_splits,
    create_temperature_interpolation_cv_splits,
)


def _structure_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            CANONICAL_SMILES: [
                "CCO",
                "CCO",
                "CCCO",
                "CCCCO",
                "CC(=O)O",
                "c1ccccc1",
                "c1ccncc1",
                "CCN",
            ]
        }
    )


def test_similarity_graph_folds_are_reproducible_and_structure_grouped() -> None:
    data = _structure_data()

    first = assign_similarity_graph_folds(data, n_folds=3, split_seed=7)
    second = assign_similarity_graph_folds(data, n_folds=3, split_seed=7)

    assert first.name == STRUCTURE_FOLD
    assert first.equals(second)
    assert first.nunique() == 3
    assert first.groupby(data[CANONICAL_SMILES]).nunique().eq(1).all()
    assert first.notna().all()


@pytest.mark.parametrize("n_folds", [0, 1, 8])
def test_similarity_graph_folds_reject_invalid_fold_count(n_folds: int) -> None:
    with pytest.raises(ValueError):
        assign_similarity_graph_folds(_structure_data(), n_folds=n_folds, split_seed=0)


def test_integrated_splits_partition_rows_without_structure_leakage() -> None:
    data = _structure_data()
    data["temperature_K"] = [280.0, 100.0, 120.0, 200.0, 300.0, 180.0, 290.0, 240.0]
    splits = create_integrated_cv_splits(
        data,
        n_folds=3,
        split_seed=7,
        high_temperature_validation_quantile=0.80,
    )

    all_indices = set(range(len(data)))
    for split in splits:
        train = set(split.train_indices)
        structure = set(split.structure_validation_indices)
        temperature = set(split.temperature_validation_indices)
        validation = set(split.validation_indices)

        assert train.isdisjoint(validation)
        assert structure.isdisjoint(temperature)
        assert train | validation == all_indices
        assert validation == structure | temperature
        development = sorted(all_indices - structure)
        cutoff = float(data.iloc[development]["temperature_K"].quantile(0.80))
        expected_temperature = {
            index
            for index in development
            if float(data.iloc[index]["temperature_K"]) >= cutoff
        }
        assert np.isclose(split.temperature_validation_cutoff_k, cutoff)
        assert temperature == expected_temperature

        structure_smiles = set(data.iloc[list(structure)][CANONICAL_SMILES])
        development_smiles = set(data.iloc[list(train | temperature)][CANONICAL_SMILES])
        assert structure_smiles.isdisjoint(development_smiles)


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.1])
def test_integrated_splits_reject_invalid_quantile(quantile: float) -> None:
    data = _structure_data()
    data["temperature_K"] = range(100, 108)

    with pytest.raises(ValueError, match="between 0 and 1"):
        create_integrated_cv_splits(
            data,
            n_folds=3,
            split_seed=0,
            high_temperature_validation_quantile=quantile,
        )


def test_temperature_interpolation_splits_protect_endpoints_and_cover_candidates() -> None:
    data = pd.DataFrame(
        {
            CANONICAL_SMILES: [
                "C",
                "CC",
                "CC",
                "CCC",
                "CCC",
                "CCC",
                "CCCC",
                "CCCC",
                "CCCC",
                "CCCC",
                "CCCC",
            ],
            "temperature_K": [
                300.0,
                290.0,
                310.0,
                280.0,
                300.0,
                320.0,
                270.0,
                285.0,
                300.0,
                315.0,
                330.0,
            ],
        }
    )

    first = create_temperature_interpolation_cv_splits(
        data,
        n_folds=3,
        split_seed=7,
    )
    second = create_temperature_interpolation_cv_splits(
        data,
        n_folds=3,
        split_seed=7,
    )

    assert [split.validation_indices.tolist() for split in first] == [
        split.validation_indices.tolist() for split in second
    ]
    assert max(len(split.validation_indices) for split in first) - min(
        len(split.validation_indices) for split in first
    ) <= 1
    validation_counts = np.zeros(len(data), dtype=int)
    all_indices = set(range(len(data)))
    for split in first:
        train = set(split.train_indices)
        validation = set(split.validation_indices)
        assert split.split_strategy == TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY
        assert train.isdisjoint(validation)
        assert train | validation == all_indices
        assert len(split.structure_validation_indices) == 0
        assert len(split.temperature_validation_indices) == 0
        assert split.temperature_validation_cutoff_k is None
        validation_counts[split.validation_indices] += 1
        train_data = data.iloc[split.train_indices]
        validation_data = data.iloc[split.validation_indices]
        assert set(validation_data[CANONICAL_SMILES]) <= set(train_data[CANONICAL_SMILES])
        ranges = train_data.groupby(CANONICAL_SMILES)["temperature_K"].agg(["min", "max"])
        for row in validation_data.itertuples(index=False):
            assert ranges.loc[row.canonical_smiles, "min"] < row.temperature_K
            assert row.temperature_K < ranges.loc[row.canonical_smiles, "max"]

    expected_candidates = np.array([4, 7, 8, 9])
    assert np.array_equal(np.flatnonzero(validation_counts), expected_candidates)
    assert np.array_equal(validation_counts[expected_candidates], np.ones(4, dtype=int))
    assert np.array_equal(
        validation_counts[np.setdiff1d(np.arange(len(data)), expected_candidates)],
        np.zeros(len(data) - len(expected_candidates), dtype=int),
    )


@pytest.mark.parametrize("n_folds", [0, 1, 5])
def test_temperature_interpolation_splits_reject_invalid_fold_count(
    n_folds: int,
) -> None:
    data = pd.DataFrame(
        {
            CANONICAL_SMILES: ["C"] * 5,
            "temperature_K": [280.0, 290.0, 300.0, 310.0, 320.0],
        }
    )
    with pytest.raises(ValueError):
        create_temperature_interpolation_cv_splits(
            data,
            n_folds=n_folds,
            split_seed=0,
        )
