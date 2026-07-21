from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from viscosity_predictor.config import load_training_config
from viscosity_predictor.data import CANONICAL_SMILES, INVERSE_TEMPERATURE, LN_VISCOSITY_PA_S
from viscosity_predictor.preprocessing import prepare_fold_data
from viscosity_predictor.splits import (
    TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY,
    CrossValidationFold,
)
from viscosity_predictor.training import (
    NNHyperparameters,
    build_native_candidates,
    native_search_fit_count,
    rank_native_results,
    run_native_search,
    train_native_candidate,
)


def test_default_native_search_contains_all_186_candidates() -> None:
    config = load_training_config("config/training.toml")
    candidates = build_native_candidates(config)

    assert len(candidates) == 31 * 6
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert native_search_fit_count(config) == 31 * 6 * 5 * 3


def test_native_candidate_uses_integrated_mae_for_best_epoch() -> None:
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
    split = CrossValidationFold(
        fold=0,
        train_indices=np.array([0, 1, 2, 3]),
        validation_indices=np.array([4, 5, 6, 7]),
        structure_validation_indices=np.array([4, 5]),
        temperature_validation_indices=np.array([6, 7]),
        temperature_validation_cutoff_k=400.0,
    )
    prepared = prepare_fold_data(
        data,
        descriptors,
        blocks=["physicochemical_descriptors"],
        split=split,
        batch_size=4,
        device_name="cpu",
        random_seed=3,
    )

    fit = train_native_candidate(
        prepared,
        NNHyperparameters(1, 8, 1.0e-2, 0.0),
        seed=3,
        max_epochs=8,
        patience=3,
    )

    assert 1 <= fit.best_epoch <= 8
    assert 1 <= fit.epochs_completed <= 8
    assert np.isfinite(fit.structure_mae)
    assert np.isfinite(fit.temperature_mae)
    assert np.isclose(fit.integrated_score, 0.5 * (fit.structure_mae + fit.temperature_mae))
    assert fit.model_handler.best_epoch == fit.best_epoch - 1


def test_native_candidate_uses_interpolation_validation_mae() -> None:
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
    split = CrossValidationFold(
        fold=0,
        train_indices=np.array([0, 1, 2, 3]),
        validation_indices=np.array([4, 5, 6, 7]),
        structure_validation_indices=np.array([], dtype=int),
        temperature_validation_indices=np.array([], dtype=int),
        temperature_validation_cutoff_k=None,
        split_strategy=TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY,
    )
    prepared = prepare_fold_data(
        data,
        descriptors,
        blocks=["physicochemical_descriptors"],
        split=split,
        batch_size=4,
        device_name="cpu",
        random_seed=3,
    )

    fit = train_native_candidate(
        prepared,
        NNHyperparameters(1, 8, 1.0e-2, 0.0),
        seed=3,
        max_epochs=4,
        patience=2,
    )

    assert np.isnan(fit.structure_mae)
    assert np.isnan(fit.temperature_mae)
    assert np.isfinite(fit.integrated_score)


def test_native_ranking_requires_complete_paired_coverage() -> None:
    rows = []
    for candidate, offset in (("a", 0.0), ("b", 1.0)):
        for fold in (0, 1):
            for seed in (1, 2):
                rows.append(
                    {
                        "candidate_id": candidate,
                        "selected_blocks": "physicochemical_descriptors",
                        "fold": fold,
                        "split_strategy": "integrated",
                        "seed": seed,
                        "hidden_layers": 1,
                        "hidden_units": 8,
                        "learning_rate": 1.0e-3,
                        "weight_decay": 0.0,
                        "best_epoch": 5,
                        "integrated_score": offset + fold + seed / 10.0,
                        "selection_score": offset + fold + seed / 10.0,
                    }
                )

    ranking = rank_native_results(
        pd.DataFrame(rows),
        expected_folds=[0, 1],
        expected_seeds=[1, 2],
    )

    assert ranking["candidate_id"].tolist() == ["a", "b"]
    assert ranking["n_fits"].tolist() == [4, 4]


def test_native_search_checkpoints_each_fit_and_resumes(tmp_path, monkeypatch) -> None:
    config = {
        "descriptors": {"candidates": ["physicochemical_descriptors"]},
        "nn_search": {
            "hidden_layers": [1],
            "hidden_units": [8],
            "learning_rates": [1.0e-2],
            "weight_decays": [0.0],
        },
        "seeds": {"search": [1]},
    }
    splits = [
        SimpleNamespace(
            fold=0,
            split_strategy="integrated",
            temperature_validation_cutoff_k=350.0,
        ),
        SimpleNamespace(
            fold=1,
            split_strategy="integrated",
            temperature_validation_cutoff_k=360.0,
        ),
    ]

    def fake_prepare(*args, **kwargs):
        return SimpleNamespace(
            split=kwargs["split"],
            selection=SimpleNamespace(selected_columns=("descriptor",)),
        )

    calls = []

    def interrupted_fit(prepared, *args, **kwargs):
        calls.append(prepared.split.fold)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return SimpleNamespace(
            best_epoch=2,
            epochs_completed=2,
            validation_mae=0.15,
            structure_mae=0.2,
            temperature_mae=0.1,
            integrated_score=0.15,
            selection_score=0.15,
        )

    monkeypatch.setattr("viscosity_predictor.training.prepare_fold_data", fake_prepare)
    monkeypatch.setattr("viscosity_predictor.training.train_native_candidate", interrupted_fit)
    checkpoint = tmp_path / "native_search_results.csv"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_native_search(
            pd.DataFrame(),
            pd.DataFrame(),
            splits,
            config,
            device_name="cpu",
            progress=False,
            checkpoint_path=checkpoint,
        )

    partial = pd.read_csv(checkpoint)
    assert partial[["fold", "seed"]].to_records(index=False).tolist() == [(0, 1)]

    resumed_calls = []

    def resumed_fit(prepared, *args, **kwargs):
        resumed_calls.append(prepared.split.fold)
        return SimpleNamespace(
            best_epoch=3,
            epochs_completed=3,
            validation_mae=0.2,
            structure_mae=0.3,
            temperature_mae=0.1,
            integrated_score=0.2,
            selection_score=0.2,
        )

    monkeypatch.setattr("viscosity_predictor.training.train_native_candidate", resumed_fit)
    results = run_native_search(
        pd.DataFrame(),
        pd.DataFrame(),
        splits,
        config,
        device_name="cpu",
        progress=False,
        existing_results=partial,
        checkpoint_path=checkpoint,
    )

    assert resumed_calls == [1]
    assert len(results) == 2
    assert len(pd.read_csv(checkpoint)) == 2
