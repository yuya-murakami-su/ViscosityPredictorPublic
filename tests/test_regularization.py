from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from viscosity_predictor.config import load_training_config
from viscosity_predictor.regularization import (
    ArrheniusCurvatureRegularization,
    DescriptorCurvatureRegularization,
    JointCurvatureRegularization,
    TrainGaussianCollocationGenerator,
    joint_search_fit_count,
    rank_joint_results,
    run_joint_search,
)
from viscosity_predictor.training import NNHyperparameters, NativeCandidate


class QuadraticModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :1].pow(2) + 3.0 * inputs[:, -1:].pow(2)


def _generator(seed: int = 1, multiplier: float = 2.0, n_samples: int = 256):
    normalizer = SimpleNamespace(
        mean=torch.zeros((1, 2)),
        std=torch.ones((1, 2)),
    )
    return TrainGaussianCollocationGenerator(
        train_inputs_norm=torch.zeros((4, 2)),
        input_normalizer=normalizer,
        descriptor_std_multiplier=multiplier,
        temperature_min_k=300.0,
        temperature_max_k=400.0,
        n_samples=n_samples,
        device_name="cpu",
        random_seed=seed,
    )


def test_gaussian_collocation_uses_requested_spread_and_unit_directions() -> None:
    inputs, directions = _generator(n_samples=5000).sample_with_direction()
    sampled_temperature = (1.0 / inputs[:, -1]).detach()

    assert np.isclose(float(inputs[:, 0].detach().std(unbiased=False)), 2.0, atol=0.08)
    assert torch.allclose(torch.linalg.norm(directions[:, :-1], dim=1), torch.ones(5000))
    assert torch.count_nonzero(directions[:, -1]) == 0
    assert float(sampled_temperature.min()) >= 300.0
    assert float(sampled_temperature.max()) <= 400.0


def test_curvature_regularizers_match_quadratic_analytic_values() -> None:
    model = QuadraticModel()
    descriptor = DescriptorCurvatureRegularization(_generator(seed=1), weight=0.1)
    arrhenius = ArrheniusCurvatureRegularization(_generator(seed=2), weight=0.2)
    joint = JointCurvatureRegularization(_generator(seed=3), _generator(seed=4), 0.1, 0.2)

    assert torch.allclose(descriptor.compute_raw_term_means(model), torch.tensor([4.0]))
    assert torch.allclose(arrhenius.compute_raw_term_means(model), torch.tensor([36.0]))
    assert torch.allclose(joint.compute_raw_term_means(model), torch.tensor([4.0, 36.0]))


def test_default_joint_grid_contains_25_conditions_and_375_fits() -> None:
    config = load_training_config("config/training.toml")

    assert len(config["regularization_search"]["hessian_lambdas"]) == 5
    assert len(config["regularization_search"]["soft_lambdas"]) == 5
    assert joint_search_fit_count(config) == 25 * 5 * 3


def test_joint_ranking_averages_complete_paired_results() -> None:
    rows = []
    for coefficient_id, hessian, soft, offset in (
        ("h0_s0", 0.0, 0.0, 1.0),
        ("h1_s0.1", 1.0, 0.1, 0.0),
    ):
        for fold in (0, 1):
            for seed in (1, 2):
                rows.append(
                    {
                        "coefficient_id": coefficient_id,
                        "lambda_hessian": hessian,
                        "lambda_soft": soft,
                        "model_family": "native" if hessian == soft == 0.0 else "joint",
                        "fold": fold,
                        "split_strategy": "integrated",
                        "seed": seed,
                        "best_epoch": 10,
                        "integrated_score": offset + fold / 10.0 + seed / 100.0,
                        "selection_score": offset + fold / 10.0 + seed / 100.0,
                    }
                )

    ranking = rank_joint_results(
        pd.DataFrame(rows),
        expected_folds=[0, 1],
        expected_seeds=[1, 2],
    )

    assert ranking["coefficient_id"].tolist() == ["h1_s0.1", "h0_s0"]
    assert ranking["fixed_epochs"].tolist() == [10, 10]


def test_joint_search_checkpoints_each_fit_and_resumes(tmp_path, monkeypatch) -> None:
    config = {
        "regularization_search": {
            "hessian_lambdas": [0.0, 1.0],
            "soft_lambdas": [0.0],
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
    base = NativeCandidate(
        candidate_id="base",
        blocks=("physicochemical_descriptors",),
        hyperparameters=NNHyperparameters(1, 8, 1.0e-2, 0.0),
    )

    monkeypatch.setattr(
        "viscosity_predictor.regularization.prepare_fold_data",
        lambda *args, **kwargs: SimpleNamespace(
            split=kwargs["split"],
            data_handler=None,
        ),
    )
    monkeypatch.setattr(
        "viscosity_predictor.regularization.build_curvature_regularization",
        lambda *args, **kwargs: None,
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

    monkeypatch.setattr("viscosity_predictor.regularization.train_candidate", interrupted_fit)
    checkpoint = tmp_path / "joint_search_results.csv"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_joint_search(
            pd.DataFrame(),
            pd.DataFrame(),
            splits,
            config,
            base,
            device_name="cpu",
            progress=False,
            checkpoint_path=checkpoint,
        )

    partial = pd.read_csv(checkpoint)
    assert partial[["coefficient_id", "fold", "seed"]].to_records(index=False).tolist() == [
        ("h0_s0", 0, 1)
    ]

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

    monkeypatch.setattr("viscosity_predictor.regularization.train_candidate", resumed_fit)
    results = run_joint_search(
        pd.DataFrame(),
        pd.DataFrame(),
        splits,
        config,
        base,
        device_name="cpu",
        progress=False,
        existing_results=partial,
        checkpoint_path=checkpoint,
    )

    assert resumed_calls == [1, 0, 1]
    assert len(results) == 4
    assert len(pd.read_csv(checkpoint)) == 4
