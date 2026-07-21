"""Train-only curvature regularization and Joint coefficient search."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from murakami_lab_modules.data import DataHandler
from murakami_lab_modules.pinn import InputGenerator, Regularization

from .artifacts import atomic_write_csv
from .preprocessing import prepare_fold_data
from .splits import CrossValidationFold
from .training import (
    NN_BATCH_SIZE,
    NN_EARLY_STOPPING_PATIENCE,
    NN_MAX_EPOCHS,
    NativeCandidate,
    train_candidate,
)


COLLOCATION_BATCH_SIZE = 1024
JOINT_RESULT_COLUMNS = (
    "coefficient_id",
    "base_candidate_id",
    "selected_blocks",
    "hidden_layers",
    "hidden_units",
    "learning_rate",
    "weight_decay",
    "lambda_hessian",
    "lambda_soft",
    "model_family",
    "fold",
    "temperature_validation_cutoff_K",
    "seed",
    "best_epoch",
    "epochs_completed",
    "structure_mae",
    "temperature_mae",
    "integrated_score",
)


class TrainGaussianCollocationGenerator(InputGenerator):
    """Sample normalized Gaussian descriptors and uniformly sampled temperatures."""

    def __init__(
        self,
        *,
        train_inputs_norm: torch.Tensor,
        input_normalizer: Any,
        descriptor_std_multiplier: float,
        temperature_min_k: float,
        temperature_max_k: float,
        n_samples: int,
        device_name: str,
        random_seed: int,
    ) -> None:
        if train_inputs_norm.ndim != 2 or train_inputs_norm.shape[1] < 2:
            raise ValueError("Collocation requires descriptors and inverse temperature.")
        if float(descriptor_std_multiplier) <= 0.0:
            raise ValueError("descriptor_std_multiplier must be positive.")
        if float(temperature_min_k) <= 0.0 or float(temperature_max_k) <= float(temperature_min_k):
            raise ValueError("The collocation temperature range is invalid.")
        super().__init__(
            n_samples=int(n_samples),
            inputs=torch.zeros((int(n_samples), int(train_inputs_norm.shape[1]))),
            device_name=device_name,
            sampling="fixed",
            requires_grad=True,
            random_seed=int(random_seed),
        )
        self.n_descriptor_features = int(train_inputs_norm.shape[1] - 1)
        self.descriptor_std_multiplier = float(descriptor_std_multiplier)
        self.temperature_min_k = float(temperature_min_k)
        self.temperature_max_k = float(temperature_max_k)
        self.input_mean = torch.as_tensor(
            input_normalizer.mean,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        self.input_std = torch.as_tensor(
            input_normalizer.std,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        if self.input_mean.shape[1] != train_inputs_norm.shape[1]:
            raise ValueError("Input normalizer dimensions do not match the training inputs.")

    def sample_with_direction(self) -> tuple[torch.Tensor, torch.Tensor]:
        descriptor_noise = torch.randn(
            (self.n_samples, self.n_descriptor_features),
            generator=self._random_generator,
            device=self.device,
            dtype=torch.float32,
        )
        descriptor_mean = self.input_mean[:, :-1]
        descriptor_std = self.input_std[:, :-1]
        descriptor_raw = (
            descriptor_mean
            + self.descriptor_std_multiplier * descriptor_std * descriptor_noise
        )
        descriptor_norm = (descriptor_raw - descriptor_mean) / descriptor_std

        direction = torch.randn(
            (self.n_samples, self.n_descriptor_features),
            generator=self._random_generator,
            device=self.device,
            dtype=torch.float32,
        )
        direction = direction / torch.linalg.norm(direction, dim=1, keepdim=True).clamp_min(1.0e-12)

        temperature = (
            torch.rand(
                (self.n_samples, 1),
                generator=self._random_generator,
                device=self.device,
            )
            * (self.temperature_max_k - self.temperature_min_k)
            + self.temperature_min_k
        )
        inverse_temperature = 1.0 / temperature
        inverse_temperature_norm = (
            inverse_temperature - self.input_mean[:, -1:]
        ) / self.input_std[:, -1:]
        inputs = torch.cat([descriptor_norm, inverse_temperature_norm], dim=1)
        full_direction = torch.cat([direction, torch.zeros_like(inverse_temperature_norm)], dim=1)
        return inputs.requires_grad_(True), full_direction.detach()

    def __call__(self) -> torch.Tensor:
        return self.sample_with_direction()[0]


class DescriptorCurvatureRegularization(Regularization):
    def __init__(self, generator: TrainGaussianCollocationGenerator, weight: float) -> None:
        super().__init__(
            input_generators=[generator],
            weights=[float(weight)],
            term_names=["descriptor_directional_curvature_loss"],
        )

    def regularization(self, data_handler, nn):
        inputs, direction = self.input_generators[0].sample_with_direction()
        prediction = nn(inputs)
        gradient = self.grad(prediction, inputs, y_idx=0, zero_if_unused=True)
        first = (gradient * direction).sum(dim=1, keepdim=True)
        directional_gradient = self.grad(
            first,
            inputs,
            y_idx=0,
            zero_if_unused=not first.requires_grad,
        )
        second = (directional_gradient * direction).sum(dim=1, keepdim=True)
        return [second]


class ArrheniusCurvatureRegularization(Regularization):
    def __init__(self, generator: TrainGaussianCollocationGenerator, weight: float) -> None:
        super().__init__(
            input_generators=[generator],
            weights=[float(weight)],
            term_names=["arrhenius_curvature_loss"],
        )

    def regularization(self, data_handler, nn):
        inputs = self.input_generators[0]()
        prediction = nn(inputs)
        _, curvature = self.partial2(
            prediction,
            inputs,
            x_idx=-1,
            y_idx=0,
            keepdim=True,
        )
        return [curvature]


class JointCurvatureRegularization(Regularization):
    def __init__(
        self,
        descriptor_generator: TrainGaussianCollocationGenerator,
        temperature_generator: TrainGaussianCollocationGenerator,
        hessian_weight: float,
        soft_weight: float,
    ) -> None:
        super().__init__(
            input_generators=[descriptor_generator, temperature_generator],
            weights=[float(hessian_weight), float(soft_weight)],
            term_names=[
                "descriptor_directional_curvature_loss",
                "arrhenius_curvature_loss",
            ],
        )

    def regularization(self, data_handler, nn):
        descriptor_inputs, direction = self.input_generators[0].sample_with_direction()
        descriptor_prediction = nn(descriptor_inputs)
        gradient = self.grad(descriptor_prediction, descriptor_inputs, y_idx=0, zero_if_unused=True)
        first = (gradient * direction).sum(dim=1, keepdim=True)
        directional_gradient = self.grad(
            first,
            descriptor_inputs,
            y_idx=0,
            zero_if_unused=not first.requires_grad,
        )
        descriptor_curvature = (directional_gradient * direction).sum(dim=1, keepdim=True)

        temperature_inputs = self.input_generators[1]()
        temperature_prediction = nn(temperature_inputs)
        _, temperature_curvature = self.partial2(
            temperature_prediction,
            temperature_inputs,
            x_idx=-1,
            y_idx=0,
            keepdim=True,
        )
        return [descriptor_curvature, temperature_curvature]


def build_curvature_regularization(
    data_handler: DataHandler,
    config: dict[str, Any],
    *,
    lambda_hessian: float,
    lambda_soft: float,
    seed: int,
    n_samples: int = COLLOCATION_BATCH_SIZE,
) -> Regularization | None:
    """Build native, Hessian, Soft, or Joint regularization from two lambdas."""

    hessian = float(lambda_hessian)
    soft = float(lambda_soft)
    if hessian < 0.0 or soft < 0.0:
        raise ValueError("Regularization coefficients must be non-negative.")
    if hessian == 0.0 and soft == 0.0:
        return None

    temperature = config["temperature"]
    collocation = config["collocation"]
    common = {
        "train_inputs_norm": data_handler.train.inputs,
        "input_normalizer": data_handler.input_normalizer,
        "temperature_min_k": float(temperature["collocation_min_K"]),
        "temperature_max_k": float(temperature["collocation_max_K"]),
        "n_samples": int(n_samples),
        "device_name": data_handler.device_name,
    }
    descriptor_generator = None
    temperature_generator = None
    if hessian > 0.0:
        descriptor_generator = TrainGaussianCollocationGenerator(
            **common,
            descriptor_std_multiplier=float(collocation["hessian_descriptor_std_multiplier"]),
            random_seed=int(seed) + 1001,
        )
    if soft > 0.0:
        temperature_generator = TrainGaussianCollocationGenerator(
            **common,
            descriptor_std_multiplier=float(collocation["soft_descriptor_std_multiplier"]),
            random_seed=int(seed) + 2001,
        )
    if descriptor_generator is not None and temperature_generator is not None:
        return JointCurvatureRegularization(
            descriptor_generator,
            temperature_generator,
            hessian,
            soft,
        )
    if descriptor_generator is not None:
        return DescriptorCurvatureRegularization(descriptor_generator, hessian)
    return ArrheniusCurvatureRegularization(temperature_generator, soft)


def joint_search_fit_count(config: dict[str, Any]) -> int:
    return (
        len(config["regularization_search"]["hessian_lambdas"])
        * len(config["regularization_search"]["soft_lambdas"])
        * int(config["cross_validation"]["n_folds"])
        * len(config["seeds"]["search"])
    )


def run_joint_search(
    data: pd.DataFrame,
    descriptor_table: pd.DataFrame,
    splits: list[CrossValidationFold],
    config: dict[str, Any],
    base_candidate: NativeCandidate,
    *,
    device_name: str,
    batch_size: int = NN_BATCH_SIZE,
    max_epochs: int = NN_MAX_EPOCHS,
    patience: int = NN_EARLY_STOPPING_PATIENCE,
    n_samples: int = COLLOCATION_BATCH_SIZE,
    progress: bool = True,
    existing_results: pd.DataFrame | None = None,
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate the configured 5 x 5 Joint coefficient grid."""

    grid = list(
        (float(hessian), float(soft))
        for hessian in config["regularization_search"]["hessian_lambdas"]
        for soft in config["regularization_search"]["soft_lambdas"]
    )
    seeds = list(map(int, config["seeds"]["search"]))
    total = len(grid) * len(splits) * len(seeds)
    expected_keys = {
        (f"h{hessian:g}_s{soft:g}", int(split.fold), seed)
        for hessian, soft in grid
        for split in splits
        for seed in seeds
    }
    rows, completed_keys = _validated_joint_checkpoint(
        existing_results,
        expected_keys=expected_keys,
        base_candidate_id=base_candidate.candidate_id,
    )
    completed = len(completed_keys)
    if progress and completed:
        print(f"joint search resume: {completed}/{total} fits already complete", flush=True)
    for hessian, soft in grid:
        for split in splits:
            for seed in seeds:
                coefficient_id = f"h{hessian:g}_s{soft:g}"
                key = (coefficient_id, int(split.fold), seed)
                if key in completed_keys:
                    continue
                prepared = prepare_fold_data(
                    data,
                    descriptor_table,
                    blocks=base_candidate.blocks,
                    split=split,
                    batch_size=int(batch_size),
                    device_name=device_name,
                    random_seed=seed,
                )
                regularization = build_curvature_regularization(
                    prepared.data_handler,
                    config,
                    lambda_hessian=hessian,
                    lambda_soft=soft,
                    seed=seed,
                    n_samples=n_samples,
                )
                fit = train_candidate(
                    prepared,
                    base_candidate.hyperparameters,
                    seed=seed,
                    regularization=regularization,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                completed += 1
                rows.append(
                    {
                        "coefficient_id": coefficient_id,
                        "base_candidate_id": base_candidate.candidate_id,
                        "selected_blocks": ";".join(base_candidate.blocks),
                        "hidden_layers": base_candidate.hyperparameters.hidden_layers,
                        "hidden_units": base_candidate.hyperparameters.hidden_units,
                        "learning_rate": base_candidate.hyperparameters.learning_rate,
                        "weight_decay": base_candidate.hyperparameters.weight_decay,
                        "lambda_hessian": hessian,
                        "lambda_soft": soft,
                        "model_family": _model_family(hessian, soft),
                        "fold": split.fold,
                        "temperature_validation_cutoff_K": (
                            split.temperature_validation_cutoff_k
                        ),
                        "seed": seed,
                        "best_epoch": fit.best_epoch,
                        "epochs_completed": fit.epochs_completed,
                        "structure_mae": fit.structure_mae,
                        "temperature_mae": fit.temperature_mae,
                        "integrated_score": fit.integrated_score,
                    }
                )
                completed_keys.add(key)
                table = pd.DataFrame(rows, columns=JOINT_RESULT_COLUMNS)
                if checkpoint_path is not None:
                    atomic_write_csv(table, checkpoint_path)
                if progress:
                    print(
                        f"joint search {completed}/{total}: hessian={hessian:g}, soft={soft:g}, "
                        f"fold={split.fold}, seed={seed}, score={fit.integrated_score:.6f}",
                        flush=True,
                    )
    return pd.DataFrame(rows, columns=JOINT_RESULT_COLUMNS)


def _validated_joint_checkpoint(
    existing_results: pd.DataFrame | None,
    *,
    expected_keys: set[tuple[str, int, int]],
    base_candidate_id: str,
) -> tuple[list[dict[str, Any]], set[tuple[str, int, int]]]:
    if existing_results is None:
        return [], set()
    missing = set(JOINT_RESULT_COLUMNS) - set(existing_results.columns)
    if missing:
        raise ValueError(f"Joint search checkpoint is missing columns: {sorted(missing)}")
    rows = existing_results.loc[:, list(JOINT_RESULT_COLUMNS)].to_dict(orient="records")
    if any(str(row["base_candidate_id"]) != base_candidate_id for row in rows):
        raise ValueError("Joint search checkpoint uses a different base candidate.")
    keys = {
        (str(row["coefficient_id"]), int(row["fold"]), int(row["seed"]))
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("Joint search checkpoint contains duplicate fit keys.")
    unexpected = keys - expected_keys
    if unexpected:
        raise ValueError("Joint search checkpoint contains unexpected fit keys.")
    return rows, keys


def rank_joint_results(
    results: pd.DataFrame,
    *,
    expected_folds: Iterable[int],
    expected_seeds: Iterable[int],
) -> pd.DataFrame:
    expected = {(int(fold), int(seed)) for fold in expected_folds for seed in expected_seeds}
    rows = []
    for coefficient_id, table in results.groupby("coefficient_id", sort=False):
        actual = set(zip(table["fold"].astype(int), table["seed"].astype(int), strict=True))
        if len(table) != len(expected) or actual != expected:
            raise ValueError(f"Incomplete fold/seed coverage for {coefficient_id}.")
        scores = table["integrated_score"].to_numpy(dtype=float)
        epochs = table["best_epoch"].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or not np.isfinite(epochs).all() or (epochs <= 0).any():
            raise ValueError(f"Non-finite score or invalid epoch for {coefficient_id}.")
        first = table.iloc[0]
        rows.append(
            {
                "coefficient_id": coefficient_id,
                "lambda_hessian": float(first["lambda_hessian"]),
                "lambda_soft": float(first["lambda_soft"]),
                "model_family": first["model_family"],
                "n_fits": len(table),
                "mean_integrated_score": float(scores.mean()),
                "std_integrated_score": float(scores.std(ddof=0)),
                "worst_integrated_score": float(scores.max()),
                "mean_best_epoch": float(epochs.mean()),
                "fixed_epochs": max(1, int(round(float(epochs.mean())))),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["mean_integrated_score", "worst_integrated_score", "lambda_hessian", "lambda_soft"],
        kind="stable",
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def _model_family(hessian: float, soft: float) -> str:
    if hessian > 0.0 and soft > 0.0:
        return "joint"
    if hessian > 0.0:
        return "hessian"
    if soft > 0.0:
        return "soft"
    return "native"
