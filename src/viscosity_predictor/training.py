"""Native neural-network search for the public training workflow."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from murakami_lab_modules.models import FeedForwardNeuralNetwork
from murakami_lab_modules.pinn import Regularization
from murakami_lab_modules.training import (
    BestModelTracker,
    Callback,
    DataFitting,
    EarlyStopping,
    ModelHandler,
    Optimizer,
)

from .artifacts import atomic_write_csv
from .descriptors import descriptor_subsets
from .preprocessing import PreparedFoldData, PreparedFullData, prepare_fold_data
from .splits import (
    INTEGRATED_SPLIT_STRATEGY,
    TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY,
    CrossValidationFold,
)


NN_BATCH_SIZE = 256
NN_MAX_EPOCHS = 3000
NN_EARLY_STOPPING_PATIENCE = 50
INTEGRATED_VALIDATION_MONITOR = "validation_integrated_mae_ln_viscosity"
STRUCTURE_VALIDATION_MONITOR = "validation_structure_mae_ln_viscosity"
TEMPERATURE_VALIDATION_MONITOR = "validation_temperature_mae_ln_viscosity"
NATIVE_RESULT_COLUMNS = (
    "candidate_id",
    "selected_blocks",
    "fold",
    "temperature_validation_cutoff_K",
    "seed",
    "hidden_layers",
    "hidden_units",
    "learning_rate",
    "weight_decay",
    "n_descriptor_features",
    "best_epoch",
    "epochs_completed",
    "structure_mae",
    "temperature_mae",
    "integrated_score",
)


@dataclass(frozen=True)
class NNHyperparameters:
    hidden_layers: int
    hidden_units: int
    learning_rate: float
    weight_decay: float

    @property
    def identifier(self) -> str:
        return (
            f"l{self.hidden_layers}_n{self.hidden_units}_"
            f"lr{self.learning_rate:g}_wd{self.weight_decay:g}"
        )


@dataclass(frozen=True)
class NativeCandidate:
    candidate_id: str
    blocks: tuple[str, ...]
    hyperparameters: NNHyperparameters


@dataclass(frozen=True)
class CandidateFit:
    model_handler: ModelHandler
    best_epoch: int
    epochs_completed: int
    structure_mae: float
    temperature_mae: float
    integrated_score: float


class ValidationMAEMonitor(Callback):
    """Record the validation metrics required by the selected split strategy."""

    def __init__(
        self,
        *,
        split: CrossValidationFold,
    ) -> None:
        super().__init__(every=1, run_on_train_end=False, priority=-20)
        self.split = split
        self.validation_indices = frozenset(map(int, split.validation_indices))
        self.structure_indices = frozenset(map(int, split.structure_validation_indices))
        self.temperature_indices = frozenset(map(int, split.temperature_validation_indices))
        if not self.validation_indices:
            raise ValueError("Validation rows must be non-empty.")
        if split.split_strategy == INTEGRATED_SPLIT_STRATEGY:
            if not self.structure_indices or not self.temperature_indices:
                raise ValueError("Both integrated validation endpoints must be non-empty.")
            if self.structure_indices & self.temperature_indices:
                raise ValueError(
                    "Structure and temperature validation indices must be disjoint."
                )
            if self.validation_indices != self.structure_indices | self.temperature_indices:
                raise ValueError(
                    "Integrated validation rows must equal the union of the two endpoints."
                )
        elif split.split_strategy == TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY:
            if self.structure_indices or self.temperature_indices:
                raise ValueError(
                    "Temperature-interpolation validation must not define integrated endpoints."
                )
        else:
            raise ValueError(f"Unsupported split strategy: {split.split_strategy!r}")
        self._structure_mask: np.ndarray | None = None
        self._temperature_mask: np.ndarray | None = None

    def on_train_begin(self, model_handler: ModelHandler) -> None:
        labels = np.asarray(model_handler.data_fitting.data_handler.valid.labels).reshape(-1)
        actual = set(map(int, labels))
        if actual != self.validation_indices:
            raise ValueError("DataHandler validation rows do not match the selected fold.")
        if self.split.split_strategy == INTEGRATED_SPLIT_STRATEGY:
            self._structure_mask = np.isin(labels, list(self.structure_indices))
            self._temperature_mask = np.isin(labels, list(self.temperature_indices))

    def on_epoch_end(self, model_handler: ModelHandler) -> None:
        fitting = model_handler.data_fitting
        valid = fitting.data_handler.valid
        model_handler.nn.eval()
        with torch.no_grad():
            prediction = fitting.to_observed_prediction(
                fitting.predict(model_handler.nn, valid.inputs, phase="valid", epoch=model_handler.epoch)
            ).reshape(-1)
            observed = fitting.to_observed_target(valid.outputs).reshape(-1)
            error = torch.abs(prediction - observed)
            validation_mae = error.mean()
            selection_score = validation_mae
            if self.split.split_strategy == INTEGRATED_SPLIT_STRATEGY:
                if self._structure_mask is None or self._temperature_mask is None:
                    raise RuntimeError("Validation endpoint masks were not initialized.")
                structure_mask = torch.as_tensor(self._structure_mask, device=error.device)
                temperature_mask = torch.as_tensor(self._temperature_mask, device=error.device)
                structure_mae = error[structure_mask].mean()
                temperature_mae = error[temperature_mask].mean()
                integrated = 0.5 * structure_mae + 0.5 * temperature_mae
                selection_score = integrated
                model_handler.current_evolution[STRUCTURE_VALIDATION_MONITOR] = float(
                    structure_mae.item()
                )
                model_handler.current_evolution[TEMPERATURE_VALIDATION_MONITOR] = float(
                    temperature_mae.item()
                )
        model_handler.current_evolution[INTEGRATED_VALIDATION_MONITOR] = float(
            selection_score.item()
        )


def build_native_candidates(config: dict[str, Any]) -> list[NativeCandidate]:
    """Create the full descriptor-subset and native-NN Cartesian product."""

    candidates = []
    subsets = descriptor_subsets(config["descriptors"]["candidates"])
    search = config["nn_search"]
    hyperparameters = [
        NNHyperparameters(int(layers), int(units), float(rate), float(decay))
        for layers, units, rate, decay in product(
            search["hidden_layers"],
            search["hidden_units"],
            search["learning_rates"],
            search["weight_decays"],
        )
    ]
    for subset_number, blocks in enumerate(subsets, start=1):
        for hp in hyperparameters:
            candidates.append(
                NativeCandidate(
                    candidate_id=f"d{subset_number:02d}_{hp.identifier}",
                    blocks=blocks,
                    hyperparameters=hp,
                )
            )
    return candidates


def native_search_fit_count(config: dict[str, Any]) -> int:
    return (
        len(build_native_candidates(config))
        * int(config["cross_validation"]["n_folds"])
        * len(config["seeds"]["search"])
    )


def native_candidate_from_ranking_row(row: pd.Series | dict[str, Any]) -> NativeCandidate:
    """Reconstruct the selected native candidate from one ranking row."""

    return NativeCandidate(
        candidate_id=str(row["candidate_id"]),
        blocks=tuple(str(row["selected_blocks"]).split(";")),
        hyperparameters=NNHyperparameters(
            hidden_layers=int(row["hidden_layers"]),
            hidden_units=int(row["hidden_units"]),
            learning_rate=float(row["learning_rate"]),
            weight_decay=float(row["weight_decay"]),
        ),
    )


def create_native_nn(input_dim: int, hp: NNHyperparameters, seed: int) -> FeedForwardNeuralNetwork:
    return FeedForwardNeuralNetwork(
        input_dim=int(input_dim),
        output_dim=1,
        n_hidden_layers=int(hp.hidden_layers),
        hidden_dim=int(hp.hidden_units),
        activation=torch.nn.Tanh(),
        dropout=0.0,
        random_seed=int(seed),
    )


def train_native_candidate(
    prepared: PreparedFoldData,
    hp: NNHyperparameters,
    *,
    seed: int,
    max_epochs: int = NN_MAX_EPOCHS,
    patience: int = NN_EARLY_STOPPING_PATIENCE,
) -> CandidateFit:
    """Train one native NN and select its epoch using the fold strategy."""

    return train_candidate(
        prepared,
        hp,
        seed=seed,
        regularization=None,
        max_epochs=max_epochs,
        patience=patience,
    )


def train_candidate(
    prepared: PreparedFoldData,
    hp: NNHyperparameters,
    *,
    seed: int,
    regularization: Regularization | None,
    max_epochs: int = NN_MAX_EPOCHS,
    patience: int = NN_EARLY_STOPPING_PATIENCE,
) -> CandidateFit:
    """Train a native or curvature-regularized NN under the same validation rule."""

    handler_data = prepared.data_handler
    network = create_native_nn(handler_data.inputs.shape[1], hp, seed)
    monitor = ValidationMAEMonitor(split=prepared.split)
    tracker = BestModelTracker(
        monitor=INTEGRATED_VALIDATION_MONITOR,
        mode="min",
        restore_best=True,
        save_optimizer=False,
    )
    handler = ModelHandler(
        nn=network,
        optimizer=Optimizer(
            torch.optim.AdamW,
            lr=float(hp.learning_rate),
            weight_decay=float(hp.weight_decay),
        ),
        data_fitting=DataFitting(data_handler=handler_data, loss_fn=torch.nn.MSELoss()),
        regularization=regularization,
        train_epochs=int(max_epochs),
        callbacks=(
            monitor,
            tracker,
            EarlyStopping(
                monitor=INTEGRATED_VALIDATION_MONITOR,
                mode="min",
                patience=int(patience),
            ),
        ),
        random_seed=int(seed),
        save_result=False,
        save_model=False,
        restore_best=False,
        history_policy="none",
        verbose=False,
    )
    handler()
    if tracker.best_epoch is None:
        raise RuntimeError("Validation did not record a best epoch.")
    handler.best_epoch = int(tracker.best_epoch)
    handler.best_loss = float(tracker.best_value)
    metrics = _validation_metrics(handler, prepared.split)
    return CandidateFit(
        model_handler=handler,
        best_epoch=int(tracker.best_epoch) + 1,
        epochs_completed=int(handler.epoch),
        structure_mae=metrics["structure_mae"],
        temperature_mae=metrics["temperature_mae"],
        integrated_score=metrics["integrated_score"],
    )


def train_fixed_candidate(
    prepared: PreparedFullData,
    hp: NNHyperparameters,
    *,
    seed: int,
    regularization: Regularization | None,
    fixed_epochs: int,
) -> ModelHandler:
    """Train one final model on all rows for the selected fixed epoch count."""

    data_handler = prepared.data_handler
    handler = ModelHandler(
        nn=create_native_nn(data_handler.inputs.shape[1], hp, seed),
        optimizer=Optimizer(
            torch.optim.AdamW,
            lr=float(hp.learning_rate),
            weight_decay=float(hp.weight_decay),
        ),
        data_fitting=DataFitting(data_handler=data_handler, loss_fn=torch.nn.MSELoss()),
        regularization=regularization,
        train_epochs=int(fixed_epochs),
        recompute_validation_loss=False,
        random_seed=int(seed),
        save_result=False,
        save_model=False,
        restore_best=False,
        history_policy="none",
        verbose=False,
    )
    handler()
    return handler


def run_native_search(
    data: pd.DataFrame,
    descriptor_table: pd.DataFrame,
    splits: list[CrossValidationFold],
    config: dict[str, Any],
    *,
    device_name: str,
    batch_size: int = NN_BATCH_SIZE,
    max_epochs: int = NN_MAX_EPOCHS,
    patience: int = NN_EARLY_STOPPING_PATIENCE,
    progress: bool = True,
    existing_results: pd.DataFrame | None = None,
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate every configured native candidate on all fold/seed pairs."""

    candidates = build_native_candidates(config)
    seeds = list(map(int, config["seeds"]["search"]))
    total = len(candidates) * len(splits) * len(seeds)
    expected_keys = {
        (candidate.candidate_id, int(split.fold), seed)
        for candidate in candidates
        for split in splits
        for seed in seeds
    }
    rows, completed_keys = _validated_native_checkpoint(
        existing_results,
        expected_keys=expected_keys,
    )
    completed = len(completed_keys)
    if progress and completed:
        print(f"native search resume: {completed}/{total} fits already complete", flush=True)
    for candidate in candidates:
        for split in splits:
            for seed in seeds:
                key = (candidate.candidate_id, int(split.fold), seed)
                if key in completed_keys:
                    continue
                prepared = prepare_fold_data(
                    data,
                    descriptor_table,
                    blocks=candidate.blocks,
                    split=split,
                    batch_size=int(batch_size),
                    device_name=device_name,
                    random_seed=seed,
                )
                fit = train_native_candidate(
                    prepared,
                    candidate.hyperparameters,
                    seed=seed,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                completed += 1
                rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "selected_blocks": ";".join(candidate.blocks),
                        "fold": split.fold,
                        "temperature_validation_cutoff_K": (
                            split.temperature_validation_cutoff_k
                        ),
                        "seed": seed,
                        "hidden_layers": candidate.hyperparameters.hidden_layers,
                        "hidden_units": candidate.hyperparameters.hidden_units,
                        "learning_rate": candidate.hyperparameters.learning_rate,
                        "weight_decay": candidate.hyperparameters.weight_decay,
                        "n_descriptor_features": len(prepared.selection.selected_columns),
                        "best_epoch": fit.best_epoch,
                        "epochs_completed": fit.epochs_completed,
                        "structure_mae": fit.structure_mae,
                        "temperature_mae": fit.temperature_mae,
                        "integrated_score": fit.integrated_score,
                    }
                )
                completed_keys.add(key)
                table = pd.DataFrame(rows, columns=NATIVE_RESULT_COLUMNS)
                if checkpoint_path is not None:
                    atomic_write_csv(table, checkpoint_path)
                if progress:
                    print(
                        f"native search {completed}/{total}: {candidate.candidate_id}, "
                        f"fold={split.fold}, seed={seed}, score={fit.integrated_score:.6f}",
                        flush=True,
                    )
    return pd.DataFrame(rows, columns=NATIVE_RESULT_COLUMNS)


def _validated_native_checkpoint(
    existing_results: pd.DataFrame | None,
    *,
    expected_keys: set[tuple[str, int, int]],
) -> tuple[list[dict[str, Any]], set[tuple[str, int, int]]]:
    if existing_results is None:
        return [], set()
    missing = set(NATIVE_RESULT_COLUMNS) - set(existing_results.columns)
    if missing:
        raise ValueError(f"Native search checkpoint is missing columns: {sorted(missing)}")
    rows = existing_results.loc[:, list(NATIVE_RESULT_COLUMNS)].to_dict(orient="records")
    keys = {
        (str(row["candidate_id"]), int(row["fold"]), int(row["seed"]))
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("Native search checkpoint contains duplicate fit keys.")
    unexpected = keys - expected_keys
    if unexpected:
        raise ValueError("Native search checkpoint contains unexpected fit keys.")
    return rows, keys


def rank_native_results(
    results: pd.DataFrame,
    *,
    expected_folds: Iterable[int],
    expected_seeds: Iterable[int],
) -> pd.DataFrame:
    """Rank only candidates with complete paired fold/seed coverage."""

    expected = {(int(fold), int(seed)) for fold in expected_folds for seed in expected_seeds}
    rows = []
    for candidate_id, table in results.groupby("candidate_id", sort=False):
        actual = set(zip(table["fold"].astype(int), table["seed"].astype(int), strict=True))
        if len(table) != len(expected) or actual != expected:
            raise ValueError(f"Incomplete fold/seed coverage for {candidate_id}.")
        scores = table["integrated_score"].to_numpy(dtype=float)
        epochs = table["best_epoch"].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or not np.isfinite(epochs).all():
            raise ValueError(f"Non-finite result for {candidate_id}.")
        first = table.iloc[0]
        rows.append(
            {
                "candidate_id": candidate_id,
                "selected_blocks": first["selected_blocks"],
                "hidden_layers": int(first["hidden_layers"]),
                "hidden_units": int(first["hidden_units"]),
                "learning_rate": float(first["learning_rate"]),
                "weight_decay": float(first["weight_decay"]),
                "n_fits": len(table),
                "mean_integrated_score": float(scores.mean()),
                "std_integrated_score": float(scores.std(ddof=0)),
                "worst_integrated_score": float(scores.max()),
                "mean_best_epoch": float(epochs.mean()),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["mean_integrated_score", "worst_integrated_score", "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def _validation_metrics(
    handler: ModelHandler,
    split: CrossValidationFold,
) -> dict[str, float]:
    fitting = handler.data_fitting
    valid = fitting.data_handler.valid
    labels = np.asarray(valid.labels).reshape(-1)
    handler.nn.eval()
    with torch.no_grad():
        prediction = fitting.to_observed_prediction(
            fitting.predict(handler.nn, valid.inputs, phase="valid", epoch=handler.epoch)
        ).reshape(-1)
        observed = fitting.to_observed_target(valid.outputs).reshape(-1)
        error = torch.abs(prediction - observed)
    validation_mae = float(error.mean().item())
    if split.split_strategy == TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY:
        return {
            "structure_mae": float("nan"),
            "temperature_mae": float("nan"),
            "integrated_score": validation_mae,
        }
    structure = torch.as_tensor(
        np.isin(labels, split.structure_validation_indices),
        dtype=torch.bool,
        device=error.device,
    )
    temperature = torch.as_tensor(
        np.isin(labels, split.temperature_validation_indices),
        dtype=torch.bool,
        device=error.device,
    )
    structure_mae = float(error[structure].mean().item())
    temperature_mae = float(error[temperature].mean().item())
    integrated_score = 0.5 * structure_mae + 0.5 * temperature_mae
    return {
        "structure_mae": structure_mae,
        "temperature_mae": temperature_mae,
        "integrated_score": integrated_score,
    }
