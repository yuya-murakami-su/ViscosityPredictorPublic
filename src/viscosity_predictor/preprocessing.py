"""Train-fitted feature selection and DataHandler construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from murakami_lab_modules.data import DataHandler

from .data import CANONICAL_SMILES, INVERSE_TEMPERATURE, LN_VISCOSITY_PA_S
from .descriptors import descriptor_columns
from .splits import CrossValidationFold


VARIANCE_THRESHOLD = 1.0e-12
CORRELATION_THRESHOLD = 0.90
SPREAD_QUANTILE_LOW = 0.05
SPREAD_QUANTILE_HIGH = 0.95


@dataclass(frozen=True)
class DescriptorSelection:
    """Descriptor columns retained by train-only unsupervised filtering."""

    input_columns: tuple[str, ...]
    selected_indices: tuple[int, ...]
    selected_columns: tuple[str, ...]
    low_variance_columns: tuple[str, ...]
    high_correlation_columns: tuple[str, ...]


@dataclass(frozen=True)
class PreparedFoldData:
    """One CV fold after train-only feature selection and normalization."""

    data_handler: DataHandler
    selection: DescriptorSelection
    feature_names: tuple[str, ...]
    split: CrossValidationFold


@dataclass(frozen=True)
class PreparedFullData:
    """All rows prepared for fixed-epoch final training."""

    data_handler: DataHandler
    selection: DescriptorSelection
    feature_names: tuple[str, ...]


def fit_descriptor_selection(
    train_descriptors: np.ndarray,
    columns: Sequence[str],
) -> DescriptorSelection:
    """Apply the research variance and correlation filters using train rows only."""

    values = np.asarray(train_descriptors, dtype=float)
    names = tuple(map(str, columns))
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("Descriptor values and column names have incompatible shapes.")
    if not np.isfinite(values).all():
        raise ValueError("Training descriptors contain NaN or inf values.")

    variance = np.var(values, axis=0)
    spread = (
        np.quantile(values, SPREAD_QUANTILE_HIGH, axis=0)
        - np.quantile(values, SPREAD_QUANTILE_LOW, axis=0)
    )
    candidates = np.flatnonzero(variance > VARIANCE_THRESHOLD)
    selected = _low_correlation_representatives(values, candidates, spread)
    selected_set = set(selected)
    low_variance = tuple(names[index] for index in range(len(names)) if index not in candidates)
    high_correlation = tuple(
        names[index]
        for index in candidates
        if int(index) not in selected_set
    )
    return DescriptorSelection(
        input_columns=names,
        selected_indices=tuple(sorted(selected)),
        selected_columns=tuple(names[index] for index in sorted(selected)),
        low_variance_columns=low_variance,
        high_correlation_columns=high_correlation,
    )


def prepare_fold_data(
    data: pd.DataFrame,
    descriptor_table: pd.DataFrame,
    *,
    blocks: Sequence[str],
    split: CrossValidationFold,
    batch_size: int | None,
    device_name: str,
    random_seed: int,
) -> PreparedFoldData:
    """Create an MLM DataHandler whose filters and normalizers use train only."""

    columns = descriptor_columns(descriptor_table, blocks)
    indexed = descriptor_table.set_index(CANONICAL_SMILES, verify_integrity=True)
    descriptor_matrix = indexed.loc[data[CANONICAL_SMILES], columns].to_numpy(dtype=float)
    selection = fit_descriptor_selection(
        descriptor_matrix[split.train_indices],
        columns,
    )
    selected = descriptor_matrix[:, selection.selected_indices]
    inverse_temperature = data[INVERSE_TEMPERATURE].to_numpy(dtype=float).reshape(-1, 1)
    inputs = np.hstack([selected, inverse_temperature]).astype(np.float32, copy=False)
    outputs = data[LN_VISCOSITY_PA_S].to_numpy(dtype=np.float32).reshape(-1, 1)

    handler = DataHandler.from_tensors(
        inputs=inputs,
        outputs=outputs,
        labels=np.arange(len(data), dtype=int).reshape(-1, 1),
        batch_size=batch_size,
        device_name=device_name,
        split_type="index_split",
        train_indices=split.train_indices,
        valid_indices=split.validation_indices,
        random_seed=int(random_seed),
        normalize_output=True,
    )
    return PreparedFoldData(
        data_handler=handler,
        selection=selection,
        feature_names=selection.selected_columns + (INVERSE_TEMPERATURE,),
        split=split,
    )


def prepare_full_data(
    data: pd.DataFrame,
    descriptor_table: pd.DataFrame,
    *,
    blocks: Sequence[str],
    batch_size: int | None,
    device_name: str,
    random_seed: int,
) -> PreparedFullData:
    """Fit filtering and normalization on every row for final model training."""

    columns = descriptor_columns(descriptor_table, blocks)
    indexed = descriptor_table.set_index(CANONICAL_SMILES, verify_integrity=True)
    descriptor_matrix = indexed.loc[data[CANONICAL_SMILES], columns].to_numpy(dtype=float)
    selection = fit_descriptor_selection(descriptor_matrix, columns)
    selected = descriptor_matrix[:, selection.selected_indices]
    inverse_temperature = data[INVERSE_TEMPERATURE].to_numpy(dtype=float).reshape(-1, 1)
    inputs = np.hstack([selected, inverse_temperature]).astype(np.float32, copy=False)
    outputs = data[LN_VISCOSITY_PA_S].to_numpy(dtype=np.float32).reshape(-1, 1)
    train_indices = np.arange(len(data), dtype=int)

    handler = DataHandler.from_tensors(
        inputs=inputs,
        outputs=outputs,
        labels=train_indices.reshape(-1, 1),
        batch_size=batch_size,
        device_name=device_name,
        split_type="index_split",
        train_indices=train_indices,
        valid_indices=np.asarray([], dtype=int),
        random_seed=int(random_seed),
        normalize_output=True,
    )
    return PreparedFullData(
        data_handler=handler,
        selection=selection,
        feature_names=selection.selected_columns + (INVERSE_TEMPERATURE,),
    )


def _low_correlation_representatives(
    values: np.ndarray,
    candidate_indices: np.ndarray,
    spread: np.ndarray,
) -> list[int]:
    if len(candidate_indices) == 0:
        return []

    correlation = np.corrcoef(values[:, candidate_indices], rowvar=False)
    correlation = np.atleast_2d(correlation)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    order = sorted(
        range(len(candidate_indices)),
        key=lambda local: (-float(spread[candidate_indices[local]]), int(candidate_indices[local])),
    )
    selected_local: list[int] = []
    selected_global: list[int] = []
    for local in order:
        if selected_local and np.abs(correlation[local, selected_local]).max() > CORRELATION_THRESHOLD:
            continue
        selected_local.append(local)
        selected_global.append(int(candidate_indices[local]))
    return selected_global
