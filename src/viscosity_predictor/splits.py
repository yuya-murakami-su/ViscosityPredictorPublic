"""Reproducible cross-validation splits for model selection."""

from __future__ import annotations

from array import array
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymetis
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .data import CANONICAL_SMILES


MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048
SIMILARITY_EXPONENT = 5.0
WEIGHT_SCALE = 1000
STRUCTURE_FOLD = "structure_fold"
INTEGRATED_SPLIT_STRATEGY = "integrated"
TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY = "temperature_interpolation"
VALID_SPLIT_STRATEGIES = frozenset(
    {INTEGRATED_SPLIT_STRATEGY, TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY}
)


@dataclass(frozen=True)
class CrossValidationFold:
    """Positional row indices and endpoint roles for one validation fold."""

    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    structure_validation_indices: np.ndarray
    temperature_validation_indices: np.ndarray
    temperature_validation_cutoff_k: float | None
    split_strategy: str = INTEGRATED_SPLIT_STRATEGY


def create_cv_splits(
    data: pd.DataFrame,
    *,
    split_strategy: str,
    n_folds: int,
    split_seed: int,
    high_temperature_validation_quantile: float,
) -> list[CrossValidationFold]:
    """Create the selected internal validation split without TOML configuration."""

    strategy = validate_split_strategy(split_strategy)
    if strategy == TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY:
        return create_temperature_interpolation_cv_splits(
            data,
            n_folds=n_folds,
            split_seed=split_seed,
        )
    return create_integrated_cv_splits(
        data,
        n_folds=n_folds,
        split_seed=split_seed,
        high_temperature_validation_quantile=high_temperature_validation_quantile,
    )


def validate_split_strategy(value: str) -> str:
    """Return a normalized supported internal split-strategy name."""

    strategy = str(value).strip().lower()
    if strategy not in VALID_SPLIT_STRATEGIES:
        allowed = ", ".join(sorted(VALID_SPLIT_STRATEGIES))
        raise ValueError(f"split_strategy must be one of: {allowed}.")
    return strategy


def create_temperature_interpolation_cv_splits(
    data: pd.DataFrame,
    *,
    n_folds: int,
    split_seed: int,
) -> list[CrossValidationFold]:
    """Hold out only temperatures strictly inside each compound's observed range."""

    folds = int(n_folds)
    n_rows = int(len(data))
    if folds < 2:
        raise ValueError("n_folds must be at least 2.")
    required = {CANONICAL_SMILES, "temperature_K"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    temperature = pd.to_numeric(data["temperature_K"], errors="coerce")
    if not np.isfinite(temperature).all():
        raise ValueError("temperature_K must contain only finite numeric values.")
    grouped = temperature.groupby(data[CANONICAL_SMILES], sort=False)
    minimum = grouped.transform("min").to_numpy(dtype=float)
    maximum = grouped.transform("max").to_numpy(dtype=float)
    values = temperature.to_numpy(dtype=float)
    eligible = np.flatnonzero((values > minimum) & (values < maximum))
    if len(eligible) < folds:
        raise ValueError(
            "n_folds cannot exceed the number of temperature-interpolation candidates."
        )

    generator = np.random.default_rng(int(split_seed))
    permutation = generator.permutation(eligible)
    fold_by_position = np.full(n_rows, -1, dtype=int)
    fold_by_position[permutation] = np.arange(len(permutation), dtype=int) % folds

    splits = []
    for fold in range(folds):
        validation = fold_by_position == fold
        train = ~validation
        if not train.any() or not validation.any():
            raise ValueError(
                "Each interpolation fold must contain training and validation rows."
            )
        splits.append(
            CrossValidationFold(
                fold=fold,
                train_indices=np.flatnonzero(train),
                validation_indices=np.flatnonzero(validation),
                structure_validation_indices=np.asarray([], dtype=int),
                temperature_validation_indices=np.asarray([], dtype=int),
                temperature_validation_cutoff_k=None,
                split_strategy=TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY,
            )
        )
    return splits


def create_integrated_cv_splits(
    data: pd.DataFrame,
    *,
    n_folds: int,
    split_seed: int,
    high_temperature_validation_quantile: float,
) -> list[CrossValidationFold]:
    """Combine structure folds with fold-specific upper-temperature validation."""

    quantile = float(high_temperature_validation_quantile)
    if not 0.0 < quantile < 1.0:
        raise ValueError("high_temperature_validation_quantile must be between 0 and 1.")
    if "temperature_K" not in data.columns:
        raise ValueError("Missing required column: temperature_K")

    temperatures = pd.to_numeric(data["temperature_K"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(temperatures).all():
        raise ValueError("temperature_K must contain only finite numeric values.")
    structure_folds = assign_similarity_graph_folds(
        data,
        n_folds=n_folds,
        split_seed=split_seed,
    ).to_numpy(dtype=int)

    splits = []
    for fold in range(int(n_folds)):
        structure_validation = structure_folds == fold
        development = ~structure_validation
        cutoff = float(np.quantile(temperatures[development], quantile))
        temperature_validation = development & (temperatures >= cutoff)
        train = development & (temperatures < cutoff)
        validation = structure_validation | temperature_validation

        if not train.any() or not temperature_validation.any():
            raise ValueError(
                "Each fold must contain both training and temperature-validation rows."
            )
        splits.append(
            CrossValidationFold(
                fold=fold,
                train_indices=np.flatnonzero(train),
                validation_indices=np.flatnonzero(validation),
                structure_validation_indices=np.flatnonzero(structure_validation),
                temperature_validation_indices=np.flatnonzero(temperature_validation),
                temperature_validation_cutoff_k=cutoff,
                split_strategy=INTEGRATED_SPLIT_STRATEGY,
            )
        )
    return splits


def assign_similarity_graph_folds(
    data: pd.DataFrame,
    *,
    n_folds: int,
    split_seed: int,
) -> pd.Series:
    """Assign each canonical structure to one balanced PyMetis graph fold."""

    if int(n_folds) < 2:
        raise ValueError("n_folds must be at least 2.")
    if CANONICAL_SMILES not in data.columns:
        raise ValueError(f"Missing required column: {CANONICAL_SMILES}")

    structures = sorted(data[CANONICAL_SMILES].drop_duplicates().astype(str).tolist())
    if int(n_folds) > len(structures):
        raise ValueError("n_folds cannot exceed the number of unique structures.")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_N_BITS,
    )
    fingerprints = []
    for smiles in structures:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"RDKit could not parse canonical SMILES: {smiles!r}")
        fingerprints.append(generator.GetFingerprint(molecule))

    similarity = _tanimoto_similarity_matrix(fingerprints)
    weights = np.power(similarity, SIMILARITY_EXPONENT).astype(np.float32)
    np.fill_diagonal(weights, 0.0)
    xadj, adjncy, eweights = _pymetis_csr(weights)

    partition = pymetis.part_graph(
        int(n_folds),
        adjacency=pymetis.CSRAdjacency(xadj, adjncy),
        eweights=eweights,
        tpwgts=[1.0 / int(n_folds)] * int(n_folds),
        options=pymetis.Options(seed=int(split_seed)),
    )
    fold_by_structure = dict(zip(structures, map(int, partition.vertex_part), strict=True))
    folds = data[CANONICAL_SMILES].astype(str).map(fold_by_structure)
    return folds.astype(int).rename(STRUCTURE_FOLD)


def _tanimoto_similarity_matrix(fingerprints: list[DataStructs.ExplicitBitVect]) -> np.ndarray:
    n_structures = len(fingerprints)
    matrix = np.eye(n_structures, dtype=np.float32)
    for index in range(1, n_structures):
        values = DataStructs.BulkTanimotoSimilarity(
            fingerprints[index],
            fingerprints[:index],
        )
        matrix[index, :index] = values
        matrix[:index, index] = values
    return matrix


def _pymetis_csr(weight_matrix: np.ndarray) -> tuple[array, array, array]:
    xadj = array("i", [0])
    adjncy = array("i")
    eweights = array("i")
    for index in range(len(weight_matrix)):
        weights = np.rint(weight_matrix[index] * WEIGHT_SCALE).astype(np.int32)
        weights[index] = 0
        neighbors = np.flatnonzero(weights > 0)
        adjncy.extend(int(value) for value in neighbors)
        eweights.extend(int(weights[value]) for value in neighbors)
        xadj.append(len(adjncy))
    return xadj, adjncy, eweights
