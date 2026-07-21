"""Molecular descriptor blocks used by the public training workflow."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Fragments, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.EState import EState_VSA
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from .data import CANONICAL_SMILES


DESCRIPTOR_BLOCK_LABELS = {
    "physicochemical_descriptors": "physicochemical descriptors",
    "structural_counts": "structural counts",
    "functional_group_counts": "functional-group counts",
    "topological_indices": "topological indices",
    "e_state_indices": "e-state indices",
    "morgan_fingerprint": "Morgan fingerprint",
    "complexity_related_indices": "Complexity-related indices",
    "magpie_elemental_properties": "Magpie elemental properties",
    "vsa_descriptors": "VSA descriptors",
    "rdkitjs_physicochemical_descriptors": "RDKit.js-compatible physicochemical descriptors",
    "rdkitjs_structural_counts": "RDKit.js-compatible structural counts",
    "rdkitjs_topological_indices": "RDKit.js-compatible topological indices",
}
SUPPORTED_BLOCKS = tuple(DESCRIPTOR_BLOCK_LABELS)
MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048


def descriptor_subsets(candidates: Sequence[str]) -> list[tuple[str, ...]]:
    """Enumerate every non-empty descriptor-block subset in stable order."""

    blocks = tuple(candidates)
    _validate_blocks(blocks)
    if len(blocks) != len(set(blocks)):
        raise ValueError("Descriptor candidates must not contain duplicates.")
    return [subset for size in range(1, len(blocks) + 1) for subset in combinations(blocks, size)]


def descriptor_block_label(block: str) -> str:
    """Return the manuscript-facing label for a descriptor block."""

    _validate_blocks((block,))
    return DESCRIPTOR_BLOCK_LABELS[block]


def calculate_descriptor_table(
    canonical_smiles: Iterable[str],
    *,
    blocks: Sequence[str] = SUPPORTED_BLOCKS,
) -> pd.DataFrame:
    """Calculate requested descriptor blocks once for each unique SMILES."""

    selected = tuple(blocks)
    _validate_blocks(selected)
    unique_smiles = list(dict.fromkeys(str(smiles) for smiles in canonical_smiles))
    if not unique_smiles:
        raise ValueError("At least one canonical SMILES is required.")

    matminer_featurizer = (
        _matminer_featurizer()
        if "magpie_elemental_properties" in selected
        else None
    )
    matminer_columns = (
        _matminer_columns(matminer_featurizer)
        if matminer_featurizer is not None
        else []
    )
    rows: list[dict[str, float | int | str]] = []
    for smiles in unique_smiles:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"RDKit could not parse canonical SMILES: {smiles!r}")
        row: dict[str, float | int | str] = {CANONICAL_SMILES: smiles}
        for block in selected:
            row.update(_calculate_block(molecule, block, matminer_featurizer, matminer_columns))
        rows.append(row)

    table = pd.DataFrame(rows)
    columns = descriptor_columns(table)
    values = table[columns].apply(pd.to_numeric, errors="coerce")
    table[columns] = values
    return table


def descriptor_columns(table: pd.DataFrame, blocks: Sequence[str] | None = None) -> list[str]:
    """Return descriptor columns in their stored order."""

    selected = tuple(blocks or SUPPORTED_BLOCKS)
    _validate_blocks(selected)
    prefixes = tuple(f"{block}__" for block in selected)
    return [column for column in table.columns if str(column).startswith(prefixes)]


def finite_descriptor_columns(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return descriptor columns with finite values for every structure."""

    columns = descriptor_columns(table)
    values = table[columns].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=0)
    kept = [column for column, is_finite in zip(columns, finite, strict=True) if is_finite]
    removed = [column for column, is_finite in zip(columns, finite, strict=True) if not is_finite]
    return kept, removed


def _calculate_block(
    molecule: Chem.Mol,
    block: str,
    matminer_featurizer,
    matminer_columns: list[str],
) -> dict[str, float | int]:
    if block == "physicochemical_descriptors":
        return _calculate_physicochemical_descriptors(molecule)
    if block == "structural_counts":
        return _calculate_structural_counts(molecule)
    if block == "functional_group_counts":
        return _calculate_functional_group_counts(molecule)
    if block == "topological_indices":
        return _calculate_topological_indices(molecule)
    if block == "e_state_indices":
        return _calculate_e_state_indices(molecule)
    if block == "morgan_fingerprint":
        return _calculate_morgan_fingerprint(molecule)
    if block == "complexity_related_indices":
        return _calculate_complexity_related_indices(molecule)
    if block == "magpie_elemental_properties":
        return _calculate_magpie_elemental_properties(
            molecule,
            matminer_featurizer,
            matminer_columns,
        )
    if block == "vsa_descriptors":
        return _calculate_vsa_descriptors(molecule)
    if block == "rdkitjs_physicochemical_descriptors":
        return _calculate_rdkitjs_physicochemical_descriptors(molecule)
    if block == "rdkitjs_structural_counts":
        return _calculate_rdkitjs_structural_counts(molecule)
    if block == "rdkitjs_topological_indices":
        return _calculate_rdkitjs_topological_indices(molecule)
    raise ValueError(f"Unsupported descriptor block: {block}")


def _calculate_physicochemical_descriptors(molecule: Chem.Mol) -> dict[str, float]:
    functions: dict[str, Callable] = {
        "MolWt": Descriptors.MolWt,
        "ExactMolWt": Descriptors.ExactMolWt,
        "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt,
        "LogP": Crippen.MolLogP,
        "MolMR": Crippen.MolMR,
        "TPSA": rdMolDescriptors.CalcTPSA,
        "LabuteASA": rdMolDescriptors.CalcLabuteASA,
        "HBD": Lipinski.NumHDonors,
        "HBA": Lipinski.NumHAcceptors,
        "NHOHCount": Lipinski.NHOHCount,
        "NOCount": Lipinski.NOCount,
        "RotBonds": Lipinski.NumRotatableBonds,
        "RingCount": rdMolDescriptors.CalcNumRings,
        "HeavyAtoms": Lipinski.HeavyAtomCount,
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
        "NumValenceElectrons": Descriptors.NumValenceElectrons,
    }
    return {
        f"physicochemical_descriptors__{name}": float(function(molecule))
        for name, function in functions.items()
    }


def _calculate_structural_counts(molecule: Chem.Mol) -> dict[str, float]:
    functions: dict[str, Callable] = {
        "NumAliphaticCarbocycles": Descriptors.NumAliphaticCarbocycles,
        "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles,
        "NumAliphaticRings": Descriptors.NumAliphaticRings,
        "NumAmideBonds": Descriptors.NumAmideBonds,
        "NumAromaticCarbocycles": Descriptors.NumAromaticCarbocycles,
        "NumAromaticHeterocycles": Descriptors.NumAromaticHeterocycles,
        "NumAromaticRings": Descriptors.NumAromaticRings,
        "NumAtomStereoCenters": Descriptors.NumAtomStereoCenters,
        "NumBridgeheadAtoms": Descriptors.NumBridgeheadAtoms,
        "NumHeteroatoms": Descriptors.NumHeteroatoms,
        "NumHeterocycles": Descriptors.NumHeterocycles,
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
        "NumSaturatedCarbocycles": Descriptors.NumSaturatedCarbocycles,
        "NumSaturatedHeterocycles": Descriptors.NumSaturatedHeterocycles,
        "NumSaturatedRings": Descriptors.NumSaturatedRings,
        "NumSpiroAtoms": Descriptors.NumSpiroAtoms,
        "NumUnspecifiedAtomStereoCenters": Descriptors.NumUnspecifiedAtomStereoCenters,
    }
    return {
        f"structural_counts__{name}": float(function(molecule))
        for name, function in functions.items()
    }


def _calculate_functional_group_counts(molecule: Chem.Mol) -> dict[str, float]:
    return {
        f"functional_group_counts__{name}": float(getattr(Fragments, name)(molecule))
        for name in _fragment_names()
    }


def _calculate_topological_indices(molecule: Chem.Mol) -> dict[str, float]:
    functions: dict[str, Callable] = {
        "Chi0": Descriptors.Chi0,
        "Chi1": Descriptors.Chi1,
        "Chi0n": Descriptors.Chi0n,
        "Chi1n": Descriptors.Chi1n,
        "Chi0v": Descriptors.Chi0v,
        "Chi1v": Descriptors.Chi1v,
        "Chi2n": Descriptors.Chi2n,
        "Chi2v": Descriptors.Chi2v,
        "Chi3n": Descriptors.Chi3n,
        "Chi3v": Descriptors.Chi3v,
        "Chi4n": Descriptors.Chi4n,
        "Chi4v": Descriptors.Chi4v,
        "Kappa1": Descriptors.Kappa1,
        "Kappa2": Descriptors.Kappa2,
        "Kappa3": Descriptors.Kappa3,
        "BalabanJ": Descriptors.BalabanJ,
        "BertzCT": Descriptors.BertzCT,
        "HallKierAlpha": Descriptors.HallKierAlpha,
        "Ipc": Descriptors.Ipc,
    }
    return {
        f"topological_indices__{name}": float(function(molecule))
        for name, function in functions.items()
    }


def _calculate_rdkitjs_physicochemical_descriptors(
    molecule: Chem.Mol,
) -> dict[str, float]:
    # NumHBA is intentionally omitted because RDKit.js 2025.03.4 and Python
    # RDKit 2026.03 use different acceptor definitions for some structures.
    functions: dict[str, Callable] = {
        "exactmw": Descriptors.ExactMolWt,
        "amw": Descriptors.MolWt,
        "NumRotatableBonds": Descriptors.NumRotatableBonds,
        "NumHBD": Descriptors.NumHDonors,
        "NumHeavyAtoms": Descriptors.HeavyAtomCount,
        "FractionCSP3": Descriptors.FractionCSP3,
        "NumRings": Descriptors.RingCount,
        "labuteASA": Descriptors.LabuteASA,
        "tpsa": Descriptors.TPSA,
        "CrippenClogP": Descriptors.MolLogP,
        "CrippenMR": Descriptors.MolMR,
    }
    return _calculate_rdkitjs_descriptor_values(
        molecule,
        "rdkitjs_physicochemical_descriptors",
        functions,
    )


def _calculate_rdkitjs_structural_counts(
    molecule: Chem.Mol,
) -> dict[str, float]:
    functions: dict[str, Callable] = {
        "NumHeteroatoms": Descriptors.NumHeteroatoms,
        "NumAmideBonds": Descriptors.NumAmideBonds,
        "NumAromaticRings": Descriptors.NumAromaticRings,
        "NumAliphaticRings": Descriptors.NumAliphaticRings,
        "NumSaturatedRings": Descriptors.NumSaturatedRings,
        "NumHeterocycles": Descriptors.NumHeterocycles,
        "NumAromaticHeterocycles": Descriptors.NumAromaticHeterocycles,
        "NumSaturatedHeterocycles": Descriptors.NumSaturatedHeterocycles,
        "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles,
        "NumSpiroAtoms": Descriptors.NumSpiroAtoms,
        "NumBridgeheadAtoms": Descriptors.NumBridgeheadAtoms,
        "NumAtomStereoCenters": Descriptors.NumAtomStereoCenters,
        "NumUnspecifiedAtomStereoCenters": Descriptors.NumUnspecifiedAtomStereoCenters,
    }
    return _calculate_rdkitjs_descriptor_values(
        molecule,
        "rdkitjs_structural_counts",
        functions,
    )


def _calculate_rdkitjs_topological_indices(molecule: Chem.Mol) -> dict[str, float]:
    # RDKit.js 2025.03.4 exposes chi2v and chi2n keys whose values do not match
    # the corresponding Python RDKit descriptors, so they are intentionally omitted.
    functions: dict[str, Callable] = {
        "chi0v": Descriptors.Chi0v,
        "chi1v": Descriptors.Chi1v,
        "chi3v": Descriptors.Chi3v,
        "chi4v": Descriptors.Chi4v,
        "chi0n": Descriptors.Chi0n,
        "chi1n": Descriptors.Chi1n,
        "chi3n": Descriptors.Chi3n,
        "chi4n": Descriptors.Chi4n,
        "hallKierAlpha": Descriptors.HallKierAlpha,
        "kappa1": Descriptors.Kappa1,
        "kappa2": Descriptors.Kappa2,
        "kappa3": Descriptors.Kappa3,
        "Phi": Descriptors.Phi,
    }
    return _calculate_rdkitjs_descriptor_values(
        molecule,
        "rdkitjs_topological_indices",
        functions,
    )


def _calculate_rdkitjs_descriptor_values(
    molecule: Chem.Mol,
    block: str,
    functions: dict[str, Callable],
) -> dict[str, float]:
    return {
        f"{block}__{name}": float(function(molecule))
        for name, function in functions.items()
    }


def _calculate_e_state_indices(molecule: Chem.Mol) -> dict[str, float]:
    names = ("MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex", "MinEStateIndex")
    return {
        f"e_state_indices__{name}": float(getattr(Descriptors, name)(molecule))
        for name in names
    }


def _calculate_morgan_fingerprint(molecule: Chem.Mol) -> dict[str, int]:
    generator = GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_N_BITS)
    fingerprint = generator.GetFingerprint(molecule)
    values = np.zeros((MORGAN_N_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, values)
    return {
        f"morgan_fingerprint__bit_{index:04d}": int(value)
        for index, value in enumerate(values)
    }


def _calculate_complexity_related_indices(molecule: Chem.Mol) -> dict[str, float]:
    functions: dict[str, Callable] = {
        "AvgIpc": Descriptors.AvgIpc,
        "FpDensityMorgan1": Descriptors.FpDensityMorgan1,
        "FpDensityMorgan2": Descriptors.FpDensityMorgan2,
        "FpDensityMorgan3": Descriptors.FpDensityMorgan3,
        "Phi": Descriptors.Phi,
        "SPS": Descriptors.SPS,
        "qed": QED.qed,
    }
    return {
        f"complexity_related_indices__{name}": float(function(molecule))
        for name, function in functions.items()
    }


def _calculate_magpie_elemental_properties(
    molecule: Chem.Mol,
    featurizer,
    columns: list[str],
) -> dict[str, float]:
    from pymatgen.core import Composition

    formula = rdMolDescriptors.CalcMolFormula(molecule)
    values = featurizer.featurize(Composition(formula))
    return {column: float(value) for column, value in zip(columns, values, strict=True)}


def _matminer_featurizer():
    from matminer.featurizers.composition import ElementProperty

    return ElementProperty.from_preset("magpie")


def _matminer_columns(featurizer) -> list[str]:
    names = [
        f"magpie_elemental_properties__{_sanitize_name(label)}"
        for label in featurizer.feature_labels()
    ]
    unique = _make_unique(names)
    if len(unique) != 132:
        raise RuntimeError(f"Expected 132 Magpie descriptors, got {len(unique)}.")
    return unique


def _calculate_vsa_descriptors(molecule: Chem.Mol) -> dict[str, float]:
    values = {}
    for index in range(1, 15):
        name = f"PEOE_VSA{index}"
        values[f"vsa_descriptors__{name}"] = float(getattr(Descriptors, name)(molecule))
    for index in range(1, 11):
        name = f"SMR_VSA{index}"
        values[f"vsa_descriptors__{name}"] = float(getattr(Descriptors, name)(molecule))
    for index in range(1, 13):
        name = f"SlogP_VSA{index}"
        values[f"vsa_descriptors__{name}"] = float(getattr(Descriptors, name)(molecule))
    for index in range(1, 11):
        name = f"VSA_EState{index}"
        values[f"vsa_descriptors__{name}"] = float(getattr(Descriptors, name)(molecule))
    for index in range(1, 12):
        name = f"EState_VSA{index}"
        values[f"vsa_descriptors__{name}"] = float(getattr(EState_VSA, name)(molecule))
    return values


def _fragment_names() -> list[str]:
    return sorted(
        name
        for name in dir(Fragments)
        if name.startswith("fr_") and callable(getattr(Fragments, name))
    )


def _sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", str(name).strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "feature"


def _make_unique(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    output = []
    for name in names:
        count = counts.get(name, 0)
        output.append(name if count == 0 else f"{name}_{count}")
        counts[name] = count + 1
    return output


def _validate_blocks(blocks: Sequence[str]) -> None:
    unknown = [block for block in blocks if block not in SUPPORTED_BLOCKS]
    if unknown:
        raise ValueError(f"Unsupported descriptor blocks: {unknown}")
