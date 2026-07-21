from __future__ import annotations

import numpy as np
import pytest

from viscosity_predictor.data import CANONICAL_SMILES
from viscosity_predictor.descriptors import (
    DESCRIPTOR_BLOCK_LABELS,
    SUPPORTED_BLOCKS,
    calculate_descriptor_table,
    descriptor_block_label,
    descriptor_columns,
    descriptor_subsets,
    finite_descriptor_columns,
)


EXPECTED_BLOCK_DIMENSIONS = {
    "physicochemical_descriptors": 16,
    "structural_counts": 17,
    "functional_group_counts": 85,
    "topological_indices": 19,
    "e_state_indices": 4,
    "morgan_fingerprint": 2048,
    "complexity_related_indices": 7,
    "magpie_elemental_properties": 132,
    "vsa_descriptors": 57,
    "rdkitjs_physicochemical_descriptors": 11,
    "rdkitjs_structural_counts": 13,
    "rdkitjs_topological_indices": 13,
}


def test_descriptor_subsets_enumerates_all_nonempty_combinations() -> None:
    subsets = descriptor_subsets(SUPPORTED_BLOCKS)

    assert len(subsets) == 4095
    assert subsets[0] == ("physicochemical_descriptors",)
    assert subsets[-1] == SUPPORTED_BLOCKS
    assert len(set(subsets)) == 4095


def test_descriptor_block_labels_match_the_manuscript_names() -> None:
    assert [descriptor_block_label(block) for block in SUPPORTED_BLOCKS] == list(
        DESCRIPTOR_BLOCK_LABELS.values()
    )


@pytest.mark.parametrize("block, expected_dimension", EXPECTED_BLOCK_DIMENSIONS.items())
def test_descriptor_block_dimensions_are_stable(block: str, expected_dimension: int) -> None:
    table = calculate_descriptor_table(["CCO"], blocks=[block])
    columns = descriptor_columns(table, [block])

    assert table[CANONICAL_SMILES].tolist() == ["CCO"]
    assert len(columns) == expected_dimension
    assert np.isfinite(table[columns].to_numpy(dtype=float)).all()


def test_descriptor_table_calculates_each_unique_structure_once() -> None:
    table = calculate_descriptor_table(
        ["CCO", "O", "CCO"],
        blocks=["physicochemical_descriptors", "e_state_indices"],
    )
    kept, removed = finite_descriptor_columns(table)

    assert table[CANONICAL_SMILES].tolist() == ["CCO", "O"]
    assert len(kept) == 20
    assert removed == []


def test_rdkitjs_descriptor_blocks_match_expected_contract() -> None:
    blocks = [
        "rdkitjs_physicochemical_descriptors",
        "rdkitjs_structural_counts",
        "rdkitjs_topological_indices",
    ]
    suffixes = [
        "exactmw",
        "amw",
        "NumRotatableBonds",
        "NumHBD",
        "NumHeavyAtoms",
        "FractionCSP3",
        "NumRings",
        "labuteASA",
        "tpsa",
        "CrippenClogP",
        "CrippenMR",
        "NumHeteroatoms",
        "NumAmideBonds",
        "NumAromaticRings",
        "NumAliphaticRings",
        "NumSaturatedRings",
        "NumHeterocycles",
        "NumAromaticHeterocycles",
        "NumSaturatedHeterocycles",
        "NumAliphaticHeterocycles",
        "NumSpiroAtoms",
        "NumBridgeheadAtoms",
        "NumAtomStereoCenters",
        "NumUnspecifiedAtomStereoCenters",
        "chi0v",
        "chi1v",
        "chi3v",
        "chi4v",
        "chi0n",
        "chi1n",
        "chi3n",
        "chi4n",
        "hallKierAlpha",
        "kappa1",
        "kappa2",
        "kappa3",
        "Phi",
    ]
    expected_columns = [
        *(f"rdkitjs_physicochemical_descriptors__{name}" for name in suffixes[:11]),
        *(f"rdkitjs_structural_counts__{name}" for name in suffixes[11:24]),
        *(f"rdkitjs_topological_indices__{name}" for name in suffixes[24:]),
    ]
    expected_values = np.array(
        [
            46.041864812,
            46.069,
            0.0,
            1.0,
            3.0,
            1.0,
            0.0,
            19.89842689442217,
            20.23,
            -0.0014,
            12.7598,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.1543203766865053,
            1.0233345472033855,
            0.0,
            0.0,
            2.1543203766865053,
            1.0233345472033855,
            0.0,
            0.0,
            -0.04,
            2.96,
            1.96,
            1.96,
            1.9338666666666668,
        ]
    )

    table = calculate_descriptor_table(["CCO"], blocks=blocks)
    columns = descriptor_columns(table, blocks)

    assert columns == expected_columns
    np.testing.assert_allclose(
        table.loc[0, columns].to_numpy(dtype=float),
        expected_values,
        rtol=0.0,
        atol=1e-12,
    )
