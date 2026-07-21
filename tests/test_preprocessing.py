from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from viscosity_predictor.data import (
    CANONICAL_SMILES,
    INVERSE_TEMPERATURE,
    LN_VISCOSITY_PA_S,
)
from viscosity_predictor.preprocessing import (
    fit_descriptor_selection,
    prepare_fold_data,
)
from viscosity_predictor.splits import CrossValidationFold


def test_descriptor_selection_uses_variance_then_spread_ordered_correlation() -> None:
    values = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 1.0],
        ]
    )
    selection = fit_descriptor_selection(values, ["a", "b", "constant"])

    assert selection.selected_columns == ("b",)
    assert selection.high_correlation_columns == ("a",)
    assert selection.low_variance_columns == ("constant",)


def test_prepare_fold_data_fits_selection_and_normalizers_on_train_only() -> None:
    data = pd.DataFrame(
        {
            CANONICAL_SMILES: ["C", "CC", "CCC", "O", "CO", "CCO"],
            INVERSE_TEMPERATURE: [0.0040, 0.0035, 0.0030, 0.0025, 0.0020, 0.0015],
            LN_VISCOSITY_PA_S: [-1.0, 0.0, 1.0, 20.0, 30.0, 40.0],
        }
    )
    descriptors = pd.DataFrame(
        {
            CANONICAL_SMILES: data[CANONICAL_SMILES],
            "physicochemical_descriptors__varying": [0.0, 1.0, 2.0, 100.0, 200.0, 300.0],
            "physicochemical_descriptors__correlated": [0.0, 2.0, 4.0, -100.0, -200.0, -300.0],
            "physicochemical_descriptors__train_constant": [1.0, 1.0, 1.0, 5.0, 6.0, 7.0],
        }
    )
    split = CrossValidationFold(
        fold=0,
        train_indices=np.array([0, 1, 2]),
        validation_indices=np.array([3, 4, 5]),
        structure_validation_indices=np.array([3, 4]),
        temperature_validation_indices=np.array([5]),
        temperature_validation_cutoff_k=400.0,
    )

    prepared = prepare_fold_data(
        data,
        descriptors,
        blocks=["physicochemical_descriptors"],
        split=split,
        batch_size=2,
        device_name="cpu",
        random_seed=1,
    )

    assert prepared.selection.selected_columns == (
        "physicochemical_descriptors__correlated",
    )
    assert prepared.feature_names[-1] == INVERSE_TEMPERATURE
    assert prepared.data_handler.n_data == {"all": 6, "train": 3, "valid": 3, "test": 0}
    expected_input_mean = torch.tensor([[2.0, 0.0035]], dtype=torch.float32)
    expected_output_mean = torch.tensor([[0.0]], dtype=torch.float32)
    assert torch.allclose(prepared.data_handler.input_normalizer.mean.cpu(), expected_input_mean)
    assert torch.allclose(prepared.data_handler.output_normalizer.mean.cpu(), expected_output_mean)
