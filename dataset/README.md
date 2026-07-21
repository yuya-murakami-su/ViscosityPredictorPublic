# Dataset directory

The training data are not distributed with this repository.

## User-provided input

Place the training CSV at `dataset/input/viscosity_data.csv`. The required columns are:

| Column | Description |
| --- | --- |
| `compound_id` | User-defined identifier or display name used for data management |
| `smiles` | Molecular structure represented as SMILES |
| `temperature_K` | Measurement temperature in kelvin |
| `viscosity_cP` | Dynamic viscosity in centipoise (numerically equal to mPa s) |

`compound_id` is used only for user-side identification and data management. It is not used to identify compounds during model training. Compound identity for the training workflow is determined exclusively from the `smiles` column.

SMILES are converted with RDKit to canonical isomeric SMILES. The workflow does not remove salts, neutralize charges, or select a largest fragment.

## Prediction input

Prediction input requires three columns:

| Column | Description |
| --- | --- |
| `compound_id` | User-defined identifier copied to the prediction output |
| `smiles` | Molecular structure represented as SMILES |
| `temperature_K` | Prediction temperature in kelvin |

Viscosity is not supplied for prediction rows.
