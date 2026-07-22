# ViscosityPredictorPublic Pretrained Models

## Model overview

This release contains two five-member neural-network ensembles for estimating the temperature-dependent dynamic viscosity of pure substances. Users must select the appropriate model explicitly. The software does not inspect a query structure or automatically determine whether that structure was represented in the training dataset.

| CLI name | Directory | Intended use | Model-selection validation |
|---|---|---|---|
| `seen-structure` | `seen/` | Substances whose molecular structure was represented in the training dataset | Five-fold temperature-interpolation validation within represented structures |
| `unseen-structure` | `unseen/` | Substances whose molecular structure was not represented in the training dataset | Five-fold integrated validation combining similarity-graph structure holdout and upper-temperature holdout |

The `seen-structure` model may be used for a represented substance at a temperature outside that substance's measured range. Such a query is still a temperature extrapolation and should not be interpreted as an interpolation merely because the structure is known.

## Training data

Both ensembles were trained exclusively on the viscosity data published in the Supporting Information of:

> Chew, A. K., Sender, M., Kaplan, Z., et al. "Advancing material property prediction: using physics-informed machine learning models for viscosity." *Journal of Cheminformatics* **16**, 31 (2024). https://doi.org/10.1186/s13321-024-00820-5

The effective preprocessed dataset contained 3,544 temperature-dependent observations for 957 unique canonical molecular structures. Its observed temperature range was 227.45-404.10 K. The regularization domain used during training was 200-450 K.

The publication makes this public dataset subset available under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

No source viscosity measurements, compound table, or training CSV are distributed with this repository. Users should consult the cited publication and its Supporting Information for the source data and comply with that license.

## Inputs and outputs

Required input columns are:

- `compound_id`: a user-managed row label;
- `smiles`: a molecular SMILES string;
- `temperature_K`: absolute temperature in kelvin.

Compound identity and molecular descriptors are generated from RDKit canonical isomeric SMILES. Each model uses its saved descriptor selection, feature order, input normalization, network architecture, and output normalization. The ensemble predicts the natural logarithm of viscosity in Pa s and reports converted viscosity values in Pa s and cP.

## Model details

### Seen-structure model

- Selection strategy: temperature interpolation within represented structures
- Descriptor blocks: physicochemical descriptors, complexity-related indices, and E-state indices
- Final input dimension: 17, including inverse temperature
- Network: three hidden layers, 256 units per layer, tanh activation
- Regularization: Soft Arrhenius coefficient 0.1; Hessian coefficient 0
- Fixed final training epochs: 342
- Ensemble seeds: 1, 2, 3, 4, and 5
- Training fingerprint: `9832e19721abcfb1b7875e90e91b1ca24819293584c05ad324759a910d4dffd3`

### Unseen-structure model

- Selection strategy: integrated structure and high-temperature validation
- Descriptor blocks: physicochemical descriptors, Magpie elemental properties, topological indices, and E-state indices
- Final input dimension: 67, including inverse temperature
- Network: three hidden layers, 512 units per layer, tanh activation
- Regularization: Hessian coefficient 10; Soft Arrhenius coefficient 0.1
- Fixed final training epochs: 169
- Ensemble seeds: 1, 2, 3, 4, and 5
- Training fingerprint: `fdc43fdbbd92610d0dcde4ec5e3d9152ee544c28ce6c6322c6dc39f72b0e8a27`

The final ensemble members were refitted on all available training data after model selection. Consequently, model-selection validation results are not independent test results for the final refitted weight files.

## Intended use

The models are intended for research use involving single, neutral pure substances at a single temperature. They may support exploratory property estimation, model comparison, and reproducibility studies.

Use `seen-structure` only when the user has independently established that the queried molecular structure was represented in the training dataset. Use `unseen-structure` when the molecular structure was not represented or when its status is uncertain.

## Limitations and out-of-scope uses

- The software does not determine whether the selected model is appropriate for a query.
- Predictions for unseen structures and temperatures outside the observed data range are extrapolations and generally carry greater uncertainty.
- The models do not determine whether a substance is liquid at the requested temperature.
- Mixtures, salts, multi-component SMILES, polymers, reactions, and safety-critical calculations are outside the intended scope.
- Ensemble-member dispersion is not a calibrated prediction interval.
- Predictions should not replace experimental measurements in process design, safety assessment, regulatory work, or other high-consequence decisions.
- Performance reported for a different dataset, split, or model must not be attributed to these released weights.

## License

All artifacts in `pretrained_model/`, including the model metadata and weight files, are distributed under [CC BY-NC 4.0](LICENSE.md). Commercial use of these pretrained model artifacts is not permitted under this license. The repository's source code is separately distributed under the MIT License.

Artifact paths and SHA-256 digests are recorded in `manifest.json`.

Run `python scripts/verify_pretrained_models.py` from the repository root to verify every digest and load each weight file with PyTorch's restricted `weights_only=True` mode.
