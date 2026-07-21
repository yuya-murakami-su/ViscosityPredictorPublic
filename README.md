# ViscosityPredictorPublic

This repository provides a compact, reproducible workflow for selecting, training, saving, and using a jointly regularized neural network for pure-component viscosity prediction.

The workflow combines two curvature penalties:

- Hessian regularization along molecular-descriptor directions
- Soft Arrhenius regularization along inverse temperature

## Installation

Python 3.11 or later and Git are required. The recommended environment setup depends on the operating system.

### Windows

Use Conda on Windows because PyMetis may otherwise require a local C/C++ compiler. Install [Miniforge](https://github.com/conda-forge/miniforge) or another Conda distribution, then create and activate an environment as follows:

```bash
conda create -n viscosity_predictor_public -c conda-forge python=3.11 pip git pymetis=2025.2.2 -y
conda activate viscosity_predictor_public
python -m pip install --upgrade pip
```

Install the Conda packages before installing any packages with pip in this environment.

### Linux and macOS

Create and activate a standard virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Optional CUDA acceleration

If you have a CUDA-capable NVIDIA GPU, install the PyTorch build appropriate for your operating system and CUDA environment using the [official PyTorch installation selector](https://docs.pytorch.org/get-started/locally/). Do this after activating the environment and before installing this project. CPU users can skip this step.

### Install the project

Run the following command from the repository root:

```bash
python -m pip install -e .
```

This command installs the project dependencies, including `murakami_lab_modules` from its `v1.1.3` tag. Confirm the installation and check GPU availability with:

```bash
python -c "import torch, pymetis; from importlib.metadata import version; print('PyTorch:', torch.__version__); print('PyMetis:', version('pymetis')); print('murakami_lab_modules:', version('murakami_lab_modules')); print('CUDA available:', torch.cuda.is_available())"
```

The training workflow uses CUDA automatically when this command returns `True`; otherwise it uses the CPU.

## Training data

Training data are not distributed with this repository. Prepare `dataset/input/viscosity_data.csv` with the four columns described in [dataset/README.md](dataset/README.md).

Compound identity is determined exclusively from RDKit canonical isomeric SMILES. `compound_id` is retained only as a user-managed label. Measurements for the same canonical structure within a total temperature width of 0.03 K are represented by their median temperature and median log viscosity.

After this preparation, the workflow reports the effective numbers of rows and unique compounds. It issues a stability warning, without stopping training, when the effective dataset contains 500 or fewer rows or 50 or fewer compounds.

## Configuration

Edit [config/training.toml](config/training.toml) before training. Each setting is documented directly in the TOML file. The main configurable groups are:

- number and seed of cross-validation folds;
- high-temperature validation quantile;
- candidate descriptor blocks;
- native-NN architecture and optimizer search grids;
- Hessian and Soft Arrhenius coefficient grids;
- search and final-ensemble seeds;
- training and collocation budgets.

`collocation_min_K` and `collocation_max_K` define the temperature domain over which curvature regularization is applied. They may extend beyond the observed data to represent the intended application range, but they must cover the complete effective dataset range. The workflow reports both ranges at startup and stops before descriptor calculation or model fitting if the configured collocation range is narrower than the dataset range.

All non-empty combinations of the configured descriptor blocks are evaluated. With the default configuration, the workflow performs:

- 2,790 native-NN validation fits;
- 375 Joint coefficient validation fits;
- 5 final full-data ensemble fits;
- 3,170 model fits in total.

This is computationally demanding. CUDA is selected automatically when available; otherwise CPU is used.

The following descriptor blocks are supported:

| TOML identifier | Descriptor block | Raw dimension |
|---|---|---:|
| `physicochemical_descriptors` | physicochemical descriptors | 16 |
| `structural_counts` | structural counts | 17 |
| `functional_group_counts` | functional-group counts | 85 |
| `topological_indices` | topological indices | 19 |
| `e_state_indices` | e-state indices | 4 |
| `morgan_fingerprint` | Morgan fingerprint | 2,048 |
| `complexity_related_indices` | Complexity-related indices | 7 |
| `magpie_elemental_properties` | Magpie elemental properties | 132 |
| `vsa_descriptors` | VSA descriptors | 57 |

Joback–Reid group descriptors are intentionally omitted from this public release to avoid uncertainty about redistribution rights for the JRgui-derived SMARTS definitions used in the research implementation.

The default configuration uses five candidate blocks, giving 31 non-empty subsets. Users may add any of the other supported identifiers. Selecting all nine listed blocks at once produces 511 descriptor subsets and greatly increases the exhaustive-search cost.

## Model selection and training

Run the full workflow from the repository root:

```bash
python scripts/train.py
```

Use an alternative TOML file with:

```bash
python scripts/train.py --config path/to/training.toml
```

The configured output directory must be empty when starting a new run. After every completed model fit, the search table is saved using an atomic file replacement. If training is interrupted, resume it with:

```bash
python scripts/train.py --resume
```

When using an alternative configuration file, provide it again during resumption:

```bash
python scripts/train.py --config path/to/training.toml --resume
```

Resumption is allowed only when the input CSV, training configuration, workflow source code, dependency versions, operating system, and compute device match the original run manifest. If any condition differs, the workflow stops without combining the results and asks for a new output directory. Fits already recorded in the search CSV files and completed final ensemble members are skipped; only the fit that was running at the moment of interruption is repeated.

For each cross-validation fold, one Similarity-graph partition is used as structure validation. The temperature cutoff is then calculated from the remaining structures only, and observations at or above the configured quantile are used as temperature validation. The default quantile is 0.90, corresponding to the upper 10% of the non-structure-validation rows. The actual cutoff, validation size, and number of represented compounds are reported for every fold. Structure and temperature validation MAEs receive equal weight.

The workflow first selects the descriptor subset and native-NN hyperparameters. It then evaluates the full Cartesian product of the configured Hessian and Soft Arrhenius coefficients. The final epoch count is the rounded arithmetic mean of the selected condition's fold/seed best epochs. Final ensemble members are trained on all available data.

Results are written to the configured output directory, `outputs/training` by default:

```text
outputs/training/
├── run_manifest.json
├── native_search_results.csv
├── native_search_ranking.csv
├── joint_search_results.csv
├── joint_search_ranking.csv
├── workflow_summary.json
└── final_model/
    ├── metadata.json
    └── models/
        ├── seed_1.pt
        └── ...
```

The run manifest protects resumed calculations from incompatible conditions. The model metadata records the workflow configuration, selected descriptor columns, train-fitted filters and normalizers, NN architecture, regularization coefficients, collocation settings, fixed epoch count, ensemble seeds, and dependency versions.

## Prediction

Prepare a CSV with `compound_id`, `smiles`, and `temperature_K`, then specify the model bundle produced by the training workflow:

```bash
python scripts/predict.py \
  --input path/to/prediction_data.csv \
  --model outputs/training/final_model \
  --output outputs/predictions.csv
```

A model bundle must contain `metadata.json` and the referenced `models/seed_*.pt` ensemble members.

The output contains each seed's prediction, the arithmetic ensemble mean in `ln(viscosity_Pa_s)`, the sample standard deviation across seeds, and the corresponding predictions in Pa s and cP. The same canonicalization, descriptor order, feature filter, and normalizers saved during training are reapplied automatically.

## Data and generated artifacts

The `dataset/input` and `outputs` directories are excluded from Git by default. Training data, search results, rankings, run manifests, workflow summaries, and trained model bundles are not distributed with this repository.

## License

The source code is available under the [MIT License](LICENSE).
