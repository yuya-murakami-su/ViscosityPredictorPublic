"""Train an internal model for known-compound temperature interpolation."""

from __future__ import annotations

import argparse
from pathlib import Path

from viscosity_predictor.config import load_training_config
from viscosity_predictor.splits import TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY
from viscosity_predictor.workflow import run_training_workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = "outputs/development_interpolation_training"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "training.toml"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_training_config(args.config)
    config["paths"]["output_dir"] = OUTPUT_DIR
    run_training_workflow(
        config,
        project_root=PROJECT_ROOT,
        resume=args.resume,
        split_strategy=TEMPERATURE_INTERPOLATION_SPLIT_STRATEGY,
    )


if __name__ == "__main__":
    main()
