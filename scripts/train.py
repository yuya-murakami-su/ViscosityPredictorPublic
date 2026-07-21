"""Run the configured public training workflow."""

import argparse
from pathlib import Path

from viscosity_predictor.config import load_training_config
from viscosity_predictor.workflow import run_training_workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "training.toml"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching interrupted run from its saved checkpoints.",
    )
    args = parser.parse_args()
    config = load_training_config(args.config)
    run_training_workflow(config, project_root=PROJECT_ROOT, resume=args.resume)


if __name__ == "__main__":
    main()
