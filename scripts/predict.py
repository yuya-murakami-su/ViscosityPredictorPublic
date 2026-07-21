"""Predict viscosity from a saved ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/input/prediction_data.csv")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a model bundle containing metadata.json and models/.",
    )
    parser.add_argument("--output", default="outputs/predictions.csv")
    args = parser.parse_args()
    from viscosity_predictor.prediction import predict_csv
    from viscosity_predictor.runtime import auto_device_name

    predictions = predict_csv(
        _resolve(args.input),
        _resolve(args.model),
        _resolve(args.output),
        device_name=auto_device_name(),
    )
    print(f"Saved {len(predictions)} predictions to {_resolve(args.output)}", flush=True)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
