"""Predict viscosity from a saved ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_MODEL_ROOT = PROJECT_ROOT / "pretrained_model"
PRETRAINED_MODEL_MANIFEST = PRETRAINED_MODEL_ROOT / "manifest.json"
PRETRAINED_MODEL_CHOICES = ("seen-structure", "unseen-structure")
PRETRAINED_MANIFEST_FORMAT_VERSION = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/input/prediction_data.csv")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--pretrained-model",
        choices=PRETRAINED_MODEL_CHOICES,
        help=(
            "Select a published model explicitly: use 'seen-structure' when the "
            "substance was represented in the training dataset, or 'unseen-structure' "
            "for a new or uncertain molecular structure."
        ),
    )
    model_group.add_argument(
        "--model",
        help="Path to a custom model bundle containing metadata.json and models/.",
    )
    parser.add_argument("--output", default="outputs/predictions.csv")
    args = parser.parse_args()
    model_path = (
        resolve_pretrained_model(args.pretrained_model)
        if args.pretrained_model is not None
        else _resolve(args.model)
    )
    from viscosity_predictor.prediction import predict_csv
    from viscosity_predictor.runtime import auto_device_name

    predictions = predict_csv(
        _resolve(args.input),
        model_path,
        _resolve(args.output),
        device_name=auto_device_name(),
    )
    print(f"Saved {len(predictions)} predictions to {_resolve(args.output)}", flush=True)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_pretrained_model(
    model_name: str,
    *,
    manifest_path: str | Path = PRETRAINED_MODEL_MANIFEST,
) -> Path:
    """Resolve one explicitly selected published model from its manifest."""

    manifest = Path(manifest_path).resolve()
    registry = json.loads(manifest.read_text(encoding="utf-8"))
    if registry.get("format_version") != PRETRAINED_MANIFEST_FORMAT_VERSION:
        raise ValueError("Unsupported pretrained-model manifest format.")
    try:
        relative_path = Path(registry["models"][model_name]["path"])
    except KeyError as error:
        raise ValueError(f"Unknown pretrained model: {model_name!r}") from error
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Pretrained-model paths must stay within pretrained_model/.")
    root = manifest.parent.resolve()
    model_path = (root / relative_path).resolve()
    if not model_path.is_relative_to(root):
        raise ValueError("Pretrained-model paths must stay within pretrained_model/.")
    return model_path


if __name__ == "__main__":
    main()
