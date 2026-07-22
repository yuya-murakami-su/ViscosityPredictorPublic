"""Verify the integrity and safe loadability of published model artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_ROOT = PROJECT_ROOT / "pretrained_model"
MANIFEST_PATH = PRETRAINED_ROOT / "manifest.json"
MANIFEST_FORMAT_VERSION = 1


def main() -> None:
    registry = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if registry.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError("Unsupported pretrained-model manifest format.")

    verified_models = 0
    verified_artifacts = 0
    for model_name, record in registry["models"].items():
        model_dir = _resolve_model_directory(record["path"])
        artifact_paths = set(record["artifacts"])
        for relative_path, expected_sha256 in record["artifacts"].items():
            artifact = _resolve_artifact(model_dir, relative_path)
            actual_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(f"SHA-256 mismatch: {artifact}")
            verified_artifacts += 1

        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        _validate_metadata(model_name, metadata, record, artifact_paths)
        _validate_members(model_name, model_dir, metadata)
        verified_models += 1

    print(
        f"Verified {verified_models} pretrained models and "
        f"{verified_artifacts} manifest artifacts.",
        flush=True,
    )


def _resolve_model_directory(value: str) -> Path:
    relative_path = Path(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Pretrained-model paths must stay within pretrained_model/.")
    model_dir = (PRETRAINED_ROOT / relative_path).resolve()
    if not model_dir.is_relative_to(PRETRAINED_ROOT.resolve()):
        raise ValueError("Pretrained-model paths must stay within pretrained_model/.")
    if not model_dir.is_dir():
        raise ValueError(f"Missing pretrained-model directory: {model_dir}")
    return model_dir


def _resolve_artifact(model_dir: Path, value: str) -> Path:
    relative_path = Path(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Artifact paths must stay within their model directory.")
    artifact = (model_dir / relative_path).resolve()
    if not artifact.is_relative_to(model_dir):
        raise ValueError("Artifact paths must stay within their model directory.")
    if not artifact.is_file():
        raise ValueError(f"Missing manifest artifact: {artifact}")
    return artifact


def _validate_metadata(
    model_name: str,
    metadata: dict,
    record: dict,
    artifact_paths: set[str],
) -> None:
    if metadata.get("training_fingerprint") != record.get("training_fingerprint"):
        raise ValueError(f"Training fingerprint mismatch: {model_name}")
    feature_names = metadata.get("feature_names", [])
    input_dim = int(metadata.get("network", {}).get("input_dim", -1))
    if input_dim != len(feature_names) or not feature_names:
        raise ValueError(f"Input dimension mismatch: {model_name}")
    if feature_names[-1] != "inverse_temperature":
        raise ValueError(f"Missing inverse-temperature input: {model_name}")
    if len(metadata.get("input_normalizer", {}).get("mean", [[]])[0]) != input_dim:
        raise ValueError(f"Input normalizer mismatch: {model_name}")
    if len(metadata.get("input_normalizer", {}).get("std", [[]])[0]) != input_dim:
        raise ValueError(f"Input normalizer mismatch: {model_name}")

    model_files = metadata.get("model_files", [])
    expected_artifacts = {"metadata.json", *(entry["path"] for entry in model_files)}
    if artifact_paths != expected_artifacts:
        raise ValueError(f"Manifest and metadata artifact lists differ: {model_name}")
    seeds = [int(entry["seed"]) for entry in model_files]
    if not model_files or len(seeds) != len(set(seeds)):
        raise ValueError(f"Invalid ensemble seed list: {model_name}")


def _validate_members(model_name: str, model_dir: Path, metadata: dict) -> None:
    for model_file in metadata["model_files"]:
        artifact = _resolve_artifact(model_dir, model_file["path"])
        bundle = torch.load(artifact, map_location="cpu", weights_only=True)
        if (
            bundle.get("format_version") != metadata.get("format_version")
            or int(bundle.get("seed", -1)) != int(model_file["seed"])
            or bundle.get("training_fingerprint") != metadata.get("training_fingerprint")
            or not bundle.get("model_state_dict")
        ):
            raise ValueError(f"Invalid model member: {model_name}/{model_file['path']}")
        _validate_state_dict_shapes(
            model_name,
            model_file["path"],
            bundle["model_state_dict"],
            metadata["network"],
        )


def _validate_state_dict_shapes(
    model_name: str,
    model_path: str,
    state_dict: dict,
    network: dict,
) -> None:
    input_dim = int(network["input_dim"])
    output_dim = int(network["output_dim"])
    hidden_layers = int(network["hidden_layers"])
    hidden_units = int(network["hidden_units"])
    expected = {}
    previous = input_dim
    for layer in range(hidden_layers):
        module_index = layer * 2
        expected[f"nn.{module_index}.weight"] = (hidden_units, previous)
        expected[f"nn.{module_index}.bias"] = (hidden_units,)
        previous = hidden_units
    output_index = hidden_layers * 2
    expected[f"nn.{output_index}.weight"] = (output_dim, previous)
    expected[f"nn.{output_index}.bias"] = (output_dim,)
    actual = {name: tuple(value.shape) for name, value in state_dict.items()}
    if actual != expected:
        raise ValueError(f"Network shape mismatch: {model_name}/{model_path}")


if __name__ == "__main__":
    main()
