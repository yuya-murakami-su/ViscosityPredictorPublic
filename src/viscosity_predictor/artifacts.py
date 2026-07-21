"""Atomic artifact writing and safe workflow resumption."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


RUN_MANIFEST_FORMAT_VERSION = 1
RUN_MANIFEST_FILENAME = "run_manifest.json"


def initialize_run_manifest(
    config: dict[str, Any],
    *,
    training_csv: str | Path,
    output_dir: str | Path,
    device_name: str,
    resume: bool,
) -> dict[str, Any]:
    """Create a run manifest or validate it before resuming an existing run."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / RUN_MANIFEST_FILENAME
    contract = _run_contract(config, Path(training_csv), device_name)
    fingerprint = json_fingerprint(contract)

    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume because {manifest_path} does not exist. "
                "Use a new output directory for a fresh run."
            )
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        saved_contract = manifest.get("contract")
        mismatches = [
            key for key, value in contract.items()
            if not isinstance(saved_contract, dict) or saved_contract.get(key) != value
        ]
        if mismatches or manifest.get("fingerprint") != fingerprint:
            details = ", ".join(mismatches or ["fingerprint"])
            raise ValueError(
                "The existing run cannot be resumed because its conditions differ "
                f"from the current run: {details}. Use a new output directory."
            )
        return manifest

    existing = [
        path.name
        for path in destination.iterdir()
        if not path.name.endswith(".tmp")
    ]
    if existing:
        raise FileExistsError(
            f"The output directory already contains training artifacts: {destination}. "
            "Use --resume for the same run or choose a new output directory."
        )

    manifest = {
        "format_version": RUN_MANIFEST_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "training_csv": str(Path(training_csv).resolve()),
        "contract": contract,
    }
    atomic_write_json(manifest, manifest_path)
    return manifest


def atomic_write_csv(table: pd.DataFrame, path: str | Path) -> None:
    """Replace a CSV only after its complete temporary file has been written."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(destination)


def atomic_write_json(values: dict[str, Any], path: str | Path) -> None:
    """Replace a JSON file only after its complete temporary file has been written."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(values, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(destination)


def json_fingerprint(values: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible values."""

    serialized = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _run_contract(
    config: dict[str, Any],
    training_csv: Path,
    device_name: str,
) -> dict[str, Any]:
    if not training_csv.is_file():
        raise FileNotFoundError(f"Training CSV not found: {training_csv}")
    return {
        "format_version": RUN_MANIFEST_FORMAT_VERSION,
        "input_sha256": _file_sha256(training_csv),
        "configuration": {
            section: values
            for section, values in config.items()
            if section != "paths"
        },
        "source_sha256": _source_sha256(),
        "runtime": _runtime_contract(device_name),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_contract(device_name: str) -> dict[str, Any]:
    versions = {
        distribution: importlib.metadata.version(distribution)
        for distribution in (
            "numpy",
            "pandas",
            "torch",
            "murakami-lab-modules",
            "rdkit",
            "matminer",
            "pymetis",
        )
    }
    runtime = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "device": device_name,
        "dependencies": versions,
    }
    if device_name.startswith("cuda"):
        index = torch.cuda.current_device()
        runtime["cuda_device"] = torch.cuda.get_device_name(index)
        runtime["cuda_capability"] = list(torch.cuda.get_device_capability(index))
        runtime["torch_cuda"] = torch.version.cuda
        runtime["cudnn"] = torch.backends.cudnn.version()
    return runtime
