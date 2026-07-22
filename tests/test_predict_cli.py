from __future__ import annotations

import hashlib
import json

import pytest

from scripts.predict import (
    PRETRAINED_MODEL_CHOICES,
    PRETRAINED_MODEL_MANIFEST,
    resolve_pretrained_model,
)


def test_published_model_choices_require_manual_structure_selection() -> None:
    assert PRETRAINED_MODEL_CHOICES == ("seen-structure", "unseen-structure")


def test_pretrained_model_manifest_resolves_selected_bundle(tmp_path) -> None:
    root = tmp_path / "pretrained_model"
    bundle = root / "seen"
    bundle.mkdir(parents=True)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "models": {"seen-structure": {"path": "seen"}},
            }
        ),
        encoding="utf-8",
    )

    assert resolve_pretrained_model(
        "seen-structure",
        manifest_path=manifest,
    ) == bundle.resolve()


def test_pretrained_model_manifest_rejects_path_traversal(tmp_path) -> None:
    root = tmp_path / "pretrained_model"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "models": {"seen-structure": {"path": "../private"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stay within"):
        resolve_pretrained_model("seen-structure", manifest_path=manifest)


def test_published_pretrained_model_artifacts_match_manifest() -> None:
    registry = json.loads(PRETRAINED_MODEL_MANIFEST.read_text(encoding="utf-8"))
    for model_name in PRETRAINED_MODEL_CHOICES:
        model_dir = resolve_pretrained_model(model_name)
        record = registry["models"][model_name]
        for relative_path, expected_sha256 in record["artifacts"].items():
            artifact = model_dir / relative_path
            assert artifact.is_file()
            assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected_sha256
