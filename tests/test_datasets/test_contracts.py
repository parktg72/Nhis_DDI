from __future__ import annotations

import json

import pytest

from scripts.datasets.contracts import (
    BundleArtifactEmptyError,
    BundleHashMismatchError,
    DL_BUNDLE_REQUIRED_FILES,
    DL_DATASET_REQUIRED_COLUMNS,
    HASH_ALG_SHA256,
    LOOKBACK_DAYS_DEFAULT,
    LookbackMismatchError,
    ML_DATASET_REQUIRED_COLUMNS,
    validate_dl_bundle_manifest,
    validate_lookback_consistency,
    validate_lookback_days,
    validate_required_columns,
    write_dl_bundle_manifest,
)


def test_ml_and_dl_contracts_are_distinct() -> None:
    assert "drug_count" in ML_DATASET_REQUIRED_COLUMNS
    assert "drug_code" not in ML_DATASET_REQUIRED_COLUMNS
    assert "drug_code" in DL_DATASET_REQUIRED_COLUMNS
    assert "drug_count" not in DL_DATASET_REQUIRED_COLUMNS


def test_validate_required_columns_raises_for_missing_columns() -> None:
    with pytest.raises(ValueError, match="drug_code"):
        validate_required_columns(["patient_id", "prescription_date"], DL_DATASET_REQUIRED_COLUMNS)


def test_dl_bundle_manifest_roundtrip(tmp_path) -> None:
    for name in DL_BUNDLE_REQUIRED_FILES:
        (tmp_path / name).write_bytes(f"artifact:{name}".encode("utf-8"))

    manifest_path = write_dl_bundle_manifest(
        tmp_path,
        run_id="run-001",
        schema_version="dl.v1",
        lookback_days=365,
    )

    manifest = validate_dl_bundle_manifest(tmp_path)
    assert manifest_path.name == "MANIFEST.json"
    assert manifest["track"] == "dl"
    assert manifest["run_id"] == "run-001"
    assert manifest["hash_alg"] == HASH_ALG_SHA256
    assert manifest["lookback_days"] == 365
    assert manifest["created_at"]
    assert manifest["drug_vocab_sha256"] == manifest["files"]["drug_vocab.json"]["sha256"]
    assert manifest["edge_index_sha256"] == manifest["files"]["edge_index.pt"]["sha256"]
    assert set(manifest["files"]) == set(DL_BUNDLE_REQUIRED_FILES)
    assert not (tmp_path / "MANIFEST.json.tmp").exists()


def test_dl_bundle_manifest_rejects_hash_mismatch(tmp_path) -> None:
    for name in DL_BUNDLE_REQUIRED_FILES:
        (tmp_path / name).write_bytes(f"artifact:{name}".encode("utf-8"))
    write_dl_bundle_manifest(tmp_path, run_id="run-001", schema_version="dl.v1")

    (tmp_path / "drug_vocab.json").write_text("changed", encoding="utf-8")

    with pytest.raises(BundleHashMismatchError, match="hash mismatch"):
        validate_dl_bundle_manifest(tmp_path)


def test_dl_bundle_manifest_rejects_wrong_track(tmp_path) -> None:
    for name in DL_BUNDLE_REQUIRED_FILES:
        (tmp_path / name).write_bytes(f"artifact:{name}".encode("utf-8"))
    write_dl_bundle_manifest(tmp_path, run_id="run-001", schema_version="dl.v1")

    manifest_path = tmp_path / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["track"] = "ml"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="track"):
        validate_dl_bundle_manifest(tmp_path)


def test_dl_bundle_manifest_rejects_empty_required_artifact(tmp_path) -> None:
    for name in DL_BUNDLE_REQUIRED_FILES:
        (tmp_path / name).write_bytes(f"artifact:{name}".encode("utf-8"))
    (tmp_path / "model.pt").write_bytes(b"")

    with pytest.raises(BundleArtifactEmptyError, match="empty"):
        write_dl_bundle_manifest(tmp_path, run_id="run-001", schema_version="dl.v1")


def test_dl_bundle_manifest_rejects_top_level_hash_drift(tmp_path) -> None:
    for name in DL_BUNDLE_REQUIRED_FILES:
        (tmp_path / name).write_bytes(f"artifact:{name}".encode("utf-8"))
    write_dl_bundle_manifest(tmp_path, run_id="run-001", schema_version="dl.v1")

    manifest_path = tmp_path / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["edge_index_sha256"] = "bad"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleHashMismatchError, match="edge_index_sha256"):
        validate_dl_bundle_manifest(tmp_path)


def test_dl_bundle_manifest_rejects_missing_run_id(tmp_path) -> None:
    for name in DL_BUNDLE_REQUIRED_FILES:
        (tmp_path / name).write_bytes(f"artifact:{name}".encode("utf-8"))
    write_dl_bundle_manifest(tmp_path, run_id="run-001", schema_version="dl.v1")

    manifest_path = tmp_path / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = ""
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="run_id"):
        validate_dl_bundle_manifest(tmp_path)


def test_validate_lookback_days_range() -> None:
    assert validate_lookback_days(LOOKBACK_DAYS_DEFAULT) == LOOKBACK_DAYS_DEFAULT
    with pytest.raises(ValueError, match="out of range"):
        validate_lookback_days(0)


def test_validate_lookback_consistency_strict_policy() -> None:
    validate_lookback_consistency(365, 365, context="request")
    with pytest.raises(LookbackMismatchError, match="artifact=365, runtime=180"):
        validate_lookback_consistency(365, 180, context="request")
