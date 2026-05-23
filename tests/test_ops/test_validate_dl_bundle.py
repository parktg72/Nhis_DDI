from __future__ import annotations

import json
import pickle

from scripts.datasets.contracts import write_dl_bundle_manifest


def _write_bundle(
    root,
    *,
    architecture: str | None = "linear",
    input_dim: int = 3,
    output_labels: list[str] | None = None,
    drug_vocab: dict[str, int] | None = None,
    schema_version: str = "dl.v1",
    schema_sidecar_version: str | None = "dl.v1",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    output_labels = output_labels or ["low", "high"]
    drug_vocab = drug_vocab or {"D1": 0, "D2": 1, "D3": 2}
    model_config = {
        "encoding_strategy": "multi_hot",
        "input_dim": input_dim,
        "output_labels": output_labels,
    }
    if architecture is not None:
        model_config["architecture"] = architecture
    (root / "model.pt").write_bytes(b"fake model")
    (root / "model_config.json").write_text(
        json.dumps(model_config),
        encoding="utf-8",
    )
    (root / "drug_vocab.json").write_text(
        json.dumps(drug_vocab),
        encoding="utf-8",
    )
    (root / "edge_index.pt").write_bytes(b"fake edge index")
    (root / "feature_normalizer.pkl").write_bytes(pickle.dumps({"type": "identity"}))
    sidecar = {"schema_version": schema_sidecar_version} if schema_sidecar_version else {}
    (root / "schema_version.json").write_text(json.dumps(sidecar), encoding="utf-8")
    write_dl_bundle_manifest(
        root,
        run_id="validate-test",
        schema_version=schema_version,
        lookback_days=365,
    )


def test_validate_dl_bundle_accepts_semantically_valid_bundle(tmp_path) -> None:
    from scripts.ops.validate_dl_bundle import validate_bundle

    _write_bundle(tmp_path)

    report = validate_bundle(tmp_path)

    assert report.ok is True
    assert report.errors == []
    assert report.warnings == []


def test_validate_dl_bundle_rejects_drug_vocab_index_out_of_range(tmp_path) -> None:
    from scripts.ops.validate_dl_bundle import validate_bundle

    _write_bundle(tmp_path, drug_vocab={"D1": 0, "D_BAD": 3}, input_dim=3)

    report = validate_bundle(tmp_path)

    assert report.ok is False
    assert any("drug_vocab" in error and "D_BAD" in error for error in report.errors)


def test_validate_dl_bundle_rejects_duplicate_output_labels(tmp_path) -> None:
    from scripts.ops.validate_dl_bundle import validate_bundle

    _write_bundle(tmp_path, output_labels=["high", "high"])

    report = validate_bundle(tmp_path)

    assert report.ok is False
    assert any("output_labels" in error and "duplicate" in error for error in report.errors)


def test_validate_dl_bundle_reports_invalid_input_dim_without_crashing(tmp_path) -> None:
    from scripts.ops.validate_dl_bundle import validate_bundle

    _write_bundle(tmp_path, input_dim=3)
    config_path = tmp_path / "model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["input_dim"] = "not-an-integer"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    report = validate_bundle(tmp_path)

    assert report.ok is False
    assert any("input_dim" in error for error in report.errors)


def test_validate_dl_bundle_warns_for_unknown_architecture(tmp_path) -> None:
    from scripts.ops.validate_dl_bundle import validate_bundle

    _write_bundle(tmp_path, architecture="future-model")

    report = validate_bundle(tmp_path)

    assert report.ok is True
    assert report.errors == []
    assert any("architecture" in warning for warning in report.warnings)


def test_validate_dl_bundle_warns_for_schema_sidecar_mismatch(tmp_path) -> None:
    from scripts.ops.validate_dl_bundle import validate_bundle

    _write_bundle(tmp_path, schema_version="dl.v1", schema_sidecar_version="dl.v2")

    report = validate_bundle(tmp_path)

    assert report.ok is True
    assert report.errors == []
    assert any("schema_version" in warning for warning in report.warnings)


def test_validate_dl_bundle_cli_returns_nonzero_for_errors(tmp_path) -> None:
    from scripts.ops.validate_dl_bundle import main

    _write_bundle(tmp_path, output_labels=["high", "high"])

    assert main([str(tmp_path)]) == 1
