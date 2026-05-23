from __future__ import annotations

import asyncio
import json
import threading

import pandas as pd
import pytest

from serving.dl_predictor import DLModel
from serving.predictor import HybridPredictor
from serving.routers import health


def _make_predictor(*, runtime_lookback_days: int = 365) -> HybridPredictor:
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._dl = DLModel(runtime_lookback_days=runtime_lookback_days)
    pred._dl_lock = threading.RLock()
    return pred


@pytest.mark.filterwarnings("ignore:.*torch\\.jit.*deprecated.*:DeprecationWarning")
def test_smoke_dl_bundle_main_creates_reloadable_predictable_bundle(monkeypatch, tmp_path) -> None:
    pytest.importorskip("torch")
    from scripts.datasets.contracts import (
        DL_BUNDLE_REQUIRED_FILES,
        validate_dl_bundle_manifest,
    )
    from scripts.datasets.smoke_dl_bundle import main

    model_dir = tmp_path / "models"
    bundle = model_dir / "dl" / "smoke"

    assert main([
        str(bundle),
        "--run-id",
        "smoke-test",
        "--schema-version",
        "dl.v1.smoke",
        "--lookback-days",
        "365",
    ]) == 0

    manifest = validate_dl_bundle_manifest(bundle)
    assert manifest["run_id"] == "smoke-test"
    assert manifest["schema_version"] == "dl.v1.smoke"
    assert manifest["lookback_days"] == 365
    assert {path.name for path in bundle.iterdir()} == {
        *DL_BUNDLE_REQUIRED_FILES,
        "MANIFEST.json",
    }
    model_config = json.loads((bundle / "model_config.json").read_text(encoding="utf-8"))
    assert model_config["architecture"] == "linear"

    pred = _make_predictor(runtime_lookback_days=365)
    monkeypatch.setattr(health, "_DL_MODEL_DIR", (model_dir / "dl").resolve())
    monkeypatch.setattr(health, "get_predictor", lambda: pred)

    reload_result = asyncio.run(
        health.reload_dl_model(
            health.DLReloadRequest(bundle_dir=str(bundle)),
            _=None,
        )
    )

    assert reload_result["status"] == "ok"
    assert reload_result["dl_loaded"] is True
    assert reload_result["dl_bundle_run_id"] == "smoke-test"
    assert reload_result["dl_schema_version"] == "dl.v1.smoke"

    result = pred._dl.predict(pd.DataFrame({
        "patient_id": ["P1", "P1", "P1"],
        "drug_code": ["D1", "D2", "UNKNOWN"],
        "prescription_date": ["20260515", "20260516", "20260517"],
    }))

    assert result["run_id"] == "smoke-test"
    assert result["encoding_strategy"] == "multi_hot"
    assert result["predicted_label"] == "high"
    assert result["known_drug_count"] == 2
    assert result["unknown_drug_count"] == 1
    assert set(result["probabilities"]) == {"low", "high"}
