from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from scripts.datasets.contracts import (
    DL_BUNDLE_REQUIRED_FILES,
    write_dl_bundle_manifest,
)
from serving.dl_predictor import DLModel
from serving.predictor import HybridPredictor
from serving.routers import health


def _write_bundle(root, *, run_id: str, lookback_days: int = 365) -> None:
    for name in DL_BUNDLE_REQUIRED_FILES:
        (root / name).write_bytes(f"artifact:{run_id}:{name}".encode("utf-8"))
    write_dl_bundle_manifest(
        root,
        run_id=run_id,
        schema_version="dl.v1",
        lookback_days=lookback_days,
    )


def _make_predictor(*, runtime_lookback_days: int = 365) -> HybridPredictor:
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml = MagicMock()
    pred._ml.loaded = False
    pred._ml._model_type = None
    pred._ml._partition = None
    pred._ml._feature_names = None
    pred._ml._threshold = None
    pred._ml._schema_drift = []
    pred._hierarchical = None
    pred._safety_net = None
    pred._ml_lock = threading.RLock()
    pred._hier_lock = threading.RLock()
    pred._dl_lock = threading.RLock()
    pred._dl = DLModel(runtime_lookback_days=runtime_lookback_days)
    return pred


def test_require_admin_rejects_bad_key(monkeypatch) -> None:
    monkeypatch.setattr(health, "_ADMIN_KEY", "secret")

    with pytest.raises(HTTPException) as exc:
        health._require_admin("bad")

    assert exc.value.status_code == 401


def test_reload_dl_route_loads_valid_bundle(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "models"
    bundle = model_dir / "dl" / "bundle-v1"
    bundle.mkdir(parents=True)
    _write_bundle(bundle, run_id="v1", lookback_days=365)
    pred = _make_predictor(runtime_lookback_days=365)

    monkeypatch.setattr(health, "_DL_MODEL_DIR", (model_dir / "dl").resolve())
    monkeypatch.setattr(health, "get_predictor", lambda: pred)

    result = asyncio.run(
        health.reload_dl_model(
            health.DLReloadRequest(bundle_dir=str(bundle)),
            _=None,
        )
    )

    assert result["status"] == "ok"
    assert result["dl_loaded"] is True
    assert result["dl_bundle_run_id"] == "v1"
    assert pred._dl.loaded
    assert pred._dl.bundle_dir == bundle


def test_reload_dl_route_rejects_path_outside_dl_model_dir(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "models"
    outside = tmp_path / "outside"
    outside.mkdir()

    monkeypatch.setattr(health, "_DL_MODEL_DIR", (model_dir / "dl").resolve())

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            health.reload_dl_model(
                health.DLReloadRequest(bundle_dir=str(outside)),
                _=None,
            )
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["error_code"] == "path_outside_model_dir"


def test_reload_dl_route_maps_lookback_mismatch_and_keeps_previous(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "models"
    good = model_dir / "dl" / "bundle-v1"
    bad = model_dir / "dl" / "bundle-v2"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    _write_bundle(good, run_id="v1", lookback_days=365)
    _write_bundle(bad, run_id="v2", lookback_days=180)
    pred = _make_predictor(runtime_lookback_days=365)
    pred.reload_dl(good)

    monkeypatch.setattr(health, "_DL_MODEL_DIR", (model_dir / "dl").resolve())
    monkeypatch.setattr(health, "get_predictor", lambda: pred)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            health.reload_dl_model(
                health.DLReloadRequest(bundle_dir=str(bad)),
                _=None,
            )
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["error_code"] == "lookback_mismatch"
    assert pred._dl.loaded
    assert pred._dl.bundle_dir == good
    assert pred._dl.run_id == "v1"


def test_health_and_model_info_expose_dl_metadata(monkeypatch, tmp_path) -> None:
    bundle = tmp_path / "models" / "dl" / "bundle-v1"
    bundle.mkdir(parents=True)
    _write_bundle(bundle, run_id="v1", lookback_days=365)
    pred = _make_predictor(runtime_lookback_days=365)
    pred.reload_dl(bundle)

    monkeypatch.setattr(health, "get_predictor", lambda: pred)

    health_body = asyncio.run(health.health_check())
    assert health_body.dl_loaded is True
    assert health_body.dl_bundle_run_id == "v1"
    assert health_body.dl_lookback_days == 365
    assert health_body.dl_schema_version == "dl.v1"

    info_body = asyncio.run(health.model_info())
    assert info_body.model_type == "none"
    assert info_body.dl_loaded is True
    assert info_body.dl_bundle_run_id == "v1"
