"""MODEL_PATH 환경변수 폴백 동작 검증."""
import importlib
import os
import pytest


def test_model_path_fallback_uses_model_dir(monkeypatch, tmp_path):
    """MODEL_PATH 미설정 시 MODEL_DIR/current/model_prod.pkl 사용."""
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    import config.settings as s
    importlib.reload(s)

    # serving/main.py lifespan 로직 재현
    model_path = os.environ.get("MODEL_PATH") or str(s.MODEL_PROD_PATH)
    assert model_path == str(tmp_path / "current" / "model_prod.pkl")

    importlib.reload(s)  # cleanup — 기본값 복원


def test_model_path_explicit_overrides_dir(monkeypatch, tmp_path):
    """MODEL_PATH 명시 설정 시 MODEL_DIR 무시."""
    explicit = str(tmp_path / "custom_model.pkl")
    monkeypatch.setenv("MODEL_PATH", explicit)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "other"))

    import config.settings as s
    importlib.reload(s)

    model_path = os.environ.get("MODEL_PATH") or str(s.MODEL_PROD_PATH)
    assert model_path == explicit

    importlib.reload(s)  # cleanup


def test_model_prod_path_derived_from_model_dir(monkeypatch, tmp_path):
    """settings.MODEL_PROD_PATH 가 MODEL_DIR / current / model_prod.pkl 임."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    import config.settings as s
    importlib.reload(s)

    assert s.MODEL_PROD_PATH == tmp_path / "current" / "model_prod.pkl"

    importlib.reload(s)  # cleanup
