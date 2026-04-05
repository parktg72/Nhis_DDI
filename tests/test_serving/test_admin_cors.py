"""
test_admin_cors.py — /admin/reload Pydantic body + CORS 기본값 테스트
"""
from __future__ import annotations

import sys
import types


def _ensure_airflow_mock():
    """airflow 패키지를 최소 Mock으로 대체 (미설치/버전 불일치 환경 대응)."""
    # Remove real airflow modules that may be incompatible
    to_remove = [k for k in list(sys.modules.keys()) if k.startswith("airflow")]
    for k in to_remove:
        del sys.modules[k]

    airflow_mod = types.ModuleType("airflow")

    class MockDAG:
        def __init__(self, dag_id, **kwargs):
            self.dag_id = dag_id
            self.tasks = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    airflow_mod.DAG = MockDAG

    ops = types.ModuleType("airflow.operators")
    python_mod = types.ModuleType("airflow.operators.python")
    empty_mod = types.ModuleType("airflow.operators.empty")

    class MockOperator:
        def __init__(self, task_id, python_callable=None, **kwargs):
            self.task_id = task_id
            self.python_callable = python_callable

        def __rshift__(self, other):
            return other

        def __lshift__(self, other):
            return other

    class BranchPythonOperator(MockOperator):
        pass

    python_mod.PythonOperator = MockOperator
    python_mod.BranchPythonOperator = BranchPythonOperator
    empty_mod.EmptyOperator = MockOperator

    sensors_mod = types.ModuleType("airflow.sensors")
    ext_task_mod = types.ModuleType("airflow.sensors.external_task")

    class MockSensor(MockOperator):
        pass

    ext_task_mod.ExternalTaskSensor = MockSensor

    from datetime import datetime
    utils_mod = types.ModuleType("airflow.utils")
    dates_mod = types.ModuleType("airflow.utils.dates")
    dates_mod.days_ago = lambda n: datetime(2026, 1, 1)
    utils_mod.dates = dates_mod

    for name, mod in [
        ("airflow", airflow_mod),
        ("airflow.operators", ops),
        ("airflow.operators.python", python_mod),
        ("airflow.operators.empty", empty_mod),
        ("airflow.sensors", sensors_mod),
        ("airflow.sensors.external_task", ext_task_mod),
        ("airflow.utils", utils_mod),
        ("airflow.utils.dates", dates_mod),
    ]:
        sys.modules[name] = mod


def test_cors_default_is_not_wildcard():
    """CORS_ORIGINS unset must not default to wildcard."""
    import importlib, sys, os
    env_backup = os.environ.pop("CORS_ORIGINS", None)
    try:
        if "serving.main" in sys.modules:
            del sys.modules["serving.main"]
        import serving.main as m
        assert m._cors_origins_env != "*", "CORS default must not be '*'"
        assert "*" not in m._cors_origins
    finally:
        if env_backup is not None:
            os.environ["CORS_ORIGINS"] = env_backup
        if "serving.main" in sys.modules:
            del sys.modules["serving.main"]


def test_reload_endpoint_uses_body_not_query(tmp_path):
    """Smoke-check that ReloadRequest model exists and has model_path field."""
    from serving.routers.health import ReloadRequest
    r = ReloadRequest(model_path="/app/models/model.pkl")
    assert r.model_path == "/app/models/model.pkl"


def test_deploy_dag_sends_admin_key(monkeypatch, tmp_path):
    """_deploy_model must send X-Admin-Key header."""
    import sys, os
    # 구버전 임시 경로 제거
    sys.path[:] = [p for p in sys.path if "/tmp/codex-review-fixes" not in p]
    _ensure_airflow_mock()

    # Remove cached dag/config modules so they reimport from current source
    for key in list(sys.modules.keys()):
        if "ddi_train_dag" in key or key in ("config.settings", "dags"):
            del sys.modules[key]

    monkeypatch.setenv("ADMIN_API_KEY", "secret-key")
    monkeypatch.setenv("DDI_SERVING_URL", "http://localhost:8000")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))  # _deploy_model이 실제 디렉터리 생성

    import hashlib, pickle as _pickle
    model_file = tmp_path / "model.pkl"
    _model_payload = _pickle.dumps({"trainer_class": "EnsembleTrainer", "weights": (0.5, 0.5)})
    model_file.write_bytes(_model_payload)
    _sha = hashlib.sha256(_model_payload).hexdigest()
    (tmp_path / "model.pkl.sha256").write_text(f"{_sha}  model.pkl\n")
    # sub-model stubs
    for _ext in (".xgb.pkl", ".lgb.pkl"):
        _sub = tmp_path / f"model{_ext}"
        _sub.write_bytes(b"submodel_stub")
        _sub_sha = hashlib.sha256(b"submodel_stub").hexdigest()
        (tmp_path / f"model{_ext}.sha256").write_text(f"{_sub_sha}  model{_ext}\n")

    captured = {}
    from unittest.mock import MagicMock, patch

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["headers"] = headers or {}
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"status": "ok"}
        return r

    class FakeTI:
        def xcom_pull(self, key, task_ids): return str(model_file)

    with patch("requests.post", side_effect=fake_post), \
         patch("shutil.copy2"):
        from dags.ddi_train_dag import _deploy_model
        _deploy_model(ti=FakeTI())

    assert captured.get("headers", {}).get("X-Admin-Key") == "secret-key"
