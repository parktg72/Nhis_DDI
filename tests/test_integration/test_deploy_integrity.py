"""배포 원자성·핫스왑 실패 시나리오 통합 테스트."""
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _ensure_airflow_mock():
    """airflow 미설치 환경에서 DAG 모듈 import 가능하도록 mock."""
    if "airflow" not in sys.modules:
        import types
        airflow = types.ModuleType("airflow")

        class DAG:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass

        airflow.DAG = DAG
        sys.modules["airflow"] = airflow
        for sub in ("airflow.operators.python", "airflow.operators.empty",
                    "airflow.sensors.external_task", "airflow.utils.dates"):
            m = types.ModuleType(sub)
            m.PythonOperator = lambda **kw: None
            m.BranchPythonOperator = lambda **kw: None
            m.EmptyOperator = lambda **kw: None
            m.ExternalTaskSensor = lambda **kw: None
            m.days_ago = lambda n: None
            sys.modules[sub] = m


class _FakeTI:
    def __init__(self, model_path: str):
        self._model_path = model_path

    def xcom_pull(self, key, task_ids):
        return self._model_path


def _make_full_artifacts(staging: Path, base_name: str = "model_v1") -> Path:
    """완전한 앙상블 아티팩트 세트 생성."""
    main_pkl = staging / f"{base_name}.pkl"
    main_pkl.write_bytes(b"model")
    (staging / f"{base_name}.pkl.sha256").write_text("aabbcc  model_v1.pkl\n")
    (staging / f"{base_name}.xgb.pkl").write_bytes(b"xgb")
    (staging / f"{base_name}.xgb.pkl.sha256").write_text("aabbcc  model_v1.xgb.pkl\n")
    (staging / f"{base_name}.lgb.pkl").write_bytes(b"lgb")
    (staging / f"{base_name}.lgb.pkl.sha256").write_text("aabbcc  model_v1.lgb.pkl\n")
    return main_pkl


@pytest.fixture(autouse=True)
def _fresh_dag_module():
    """각 테스트마다 DAG 모듈 + config.settings 새로 import."""
    _ensure_airflow_mock()
    for key in list(sys.modules.keys()):
        if "ddi_train_dag" in key or key == "config.settings":
            del sys.modules[key]
    yield
    for key in list(sys.modules.keys()):
        if "ddi_train_dag" in key or key == "config.settings":
            del sys.modules[key]


def test_deploy_atomic_no_files_on_missing_submodel_sha256(tmp_path):
    """서브모델 sha256 누락 시 prod_dir 파일 변경 없음 (원자성 보장)."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    # 서브모델 sha256 하나 누락
    main_pkl = staging / "model_v1.pkl"
    main_pkl.write_bytes(b"model")
    (staging / "model_v1.pkl.sha256").write_text("hash\n")
    (staging / "model_v1.xgb.pkl").write_bytes(b"xgb")
    # model_v1.xgb.pkl.sha256 고의로 누락
    (staging / "model_v1.lgb.pkl").write_bytes(b"lgb")
    (staging / "model_v1.lgb.pkl.sha256").write_text("hash\n")

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"

    from dags.ddi_train_dag import _deploy_model
    with pytest.raises(RuntimeError, match="배포 중단"):
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    assert list(prod_dir.glob("model_prod*")) == [], \
        "RuntimeError 발생 전 파일이 prod_dir에 복사됨 — 원자성 깨짐"


def test_deploy_atomic_no_files_on_missing_main_sha256(tmp_path):
    """메인 sha256 누락 시 prod_dir 파일 변경 없음."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    main_pkl = staging / "model_v1.pkl"
    main_pkl.write_bytes(b"model")
    # model_v1.pkl.sha256 고의로 누락

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"

    from dags.ddi_train_dag import _deploy_model
    with pytest.raises(RuntimeError, match="배포 중단"):
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    assert list(prod_dir.glob("model_prod*")) == []


def test_deploy_success_creates_all_files(tmp_path):
    """완전한 아티팩트로 배포 시 model_prod* 파일 전부 생성."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.ADMIN_API_KEY = "test-key"

    def fake_post(url, **kw):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"status": "ok"}
        return r

    with patch("requests.post", side_effect=fake_post):
        from dags.ddi_train_dag import _deploy_model
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    assert (prod_dir / "model_prod.pkl").exists()
    assert (prod_dir / "model_prod.pkl.sha256").exists()
    assert (prod_dir / "model_prod.xgb.pkl").exists()
    assert (prod_dir / "model_prod.xgb.pkl.sha256").exists()
    assert (prod_dir / "model_prod.lgb.pkl").exists()
    assert (prod_dir / "model_prod.lgb.pkl.sha256").exists()


def test_deploy_backup_covers_all_files(tmp_path):
    """배포 성공 시 backup/ 에 기존 model_prod* 전체 보관."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    # 기존 prod 파일 준비
    (prod_dir / "model_prod.pkl").write_bytes(b"old_model")
    (prod_dir / "model_prod.pkl.sha256").write_text("oldhash\n")
    (prod_dir / "model_prod.xgb.pkl").write_bytes(b"old_xgb")
    (prod_dir / "model_prod.xgb.pkl.sha256").write_text("oldhash\n")

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.ADMIN_API_KEY = "key"

    def fake_post(url, **kw):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"status": "ok"}
        return r

    with patch("requests.post", side_effect=fake_post):
        from dags.ddi_train_dag import _deploy_model
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    backup_dir = prod_dir / "backup"
    assert (backup_dir / "model_prod.pkl").read_bytes() == b"old_model"
    assert (backup_dir / "model_prod.pkl.sha256").exists()
    assert (backup_dir / "model_prod.xgb.pkl").read_bytes() == b"old_xgb"


def test_hotswap_failure_raises(tmp_path):
    """serving reload 503 시 RuntimeError."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()
    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.ADMIN_API_KEY = "key"

    import requests as _req

    def fake_post_503(url, **kw):
        r = MagicMock()
        r.raise_for_status.side_effect = _req.HTTPError("503 Server Error")
        return r

    with patch("requests.post", side_effect=fake_post_503):
        from dags.ddi_train_dag import _deploy_model
        with pytest.raises(RuntimeError, match="핫스왑 실패"):
            _deploy_model(ti=_FakeTI(str(main_pkl)))


def test_hotswap_timeout_raises(tmp_path):
    """serving reload 타임아웃 시 RuntimeError."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()
    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.ADMIN_API_KEY = "key"

    import requests as _req

    def fake_post_timeout(url, **kw):
        raise _req.Timeout("connection timed out")

    with patch("requests.post", side_effect=fake_post_timeout):
        from dags.ddi_train_dag import _deploy_model
        with pytest.raises(RuntimeError, match="핫스왑 실패"):
            _deploy_model(ti=_FakeTI(str(main_pkl)))
