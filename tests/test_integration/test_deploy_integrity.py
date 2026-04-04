"""배포 원자성·핫스왑 실패 시나리오 통합 테스트."""
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _ensure_airflow_mock():
    """airflow 미설치 환경에서 DAG 모듈 import 가능하도록 mock."""
    import types

    # 기존 airflow 모듈 전체 제거 후 재구성
    for key in list(sys.modules.keys()):
        if key.startswith("airflow"):
            del sys.modules[key]

    airflow = types.ModuleType("airflow")

    class DAG:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    airflow.DAG = DAG
    sys.modules["airflow"] = airflow

    class MockOperator:
        def __init__(self, *a, **kw): pass
        def __rshift__(self, other): return other
        def __lshift__(self, other): return other

    for sub in ("airflow.operators.python", "airflow.operators.empty",
                "airflow.sensors.external_task", "airflow.utils.dates"):
        m = types.ModuleType(sub)
        m.PythonOperator = MockOperator
        m.BranchPythonOperator = MockOperator
        m.EmptyOperator = MockOperator
        m.ExternalTaskSensor = MockOperator
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
    """각 테스트마다 DAG 모듈 + config.settings 새로 import.
    /tmp/codex-review-fixes 같은 임시 경로가 sys.path에 있으면 구버전이 로드되므로 제거.
    """
    # 임시 경로 제거 (이전 테스트가 삽입한 구버전 경로)
    sys.path[:] = [p for p in sys.path if "/tmp/codex-review-fixes" not in p]
    _ensure_airflow_mock()
    # dags 네임스페이스 패키지도 제거 — 캐시된 __path__에 구버전 경로 포함 가능
    for key in list(sys.modules.keys()):
        if "ddi_train_dag" in key or key in ("config.settings", "dags"):
            del sys.modules[key]
    yield
    for key in list(sys.modules.keys()):
        if "ddi_train_dag" in key or key in ("config.settings", "dags"):
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
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.SERVING_URLS = ["http://localhost:9999"]

    from dags.ddi_train_dag import _deploy_model
    with pytest.raises(RuntimeError, match="배포 중단"):
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    assert not (prod_dir / "current").exists(), \
        "RuntimeError 발생 전 current 심링크가 생성됨 — 원자성 깨짐"
    assert not list(prod_dir.glob(".v_*")), \
        "RuntimeError 발생 전 버전 디렉터리가 생성됨 — 원자성 깨짐"


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
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.SERVING_URLS = ["http://localhost:9999"]

    from dags.ddi_train_dag import _deploy_model
    with pytest.raises(RuntimeError, match="배포 중단"):
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    assert not (prod_dir / "current").exists()
    assert not list(prod_dir.glob(".v_*"))


def test_deploy_success_creates_all_files(tmp_path):
    """완전한 아티팩트로 배포 시 current 심링크 + 버전 디렉터리에 파일 전부 생성."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.SERVING_URLS = ["http://localhost:9999"]
    s.ADMIN_API_KEY = "test-key"

    def fake_post(url, **kw):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"status": "ok"}
        return r

    with patch("requests.post", side_effect=fake_post):
        from dags.ddi_train_dag import _deploy_model
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    current = prod_dir / "current"
    assert current.is_symlink(), "current 심링크 없음"
    assert (current / "model_prod.pkl").exists()
    assert (current / "model_prod.pkl.sha256").exists()
    assert (current / "model_prod.xgb.pkl").exists()
    assert (current / "model_prod.xgb.pkl.sha256").exists()
    assert (current / "model_prod.lgb.pkl").exists()
    assert (current / "model_prod.lgb.pkl.sha256").exists()


def test_deploy_backup_covers_all_files(tmp_path):
    """배포 성공 시 backup/ 에 기존 current→versioned_dir 파일 전체 보관."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    # 기존 버전 디렉터리 + current 심링크 준비
    old_dir = prod_dir / ".v_old"
    old_dir.mkdir()
    (old_dir / "model_prod.pkl").write_bytes(b"old_model")
    (old_dir / "model_prod.pkl.sha256").write_text("oldhash\n")
    (old_dir / "model_prod.xgb.pkl").write_bytes(b"old_xgb")
    (old_dir / "model_prod.xgb.pkl.sha256").write_text("oldhash\n")
    (prod_dir / "current").symlink_to(".v_old")

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.SERVING_URLS = ["http://localhost:9999"]
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


def test_hotswap_failure_rolls_back_current_symlink(tmp_path):
    """핫스왑 실패 시 current 심링크가 이전 버전으로 복구됨."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    # 기존 버전 디렉터리 + current 심링크 준비
    old_dir = prod_dir / ".v_old"
    old_dir.mkdir()
    (old_dir / "model_prod.pkl").write_bytes(b"old_model")
    (old_dir / "model_prod.pkl.sha256").write_text("oldhash\n")
    (old_dir / "model_prod.xgb.pkl").write_bytes(b"old_xgb")
    (old_dir / "model_prod.xgb.pkl.sha256").write_text("oldhash\n")
    (old_dir / "model_prod.lgb.pkl").write_bytes(b"old_lgb")
    (old_dir / "model_prod.lgb.pkl.sha256").write_text("oldhash\n")
    (prod_dir / "current").symlink_to(".v_old")

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.SERVING_URLS = ["http://localhost:9999"]
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

    current = prod_dir / "current"
    assert current.is_symlink(), "current 심링크 없음"
    assert current.resolve() == old_dir.resolve(), \
        "핫스왑 실패 후 current가 이전 버전(.v_old)으로 복구되지 않음"
    assert (current / "model_prod.pkl").read_bytes() == b"old_model"


def test_hotswap_multi_url_all_called(tmp_path):
    """SERVING_URLS 여러 인스턴스 모두에 reload 요청 전송."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()
    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://inst1:8000"
    s.SERVING_URLS = ["http://inst1:8000", "http://inst2:8000"]
    s.ADMIN_API_KEY = "key"

    called_urls: list = []

    def fake_post(url, **kw):
        called_urls.append(url)
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"status": "ok"}
        return r

    with patch("requests.post", side_effect=fake_post):
        from dags.ddi_train_dag import _deploy_model
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    reload_urls = [u for u in called_urls if "/admin/reload" in u]
    assert "http://inst1:8000/admin/reload" in reload_urls
    assert "http://inst2:8000/admin/reload" in reload_urls


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
    s.SERVING_URLS = ["http://localhost:9999"]
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
    s.SERVING_URLS = ["http://localhost:9999"]
    s.ADMIN_API_KEY = "key"

    import requests as _req

    def fake_post_timeout(url, **kw):
        raise _req.Timeout("connection timed out")

    with patch("requests.post", side_effect=fake_post_timeout):
        from dags.ddi_train_dag import _deploy_model
        with pytest.raises(RuntimeError, match="핫스왑 실패"):
            _deploy_model(ti=_FakeTI(str(main_pkl)))


def test_prune_keeps_backup_keep_n_versioned_dirs(tmp_path):
    """배포 후 .v_* 디렉터리가 BACKUP_KEEP_N 개만 남고 오래된 것은 삭제됨."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    # 기존 .v_* 디렉터리 6개 생성 (ts=1~6)
    for i in range(1, 7):
        d = prod_dir / f".v_{i}"
        d.mkdir()
        (d / "model_prod.pkl").write_bytes(b"old")
        (d / "model_prod.pkl.sha256").write_text("hash\n")
        (d / "model_prod.xgb.pkl").write_bytes(b"old_xgb")
        (d / "model_prod.xgb.pkl.sha256").write_text("hash\n")
        (d / "model_prod.lgb.pkl").write_bytes(b"old_lgb")
        (d / "model_prod.lgb.pkl.sha256").write_text("hash\n")
    # current는 가장 최신(.v_6) 가리킴
    (prod_dir / "current").symlink_to(".v_6")

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.SERVING_URLS = ["http://localhost:9999"]
    s.ADMIN_API_KEY = "key"
    s.BACKUP_KEEP_N = 3  # 최신 3개만 유지

    def fake_post(url, **kw):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"status": "ok"}
        return r

    with patch("requests.post", side_effect=fake_post):
        from dags.ddi_train_dag import _deploy_model
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    versioned = sorted(prod_dir.glob(".v_*"), key=lambda p: p.name)
    # 신규 .v_<ts> 포함해서 총 3개만 남아야 함
    assert len(versioned) == 3, (
        f"BACKUP_KEEP_N=3 인데 {len(versioned)}개 남음: {[d.name for d in versioned]}"
    )
    # 가장 오래된 .v_1~.v_4 는 삭제, .v_5 .v_6 + 신규 유지
    assert not (prod_dir / ".v_1").exists()
    assert not (prod_dir / ".v_2").exists()
    assert not (prod_dir / ".v_3").exists()
    assert not (prod_dir / ".v_4").exists()
