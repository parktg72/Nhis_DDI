"""배포 원자성·핫스왑 실패 시나리오 통합 테스트."""
import hashlib
import os
import pickle
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _has_symlink_privilege() -> bool:
    """Windows에서 심볼릭 링크 생성 권한 여부 확인."""
    with tempfile.TemporaryDirectory() as td:
        try:
            os.symlink("nonexistent", os.path.join(td, "_probe"))
            return True
        except OSError:
            return False


pytestmark = pytest.mark.skipif(
    not _has_symlink_privilege(),
    reason="Windows 심볼릭 링크 권한 없음 (관리자 또는 개발자 모드 필요)",
)


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


def _write_with_sha256(path: Path, content: bytes, sha_filename: str = None) -> None:
    """파일 쓰기 + 실제 sha256 사이드카 생성."""
    path.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    sha_path = path.parent / (sha_filename or (path.name + ".sha256"))
    sha_path.write_text(f"{sha}  {path.name}\n")


def _make_full_artifacts(staging: Path, base_name: str = "model_v1") -> Path:
    """완전한 앙상블 아티팩트 세트 생성 (실제 sha256 + 유효한 pickle)."""
    staging.mkdir(parents=True, exist_ok=True)

    # 메인 pkl — pickle.loads()로 trainer_class 읽을 수 있도록 유효한 dict
    main_payload = pickle.dumps({"trainer_class": "EnsembleTrainer", "weights": (0.5, 0.5)})
    main_pkl = staging / f"{base_name}.pkl"
    _write_with_sha256(main_pkl, main_payload)

    # 서브모델
    for ext in (".xgb.pkl", ".lgb.pkl"):
        _write_with_sha256(staging / f"{base_name}{ext}", b"submodel_stub")

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
    main_payload = pickle.dumps({"trainer_class": "EnsembleTrainer", "weights": (0.5, 0.5)})
    main_pkl = staging / "model_v1.pkl"
    _write_with_sha256(main_pkl, main_payload)
    _write_with_sha256(staging / "model_v1.xgb.pkl", b"xgb_stub")
    # model_v1.xgb.pkl.sha256 고의로 누락 — .sha256 삭제
    (staging / "model_v1.xgb.pkl.sha256").unlink()
    _write_with_sha256(staging / "model_v1.lgb.pkl", b"lgb_stub")

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
    _write_with_sha256(old_dir / "model_prod.pkl", b"old_model")
    _write_with_sha256(old_dir / "model_prod.xgb.pkl", b"old_xgb")
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
    _write_with_sha256(old_dir / "model_prod.pkl", b"old_model")
    _write_with_sha256(old_dir / "model_prod.xgb.pkl", b"old_xgb")
    _write_with_sha256(old_dir / "model_prod.lgb.pkl", b"old_lgb")
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


def test_hotswap_timeout_raises(tmp_path):
    """serving reload 타임아웃 시 RuntimeError."""
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
        _write_with_sha256(d / "model_prod.pkl", b"old")
        _write_with_sha256(d / "model_prod.xgb.pkl", b"old_xgb")
        _write_with_sha256(d / "model_prod.lgb.pkl", b"old_lgb")
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


def test_hotswap_failure_with_keep_n1_rollback_is_valid(tmp_path):
    """BACKUP_KEEP_N=1 환경에서 hotswap 실패 시 롤백 심링크가 유효한 디렉터리를 가리킴.

    pruning이 hotswap 전에 실행되면 prev 버전이 삭제되어 broken symlink 발생.
    pruning은 hotswap 성공 후에만 실행돼야 함.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    # 기존 버전 + current 심링크
    old_dir = prod_dir / ".v_1000"
    old_dir.mkdir()
    for fname in ("model_prod.pkl", "model_prod.pkl.sha256",
                  "model_prod.xgb.pkl", "model_prod.xgb.pkl.sha256",
                  "model_prod.lgb.pkl", "model_prod.lgb.pkl.sha256"):
        (old_dir / fname).write_bytes(b"old")
    (prod_dir / "current").symlink_to(".v_1000")

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.SERVING_URLS = ["http://localhost:9999"]
    s.ADMIN_API_KEY = "key"
    s.BACKUP_KEEP_N = 1  # 신규 버전만 유지 — 구버전이 pruning될 수 있음

    import requests as _req

    def fake_post_503(url, **kw):
        r = MagicMock()
        r.raise_for_status.side_effect = _req.HTTPError("503")
        return r

    with patch("requests.post", side_effect=fake_post_503):
        from dags.ddi_train_dag import _deploy_model
        with pytest.raises(RuntimeError, match="핫스왑 실패"):
            _deploy_model(ti=_FakeTI(str(main_pkl)))

    # 롤백 후 current가 유효한 디렉터리를 가리켜야 함 (broken symlink 아님)
    current = prod_dir / "current"
    assert current.is_symlink(), "current 심링크 없음"
    assert current.resolve().exists(), (
        f"롤백 후 current가 존재하지 않는 디렉터리를 가리킴: {os.readlink(current)}"
    )


def test_hotswap_partial_failure_sends_compensating_reload(tmp_path):
    """URL[0] 성공·URL[1] 실패 시 URL[0]에 구버전 model_path로 보상 롤백 전송."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    # 기존 버전 + current 심링크
    old_dir = prod_dir / ".v_1000"
    old_dir.mkdir()
    for fname in ("model_prod.pkl", "model_prod.pkl.sha256",
                  "model_prod.xgb.pkl", "model_prod.xgb.pkl.sha256",
                  "model_prod.lgb.pkl", "model_prod.lgb.pkl.sha256"):
        (old_dir / fname).write_bytes(b"old")
    (prod_dir / "current").symlink_to(".v_1000")

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://inst1:8000"
    s.SERVING_URLS = ["http://inst1:8000", "http://inst2:8000"]
    s.ADMIN_API_KEY = "key"
    s.BACKUP_KEEP_N = 5

    import requests as _req
    call_log: list = []  # (url, model_path)

    def fake_post(url, json=None, **kw):
        call_log.append((url, (json or {}).get("model_path", "")))
        if "inst2" in url:
            r = MagicMock()
            r.raise_for_status.side_effect = _req.HTTPError("503")
            return r
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"status": "ok"}
        return r

    with patch("requests.post", side_effect=fake_post):
        from dags.ddi_train_dag import _deploy_model
        with pytest.raises(RuntimeError, match="핫스왑 실패"):
            _deploy_model(ti=_FakeTI(str(main_pkl)))

    reload_calls = [(u, p) for u, p in call_log if "/admin/reload" in u]

    # inst1은 신버전으로 처음 호출, 실패 후 구버전으로 보상 호출
    inst1_paths = [p for u, p in reload_calls if "inst1" in u]
    assert len(inst1_paths) == 2, f"inst1 reload 호출 횟수 오류: {inst1_paths}"
    new_path = inst1_paths[0]
    old_path = inst1_paths[1]
    assert "current" in new_path or "model_prod" in new_path, "첫 호출이 신버전 경로 아님"
    assert ".v_1000" in old_path, f"보상 롤백 호출이 구버전 경로 아님: {old_path}"


def test_compensating_rollback_failure_included_in_error(tmp_path):
    """보상 롤백 자체 실패 시 최종 RuntimeError 메시지에 보상롤백 실패 정보 포함."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    old_dir = prod_dir / ".v_1000"
    old_dir.mkdir()
    for fname in ("model_prod.pkl", "model_prod.pkl.sha256",
                  "model_prod.xgb.pkl", "model_prod.xgb.pkl.sha256",
                  "model_prod.lgb.pkl", "model_prod.lgb.pkl.sha256"):
        (old_dir / fname).write_bytes(b"old")
    (prod_dir / "current").symlink_to(".v_1000")

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://inst1:8000"
    s.SERVING_URLS = ["http://inst1:8000", "http://inst2:8000"]
    s.ADMIN_API_KEY = "key"
    s.BACKUP_KEEP_N = 5

    import requests as _req
    call_count = [0]

    def fake_post(url, **kw):
        call_count[0] += 1
        r = MagicMock()
        # inst1 첫 호출 성공, 이후 모든 호출 실패 (inst2 실패 + inst1 보상 롤백 실패)
        if "inst1" in url and call_count[0] == 1:
            r.raise_for_status.return_value = None
            r.json.return_value = {"status": "ok"}
        else:
            r.raise_for_status.side_effect = _req.HTTPError("503")
        return r

    with patch("requests.post", side_effect=fake_post):
        from dags.ddi_train_dag import _deploy_model
        with pytest.raises(RuntimeError) as exc_info:
            _deploy_model(ti=_FakeTI(str(main_pkl)))

    msg = str(exc_info.value)
    assert "핫스왑 실패" in msg
    assert "보상롤백 실패" in msg, f"보상롤백 실패 정보가 에러 메시지에 없음: {msg}"


def test_versioned_dir_name_is_unique_nanoseconds(tmp_path):
    """두 번 연속 배포해도 .v_<ts_ns> 이름이 충돌하지 않음 (나노초 해상도)."""
    import time as _time

    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    def fake_post(url, **kw):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"status": "ok"}
        return r

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.SERVING_URLS = ["http://localhost:9999"]
    s.ADMIN_API_KEY = "key"
    s.BACKUP_KEEP_N = 5

    # 첫 번째 배포
    main_pkl = _make_full_artifacts(staging, base_name="model_v1")
    with patch("requests.post", side_effect=fake_post):
        from dags.ddi_train_dag import _deploy_model
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    # 두 번째 배포 (같은 초 내 가능)
    main_pkl2 = _make_full_artifacts(staging, base_name="model_v2")
    with patch("requests.post", side_effect=fake_post):
        _deploy_model(ti=_FakeTI(str(main_pkl2)))

    versioned = list(prod_dir.glob(".v_*"))
    names = [d.name for d in versioned]
    assert len(names) == len(set(names)), f"버전 디렉터리 이름 충돌: {names}"
    assert len(versioned) == 2, f"배포 2회인데 .v_* 개수 오류: {names}"


def test_atomic_symlink_update_unique_tmp_name(tmp_path):
    """_atomic_symlink_update tmp 심링크 이름이 PID 포함 unique 이름 사용."""
    from dags.ddi_train_dag import _atomic_symlink_update
    import os as _os

    target_dir = tmp_path / ".v_1"
    target_dir.mkdir()
    link = tmp_path / "current"

    _atomic_symlink_update(link, ".v_1")
    assert link.is_symlink()
    assert link.resolve() == target_dir.resolve()

    # tmp 심링크 잔재 없음 확인
    tmp_artifacts = list(tmp_path.glob("current.tmp*"))
    assert tmp_artifacts == [], f"tmp 심링크 잔재: {tmp_artifacts}"


def test_mid_deploy_crash_leftover_tmp_dir_does_not_block_next_deploy(tmp_path):
    """Phase 2 도중 크래시 → .deploy_tmp_* 잔재가 다음 배포를 방해하지 않음.

    per-run tempfile.mkdtemp 격리로 다음 배포는 별도 tmp 디렉터리 사용.
    잔재 .deploy_tmp_* 는 배포 성공 후 prune 대상이 아니므로 수동 정리 필요하지만
    배포 자체는 정상 완료돼야 한다.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    # 이전 배포 크래시로 남겨진 잔재 tmp 디렉터리
    leftover = prod_dir / ".deploy_tmp_crashed"
    leftover.mkdir()
    (leftover / "model_prod.pkl").write_bytes(b"crashed_deploy")

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.SERVING_URLS = ["http://localhost:9999"]
    s.ADMIN_API_KEY = "key"
    s.BACKUP_KEEP_N = 5

    def fake_post(url, **kw):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"status": "ok"}
        return r

    with patch("requests.post", side_effect=fake_post):
        from dags.ddi_train_dag import _deploy_model
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    # 새 배포가 성공하고 current 심링크가 존재해야 함
    current = prod_dir / "current"
    assert current.is_symlink(), "잔재 tmp 디렉터리로 인해 배포 실패"
    assert (current / "model_prod.pkl").exists()
    # M1 수정: 배포 시작 시 .deploy_tmp_* 잔재 자동 정리
    assert not leftover.exists(), "잔재 tmp 디렉터리가 자동 정리되지 않음"


def test_phase2_copy_failure_leaves_prod_dir_intact(tmp_path):
    """Phase 2 복사 실패(디스크 풀 등) → prod_dir 변경 없음, tmp 자동 정리."""
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
    s.ADMIN_API_KEY = "key"

    # shutil.copy2 실패 시뮬레이션 (디스크 풀)
    with patch("shutil.copy2", side_effect=OSError("No space left on device")):
        from dags.ddi_train_dag import _deploy_model
        with pytest.raises(OSError, match="No space left"):
            _deploy_model(ti=_FakeTI(str(main_pkl)))

    # prod_dir에 current 심링크나 .v_* 디렉터리 없음
    assert not (prod_dir / "current").exists(), "Phase 2 실패 후 current 생성됨"
    assert not list(prod_dir.glob(".v_*")), "Phase 2 실패 후 버전 디렉터리 생성됨"
    # tmp 디렉터리도 정리됨
    assert not list(prod_dir.glob(".deploy_tmp_*")), "Phase 2 실패 후 tmp 잔재 존재"


def test_hotswap_sends_versioned_path_not_symlink(tmp_path):
    """C1 수정: 핫스왑 reload 요청이 symlink 경로(current/)가 아닌 versioned 절대 경로 전송.

    동시 배포 경쟁 시 serving이 정확히 의도한 버전을 로드함을 보장.
    """
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
    s.ADMIN_API_KEY = "key"
    s.BACKUP_KEEP_N = 5

    received_paths: list = []

    def fake_post(url, json=None, **kw):
        received_paths.append((json or {}).get("model_path", ""))
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"status": "ok"}
        return r

    with patch("requests.post", side_effect=fake_post):
        from dags.ddi_train_dag import _deploy_model
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    reload_paths = [p for p in received_paths if "model_prod" in p]
    assert reload_paths, "reload 요청 없음"
    for rp in reload_paths:
        assert "current" not in rp, f"C1: symlink 경로가 전송됨 — {rp}"
        assert ".v_" in rp, f"C1: versioned 경로가 아님 — {rp}"


def test_missing_admin_api_key_raises_before_filesystem_changes(tmp_path):
    """H2 수정: ADMIN_API_KEY 미설정 시 파일시스템 변경 없이 즉시 RuntimeError."""
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
    s.ADMIN_API_KEY = ""  # 미설정

    from dags.ddi_train_dag import _deploy_model
    with pytest.raises(RuntimeError, match="ADMIN_API_KEY"):
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    # 파일시스템 변경 없음
    assert not (prod_dir / "current").exists(), "pre-flight 실패임에도 current 생성됨"
    assert not list(prod_dir.glob(".v_*")), "pre-flight 실패임에도 버전 디렉터리 생성됨"
