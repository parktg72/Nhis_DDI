"""DAG↔serving 환경변수 계약 및 하드코딩 경로 검증."""
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent


def test_settings_model_dir_default():
    """MODEL_DIR 기본값 /app/models."""
    import config.settings as s
    importlib.reload(s)
    assert str(s.MODEL_DIR) == "/app/models"


def test_settings_model_dir_env_override(monkeypatch, tmp_path):
    """MODEL_DIR 환경변수 오버라이드."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    import config.settings as s
    importlib.reload(s)
    assert s.MODEL_DIR == tmp_path
    importlib.reload(s)  # cleanup


def test_settings_admin_api_key_default():
    """ADMIN_API_KEY 기본값 빈 문자열."""
    import config.settings as s
    importlib.reload(s)
    assert s.ADMIN_API_KEY == ""


def test_no_hardcoded_app_models():
    """/app/models 리터럴이 config/settings.py 외 Python 소스에 없음."""
    violations = []
    exclude = {".venv", ".venv_macos", "__pycache__", "docs", ".git", ".worktrees"}
    # 이 테스트 파일 자체는 검사 문자열을 포함하므로 제외
    this_file = Path(__file__).relative_to(REPO_ROOT)
    for py_file in REPO_ROOT.rglob("*.py"):
        if any(part in exclude for part in py_file.parts):
            continue
        rel = py_file.relative_to(REPO_ROOT)
        if rel == Path("config/settings.py") or rel == this_file:
            continue
        content = py_file.read_text(errors="replace")
        if '"/app/models"' in content or "'/app/models'" in content:
            violations.append(str(rel))
    assert violations == [], (
        "하드코딩된 /app/models 발견 — config.settings 로 교체하세요:\n"
        + "\n".join(violations)
    )


def test_settings_log_level_default(monkeypatch):
    """LOG_LEVEL 기본값은 INFO."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    import config.settings as s
    importlib.reload(s)
    assert s.LOG_LEVEL == "INFO"
    importlib.reload(s)


def test_settings_cors_origins_default(monkeypatch):
    """CORS_ORIGINS 기본값은 빈 문자열."""
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    import config.settings as s
    importlib.reload(s)
    assert s.CORS_ORIGINS == ""
    importlib.reload(s)


def test_settings_serving_urls_single(monkeypatch):
    """DDI_SERVING_URLS 미설정 시 DDI_SERVING_URL 단일 URL 사용."""
    monkeypatch.delenv("DDI_SERVING_URLS", raising=False)
    monkeypatch.setenv("DDI_SERVING_URL", "http://host1:8000")
    import config.settings as s
    importlib.reload(s)
    assert s.SERVING_URLS == ["http://host1:8000"]
    importlib.reload(s)


def test_settings_serving_urls_multi(monkeypatch):
    """DDI_SERVING_URLS 설정 시 쉼표 구분 URL 목록 반환."""
    monkeypatch.setenv("DDI_SERVING_URLS", "http://a:8000,http://b:8000")
    import config.settings as s
    importlib.reload(s)
    assert s.SERVING_URLS == ["http://a:8000", "http://b:8000"]
    importlib.reload(s)


def test_settings_backup_keep_n_default(monkeypatch):
    """BACKUP_KEEP_N 기본값은 5."""
    monkeypatch.delenv("DDI_BACKUP_KEEP_N", raising=False)
    import config.settings as s
    importlib.reload(s)
    assert s.BACKUP_KEEP_N == 5
    importlib.reload(s)


def test_admin_api_key_no_drift():
    """DDI_ADMIN_API_KEY 잔재 없음 — ADMIN_API_KEY 로 통일됨."""
    violations = []
    exclude = {".venv", ".venv_macos", "__pycache__", "docs", ".git", ".worktrees"}
    this_file = Path(__file__).relative_to(REPO_ROOT)
    for py_file in REPO_ROOT.rglob("*.py"):
        if any(part in exclude for part in py_file.parts):
            continue
        rel = py_file.relative_to(REPO_ROOT)
        if rel == this_file:
            continue
        content = py_file.read_text(errors="replace")
        if "DDI_ADMIN_API_KEY" in content:
            violations.append(str(rel))
    assert violations == [], (
        "DDI_ADMIN_API_KEY 잔재 발견:\n" + "\n".join(violations)
    )
