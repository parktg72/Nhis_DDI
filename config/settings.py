"""
중앙 환경변수 설정 모듈.

모든 환경변수 기본값은 이 파일이 유일한 소스다.
DAG, serving, 테스트 모두 여기서 import한다.

주의: 모듈 레벨 상수는 프로세스 시작 시 1회 평가된다.
장수 프로세스(Airflow webserver, uvicorn)에서 런타임 오버라이드는
반영되지 않는다. 테스트에서는 monkeypatch 후 importlib.reload(settings) 사용.
순환 import 방지: 이 모듈은 프로젝트 내 다른 모듈을 일절 import하지 않는다.
"""
import os
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────────────────────────
MODEL_DIR       = Path(os.environ.get("MODEL_DIR",          "/app/models"))
FEATURES_DIR    = Path(os.environ.get("DDI_FEATURES_DIR",   "/app/data/features"))
PROCESSED_DIR   = Path(os.environ.get("DDI_PROCESSED_DIR",  "/app/data/processed"))
PREDICTIONS_DIR = Path(os.environ.get("DDI_PREDICTIONS_DIR","/app/data/predictions"))
RAW_DATA_DIR    = Path(os.environ.get("DDI_RAW_DATA_DIR",   "/app/data/raw"))

# 파생 경로 (MODEL_DIR 기반)
MODEL_PROD_PATH   = MODEL_DIR / "current" / "model_prod.pkl"
MODEL_BACKUP_PATH = MODEL_DIR / "backup" / "model_prod.pkl"

# ── 로깅 / CORS ────────────────────────────────────────────────────────────────
LOG_LEVEL    = os.environ.get("LOG_LEVEL",    "INFO")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "")

# ── API / 서비스 ───────────────────────────────────────────────────────────────
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
SERVING_URL   = os.environ.get("DDI_SERVING_URL", "http://localhost:8000")

# 다중 인스턴스 핫스왑: 쉼표 구분 URL 목록
# DDI_SERVING_URLS=http://inst1:8000,http://inst2:8000
# 미설정 시 DDI_SERVING_URL 단일 인스턴스 사용
_serving_urls_raw = os.environ.get("DDI_SERVING_URLS", "")
SERVING_URLS: list = (
    [u.strip() for u in _serving_urls_raw.split(",") if u.strip()]
    if _serving_urls_raw
    else ([SERVING_URL] if SERVING_URL else [])
)

# ── 배포 보존 정책 ──────────────────────────────────────────────────────────────
BACKUP_KEEP_N = max(1, int(os.environ.get("DDI_BACKUP_KEEP_N", "5")))

# ── 훈련 파라미터 ──────────────────────────────────────────────────────────────
TRAIN_WEEKS      = int(os.environ.get("DDI_TRAIN_WEEKS",        "4"))
MODEL_TYPE       = os.environ.get("DDI_MODEL_TYPE",             "ensemble")
OPTUNA_TRIALS    = int(os.environ.get("DDI_OPTUNA_TRIALS",      "50"))
RECALL_THRESHOLD = float(os.environ.get("DDI_RECALL_THRESHOLD", "0.90"))
AUC_THRESHOLD    = float(os.environ.get("DDI_AUC_THRESHOLD",    "0.85"))
BATCH_SIZE       = max(1, min(10_000, int(os.environ.get("DDI_BATCH_SIZE", "500"))))

# ── 데이터 파생 경로 ───────────────────────────────────────────────────────────
DDI_MATRIX_PATH = Path(os.environ.get(
    "DDI_MATRIX_PATH", "/app/data/processed/ddi_matrix_final.parquet"
))
DRUG_INDEX_PATH = Path(os.environ.get(
    "DRUG_INDEX_PATH", "/app/data/processed/drug_name_index.parquet"
))
CYP_MATRIX_PATH = Path(os.environ.get(
    "CYP_MATRIX_PATH", "/app/data/processed/cyp_matrix.parquet"
))
# DDI_DRUG_INDEX_PATH 키로도 접근 가능 (ddi_etl_dag.py 호환)
DRUG_INDEX_PARQUET = Path(os.environ.get(
    "DDI_DRUG_INDEX_PATH", "/app/data/processed/drug_name_index.parquet"
))

# ── 모니터링 ────────────────────────────────────────────────────────────────────
MONITORING_DIR             = Path(os.environ.get("DDI_MONITORING_DIR",        "/app/data/monitoring"))
METRICS_JSONL_PATH         = Path(os.environ.get("DDI_METRICS_JSONL_PATH",    "/app/data/monitoring/metrics_live.jsonl"))
DRIFT_REFERENCE_PATH       = Path(os.environ.get("DDI_DRIFT_REFERENCE_PATH",  "/app/models/current/drift_reference.pkl"))
METRICS_JSONL_LOCK_TIMEOUT = float(os.environ.get("DDI_METRICS_JSONL_LOCK_TIMEOUT", "5.0"))
