"""
FastAPI 애플리케이션 진입점

실행:
  uvicorn serving.main:app --host 0.0.0.0 --port 8000 --workers 4

환경변수:
  MODEL_PATH      : ML 모델 경로 (없으면 Rule-only 모드)
  DDI_MATRIX_PATH : DDI 매트릭스 경로 (기본: data/processed/ddi_matrix_final.parquet)
  DRUG_INDEX_PATH : 약물 인덱스 경로 (기본: data/processed/drug_name_index.parquet)
  CYP_MATRIX_PATH : CYP 매트릭스 경로 (기본: data/processed/cyp_matrix.parquet)
  LOG_LEVEL       : 로그 레벨 (기본: INFO)
  ADMIN_API_KEY   : /admin/reload 인증 키 (미설정 시 엔드포인트 비활성화)
  MODEL_DIR       : 모델 핫스왑 허용 디렉토리 (기본: data/models)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from serving.middleware import RequestLoggingMiddleware
from serving.predictor import init_predictor
from serving.routers import health, predict

# ─────────────────────────────────────────────────────────────────────────────
# 로깅 설정
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 앱 생명주기
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 리소스 초기화/해제."""
    logger.info("DDI 위험도 분류 서버 시작")
    init_predictor(
        model_path=os.environ.get("MODEL_PATH"),
        ddi_matrix_path=os.environ.get("DDI_MATRIX_PATH", "data/processed/ddi_matrix_final.parquet"),
        drug_index_path=os.environ.get("DRUG_INDEX_PATH", "data/processed/drug_name_index.parquet"),
        cyp_matrix_path=os.environ.get("CYP_MATRIX_PATH", "data/processed/cyp_matrix.parquet"),
    )
    logger.info("예측기 초기화 완료")
    yield
    logger.info("서버 종료")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 앱 생성
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DDI 다재약물 위험도 분류 API",
    description=(
        "국민건강보험공단 다재약물 환자 위험도 자동 분류 서비스.\n\n"
        "- Rule-based Safety Net (Top 10 DDI 100% 탐지)\n"
        "- ML 모델 (XGBoost/LightGBM)\n"
        "- 최종등급 = max(Rule, ML)"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────────────────────────────────────
# 미들웨어
# ─────────────────────────────────────────────────────────────────────────────

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 폐쇄망 내부이므로 전체 허용
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# 라우터
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(predict.router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "DDI Risk Classifier",
        "version": "1.0.0",
        "docs": "/docs",
    }
