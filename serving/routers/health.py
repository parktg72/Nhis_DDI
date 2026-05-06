"""
헬스체크 및 모델 정보 엔드포인트
GET /health        - 서버 상태
GET /model/info    - 로드된 모델 정보
POST /admin/reload - 모델 핫스왑 (운영, X-Admin-Key 헤더 필수)

환경변수:
  ADMIN_API_KEY : /admin/* 엔드포인트 인증 키 (미설정 시 엔드포인트 비활성화)
  MODEL_DIR     : 허용된 모델 파일 디렉토리 (기본: /app/models)
                  경로를 이 디렉토리 밖으로 지정하면 거부됩니다.
"""
import hmac
import os
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from serving.predictor import get_predictor
from serving.schemas import HealthResponse, ModelInfoResponse
from config import settings as _settings

router = APIRouter(tags=["health"])


class ReloadRequest(BaseModel):
    model_path: str


class HierarchicalReloadRequest(BaseModel):
    model_dir: str

APP_VERSION = "1.0.0"

_ADMIN_KEY: str = _settings.ADMIN_API_KEY
_MODEL_DIR: Path = _settings.MODEL_DIR.resolve()


def _require_admin(x_admin_key: str = Header(..., alias="X-Admin-Key")) -> None:
    """X-Admin-Key 헤더로 관리자 인증. ADMIN_API_KEY 미설정 시 엔드포인트 전체 비활성화."""
    if not _ADMIN_KEY:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_API_KEY 환경변수 미설정: /admin 엔드포인트 비활성화",
        )
    if not hmac.compare_digest(x_admin_key, _ADMIN_KEY):
        raise HTTPException(status_code=401, detail="관리자 인증 실패")


def _model_mode(pred) -> str:
    """단일 ML / 계층 / 양쪽 / 없음 — model_mode 라벨 도출."""
    ml_on = bool(pred._ml.loaded)
    hier_on = pred._hierarchical is not None and pred._hierarchical.loaded
    if ml_on and hier_on:
        return "both"
    if hier_on:
        return "hierarchical"
    if ml_on:
        return "single"
    return "none"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 및 모델 상태 확인. 계층 모드만 로드된 경우도 model_loaded=True."""
    try:
        pred = get_predictor()
        ml_loaded = bool(pred._ml.loaded)
        hier_loaded = pred._hierarchical is not None and pred._hierarchical.loaded
        return HealthResponse(
            status="ok",
            model_loaded=ml_loaded or hier_loaded,
            rule_loaded=pred._safety_net is not None,
            version=APP_VERSION,
            uptime_sec=round(pred.uptime, 1),
            model_mode=_model_mode(pred),
            hierarchical_loaded=hier_loaded,
        )
    except RuntimeError:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            rule_loaded=False,
            version=APP_VERSION,
            uptime_sec=0.0,
            model_mode="none",
            hierarchical_loaded=False,
        )


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """로드된 모델 정보. 계층 모드만 로드된 경우 stage1/stage2 정보로 채움."""
    pred = get_predictor()
    ml = pred._ml
    hier = pred._hierarchical

    if ml.loaded:
        return ModelInfoResponse(
            model_type=ml._model_type,
            partition=ml._partition,
            n_features=len(ml._feature_names) if ml._feature_names else None,
            threshold=ml._threshold,
        )
    if hier is not None and hier.loaded:
        return ModelInfoResponse(
            model_type="hierarchical",
            partition=None,
            n_features=len(hier.feature_cols) if hier.feature_cols else None,
            threshold=hier._thresholds.get("tau_red"),
        )
    return ModelInfoResponse(
        model_type="none",
        partition=ml._partition,
        n_features=len(ml._feature_names) if ml._feature_names else None,
        threshold=None,
    )


@router.post("/admin/reload")
async def reload_model(
    body: ReloadRequest,
    _: None = Depends(_require_admin),
):
    """모델 핫스왑 (무중단 교체). X-Admin-Key 헤더 인증 필수.

    model_path는 MODEL_DIR 환경변수로 지정된 디렉토리 내부 경로만 허용됩니다.
    """
    resolved = Path(body.model_path).resolve()
    try:
        resolved.relative_to(_MODEL_DIR)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"모델 경로는 허용된 디렉토리({_MODEL_DIR}) 내부여야 합니다: {body.model_path}",
        )
    pred = get_predictor()
    ok = pred.reload_model(resolved)
    if not ok:
        raise HTTPException(status_code=400, detail=f"모델 로드 실패: {body.model_path}")
    return {"status": "ok", "model_path": str(resolved)}


@router.post("/admin/reload/hierarchical")
async def reload_hierarchical_model(
    body: HierarchicalReloadRequest,
    _: None = Depends(_require_admin),
):
    """계층 모델 핫스왑 (무중단 교체). X-Admin-Key 헤더 인증 필수.

    model_dir은 MODEL_DIR 환경변수로 지정된 디렉토리 내부 경로만 허용됩니다.
    """
    resolved = Path(body.model_dir).resolve()
    try:
        resolved.relative_to(_MODEL_DIR)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"모델 경로는 허용된 디렉토리({_MODEL_DIR}) 내부여야 합니다: {body.model_dir}",
        )
    pred = get_predictor()
    ok = pred.reload_hierarchical(resolved)
    if not ok:
        raise HTTPException(status_code=400, detail=f"계층 모델 로드 실패: {body.model_dir}")
    return {"status": "ok", "model_dir": str(resolved)}
