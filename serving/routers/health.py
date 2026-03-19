"""
헬스체크 및 모델 정보 엔드포인트
GET /health      - 서버 상태
GET /model/info  - 로드된 모델 정보
POST /admin/reload - 모델 핫스왑 (운영)
"""
from fastapi import APIRouter, HTTPException
from serving.predictor import get_predictor
from serving.schemas import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["health"])

APP_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 및 모델 상태 확인."""
    try:
        pred = get_predictor()
        return HealthResponse(
            status="ok",
            model_loaded=pred._ml.loaded,
            rule_loaded=True,
            version=APP_VERSION,
            uptime_sec=round(pred.uptime, 1),
        )
    except RuntimeError:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            rule_loaded=False,
            version=APP_VERSION,
            uptime_sec=0.0,
        )


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """로드된 모델 정보."""
    pred = get_predictor()
    ml = pred._ml
    return ModelInfoResponse(
        model_type=ml._model_type if ml.loaded else "none",
        partition=ml._partition,
        n_features=len(ml._feature_names) if ml._feature_names else None,
        threshold=ml._threshold if ml.loaded else None,
    )


@router.post("/admin/reload")
async def reload_model(model_path: str):
    """모델 핫스왑 (무중단 교체)."""
    pred = get_predictor()
    ok = pred.reload_model(model_path)
    if not ok:
        raise HTTPException(status_code=400, detail=f"모델 로드 실패: {model_path}")
    return {"status": "ok", "model_path": model_path}
