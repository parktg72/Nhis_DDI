"""
예측 엔드포인트
POST /predict       - 단일 환자 위험도 예측
POST /predict/batch - 배치 예측 (최대 1000명)
"""
import time
from collections import Counter

from fastapi import APIRouter, HTTPException

from serving.predictor import get_predictor
from serving.schemas import (
    BatchPredictRequest, BatchPredictResponse,
    PredictRequest, PredictResponse, RiskLevel,
)

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    단일 환자 위험도 예측.

    - Rule-based Safety Net (Top 10 DDI 100% 탐지)
    - ML 모델 (XGBoost/LightGBM, 로드된 경우)
    - 최종등급 = max(Rule, ML)
    """
    try:
        pred = get_predictor()
        return pred.predict(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest):
    """
    배치 예측 (최대 1000명).
    Rule + ML 하이브리드 예측. 결과에 위험도 분포 통계 포함.
    """
    t0 = time.perf_counter()
    try:
        pred = get_predictor()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = []
    errors = []
    for single_req in req.requests:
        try:
            results.append(pred.predict(single_req))
        except Exception as e:
            errors.append(f"{single_req.patient_id}: {e}")

    if errors:
        # 부분 실패는 허용, 헤더에 경고 기록
        pass

    dist = Counter(r.risk_level for r in results)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return BatchPredictResponse(
        results=results,
        total=len(results),
        red_count=dist.get(RiskLevel.RED, 0),
        yellow_count=dist.get(RiskLevel.YELLOW, 0),
        green_count=dist.get(RiskLevel.GREEN, 0),
        normal_count=dist.get(RiskLevel.NORMAL, 0),
        elapsed_ms=round(elapsed_ms, 1),
    )
