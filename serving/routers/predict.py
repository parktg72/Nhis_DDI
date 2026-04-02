"""
예측 엔드포인트
POST /predict       - 단일 환자 위험도 예측
POST /predict/batch - 배치 예측 (최대 1000명)
"""
import logging
import time
from collections import Counter

from fastapi import APIRouter, HTTPException

from serving.predictor import get_predictor
from serving.schemas import (
    BatchPredictRequest, BatchPredictResponse,
    PredictRequest, PredictResponse, RiskLevel,
)

logger = logging.getLogger(__name__)

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
        logger.exception("예측 처리 중 오류 (patient_id=%s)", req.patient_id)
        raise HTTPException(status_code=500, detail="내부 서버 오류: 예측 처리 실패")


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
        logger.exception("배치 예측 초기화 오류")
        raise HTTPException(status_code=500, detail="내부 서버 오류: 예측기 초기화 실패")

    results = []
    warnings = []
    for single_req in req.requests:
        try:
            results.append(pred.predict(single_req))
        except Exception as e:
            logger.warning("배치 부분 실패 (patient_id=%s): %s", single_req.patient_id, e)
            warnings.append(f"{single_req.patient_id}: 예측 처리 실패")

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
        warnings=warnings,
    )
