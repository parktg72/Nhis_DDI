"""
예측 엔드포인트
POST /predict       - 단일 환자 위험도 예측
POST /predict/batch - 배치 예측 (최대 1000명)
"""
import logging
import time
from collections import Counter
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from monitoring.metrics_writer import get_metrics_writer
from serving.predictor import get_predictor
from serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    RiskLevel,
)

logger = logging.getLogger(__name__)


def _rule_ids(result) -> list[str]:
    """사유 목록에서 **규칙 ID 만** 뽑는다 (A5 발화 측정용).

    설명 문구와 약물명은 넣지 않는다. 이 파일은 환자 단위로 append 되므로,
    필요한 최소치(어느 규칙이 발화했는가)만 남겨 환자별 약물 정보가 축적되지
    않게 한다. 규칙별 발화 환자 수와 "사유 없는 Red" 건수는 이것으로 산출된다.
    """
    out: list[str] = []
    for r in (getattr(result, "risk_reasons", None) or []):
        rid = str(r).split(":", 1)[0].strip()
        if rid and rid not in out:
            out.append(rid)
    return out


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
        t0 = time.perf_counter()
        result = pred.predict(req)
        latency_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        logger.exception("예측 처리 중 오류 (patient_id=%s)", req.patient_id)
        raise HTTPException(status_code=500, detail="내부 서버 오류: 예측 처리 실패")

    try:
        _now = datetime.now(timezone.utc)
        get_metrics_writer().append({
            "timestamp": _now.isoformat(),
            "partition": _now.strftime("%Y-%m-%d"),
            "patient_id": req.patient_id,
            "risk_level": result.risk_level.value,
            "rule_level": result.rule_level.value if result.rule_level else None,
            "ml_level": result.ml_level.value if result.ml_level else None,
            "disagree": (
                result.rule_level != result.ml_level
                if result.ml_level else False
            ),
            "latency_ms": round(latency_ms, 1),
            "source": "api",
            "rule_ids": _rule_ids(result),
            "n_reasons": len(result.risk_reasons or []),
        })
    except Exception:
        logger.warning("메트릭 기록 실패 — 예측은 정상 반환", exc_info=True)

    return result


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
            t_single = time.perf_counter()
            single_result = pred.predict(single_req)
            single_latency_ms = (time.perf_counter() - t_single) * 1000
            results.append(single_result)
            try:
                _now = datetime.now(timezone.utc)
                get_metrics_writer().append({
                    "timestamp": _now.isoformat(),
                    "partition": _now.strftime("%Y-%m-%d"),
                    "patient_id": single_req.patient_id,
                    "risk_level": single_result.risk_level.value,
                    "rule_level": single_result.rule_level.value if single_result.rule_level else None,
                    "ml_level": single_result.ml_level.value if single_result.ml_level else None,
                    "disagree": (
                        single_result.rule_level != single_result.ml_level
                        if single_result.ml_level else False
                    ),
                    "latency_ms": round(single_latency_ms, 1),
                    "source": "batch",
                    "rule_ids": _rule_ids(single_result),
                    "n_reasons": len(single_result.risk_reasons or []),
                })
            except Exception:
                logger.warning("배치 메트릭 기록 실패 (patient_id=%s)", single_req.patient_id)
        except Exception as e:
            logger.warning("배치 부분 실패 (patient_id=%s): %s", single_req.patient_id, e)
            warnings.append(f"{single_req.patient_id}: 예측 처리 실패")

    dist = Counter(r.risk_level for r in results)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    requested_count = len(req.requests)
    success_count = len(results)
    failed_count = requested_count - success_count
    return BatchPredictResponse(
        results=results,
        requested_count=requested_count,
        success_count=success_count,
        failed_count=failed_count,
        total=success_count,  # DEPRECATED alias (Codex 2026-05-07 #6)
        red_count=dist.get(RiskLevel.RED, 0),
        yellow_count=dist.get(RiskLevel.YELLOW, 0),
        green_count=dist.get(RiskLevel.GREEN, 0),
        normal_count=dist.get(RiskLevel.NORMAL, 0),
        elapsed_ms=round(elapsed_ms, 1),
        warnings=warnings,
    )
