"""
예측 엔드포인트
POST /predict       - 단일 환자 위험도 예측
POST /predict/batch - 배치 예측 (최대 1000명)
"""
import logging
import re
import time
from collections import Counter
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from monitoring.metrics_writer import get_metrics_writer
from serving.predictor import serving_flag_state
from serving.predictor import get_predictor
from serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    RiskLevel,
)

logger = logging.getLogger(__name__)


# 사유 목록에서 규칙 ID 로 인정하는 문법. **네임스페이스를 명시적으로 제한한다.**
# 대문자 토큰이면 다 통과시키면 TIMEOUT·UNKNOWN 같은 비규칙 토큰도 규칙으로 세게
# 된다. 실제로 사유에 실리는 것은 아래 다섯 갈래뿐이다.
#   TOP01~TOP10   Top-10 DDI 규칙
#   GRADE_*       등급 산출 조건 (10종+고위험약 · 75세+신/간 등)
#   SEV_*         rule_floor 의 중증 하한 트리거
#   RED_*         결정적 Red 백스톱 (금기)
#   DDI_*         subtype 판정에서 온 DDI 사유
# 사유 문자열에는 ID 형식이 아닌 것도 섞인다 — `동일성분중복 3건`(건수마다
# 달라진다) · `ML 모델 Red 확률: 45.2%`(값이 매번 다르다). 이것들을 ID 로 세면
# 집계표가 오염되고 카디널리티가 무한히 늘어난다.
_RULE_ID_RE = re.compile(r"^(?:TOP\d{2}|(?:GRADE|SEV|RED|DDI)_[A-Z0-9_]+)$")

METRICS_SCHEMA_VERSION = 3


def _split_reasons(result) -> tuple[list[str], int]:
    """사유 목록 → (규칙 ID 목록, ID 형식이 아닌 사유 수).

    설명 문구와 약물명은 기록하지 않는다. 이 파일은 환자 단위로 누적되므로
    필요한 최소치(어느 규칙이 발화했는가)만 남긴다.

    ID 가 아닌 사유는 **개수만** 센다. 버리지 않는 이유는, 0 으로 보이면 사유가
    없는 것으로 오독되기 때문이다 — 사유 없는 Red 판정이 그 위에 선다.
    """
    ids: list[str] = []
    other = 0
    for r in (getattr(result, "risk_reasons", None) or []):
        token = str(r).split(":", 1)[0].strip()
        if _RULE_ID_RE.match(token):
            if token not in ids:
                ids.append(token)
        else:
            other += 1
    return ids, other


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
        _ids, _other = _split_reasons(result)
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
            "schema_version": METRICS_SCHEMA_VERSION,
            # 플래그 상태를 행에 남긴다. 관측 기간 중 플래그가 바뀌면 꺼진 날과
            # 켜진 날의 행이 한 파일에 섞이고, 집계가 그것을 구분할 수 없다.
            "serving_flags": serving_flag_state(),
            "rule_ids": _ids,
            "n_reasons": len(result.risk_reasons or []),
            "n_other_reasons": _other,
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
                _b_ids, _b_other = _split_reasons(single_result)
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
                    "schema_version": METRICS_SCHEMA_VERSION,
                    "serving_flags": serving_flag_state(),
                    "rule_ids": _b_ids,
                    "n_reasons": len(single_result.risk_reasons or []),
                    "n_other_reasons": _b_other,
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
