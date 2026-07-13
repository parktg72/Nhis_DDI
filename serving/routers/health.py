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
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from config import settings as _settings
from scripts.datasets.contracts import (
    BundleArtifactEmptyError,
    BundleHashMismatchError,
    LookbackMismatchError,
)
from serving.predictor import get_predictor
from serving.schemas import HealthResponse, ModelInfoResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class ReloadRequest(BaseModel):
    model_path: str


class HierarchicalReloadRequest(BaseModel):
    model_dir: str


class DLReloadRequest(BaseModel):
    bundle_dir: str

APP_VERSION = "1.0.0"

_ADMIN_KEY: str = _settings.ADMIN_API_KEY
_MODEL_DIR: Path = _settings.MODEL_DIR.resolve()
_DL_MODEL_DIR: Path = (_MODEL_DIR / "dl").resolve()


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


def _collect_schema_drift(pred) -> list[str]:
    """단일 ML + 계층 모델의 schema drift trail 수집 (Codex 2026-05-07 #1).

    현재는 단일 ML 의 `_schema_drift` 만 있음 (FEATURE_SCHEMA_LENIENT=1 로 unknown
    feature 가 lenient 통과한 경우). 계층 모델 쪽은 schema_drift trail 없이 strict
    reject 만 하지만, 미래 확장을 위해 helper 가 양쪽 모두 검사하는 형태로 둠.
    """
    drift: list[str] = []
    ml = getattr(pred, "_ml", None)
    if ml is not None:
        drift.extend(getattr(ml, "_schema_drift", []) or [])
    hier = getattr(pred, "_hierarchical", None)
    if hier is not None:
        drift.extend(getattr(hier, "_schema_drift", []) or [])
    return drift


def _dl_info(pred) -> dict:
    dl = getattr(pred, "_dl", None)
    loaded = bool(dl is not None and dl.loaded)
    return {
        "dl_loaded": loaded,
        "dl_lookback_days": dl.lookback_days if loaded else None,
        "dl_bundle_run_id": dl.run_id if loaded else None,
        "dl_schema_version": dl.schema_version if loaded else None,
    }


def _is_schema_lenient_active() -> bool:
    """FEATURE_SCHEMA_LENIENT 운영 escape hatch 활성 상태 (env 기준).

    sunset 효력은 별도 — `_is_schema_lenient_allowed()` 참고.
    """
    return os.environ.get("FEATURE_SCHEMA_LENIENT", "").strip().lower() in (
        "1", "true", "yes",
    )


def _lenient_sunset_date_iso() -> str:
    """현재 적용 중인 sunset date 의 ISO 문자열 (운영자 가시성용 — Codex #6-followup)."""
    from serving.predictor import (
        _FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT,
    )
    raw = os.environ.get("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "").strip()
    if raw:
        # invalid 도 그대로 노출 — 운영자가 형식 오류 인지 가능
        return raw
    return _FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT.isoformat()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 및 모델 상태 확인. 계층 모드만 로드된 경우도 model_loaded=True.

    Codex 2026-05-07 #1 — schema_drift 가 non-empty 면 status='degraded' 자동
    전환. lenient env 가 켜져 있어도 실제 drift 가 없으면 status='ok' 유지
    ("우회 가능 상태"가 아닌 "실제 drift 모델 로드" 기준).
    """
    from serving.predictor import _is_feature_schema_lenient_allowed
    try:
        pred = get_predictor()
        ml_loaded = bool(pred._ml.loaded)
        hier_loaded = pred._hierarchical is not None and pred._hierarchical.loaded
        schema_drift = _collect_schema_drift(pred)
        lenient = _is_schema_lenient_active()
        lenient_allowed = lenient and _is_feature_schema_lenient_allowed()
        sunset_iso = _lenient_sunset_date_iso()

        degraded_reasons: list[str] = []
        if schema_drift:
            degraded_reasons.append(
                f"feature_schema_drift: {len(schema_drift)} unknown columns "
                f"({', '.join(schema_drift[:5])}{'...' if len(schema_drift) > 5 else ''})"
            )
        # Codex #6-followup — env 켜졌지만 sunset 으로 차단된 상태 명시
        if lenient and not lenient_allowed:
            degraded_reasons.append(
                f"feature_schema_lenient_blocked_by_sunset: {sunset_iso}"
            )
        status = "degraded" if degraded_reasons else "ok"

        return HealthResponse(
            status=status,
            model_loaded=ml_loaded or hier_loaded,
            rule_loaded=pred._safety_net is not None,
            version=APP_VERSION,
            uptime_sec=round(pred.uptime, 1),
            model_mode=_model_mode(pred),
            hierarchical_loaded=hier_loaded,
            schema_drift=schema_drift,
            feature_schema_lenient=lenient,
            feature_schema_lenient_allowed=lenient_allowed,
            feature_schema_lenient_sunset_date=sunset_iso,
            degraded_reasons=degraded_reasons,
            **_dl_info(pred),
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
            schema_drift=[],
            feature_schema_lenient=_is_schema_lenient_active(),
            feature_schema_lenient_allowed=False,
            feature_schema_lenient_sunset_date=_lenient_sunset_date_iso(),
            degraded_reasons=["predictor_not_initialized"],
            dl_loaded=False,
        )


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """로드된 모델 정보. 계층 모드만 로드된 경우 stage1/stage2 정보로 채움.

    Codex 2026-05-07 #1 — schema_drift 도 노출 (디버깅/감사용). /health 는
    운영 알림 기준, /model/info 는 staff 가 깊이 들여다볼 때 사용.
    """
    pred = get_predictor()
    ml = pred._ml
    hier = pred._hierarchical
    drift = _collect_schema_drift(pred)

    if ml.loaded:
        return ModelInfoResponse(
            model_type=ml._model_type,
            partition=ml._partition,
            n_features=len(ml._feature_names) if ml._feature_names else None,
            threshold=ml._threshold,
            schema_drift=drift,
            **_dl_info(pred),
        )
    if hier is not None and hier.loaded:
        return ModelInfoResponse(
            model_type="hierarchical",
            partition=None,
            n_features=len(hier.feature_cols) if hier.feature_cols else None,
            threshold=hier._thresholds.get("tau_red"),
            schema_drift=drift,
            **_dl_info(pred),
        )
    return ModelInfoResponse(
        model_type="none",
        partition=ml._partition,
        n_features=len(ml._feature_names) if ml._feature_names else None,
        threshold=None,
        schema_drift=drift,
        **_dl_info(pred),
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


@router.post("/admin/reload/dl")
async def reload_dl_model(
    body: DLReloadRequest,
    _: None = Depends(_require_admin),
):
    """DL bundle manifest/hash/lookback 핫스왑. 실제 DL 추론은 아직 비활성."""
    resolved = Path(body.bundle_dir).resolve()
    try:
        resolved.relative_to(_DL_MODEL_DIR)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "path_outside_model_dir",
                "message": (
                    f"DL bundle 경로는 허용된 디렉토리({_DL_MODEL_DIR}) 내부여야 합니다: "
                    f"{body.bundle_dir}"
                ),
            },
        )

    pred = get_predictor()
    try:
        pred.reload_dl(resolved)
    except LookbackMismatchError as e:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "lookback_mismatch", "message": str(e)},
        )
    except BundleHashMismatchError as e:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "bundle_hash_mismatch", "message": str(e)},
        )
    except BundleArtifactEmptyError as e:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "bundle_artifact_empty", "message": str(e)},
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "bundle_not_found", "message": str(e)},
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "bundle_invalid", "message": str(e)},
        )
    return {"status": "ok", "bundle_dir": str(resolved), **_dl_info(pred)}
