"""
하이브리드 예측기 (Rule + ML 앙상블)

최종등급 = max(Rule등급, ML등급)

Rule Safety Net:
  - rules/safety_net.py (Top 10 DDI, Triple Whammy, QT)
  - rules/duplicate_detector.py (ATC 중복약물)

ML 모델:
  - models/ddi_model_{partition}.pkl (XGBoost/LightGBM)
  - scripts/features/ (피처 추출)

로딩 전략:
  - 앱 시작 시 모든 데이터(DDI 매트릭스, 규칙, 모델) 메모리에 로드
  - 요청당 재로드 없음 (성능)
  - 모델 핫스왑: /admin/reload 엔드포인트로 무중단 교체
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .schemas import (
    DDIAlert, DLPredictionResult, DrugItem, PredictRequest, PredictResponse,
    RiskLevel, Severity, INTERVENTION_MAP,
)
from .dl_predictor import DLModel
from .hana_history import HANAHistoryProvider

logger = logging.getLogger(__name__)

# RequestFeatureBuilder.build() 가 생성하는 컬럼 집합 — 계층 모델 호환성 검증용
# dup_efmdc: edi→efmdc(HIRA 분류) 브릿지로 학습 정합 산출(P2). 브릿지 없으면 0.0 degraded.
_BUILDER_KNOWN_COLS: frozenset[str] = frozenset({
    "drug_count", "institution_count", "age", "sex_m",
    "ddi_contraindicated", "ddi_major", "ddi_moderate", "ddi_minor",
    "avg_drug_duration", "long_term_drug_count",
    "dup_same_ingredient", "dup_atc5", "dup_atc4", "dup_atc3", "dup_efmdc",
    "has_high_risk_drug", "has_renal_risk_drug", "has_hepatic_risk_drug",
    "cyp_risk_score", "cyp_high_risk_pairs", "cyp_max_enzyme_risk",
    "triple_whammy", "qt_risk_count", "drug_count_7d",
})

# serving 이 산출 못 하는 의도된 컬럼(현재 없음 — dup_efmdc 는 P2 에서 산출 추가됨).
# 본 allowlist 외 컬럼이 모델 feature_names 에 있으면 silent 0.0 fallback drift 위험 →
# _validate_feature_schema 가 strict fail (Codex 2026-05-07 P1).
_INTENTIONAL_FEATURE_ALLOWLIST: frozenset[str] = frozenset()
_FEATURE_ALLOWED: frozenset[str] = _BUILDER_KNOWN_COLS | _INTENTIONAL_FEATURE_ALLOWLIST


# Codex 2026-05-07 #6 — FEATURE_SCHEMA_LENIENT escape hatch sunset.
# env 미설정 시 코드 default deadline. today >= sunset 면 lenient 강제 차단.
# dup_efmdc allowlist sunset 은 별도 개념 (Codex 합의 — 본 PR 범위 밖).
_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT: date = date(2026, 8, 1)


def _is_feature_schema_lenient_allowed(today: "Optional[date]" = None) -> bool:
    """FEATURE_SCHEMA_LENIENT=1 escape hatch 가 sunset deadline 안인지 확인.

    today >= sunset_date 면 lenient 차단 (strict 강제). today 인자는
    monkeypatch 가능 — 테스트 시 임의 날짜 주입.

    env FEATURE_SCHEMA_LENIENT_SUNSET_DATE (YYYY-MM-DD) 미설정 → 코드 default.
    Invalid env date → 안전 측: lenient 차단 (운영 escape hatch 정책 — Codex 권고).
    """
    today = today or date.today()
    raw = os.environ.get("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "").strip()
    if raw:
        try:
            sunset = datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            logger.warning(
                "FEATURE_SCHEMA_LENIENT_SUNSET_DATE 형식 오류 — 안전 측으로 lenient "
                "차단 (strict 강제). input=%r (YYYY-MM-DD 필요)", raw,
            )
            return False
    else:
        sunset = _FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT
    return today < sunset


def _validate_feature_schema(
    feature_names: list[str] | None,
    model_label: str,
) -> tuple[list[str], bool]:
    """모델 feature_names ⊆ _FEATURE_ALLOWED 검증 — silent 0.0 drift 방지.

    기본 strict: 미허용 컬럼 발견 시 logger.error + return ok=False → 모델 로드 거부.
    FEATURE_SCHEMA_LENIENT=1 (env) 로 legacy 호환 우회 (warning + degraded 로 로드).
    단 Codex 2026-05-07 #6 sunset: today >= FEATURE_SCHEMA_LENIENT_SUNSET_DATE 면
    lenient 무시하고 strict 강제 (escape hatch 영구 고착 방지).

    오늘 추가된 hana_etl `_normalize_yyyymmdd` 와 같은 input contract validation
    패턴 — boundary 에서 strict, sunset 윈도우에 한해 lenient.

    Returns: (sorted_missing, ok)
    """
    if not feature_names:
        return [], True
    missing = sorted(set(feature_names) - _FEATURE_ALLOWED)
    if not missing:
        return [], True
    lenient_env = os.environ.get("FEATURE_SCHEMA_LENIENT", "").strip().lower() in (
        "1", "true", "yes",
    )
    lenient_active = lenient_env and _is_feature_schema_lenient_allowed()
    if lenient_active:
        logger.warning(
            "%s feature_names 중 RequestFeatureBuilder 미산출/허용 외 %d개: %s — "
            "FEATURE_SCHEMA_LENIENT=1 로 0.0 fallback 으로 로드 (degraded). "
            "운영에선 LENIENT 해제 후 학습/서빙 schema 정렬 필수.",
            model_label, len(missing), missing,
        )
        return missing, True
    if lenient_env and not lenient_active:
        logger.error(
            "%s feature_names 미허용 컬럼 %d개 — FEATURE_SCHEMA_LENIENT=1 이지만 "
            "sunset deadline 경과로 차단됨 (Codex #6). 학습/서빙 schema 정렬 필수. "
            "missing=%s", model_label, len(missing), missing,
        )
    else:
        logger.error(
            "%s feature_names 중 RequestFeatureBuilder 미산출/허용 외 %d개 — 로드 거부 "
            "(silent 0.0 drift 방지). 학습/서빙 schema 정렬 필요. 임시 우회 "
            "(sunset 안에서만): FEATURE_SCHEMA_LENIENT=1. missing=%s",
            model_label, len(missing), missing,
        )
    return missing, False

# 위험 약물 판정 상수 — 단일 출처: rules/risk_drug_constants.py
# (Codex 2026-05-06 ISSUE-3 단일화. 기준: drug_rules.yaml :123 high_risk_drugs).
from rules.risk_drug_constants import (
    HIGH_RISK_KEYWORDS as _HIGH_RISK_KEYWORDS,
    HIGH_RISK_ATC_PREFIXES as _HIGH_RISK_ATC_PREFIXES,
    RENAL_RISK_KEYWORDS as _RENAL_RISK_KEYWORDS,
    RENAL_RISK_ATC_PREFIXES as _RENAL_RISK_ATC_PREFIXES,
    HEPATIC_RISK_KEYWORDS as _HEPATIC_RISK_KEYWORDS,
    HEPATIC_RISK_ATC_PREFIXES as _HEPATIC_RISK_ATC_PREFIXES,
)


# ─────────────────────────────────────────────────────────────────────────────
# 규칙 기반 Safety Net 브릿지
# ─────────────────────────────────────────────────────────────────────────────

def _drugs_to_dup_input(drugs: list[DrugItem]) -> list[dict]:
    """DrugItem 목록 → DuplicateDetector 입력 형식."""
    return [
        {
            "name": d.drug_name or d.edi_code,
            "atc":  d.atc_code or "",
            "edi":  d.edi_code,
        }
        for d in drugs
    ]


def _detect_risk_flags(drugs: list[DrugItem]) -> tuple[bool, bool]:
    """약물 목록에서 신기능/간기능 저하 위험 플래그 계산."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.etl.prescription_aggregator import (
            _RENAL_RISK_KEYWORDS, _RENAL_RISK_ATC_PREFIXES,
            _HEPATIC_RISK_KEYWORDS, _HEPATIC_RISK_ATC_PREFIXES,
        )
        renal_prefixes = tuple(_RENAL_RISK_ATC_PREFIXES)
        hepatic_prefixes = tuple(_HEPATIC_RISK_ATC_PREFIXES)

        has_renal = any(
            any(kw in (d.drug_name or "").lower() for kw in _RENAL_RISK_KEYWORDS)
            or bool(d.atc_code and d.atc_code.startswith(renal_prefixes))
            for d in drugs
        )
        has_hepatic = any(
            any(kw in (d.drug_name or "").lower() for kw in _HEPATIC_RISK_KEYWORDS)
            or bool(d.atc_code and d.atc_code.startswith(hepatic_prefixes))
            for d in drugs
        )
        return has_renal, has_hepatic
    except Exception:
        return False, False


def _run_safety_net(
    drugs: list[DrugItem],
    patient_age: Optional[int] = None,
    sn_instance=None,
) -> tuple[RiskLevel, list[str], list[DDIAlert]]:
    """
    rules/safety_net.py 실행 → (등급, 이유 목록, DDI 알림 목록).

    sn_instance 제공 + assess() 런타임 오류 → RuntimeError 전파
      (초기화된 SafetyNet이 충돌하면 DDI 탐지 실패를 숨겨선 안 됨)
    sn_instance 미제공 + 모듈 없음/초기화 실패 → Normal 묵과
      (선택적 기능 미설치 환경 지원)
    """
    try:
        if sn_instance is None:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from rules.safety_net import SafetyNet
            sn = SafetyNet()
        else:
            sn = sn_instance

        has_renal, has_hepatic = _detect_risk_flags(drugs)

        # SafetyNet.assess는 list[str] (약물명 목록)을 기대
        drug_names = [d.drug_name or d.edi_code for d in drugs]
        assessment = sn.assess(
            drugs=drug_names,
            patient_age=patient_age,
            concurrent_drug_count=len(drugs),
            has_renal_risk=has_renal,
            has_hepatic_risk=has_hepatic,
        )

        level = RiskLevel(assessment.risk_grade)
        reasons = list(assessment.triggered_rules)

        alerts: list[DDIAlert] = []
        for ddi in assessment.ddi_pairs:
            try:
                severity = Severity(ddi.severity)
            except ValueError:
                severity = Severity.UNKNOWN
            alerts.append(DDIAlert(
                drug_a=ddi.drug_a,
                drug_b=ddi.drug_b,
                severity=severity,
                description=ddi.description,
                source=ddi.source,
            ))

        return level, reasons, alerts

    except ImportError:
        # sn_instance=None 경로에서만 발생 — 모듈 미설치 → 묵과
        logger.warning("Safety Net 미설치 (Normal 반환)")
        return RiskLevel.NORMAL, [], []
    except Exception as e:
        if sn_instance is not None:
            # 초기화된 Safety Net 런타임 오류(AttributeError 포함) → DDI 탐지 실패, 전파
            logger.error("Safety Net assess() 런타임 오류: %s", e)
            raise
        logger.warning("Safety Net 실행 오류 (Normal 반환): %s", e)
        return RiskLevel.NORMAL, [], []


def _run_duplicate_detector(drugs: list[DrugItem], dd_instance=None) -> tuple[int, list[str]]:
    """중복약물 탐지 → (중복건수, 이유 목록).

    dd_instance 제공 + detect() 런타임 오류 → 전파 (중복약물 탐지 실패 은닉 방지)
    dd_instance 미제공 + 모듈 없음/초기화 실패 → (0, []) 묵과
    """
    try:
        if dd_instance is None:
            from rules.duplicate_detector import DuplicateDetector
            dd = DuplicateDetector()
        else:
            dd = dd_instance

        drug_input = _drugs_to_dup_input(drugs)
        result = dd.detect(drug_input)

        dup_count = result.duplicate_level1_count + result.duplicate_level2_count
        reasons = []
        if result.duplicate_level1_count:
            reasons.append(f"동일성분중복 {result.duplicate_level1_count}건")
        if result.duplicate_level2_count:
            reasons.append(f"동일약리군중복 {result.duplicate_level2_count}건")
        return dup_count, reasons
    except ImportError:
        # dd_instance=None 경로에서만 발생 — 모듈 미설치 → 묵과
        logger.warning("DuplicateDetector 미설치 (중복 탐지 스킵)")
        return 0, []
    except Exception as e:
        if dd_instance is not None:
            # 초기화된 DuplicateDetector 런타임 오류 → 탐지 실패, 전파
            logger.error("DuplicateDetector detect() 런타임 오류: %s", e)
            raise
        logger.warning("DuplicateDetector 오류: %s", e)
        return 0, []


# ─────────────────────────────────────────────────────────────────────────────
# ML 모델 래퍼
# ─────────────────────────────────────────────────────────────────────────────

class MLModel:
    """저장된 ML 모델 로드 및 예측."""

    def __init__(self):
        self._model = None
        self._threshold: float = 0.5
        self._feature_names: list[str] = []
        self._artifact_version: int = 1
        self._partition: Optional[str] = None
        self._model_type: str = "none"
        self._scaler = None
        self._selector = None
        self._ensemble_weights = (1/3, 1/3, 1/3)
        self._gat_trainer = None   # GATTrainer instance (EnsembleTrainer3Way용)
        self._gat_graph_age_warned = False
        self._schema_drift: list[str] = []  # FEATURE_SCHEMA_LENIENT 로 로드된 missing 컬럼 trail

    @staticmethod
    def _verify_hash(path: Path, content: bytes) -> bool:
        """SHA-256 사이드카 파일(.sha256)이 있으면 content bytes로 무결성 검증.

        content를 인자로 받아 검증과 역직렬화가 동일한 바이트를 사용하도록 보장
        (TOCTOU 방지).
        """
        hash_path = path.with_suffix(path.suffix + ".sha256")
        if not hash_path.exists():
            logger.error("모델 해시 파일 없음 — 무결성 검증 불가, 로드 거부: %s", hash_path)
            return False
        expected = hash_path.read_text().strip().split()[0]
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            logger.error(
                "모델 파일 해시 불일치 — 로드 거부 (expected=%s, actual=%s)",
                expected[:16] + "…", actual[:16] + "…",
            )
            return False
        logger.info("모델 해시 검증 통과: %s", path)
        return True

    @staticmethod
    def _load_sidecar(path: Path) -> "tuple[Optional[Any], bool]":
        """sidecar pickle artifact (scaler/selector 등) hash 검증 + TOCTOU-safe 로드.

        Codex 2026-05-07 #2 — 주 모델 무결성 정책을 sidecar 까지 일관 확장.
        sidecar 도 read_bytes → _verify_hash → pickle.loads 동일 패턴.

        Returns: (loaded_obj, ok). ok=False 면 호출자(MLModel.load) 가 False return.
        """
        if not path.exists():
            logger.error("sidecar 파일 없음 — 로드 거부: %s", path)
            return None, False
        try:
            content = path.read_bytes()
        except Exception as e:
            logger.error("sidecar 읽기 실패: %s — %s", path, e)
            return None, False
        if not MLModel._verify_hash(path, content):
            return None, False
        try:
            obj = pickle.loads(content)
        except Exception as e:
            logger.error("sidecar pickle.loads 실패: %s — %s", path, e)
            return None, False
        logger.info("sidecar 로드 성공: %s", path)
        return obj, True

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        try:
            # 파일을 1회만 읽어 검증과 역직렬화에 동일한 바이트 사용 (TOCTOU 방지)
            content = path.read_bytes()
            if not self._verify_hash(path, content):
                return False
            state = pickle.loads(content)
            if not isinstance(state, dict):
                logger.error("모델 파일 형식 오류: dict가 아님 (%s)", type(state))
                return False
            self._model = state.get("model")
            self._threshold = state.get("best_threshold", 0.5)
            self._model_type = state.get("trainer_class", "unknown")
            self._feature_names = state.get("feature_names", [])
            self._artifact_version = state.get("artifact_version", 1)
            self._partition = state.get("partition")

            # DDI 시맨틱 버전 가드 (Q5) — ddi_* 피처를 쓰는 모델만 해당. 구 ddi 의미(ATC/
            # ddi=0)로 학습된 모델이 실 DDI 계산 서빙에 로드되면 train/serve 스큐(d201743).
            # pipeline.py 패키징이 ddi_feature_semantics_version 을 스탬프하므로(전 트레이너),
            # 불일치/누락 모두 하드 거부(계층 가드와 동일 정책). 구 미스탬프 번들은 재학습 필요.
            if any(str(c).startswith("ddi_") for c in self._feature_names):
                from scripts.etl.prescription_aggregator import (
                    DDI_FEATURE_SEMANTICS_VERSION as _cur_ddi_ver,
                )
                _bundle_ddi_ver = state.get("ddi_feature_semantics_version")
                if _bundle_ddi_ver != _cur_ddi_ver:
                    logger.error(
                        "단일 ML 모델 DDI 시맨틱 버전 불일치/누락 — 로드 거부(재학습 필요): "
                        "번들=%r vs 현재=%r",
                        _bundle_ddi_ver, _cur_ddi_ver,
                    )
                    self._model = None
                    self._feature_names = []
                    return False

            # Schema strict validation (Codex 2026-05-07 P1) — silent 0.0 drift 방지.
            # 학습 모델이 RequestFeatureBuilder 미산출 컬럼을 사용 중이면 로드 거부.
            _missing, _ok = _validate_feature_schema(
                self._feature_names, "단일 ML 모델",
            )
            if not _ok:
                # state 부분 적용 상태 정리
                self._model = None
                self._feature_names = []
                return False
            self._schema_drift = _missing

            # Codex 2026-05-07 #2 — sidecar (scaler/selector) 무결성 검증.
            # 직전까지: traversal continue + hash 미검증 + 파일 부재 warning 만.
            # 정책 일관성: state 에 path 명시되면 artifact 구성요소 — 부재/불일치/
            # traversal 모두 모델 로드 실패. 모든 검증 통과 후 instance state 반영.
            model_dir = path.parent
            loaded_sidecars: dict[str, "Any"] = {}
            for attr, key in [("_scaler", "scaler_path"), ("_selector", "selector_path")]:
                stored = state.get(key)
                if not stored:
                    continue
                candidate = (model_dir / stored).resolve()
                # path traversal 방어 — model_dir 외부 경로 거부
                try:
                    candidate.relative_to(model_dir.resolve())
                except ValueError:
                    logger.error(
                        "%s 경로가 model_dir 외부 — 로드 거부: %s", key, candidate
                    )
                    self._model = None
                    self._feature_names = []
                    self._schema_drift = []
                    return False
                obj, ok = MLModel._load_sidecar(candidate)
                if not ok:
                    self._model = None
                    self._feature_names = []
                    self._schema_drift = []
                    return False
                loaded_sidecars[attr] = obj

            # 모든 sidecar 검증 통과 → instance state 반영 (partial state 오염 방지)
            for attr, obj in loaded_sidecars.items():
                setattr(self, attr, obj)

            # Ensemble model: load from sub-model files
            if self._model is None and state.get("trainer_class") in ("EnsembleTrainer", "EnsembleTrainer3Way"):
                xgb_path = path.with_suffix(".xgb.pkl")
                lgb_path = path.with_suffix(".lgb.pkl")
                if xgb_path.exists() and lgb_path.exists():
                    try:
                        import pickle as _pk
                        xgb_content = xgb_path.read_bytes()
                        lgb_content = lgb_path.read_bytes()
                        if not self._verify_hash(xgb_path, xgb_content):
                            raise ValueError(f"xgb 서브모델 해시 불일치: {xgb_path}")
                        if not self._verify_hash(lgb_path, lgb_content):
                            raise ValueError(f"lgb 서브모델 해시 불일치: {lgb_path}")
                        xgb_state = _pk.loads(xgb_content)
                        lgb_state = _pk.loads(lgb_content)
                        weights = state.get("weights", (0.5, 0.5))
                        # Create a simple callable ensemble wrapper
                        class _EnsembleWrapper:
                            def __init__(self, xgb_model, lgb_model, w):
                                self._xgb = xgb_model
                                self._lgb = lgb_model
                                self._w = w
                            def predict_proba(self, X):
                                import numpy as np
                                p_xgb = self._xgb.predict_proba(X)[:, 1]
                                p_lgb = self._lgb.predict_proba(X)[:, 1]
                                prob = self._w[0] * p_xgb + self._w[1] * p_lgb
                                return np.column_stack([1 - prob, prob])
                        self._model = _EnsembleWrapper(xgb_state["model"], lgb_state["model"], weights)
                        logger.info("앙상블 모델 로드 완료: %s + %s", xgb_path.name, lgb_path.name)
                    except Exception as e:
                        logger.warning("앙상블 로드 실패: %s", e)

            # EnsembleTrainer3Way: GAT 서브모델 추가 로드
            if state.get("trainer_class") == "EnsembleTrainer3Way":
                gat_model_path = path.parent / "gat_model.pt"
                if not gat_model_path.exists():
                    raise RuntimeError(
                        f"EnsembleTrainer3Way는 gat_model.pt가 필수입니다: {gat_model_path}"
                    )
                try:
                    from scripts.train.gat_trainer import GATTrainer
                    self._gat_trainer = GATTrainer.load_gat(gat_model_path)
                    # 그래프 나이 경고
                    import json
                    from datetime import datetime, timezone
                    meta_path = path.parent / "gat_graph_meta.json"
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text())
                        built_at_str = meta.get("built_at", "")
                        if built_at_str:
                            try:
                                built_at = datetime.fromisoformat(built_at_str)
                                # Handle timezone-aware and naive datetimes
                                now = datetime.now(timezone.utc)
                                if built_at.tzinfo is None:
                                    built_at = built_at.replace(tzinfo=timezone.utc)
                                age_days = (now - built_at).days
                                if age_days > 180 and not self._gat_graph_age_warned:
                                    logger.warning(
                                        "gat_graph.pt 나이 %d일 (>180일) — 그래프 재빌드 권장",
                                        age_days,
                                    )
                                    self._gat_graph_age_warned = True
                            except ValueError:
                                pass
                    self._ensemble_weights = state.get("weights", (1/3, 1/3, 1/3))
                    logger.info("GATTrainer 로드 완료: %s", gat_model_path)
                except RuntimeError:
                    raise
                except Exception as e:
                    logger.warning("GATTrainer 로드 실패 (GAT 제외 모드): %s", e)
                    self._gat_trainer = None

            if self._model is None:
                logger.error("모델 로드 실패: _model이 None (앙상블 복원 실패 포함): %s", path)
                return False
            logger.info("ML 모델 로드: %s (threshold=%.3f)", path, self._threshold)
            return True
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning("ML 모델 로드 실패: %s", e)
            return False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def predict_proba(self, X: np.ndarray) -> float:
        """단일 샘플 Red 확률 반환."""
        if self._model is None:
            return 0.0
        try:
            prob = self._model.predict_proba(X.reshape(1, -1))[0, 1]
            return float(prob)
        except Exception as e:
            logger.warning("ML 예측 오류: %s", e)
            return 0.0

    def predict_proba_gat(
        self,
        X: np.ndarray,
        drug_codes: list[str],
    ) -> float:
        """
        GAT 포함 앙상블 예측.

        Parameters
        ----------
        X          : [1, feature_dim] tabular 피처 (스케일링 적용 후)
        drug_codes : 요청 내 약물 코드 목록

        Returns
        -------
        최종 DDI 위험 확률 (0~1)
        """
        from itertools import combinations

        # tabular 예측 (기존 경로) — predict_proba()는 이미 float 반환
        base_prob = self.predict_proba(X)

        if self._gat_trainer is None or not getattr(self._gat_trainer, "_trained", False) or len(drug_codes) < 2:
            return base_prob

        known_drug_to_idx = self._gat_trainer._graph_builder.drug_to_idx
        unknown_codes = sorted({code for code in drug_codes if code not in known_drug_to_idx})
        if unknown_codes:
            logger.warning(
                "알 수 없는 약물 코드 포함 — 요청 전체에서 GAT 제외: %s",
                ", ".join(unknown_codes),
            )
            return base_prob

        # 모든 약물쌍 GAT 스코어 → max 집계
        valid_scores = []
        for drug_a, drug_b in combinations(drug_codes, 2):
            score = self._gat_trainer.predict_pair_proba(drug_a, drug_b)
            if score is not None:
                valid_scores.append(score)

        if not valid_scores:
            # 모든 쌍 미지 약물 → GAT 제외, tabular만 사용
            return base_prob

        p_gat = float(max(valid_scores))

        weights = getattr(self, "_ensemble_weights", (1/3, 1/3, 1/3))
        w1, w2, w3 = weights
        tab_weight = w1 + w2
        total = tab_weight + w3
        return (tab_weight * base_prob + w3 * p_gat) / (total or 1.0)

    def classify(self, prob: float) -> RiskLevel:
        """확률 → 위험등급 변환."""
        if prob >= self._threshold:
            return RiskLevel.RED
        elif prob >= self._threshold * 0.6:
            return RiskLevel.YELLOW
        elif prob >= self._threshold * 0.3:
            return RiskLevel.GREEN
        return RiskLevel.NORMAL


# ─────────────────────────────────────────────────────────────────────────────
# 계층 예측기 (Stage 1 Red + Stage 2 Yellow-subtype)
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalPredictor:
    """계층 분류 모델 래퍼 — stage1_red.joblib + stage2_yellow.joblib + stage_meta.json.

    stage_meta.json 의 stage{1,2}_sha256 로 joblib 무결성 검증.
    predict_risk_single() 이 2단 임계값(τ_red, τ_review) 분기 결과 dict 반환.
    """

    def __init__(self):
        self._stage1 = None
        self._stage2 = None
        self._encoder = None
        self._classes_present: list[int] = []
        self._thresholds: dict[str, float] = {}
        self._feature_cols: list[str] = []
        self._meta: dict = {}

    def _clear_state(self) -> None:
        """Clear loaded/partial hierarchical model state after a failed load."""
        self._stage1 = None
        self._stage2 = None
        self._encoder = None
        self._classes_present = []
        self._thresholds = {}
        self._feature_cols = []
        self._meta = {}

    def load(self, model_dir: str | Path) -> bool:
        model_dir = Path(model_dir)
        meta_path = model_dir / "stage_meta.json"
        p1 = model_dir / "stage1_red.joblib"
        p2 = model_dir / "stage2_yellow.joblib"
        if not meta_path.exists() or not p1.exists() or not p2.exists():
            logger.error(
                "계층 모델 파일 누락 — 로드 실패: meta=%s stage1=%s stage2=%s",
                meta_path.exists(), p1.exists(), p2.exists(),
            )
            self._clear_state()
            return False
        try:
            import json
            import joblib
            self._meta = json.loads(meta_path.read_text())
            self._thresholds = self._meta["thresholds"]
            self._feature_cols = self._meta["feature_cols"]
            # 라벨 공간 정합 가드 (fail-fast, joblib 로드 전) — 구 번들(예: 6-class Y_MIX)이
            # 현재 STAGE2_LABELS(7-class)와 불일치하면 predict_risk 의 local→global 슬롯이
            # 어긋나 silent train/serve skew(d201743 전례)가 발생한다. 메타의 stage2_labels 가
            # 현재와 정확히 같지 않거나 없으면 로드 거부(재학습 필요).
            from hana_app.core.hierarchical_runner import (
                STAGE2_LABELS as _CURRENT_STAGE2_LABELS,
            )
            _meta_labels = self._meta.get("stage2_labels")
            if _meta_labels is None or list(_meta_labels) != list(_CURRENT_STAGE2_LABELS):
                logger.error(
                    "계층 모델 stage2 라벨 공간 불일치 — 로드 거부(재학습 필요): "
                    "번들=%s vs 현재(%d)=%s",
                    _meta_labels, len(_CURRENT_STAGE2_LABELS), list(_CURRENT_STAGE2_LABELS),
                )
                self._clear_state()
                return False
            # DDI 피처 시맨틱 버전 가드 (Q5) — 구 ddi=0/ATC 경로로 학습된 번들이 실 DDI
            # 계산 서빙에 로드되면 train/serve 스큐(d201743). 메타 버전이 현재와 다르거나
            # 없으면 로드 거부(재학습 필요). ddi_* 가 feature_cols 에 포함되므로 적용.
            from scripts.etl.prescription_aggregator import (
                DDI_FEATURE_SEMANTICS_VERSION as _CUR_DDI_VER,
            )
            _ddi_ver = self._meta.get("ddi_feature_semantics_version")
            if _ddi_ver != _CUR_DDI_VER:
                logger.error(
                    "계층 모델 DDI 시맨틱 버전 불일치 — 로드 거부(재학습 필요): "
                    "번들=%r vs 현재=%r",
                    _ddi_ver, _CUR_DDI_VER,
                )
                self._clear_state()
                return False
            if not self._feature_cols:
                logger.error(
                    "계층 모델 feature_cols 비어 있음 — 로드 거부: %s",
                    model_dir,
                )
                self._clear_state()
                return False
            _missing, _schema_ok = _validate_feature_schema(
                self._feature_cols, "계층 모델 (load)",
            )
            if not _schema_ok:
                logger.error(
                    "계층 모델 feature schema 검증 실패 — 로드 거부: %s (missing=%s)",
                    model_dir, _missing,
                )
                self._clear_state()
                return False

            for p, key in ((p1, "stage1_sha256"), (p2, "stage2_sha256")):
                expected = self._meta.get(key)
                if not expected:
                    logger.warning("%s 메타 누락 — 무결성 검증 스킵: %s", key, p.name)
                    continue
                actual = hashlib.sha256(p.read_bytes()).hexdigest()
                if actual != expected:
                    logger.error(
                        "계층 모델 해시 불일치 — 로드 거부: %s (expected=%s, actual=%s)",
                        p.name, expected[:16] + "…", actual[:16] + "…",
                    )
                    self._clear_state()
                    return False

            self._stage1 = joblib.load(p1)
            bundle = joblib.load(p2)
            self._stage2 = bundle["model"]
            self._encoder = bundle["encoder"]
            self._classes_present = list(bundle["classes_present"])
            # 인코더 객체 정합 가드 (belt-and-suspenders) — encoder.classes_ 는 학습 시
            # 항상 full STAGE2_LABELS 로 강제되고(_make_stage2_encoder), classes_present 는
            # 학습에 존재한 클래스의 global 인덱스 부분집합이다. 직렬화된 인코더가 현재
            # STAGE2_LABELS 와 정확히 같지 않거나 classes_present 인덱스가 범위를 벗어나면
            # (메타만 손편집된 혼합 번들 등) 거부.
            _enc_classes = [str(c) for c in self._encoder.classes_]
            if _enc_classes != list(_CURRENT_STAGE2_LABELS) or any(
                not (0 <= int(gi) < len(_CURRENT_STAGE2_LABELS))
                for gi in self._classes_present
            ):
                logger.error(
                    "계층 모델 encoder.classes_/classes_present 정합 실패 — 로드 거부: "
                    "enc=%s classes_present=%s 현재=%s",
                    _enc_classes, self._classes_present, list(_CURRENT_STAGE2_LABELS),
                )
                self._clear_state()
                return False
            logger.info(
                "계층 모델 로드 완료: %s (τ_red=%.3f, τ_review=%.3f, %d features)",
                model_dir,
                self._thresholds["tau_red"],
                self._thresholds["tau_review"],
                len(self._feature_cols),
            )
            return True
        except Exception as e:
            logger.warning("계층 모델 로드 실패: %s", e)
            self._clear_state()
            return False

    @property
    def loaded(self) -> bool:
        return self._stage1 is not None and self._stage2 is not None

    @property
    def feature_cols(self) -> list[str]:
        return list(self._feature_cols)

    @property
    def feature_semantics_version(self):
        """번들 메타의 feature_semantics_version (triple_whammy/위험플래그 활성 버전). 없으면 None."""
        return self._meta.get("feature_semantics_version")

    def predict_risk_single(self, X: np.ndarray) -> dict:
        """단일 샘플 계층 추론 — 반환: {risk_level, p_red, stage2_probs, red_suspect, action}."""
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from hana_app.core.hierarchical_runner import predict_risk
        X_row = np.asarray(X).reshape(1, -1)
        results = predict_risk(
            X_row,
            self._stage1,
            self._stage2,
            self._encoder,
            self._thresholds,
            classes_present=self._classes_present,
        )
        return results[0]


# ─────────────────────────────────────────────────────────────────────────────
# 피처 빌더 (요청 → ML 입력 벡터)
# ─────────────────────────────────────────────────────────────────────────────

class RequestFeatureBuilder:
    """
    PredictRequest → ML 피처 벡터 (numpy).
    ETL+Feature 파이프라인 없이 온라인 피처 계산.
    """

    def __init__(
        self,
        ddi_matrix: Optional[pd.DataFrame] = None,
        cyp_extractor=None,
        code_standardizer=None,
    ):
        self._ddi = ddi_matrix
        self._cyp = cyp_extractor
        self._std = code_standardizer

        # DDI 조회용 인덱스
        self._ddi_index: dict[frozenset, str] = {}
        if ddi_matrix is not None and "drug_a_atc" in ddi_matrix.columns:
            for row in ddi_matrix.itertuples(index=False):
                key = frozenset({str(row.drug_a_atc), str(row.drug_b_atc)})
                sev = str(row.severity)
                existing = self._ddi_index.get(key)
                order = {"Contraindicated": 4, "Major": 3, "Moderate": 2, "Minor": 1}
                if existing is None or order.get(sev, 0) > order.get(existing, 0):
                    self._ddi_index[key] = sev

    def _drug_master(self):
        try:
            return self._std.drug_master if self._std is not None else None
        except Exception:
            return None

    def _build_ddi_records(self, drugs: list, ref: date):
        """edi→wk 로 PrescriptionRecord 재구성 + wk→표시명 맵. (recs, wk_to_name) 반환.

        미매핑 EDI 는 DDI 평가서 제외(degraded, "미매핑≠음성") + 경고. 학습 DDI 피처와
        DDI 알림이 이 동일 재구성을 공유한다.
        """
        from scripts.etl.models import PrescriptionRecord
        recs = []
        wk_to_name: dict[str, str] = {}
        n_unmapped = 0
        for i, d in enumerate(drugs):
            wk = self._std.get_wk(d.edi_code) if self._std is not None else None
            if not wk:
                n_unmapped += 1
                continue
            sd = d.start_date or ref
            td = int(d.total_days or 1)
            recs.append(PrescriptionRecord(
                patient_id="req", institution_id=str(d.institution_id or ""),
                bill_no=f"R{i}", wk_compn_cd=wk, edi_code=d.edi_code,
                start_date=sd, end_date=sd + timedelta(days=td - 1), total_days=td,
                source="serving",
            ))
            wk_to_name.setdefault(wk, d.drug_name or d.edi_code)
        if n_unmapped:
            logger.warning(
                "DDI: edi→wk 미매핑 %d/%d 건 — DDI 평가서 제외(degraded, 미매핑≠음성)",
                n_unmapped, len(drugs),
            )
        return recs, wk_to_name

    def _count_ddi(self, drugs: list, ref: date) -> dict[str, int]:
        """학습과 동일 경로로 DDI 심각도 카운트 (train/serve parity).

        edi→wk → PrescriptionRecord 재구성 → calculate_overlaps_for_patient(학습과 동일)
        → count_ddi_severities(공용 단일출처). std/ddi_matrix/drug_master 부재·약물<2 면 0.
        """
        zero = {"Contraindicated": 0, "Major": 0, "Moderate": 0, "Minor": 0}
        drug_master = self._drug_master()
        if self._std is None or self._ddi is None or drug_master is None:
            return dict(zero)
        from scripts.etl.overlap_calculator import calculate_overlaps_for_patient
        from scripts.etl.prescription_aggregator import count_ddi_severities
        recs, _ = self._build_ddi_records(drugs, ref)
        if len(recs) < 2:
            return dict(zero)
        overlaps = calculate_overlaps_for_patient(recs, window_days=90)
        return count_ddi_severities(overlaps, self._ddi, drug_master)

    def ddi_alert_pairs(self, drugs: list, ref: date) -> list[tuple[str, str, str]]:
        """동시복용 DDI 쌍 → (name_a, name_b, severity). DDI 피처와 **동일 경로**(일관성)."""
        drug_master = self._drug_master()
        if self._std is None or self._ddi is None or drug_master is None:
            return []
        from scripts.etl.overlap_calculator import calculate_overlaps_for_patient
        from scripts.etl.prescription_aggregator import ddi_pair_severities
        recs, wk_to_name = self._build_ddi_records(drugs, ref)
        if len(recs) < 2:
            return []
        overlaps = calculate_overlaps_for_patient(recs, window_days=90)
        return [
            (wk_to_name.get(pair.drug_a_wk_compn, pair.drug_a_wk_compn),
             wk_to_name.get(pair.drug_b_wk_compn, pair.drug_b_wk_compn), sev)
            for pair, sev in ddi_pair_severities(overlaps, self._ddi, drug_master)
        ]

    def _count_dup_features(self, drugs: list, ref: date) -> dict | None:
        """학습 정합 drug_count / drug_count_7d / dup_same_ingredient (P2).

        edi→wk(code_standardizer)→DrugMaster 로 학습과 동일 함수 호출:
          drug_count        = len(expand_drug_count(매핑 wk)) + 미매핑 약물 수
          drug_count_7d     = get_concurrent_drug_count(전 약물 record, ref)
          dup_same_ingredient/dup_atc5/4/3 = _fill_dup_features(매핑 record, drug_master)
        브릿지/DrugMaster 부재 시 None → build() 가 기존(edi 기반) fallback 사용.
        미매핑 EDI: drug_count 엔 +1(약물은 약물), dup 엔 제외(성분 모름=degraded).
        """
        drug_master = self._drug_master()
        if self._std is None or drug_master is None:
            return None
        from scripts.etl.models import PrescriptionRecord, PatientFeatures
        from scripts.etl.overlap_calculator import get_concurrent_drug_count
        from scripts.etl.prescription_aggregator import _fill_dup_features

        all_recs, mapped_recs, mapped_wks = [], [], []
        n_unmapped = 0
        for i, d in enumerate(drugs):
            wk = self._std.get_wk(d.edi_code)
            sd = d.start_date or ref
            td = int(d.total_days or 1)
            rec = PrescriptionRecord(
                patient_id="req", institution_id=str(d.institution_id or ""), bill_no=f"R{i}",
                wk_compn_cd=(wk or str(d.edi_code)), edi_code=d.edi_code,
                efmdc_clsf_no=(self._std.get_efmdc(d.edi_code) if wk else None),
                start_date=sd, end_date=sd + timedelta(days=td - 1), total_days=td, source="serving",
            )
            all_recs.append(rec)
            if wk:
                mapped_recs.append(rec)
                mapped_wks.append(wk)
            else:
                n_unmapped += 1
        # dup 피처: 학습 _fill_dup_features 공용 호출(단일출처, dup_groups 미사용→None).
        # atc_code 없음(학습 df_row_to_record 도 미세팅) → dup_same_ingredient=성분경로,
        # dup_atc5/4/3=0 학습 정합. dup_efmdc 는 edi→efmdc(HIRA 분류, records efmdc 와
        # 99.9% 동일) 로 채워 학습과 정합.
        _dup = PatientFeatures(patient_id="req", window_start=ref, window_end=ref)
        _fill_dup_features(_dup, mapped_recs, dup_groups=None, drug_master=drug_master)
        return {
            "drug_count": float(len(drug_master.expand_drug_count(mapped_wks)) + n_unmapped),
            "drug_count_7d": float(get_concurrent_drug_count(all_recs, ref)),
            "dup_same_ingredient": float(_dup.dup_same_ingredient),
            "dup_atc5": float(_dup.dup_atc5),
            "dup_atc4": float(_dup.dup_atc4),
            "dup_atc3": float(_dup.dup_atc3),
            "dup_efmdc": float(_dup.dup_efmdc),
        }

    def _rule_namespace(self, drugs: list, ref: date, patient_age=None):
        """학습 룰 함수가 기대하는 피처 객체를 edi→wk→DrugMaster 로 구성(학습과 동일 식별자).

        결정적 백스톱(red_triggers/severe_triggers) 공용. std/drug_master 부재 시 None.
        """
        from types import SimpleNamespace
        drug_master = self._drug_master()
        if self._std is None or drug_master is None:
            return None
        from scripts.etl.prescription_aggregator import (
            detect_triple_whammy, detect_risk_drug,
            _HIGH_RISK_KEYWORDS, _RENAL_RISK_KEYWORDS, _HEPATIC_RISK_KEYWORDS,
        )
        ddi = self._count_ddi(drugs, ref)
        cdup = self._count_dup_features(drugs, ref) or {}
        mapped_wks = [w for w in (self._std.get_wk(d.edi_code) for d in drugs) if w]
        return SimpleNamespace(
            ddi_contraindicated=ddi.get("Contraindicated", 0),
            ddi_major=ddi.get("Major", 0),
            triple_whammy=detect_triple_whammy(mapped_wks, drug_master),
            drug_count=cdup.get("drug_count", float(len({d.edi_code for d in drugs}))),
            has_high_risk_drug=detect_risk_drug(mapped_wks, drug_master, _HIGH_RISK_KEYWORDS),
            has_renal_risk_drug=detect_risk_drug(mapped_wks, drug_master, _RENAL_RISK_KEYWORDS),
            has_hepatic_risk_drug=detect_risk_drug(mapped_wks, drug_master, _HEPATIC_RISK_KEYWORDS),
            age=patient_age,
        )

    def red_triggers(self, drugs: list, ref: date, patient_age=None) -> set:
        """학습 collect_red_triggers 를 edi→wk 피처에 적용 → 결정적 Red 트리거 집합.

        model-independent 안전 백스톱(Phase 2-2). std/drug_master 부재 시 빈 set(비차단).
        """
        from scripts.etl.clinical_rules import collect_red_triggers
        f = self._rule_namespace(drugs, ref, patient_age)
        return collect_red_triggers(f) if f is not None else set()

    def severe_triggers(self, drugs: list, ref: date, patient_age=None) -> set:
        """학습 collect_severe_immediate_triggers 를 edi→wk 피처에 적용 → 중증(Y_TRIPLE) 트리거.

        triple_whammy/10drug+고위험/고령+장기. model-independent severe→Y_TRIPLE 백스톱(floor)용.
        std/drug_master 부재 시 빈 set(비차단).
        """
        from scripts.etl.clinical_rules import collect_severe_immediate_triggers
        f = self._rule_namespace(drugs, ref, patient_age)
        return collect_severe_immediate_triggers(f) if f is not None else set()

    def build(self, req: PredictRequest, feature_names=None, scaler=None, selector=None,
              rule_features_active: bool = False) -> tuple[np.ndarray, dict]:
        """
        요청 → (피처 벡터, 피처 딕셔너리).
        피처 딕셔너리는 logging/debugging용.
        """
        ref = req.reference_date or date.today()
        drugs = req.drugs

        # EDI → ATC 코드 보완
        if self._std is not None:
            for d in drugs:
                if not d.atc_code:
                    atc, name = self._std.lookup_edi(d.edi_code)
                    d.atc_code = atc
                    if not d.drug_name and name:
                        d.drug_name = name

        atc_codes = [d.atc_code for d in drugs if d.atc_code]

        # ── 기본 피처 ──────────────────────────────────────────────────────
        # P2: drug_count/drug_count_7d/dup_same_ingredient 를 학습과 동일 함수로(edi→wk).
        # 브릿지 없으면 _cdup=None → 기존 edi 기반 fallback.
        _cdup = self._count_dup_features(drugs, ref)
        feat: dict[str, float] = {}
        feat["drug_count"]        = (_cdup["drug_count"] if _cdup is not None
                                     else float(len({d.edi_code for d in drugs})))
        feat["institution_count"] = float(len({d.institution_id for d in drugs
                                               if d.institution_id}))
        feat["age"]               = float(req.patient_age or 0)
        feat["sex_m"]             = float(req.patient_sex == "M") if req.patient_sex else 0.5

        # ── DDI 피처 (학습 정합: edi→wk→DrugMaster→DB-code, overlap 쌍 기준) ──
        # 학습(_fill_ddi_features)과 동일하게 count_ddi_severities 를 호출해 train/serve
        # 스큐를 구조적으로 차단(d201743 전례). 날짜 미겹침 쌍은 overlap 0 → 카운트 0.
        ddi_counts = self._count_ddi(drugs, ref)
        feat["ddi_contraindicated"] = float(ddi_counts["Contraindicated"])
        feat["ddi_major"]           = float(ddi_counts["Major"])
        feat["ddi_moderate"]        = float(ddi_counts["Moderate"])
        feat["ddi_minor"]           = float(ddi_counts["Minor"])

        # ── 투여일수 피처 ──────────────────────────────────────────────────
        durations = [d.total_days for d in drugs]
        feat["avg_drug_duration"]    = float(sum(durations) / len(durations)) if durations else 0.0
        feat["long_term_drug_count"] = float(sum(1 for d in durations if d >= 30))

        # ── ATC 중복 피처 ──────────────────────────────────────────────────
        # ETL 계약(scripts/etl/prescription_aggregator.py:325-343)과 정렬:
        #   dup_atc5 = full 7-char ATC, dup_atc4 = 5-prefix, dup_atc3 = 4-prefix
        from collections import Counter
        cnt5 = Counter(atc_codes)                                       # full 7-char
        cnt4 = Counter(c[:5] for c in atc_codes if len(c) >= 5)         # 5-prefix
        cnt3 = Counter(c[:4] for c in atc_codes if len(c) >= 4)         # 4-prefix
        feat["dup_same_ingredient"] = (_cdup["dup_same_ingredient"] if _cdup is not None
                                       else float(sum(1 for v in cnt5.values() if v >= 2)))
        # dup_atc5/4/3: 브릿지 있으면 학습 정합(_fill_dup_features, atc_code 미보유→0;
        # 학습 records 도 atc_code 없어 항상 0). 브릿지 없으면 기존 ATC 기반 fallback.
        feat["dup_atc5"] = (_cdup["dup_atc5"] if _cdup is not None
                            else float(sum(1 for v in cnt5.values() if v >= 2)))
        feat["dup_atc4"] = (_cdup["dup_atc4"] if _cdup is not None
                            else float(sum(1 for v in cnt4.values() if v >= 2)))
        feat["dup_atc3"] = (_cdup["dup_atc3"] if _cdup is not None
                            else float(sum(1 for v in cnt3.values() if v >= 2)))
        # dup_efmdc: 약효분류 중복 — 브릿지 있으면 edi→efmdc(HIRA 분류)로 학습 정합,
        # 없으면 0.0(efmdc 출처 부재 degraded).
        feat["dup_efmdc"] = (_cdup["dup_efmdc"] if _cdup is not None else 0.0)

        # ── CYP 피처 ──────────────────────────────────────────────────────
        if self._cyp and atc_codes:
            cyp_feat = self._cyp.extract(atc_codes)
            feat.update(cyp_feat)
        else:
            feat["cyp_risk_score"]      = 0.0
            feat["cyp_high_risk_pairs"] = 0.0
            feat["cyp_max_enzyme_risk"] = 0.0

        # ── 위험 약물 플래그 + triple_whammy ──────────────────────────────
        # rule_features_active(로드된 번들 feature_semantics_version=rulefeat.v1) 면 학습과
        # 동일 edi→wk→DrugMaster components 키워드로 산출(Phase 2-2). 아니면(구 번들/tabular)
        # 기존 atc/name 경로 — 실 edi 요청은 ~0 으로 구 번들(=0)과 정합 유지(skew 방지 게이팅).
        _rf_dm = self._drug_master() if rule_features_active else None
        if _rf_dm is not None and self._std is not None:
            from scripts.etl.prescription_aggregator import detect_triple_whammy, detect_risk_drug
            _wks = [w for w in (self._std.get_wk(d.edi_code) for d in drugs) if w]
            feat["has_high_risk_drug"]    = float(detect_risk_drug(_wks, _rf_dm, _HIGH_RISK_KEYWORDS))
            feat["has_renal_risk_drug"]   = float(detect_risk_drug(_wks, _rf_dm, _RENAL_RISK_KEYWORDS))
            feat["has_hepatic_risk_drug"] = float(detect_risk_drug(_wks, _rf_dm, _HEPATIC_RISK_KEYWORDS))
            feat["triple_whammy"]         = float(detect_triple_whammy(_wks, _rf_dm))
        else:
            drug_names_lower = [(d.drug_name or "").lower() for d in drugs]
            feat["has_high_risk_drug"]    = float(_has_risk_drug(
                drug_names_lower, atc_codes, _HIGH_RISK_KEYWORDS, _HIGH_RISK_ATC_PREFIXES))
            feat["has_renal_risk_drug"]   = float(_has_risk_drug(
                drug_names_lower, atc_codes, _RENAL_RISK_KEYWORDS, _RENAL_RISK_ATC_PREFIXES))
            feat["has_hepatic_risk_drug"] = float(_has_risk_drug(
                drug_names_lower, atc_codes, _HEPATIC_RISK_KEYWORDS, _HEPATIC_RISK_ATC_PREFIXES))
            feat["triple_whammy"] = float(_check_triple_whammy(atc_codes))

        # QT 카운트 (단순 ATC prefix 기반)
        feat["qt_risk_count"] = float(_count_qt_drugs(atc_codes))
        feat["drug_count_7d"] = (_cdup["drug_count_7d"] if _cdup is not None
                                 else feat["drug_count"])  # 학습=concurrent@ref, fallback=copy

        # Align to training feature order
        if feature_names:
            aligned = {name: feat.get(name, 0.0) for name in feature_names}
        else:
            aligned = feat

        # Apply scaler (expects DataFrame)
        import pandas as pd
        df = pd.DataFrame([aligned])
        if scaler is not None:
            try:
                df = pd.DataFrame(scaler.transform(df), columns=df.columns)
            except Exception as e:
                logger.warning("Scaler 적용 실패 (원본 사용): %s", e)

        # Apply selector (expects DataFrame)
        if selector is not None:
            try:
                cols_before = list(df.columns)
                arr = selector.transform(df)
                if hasattr(selector, 'get_support'):
                    # get_support() returns boolean mask → apply to column names
                    selected_cols = [c for c, keep in zip(cols_before, selector.get_support()) if keep]
                else:
                    selected_cols = cols_before[:arr.shape[1]]
                df = pd.DataFrame(arr, columns=selected_cols)
            except Exception as e:
                logger.warning("Selector 적용 실패 (원본 사용): %s", e)

        vec = df.values.flatten().astype(float)
        return vec, feat


def _has_risk_drug(
    names_lower: list[str],
    atc_codes: list[str],
    keywords: frozenset[str],
    atc_prefixes: tuple[str, ...],
) -> bool:
    """약물 이름 또는 ATC prefix 기반 위험 약물 포함 여부."""
    for name in names_lower:
        if any(kw in name for kw in keywords):
            return True
    for atc in atc_codes:
        if atc and atc.startswith(atc_prefixes):
            return True
    return False


def _check_triple_whammy(atc_codes: list[str]) -> bool:
    """ACEi/ARB + K보존이뇨제 + NSAIDs 동시 복용 체크."""
    has_acei_arb  = any(c.startswith(("C09AA", "C09CA")) for c in atc_codes)
    has_k_sparing = any(c.startswith(("C03DA", "C03DB")) for c in atc_codes)
    has_nsaid     = any(c.startswith(("M01A", "M01B", "N02BA01")) for c in atc_codes)
    return has_acei_arb and has_k_sparing and has_nsaid


def _count_qt_drugs(atc_codes: list[str]) -> int:
    """QT 연장 위험 약물 수 (주요 ATC prefix 기반)."""
    qt_prefixes = ("N05AD", "J01MA", "P01BA", "C01BD", "J01FA")
    return sum(1 for c in atc_codes if c.startswith(qt_prefixes))


# ─────────────────────────────────────────────────────────────────────────────
# 하이브리드 예측기 (메인)
# ─────────────────────────────────────────────────────────────────────────────

class HybridPredictor:
    """
    Rule + ML 하이브리드 예측기.
    앱 시작 시 싱글턴으로 생성, 요청마다 재사용.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        ddi_matrix_path: str | Path = "data/processed/ddi_matrix_final.parquet",
        drug_index_path: str | Path = "data/processed/drug_name_index.parquet",
        cyp_matrix_path: str | Path = "data/processed/cyp_matrix.parquet",
        hierarchical_model_dir: Optional[str | Path] = None,
        dl_history_provider: Optional[HANAHistoryProvider] = None,
    ):
        self._start_time = time.time()
        self._ml_lock = threading.RLock()
        self._hier_lock = threading.RLock()
        self._dl_lock = threading.RLock()
        self._ml = MLModel()
        self._hierarchical: Optional[HierarchicalPredictor] = None
        self._dl = DLModel()
        self._dl_history_provider = dl_history_provider
        self._ddi_matrix: Optional[pd.DataFrame] = None
        self._cyp = None
        self._std = None

        # DDI 매트릭스 로드
        if Path(ddi_matrix_path).exists():
            self._ddi_matrix = pd.read_parquet(ddi_matrix_path)
            logger.info("DDI 매트릭스 로드: %d행", len(self._ddi_matrix))

        # 코드 표준화기
        if Path(drug_index_path).exists():
            try:
                from scripts.etl.code_standardizer import CodeStandardizer
                self._std = CodeStandardizer(index_path=drug_index_path)
                logger.info("코드 표준화기 로드")
            except Exception as e:
                logger.warning("코드 표준화기 로드 실패: %s", e)

        # CYP 피처 추출기
        if Path(cyp_matrix_path).exists() and Path(drug_index_path).exists():
            try:
                from scripts.features.cyp_features import CYPFeatureExtractor
                self._cyp = CYPFeatureExtractor(
                    cyp_matrix_path=cyp_matrix_path,
                    drug_index_path=drug_index_path,
                )
                logger.info("CYP 피처 추출기 로드")
            except Exception as e:
                logger.warning("CYP 추출기 로드 실패: %s", e)

        # ML 모델 로드 — 계층 모드 디렉터리 우선, 실패/미설정 시 단일 모델 fallback
        if hierarchical_model_dir:
            _hdir = Path(hierarchical_model_dir)
            if not _hdir.exists():
                logger.warning(
                    "HIERARCHICAL_MODEL_DIR 경로 없음 — 단일 모델 fallback: %s",
                    hierarchical_model_dir,
                )
            else:
                hp = HierarchicalPredictor()
                if hp.load(str(_hdir)):
                    _missing, _ok = _validate_feature_schema(
                        hp.feature_cols, "계층 모델 (init)",
                    )
                    if not _ok:
                        logger.warning(
                            "HIERARCHICAL_MODEL_DIR schema 거부 — 단일 모델 fallback: %s",
                            hierarchical_model_dir,
                        )
                    else:
                        self._hierarchical = hp
                else:
                    logger.warning(
                        "HIERARCHICAL_MODEL_DIR 로드 실패 — 단일 모델 fallback: %s",
                        hierarchical_model_dir,
                    )
        if self._hierarchical is None and model_path and Path(model_path).exists():
            self._ml.load(model_path)

        # Safety Net 싱글턴 (요청당 재생성 방지)
        self._safety_net = None
        self._dup_detector = None
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent))
            from rules.safety_net import SafetyNet
            self._safety_net = SafetyNet()
            logger.info("SafetyNet 싱글턴 초기화 완료")
        except Exception as e:
            logger.warning("SafetyNet 초기화 실패: %s", e)
        try:
            from rules.duplicate_detector import DuplicateDetector
            self._dup_detector = DuplicateDetector()
            logger.info("DuplicateDetector 싱글턴 초기화 완료")
        except Exception as e:
            logger.warning("DuplicateDetector 초기화 실패: %s", e)

        # 피처 빌더
        self._builder = RequestFeatureBuilder(
            ddi_matrix=self._ddi_matrix,
            cyp_extractor=self._cyp,
            code_standardizer=self._std,
        )

    def reload_model(self, model_path: str | Path) -> bool:
        """무중단 모델 핫스왑 (스레드 안전)."""
        new_ml = MLModel()
        ok = new_ml.load(model_path)
        if ok:
            with self._ml_lock:
                self._ml = new_ml
            logger.info("모델 핫스왑 완료: %s", model_path)
        return ok

    def reload_hierarchical(self, model_dir: str | Path) -> bool:
        """계층 모델 무중단 핫스왑 (스레드 안전)."""
        new_hp = HierarchicalPredictor()
        ok = new_hp.load(str(model_dir))
        if ok:
            if not new_hp.feature_cols:
                logger.warning("계층 모델 핫스왑 feature_cols 비어 있음 — 거부: %s", model_dir)
                return False
            _missing, _schema_ok = _validate_feature_schema(
                new_hp.feature_cols, "계층 모델 (reload)",
            )
            if not _schema_ok:
                logger.warning("계층 모델 핫스왑 schema 거부: %s", model_dir)
                return False
            with self._hier_lock:
                self._hierarchical = new_hp
            logger.info("계층 모델 핫스왑 완료: %s", model_dir)
        else:
                logger.warning("계층 모델 핫스왑 실패: %s", model_dir)
        return ok

    def reload_dl(self, bundle_dir: str | Path) -> bool:
        """DL bundle hot-swap.

        Hot-swap keeps manifest/hash/lookback validation eager. Torch runtime
        artifacts are loaded lazily by DLModel.predict().
        """
        new_dl = DLModel(runtime_lookback_days=self._dl.runtime_lookback_days)
        new_dl.load(bundle_dir)
        with self._dl_lock:
            self._dl = new_dl
        logger.info("DL bundle 핫스왑 완료: %s", bundle_dir)
        return True

    def set_dl_history_provider(
        self,
        provider: Optional[HANAHistoryProvider],
    ) -> None:
        """Attach or clear the provider used by DL auxiliary inference."""
        with self._dl_lock:
            self._dl_history_provider = provider

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time

    # ──────────────────────────────────────────────────────────────────────────
    # 예측
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _stage2_label_to_risk(label: str) -> RiskLevel:
        """Stage 2 6-class 라벨 (+ Red) → 기존 4-class RiskLevel enum 매핑.

        Red          → RED
        Y_*          → YELLOW (Y_OTHER 포함)
        No_Alert     → NORMAL
        """
        if label == "Red":
            return RiskLevel.RED
        if label.startswith("Y_"):
            return RiskLevel.YELLOW
        return RiskLevel.NORMAL

    def predict(self, req: PredictRequest) -> PredictResponse:
        """단일 환자 위험도 예측."""
        ref = req.reference_date or date.today()

        # Step 1: Rule Safety Net
        rule_level, rule_reasons, ddi_alerts = _run_safety_net(req.drugs, patient_age=req.patient_age, sn_instance=self._safety_net)

        # Step 2: 중복약물 탐지
        dup_count, dup_reasons = _run_duplicate_detector(req.drugs, dd_instance=self._dup_detector)

        # Rule 등급 보완 (중복약물) — rule_level이 NORMAL일 때만 등급 상향
        if dup_count >= 1 and rule_level == RiskLevel.NORMAL:
            rule_level = RiskLevel.YELLOW

        # Step 3: ML 예측 — 계층 모드 우선, 아니면 단일 모델
        ml_level: Optional[RiskLevel] = None
        ml_prob: Optional[float] = None
        yellow_subtype: Optional[str] = None
        stage2_probs: Optional[dict[str, float]] = None
        red_suspect: bool = False
        action: Optional[str] = None
        dl_prediction: Optional[DLPredictionResult | dict[str, object]] = None
        dl_error: Optional[str] = None

        with self._hier_lock:
            _hier = self._hierarchical
        if _hier is not None and _hier.loaded:
            # rulefeat.v1 번들이면 triple_whammy/위험플래그를 edi→wk 로 산출(Phase 2-2 모델
            # 피처 정합). 구 번들이면 비활성(0) → 구 번들 학습값과 정합 유지(skew 방지 게이팅).
            from scripts.etl.prescription_aggregator import FEATURE_SEMANTICS_VERSION
            _rf_active = (_hier.feature_semantics_version == FEATURE_SEMANTICS_VERSION)
            feat_vec, _ = self._builder.build(
                req,
                feature_names=_hier.feature_cols or None,
                rule_features_active=_rf_active,
            )
            h = _hier.predict_risk_single(feat_vec)
            ml_prob = float(h["p_red"])
            ml_level = self._stage2_label_to_risk(h["risk_level"])
            if h["risk_level"].startswith("Y_"):
                yellow_subtype = h["risk_level"]
            stage2_probs = h.get("stage2_probs")
            red_suspect = bool(h.get("red_suspect", False))
            action = h.get("action")
        else:
            with self._ml_lock:
                ml_snapshot = self._ml  # 핫스왑 중 교체되더라도 이 참조는 안전
            if ml_snapshot.loaded:
                feat_vec, _ = self._builder.build(
                    req,
                    feature_names=ml_snapshot._feature_names or None,
                    scaler=ml_snapshot._scaler,
                    selector=ml_snapshot._selector,
                )
                ml_prob  = ml_snapshot.predict_proba(feat_vec)
                ml_level = ml_snapshot.classify(ml_prob)

        dl_lock = getattr(self, "_dl_lock", None)
        if dl_lock is None:
            dl_snapshot = getattr(self, "_dl", None)
            history_provider = getattr(self, "_dl_history_provider", None)
        else:
            with dl_lock:
                dl_snapshot = getattr(self, "_dl", None)
                history_provider = getattr(self, "_dl_history_provider", None)
        if dl_snapshot is not None and dl_snapshot.loaded and history_provider is not None:
            try:
                history_df = history_provider.fetch_patient_history(
                    req.patient_id,
                    ref,
                    dl_snapshot.lookback_days or dl_snapshot.runtime_lookback_days,
                )
                dl_prediction = dl_snapshot.predict(history_df)
            except Exception as e:
                dl_error = str(e)
                logger.warning("DL auxiliary inference failed: %s", e)

        # Step 3.5: 결정적 Red 백스톱 — 학습 collect_red_triggers(=금기만, 2026-06-06 재설계)를
        # edi→wk 피처에 적용(model-independent, 단방향 escalation). ML Stage1(degenerate τ,
        # ~10% 누락)·SafetyNet(약물명 미해석) 의존 제거. Red=절대 금기(contraindicated)로 한정
        # (25.8% 과도 → major3/triple/10drug/elderly 는 Y_TRIPLE 즉시개입으로 강등). collect_red_
        # triggers 가 금기만 반환하므로 필터는 사실상 RED_CONTRAINDICATED.
        _BACKSTOP_ACTIVE = {"RED_CONTRAINDICATED"}
        _red_reasons: list[str] = []
        try:
            _red_reasons = sorted(t for t in self._builder.red_triggers(req.drugs, ref, req.patient_age)
                                  if t in _BACKSTOP_ACTIVE)
        except Exception as e:  # 백스톱 실패는 비차단(기존 경로 유지)
            logger.warning("결정적 Red 백스톱 계산 실패(무시): %s", e)

        # Step 4: 최종 등급 = max(Rule, ML, 결정적 Red 백스톱)
        final_level = rule_level
        if ml_level is not None:
            final_level = RiskLevel.max(rule_level, ml_level)
        if _red_reasons:
            final_level = RiskLevel.max(final_level, RiskLevel.RED)

        # Step 4.5: 결정적 severe→Y_TRIPLE 백스톱 (model-independent floor) — 중증 조건
        # (triple_whammy/10drug+고위험/고령+장기)은 학습상 Y_TRIPLE(문자안내). 모델이 하향분류해도
        # 최소 Y_TRIPLE 보장(Red·Y_DDI_MAJOR 등 상위 subtype 은 유지). 단방향 escalation.
        _severe_reasons: list[str] = []
        if final_level != RiskLevel.RED:
            try:
                _severe_reasons = sorted(self._builder.severe_triggers(req.drugs, ref, req.patient_age))
            except Exception as e:  # 백스톱 실패는 비차단
                logger.warning("결정적 severe 백스톱 계산 실패(무시): %s", e)
        if _severe_reasons:
            from hana_app.core.hierarchical_runner import ACTION_BY_LABEL
            _SUB_RANK = {"Y_DDI_MAJOR": 4, "Y_TRIPLE": 3, "Y_DOUBLE": 2,
                         "Y_DDI_MOD": 1, "Y_DUP": 1, "Y_FRAG": 1}
            if _SUB_RANK.get(yellow_subtype or "", 0) < _SUB_RANK["Y_TRIPLE"]:
                yellow_subtype = "Y_TRIPLE"
                action = ACTION_BY_LABEL["Y_TRIPLE"]   # 문자 안내
            final_level = RiskLevel.max(final_level, RiskLevel.YELLOW)

        # Step 5: DDI 알림 보완 (ddi_matrix에서 추가)
        if self._ddi_matrix is not None:
            extra_alerts = self._build_ddi_alerts(req.drugs, ref)
            existing_pairs = {(a.drug_a, a.drug_b) for a in ddi_alerts}
            for alert in extra_alerts:
                if (alert.drug_a, alert.drug_b) not in existing_pairs:
                    ddi_alerts.append(alert)

        # dup_reasons는 등급과 무관하게 항상 포함 (설명 가능성)
        all_reasons = list(rule_reasons) + [r for r in dup_reasons if r not in rule_reasons]
        for _r in _red_reasons:   # 결정적 Red 백스톱 사유 (canonical rule ID, dedupe)
            if _r not in all_reasons:
                all_reasons.append(_r)
        for _r in _severe_reasons:   # 결정적 severe→Y_TRIPLE 백스톱 사유 (SEV_*, dedupe)
            if _r not in all_reasons:
                all_reasons.append(_r)
        if ml_prob is not None and ml_prob > 0.3:
            all_reasons.append(f"ML 모델 Red 확률: {ml_prob:.1%}")
        if red_suspect:
            all_reasons.append(
                "Red 의심 (τ_review ≤ p_red < τ_red) — 운영팀 검수 큐"
            )

        return PredictResponse(
            patient_id=req.patient_id,
            risk_level=final_level,
            rule_level=rule_level,
            ml_level=ml_level,
            ml_probability=ml_prob,
            drug_count=len({d.edi_code for d in req.drugs}),
            ddi_alerts=ddi_alerts,
            risk_reasons=all_reasons,
            intervention=INTERVENTION_MAP[final_level],
            reference_date=ref,
            yellow_subtype=yellow_subtype,
            stage2_probs=stage2_probs,
            red_suspect=red_suspect,
            action=action,
            dl_prediction=dl_prediction,
            dl_error=dl_error,
        )

    def _build_ddi_alerts(self, drugs: list[DrugItem], ref: Optional[date] = None) -> list[DDIAlert]:
        """DDI 알림 생성 — DDI 피처와 **동일 경로**(edi→wk→overlap→DB-code, 단일출처).

        구 ATC all-pairs 경로(_builder._ddi_index)는 ddi_matrix 에 ATC 컬럼이 없어 항상
        빈 dict 였고 features 와 불일치했다. 이제 RequestFeatureBuilder.ddi_alert_pairs 로
        통일해 features(ddi_*)와 alerts 가 같은 overlap·식별자 공간을 쓴다(d201743 회귀 방지).
        """
        if self._ddi_matrix is None:
            return []
        pairs = self._builder.ddi_alert_pairs(drugs, ref or date.today())
        alerts: list[DDIAlert] = []
        for name_a, name_b, sev in pairs:
            if sev in ("Contraindicated", "Major"):
                try:
                    severity = Severity(sev)
                except ValueError:
                    severity = Severity.UNKNOWN
                alerts.append(DDIAlert(
                    drug_a=name_a, drug_b=name_b, severity=severity, source="DDI_Matrix",
                ))
        return alerts


# ─────────────────────────────────────────────────────────────────────────────
# 글로벌 싱글턴
# ─────────────────────────────────────────────────────────────────────────────

_predictor: Optional[HybridPredictor] = None


def get_predictor() -> HybridPredictor:
    global _predictor
    if _predictor is None:
        raise RuntimeError("Predictor가 초기화되지 않았습니다. lifespan에서 init_predictor() 호출 필요")
    return _predictor


def init_predictor(**kwargs) -> HybridPredictor:
    global _predictor
    _predictor = HybridPredictor(**kwargs)
    return _predictor
