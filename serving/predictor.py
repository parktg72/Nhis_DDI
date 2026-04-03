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
import pickle
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .schemas import (
    DDIAlert, DrugItem, PredictRequest, PredictResponse,
    RiskLevel, Severity, INTERVENTION_MAP,
)

logger = logging.getLogger(__name__)


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
) -> tuple[RiskLevel, list[str], list[DDIAlert]]:
    """
    rules/safety_net.py 실행 → (등급, 이유 목록, DDI 알림 목록).
    safety_net 미설치/오류 시 Normal 반환.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from rules.safety_net import SafetyNet

        has_renal, has_hepatic = _detect_risk_flags(drugs)

        sn = SafetyNet()
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
    except Exception as e:
        logger.warning("Safety Net 실행 오류 (Normal 반환): %s", e)
        return RiskLevel.NORMAL, [], []


def _run_duplicate_detector(drugs: list[DrugItem]) -> tuple[int, list[str]]:
    """중복약물 탐지 → (중복건수, 이유 목록)."""
    try:
        from rules.duplicate_detector import DuplicateDetector

        dd = DuplicateDetector()
        drug_input = _drugs_to_dup_input(drugs)
        result = dd.detect(drug_input)

        dup_count = result.duplicate_level1_count + result.duplicate_level2_count
        reasons = []
        if result.duplicate_level1_count:
            reasons.append(f"동일성분중복 {result.duplicate_level1_count}건")
        if result.duplicate_level2_count:
            reasons.append(f"동일약리군중복 {result.duplicate_level2_count}건")
        return dup_count, reasons
    except Exception as e:
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

            # Resolve scaler/selector paths relative to model file directory
            import os
            model_dir = path.parent
            for attr, key in [("_scaler", "scaler_path"), ("_selector", "selector_path")]:
                stored = state.get(key)
                if stored:
                    candidate = (model_dir / stored).resolve()
                    if candidate.exists():
                        import pickle as _pk
                        with open(candidate, "rb") as f:
                            setattr(self, attr, _pk.load(f))
                        logger.info("%s 로드: %s", key, candidate)
                    else:
                        logger.warning("%s 없음 — 미적용: %s", key, candidate)

            # Ensemble model: load from sub-model files
            if self._model is None and state.get("trainer_class") == "EnsembleTrainer":
                xgb_path = path.with_suffix(".xgb.pkl")
                lgb_path = path.with_suffix(".lgb.pkl")
                if xgb_path.exists() and lgb_path.exists():
                    try:
                        import pickle as _pk
                        xgb_content = xgb_path.read_bytes()
                        lgb_content = lgb_path.read_bytes()
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

            logger.info("ML 모델 로드: %s (threshold=%.3f)", path, self._threshold)
            return True
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

    def build(self, req: PredictRequest, feature_names=None, scaler=None, selector=None) -> tuple[np.ndarray, dict]:
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
        feat: dict[str, float] = {}
        feat["drug_count"]        = float(len({d.edi_code for d in drugs}))
        feat["institution_count"] = float(len({d.institution_id for d in drugs
                                               if d.institution_id}))
        feat["age"]               = float(req.patient_age or 0)
        feat["sex_male"]          = float(req.patient_sex == "M") if req.patient_sex else 0.5

        # ── DDI 피처 (ATC 조합 기반) ──────────────────────────────────────
        ddi_counts = {"Contraindicated": 0, "Major": 0, "Moderate": 0, "Minor": 0}
        if self._ddi_index and len(atc_codes) >= 2:
            for i in range(len(atc_codes)):
                for j in range(i + 1, len(atc_codes)):
                    key = frozenset({atc_codes[i], atc_codes[j]})
                    sev = self._ddi_index.get(key)
                    if sev in ddi_counts:
                        ddi_counts[sev] += 1

        feat["ddi_contraindicated"] = float(ddi_counts["Contraindicated"])
        feat["ddi_major"]           = float(ddi_counts["Major"])
        feat["ddi_moderate"]        = float(ddi_counts["Moderate"])
        feat["ddi_minor"]           = float(ddi_counts["Minor"])

        # ── 투여일수 피처 ──────────────────────────────────────────────────
        durations = [d.total_days for d in drugs]
        feat["avg_drug_duration"]    = float(sum(durations) / len(durations)) if durations else 0.0
        feat["long_term_drug_count"] = float(sum(1 for d in durations if d >= 30))

        # ── ATC 중복 피처 ──────────────────────────────────────────────────
        from collections import Counter
        cnt5 = Counter(atc_codes)
        cnt4 = Counter(c[:5] for c in atc_codes if len(c) >= 5)
        cnt3 = Counter(c[:4] for c in atc_codes if len(c) >= 4)
        feat["dup_same_ingredient"] = float(sum(1 for v in cnt5.values() if v >= 2))
        feat["dup_atc5"]            = float(sum(1 for v in cnt4.values() if v >= 2))
        feat["dup_atc4"]            = float(sum(1 for v in cnt3.values() if v >= 2))

        # ── CYP 피처 ──────────────────────────────────────────────────────
        if self._cyp and atc_codes:
            cyp_feat = self._cyp.extract(atc_codes)
            feat.update(cyp_feat)
        else:
            feat["cyp_risk_score"]     = 0.0
            feat["cyp_high_risk_pairs"] = 0.0

        # Triple Whammy, QT 카운트 (단순 ATC prefix 기반)
        feat["triple_whammy"] = float(_check_triple_whammy(atc_codes))
        feat["qt_risk_count"] = float(_count_qt_drugs(atc_codes))
        feat["drug_count_7d"] = feat["drug_count"]  # 온라인에서는 동일

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
                df = pd.DataFrame(selector.transform(df), columns=selector.get_support(indices=False) if hasattr(selector, 'get_support') else df.columns)
            except Exception as e:
                logger.warning("Selector 적용 실패 (원본 사용): %s", e)

        vec = df.values.flatten().astype(float)
        return vec, feat


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
    ):
        self._start_time = time.time()
        self._ml_lock = threading.RLock()
        self._ml = MLModel()
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

        # ML 모델 로드
        if model_path and Path(model_path).exists():
            self._ml.load(model_path)

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

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time

    # ──────────────────────────────────────────────────────────────────────────
    # 예측
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, req: PredictRequest) -> PredictResponse:
        """단일 환자 위험도 예측."""
        ref = req.reference_date or date.today()

        # Step 1: Rule Safety Net
        rule_level, rule_reasons, ddi_alerts = _run_safety_net(req.drugs, patient_age=req.patient_age)

        # Step 2: 중복약물 탐지
        dup_count, dup_reasons = _run_duplicate_detector(req.drugs)

        # Rule 등급 보완 (중복약물)
        if dup_count >= 1 and rule_level == RiskLevel.NORMAL:
            rule_level = RiskLevel.YELLOW
            rule_reasons.extend(dup_reasons)

        # Step 3: ML 예측 (모델 있을 때만)
        ml_level: Optional[RiskLevel] = None
        ml_prob: Optional[float] = None
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

        # Step 4: 최종 등급 = max(Rule, ML)
        final_level = rule_level
        if ml_level is not None:
            final_level = RiskLevel.max(rule_level, ml_level)

        # Step 5: DDI 알림 보완 (ddi_matrix에서 추가)
        if self._ddi_matrix is not None:
            extra_alerts = self._build_ddi_alerts(req.drugs)
            existing_pairs = {(a.drug_a, a.drug_b) for a in ddi_alerts}
            for alert in extra_alerts:
                if (alert.drug_a, alert.drug_b) not in existing_pairs:
                    ddi_alerts.append(alert)

        all_reasons = rule_reasons + dup_reasons
        if ml_prob is not None and ml_prob > 0.3:
            all_reasons.append(f"ML 모델 Red 확률: {ml_prob:.1%}")

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
        )

    def _build_ddi_alerts(self, drugs: list[DrugItem]) -> list[DDIAlert]:
        """DDI 매트릭스에서 ATC 기반 DDI 알림 생성."""
        if self._ddi_matrix is None:
            return []

        atc_to_name = {d.atc_code: (d.drug_name or d.edi_code)
                       for d in drugs if d.atc_code}
        atc_codes = list(atc_to_name.keys())
        alerts: list[DDIAlert] = []

        ddi_lookup = self._builder._ddi_index
        severity_order = {"Contraindicated": 4, "Major": 3, "Moderate": 2, "Minor": 1}

        for i in range(len(atc_codes)):
            for j in range(i + 1, len(atc_codes)):
                key = frozenset({atc_codes[i], atc_codes[j]})
                sev = ddi_lookup.get(key)
                if sev and sev in ("Contraindicated", "Major"):
                    try:
                        severity = Severity(sev)
                    except ValueError:
                        severity = Severity.UNKNOWN
                    alerts.append(DDIAlert(
                        drug_a=atc_to_name.get(atc_codes[i], atc_codes[i]),
                        drug_b=atc_to_name.get(atc_codes[j], atc_codes[j]),
                        severity=severity,
                        source="DDI_Matrix",
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
