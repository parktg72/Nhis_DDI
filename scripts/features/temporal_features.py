"""
시계열 처방 패턴 피처 추출

90일 윈도우 내 처방 패턴:
  - 약물 증감 추세 (전반 45일 vs 후반 45일)
  - 기관 분산 (기관별 처방 비율 엔트로피)
  - 처방 집중도 (특정 기관에 집중 vs 분산)
  - 신규 약물 추가 여부 (후반 45일에 처음 등장)
  - 처방 주기성 (규칙적 복용 vs 간헐적)
"""
from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from scripts.etl.models import PrescriptionRecord

logger = logging.getLogger(__name__)

TEMPORAL_FEATURE_COLS = [
    "drug_count_early",          # 전반 45일 약물 수
    "drug_count_late",           # 후반 45일 약물 수
    "drug_trend",                # 약물 증감 (late - early)
    "new_drug_in_late",          # 후반 신규 약물 수
    "institution_entropy",       # 기관 분산 엔트로피 (높을수록 분산)
    "max_institution_ratio",     # 단일 기관 최대 처방 비율
    "multi_institution_flag",    # 3개↑ 기관 처방 여부 (0/1)
    "prescription_density",      # 처방 밀도 (처방건수 / 90일)
    "avg_drug_duration",         # 평균 투여일수
    "long_term_drug_count",      # 30일↑ 장기 처방 약물 수
    "chronic_drug_ratio",        # 장기처방 약물 비율
]


def _entropy(counts: list[int]) -> float:
    """Shannon 엔트로피 (기관 분산 지표)."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def extract_temporal(
    prescriptions: list[PrescriptionRecord],
    window_start: date | None = None,
    window_end: date | None = None,
) -> dict[str, float]:
    """
    단일 환자 처방 목록 → 시계열 피처 딕셔너리.

    Parameters
    ----------
    prescriptions : ETL PrescriptionRecord 목록
    window_start, window_end : 90일 윈도우 경계 (None이면 데이터에서 추론)
    """
    features: dict[str, float] = {col: 0.0 for col in TEMPORAL_FEATURE_COLS}

    if not prescriptions:
        return features

    # 윈도우 결정
    starts = [p.start_date for p in prescriptions]
    ends = [p.end_date for p in prescriptions]
    w_start = window_start or min(starts)
    w_end = window_end or min(max(ends), w_start + timedelta(days=89))
    mid = w_start + timedelta(days=44)  # 전반/후반 분기점

    # ── 전반/후반 분할 ──────────────────────────────────────────────────────
    early = [p for p in prescriptions if p.start_date <= mid]
    late  = [p for p in prescriptions if p.start_date > mid]

    early_drugs = {p.edi_code for p in early}
    late_drugs  = {p.edi_code for p in late}
    all_drugs   = early_drugs | late_drugs

    features["drug_count_early"] = float(len(early_drugs))
    features["drug_count_late"]  = float(len(late_drugs))
    features["drug_trend"]       = float(len(late_drugs) - len(early_drugs))
    features["new_drug_in_late"] = float(len(late_drugs - early_drugs))

    # ── 기관 분산 ──────────────────────────────────────────────────────────
    inst_counts: dict[str, int] = {}
    for p in prescriptions:
        if p.institution_id:
            inst_counts[p.institution_id] = inst_counts.get(p.institution_id, 0) + 1

    if inst_counts:
        total_rx = sum(inst_counts.values())
        features["institution_entropy"]    = _entropy(list(inst_counts.values()))
        features["max_institution_ratio"]  = max(inst_counts.values()) / total_rx
        features["multi_institution_flag"] = float(len(inst_counts) >= 3)
    else:
        features["max_institution_ratio"] = 1.0

    # ── 처방 밀도 ────────────────────────────────────────────────────────────
    window_len = max(1, (w_end - w_start).days + 1)
    features["prescription_density"] = len(prescriptions) / window_len

    # ── 투여일수 통계 ─────────────────────────────────────────────────────────
    durations = [p.total_days for p in prescriptions]
    if durations:
        features["avg_drug_duration"]   = sum(durations) / len(durations)
        long_term = [d for d in durations if d >= 30]
        features["long_term_drug_count"] = float(len(long_term))
        features["chronic_drug_ratio"]   = len(long_term) / len(durations)

    return features


def extract_temporal_batch(
    patient_records: dict[str, list[PrescriptionRecord]],
) -> pd.DataFrame:
    """
    환자별 처방 레코드 딕셔너리 → 시계열 피처 DataFrame.

    Returns
    -------
    DataFrame with columns: patient_id + TEMPORAL_FEATURE_COLS
    """
    rows = []
    for patient_id, prescriptions in patient_records.items():
        feat = extract_temporal(prescriptions)
        feat["patient_id"] = patient_id
        rows.append(feat)

    if not rows:
        return pd.DataFrame(columns=["patient_id"] + TEMPORAL_FEATURE_COLS)

    result = pd.DataFrame(rows)
    cols = ["patient_id"] + [c for c in result.columns if c != "patient_id"]
    return result[cols]
