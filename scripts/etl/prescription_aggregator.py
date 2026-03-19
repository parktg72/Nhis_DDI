"""
처방 패턴 집계
T20+T30+T40+T50 조인 결과 → 환자별 90일 윈도우 피처 집계

집계 항목:
- 고유 약물 수 (drug_count)
- 처방 기관 수 (institution_count)
- 동시복용 피크 수 (max_concurrent)
- DDI 심각도별 카운트
- 중복약물 레벨별 카운트
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from .models import DrugOverlapPair, PatientFeatures, PrescriptionRecord
from .overlap_calculator import calculate_overlaps_for_patient, get_concurrent_drug_count

logger = logging.getLogger(__name__)


def _parse_date(s: str) -> Optional[date]:
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except (ValueError, TypeError):
        return None


def _calc_age(birth_year: str, window_end: date) -> Optional[int]:
    try:
        return window_end.year - int(birth_year)
    except (ValueError, TypeError):
        return None


def aggregate_patient_features(
    patient_id: str,
    prescriptions: list[PrescriptionRecord],
    overlap_pairs: list[DrugOverlapPair],
    ddi_matrix: pd.DataFrame | None,
    dup_groups: pd.DataFrame | None,
    age: int | None = None,
    sex: str | None = None,
    window_start: date | None = None,
    window_end: date | None = None,
) -> PatientFeatures:
    """
    단일 환자의 피처 벡터 계산.

    Parameters
    ----------
    ddi_matrix : ddi_matrix_final.parquet (drug_a_atc, drug_b_atc, severity)
    dup_groups : efcy_duplicate_groups.parquet (drug_code, efcy_class_no)
    """
    if not prescriptions:
        return PatientFeatures(
            patient_id=patient_id,
            window_start=window_start or date.today(),
            window_end=window_end or date.today(),
            age=age, sex=sex,
        )

    # 윈도우 결정 (처방 최소/최대 날짜)
    all_starts = [p.start_date for p in prescriptions]
    all_ends = [p.end_date for p in prescriptions]
    w_start = window_start or min(all_starts)
    w_end = window_end or min(max(all_ends), w_start + timedelta(days=89))

    features = PatientFeatures(
        patient_id=patient_id,
        window_start=w_start,
        window_end=w_end,
        age=age,
        sex=sex,
    )

    # ── 기본 피처 ──────────────────────────────────────────────────────────
    unique_edis = {p.edi_code for p in prescriptions}
    features.drug_count = len(unique_edis)

    unique_insts = {p.institution_id for p in prescriptions if p.institution_id}
    features.institution_count = len(unique_insts)

    # 7일 내 동시 복용 수 (기준: 윈도우 종료일)
    features.drug_count_7d = get_concurrent_drug_count(prescriptions, w_end)

    # ── DDI 피처 ────────────────────────────────────────────────────────────
    if ddi_matrix is not None and overlap_pairs:
        _fill_ddi_features(features, overlap_pairs, ddi_matrix)

    # ── 중복약물 피처 ────────────────────────────────────────────────────────
    if dup_groups is not None:
        _fill_dup_features(features, prescriptions, dup_groups)

    # ── 위험도 결정 ──────────────────────────────────────────────────────────
    _assign_risk_level(features)

    return features


def _fill_ddi_features(
    features: PatientFeatures,
    pairs: list[DrugOverlapPair],
    ddi_matrix: pd.DataFrame,
) -> None:
    """동시복용 쌍 × DDI 매트릭스 → 심각도별 카운트."""
    # ddi_matrix 인덱스: (drug_a_atc, drug_b_atc) → severity
    if "drug_a_atc" not in ddi_matrix.columns or "severity" not in ddi_matrix.columns:
        return

    # ATC 코드 집합으로 빠른 조회를 위한 딕셔너리 구성
    ddi_lookup: dict[frozenset, str] = {}
    for row in ddi_matrix.itertuples(index=False):
        key = frozenset({str(row.drug_a_atc), str(row.drug_b_atc)})
        # 더 심각한 것을 유지
        severity_order = {"Contraindicated": 4, "Major": 3, "Moderate": 2, "Minor": 1}
        existing = ddi_lookup.get(key)
        new_sev = str(row.severity)
        if existing is None or severity_order.get(new_sev, 0) > severity_order.get(existing, 0):
            ddi_lookup[key] = new_sev

    for pair in pairs:
        a_atc = pair.drug_a_atc
        b_atc = pair.drug_b_atc
        if not a_atc or not b_atc:
            continue
        key = frozenset({a_atc, b_atc})
        severity = ddi_lookup.get(key)
        if severity == "Contraindicated":
            features.ddi_contraindicated += 1
        elif severity == "Major":
            features.ddi_major += 1
        elif severity == "Moderate":
            features.ddi_moderate += 1
        elif severity == "Minor":
            features.ddi_minor += 1


def _fill_dup_features(
    features: PatientFeatures,
    prescriptions: list[PrescriptionRecord],
    dup_groups: pd.DataFrame,
) -> None:
    """ATC 코드 기반 중복약물 레벨 계산."""
    atc_codes = [p.atc_code for p in prescriptions if p.atc_code]
    if len(atc_codes) < 2:
        return

    # ATC 레벨별 prefix 집합
    level5 = set(atc_codes)                         # 전체 (7자리)
    level4 = {c[:5] for c in atc_codes if len(c) >= 5}  # 5자리
    level3 = {c[:4] for c in atc_codes if len(c) >= 4}  # 4자리
    level2 = {c[:3] for c in atc_codes if len(c) >= 3}  # 3자리

    # Level 5 (동일성분): 동일 ATC 7자리 2개 이상
    from collections import Counter
    cnt5 = Counter(atc_codes)
    features.dup_same_ingredient = sum(1 for c in cnt5.values() if c >= 2)

    # Level 4: 동일 5자리 prefix 중 다른 ATC를 가진 쌍
    cnt4: Counter = Counter()
    for code in atc_codes:
        if len(code) >= 5:
            cnt4[code[:5]] += 1
    features.dup_atc5 = sum(1 for c in cnt4.values() if c >= 2)

    # Level 3: 동일 4자리 prefix
    cnt3: Counter = Counter()
    for code in atc_codes:
        if len(code) >= 4:
            cnt3[code[:4]] += 1
    features.dup_atc4 = sum(1 for c in cnt3.values() if c >= 2)

    # Level 2: 동일 3자리 prefix (더 넓은 범위)
    cnt2: Counter = Counter()
    for code in atc_codes:
        if len(code) >= 3:
            cnt2[code[:3]] += 1
    features.dup_atc3 = sum(1 for c in cnt2.values() if c >= 2)


def _assign_risk_level(features: PatientFeatures) -> None:
    """
    위험도 4단계 판정 규칙 (Rule-based Safety Net과 동일 기준).
    Safety Net에서 이미 상세 판정하므로 여기서는 DDI 카운트 기반 단순 판정.
    최종 등급 = max(Safety Net 등급, ML 등급)은 pipeline.py에서 처리.
    """
    reasons: list[str] = []

    # Red 조건
    if features.ddi_contraindicated >= 1:
        features.risk_level = "Red"
        reasons.append(f"Contraindicated DDI {features.ddi_contraindicated}건")
    elif features.ddi_major >= 3:
        features.risk_level = "Red"
        reasons.append(f"Major DDI {features.ddi_major}건 (≥3)")
    elif features.triple_whammy:
        features.risk_level = "Red"
        reasons.append("Triple Whammy")
    elif features.drug_count >= 10 and features.qt_risk_count >= 3:
        features.risk_level = "Red"
        reasons.append(f"10종↑+QT위험약물{features.qt_risk_count}종")
    elif (
        features.age is not None
        and features.age >= 75
        and features.drug_count >= 5
    ):
        features.risk_level = "Red"
        reasons.append(f"75세↑+5종↑ (나이={features.age}, 약물={features.drug_count})")

    # Yellow 조건
    elif features.ddi_major >= 1:
        features.risk_level = "Yellow"
        reasons.append(f"Major DDI {features.ddi_major}건")
    elif features.ddi_moderate >= 2:
        features.risk_level = "Yellow"
        reasons.append(f"Moderate DDI {features.ddi_moderate}건 (≥2)")
    elif features.dup_same_ingredient >= 1:
        features.risk_level = "Yellow"
        reasons.append(f"동일성분중복 {features.dup_same_ingredient}건")
    elif features.institution_count >= 3:
        features.risk_level = "Yellow"
        reasons.append(f"3기관↑ 동시처방 ({features.institution_count}개)")

    # Green 조건
    elif features.ddi_minor >= 1:
        features.risk_level = "Green"
        reasons.append(f"Minor DDI {features.ddi_minor}건")
    elif features.drug_count >= 5:
        features.risk_level = "Green"
        reasons.append(f"5종↑ ({features.drug_count}종)")

    else:
        features.risk_level = "Normal"

    features.risk_reasons = reasons


def aggregate_batch(
    df_prescriptions: pd.DataFrame,
    df_t40: pd.DataFrame | None,
    overlap_df: pd.DataFrame,
    ddi_matrix: pd.DataFrame | None,
    dup_groups: pd.DataFrame | None,
) -> list[PatientFeatures]:
    """
    전체 환자 배치 집계.
    overlap_df: calculate_overlaps_batch() 결과 DataFrame.
    """
    # 처방 레코드 → 환자별 그룹
    from .overlap_calculator import prescriptions_from_df
    all_records = prescriptions_from_df(df_prescriptions)
    patient_records: dict[str, list[PrescriptionRecord]] = {}
    for r in all_records:
        patient_records.setdefault(r.patient_id, []).append(r)

    # 환자 인구통계
    patient_demo: dict[str, dict] = {}
    if df_t40 is not None and "BNFCR_PSEUDO" in df_t40.columns:
        for row in df_t40.itertuples(index=False):
            pid = str(row.BNFCR_PSEUDO)
            patient_demo[pid] = {
                "birth_year": str(getattr(row, "BTH_YYYY", "")),
                "sex": "M" if str(getattr(row, "SEX_TP_CD", "")) == "1" else "F",
            }

    # 동시복용 쌍 → 환자별 그룹
    patient_pairs: dict[str, list[DrugOverlapPair]] = {}
    if not overlap_df.empty:
        for row in overlap_df.itertuples(index=False):
            pid = str(row.patient_id)
            patient_pairs.setdefault(pid, []).append(DrugOverlapPair(
                patient_id=pid,
                drug_a_edi=str(row.drug_a_edi),
                drug_b_edi=str(row.drug_b_edi),
                drug_a_atc=getattr(row, "drug_a_atc", None) or None,
                drug_b_atc=getattr(row, "drug_b_atc", None) or None,
                drug_a_name=getattr(row, "drug_a_name", None) or None,
                drug_b_name=getattr(row, "drug_b_name", None) or None,
                overlap_start=row.overlap_start,
                overlap_end=row.overlap_end,
                overlap_days=int(row.overlap_days),
                window_start=row.window_start,
                window_end=row.window_end,
            ))

    all_features: list[PatientFeatures] = []
    for patient_id, prx_list in patient_records.items():
        demo = patient_demo.get(patient_id, {})
        birth_year = demo.get("birth_year", "")
        # 윈도우 종료 기준 나이
        w_end = max(p.end_date for p in prx_list)
        age = _calc_age(birth_year, w_end) if birth_year else None
        sex = demo.get("sex")

        pairs = patient_pairs.get(patient_id, [])
        feat = aggregate_patient_features(
            patient_id=patient_id,
            prescriptions=prx_list,
            overlap_pairs=pairs,
            ddi_matrix=ddi_matrix,
            dup_groups=dup_groups,
            age=age,
            sex=sex,
        )
        all_features.append(feat)

    return all_features
