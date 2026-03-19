"""
동시복용(Concurrent Drug Use) 기간 계산
핵심 알고리즘:
  1. 환자별로 처방 구간 [start, end] 목록 수집
  2. 90일 슬라이딩 윈도우 내 모든 약물 쌍 검사
  3. 두 구간의 교집합이 ≥ 7일이면 동시복용으로 판정
  4. 결과: DrugOverlapPair 목록

성능 고려:
  - 약물 수가 많을 경우 O(n²) → 윈도우 내 약물 수 상한(MAX_DRUGS_PER_WINDOW=30)
  - 대용량은 Spark 배치에서 처리; 이 모듈은 단일 환자 단위
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterator

import pandas as pd

from .models import DrugOverlapPair, PrescriptionRecord

# ─────────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_DAYS = 90
MIN_OVERLAP_DAYS = 7
MAX_DRUGS_PER_WINDOW = 50   # 윈도우 내 최대 약물 수 (성능 안전장치)


def _date_from_str(s: str) -> date:
    """YYYYMMDD → date."""
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def prescriptions_from_df(df: pd.DataFrame) -> list[PrescriptionRecord]:
    """
    T20+T30 조인 결과 DataFrame → PrescriptionRecord 목록 변환.

    필수 컬럼:
      BNFCR_PSEUDO, INST_PSEUDO, MDCARE_BILL_NO,
      EDI_CD, atc_code, drug_name,
      MDCARE_STRT_DT, MEDTIME_FRQ_CNT,
      DOSG_ONCE (optional), DOSG_FREQ_DY (optional)
    """
    records = []
    for row in df.itertuples(index=False):
        try:
            start = _date_from_str(str(row.MDCARE_STRT_DT))
            total_days = max(1, int(row.MEDTIME_FRQ_CNT))
            end = start + timedelta(days=total_days - 1)
        except (ValueError, AttributeError):
            continue

        records.append(PrescriptionRecord(
            patient_id=str(row.BNFCR_PSEUDO),
            institution_id=str(getattr(row, "INST_PSEUDO", "")),
            bill_no=str(row.MDCARE_BILL_NO),
            edi_code=str(row.EDI_CD),
            atc_code=getattr(row, "atc_code", None) or None,
            drug_name=getattr(row, "drug_name", None) or None,
            start_date=start,
            end_date=end,
            total_days=total_days,
            dose_once=float(getattr(row, "DOSG_ONCE", 0) or 0),
            dose_freq=int(getattr(row, "DOSG_FREQ_DY", 1) or 1),
            sick_code=getattr(row, "SICK_SYM", None) or None,
            institution_type=getattr(row, "CLNC_TP_CD", None) or None,
        ))
    return records


def _overlap_days(a_start: date, a_end: date, b_start: date, b_end: date) -> int:
    """두 날짜 구간의 교집합 일수. 겹치지 않으면 0."""
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    delta = (overlap_end - overlap_start).days + 1
    return max(0, delta)


def calculate_overlaps_for_patient(
    prescriptions: list[PrescriptionRecord],
    window_days: int = WINDOW_DAYS,
    min_overlap: int = MIN_OVERLAP_DAYS,
) -> list[DrugOverlapPair]:
    """
    단일 환자의 처방 목록에서 동시복용 쌍 계산.

    90일 윈도우: 각 처방의 start_date 기준으로 [start, start+89] 윈도우 생성.
    해당 윈도우 내에 활성인 다른 처방과 교집합 계산.
    """
    if len(prescriptions) < 2:
        return []

    # 시작일 기준 정렬
    prescriptions = sorted(prescriptions, key=lambda p: p.start_date)
    pairs: list[DrugOverlapPair] = []
    seen_pairs: set[frozenset] = set()  # 중복 쌍 방지

    for i, anchor in enumerate(prescriptions):
        window_start = anchor.start_date
        window_end = window_start + timedelta(days=window_days - 1)

        # 윈도우 내 활성 처방 수집
        window_drugs = [
            p for p in prescriptions
            if p.start_date <= window_end and p.end_date >= window_start
        ]

        if len(window_drugs) > MAX_DRUGS_PER_WINDOW:
            # 성능 안전장치: 투여일수 긴 순으로 상위 N개만
            window_drugs = sorted(window_drugs, key=lambda p: -p.total_days)[:MAX_DRUGS_PER_WINDOW]

        for j in range(len(window_drugs)):
            for k in range(j + 1, len(window_drugs)):
                a = window_drugs[j]
                b = window_drugs[k]

                # 동일 약물 제외
                if a.edi_code == b.edi_code:
                    continue

                # 중복 쌍 제외
                pair_key = frozenset({a.edi_code, b.edi_code})
                if pair_key in seen_pairs:
                    continue

                ov = _overlap_days(a.start_date, a.end_date, b.start_date, b.end_date)
                if ov >= min_overlap:
                    seen_pairs.add(pair_key)
                    overlap_start = max(a.start_date, b.start_date)
                    overlap_end = min(a.end_date, b.end_date)
                    pairs.append(DrugOverlapPair(
                        patient_id=anchor.patient_id,
                        drug_a_edi=a.edi_code,
                        drug_b_edi=b.edi_code,
                        drug_a_atc=a.atc_code,
                        drug_b_atc=b.atc_code,
                        drug_a_name=a.drug_name,
                        drug_b_name=b.drug_name,
                        overlap_start=overlap_start,
                        overlap_end=overlap_end,
                        overlap_days=ov,
                        window_start=window_start,
                        window_end=window_end,
                    ))

    return pairs


def calculate_overlaps_batch(
    df_prescriptions: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
    min_overlap: int = MIN_OVERLAP_DAYS,
) -> pd.DataFrame:
    """
    전체 환자 DataFrame을 환자별로 그룹화하여 동시복용 쌍 계산.
    반환: DrugOverlapPair 컬럼들로 구성된 DataFrame.
    """
    records = prescriptions_from_df(df_prescriptions)

    # 환자별 그룹화
    patient_map: dict[str, list[PrescriptionRecord]] = {}
    for r in records:
        patient_map.setdefault(r.patient_id, []).append(r)

    all_pairs: list[dict] = []
    for patient_id, prx_list in patient_map.items():
        pairs = calculate_overlaps_for_patient(prx_list, window_days, min_overlap)
        for p in pairs:
            all_pairs.append({
                "patient_id":    p.patient_id,
                "drug_a_edi":    p.drug_a_edi,
                "drug_b_edi":    p.drug_b_edi,
                "drug_a_atc":    p.drug_a_atc,
                "drug_b_atc":    p.drug_b_atc,
                "drug_a_name":   p.drug_a_name,
                "drug_b_name":   p.drug_b_name,
                "overlap_start": p.overlap_start,
                "overlap_end":   p.overlap_end,
                "overlap_days":  p.overlap_days,
                "window_start":  p.window_start,
                "window_end":    p.window_end,
            })

    if not all_pairs:
        return pd.DataFrame(columns=[
            "patient_id", "drug_a_edi", "drug_b_edi",
            "drug_a_atc", "drug_b_atc", "drug_a_name", "drug_b_name",
            "overlap_start", "overlap_end", "overlap_days",
            "window_start", "window_end",
        ])

    return pd.DataFrame(all_pairs)


def get_concurrent_drug_count(
    prescriptions: list[PrescriptionRecord],
    reference_date: date,
) -> int:
    """특정 기준일 기준 동시 복용 중인 약물 수 (피처용)."""
    return sum(
        1 for p in prescriptions
        if p.start_date <= reference_date <= p.end_date
    )
