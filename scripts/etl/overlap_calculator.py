"""
동시복용(Concurrent Drug Use) 기간 계산
핵심 알고리즘:
  1. 환자별로 처방 구간 [start, end] 목록 수집
  2. 90일 슬라이딩 윈도우 내 모든 약물 쌍 검사
  3. 두 구간의 교집합이 ≥ 7일이면 동시복용으로 판정
  4. 결과: DrugOverlapPair 목록

성능 고려:
  - 약물 수가 많을 경우 O(n²) → 윈도우 내 약물 수 상한(MAX_DRUGS_PER_WINDOW=50)
  - 대용량은 Spark 배치에서 처리; 이 모듈은 단일 환자 단위
"""
from __future__ import annotations

from datetime import date, timedelta

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

    필수 컬럼 (NHIS 실제 레이아웃 기준):
      INDI_DSCM_NO  — 개인식별번호 (환자 ID)
      CMN_KEY       — 공통키 (명세서 ID)
      MDCARE_SYM    — 요양기관기호
      WK_COMPN_CD   — 주성분코드 (DDI 매칭 핵심)
      MDCARE_STRT_DT — 요양개시일자 YYYYMMDD
      TOT_MCNT      — 총투여일수

    선택 컬럼:
      MCARE_DIV_CD   — EDI 약품코드
      RVSN_WK_COMPN_CD — 보정주성분코드 (WK_COMPN_CD 공란 시 사용)
      EFMDC_CLSF_NO  — 약효분류번호
      atc_code       — ATC 코드 (CodeStandardizer 매핑 후 추가)
      drug_name      — 약품명
      TIME1_MDCT_CPCT — 1회투여용량
      DD1_MQTY_FREQ  — 1일투여량횟수
      SICK_SYM1      — 주상병기호
      YOYANG_CLSFC_CD — 요양기관종별코드
    """
    records = []
    for row in df.itertuples(index=False):
        try:
            start = _date_from_str(str(row.MDCARE_STRT_DT))
            total_days = max(1, int(row.TOT_MCNT))
            end = start + timedelta(days=total_days - 1)
        except (ValueError, AttributeError):
            continue

        # WK_COMPN_CD 우선, 없으면 RVSN_WK_COMPN_CD 사용
        wk = str(getattr(row, "WK_COMPN_CD", "") or "").strip()
        if not wk:
            wk = str(getattr(row, "RVSN_WK_COMPN_CD", "") or "").strip()
        if not wk:
            continue  # 주성분코드 없으면 DDI 매칭 불가 → 스킵

        records.append(PrescriptionRecord(
            patient_id=str(row.INDI_DSCM_NO),
            institution_id=str(getattr(row, "MDCARE_SYM", "") or ""),
            bill_no=str(row.CMN_KEY),
            wk_compn_cd=wk,
            edi_code=str(getattr(row, "MCARE_DIV_CD", "") or "") or None,
            atc_code=getattr(row, "atc_code", None) or None,
            gnl_nm_cd=str(getattr(row, "GNL_NM_CD", "") or "") or None,
            efmdc_clsf_no=str(getattr(row, "EFMDC_CLSF_NO", "") or "") or None,
            drug_name=getattr(row, "drug_name", None) or None,
            start_date=start,
            end_date=end,
            total_days=total_days,
            dose_once=float(getattr(row, "TIME1_MDCT_CPCT", 0) or 0),
            dose_freq=int(getattr(row, "DD1_MQTY_FREQ", 1) or 1),
            sick_code=getattr(row, "SICK_SYM1", None) or None,
            sex=str(getattr(row, "SEX_TYPE", "") or "") or None,
            age_id=str(getattr(row, "SUJIN_POTM_AGE_ID", "") or "") or None,
            institution_type=str(getattr(row, "YOYANG_CLSFC_CD", "") or "") or None,
            source="T30",
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
    동일 약물 판정 기준: WK_COMPN_CD (NHIS 주성분코드)
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

                # 동일 주성분 제외 (동일 약물 쌍은 중복약물로 별도 처리)
                if a.wk_compn_cd == b.wk_compn_cd:
                    continue

                # 중복 쌍 제외
                pair_key = frozenset({a.wk_compn_cd, b.wk_compn_cd})
                if pair_key in seen_pairs:
                    continue

                ov = _overlap_days(a.start_date, a.end_date, b.start_date, b.end_date)
                if ov >= min_overlap:
                    seen_pairs.add(pair_key)
                    overlap_start = max(a.start_date, b.start_date)
                    overlap_end = min(a.end_date, b.end_date)
                    pairs.append(DrugOverlapPair(
                        patient_id=anchor.patient_id,
                        drug_a_wk_compn=a.wk_compn_cd,
                        drug_a_edi=a.edi_code,
                        drug_a_atc=a.atc_code,
                        drug_a_name=a.drug_name,
                        drug_b_wk_compn=b.wk_compn_cd,
                        drug_b_edi=b.edi_code,
                        drug_b_atc=b.atc_code,
                        drug_b_name=b.drug_name,
                        overlap_start=overlap_start,
                        overlap_end=overlap_end,
                        overlap_days=ov,
                        window_start=window_start,
                        window_end=window_end,
                    ))

    return pairs


_OVERLAP_COLUMNS = [
    "patient_id",
    "drug_a_wk_compn", "drug_b_wk_compn",
    "drug_a_edi", "drug_b_edi",
    "drug_a_atc", "drug_b_atc",
    "drug_a_name", "drug_b_name",
    "overlap_start", "overlap_end", "overlap_days",
    "window_start", "window_end",
]

_BATCH_FLUSH_SIZE = 5000  # 환자 N명 처리 후 결과 병합 + 메모리 해제


def _overlap_pair_to_dict(p: DrugOverlapPair) -> dict:
    return {
        "patient_id":      p.patient_id,
        "drug_a_wk_compn": p.drug_a_wk_compn,
        "drug_b_wk_compn": p.drug_b_wk_compn,
        "drug_a_edi":      p.drug_a_edi,
        "drug_b_edi":      p.drug_b_edi,
        "drug_a_atc":      p.drug_a_atc,
        "drug_b_atc":      p.drug_b_atc,
        "drug_a_name":     p.drug_a_name,
        "drug_b_name":     p.drug_b_name,
        "overlap_start":   p.overlap_start,
        "overlap_end":     p.overlap_end,
        "overlap_days":    p.overlap_days,
        "window_start":    p.window_start,
        "window_end":      p.window_end,
    }


def calculate_overlaps_batch(
    df_prescriptions: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
    min_overlap: int = MIN_OVERLAP_DAYS,
    guard=None,
) -> pd.DataFrame:
    """
    전체 환자 DataFrame을 환자별로 그룹화하여 동시복용 쌍 계산.

    메모리 효율화:
      - DataFrame을 환자별 청크로 직접 그룹화 (전체 PrescriptionRecord 리스트 생성 방지)
      - _BATCH_FLUSH_SIZE 환자마다 중간 결과 병합 + gc.collect()
      - MemoryGuard 연동: 배치 경계마다 RSS 체크, 한도 초과 시 안전 종료

    Parameters
    ----------
    guard : MemoryGuard 또는 None. None이면 메모리 체크 생략.
    """
    import gc

    if df_prescriptions.empty:
        return pd.DataFrame(columns=_OVERLAP_COLUMNS)

    # guard가 None이면 no-op 가드 사용
    from hana_app.core.memory_guard import get_guard
    _guard = get_guard(guard)

    # 환자별 그룹화: DataFrame groupby로 직접 처리 (PrescriptionRecord 전체 변환 방지)
    pid_col = "INDI_DSCM_NO" if "INDI_DSCM_NO" in df_prescriptions.columns else "patient_id"
    if pid_col not in df_prescriptions.columns:
        return pd.DataFrame(columns=_OVERLAP_COLUMNS)

    result_chunks: list[pd.DataFrame] = []
    batch_pairs: list[dict] = []
    patients_processed = 0

    for patient_id, group_df in df_prescriptions.groupby(pid_col, sort=False):
        prx_list = prescriptions_from_df(group_df)
        if len(prx_list) < 2:
            continue

        pairs = calculate_overlaps_for_patient(prx_list, window_days, min_overlap)
        for p in pairs:
            batch_pairs.append(_overlap_pair_to_dict(p))

        patients_processed += 1

        # 배치 flush: 누적 결과를 DataFrame으로 변환 후 메모리 해제
        if patients_processed % _BATCH_FLUSH_SIZE == 0 and batch_pairs:
            result_chunks.append(pd.DataFrame(batch_pairs))
            batch_pairs = []
            gc.collect()
            # MemoryGuard 체크: HARD_STOP 시 MemoryLimitExceeded 발생
            _guard.check()

    # 잔여 결과 flush
    if batch_pairs:
        result_chunks.append(pd.DataFrame(batch_pairs))
        del batch_pairs
        gc.collect()

    if not result_chunks:
        return pd.DataFrame(columns=_OVERLAP_COLUMNS)

    result = pd.concat(result_chunks, ignore_index=True)
    del result_chunks
    gc.collect()
    return result


def get_concurrent_drug_count(
    prescriptions: list[PrescriptionRecord],
    reference_date: date,
) -> int:
    """특정 기준일 기준 동시 복용 중인 약물 수 (피처용)."""
    return sum(
        1 for p in prescriptions
        if p.start_date <= reference_date <= p.end_date
    )
