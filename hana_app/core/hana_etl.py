"""
HANA DB -> 처방 데이터 추출 및 전처리

T20 (요양명세서)  +  T30 (원내 약품)  +  T60 (원외 처방)
-> PrescriptionRecord 리스트  ->  PatientFeatures 리스트

메모리 효율화:
  - extract_prescriptions        : 기존 방식 (소규모 데이터)
  - extract_prescriptions_chunked: 월별 청크 처리 + Parquet 저장 (대용량)
"""
from __future__ import annotations

import gc
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.etl.models import PrescriptionRecord
from hana_app.core.db import _assert_safe_identifier

logger = logging.getLogger(__name__)

# Parquet 저장 디렉토리
RAW_DIR = ROOT / "data" / "raw"

# PrescriptionRecord -> Parquet 컬럼 스키마
_RECORD_COLS = [
    "patient_id", "institution_id", "bill_no", "wk_compn_cd",
    "edi_code", "gnl_nm_cd", "efmdc_clsf_no",
    "start_date", "end_date", "total_days",
    "dose_once", "dose_freq",
    "sick_code", "sex", "age_id", "institution_type", "source",
]


# ---------------------------------------------------------------------------
# 날짜 헬퍼
# ---------------------------------------------------------------------------

def _parse_date(s: str | None) -> date | None:
    try:
        s = str(s).strip()
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except Exception:
        return None


def _yyyymm_range(year_from: str, month_from: str,
                  year_to: str, month_to: str) -> list[str]:
    result = []
    y, m = int(year_from), int(month_from)
    ye, me = int(year_to), int(month_to)
    while (y, m) <= (ye, me):
        result.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return result


def _shift_yyyymm(yyyymm: str, months: int) -> str:
    """YYYYMM 문자열에서 months 개월만큼 이동 (음수=과거)."""
    y, m = int(yyyymm[:4]), int(yyyymm[4:])
    m += months
    while m <= 0:
        m += 12
        y -= 1
    while m > 12:
        m -= 12
        y += 1
    return f"{y:04d}{m:02d}"


def _date_range_days(
    year_from: str, month_from: str,
    year_to: str, month_to: str,
    chunk_days: int = 1,
) -> list[tuple[str, str]]:
    """날짜 범위를 chunk_days 단위로 분할. [(start_YYYYMMDD, end_YYYYMMDD), ...]"""
    from calendar import monthrange
    start = date(int(year_from), int(month_from), 1)
    end_y, end_m = int(year_to), int(month_to)
    end = date(end_y, end_m, monthrange(end_y, end_m)[1])

    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        chunks.append((cursor.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        cursor = chunk_end + timedelta(days=1)
    return chunks


# ---------------------------------------------------------------------------
# PrescriptionRecord <-> DataFrame 변환
# ---------------------------------------------------------------------------

def records_to_df(records: list[PrescriptionRecord]) -> pd.DataFrame:
    """PrescriptionRecord 리스트 -> 평탄화된 DataFrame.

    컬럼별 리스트로 직접 구축하여 중간 dict 생성을 방지합니다 (메모리 절약).
    """
    if not records:
        return pd.DataFrame(columns=_RECORD_COLS)
    return pd.DataFrame({
        "patient_id":       [r.patient_id for r in records],
        "institution_id":   [r.institution_id for r in records],
        "bill_no":          [r.bill_no for r in records],
        "wk_compn_cd":      [r.wk_compn_cd for r in records],
        "edi_code":         [r.edi_code for r in records],
        "gnl_nm_cd":        [r.gnl_nm_cd for r in records],
        "efmdc_clsf_no":    [r.efmdc_clsf_no for r in records],
        "start_date":       [r.start_date.isoformat() for r in records],
        "end_date":         [r.end_date.isoformat() for r in records],
        "total_days":       [r.total_days for r in records],
        "dose_once":        [r.dose_once for r in records],
        "dose_freq":        [r.dose_freq for r in records],
        "sick_code":        [r.sick_code for r in records],
        "sex":              [r.sex for r in records],
        "age_id":           [r.age_id for r in records],
        "institution_type": [r.institution_type for r in records],
        "source":           [r.source for r in records],
    })


def df_row_to_record(row) -> PrescriptionRecord:
    """DataFrame 행 (itertuples) -> PrescriptionRecord."""
    def _d(s: str | None) -> date:
        try:
            s = str(s)
            return date(int(s[:4]), int(s[5:7]), int(s[8:10]))
        except Exception:
            return date.today()

    return PrescriptionRecord(
        patient_id=str(row.patient_id or ""),
        institution_id=str(row.institution_id or ""),
        bill_no=str(row.bill_no or ""),
        wk_compn_cd=str(row.wk_compn_cd or ""),
        edi_code=row.edi_code or None,
        gnl_nm_cd=row.gnl_nm_cd or None,
        efmdc_clsf_no=row.efmdc_clsf_no or None,
        start_date=_d(row.start_date),
        end_date=_d(row.end_date),
        total_days=int(row.total_days or 1),
        dose_once=float(row.dose_once or 1.0),
        dose_freq=int(row.dose_freq or 1),
        sick_code=row.sick_code or None,
        sex=row.sex or None,
        age_id=row.age_id or None,
        institution_type=row.institution_type or None,
        source=str(row.source or "T30"),
    )


# ---------------------------------------------------------------------------
# T40 상병 인덱스 빌더
# ---------------------------------------------------------------------------

def build_t40_index(t40: pd.DataFrame, bill_col: str, sick_col: str) -> dict[str, str]:
    """T40 DataFrame -> {bill_no: primary_sick_code} 매핑.

    한 명세서에 상병이 여러 개인 경우 첫 행(주상병)을 사용.
    """
    if t40.empty:
        return {}
    index: dict[str, str] = {}
    for row in t40.itertuples(index=False):
        bn = str(getattr(row, bill_col, "") or "").strip()
        sc = str(getattr(row, sick_col, "") or "").strip()
        if bn and sc and bn not in index:
            index[bn] = sc
    return index


# ---------------------------------------------------------------------------
# HANA SQL 쿼리 빌더
# ---------------------------------------------------------------------------

class HANAExtractor:
    """HANA DB에서 T20/T30/T60 데이터 추출."""

    def __init__(self, conn, table_cfg: dict, col_cfg: dict) -> None:
        self.conn = conn
        self.tables = table_cfg
        self.cols = col_cfg

    def _tbl(self, key: str) -> str:
        t = self.tables[key]
        _assert_safe_identifier(t["schema"], "schema")
        _assert_safe_identifier(t["table"], "table")
        return f'"{t["schema"]}"."{t["table"]}"'

    def _col(self, tbl: str, field: str) -> str:
        val = self.cols[tbl][field]
        _assert_safe_identifier(val, f"{tbl}.{field}")
        return f'"{val}"'

    # ---- T20 ----------------------------------------------------------------

    def fetch_t20(self, yyyymm_list: list[str],
                  progress_cb: Callable[[str], None] | None = None) -> pd.DataFrame:
        c = self.cols["t20"]
        tbl = self._tbl("t20")
        placeholders = ",".join(["?" for _ in yyyymm_list])
        sql = (
            f'SELECT "{c["bill_no"]}", "{c["patient_id"]}", "{c["institution_id"]}", '
            f'"{c["start_date"]}", "{c["sex"]}", "{c["age_id"]}", '
            f'"{c["institution_type"]}" '
            f"FROM {tbl} "
            f'WHERE "{c["yyyymm"]}" IN ({placeholders})'
        )
        if progress_cb:
            progress_cb(f"T20 조회: {yyyymm_list[0]}~{yyyymm_list[-1]}")
        return self.conn.query_df(sql, yyyymm_list)

    # ---- T30 ----------------------------------------------------------------

    def fetch_t30(self, yyyymm_list: list[str],
                  progress_cb: Callable[[str], None] | None = None) -> pd.DataFrame:
        c = self.cols["t30"]
        tbl = self._tbl("t30")
        placeholders = ",".join(["?" for _ in yyyymm_list])
        sql = (
            f'SELECT "{c["bill_no"]}", "{c["patient_id"]}", "{c["start_date"]}", '
            f'"{c["drug_code"]}", "{c["drug_code_alt"]}", '
            f'"{c["edi_code"]}", "{c["efmdc"]}", '
            f'"{c["dose_once"]}", "{c["dose_freq"]}", "{c["total_days"]}" '
            f"FROM {tbl} "
            f'WHERE "{c["yyyymm"]}" IN ({placeholders})'
        )
        if progress_cb:
            progress_cb(f"T30 (원내) 조회: {yyyymm_list[0]}~{yyyymm_list[-1]}")
        return self.conn.query_df(sql, yyyymm_list)

    # ---- T60 ----------------------------------------------------------------

    def fetch_t60(self, yyyymm_list: list[str],
                  progress_cb: Callable[[str], None] | None = None) -> pd.DataFrame:
        c = self.cols["t60"]
        tbl = self._tbl("t60")
        placeholders = ",".join(["?" for _ in yyyymm_list])
        sql = (
            f'SELECT "{c["bill_no"]}", "{c["patient_id"]}", "{c["start_date"]}", '
            f'"{c["drug_code"]}", "{c["drug_code_alt"]}", '
            f'"{c["edi_code"]}", '
            f'"{c["dose_once"]}", "{c["dose_freq"]}", "{c["total_days"]}", '
            f'"{c["sick_code"]}", "{c["institution_id"]}" '
            f"FROM {tbl} "
            f'WHERE "{c["yyyymm"]}" IN ({placeholders})'
        )
        if progress_cb:
            progress_cb(f"T60 (원외) 조회: {yyyymm_list[0]}~{yyyymm_list[-1]}")
        return self.conn.query_df(sql, yyyymm_list)

    # ---- T40 ----------------------------------------------------------------

    def fetch_t40(self, yyyymm_list: list[str],
                  progress_cb: Callable[[str], None] | None = None) -> pd.DataFrame:
        c = self.cols["t40"]
        tbl = self._tbl("t40")
        placeholders = ",".join(["?" for _ in yyyymm_list])
        sql = (
            f'SELECT "{c["bill_no"]}", "{c["patient_id"]}", "{c["start_date"]}", '
            f'"{c["sick_code"]}", "{c["sick_type"]}" '
            f"FROM {tbl} "
            f'WHERE "{c["yyyymm"]}" IN ({placeholders})'
        )
        if progress_cb:
            progress_cb(f"T40 (상병) 조회: {yyyymm_list[0]}~{yyyymm_list[-1]}")
        return self.conn.query_df(sql, yyyymm_list)

    # ---- 날짜 범위 쿼리 (일 단위 청크용) ------------------------------------

    def fetch_t20_by_date(self, date_from: str, date_to: str,
                          progress_cb: Callable[[str], None] | None = None) -> pd.DataFrame:
        c = self.cols["t20"]
        tbl = self._tbl("t20")
        sql = (
            f'SELECT "{c["bill_no"]}", "{c["patient_id"]}", "{c["institution_id"]}", '
            f'"{c["start_date"]}", "{c["sex"]}", "{c["age_id"]}", '
            f'"{c["institution_type"]}" '
            f"FROM {tbl} "
            f'WHERE "{c["start_date"]}" BETWEEN ? AND ?'
        )
        if progress_cb:
            progress_cb(f"T20 조회: {date_from}~{date_to}")
        return self.conn.query_df(sql, [date_from, date_to])

    def fetch_t30_by_date(self, date_from: str, date_to: str,
                          progress_cb: Callable[[str], None] | None = None) -> pd.DataFrame:
        c = self.cols["t30"]
        tbl = self._tbl("t30")
        sql = (
            f'SELECT "{c["bill_no"]}", "{c["patient_id"]}", "{c["start_date"]}", '
            f'"{c["drug_code"]}", "{c["drug_code_alt"]}", '
            f'"{c["edi_code"]}", "{c["efmdc"]}", '
            f'"{c["dose_once"]}", "{c["dose_freq"]}", "{c["total_days"]}" '
            f"FROM {tbl} "
            f'WHERE "{c["start_date"]}" BETWEEN ? AND ?'
        )
        if progress_cb:
            progress_cb(f"T30 (원내) 조회: {date_from}~{date_to}")
        return self.conn.query_df(sql, [date_from, date_to])

    def fetch_t60_by_date(self, date_from: str, date_to: str,
                          progress_cb: Callable[[str], None] | None = None) -> pd.DataFrame:
        c = self.cols["t60"]
        tbl = self._tbl("t60")
        sql = (
            f'SELECT "{c["bill_no"]}", "{c["patient_id"]}", "{c["start_date"]}", '
            f'"{c["drug_code"]}", "{c["drug_code_alt"]}", '
            f'"{c["edi_code"]}", '
            f'"{c["dose_once"]}", "{c["dose_freq"]}", "{c["total_days"]}", '
            f'"{c["sick_code"]}", "{c["institution_id"]}" '
            f"FROM {tbl} "
            f'WHERE "{c["start_date"]}" BETWEEN ? AND ?'
        )
        if progress_cb:
            progress_cb(f"T60 (원외) 조회: {date_from}~{date_to}")
        return self.conn.query_df(sql, [date_from, date_to])

    def fetch_t40_by_date(self, date_from: str, date_to: str,
                          progress_cb: Callable[[str], None] | None = None) -> pd.DataFrame:
        c = self.cols["t40"]
        tbl = self._tbl("t40")
        sql = (
            f'SELECT "{c["bill_no"]}", "{c["patient_id"]}", "{c["start_date"]}", '
            f'"{c["sick_code"]}", "{c["sick_type"]}" '
            f"FROM {tbl} "
            f'WHERE "{c["start_date"]}" BETWEEN ? AND ?'
        )
        if progress_cb:
            progress_cb(f"T40 (상병) 조회: {date_from}~{date_to}")
        return self.conn.query_df(sql, [date_from, date_to])

    # ---- 요양기관 -----------------------------------------------------------

    def fetch_yoyang(self, std_year: str) -> pd.DataFrame:
        c = self.cols["yoyang"]
        tbl = self._tbl("yoyang")
        sql = (
            f'SELECT "{c["institution_id"]}", "{c["institution_type"]}", '
            f'"{c["inst_name"]}", "{c["addr_sgg"]}" '
            f'FROM {tbl} WHERE "{c["std_year"]}" = ?'
        )
        return self.conn.query_df(sql, [std_year])

    # ---- 자격 DB (Eligibility) → 환자 나이 -----------------------------------

    def fetch_eligibility_ages(
        self,
        patient_ids: list[str] | None = None,
        reference_year: int | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, int]:
        """자격 DB에서 BYEAR 조회 → {patient_id: age} 딕셔너리 반환.

        나이 = reference_year - BYEAR.
        reference_year 미지정 시 현재 연도 사용.
        """
        from datetime import date as _date

        if "eligibility" not in self.tables:
            if progress_cb:
                progress_cb("[건너뜀] 자격DB 테이블 미설정")
            return {}

        ref_year = reference_year or _date.today().year

        elig_cfg = self.tables["eligibility"]
        tbl = f'"{elig_cfg["schema"]}"."{elig_cfg["table"]}"'
        c = self.cols.get("eligibility", {})
        pid_col = c.get("patient_id", "INDI_DSCM_NO")
        byear_col = c.get("byear", "BYEAR")

        if patient_ids and len(patient_ids) <= 50_000:
            placeholders = ",".join(["?" for _ in patient_ids])
            sql = (
                f'SELECT "{pid_col}", "{byear_col}" '
                f'FROM {tbl} WHERE "{pid_col}" IN ({placeholders})'
            )
            params = list(patient_ids)
        else:
            sql = f'SELECT "{pid_col}", "{byear_col}" FROM {tbl}'
            params = None

        if progress_cb:
            progress_cb(f"자격DB 조회: {tbl} (BYEAR → 나이 변환, 기준년도={ref_year})")

        df = self.conn.query_df(sql, params)
        age_map: dict[str, int] = {}
        for row in df.itertuples(index=False):
            pid = str(getattr(row, pid_col, "")).strip()
            byear = getattr(row, byear_col, None)
            if pid and byear is not None:
                try:
                    age_map[pid] = ref_year - int(byear)
                except (ValueError, TypeError):
                    pass

        if progress_cb:
            progress_cb(f"자격DB 완료: {len(age_map):,}명 나이 매핑")
        return age_map

    # ---- T30 DataFrame -> PrescriptionRecord 변환 ---------------------------

    def _t30_to_records(self, t30: pd.DataFrame,
                        t20_index: pd.DataFrame,
                        t40_index: dict[str, str] | None = None) -> list[PrescriptionRecord]:
        c30 = self.cols["t30"]
        c20 = self.cols["t20"]
        records: list[PrescriptionRecord] = []
        for row in t30.itertuples(index=False):
            bill_no = getattr(row, c30["bill_no"])
            start_dt = _parse_date(getattr(row, c30["start_date"]))
            if not start_dt:
                continue
            total_days = int(getattr(row, c30["total_days"]) or 1)
            wk     = str(getattr(row, c30["drug_code"]) or "").strip()
            wk_alt = str(getattr(row, c30["drug_code_alt"]) or "").strip()
            drug_code = wk if wk else wk_alt
            if not drug_code:
                continue
            t20r = t20_index.loc[bill_no] if bill_no in t20_index.index else None
            records.append(PrescriptionRecord(
                patient_id=str(getattr(row, c30["patient_id"]) or ""),
                institution_id=str(t20r[c20["institution_id"]] if t20r is not None else ""),
                bill_no=str(bill_no),
                wk_compn_cd=drug_code,
                edi_code=str(getattr(row, c30["edi_code"]) or "") or None,
                efmdc_clsf_no=str(getattr(row, c30["efmdc"]) or "") or None,
                start_date=start_dt,
                end_date=start_dt + timedelta(days=max(total_days - 1, 0)),
                total_days=total_days,
                dose_once=float(getattr(row, c30["dose_once"]) or 1.0),
                dose_freq=int(getattr(row, c30["dose_freq"]) or 1),
                sex=str(t20r[c20["sex"]] if t20r is not None else "") or None,
                age_id=str(t20r[c20["age_id"]] if t20r is not None else "") or None,
                institution_type=str(t20r[c20["institution_type"]] if t20r is not None else "") or None,
                sick_code=t40_index.get(str(bill_no)) if t40_index else None,
                source="T30",
            ))
        return records

    # ---- T60 DataFrame -> PrescriptionRecord 변환 ---------------------------

    def _t60_to_records(self, t60: pd.DataFrame,
                        t20_index: pd.DataFrame,
                        t40_index: dict[str, str] | None = None) -> list[PrescriptionRecord]:
        c60 = self.cols["t60"]
        c20 = self.cols["t20"]
        records: list[PrescriptionRecord] = []
        for row in t60.itertuples(index=False):
            bill_no = getattr(row, c60["bill_no"])
            start_dt = _parse_date(getattr(row, c60["start_date"]))
            if not start_dt:
                continue
            total_days = int(getattr(row, c60["total_days"]) or 1)
            gnl    = str(getattr(row, c60["drug_code"]) or "").strip()
            wk_alt = str(getattr(row, c60["drug_code_alt"]) or "").strip()
            drug_code = gnl if gnl else wk_alt
            if not drug_code:
                continue
            t20r = t20_index.loc[bill_no] if bill_no in t20_index.index else None
            records.append(PrescriptionRecord(
                patient_id=str(getattr(row, c60["patient_id"]) or ""),
                institution_id=str(getattr(row, c60["institution_id"]) or ""),
                bill_no=str(bill_no),
                wk_compn_cd=drug_code,
                gnl_nm_cd=gnl or None,
                edi_code=str(getattr(row, c60["edi_code"]) or "") or None,
                start_date=start_dt,
                end_date=start_dt + timedelta(days=max(total_days - 1, 0)),
                total_days=total_days,
                dose_once=float(getattr(row, c60["dose_once"]) or 1.0),
                dose_freq=int(getattr(row, c60["dose_freq"]) or 1),
                sick_code=str(getattr(row, c60["sick_code"]) or "").strip() or (t40_index.get(str(bill_no)) if t40_index else None),
                sex=str(t20r[c20["sex"]] if t20r is not None else "") or None,
                age_id=str(t20r[c20["age_id"]] if t20r is not None else "") or None,
                institution_type=str(t20r[c20["institution_type"]] if t20r is not None else "") or None,
                source="T60",
            ))
        return records

    # ---- 통합 추출 (기존 방식, 소규모용) ------------------------------------

    def extract_prescriptions(
        self,
        year_from: str, month_from: str,
        year_to: str,   month_to: str,
        window_days: int = 90,
        poly_threshold: int = 5,
        buffer_days: int = 90,
        buffer_after_days: int = 0,
        progress_cb: Callable[[str], None] | None = None,
    ) -> tuple[list[PrescriptionRecord], dict]:
        """전 기간 일괄 추출 (소규모 데이터용).

        buffer_days       : 분석 시작일 이전 버퍼 (기본 90일).
        buffer_after_days : 분석 종료일 이후 버퍼 (기본 0일).
        """
        analysis_start = f"{year_from}{month_from}"
        analysis_end = f"{year_to}{month_to}"

        # 시작 전 버퍼: 쿼리 시작을 앞당김
        buffer_before_months = max(1, (buffer_days + 29) // 30) if buffer_days > 0 else 0
        query_start = _shift_yyyymm(analysis_start, -buffer_before_months) if buffer_before_months else analysis_start

        # 종료 후 버퍼: 쿼리 종료를 뒤로 늘림
        buffer_after_months = max(1, (buffer_after_days + 29) // 30) if buffer_after_days > 0 else 0
        query_end = _shift_yyyymm(analysis_end, buffer_after_months) if buffer_after_months else analysis_end

        yyyymm_list = _yyyymm_range(query_start[:4], query_start[4:], query_end[:4], query_end[4:])
        if progress_cb:
            buf_parts = []
            if buffer_before_months:
                buf_parts.append(f"시작 전 {buffer_before_months}개월")
            if buffer_after_months:
                buf_parts.append(f"종료 후 {buffer_after_months}개월")
            buf_label = " + ".join(buf_parts) if buf_parts else "없음"
            progress_cb(
                f"분석 기간: {analysis_start} ~ {analysis_end}  "
                f"(쿼리 범위: {query_start} ~ {query_end}, "
                f"버퍼: {buf_label})"
            )

        t20 = self.fetch_t20(yyyymm_list, progress_cb)
        stats_t20 = len(t20)
        t20_index = t20.set_index(self.cols["t20"]["bill_no"])
        del t20
        gc.collect()

        t40 = self.fetch_t40(yyyymm_list, progress_cb)
        stats_t40 = len(t40)
        t40_idx = build_t40_index(t40, self.cols["t40"]["bill_no"], self.cols["t40"]["sick_code"])
        del t40
        gc.collect()

        t30 = self.fetch_t30(yyyymm_list, progress_cb)
        stats_t30 = len(t30)

        if progress_cb:
            progress_cb("PrescriptionRecord 변환 중 (T30)...")

        records = self._t30_to_records(t30, t20_index, t40_idx)
        del t30
        gc.collect()

        t60 = self.fetch_t60(yyyymm_list, progress_cb)
        stats_t60 = len(t60)

        if progress_cb:
            progress_cb("PrescriptionRecord 변환 중 (T60)...")

        records += self._t60_to_records(t60, t20_index, t40_idx)
        del t60, t20_index, t40_idx
        gc.collect()

        stats = {
            "t20_rows": stats_t20,
            "t30_rows": stats_t30,
            "t40_rows": stats_t40,
            "t60_rows": stats_t60,
            "total_records": len(records),
            "unique_patients": len({r.patient_id for r in records}),
            "period": f"{analysis_start}~{analysis_end}",
            "query_period": f"{query_start}~{query_end}",
            "buffer_before_months": buffer_before_months,
            "buffer_after_months": buffer_after_months,
        }
        if progress_cb:
            progress_cb(
                f"추출 완료 - 총 {stats['total_records']:,}건 / "
                f"환자 {stats['unique_patients']:,}명"
            )
        return records, stats

    # ---- 청크 추출 (대용량, 메모리 효율화) ----------------------------------

    def extract_prescriptions_chunked(
        self,
        year_from: str, month_from: str,
        year_to: str,   month_to: str,
        save_dir: Path | str | None = None,
        chunk_months: int = 1,
        chunk_unit: str = "month",
        chunk_days: int = 1,
        window_days: int = 90,
        poly_threshold: int = 5,
        buffer_days: int = 90,
        buffer_after_days: int = 0,
        memory_limit_mb: int = 0,
        progress_cb: Callable[[str], None] | None = None,
    ) -> tuple[list[Path], dict]:
        """
        청크 단위 추출 -> Parquet 저장 (메모리 효율화).

        chunk_unit        : "month" (월 단위) 또는 "day" (일 단위)
        chunk_months      : chunk_unit="month"일 때 청크 크기 (개월)
        chunk_days        : chunk_unit="day"일 때 청크 크기 (일)
        buffer_days       : 분석 시작일 이전 버퍼 (기본 90일).
        buffer_after_days : 분석 종료일 이후 버퍼 (기본 0일).
        memory_limit_mb   : RAM 한도(MB). 0이면 기본값.
        """
        if save_dir is None:
            save_dir = RAW_DIR
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        analysis_start = f"{year_from}{month_from}"
        analysis_end = f"{year_to}{month_to}"

        # 시작 전 버퍼
        buffer_before_months = max(1, (buffer_days + 29) // 30) if buffer_days > 0 else 0
        query_start = _shift_yyyymm(analysis_start, -buffer_before_months) if buffer_before_months else analysis_start

        # 종료 후 버퍼
        buffer_after_months = max(1, (buffer_after_days + 29) // 30) if buffer_after_days > 0 else 0
        query_end = _shift_yyyymm(analysis_end, buffer_after_months) if buffer_after_months else analysis_end

        if progress_cb:
            buf_parts = []
            if buffer_before_months:
                buf_parts.append(f"시작 전 {buffer_before_months}개월")
            if buffer_after_months:
                buf_parts.append(f"종료 후 {buffer_after_months}개월")
            buf_label = " + ".join(buf_parts) if buf_parts else "없음"
            progress_cb(
                f"분석 기간: {analysis_start} ~ {analysis_end}  "
                f"(쿼리 범위: {query_start} ~ {query_end}, "
                f"버퍼: {buf_label})"
            )

        # ── 청크 목록 생성 (월 단위 또는 일 단위) ────────────────────
        _use_daily = (chunk_unit == "day")

        if _use_daily:
            # 일 단위: [(start_YYYYMMDD, end_YYYYMMDD), ...]
            day_chunks = _date_range_days(
                query_start[:4], query_start[4:],
                query_end[:4], query_end[4:],
                chunk_days=chunk_days,
            )
            if progress_cb:
                progress_cb(
                    f"일 단위 청크: {len(day_chunks)}개 "
                    f"({chunk_days}일 단위)"
                )
        else:
            # 월 단위 (기존 방식)
            yyyymm_list = _yyyymm_range(query_start[:4], query_start[4:], query_end[:4], query_end[4:])
            day_chunks = None

        month_chunks = None if _use_daily else [
            yyyymm_list[i:i + chunk_months]
            for i in range(0, len(yyyymm_list), chunk_months)
        ]

        _total_chunks = len(day_chunks) if _use_daily else len(month_chunks)

        parquet_paths: list[Path] = []
        stats: dict = {
            "t20_rows": 0, "t30_rows": 0, "t40_rows": 0, "t60_rows": 0,
            "total_records": 0, "chunks": _total_chunks,
            "chunk_unit": chunk_unit,
            "period": f"{analysis_start}~{analysis_end}",
            "query_period": f"{query_start}~{query_end}",
            "buffer_before_months": buffer_before_months,
            "buffer_after_months": buffer_after_months,
        }

        for i in range(_total_chunks):
            if _use_daily:
                dt_from, dt_to = day_chunks[i]
                label = f"{dt_from}_{dt_to}" if dt_from != dt_to else dt_from
                if progress_cb:
                    progress_cb(
                        f"[{i+1}/{_total_chunks}] 일별 청크: "
                        f"{dt_from}~{dt_to}"
                    )

                # 일 단위 쿼리: MDCARE_STRT_DT BETWEEN
                t20 = self.fetch_t20_by_date(dt_from, dt_to, progress_cb)
                stats["t20_rows"] += len(t20)
                t20_index = t20.set_index(self.cols["t20"]["bill_no"])
                del t20
                gc.collect()

                t40 = self.fetch_t40_by_date(dt_from, dt_to, progress_cb)
                stats["t40_rows"] += len(t40)
                t40_idx = build_t40_index(t40, self.cols["t40"]["bill_no"], self.cols["t40"]["sick_code"])
                del t40
                gc.collect()

                t30 = self.fetch_t30_by_date(dt_from, dt_to, progress_cb)
                stats["t30_rows"] += len(t30)
                chunk_records = self._t30_to_records(t30, t20_index, t40_idx)
                del t30
                gc.collect()

                t60 = self.fetch_t60_by_date(dt_from, dt_to, progress_cb)
                stats["t60_rows"] += len(t60)
                chunk_records += self._t60_to_records(t60, t20_index, t40_idx)
                del t60, t20_index, t40_idx
                gc.collect()
            else:
                chunk = month_chunks[i]
                label = f"{chunk[0]}" if len(chunk) == 1 else f"{chunk[0]}_{chunk[-1]}"
                if progress_cb:
                    progress_cb(
                        f"[{i+1}/{_total_chunks}] 월별 청크: "
                        f"{chunk[0]}~{chunk[-1]}"
                    )

                # 월 단위 쿼리 (기존 방식): MDCARE_STRT_YYYYMM IN
                t20 = self.fetch_t20(chunk, progress_cb)
                stats["t20_rows"] += len(t20)
                t20_index = t20.set_index(self.cols["t20"]["bill_no"])
                del t20
                gc.collect()

                t40 = self.fetch_t40(chunk, progress_cb)
                stats["t40_rows"] += len(t40)
                t40_idx = build_t40_index(t40, self.cols["t40"]["bill_no"], self.cols["t40"]["sick_code"])
                del t40
                gc.collect()

                t30 = self.fetch_t30(chunk, progress_cb)
                stats["t30_rows"] += len(t30)
                chunk_records = self._t30_to_records(t30, t20_index, t40_idx)
                del t30
                gc.collect()

                t60 = self.fetch_t60(chunk, progress_cb)
                stats["t60_rows"] += len(t60)
                chunk_records += self._t60_to_records(t60, t20_index, t40_idx)
                del t60, t20_index, t40_idx
                gc.collect()

            stats["total_records"] += len(chunk_records)

            # Parquet 저장 후 records 해제
            if chunk_records:
                parquet_path = save_dir / f"records_{label}.parquet"
                records_to_df(chunk_records).to_parquet(parquet_path, index=False)
                parquet_paths.append(parquet_path)
                if progress_cb:
                    progress_cb(
                        f"  저장: {parquet_path.name} ({len(chunk_records):,}건)"
                    )
            del chunk_records
            gc.collect()

        stats["unique_patients"] = _count_unique_patients(parquet_paths, memory_limit_mb=memory_limit_mb)

        if progress_cb:
            progress_cb(
                f"청크 추출 완료 - 총 {stats['total_records']:,}건 / "
                f"환자 {stats['unique_patients']:,}명 / "
                f"파일 {len(parquet_paths)}개"
            )
        return parquet_paths, stats


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _count_unique_patients(parquet_paths: list[Path], memory_limit_mb: int = 0) -> int:
    """Parquet 파일들에서 고유 환자 수 집계 (메모리 절약).

    DuckDB 설치 시: COUNT(DISTINCT) 쿼리로 Python 메모리 최소화.
    미설치 시: pandas 루프 폴백.
    """
    try:
        import duckdb
        import tempfile, shutil
        _src_list = ", ".join(f"'{Path(p).as_posix()}'" for p in parquet_paths)
        _mem = max(256, memory_limit_mb // 4) if memory_limit_mb > 0 else 512
        _tmp = Path(tempfile.mkdtemp(prefix="duck_cnt_"))
        con = duckdb.connect()
        try:
            con.execute(f"SET memory_limit='{_mem}MB'")
            con.execute(f"SET temp_directory='{_tmp.as_posix()}'")
            row = con.execute(
                f"SELECT COUNT(DISTINCT patient_id) FROM read_parquet([{_src_list}])"
            ).fetchone()
            return int(row[0])
        finally:
            con.close()
            shutil.rmtree(_tmp, ignore_errors=True)
    except Exception:
        patient_ids: set[str] = set()
        for p in parquet_paths:
            try:
                df = pd.read_parquet(p, columns=["patient_id"])
                patient_ids.update(df["patient_id"].unique())
                del df
                gc.collect()
            except MemoryError:
                # 메모리 부족 시 현재까지 수집한 수라도 반환
                gc.collect()
                break
        return len(patient_ids)
