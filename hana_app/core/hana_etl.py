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

_AGE_CASE_SQL = """
CASE
    WHEN "{byear_col}" IS NULL THEN 'unknown'
    WHEN (? - CAST("{byear_col}" AS INTEGER)) < 0 THEN 'unknown'
    WHEN (? - CAST("{byear_col}" AS INTEGER)) < 20 THEN '0-19'
    WHEN (? - CAST("{byear_col}" AS INTEGER)) < 40 THEN '20-39'
    WHEN (? - CAST("{byear_col}" AS INTEGER)) < 60 THEN '40-59'
    WHEN (? - CAST("{byear_col}" AS INTEGER)) < 75 THEN '60-74'
    ELSE '75+'
END
""".strip()


def _allocate_sampling_quotas(
    counts_df: pd.DataFrame,
    sample_size: int,
) -> list[dict[str, object]]:
    """GROUP BY 결과를 받아 층별 quota를 최대잉여법으로 배분한다."""
    if counts_df.empty or sample_size <= 0:
        return []

    total = int(counts_df["POP_COUNT"].sum())
    if total <= 0:
        return []

    alloc: list[dict[str, object]] = []
    for row in counts_df.itertuples(index=False):
        cnt = int(getattr(row, "POP_COUNT"))
        exact = cnt / total * sample_size
        quota = min(int(exact), cnt)
        alloc.append({
            "sex": getattr(row, "_SEX"),
            "age_grp": getattr(row, "_AGE_GRP"),
            "addr": getattr(row, "_ADDR"),
            "count": cnt,
            "quota": quota,
            "fractional": exact - quota,
        })

    remaining = sample_size - sum(int(item["quota"]) for item in alloc)
    for item in sorted(alloc, key=lambda x: x["fractional"], reverse=True):
        if remaining <= 0:
            break
        if int(item["quota"]) < int(item["count"]):
            item["quota"] = int(item["quota"]) + 1
            remaining -= 1

    return [item for item in alloc if int(item["quota"]) > 0]


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

    _PID_BATCH = 50_000
    # T30(원내 처방)은 행 수가 압도적으로 많아 HANA 측 단일 쿼리 메모리 부담이
    # 큼. statement-timeout / workload-governor 에 의한 cancel(InternalError 139)
    # 을 방지하기 위해 IN-list를 더 작게 끊는다.
    _PID_BATCH_T30 = 5_000

    def _query_paged_by_pid(
        self,
        sql_base: str,
        params_base: list,
        pid_col: str,
        patient_ids: list[str] | None,
        pid_batch: int | None = None,
    ) -> pd.DataFrame:
        """날짜 조건 SQL에 INDI_DSCM_NO IN (...) 조건을 추가하여 조회.

        patient_ids=None 이면 필터 없이 그대로 실행.
        patient_ids=[] 이면 빈 DataFrame 반환.
        patient_ids 건수가 많으면 pid_batch (기본 _PID_BATCH) 단위로 분할·조합.
        """
        if patient_ids is not None and len(patient_ids) == 0:
            return pd.DataFrame()
        if not patient_ids:
            return self.conn.query_df(sql_base, params_base or None)
        batch_size = pid_batch if pid_batch is not None else self._PID_BATCH
        results = []
        for i in range(0, len(patient_ids), batch_size):
            batch = patient_ids[i: i + batch_size]
            phs = ",".join(["?"] * len(batch))
            sql = f'{sql_base} AND "{pid_col}" IN ({phs})'
            results.append(self.conn.query_df(sql, params_base + list(batch)))
        if len(results) == 1:
            return results[0]
        return pd.concat(results, ignore_index=True)

    # ---- T20 ----------------------------------------------------------------

    def fetch_t20(self, yyyymm_list: list[str],
                  progress_cb: Callable[[str], None] | None = None,
                  patient_ids: list[str] | None = None) -> pd.DataFrame:
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
        return self._query_paged_by_pid(sql, list(yyyymm_list), c["patient_id"], patient_ids)

    # ---- T30 ----------------------------------------------------------------

    def fetch_t30(self, yyyymm_list: list[str],
                  progress_cb: Callable[[str], None] | None = None,
                  patient_ids: list[str] | None = None) -> pd.DataFrame:
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
        return self._query_paged_by_pid(
            sql, list(yyyymm_list), c["patient_id"], patient_ids,
            pid_batch=self._PID_BATCH_T30,
        )

    # ---- T60 ----------------------------------------------------------------

    def fetch_t60(self, yyyymm_list: list[str],
                  progress_cb: Callable[[str], None] | None = None,
                  patient_ids: list[str] | None = None) -> pd.DataFrame:
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
        return self._query_paged_by_pid(sql, list(yyyymm_list), c["patient_id"], patient_ids)

    # ---- T40 ----------------------------------------------------------------

    def fetch_t40(self, yyyymm_list: list[str],
                  progress_cb: Callable[[str], None] | None = None,
                  patient_ids: list[str] | None = None) -> pd.DataFrame:
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
        return self._query_paged_by_pid(sql, list(yyyymm_list), c["patient_id"], patient_ids)

    # ---- 날짜 범위 쿼리 (일 단위 청크용) ------------------------------------

    def fetch_t20_by_date(self, date_from: str, date_to: str,
                          progress_cb: Callable[[str], None] | None = None,
                          patient_ids: list[str] | None = None) -> pd.DataFrame:
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
        return self._query_paged_by_pid(sql, [date_from, date_to], c["patient_id"], patient_ids)

    def fetch_t30_by_date(self, date_from: str, date_to: str,
                          progress_cb: Callable[[str], None] | None = None,
                          patient_ids: list[str] | None = None) -> pd.DataFrame:
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
        return self._query_paged_by_pid(
            sql, [date_from, date_to], c["patient_id"], patient_ids,
            pid_batch=self._PID_BATCH_T30,
        )

    def fetch_t60_by_date(self, date_from: str, date_to: str,
                          progress_cb: Callable[[str], None] | None = None,
                          patient_ids: list[str] | None = None) -> pd.DataFrame:
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
        return self._query_paged_by_pid(sql, [date_from, date_to], c["patient_id"], patient_ids)

    def fetch_t40_by_date(self, date_from: str, date_to: str,
                          progress_cb: Callable[[str], None] | None = None,
                          patient_ids: list[str] | None = None) -> pd.DataFrame:
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
        return self._query_paged_by_pid(sql, [date_from, date_to], c["patient_id"], patient_ids)

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

    # ---- T40 ICD-10 질환 필터 → 환자 ID 추출 ----------------------------------

    def fetch_patients_by_icd10(
        self,
        icd10_prefixes: list[str],
        yyyymm_list: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> list[str]:
        """T40 상병코드(ICD-10)로 해당 환자 ID 목록 반환.

        Parameters
        ----------
        icd10_prefixes : list[str]
            ICD-10 접두사 목록. 3자리 입력 시 하위 코드 포함 (LIKE 'E11%').
            예: ["E11", "I10", "J45"]
        yyyymm_list : list[str], optional
            월 목록 (YYYYMM 형식). date_from/date_to와 택일.
        date_from / date_to : str, optional
            일 단위 날짜 범위 (YYYYMMDD). yyyymm_list와 택일.

        Returns
        -------
        list[str]
            중복 제거된 INDI_DSCM_NO 목록.
        """
        if not icd10_prefixes:
            return []
        if "t40" not in self.tables:
            if progress_cb:
                progress_cb("[건너뜀] T40 테이블 미설정")
            return []

        c = self.cols["t40"]
        tbl = self._tbl("t40")
        pid_col  = c["patient_id"]
        sick_col = c["sick_code"]

        # ICD-10 LIKE 조건 생성
        like_clauses = " OR ".join(
            f'"{sick_col}" LIKE ?' for _ in icd10_prefixes
        )
        like_params = [f"{p.strip().upper()}%" for p in icd10_prefixes]

        if yyyymm_list:
            phs = ",".join(["?"] * len(yyyymm_list))
            sql = (
                f'SELECT DISTINCT "{pid_col}" FROM {tbl} '
                f'WHERE "{c["yyyymm"]}" IN ({phs}) AND ({like_clauses})'
            )
            params = list(yyyymm_list) + like_params
        elif date_from and date_to:
            sql = (
                f'SELECT DISTINCT "{pid_col}" FROM {tbl} '
                f'WHERE "{c["start_date"]}" BETWEEN ? AND ? AND ({like_clauses})'
            )
            params = [date_from, date_to] + like_params
        else:
            if progress_cb:
                progress_cb("[건너뜀] T40 ICD-10 조회: 날짜 범위 미지정")
            return []

        codes_str = ", ".join(icd10_prefixes)
        if progress_cb:
            progress_cb(f"T40 ICD-10 필터 조회: {codes_str} (하위 코드 포함)")

        df = self.conn.query_df(sql, params)
        result = df[pid_col].dropna().astype(str).str.strip().unique().tolist()

        if progress_cb:
            progress_cb(f"T40 ICD-10 완료: {codes_str} → {len(result):,}명")
        return result

    # ---- 자격 DB (Eligibility) → 인구통계 ------------------------------------

    def fetch_eligibility_for_sampling(
        self,
        std_year: str,
        addr_digits: int = 5,
        sample_size: int | None = None,
        seed: int = 42,
        progress_cb: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        """자격DB에서 연도별 인구통계 조회 (사전 층화 샘플링용).

        Parameters
        ----------
        std_year : str
            기준 연도 (STD_YYYY 필터).
        addr_digits : int
            RVSN_ADDR_CD 앞 몇 자리 (5 또는 8).
        sample_size : int, optional
            지정 시 전체 모집단의 층별 인원 수를 먼저 집계한 뒤 비례 quota를 계산하고,
            각 층에서 정확히 quota만큼만 추출합니다.
            None 이면 전체 연도 데이터를 반환합니다 (주의: 대용량).
        seed : int
            층별 샘플 행 선택에 사용할 결정적 시드.

        Returns
        -------
        pd.DataFrame
            INDI_DSCM_NO, BYEAR, SEX_TYPE, RVSN_ADDR_CD(잘린 값) 컬럼.
            중복 INDI_DSCM_NO는 마지막 레코드 유지.
        """
        if "eligibility" not in self.tables:
            if progress_cb:
                progress_cb("[건너뜀] 자격DB 테이블 미설정")
            return pd.DataFrame()

        elig_cfg = self.tables["eligibility"]
        tbl = f'"{elig_cfg["schema"]}"."{elig_cfg["table"]}"'
        c = self.cols.get("eligibility", {})
        pid_col    = c.get("patient_id",   "INDI_DSCM_NO")
        byear_col  = c.get("byear",        "BYEAR")
        sex_col    = c.get("sex_type",     "SEX_TYPE")
        year_col   = c.get("std_year",     "STD_YYYY")
        addr_col   = c.get("rvsn_addr_cd", "RVSN_ADDR_CD")

        for label, value in (
            ("patient_id", pid_col),
            ("byear", byear_col),
            ("sex_type", sex_col),
            ("std_year", year_col),
            ("rvsn_addr_cd", addr_col),
        ):
            _assert_safe_identifier(value, label)

        age_expr = _AGE_CASE_SQL.format(byear_col=byear_col)
        sex_expr = f"COALESCE(NULLIF(TRIM(TO_NVARCHAR(\"{sex_col}\")), ''), 'U')"
        addr_expr = (
            f"COALESCE(SUBSTRING(TO_NVARCHAR(\"{addr_col}\"), 1, {int(addr_digits)}), '')"
        )
        base_cte = f"""
WITH base AS (
    SELECT
        TO_NVARCHAR("{pid_col}") AS "{pid_col}",
        "{byear_col}" AS "{byear_col}",
        "{sex_col}" AS "{sex_col}",
        "{addr_col}" AS "{addr_col}",
        ROW_NUMBER() OVER (
            PARTITION BY TO_NVARCHAR("{pid_col}")
            ORDER BY
                TO_NVARCHAR("{pid_col}") DESC,
                TO_NVARCHAR("{byear_col}") DESC,
                TO_NVARCHAR("{sex_col}") DESC,
                TO_NVARCHAR("{addr_col}") DESC
        ) AS _PID_RN
    FROM {tbl}
    WHERE "{year_col}" = ?
),
dedup AS (
    SELECT
        "{pid_col}",
        "{byear_col}",
        "{sex_col}",
        "{addr_col}",
        {sex_expr} AS _SEX,
        {age_expr} AS _AGE_GRP,
        {addr_expr} AS _ADDR
    FROM base
    WHERE _PID_RN = 1
)
""".strip()

        if sample_size is not None and sample_size > 0:
            count_sql = (
                f"{base_cte} "
                "SELECT _SEX, _AGE_GRP, _ADDR, COUNT(*) AS POP_COUNT "
                "FROM dedup "
                "GROUP BY _SEX, _AGE_GRP, _ADDR"
            )
            count_params: list = [
                std_year,
                int(std_year),
                int(std_year),
                int(std_year),
                int(std_year),
                int(std_year),
            ]
            counts_df = self.conn.query_df(count_sql, count_params)
            quotas = _allocate_sampling_quotas(counts_df, int(sample_size))

            if progress_cb:
                progress_cb(
                    f"자격DB 층별 quota 계산: {tbl} (STD_YYYY={std_year}, seed={int(seed)})"
                )

            if not quotas:
                return pd.DataFrame(columns=[pid_col, byear_col, sex_col, addr_col])

            quota_rows_sql: list[str] = []
            quota_params: list[object] = []
            for quota in quotas:
                quota_rows_sql.append(
                    "SELECT ? AS _SEX, ? AS _AGE_GRP, ? AS _ADDR, ? AS QUOTA FROM DUMMY"
                )
                quota_params.extend([
                    quota["sex"], quota["age_grp"], quota["addr"], int(quota["quota"]),
                ])

            sample_sql = f"""
{base_cte},
quota_map AS (
    {" UNION ALL ".join(quota_rows_sql)}
),
ranked AS (
    SELECT
        d."{pid_col}",
        d."{byear_col}",
        d."{sex_col}",
        d."{addr_col}",
        ROW_NUMBER() OVER (
            PARTITION BY d._SEX, d._AGE_GRP, d._ADDR
            ORDER BY HASH_SHA256(
                COALESCE(d."{pid_col}", '') || '|' || TO_NVARCHAR(?)
            )
        ) AS _STRATA_RN,
        q.QUOTA
    FROM dedup d
    INNER JOIN quota_map q
        ON d._SEX = q._SEX
       AND d._AGE_GRP = q._AGE_GRP
       AND d._ADDR = q._ADDR
)
SELECT "{pid_col}", "{byear_col}", "{sex_col}", "{addr_col}"
FROM ranked
WHERE _STRATA_RN <= QUOTA
ORDER BY "{pid_col}" ASC
""".strip()
            params = (
                [
                    std_year,
                    int(std_year),
                    int(std_year),
                    int(std_year),
                    int(std_year),
                    int(std_year),
                ]
                + quota_params
                + [int(seed)]
            )
            if progress_cb:
                progress_cb(
                    f"자격DB quota 추출 실행: 총 {int(sample_size):,}명 목표, "
                    f"{len(quotas):,}개 층"
                )
            df = self.conn.query_df(sample_sql, params)
        else:
            sql = (
                f"{base_cte} "
                f'SELECT "{pid_col}", "{byear_col}", "{sex_col}", "{addr_col}" '
                f'FROM dedup ORDER BY "{pid_col}" ASC'
            )
            params = [
                std_year,
                int(std_year),
                int(std_year),
                int(std_year),
                int(std_year),
                int(std_year),
            ]
            if progress_cb:
                progress_cb(
                    f"자격DB 전체 조회 (층화 샘플링용): {tbl} (STD_YYYY={std_year})"
                )
            df = self.conn.query_df(sql, params)

        # 중복 INDI_DSCM_NO → 마지막 레코드 유지 (결정적)
        if not df.empty:
            df = df.drop_duplicates(subset=[pid_col], keep="last").reset_index(drop=True)
            df[addr_col] = df[addr_col].astype(str).str[:addr_digits]

        if progress_cb:
            progress_cb(f"자격DB 조회 완료: {len(df):,}명")

        return df

    def fetch_eligibility_demographics(
        self,
        patient_ids: list[str],
        std_year: str,
        addr_digits: int = 5,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, dict]:
        """자격 DB에서 BYEAR·SEX_TYPE·RVSN_ADDR_CD 조회.

        Parameters
        ----------
        patient_ids : list[str]
            처방 추출 후 수집한 고유 INDI_DSCM_NO 목록.
        std_year : str
            추출 기준 연도 (STD_YYYY 필터, 예: "2023").
        addr_digits : int
            RVSN_ADDR_CD 앞 몇 자리를 사용할지 (5 또는 8).
        progress_cb : callable, optional
            진행 메시지 콜백.

        Returns
        -------
        dict[str, dict]
            {patient_id: {"byear": int, "sex_type": str, "addr_cd": str}}
            INDI_DSCM_NO 중복 시 마지막 레코드의 RVSN_ADDR_CD 사용.
        """
        if "eligibility" not in self.tables:
            if progress_cb:
                progress_cb("[건너뜀] 자격DB 테이블 미설정")
            return {}

        elig_cfg = self.tables["eligibility"]
        tbl = f'"{elig_cfg["schema"]}"."{elig_cfg["table"]}"'
        c = self.cols.get("eligibility", {})
        pid_col    = c.get("patient_id",   "INDI_DSCM_NO")
        byear_col  = c.get("byear",        "BYEAR")
        sex_col    = c.get("sex_type",     "SEX_TYPE")
        year_col   = c.get("std_year",     "STD_YYYY")
        addr_col   = c.get("rvsn_addr_cd", "RVSN_ADDR_CD")

        _BATCH = 50_000
        result: dict[str, dict] = {}

        if progress_cb:
            progress_cb(
                f"자격DB 조회: {tbl} (STD_YYYY={std_year}, "
                f"대상 {len(patient_ids):,}명, 배치 {_BATCH:,})"
            )

        for i in range(0, max(len(patient_ids), 1), _BATCH):
            batch = patient_ids[i: i + _BATCH]
            placeholders = ",".join(["?" for _ in batch])
            # ORDER BY pid ASC → drop_duplicates(keep="last") 결과가 항상 결정적
            sql = (
                f'SELECT "{pid_col}", "{byear_col}", "{sex_col}", "{addr_col}" '
                f'FROM {tbl} '
                f'WHERE "{year_col}" = ? AND "{pid_col}" IN ({placeholders}) '
                f'ORDER BY "{pid_col}" ASC'
            )
            params = [std_year] + batch
            df = self.conn.query_df(sql, params)

            # 중복 INDI_DSCM_NO → 마지막 레코드 유지 (ORDER BY pid ASC 기준으로 결정적)
            if not df.empty:
                df = df.drop_duplicates(subset=[pid_col], keep="last")
                for row in df.itertuples(index=False):
                    pid   = str(getattr(row, pid_col, "")).strip()
                    byear = getattr(row, byear_col, None)
                    sex   = str(getattr(row, sex_col, "") or "").strip() or None
                    addr  = str(getattr(row, addr_col, "") or "").strip()
                    if pid:
                        result[pid] = {
                            "byear":    int(byear) if byear is not None else None,
                            "sex_type": sex,
                            "addr_cd":  addr[:addr_digits] if addr else None,
                        }

        if progress_cb:
            progress_cb(f"자격DB 완료: {len(result):,}명 인구통계 매핑")
        return result

    def fetch_eligibility_ages(
        self,
        patient_ids: list[str] | None = None,
        reference_year: int | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, int]:
        """자격 DB에서 BYEAR 조회 → {patient_id: age} 딕셔너리 반환.

        나이 = reference_year - BYEAR.
        reference_year 미지정 시 현재 연도 사용.

        .. deprecated::
            fetch_eligibility_demographics 사용 권장.
            STD_YYYY 필터 없이 전체 테이블을 조회하므로 메모리 사용량이 큽니다.
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
        patient_ids: list[str] | None = None,
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

        t20 = self.fetch_t20(yyyymm_list, progress_cb, patient_ids=patient_ids)
        stats_t20 = len(t20)
        t20_index = t20.set_index(self.cols["t20"]["bill_no"])
        del t20
        gc.collect()

        t40 = self.fetch_t40(yyyymm_list, progress_cb, patient_ids=patient_ids)
        stats_t40 = len(t40)
        t40_idx = build_t40_index(t40, self.cols["t40"]["bill_no"], self.cols["t40"]["sick_code"])
        del t40
        gc.collect()

        t30 = self.fetch_t30(yyyymm_list, progress_cb, patient_ids=patient_ids)
        stats_t30 = len(t30)

        if progress_cb:
            progress_cb("PrescriptionRecord 변환 중 (T30)...")

        records = self._t30_to_records(t30, t20_index, t40_idx)
        del t30
        gc.collect()

        t60 = self.fetch_t60(yyyymm_list, progress_cb, patient_ids=patient_ids)
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
        patient_ids: list[str] | None = None,
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
                t20 = self.fetch_t20_by_date(dt_from, dt_to, progress_cb, patient_ids=patient_ids)
                stats["t20_rows"] += len(t20)
                t20_index = t20.set_index(self.cols["t20"]["bill_no"])
                del t20
                gc.collect()

                t40 = self.fetch_t40_by_date(dt_from, dt_to, progress_cb, patient_ids=patient_ids)
                stats["t40_rows"] += len(t40)
                t40_idx = build_t40_index(t40, self.cols["t40"]["bill_no"], self.cols["t40"]["sick_code"])
                del t40
                gc.collect()

                t30 = self.fetch_t30_by_date(dt_from, dt_to, progress_cb, patient_ids=patient_ids)
                stats["t30_rows"] += len(t30)
                chunk_records = self._t30_to_records(t30, t20_index, t40_idx)
                del t30
                gc.collect()

                t60 = self.fetch_t60_by_date(dt_from, dt_to, progress_cb, patient_ids=patient_ids)
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
                t20 = self.fetch_t20(chunk, progress_cb, patient_ids=patient_ids)
                stats["t20_rows"] += len(t20)
                t20_index = t20.set_index(self.cols["t20"]["bill_no"])
                del t20
                gc.collect()

                t40 = self.fetch_t40(chunk, progress_cb, patient_ids=patient_ids)
                stats["t40_rows"] += len(t40)
                t40_idx = build_t40_index(t40, self.cols["t40"]["bill_no"], self.cols["t40"]["sick_code"])
                del t40
                gc.collect()

                t30 = self.fetch_t30(chunk, progress_cb, patient_ids=patient_ids)
                stats["t30_rows"] += len(t30)
                chunk_records = self._t30_to_records(t30, t20_index, t40_idx)
                del t30
                gc.collect()

                t60 = self.fetch_t60(chunk, progress_cb, patient_ids=patient_ids)
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

def collect_unique_patient_ids(parquet_paths: list[Path]) -> list[str]:
    """Parquet 청크 파일들에서 고유 patient_id(INDI_DSCM_NO) 목록 반환.

    자격DB 조회 전에 호출하여 필요한 환자 ID만 필터링하는 데 사용합니다.
    DuckDB 설치 시 메모리 사용 최소화.
    """
    if not parquet_paths:
        return []
    try:
        import duckdb
        _src_list = ", ".join(f"'{Path(p).as_posix()}'" for p in parquet_paths)
        con = duckdb.connect()
        try:
            rows = con.execute(
                f"SELECT DISTINCT patient_id FROM read_parquet([{_src_list}])"
            ).fetchall()
            return [str(r[0]) for r in rows if r[0] is not None]
        finally:
            con.close()
    except Exception:
        pid_set: set[str] = set()
        for p in parquet_paths:
            try:
                df = pd.read_parquet(p, columns=["patient_id"])
                pid_set.update(df["patient_id"].dropna().astype(str).unique())
                del df
                gc.collect()
            except Exception:
                pass
        return list(pid_set)


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
