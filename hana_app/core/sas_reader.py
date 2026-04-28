"""
SAS 데이터 파일 읽기 모듈 (HANA DB 대체 경로)

지원 형식: .sas7bdat, .xpt (SAS XPORT)
라이브러리: pyreadstat (우선) → pandas.read_sas (폴백)

NHIS 데이터 특성:
  - 인코딩: cp949 (기본) / euc-kr / utf-8
  - 대용량: 청크(chunk) 단위 처리
  - 컬럼명: HANA와 동일한 NHIS 표준 컬럼명 사용
"""
from __future__ import annotations

import gc
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.etl.models import PrescriptionRecord

from .strata_utils import byear_to_age_band

logger = logging.getLogger(__name__)

# SAS 파일 확장자
SAS_EXTENSIONS = (".sas7bdat", ".xpt", ".sas7bcat")


# ─────────────────────────────────────────────────────────────────────────────
# 파일 스캔
# ─────────────────────────────────────────────────────────────────────────────

def scan_sas_files(folder: str | Path) -> list[Path]:
    """폴더에서 SAS 파일 목록 반환 (재귀 미포함)."""
    folder = Path(folder)
    if not folder.exists():
        return []
    files = [
        f for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in SAS_EXTENSIONS
    ]
    return files


def guess_table_type(filename: str) -> str | None:
    """파일명에서 테이블 종류 추정 (T20/T30/T40/T60/요양기관)."""
    name = filename.lower()
    if "t20" in name or "tmsbj20" in name or "명세서" in name:
        return "t20"
    if "t30" in name or "tmsbj30" in name or "진료내역" in name:
        return "t30"
    if "t40" in name or "tmsbj40" in name or "상병" in name:
        return "t40"
    if "t60" in name or "tmsbj60" in name or "원외" in name or "처방" in name:
        return "t60"
    if "yoyang" in name or "요양기관" in name or "mdcin" in name or "기관" in name:
        return "yoyang"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SAS 파일 읽기 (단일 / 청크)
# ─────────────────────────────────────────────────────────────────────────────

def _read_with_pyreadstat(
    file_path: Path,
    encoding: str,
    usecols: list[str] | None,
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    """pyreadstat 기반 청크 읽기."""
    import pyreadstat

    read_fn = (
        pyreadstat.read_xport
        if file_path.suffix.lower() == ".xpt"
        else pyreadstat.read_sas7bdat
    )

    kwargs: dict = {"encoding": encoding}
    if usecols:
        kwargs["usecols"] = usecols

    reader = pyreadstat.read_file_in_chunks(
        read_fn,
        str(file_path),
        chunksize=chunksize,
        **kwargs,
    )
    for df, _meta in reader:
        yield df


def _read_with_pandas(
    file_path: Path,
    encoding: str,
    usecols: list[str] | None,
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    """pandas 폴백 – 청크 읽기."""
    fmt = "xport" if file_path.suffix.lower() == ".xpt" else "sas7bdat"
    reader = pd.read_sas(
        str(file_path),
        format=fmt,
        encoding=encoding,
        chunksize=chunksize,
    )
    if hasattr(reader, "__iter__"):
        for chunk in reader:
            if usecols:
                available = [c for c in usecols if c in chunk.columns]
                chunk = chunk[available]
            yield chunk
    else:
        df = reader
        if usecols:
            available = [c for c in usecols if c in df.columns]
            df = df[available]
        yield df


def read_sas_chunks(
    file_path: str | Path,
    encoding: str = "cp949",
    usecols: list[str] | None = None,
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """SAS 파일을 청크 단위로 읽는 제너레이터.

    pyreadstat → pandas 순으로 시도합니다.
    """
    file_path = Path(file_path)
    try:
        import pyreadstat  # noqa: F401
    except ImportError:
        logger.warning("pyreadstat 미설치 – pandas.read_sas 폴백 사용")
        yield from _read_with_pandas(file_path, encoding, usecols, chunksize)
        return

    try:
        yield from _read_with_pyreadstat(file_path, encoding, usecols, chunksize)
    except Exception as e:
        logger.warning("pyreadstat 읽기 오류 (%s) – pandas.read_sas 폴백 사용", e)
        yield from _read_with_pandas(file_path, encoding, usecols, chunksize)


def read_sas_full(
    file_path: str | Path,
    encoding: str = "cp949",
    usecols: list[str] | None = None,
    chunksize: int = 100_000,
    guard=None,
) -> pd.DataFrame:
    """SAS 파일 전체를 DataFrame으로 반환.

    메모리 안전: 청크를 하나씩 병합 (전체 list 한번에 적재 방지).
    guard 전달 시 매 청크마다 RSS 체크.
    """
    from hana_app.core.memory_guard import get_guard
    _guard = get_guard(guard)

    result = None
    for chunk in read_sas_chunks(file_path, encoding, usecols, chunksize):
        _guard.check()
        if result is None:
            result = chunk
        else:
            result = pd.concat([result, chunk], ignore_index=True)
    if result is None:
        return pd.DataFrame()
    gc.collect()
    return result


def get_sas_columns(file_path: str | Path, encoding: str = "cp949") -> list[str]:
    """SAS 파일의 컬럼 목록만 빠르게 조회 (첫 행만 읽음)."""
    try:
        import pyreadstat
        read_fn = (
            pyreadstat.read_xport
            if Path(file_path).suffix.lower() == ".xpt"
            else pyreadstat.read_sas7bdat
        )
        df, meta = read_fn(str(file_path), row_limit=1, encoding=encoding)
        return list(df.columns)
    except Exception:
        try:
            df = next(read_sas_chunks(file_path, encoding, chunksize=1))
            return list(df.columns)
        except Exception:
            return []


def get_sas_row_count(file_path: str | Path, encoding: str = "cp949") -> int:
    """행 수 추정 (pyreadstat 메타 기준, 없으면 전체 읽기)."""
    try:
        import pyreadstat
        read_fn = (
            pyreadstat.read_xport
            if Path(file_path).suffix.lower() == ".xpt"
            else pyreadstat.read_sas7bdat
        )
        _df, meta = read_fn(str(file_path), row_limit=0, encoding=encoding)
        return meta.number_rows
    except Exception:
        total = 0
        for chunk in read_sas_chunks(file_path, encoding, chunksize=100_000):
            total += len(chunk)
        return total


# ─────────────────────────────────────────────────────────────────────────────
# 날짜 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(val) -> date | None:
    try:
        s = str(val).strip()
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except Exception:
        return None


def _yyyymm_range(year_from: str, month_from: str, year_to: str, month_to: str) -> set[str]:
    result = set()
    y, m = int(year_from), int(month_from)
    ye, me = int(year_to), int(month_to)
    while (y, m) <= (ye, me):
        result.add(f"{y:04d}{m:02d}")
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


# ─────────────────────────────────────────────────────────────────────────────
# SAS ETL 추출기
# ─────────────────────────────────────────────────────────────────────────────

class SASExtractor:
    """SAS 파일 → PrescriptionRecord 변환.

    HANAExtractor와 동일한 인터페이스(extract_prescriptions)를 제공합니다.
    """

    def __init__(self, sas_cfg: dict, col_cfg: dict) -> None:
        """
        sas_cfg : config["sas"]   {"folder":..., "files":{...}, "encoding":..., "chunksize":...}
        col_cfg : config["columns"]
        """
        self.folder   = Path(sas_cfg.get("folder", ""))
        self.files    = sas_cfg.get("files", {})
        self.encoding = sas_cfg.get("encoding", "cp949")
        self.chunksize = int(sas_cfg.get("chunksize", 100_000))
        self.cols     = col_cfg

    def _file_path(self, key: str) -> Path | None:
        fname = self.files.get(key, "")
        if not fname:
            return None
        p = self.folder / fname
        return p if p.exists() else None

    def _col(self, tbl: str, field: str) -> str:
        return self.cols[tbl][field]

    # ── 단일 테이블 로드 ──────────────────────────────────────────

    def _load_filtered(
        self,
        key: str,
        needed_cols: list[str],
        yyyymm_set: set[str],
        progress_cb: Callable[[str], None] | None,
        guard=None,
    ) -> pd.DataFrame:
        """SAS 파일을 청크 단위로 읽어 날짜 필터 적용 후 반환."""
        fpath = self._file_path(key)
        if fpath is None:
            if progress_cb:
                progress_cb(f"[건너뜀] {key.upper()} 파일 미지정 또는 없음")
            return pd.DataFrame()

        yyyymm_col = self.cols[key].get("yyyymm", "MDCARE_STRT_YYYYMM")

        # 실제 컬럼명 확인 후 없는 열 제거
        actual_cols = get_sas_columns(fpath, self.encoding)
        use_cols = [c for c in needed_cols if c in actual_cols]
        if yyyymm_col in actual_cols and yyyymm_col not in use_cols:
            use_cols.append(yyyymm_col)

        if progress_cb:
            progress_cb(
                f"{key.upper()} SAS 읽기: {fpath.name} "
                f"({len(use_cols)}개 컬럼, 인코딩={self.encoding})"
            )

        from hana_app.core.memory_guard import get_guard, MemoryLimitExceeded
        _guard = get_guard(guard)

        parts: list[pd.DataFrame] = []
        total_rows = 0

        for chunk in read_sas_chunks(fpath, self.encoding, use_cols, self.chunksize):
            try:
                _guard.check()
                total_rows += len(chunk)
                # YYYYMM 필터
                if yyyymm_col in chunk.columns and yyyymm_set:
                    chunk[yyyymm_col] = chunk[yyyymm_col].astype(str).str.strip().str[:6]
                    chunk = chunk[chunk[yyyymm_col].isin(yyyymm_set)]
                if not chunk.empty:
                    parts.append(chunk)
            except MemoryLimitExceeded:
                logger.warning(
                    "%s SAS 읽기 중 RAM 한도 도달 — %d행까지 처리된 결과 사용",
                    key.upper(), total_rows,
                )
                break
            except MemoryError:
                logger.warning(
                    "%s SAS 읽기 중 메모리 부족 — %d행까지 처리된 결과 사용",
                    key.upper(), total_rows,
                )
                gc.collect()
                break

        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=use_cols)
        del parts
        gc.collect()
        if progress_cb:
            progress_cb(
                f"{key.upper()} 완료: 전체 {total_rows:,}행 → "
                f"필터 후 {len(df):,}행"
            )
        return df

    # ── DataFrame → PrescriptionRecord 변환 헬퍼 ──────────────────

    @staticmethod
    def _t20_lookup(t20_idx: pd.DataFrame, bill_no: str, col: str) -> str:
        """T20 DataFrame index에서 bill_no로 컬럼값 조회 (없으면 빈 문자열)."""
        if bill_no in t20_idx.index:
            val = t20_idx.at[bill_no, col]
            return str(val or "")
        return ""

    def _t30_to_records(
        self, t30: pd.DataFrame, t20_index: pd.DataFrame,
        t40_index: dict[str, str] | None = None,
    ) -> list[PrescriptionRecord]:
        c30 = self.cols["t30"]
        c20 = self.cols["t20"]
        _lk = self._t20_lookup
        records: list[PrescriptionRecord] = []
        for row in t30.itertuples(index=False):
            bill_no   = str(getattr(row, c30["bill_no"], "") or "")
            start_dt  = _parse_date(str(getattr(row, c30["start_date"], "") or ""))
            if not start_dt:
                continue
            total_days = int(getattr(row, c30["total_days"], 1) or 1)
            wk     = str(getattr(row, c30["drug_code"], "") or "").strip()
            wk_alt = str(getattr(row, c30["drug_code_alt"], "") or "").strip()
            drug_code = wk if wk else wk_alt
            if not drug_code:
                continue
            records.append(PrescriptionRecord(
                patient_id       = str(getattr(row, c30["patient_id"], "") or ""),
                institution_id   = _lk(t20_index, bill_no, c20["institution_id"]),
                bill_no          = bill_no,
                wk_compn_cd      = drug_code,
                edi_code         = str(getattr(row, c30["edi_code"], "") or "") or None,
                efmdc_clsf_no    = str(getattr(row, c30["efmdc"], "") or "") or None,
                start_date       = start_dt,
                end_date         = start_dt + timedelta(days=max(total_days - 1, 0)),
                total_days       = total_days,
                dose_once        = float(getattr(row, c30["dose_once"], 1.0) or 1.0),
                dose_freq        = int(getattr(row, c30["dose_freq"], 1) or 1),
                sex              = _lk(t20_index, bill_no, c20["sex"]) or None,
                age_id           = _lk(t20_index, bill_no, c20["age_id"]) or None,
                institution_type = _lk(t20_index, bill_no, c20["institution_type"]) or None,
                sick_code        = t40_index.get(str(bill_no)) if t40_index else None,
                source           = "T30",
            ))
        return records

    def _t60_to_records(
        self, t60: pd.DataFrame, t20_index: pd.DataFrame,
        t40_index: dict[str, str] | None = None,
    ) -> list[PrescriptionRecord]:
        c60 = self.cols["t60"]
        c20 = self.cols["t20"]
        _lk = self._t20_lookup
        records: list[PrescriptionRecord] = []
        for row in t60.itertuples(index=False):
            bill_no   = str(getattr(row, c60["bill_no"], "") or "")
            start_dt  = _parse_date(str(getattr(row, c60["start_date"], "") or ""))
            if not start_dt:
                continue
            total_days = int(getattr(row, c60["total_days"], 1) or 1)
            gnl    = str(getattr(row, c60["drug_code"], "") or "").strip()
            wk_alt = str(getattr(row, c60["drug_code_alt"], "") or "").strip()
            drug_code = gnl if gnl else wk_alt
            if not drug_code:
                continue
            records.append(PrescriptionRecord(
                patient_id       = str(getattr(row, c60["patient_id"], "") or ""),
                institution_id   = str(getattr(row, c60["institution_id"], "") or ""),
                bill_no          = bill_no,
                wk_compn_cd      = drug_code,
                gnl_nm_cd        = gnl or None,
                edi_code         = str(getattr(row, c60["edi_code"], "") or "") or None,
                start_date       = start_dt,
                end_date         = start_dt + timedelta(days=max(total_days - 1, 0)),
                total_days       = total_days,
                dose_once        = float(getattr(row, c60["dose_once"], 1.0) or 1.0),
                dose_freq        = int(getattr(row, c60["dose_freq"], 1) or 1),
                sick_code        = str(getattr(row, c60["sick_code"], "") or "").strip() or (t40_index.get(str(bill_no)) if t40_index else None),
                sex              = _lk(t20_index, bill_no, c20["sex"]) or None,
                age_id           = _lk(t20_index, bill_no, c20["age_id"]) or None,
                institution_type = _lk(t20_index, bill_no, c20["institution_type"]) or None,
                source           = "T60",
            ))
        return records

    # ── T40 ICD-10 질환 필터 → 환자 ID 추출 ─────────────────────

    def fetch_patients_by_icd10(
        self,
        icd10_prefixes: list[str],
        yyyymm_set: "set[str] | list[str] | None" = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> list[str]:
        """T40 SAS 파일에서 ICD-10 상병코드에 해당하는 환자 ID 반환.

        Parameters
        ----------
        icd10_prefixes : list[str]
            ICD-10 접두사 목록. 예: ["E11", "I10"]
        yyyymm_set : set/list[str], optional
            YYYYMM 필터. None이면 전체 파일을 읽습니다.

        Returns
        -------
        list[str] : 중복 제거된 INDI_DSCM_NO 목록.
        """
        if not icd10_prefixes:
            return []

        fpath = self._file_path("t40")
        if fpath is None:
            if progress_cb:
                progress_cb("[건너뜀] T40 SAS 파일 미지정 또는 없음")
            return []

        c = self.cols["t40"]
        pid_col  = c["patient_id"]
        sick_col = c["sick_code"]
        ym_col   = c.get("yyyymm", "MDCARE_STRT_YYYYMM")

        _ym_set: set[str] | None = set(yyyymm_set) if yyyymm_set is not None else None
        # prefix 대문자 정규화
        prefixes = [p.strip().upper() for p in icd10_prefixes]
        use_cols = [pid_col, sick_col]
        if _ym_set is not None:
            use_cols.append(ym_col)

        codes_str = ", ".join(prefixes)
        if progress_cb:
            progress_cb(f"T40 SAS ICD-10 필터: {codes_str}")

        pid_set: set[str] = set()
        for chunk in read_sas_chunks(fpath, self.encoding, use_cols, self.chunksize):
            if _ym_set is not None and ym_col in chunk.columns:
                chunk = chunk[chunk[ym_col].astype(str).str.strip().isin(_ym_set)]
            if chunk.empty:
                continue
            # ICD-10 prefix 매칭: startswith any prefix
            sick_series = chunk[sick_col].astype(str).str.strip().str.upper()
            mask = sick_series.apply(
                lambda x: any(x.startswith(pf) for pf in prefixes)
            )
            matched = chunk.loc[mask, pid_col].astype(str).str.strip()
            pid_set.update(matched.unique())

        result = [p for p in pid_set if p]
        if progress_cb:
            progress_cb(f"T40 SAS ICD-10 완료: {codes_str} → {len(result):,}명")
        return result

    # ── 자격 DB (Eligibility) → 전체 인구통계 (층화 샘플링용) ────

    def fetch_eligibility_for_sampling(
        self,
        std_year: str,
        addr_digits: int = 5,
        sample_size: int | None = None,
        seed: int = 42,
        progress_cb: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        """자격DB SAS 파일에서 연도별 인구통계 조회 (사전 층화 샘플링용).

        patient_ids 필터 없이 STD_YYYY = std_year 에 해당하는 레코드를 반환합니다.
        중복 INDI_DSCM_NO는 마지막 레코드 유지.

        Parameters
        ----------
        sample_size : int, optional
            지정 시 2-pass 스트리밍 저장소 샘플링을 사용합니다.
            전체 파일을 메모리에 올리지 않고 ``sample_size`` 행만 보유.
            None 이면 전체 데이터를 메모리에 로드합니다 (주의: 대용량).
        seed : int
            저장소 샘플링에 사용할 시드.

        Returns
        -------
        pd.DataFrame : INDI_DSCM_NO, BYEAR, SEX_TYPE, RVSN_ADDR_CD(잘린 값) 컬럼.
        """
        import random as _random
        from collections import defaultdict as _defaultdict

        c = self.cols.get("eligibility", {})
        pid_col    = c.get("patient_id",   "INDI_DSCM_NO")
        byear_col  = c.get("byear",        "BYEAR")
        sex_col    = c.get("sex_type",     "SEX_TYPE")
        year_col   = c.get("std_year",     "STD_YYYY")
        addr_col   = c.get("rvsn_addr_cd", "RVSN_ADDR_CD")

        fpath = self._file_path("eligibility")
        if fpath is None:
            if progress_cb:
                progress_cb("[건너뜀] 자격DB SAS 파일 미지정 또는 없음")
            return pd.DataFrame()

        use_cols = [pid_col, byear_col, sex_col, addr_col, year_col]

        if sample_size is not None and sample_size > 0:
            # ── 2-pass 스트리밍 저장소 샘플링 ────────────────────────────────
            # Pass 1: 층별 행 수 카운트 (row 저장 없음 → 메모리 최소)
            if progress_cb:
                progress_cb(
                    f"자격DB SAS 1차 스캔 (층별 카운트): {fpath.name} (STD_YYYY={std_year})"
                )
            strata_row_count: dict[str, int] = _defaultdict(int)
            ref_year_int = int(str(std_year).strip())
            for chunk in read_sas_chunks(fpath, self.encoding, use_cols, self.chunksize):
                if year_col in chunk.columns:
                    chunk = chunk[chunk[year_col].astype(str).str.strip() == str(std_year)]
                if chunk.empty:
                    continue
                for row in chunk.itertuples(index=False):
                    sex_v    = str(getattr(row, sex_col,   "") or "").strip()
                    byear_raw = getattr(row, byear_col, None)
                    age_band = byear_to_age_band(byear_raw, ref_year_int)
                    addr_v   = str(getattr(row, addr_col,  "") or "")[:addr_digits]
                    strata_row_count[f"{sex_v}|{age_band}|{addr_v}"] += 1

            total_rows = sum(strata_row_count.values())
            if total_rows == 0:
                return pd.DataFrame(columns=[pid_col, byear_col, sex_col, addr_col])

            # floor + 최대잉여법으로 층별 할당
            alloc: dict[str, int] = {}
            frac:  dict[str, float] = {}
            for key, cnt in strata_row_count.items():
                exact = cnt / total_rows * sample_size
                floor_v = min(int(exact), cnt)
                alloc[key] = floor_v
                frac[key]  = exact - floor_v
            remaining = sample_size - sum(alloc.values())
            for key in sorted(frac, key=frac.__getitem__, reverse=True):
                if remaining <= 0:
                    break
                if alloc[key] < strata_row_count[key]:
                    alloc[key] += 1
                    remaining -= 1

            # Pass 2: 층별 저장소(reservoir) 샘플링
            if progress_cb:
                progress_cb(
                    f"자격DB SAS 2차 스캔 (저장소 샘플링 {sample_size:,}명): {fpath.name}"
                )
            rng_local = _random.Random(int(seed))
            reservoirs: dict[str, list] = {k: [] for k in alloc if alloc[k] > 0}
            counts_seen: dict[str, int] = _defaultdict(int)

            for chunk in read_sas_chunks(fpath, self.encoding, use_cols, self.chunksize):
                if year_col in chunk.columns:
                    chunk = chunk[chunk[year_col].astype(str).str.strip() == str(std_year)]
                if chunk.empty:
                    continue
                for row in chunk.itertuples(index=False):
                    pid_v    = str(getattr(row, pid_col,   "") or "").strip()
                    sex_v    = str(getattr(row, sex_col,   "") or "").strip()
                    byear_v  = str(getattr(row, byear_col, "") or "").strip()
                    age_band = byear_to_age_band(byear_v, ref_year_int)
                    addr_v   = str(getattr(row, addr_col,  "") or "")[:addr_digits]
                    if not pid_v:
                        continue
                    key = f"{sex_v}|{age_band}|{addr_v}"
                    if key not in reservoirs:
                        continue
                    k = alloc[key]
                    counts_seen[key] += 1
                    n = counts_seen[key]
                    entry = {
                        pid_col: pid_v, byear_col: byear_v,
                        sex_col: sex_v, addr_col: addr_v,
                    }
                    if len(reservoirs[key]) < k:
                        reservoirs[key].append(entry)
                    else:
                        j = rng_local.randint(0, n - 1)
                        if j < k:
                            reservoirs[key][j] = entry

            rows_sampled = [entry for res in reservoirs.values() for entry in res]
            if not rows_sampled:
                return pd.DataFrame(columns=[pid_col, byear_col, sex_col, addr_col])

            df = pd.DataFrame(rows_sampled)
            df = df.drop_duplicates(subset=[pid_col], keep="last").reset_index(drop=True)
            if progress_cb:
                progress_cb(f"자격DB SAS 샘플링 완료: {len(df):,}명")
            return df

        # ── 전체 로드 경로 (sample_size 미지정 / 소규모 데이터) ──────────────
        if progress_cb:
            progress_cb(
                f"자격DB SAS 전체 읽기 (층화 샘플링용): {fpath.name} (STD_YYYY={std_year})"
            )

        rows_acc: list[pd.DataFrame] = []
        for chunk in read_sas_chunks(fpath, self.encoding, use_cols, self.chunksize):
            if year_col in chunk.columns:
                chunk = chunk[chunk[year_col].astype(str).str.strip() == str(std_year)]
            if chunk.empty:
                continue
            chunk = chunk[[pid_col, byear_col, sex_col, addr_col]].copy()
            chunk[addr_col] = chunk[addr_col].astype(str).str[:addr_digits]
            rows_acc.append(chunk)

        if not rows_acc:
            return pd.DataFrame(columns=[pid_col, byear_col, sex_col, addr_col])

        df = pd.concat(rows_acc, ignore_index=True)
        # pid 오름차순 정렬 후 dedup → 결과가 항상 결정적
        df = df.sort_values(pid_col, kind="mergesort").reset_index(drop=True)
        df = df.drop_duplicates(subset=[pid_col], keep="last").reset_index(drop=True)

        if progress_cb:
            progress_cb(f"자격DB SAS 전체 완료: {len(df):,}명")
        return df

    # ── 자격 DB (Eligibility) → 인구통계 ────────────────────────

    def fetch_eligibility_demographics(
        self,
        patient_ids: list[str],
        std_year: str,
        addr_digits: int = 5,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, dict]:
        """자격 DB SAS 파일에서 BYEAR·SEX_TYPE·RVSN_ADDR_CD 조회.

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
        c = self.cols.get("eligibility", {})
        pid_col    = c.get("patient_id",   "INDI_DSCM_NO")
        byear_col  = c.get("byear",        "BYEAR")
        sex_col    = c.get("sex_type",     "SEX_TYPE")
        year_col   = c.get("std_year",     "STD_YYYY")
        addr_col   = c.get("rvsn_addr_cd", "RVSN_ADDR_CD")

        fpath = self._file_path("eligibility")
        if fpath is None:
            if progress_cb:
                progress_cb("[건너뜀] 자격DB SAS 파일 미지정 또는 없음")
            return {}

        if progress_cb:
            progress_cb(
                f"자격DB SAS 읽기: {fpath.name} "
                f"(STD_YYYY={std_year}, 대상 {len(patient_ids):,}명)"
            )

        pid_set = set(patient_ids)
        use_cols = [pid_col, byear_col, sex_col, addr_col, year_col]
        # 청크별로 누적 후 최종 drop_duplicates(keep='last')
        rows_acc: list[dict] = []

        for chunk in read_sas_chunks(fpath, self.encoding, use_cols, self.chunksize):
            # STD_YYYY 필터
            year_vals = chunk.get(year_col) if hasattr(chunk, "get") else None
            if year_col in chunk.columns:
                chunk = chunk[chunk[year_col].astype(str).str.strip() == str(std_year)]
            # 대상 환자 필터
            chunk = chunk[chunk[pid_col].astype(str).str.strip().isin(pid_set)]
            for row in chunk.itertuples(index=False):
                pid  = str(getattr(row, pid_col, "")).strip()
                addr = str(getattr(row, addr_col, "") or "").strip()
                rows_acc.append({
                    pid_col:   pid,
                    byear_col: getattr(row, byear_col, None),
                    sex_col:   str(getattr(row, sex_col, "") or "").strip() or None,
                    addr_col:  addr[:addr_digits] if addr else None,
                })

        # 중복 INDI_DSCM_NO → pid 오름차순 정렬 후 마지막 레코드 유지 (결정적)
        rows_acc.sort(key=lambda r: r.get(pid_col, "") or "")
        result: dict[str, dict] = {}
        for r in rows_acc:
            pid = r[pid_col]
            if pid:
                byear = r[byear_col]
                result[pid] = {
                    "byear":    int(float(byear)) if byear is not None else None,
                    "sex_type": r[sex_col],
                    "addr_cd":  r[addr_col],
                }

        if progress_cb:
            progress_cb(f"자격DB 완료: {len(result):,}명 인구통계 매핑")
        return result

    # ── 자격 DB (Eligibility) → 환자 나이 (구버전, 호환성 유지) ─────

    def fetch_eligibility_ages(
        self,
        patient_ids: list[str] | None = None,
        reference_year: int | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, int]:
        """자격 DB SAS 파일에서 BYEAR 조회 → {patient_id: age} 딕셔너리.

        나이 = reference_year - BYEAR.

        .. deprecated::
            fetch_eligibility_demographics 사용 권장.
            STD_YYYY 필터 없이 전체 파일을 읽으므로 메모리 사용량이 큽니다.
        """
        from datetime import date as _date

        ref_year = reference_year or _date.today().year
        c = self.cols.get("eligibility", {})
        pid_col = c.get("patient_id", "INDI_DSCM_NO")
        byear_col = c.get("byear", "BYEAR")

        # SAS 파일 경로 확인
        fpath = self._file_path("eligibility")
        if fpath is None:
            if progress_cb:
                progress_cb("[건너뜀] 자격DB SAS 파일 미지정 또는 없음")
            return {}

        if progress_cb:
            progress_cb(f"자격DB SAS 읽기: {fpath.name} (BYEAR → 나이, 기준년도={ref_year})")

        age_map: dict[str, int] = {}
        use_cols = [pid_col, byear_col]

        for chunk in read_sas_chunks(fpath, self.encoding, use_cols, self.chunksize):
            for row in chunk.itertuples(index=False):
                pid = str(getattr(row, pid_col, "")).strip()
                byear = getattr(row, byear_col, None)
                if pid and byear is not None:
                    try:
                        age_map[pid] = ref_year - int(float(byear))
                    except (ValueError, TypeError):
                        pass

        # patient_ids 필터 (지정된 경우)
        if patient_ids is not None:
            pid_set = set(patient_ids)
            age_map = {k: v for k, v in age_map.items() if k in pid_set}

        if progress_cb:
            progress_cb(f"자격DB 완료: {len(age_map):,}명 나이 매핑")
        return age_map

    # ── 통합 추출 (HANAExtractor와 동일한 시그니처) ──────────────

    def extract_prescriptions(
        self,
        year_from: str,
        month_from: str,
        year_to: str,
        month_to: str,
        window_days: int = 90,
        poly_threshold: int = 5,
        buffer_days: int = 0,
        buffer_after_days: int = 0,
        progress_cb: Callable[[str], None] | None = None,
        patient_ids: list[str] | None = None,
    ) -> tuple[list[PrescriptionRecord], dict]:

        _pid_set: set[str] | None = set(patient_ids) if patient_ids is not None else None

        analysis_start = f"{year_from}{month_from}"
        analysis_end = f"{year_to}{month_to}"

        # 시작 전 버퍼
        buffer_before_months = max(1, (buffer_days + 29) // 30) if buffer_days > 0 else 0
        query_start = _shift_yyyymm(analysis_start, -buffer_before_months) if buffer_before_months else analysis_start

        # 종료 후 버퍼
        buffer_after_months = max(1, (buffer_after_days + 29) // 30) if buffer_after_days > 0 else 0
        query_end = _shift_yyyymm(analysis_end, buffer_after_months) if buffer_after_months else analysis_end

        yyyymm_set = _yyyymm_range(query_start[:4], query_start[4:], query_end[:4], query_end[4:])
        if progress_cb:
            buf_parts = []
            if buffer_before_months:
                buf_parts.append(f"시작 전 {buffer_before_months}개월")
            if buffer_after_months:
                buf_parts.append(f"종료 후 {buffer_after_months}개월")
            buf_label = " + ".join(buf_parts) if buf_parts else "없음"
            progress_cb(
                f"SAS 파일 추출 시작 "
                f"[분석: {analysis_start}~{analysis_end}]  "
                f"(쿼리: {query_start}~{query_end}, 버퍼: {buf_label})  "
                f"대상 YYYYMM {len(yyyymm_set)}개월"
            )

        c20 = self.cols["t20"]
        c30 = self.cols["t30"]
        c40 = self.cols["t40"]
        c60 = self.cols["t60"]

        # ── T20 ──────────────────────────────────────────────────
        t20_cols = [
            c20["bill_no"], c20["patient_id"], c20["institution_id"],
            c20["start_date"], c20["sex"], c20["age_id"], c20["institution_type"],
        ]
        t20 = self._load_filtered("t20", t20_cols, yyyymm_set, progress_cb)
        if _pid_set is not None and not t20.empty and c20["patient_id"] in t20.columns:
            t20 = t20[t20[c20["patient_id"]].isin(_pid_set)]
        stats_t20 = len(t20)
        t20_index = pd.DataFrame()
        if not t20.empty and c20["bill_no"] in t20.columns:
            t20_index = t20.set_index(c20["bill_no"])
        del t20
        gc.collect()

        # ── T40 (상병) ────────────────────────────────────────────
        t40_cols = [c40["bill_no"], c40["sick_code"], c40["sick_type"]]
        t40 = self._load_filtered("t40", t40_cols, yyyymm_set, progress_cb)
        stats_t40 = len(t40)
        from hana_app.core.hana_etl import build_t40_index
        t40_idx = build_t40_index(t40, c40["bill_no"], c40["sick_code"])
        del t40
        gc.collect()

        # ── T30 (원내) ────────────────────────────────────────────
        t30_cols = [
            c30["bill_no"], c30["patient_id"], c30["start_date"],
            c30["drug_code"], c30["drug_code_alt"], c30["edi_code"],
            c30["efmdc"], c30["dose_once"], c30["dose_freq"], c30["total_days"],
        ]
        t30 = self._load_filtered("t30", t30_cols, yyyymm_set, progress_cb)
        if _pid_set is not None and not t30.empty and c30["patient_id"] in t30.columns:
            t30 = t30[t30[c30["patient_id"]].isin(_pid_set)]
        stats_t30 = len(t30)

        if progress_cb:
            progress_cb("PrescriptionRecord 변환 중 (T30)...")
        records = self._t30_to_records(t30, t20_index, t40_idx)
        del t30
        gc.collect()

        # ── T60 (원외) ────────────────────────────────────────────
        t60_cols = [
            c60["bill_no"], c60["patient_id"], c60["start_date"],
            c60["drug_code"], c60["drug_code_alt"], c60["edi_code"],
            c60["dose_once"], c60["dose_freq"], c60["total_days"],
            c60["sick_code"], c60["institution_id"],
        ]
        t60 = self._load_filtered("t60", t60_cols, yyyymm_set, progress_cb)
        if _pid_set is not None and not t60.empty and c60["patient_id"] in t60.columns:
            t60 = t60[t60[c60["patient_id"]].isin(_pid_set)]
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
            "total_records":   len(records),
            "unique_patients": len({r.patient_id for r in records}),
            "period": f"{analysis_start}~{analysis_end}",
            "query_period": f"{query_start}~{query_end}",
            "buffer_before_months": buffer_before_months,
            "buffer_after_months": buffer_after_months,
            "source": "SAS",
        }
        if progress_cb:
            progress_cb(
                f"SAS 추출 완료 – 총 {stats['total_records']:,}건 / "
                f"환자 {stats['unique_patients']:,}명"
            )
        return records, stats

    # ── 청크 추출 → Parquet 저장 (HANAExtractor와 동일 인터페이스) ─

    def extract_prescriptions_chunked(
        self,
        year_from: str,
        month_from: str,
        year_to: str,
        month_to: str,
        save_dir: Path | str | None = None,
        chunk_months: int = 1,
        chunk_unit: str = "month",
        chunk_days: int = 1,
        window_days: int = 90,
        poly_threshold: int = 5,
        buffer_days: int = 0,
        buffer_after_days: int = 0,
        memory_limit_mb: int = 0,
        progress_cb: Callable[[str], None] | None = None,
        patient_ids: list[str] | None = None,
    ) -> tuple[list[Path], dict]:
        """
        SAS 파일 → 청크 Parquet 저장 (메모리 효율화).

        SAS 파일은 1회만 읽고, 월/일별 청크로 분배하여 Parquet 저장.
        HANAExtractor.extract_prescriptions_chunked 와 동일한 반환 형식.
        chunk_unit: "month" 또는 "day". SAS는 파일 기반이므로 날짜 컬럼 필터.
        """
        from hana_app.core.hana_etl import records_to_df

        if save_dir is None:
            save_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        analysis_start = f"{year_from}{month_from}"
        analysis_end = f"{year_to}{month_to}"

        # 버퍼 계산
        buffer_before_months = max(1, (buffer_days + 29) // 30) if buffer_days > 0 else 0
        query_start = _shift_yyyymm(analysis_start, -buffer_before_months) if buffer_before_months else analysis_start
        buffer_after_months = max(1, (buffer_after_days + 29) // 30) if buffer_after_days > 0 else 0
        query_end = _shift_yyyymm(analysis_end, buffer_after_months) if buffer_after_months else analysis_end

        full_yyyymm_set = _yyyymm_range(
            query_start[:4], query_start[4:],
            query_end[:4], query_end[4:],
        )
        yyyymm_sorted = sorted(full_yyyymm_set)
        chunks = [
            yyyymm_sorted[i:i + chunk_months]
            for i in range(0, len(yyyymm_sorted), chunk_months)
        ]

        if progress_cb:
            buf_parts = []
            if buffer_before_months:
                buf_parts.append(f"시작 전 {buffer_before_months}개월")
            if buffer_after_months:
                buf_parts.append(f"종료 후 {buffer_after_months}개월")
            buf_label = " + ".join(buf_parts) if buf_parts else "없음"
            progress_cb(
                f"SAS 청크 추출 시작 "
                f"[분석: {analysis_start}~{analysis_end}]  "
                f"(쿼리: {query_start}~{query_end}, 버퍼: {buf_label})  "
                f"대상 YYYYMM {len(yyyymm_sorted)}개월"
            )

        _pid_set: set[str] | None = set(patient_ids) if patient_ids is not None else None

        c20 = self.cols["t20"]
        c30 = self.cols["t30"]
        c40 = self.cols["t40"]
        c60 = self.cols["t60"]

        # ── T20: 전체 1회 로드 → 인덱스 (조회용) ────────────────
        t20_cols = [
            c20["bill_no"], c20["patient_id"], c20["institution_id"],
            c20["start_date"], c20["sex"], c20["age_id"],
            c20["institution_type"],
        ]
        t20 = self._load_filtered("t20", t20_cols, full_yyyymm_set, progress_cb)
        if _pid_set is not None and not t20.empty and c20["patient_id"] in t20.columns:
            t20 = t20[t20[c20["patient_id"]].isin(_pid_set)]
        t20_index = pd.DataFrame()
        if not t20.empty and c20["bill_no"] in t20.columns:
            t20_index = t20.set_index(c20["bill_no"])
            # 메모리 최적화: 반복값이 많은 컬럼을 category로 변환 (50-80% 절감)
            for _col in t20_index.columns:
                if t20_index[_col].dtype == 'object':
                    t20_index[_col] = t20_index[_col].astype('category')
        stats_t20 = len(t20)
        del t20
        gc.collect()
        if progress_cb and not t20_index.empty:
            _t20_mb = t20_index.memory_usage(deep=True).sum() / 1024 / 1024
            progress_cb(f"T20 인덱스: {len(t20_index):,}건 / {_t20_mb:.0f} MB (category 최적화 적용)")

        # ── T30 / T60: SAS → Parquet 캐시 (1회 읽기) ──────────────
        t30_cols = [
            c30["bill_no"], c30["patient_id"], c30["start_date"],
            c30["drug_code"], c30["drug_code_alt"], c30["edi_code"],
            c30["efmdc"], c30["dose_once"], c30["dose_freq"],
            c30["total_days"],
        ]
        t60_cols = [
            c60["bill_no"], c60["patient_id"], c60["start_date"],
            c60["drug_code"], c60["drug_code_alt"], c60["edi_code"],
            c60["dose_once"], c60["dose_freq"], c60["total_days"],
            c60["sick_code"], c60["institution_id"],
        ]

        # SAS 파일을 1회만 읽어 Parquet 캐시 저장 (N번 SAS 스캔 방지)
        import tempfile, shutil
        _cache_dir = Path(tempfile.mkdtemp(prefix="sas_cache_"))
        _ym_col_t30 = self.cols["t30"].get("yyyymm", "MDCARE_STRT_YYYYMM")
        _ym_col_t60 = self.cols["t60"].get("yyyymm", "MDCARE_STRT_YYYYMM")

        if progress_cb:
            progress_cb("T30/T40/T60 SAS → Parquet 캐시 변환 (1회 읽기)...")

        _t30_full = self._load_filtered("t30", t30_cols, full_yyyymm_set, progress_cb)
        if _pid_set is not None and not _t30_full.empty and c30["patient_id"] in _t30_full.columns:
            _t30_full = _t30_full[_t30_full[c30["patient_id"]].isin(_pid_set)]
        _t30_cache = None
        _stats_t30 = len(_t30_full)
        if not _t30_full.empty:
            _t30_cache = _cache_dir / "t30_cache.parquet"
            _t30_full.to_parquet(_t30_cache, index=False)
        del _t30_full
        gc.collect()

        # ── T40 (상병): SAS → Parquet 캐시 ──────────────────────
        t40_cols = [c40["bill_no"], c40["sick_code"], c40["sick_type"]]
        _t40_full = self._load_filtered("t40", t40_cols, full_yyyymm_set, progress_cb)
        _t40_cache = None
        _stats_t40 = len(_t40_full)
        if not _t40_full.empty:
            _t40_cache = _cache_dir / "t40_cache.parquet"
            _t40_full.to_parquet(_t40_cache, index=False)
        del _t40_full
        gc.collect()

        _t60_full = self._load_filtered("t60", t60_cols, full_yyyymm_set, progress_cb)
        if _pid_set is not None and not _t60_full.empty and c60["patient_id"] in _t60_full.columns:
            _t60_full = _t60_full[_t60_full[c60["patient_id"]].isin(_pid_set)]
        _t60_cache = None
        _stats_t60 = len(_t60_full)
        if not _t60_full.empty:
            _t60_cache = _cache_dir / "t60_cache.parquet"
            _t60_full.to_parquet(_t60_cache, index=False)
        del _t60_full
        gc.collect()

        if progress_cb:
            progress_cb("Parquet 캐시 완료 — 청크별 고속 필터 시작")

        parquet_paths: list[Path] = []
        stats: dict = {
            "t20_rows": stats_t20,
            "t30_rows": _stats_t30,
            "t40_rows": _stats_t40,
            "t60_rows": _stats_t60,
            "total_records": 0,
            "chunks": len(chunks),
            "period": f"{analysis_start}~{analysis_end}",
            "query_period": f"{query_start}~{query_end}",
            "buffer_before_months": buffer_before_months,
            "buffer_after_months": buffer_after_months,
            "source": "SAS",
        }

        # ── 월 청크별 처리 (Parquet 캐시에서 필터) ────────────────
        for ci, chunk_yms in enumerate(chunks):
            chunk_set = set(chunk_yms)
            label = (
                chunk_yms[0]
                if len(chunk_yms) == 1
                else f"{chunk_yms[0]}_{chunk_yms[-1]}"
            )
            if progress_cb:
                progress_cb(
                    f"[{ci+1}/{len(chunks)}] 청크 처리: "
                    f"{chunk_yms[0]}~{chunk_yms[-1]}"
                )

            # Parquet 캐시에서 YYYYMM 필터 (PyArrow predicate pushdown)
            # 전체 캐시 로드 대신 필요한 YYYYMM 행만 읽어 메모리 피크 방지
            _chunk_list = list(chunk_set)
            if _t30_cache is not None:
                try:
                    t30_c = pd.read_parquet(
                        _t30_cache,
                        filters=[(_ym_col_t30, "in", _chunk_list)],
                    )
                except Exception:
                    # PyArrow filters 미지원 시 폴백 (컬럼 타입 불일치 등)
                    t30_c = pd.read_parquet(_t30_cache)
                    if _ym_col_t30 in t30_c.columns:
                        t30_c = t30_c[t30_c[_ym_col_t30].isin(chunk_set)]
            else:
                t30_c = pd.DataFrame()

            # T40 캐시에서 YYYYMM 필터 → t40_index 빌드
            _ym_col_t40 = self.cols["t40"].get("yyyymm", "MDCARE_STRT_YYYYMM")
            t40_idx: dict[str, str] = {}
            if _t40_cache is not None:
                try:
                    t40_c = pd.read_parquet(
                        _t40_cache,
                        filters=[(_ym_col_t40, "in", _chunk_list)],
                    )
                except Exception:
                    t40_c = pd.read_parquet(_t40_cache)
                    if _ym_col_t40 in t40_c.columns:
                        t40_c = t40_c[t40_c[_ym_col_t40].isin(chunk_set)]
                from hana_app.core.hana_etl import build_t40_index
                t40_idx = build_t40_index(t40_c, c40["bill_no"], c40["sick_code"])
                del t40_c
                gc.collect()

            if _t60_cache is not None:
                try:
                    t60_c = pd.read_parquet(
                        _t60_cache,
                        filters=[(_ym_col_t60, "in", _chunk_list)],
                    )
                except Exception:
                    t60_c = pd.read_parquet(_t60_cache)
                    if _ym_col_t60 in t60_c.columns:
                        t60_c = t60_c[t60_c[_ym_col_t60].isin(chunk_set)]
            else:
                t60_c = pd.DataFrame()

            chunk_records = self._t30_to_records(t30_c, t20_index, t40_idx)
            del t30_c
            chunk_records += self._t60_to_records(t60_c, t20_index, t40_idx)
            del t60_c, t40_idx
            gc.collect()

            stats["total_records"] += len(chunk_records)

            if chunk_records:
                parquet_path = save_dir / f"records_{label}.parquet"
                records_to_df(chunk_records).to_parquet(
                    parquet_path, index=False,
                )
                parquet_paths.append(parquet_path)
                if progress_cb:
                    progress_cb(
                        f"  저장: {parquet_path.name} "
                        f"({len(chunk_records):,}건)"
                    )
            del chunk_records
            gc.collect()

        # Parquet 캐시 정리
        shutil.rmtree(_cache_dir, ignore_errors=True)

        # 고유 환자 수 (DuckDB 사용 시 메모리 효율적)
        from hana_app.core.hana_etl import _count_unique_patients
        stats["unique_patients"] = _count_unique_patients(
            parquet_paths, memory_limit_mb=memory_limit_mb,
        )

        if progress_cb:
            progress_cb(
                f"SAS 청크 추출 완료 – 총 {stats['total_records']:,}건 / "
                f"환자 {stats['unique_patients']:,}명 / "
                f"파일 {len(parquet_paths)}개"
            )
        return parquet_paths, stats
