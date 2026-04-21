"""
ML 학습 / 평가 파이프라인

HANA에서 추출한 PrescriptionRecord → PatientFeatures → ML 모델 학습
"""
from __future__ import annotations

import contextlib
import json
import logging
import pickle
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Callable

# Windows OpenMP DLL 충돌 방지 ─────────────────────────────────────────────
# numpy의 OpenBLAS가 OpenMP를 선점하면 torch의 libiomp5md.dll 로드 실패.
# torch를 numpy보다 먼저 import하여 Intel OpenMP가 우선 초기화되도록 함.
if sys.platform == 'win32':
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        pass  # torch 미설치 또는 DLL 초기화 실패 시 Phase 1~2만 사용 (정상)
# ──────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.etl.models import (
    PatientFeatures,
    PrescriptionRecord,
)
from scripts.etl.prescription_aggregator import aggregate_patient_features
from scripts.etl.overlap_calculator import calculate_overlaps_for_patient

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS_DIR = Path(__file__).parent.parent / "models"

FEATURE_COLS = [
    "drug_count",
    "drug_count_7d",
    "institution_count",
    "ddi_contraindicated",
    "ddi_major",
    "ddi_moderate",
    "ddi_minor",
    "triple_whammy",
    "qt_risk_count",
    "dup_same_ingredient",
    "dup_atc5",
    "dup_atc4",
    "dup_atc3",
    "dup_efmdc",
    "has_high_risk_drug",
    "has_renal_risk_drug",
    "has_hepatic_risk_drug",
    "cyp_risk_score",
    "cyp_max_enzyme_risk",
    "cyp_high_risk_pairs",
    "age",
    "sex_m",
]

RISK_LABEL_MAP = {
    "Red": 3,
    "Yellow": 2,
    "Green": 1,
    "Normal": 0,
}

RISK_COLOR_MAP = {
    "Red": "🔴",
    "Yellow": "🟡",
    "Green": "🟢",
    "Normal": "⚪",
}

GPU_MEMORY_FRACTION: float = 0.70  # GPU 메모리 최대 사용 비율 (기본 70%)


# ─────────────────────────────────────────────────────────────────────────────
# GPU 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _has_cuda() -> bool:
    """CUDA GPU 가용 여부를 nvidia-smi 로 확인."""
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def _get_gpu_total_mb() -> int:
    """GPU 총 메모리(MB) 반환. 실패 시 0."""
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return int(r.stdout.strip().split("\n")[0].strip())
    except Exception:
        pass
    return 0


class _GpuMemoryGuard:
    """
    GPU 메모리 사용량을 fraction 이하로 제한하는 컨텍스트 매니저.

    시도 순서:
      1. torch.cuda.set_per_process_memory_fraction  (torch 설치 시)
      2. cupy MemoryPool.set_limit                   (cupy 설치 시)
      3. ctypes cudaMalloc 선점 예약                  (CUDA DLL 직접 호출)
      4. 제한 불가 – GPU 사용은 허용하되 경고 로그
    """

    def __init__(self, fraction: float = GPU_MEMORY_FRACTION):
        self.fraction = max(0.1, min(1.0, fraction))
        self._method = ""
        self._cuda_lib = None
        self._cuda_ptr = None

    # ── 진입 ──────────────────────────────────────────────────────────────
    def __enter__(self) -> "_GpuMemoryGuard":
        if not _has_cuda():
            return self

        # 1) PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(self.fraction)
                name = torch.cuda.get_device_name(0)
                total = torch.cuda.get_device_properties(0).total_memory
                self._method = (
                    f"torch | {name} | "
                    f"상한 {self.fraction:.0%} ({int(total * self.fraction)//1024//1024} MB)"
                )
                return self
        except (ImportError, OSError):
            pass

        # 2) CuPy
        try:
            import cupy as cp
            pool = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(pool.malloc)
            _, total = cp.cuda.Device(0).mem_info
            limit = int(total * self.fraction)
            pool.set_limit(limit)
            self._method = (
                f"cupy | 상한 {self.fraction:.0%} "
                f"({limit//1024//1024} MB / {total//1024//1024} MB)"
            )
            return self
        except ImportError:
            pass

        # 3) ctypes – (1-fraction) 만큼 선점 예약
        self._try_ctypes_reserve()
        return self

    def _try_ctypes_reserve(self) -> None:
        import ctypes
        total_mb = _get_gpu_total_mb()
        if total_mb <= 0:
            return

        reserve_mb = max(0, int(total_mb * (1.0 - self.fraction)) - 128)
        if reserve_mb <= 0:
            return

        # CUDA Runtime DLL 탐색 (Windows)
        for dll in ["cudart64_12", "cudart64_120", "cudart64_121",
                    "cudart64_11", "cudart64_110", "cudart64_112"]:
            try:
                lib = ctypes.CDLL(dll + ".dll")
                self._cuda_lib = lib
                break
            except OSError:
                continue

        if self._cuda_lib is None:
            self._method = f"GPU 감지됨 (제한 미적용 – CUDA DLL 없음)"
            return

        ptr = ctypes.c_void_p()
        size = ctypes.c_size_t(reserve_mb * 1024 * 1024)
        rc = self._cuda_lib.cudaMalloc(ctypes.byref(ptr), size)
        if rc == 0:  # cudaSuccess
            self._cuda_ptr = ptr
            self._method = (
                f"ctypes CUDA | 상한 {self.fraction:.0%} "
                f"({reserve_mb} MB 선점 예약 / 전체 {total_mb} MB)"
            )
        else:
            self._method = f"GPU 감지됨 (cudaMalloc 실패 rc={rc} – 제한 미적용)"

    # ── 종료 ──────────────────────────────────────────────────────────────
    def __exit__(self, *_) -> None:
        if self._cuda_lib and self._cuda_ptr and self._cuda_ptr.value:
            try:
                self._cuda_lib.cudaFree(self._cuda_ptr)
            except Exception:
                pass
            self._cuda_ptr = None

    def __del__(self):
        """안전망: 예외 등으로 __exit__이 호출되지 않은 경우 GC 시 GPU 메모리 해제."""
        try:
            self.__exit__(None, None, None)
        except Exception:
            pass

    @property
    def info(self) -> str:
        return self._method


# ─────────────────────────────────────────────────────────────────────────────
# DuckDB 디스크 스필 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

DUCKDB_MEMORY_LIMIT_MB: int = 512  # DuckDB 내부 메모리 한도 기본값 (초과 시 디스크 스필)


def _detect_system_ram_mb() -> int:
    """시스템 총 RAM(MB) 감지. psutil 없으면 4096 기본값."""
    try:
        import psutil
        return int(psutil.virtual_memory().total / 1024 / 1024)
    except Exception:
        return 4096


# 시스템 RAM의 75%를 기본 한도로 사용 (최소 512, 최대 16384)
PROCESS_MEMORY_LIMIT_MB: int = max(512, min(65536, _detect_system_ram_mb() * 3 // 4))


def _mem_usage_mb() -> int:
    """현재 프로세스의 RSS 메모리 사용량(MB). psutil 없으면 -1."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss / 1024 / 1024)
    except Exception:
        return -1


def _log_mem(label: str, progress_cb=None):
    """메모리 사용량을 로그에 기록."""
    mb = _mem_usage_mb()
    if mb > 0:
        msg = f"[MEM] {label}: {mb:,} MB"
        if progress_cb:
            progress_cb(msg)
        else:
            logger.info(msg)


def _duckdb_available() -> bool:
    """duckdb 패키지 설치 여부 확인."""
    try:
        import duckdb  # noqa: F401
        return True
    except ImportError:
        return False


@contextlib.contextmanager
def _duck_con(
    memory_limit_mb: int = DUCKDB_MEMORY_LIMIT_MB,
    tmp_dir: "Path | None" = None,
):
    """
    DuckDB 연결 컨텍스트 매니저.

    memory_limit_mb 초과 시 tmp_dir로 중간 결과를 디스크에 스필하여
    Python RAM을 보호합니다.
    tmp_dir=None 이면 자동 생성 후 종료 시 삭제.
    """
    import duckdb
    import shutil
    import tempfile

    _auto_tmp = tmp_dir is None
    _tmp = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="duck_tmp_"))
    _tmp.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit='{memory_limit_mb}MB'")
        con.execute(f"SET temp_directory='{_tmp.as_posix()}'")
        yield con
    finally:
        try:
            con.close()
        except Exception:
            pass
        if _auto_tmp:
            shutil.rmtree(_tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 피처 변환
# ─────────────────────────────────────────────────────────────────────────────

def _patient_features_to_row(f: PatientFeatures) -> dict:
    """PatientFeatures 1건 → dict (DataFrame 행)."""
    return {
        "patient_id": f.patient_id,
        "drug_count": f.drug_count,
        "drug_count_7d": f.drug_count_7d,
        "institution_count": f.institution_count,
        "ddi_contraindicated": f.ddi_contraindicated,
        "ddi_major": f.ddi_major,
        "ddi_moderate": f.ddi_moderate,
        "ddi_minor": f.ddi_minor,
        "triple_whammy": int(f.triple_whammy),
        "qt_risk_count": f.qt_risk_count,
        "dup_same_ingredient": f.dup_same_ingredient,
        "dup_atc5": f.dup_atc5,
        "dup_atc4": f.dup_atc4,
        "dup_atc3": f.dup_atc3,
        "dup_efmdc": f.dup_efmdc,
        "has_high_risk_drug": int(f.has_high_risk_drug),
        "has_renal_risk_drug": int(f.has_renal_risk_drug),
        "has_hepatic_risk_drug": int(f.has_hepatic_risk_drug),
        "cyp_risk_score": f.cyp_risk_score,
        "cyp_max_enzyme_risk": f.cyp_max_enzyme_risk,
        "cyp_high_risk_pairs": f.cyp_high_risk_pairs,
        "age": f.age if f.age is not None else -1,
        "sex_m": 1 if f.sex == "1" else 0,
        "risk_level": f.risk_level,
        "risk_label": RISK_LABEL_MAP.get(f.risk_level, 0),
        "risk_binary": 1 if f.risk_level in ("Red", "Yellow") else 0,
    }


def features_to_dataframe(features_list: list[PatientFeatures]) -> pd.DataFrame:
    """PatientFeatures 리스트 → DataFrame (소량 데이터용, 하위 호환)."""
    return pd.DataFrame([_patient_features_to_row(f) for f in features_list])


def _flush_features_to_parquet(
    features_list: list[PatientFeatures],
    out_dir: Path,
    batch_idx: int,
) -> Path:
    """features 배치를 Parquet에 저장하고 리스트를 비움. 경로 반환."""
    df = pd.DataFrame([_patient_features_to_row(f) for f in features_list])
    out_path = out_dir / f"features_batch_{batch_idx:04d}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def load_features_from_parquet(
    parquet_path: "str | Path",
    columns: list[str] | None = None,
    memory_limit_mb: int = 0,
) -> pd.DataFrame:
    """DuckDB로 피처 Parquet 읽기 (디스크 스필 지원). 단일 파일 또는 glob."""
    p = Path(parquet_path)
    mem = max(256, (memory_limit_mb // 2) if memory_limit_mb > 0 else PROCESS_MEMORY_LIMIT_MB // 2)
    cols = ", ".join(columns) if columns else "*"

    if _duckdb_available():
        # glob 패턴 또는 단일 파일
        pattern = p.as_posix()
        with _duck_con(memory_limit_mb=mem) as con:
            return con.execute(f"SELECT {cols} FROM read_parquet('{pattern}')").df()
    else:
        return pd.read_parquet(p, columns=columns)


# ── 피처 Parquet 디렉토리 ────────────────────────────────────────────────

FEATURES_CACHE_DIR = Path(__file__).parent.parent / "data" / "features_cache"


# ─────────────────────────────────────────────────────────────────────────────
# 사전 층화 샘플링 (자격DB 인구통계 기반)
# ─────────────────────────────────────────────────────────────────────────────

# 의학적 연령 구간 (기본값)
_DEFAULT_AGE_BINS   = [0,  20,  40,  60,  75,  200]
_DEFAULT_AGE_LABELS = ["0-19", "20-39", "40-59", "60-74", "75+"]


def _allocate_stratum_quotas(
    strata_counts: pd.Series,
    sample_size: int,
) -> dict[str, int]:
    """층별 모집단 수에 비례해 총 sample_size를 정확히 배분한다."""
    total = int(strata_counts.sum())
    if sample_size <= 0 or total <= 0:
        return {}

    alloc: dict[str, int] = {}
    fractional: dict[str, float] = {}
    for st, cnt_raw in strata_counts.items():
        cnt = int(cnt_raw)
        exact = cnt / total * sample_size
        floor_val = min(int(exact), cnt)
        alloc[st] = floor_val
        fractional[st] = exact - floor_val

    remaining = sample_size - sum(alloc.values())
    for st in sorted(fractional, key=fractional.__getitem__, reverse=True):
        if remaining <= 0:
            break
        if alloc[st] < int(strata_counts[st]):
            alloc[st] += 1
            remaining -= 1

    return alloc


def stratify_and_sample_patients(
    elig_df: pd.DataFrame,
    sample_size: int,
    reference_year: int,
    seed: int = 42,
    pid_col: str = "INDI_DSCM_NO",
    byear_col: str = "BYEAR",
    sex_col: str = "SEX_TYPE",
    addr_col: str = "RVSN_ADDR_CD",
    addr_digits: int = 5,
    age_bins: "list[int] | None" = None,
    age_labels: "list[str] | None" = None,
) -> "tuple[list[str], dict[str, dict], dict]":
    """자격DB DataFrame에서 성별·연령·지역 층화 샘플링.

    Parameters
    ----------
    elig_df : pd.DataFrame
        fetch_eligibility_for_sampling() 반환값.
        INDI_DSCM_NO, BYEAR, SEX_TYPE, RVSN_ADDR_CD 컬럼 필요.
    sample_size : int
        추출할 총 환자 수.
    reference_year : int
        나이 계산 기준 연도 (나이 = reference_year - BYEAR).
    seed : int
        랜덤 시드.
    addr_digits : int
        RVSN_ADDR_CD 앞 몇 자리로 지역 구분 (5=시군구, 8=읍면동).
    age_bins / age_labels : list, optional
        연령 구간 경계·레이블. 미지정 시 기본값 사용.

    Returns
    -------
    sampled_pids : list[str]
        샘플링된 INDI_DSCM_NO 목록.
    demographics : dict[str, dict]
        {patient_id: {"byear": int, "sex_type": str, "addr_cd": str}}
        save_demographics() 에 직접 전달 가능.
    strata_summary : dict
        층별 샘플 수 요약.
    """
    import numpy as np

    bins   = age_bins   or _DEFAULT_AGE_BINS
    labels = age_labels or _DEFAULT_AGE_LABELS

    df = elig_df.copy()
    if df.empty or sample_size <= 0:
        return [], {}, {}

    df[pid_col] = df[pid_col].astype(str).str.strip()
    df = df[df[pid_col] != ""].copy()
    df = df.drop_duplicates(subset=[pid_col], keep="last").reset_index(drop=True)
    unique_patients = int(df[pid_col].nunique())
    if unique_patients < sample_size:
        raise ValueError(
            "사전 조회된 코호트의 고유 환자 수가 요청 샘플 수보다 적습니다: "
            f"{unique_patients:,}명 < {sample_size:,}명"
        )

    # ── 나이·층화 키 생성 ─────────────────────────────────────────────
    df["_age"] = reference_year - pd.to_numeric(df[byear_col], errors="coerce")
    df["_age_grp"] = pd.cut(
        df["_age"], bins=bins, labels=labels, right=False
    ).astype(str).fillna("unknown")
    df["_sex"] = df[sex_col].astype(str).str.strip().fillna("U")
    df["_addr"] = df[addr_col].astype(str).str[:addr_digits].fillna("00000")
    df["_strata"] = df["_sex"] + "|" + df["_age_grp"] + "|" + df["_addr"]

    total = len(df)
    if total == sample_size:
        sampled = df
    else:
        strata_counts = df["_strata"].value_counts()
        alloc = _allocate_stratum_quotas(strata_counts, sample_size)

        rng = np.random.default_rng(seed)
        parts: list[pd.DataFrame] = []
        zero_eligible_strata: list[str] = []
        for st, a in alloc.items():
            if a <= 0:
                continue
            sub = df[df["_strata"] == st]
            if sub.empty:
                zero_eligible_strata.append(st)
                continue
            if len(sub) <= a:
                parts.append(sub)
            else:
                parts.append(
                    sub.sample(n=a, random_state=int(rng.integers(0, 2**31)))
                )

        if zero_eligible_strata:
            raise ValueError(
                "할당된 층 중 추출 가능한 환자가 없는 층이 있습니다: "
                + ", ".join(zero_eligible_strata[:10])
            )

        sampled = pd.concat(parts, ignore_index=True) if parts else df.head(0)
        if len(sampled) != sample_size:
            raise ValueError(
                "층화 샘플링 quota를 정확히 충족하지 못했습니다: "
                f"요청 {sample_size:,}명, 실제 {len(sampled):,}명"
            )

    # ── 결과 변환 ─────────────────────────────────────────────────────
    sampled_pids: list[str] = []
    demographics: dict[str, dict] = {}
    for row in sampled.itertuples(index=False):
        pid  = str(getattr(row, pid_col, "")).strip()
        if not pid:
            continue
        byear = getattr(row, byear_col, None)
        sex   = str(getattr(row, sex_col, "") or "").strip() or None
        addr  = str(getattr(row, addr_col, "") or "").strip()
        sampled_pids.append(pid)
        demographics[pid] = {
            "byear":    int(float(byear)) if byear is not None else None,
            "sex_type": sex,
            "addr_cd":  addr[:addr_digits] if addr else None,
        }

    strata_summary = sampled["_strata"].value_counts().to_dict()
    logger.info(
        "층화 샘플링 완료: %d명 → %d명 (%d 층)", total, len(sampled_pids), len(strata_summary)
    )
    return sampled_pids, demographics, strata_summary


# ─────────────────────────────────────────────────────────────────────────────
# 층화 샘플링 (DuckDB 기반, 학습 시 사용)
# ─────────────────────────────────────────────────────────────────────────────

def stratified_sample_from_parquet(
    parquet_paths: "list[Path] | str | Path",
    target_col: str,
    sample_size: int,
    seed: int = 42,
    memory_limit_mb: int = 0,
) -> pd.DataFrame:
    """
    DuckDB 기반 층화 샘플링 — 전체 데이터를 메모리에 올리지 않음.

    각 target_col 값(위험도 등급)의 비율을 유지하면서 sample_size 건 추출.
    예: 전체 4천만 명 중 Red 5%, Yellow 15%, Green 30%, Normal 50% 비율이면
        sample_size=100만 → Red 5만, Yellow 15만, Green 30만, Normal 50만

    Parameters
    ----------
    parquet_paths : Parquet 파일 경로 (단일 또는 리스트)
    target_col    : 층화 기준 컬럼 (예: "risk_level", "risk_binary")
    sample_size   : 추출할 총 건수
    seed          : 랜덤 시드 (재현성)
    memory_limit_mb : DuckDB 메모리 한도

    Returns
    -------
    층화 샘플링된 DataFrame (sample_size행)
    """
    mem = max(256, (memory_limit_mb // 2) if memory_limit_mb > 0 else PROCESS_MEMORY_LIMIT_MB // 2)

    if isinstance(parquet_paths, (str, Path)):
        _src_expr = f"read_parquet('{Path(parquet_paths).as_posix()}')"
    else:
        _src = ", ".join(f"'{Path(p).as_posix()}'" for p in parquet_paths)
        _src_expr = f"read_parquet([{_src}])"

    if not _duckdb_available():
        # pandas 폴백: 파일별 청크 로드 후 샘플링 (OOM 방지)
        import gc as _gc
        _files = parquet_paths if isinstance(parquet_paths, list) else [parquet_paths]
        parts: list[pd.DataFrame] = []
        for _fp in _files:
            parts.append(pd.read_parquet(_fp))
            if len(parts) > 1:
                parts = [pd.concat(parts, ignore_index=True)]
                _gc.collect()
        df = parts[0] if parts else pd.DataFrame()
        del parts
        _gc.collect()
        if len(df) <= sample_size:
            return df
        from sklearn.model_selection import train_test_split
        sampled, _ = train_test_split(
            df, train_size=sample_size, random_state=seed, stratify=df[target_col],
        )
        del df
        _gc.collect()
        return sampled.reset_index(drop=True)

    with _duck_con(memory_limit_mb=mem) as con:
        # 1. 각 클래스별 건수 조회
        dist = con.execute(f"""
            SELECT {target_col}, COUNT(*) AS cnt
            FROM {_src_expr}
            GROUP BY {target_col}
        """).df()

        total = int(dist["cnt"].sum())
        if total <= sample_size:
            # 전체가 샘플 크기 이하 → 전부 반환
            return con.execute(f"SELECT * FROM {_src_expr}").df()

        # 2. 클래스별 샘플 수 계산 (비율 유지)
        dist["sample_n"] = (dist["cnt"] / total * sample_size).astype(int)
        # 반올림 오차 보정: 부족분을 가장 큰 클래스에 추가
        _diff = sample_size - int(dist["sample_n"].sum())
        if _diff > 0:
            dist.loc[dist["cnt"].idxmax(), "sample_n"] += _diff

        # 3. 클래스별 SAMPLE 쿼리 실행 (디스크 기반)
        parts = []
        for _, row in dist.iterrows():
            # numpy scalar → Python native (DuckDB 파라미터 바인딩용)
            raw_val = row[target_col]
            cls_val = raw_val.item() if hasattr(raw_val, "item") else raw_val
            n = int(row["sample_n"])
            if n <= 0:
                continue
            # DuckDB USING SAMPLE은 비율만 지원 → ORDER BY + LIMIT 사용
            # `||` : DuckDB 표준 문자열 결합 (`+` 은 버전에 따라 동작 다름)
            _cls_df = con.execute(f"""
                SELECT * FROM {_src_expr}
                WHERE {target_col} = ?
                ORDER BY hash(CAST(patient_id AS VARCHAR) || CAST({seed} AS VARCHAR))
                LIMIT {n}
            """, [cls_val]).df()
            parts.append(_cls_df)

        if not parts:
            return pd.DataFrame()

        result = pd.concat(parts, ignore_index=True)
        # 셔플
        result = result.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return result


def stratified_sample_from_df(
    df: pd.DataFrame,
    target_col: str,
    sample_size: int,
    seed: int = 42,
) -> pd.DataFrame:
    """DataFrame 기반 층화 샘플링 (소량 데이터용)."""
    if len(df) <= sample_size:
        return df
    from sklearn.model_selection import train_test_split
    sampled, _ = train_test_split(
        df, train_size=sample_size, random_state=seed, stratify=df[target_col],
    )
    return sampled.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 환자별 집계
# ─────────────────────────────────────────────────────────────────────────────

def build_patient_features(
    records: list[PrescriptionRecord],
    window_days: int = 90,
    poly_threshold: int = 5,
    progress_cb: Callable[[str], None] | None = None,
    progress_pct_cb: Callable[[float], None] | None = None,
    guard=None,
) -> list[PatientFeatures]:
    """PrescriptionRecord → PatientFeatures 변환."""
    from hana_app.core.memory_guard import get_guard
    _guard = get_guard(guard)

    # DDI 매트릭스 로드 시도
    ddi_matrix = _load_ddi_matrix()
    dup_groups = _load_dup_groups()
    age_map = _load_age_map()
    cyp_ext = _load_cyp_extractor()

    # 환자별 그룹화
    import gc as _gc
    by_patient: dict[str, list[PrescriptionRecord]] = defaultdict(list)
    for r in records:
        by_patient[r.patient_id].append(r)
    # records 원본은 호출자가 소유하므로 여기서 해제하지 않음.
    # by_patient 구축 후 추가적인 records 순회는 없으므로 루프 변수만 정리.
    del r
    _gc.collect()

    features_list: list[PatientFeatures] = []
    total = len(by_patient)

    for i, (pid, precs) in enumerate(by_patient.items()):
        if i % 500 == 0:
            _guard.check()
            if progress_cb:
                progress_cb(f"피처 계산 중... {i:,}/{total:,} ({i/total*100:.0f}%)")
            if progress_pct_cb:
                progress_pct_cb(i / max(total, 1))

        # 다재약물 기준 미달 환자 제외
        if len({p.wk_compn_cd for p in precs}) < poly_threshold:
            continue

        overlaps = calculate_overlaps_for_patient(precs, window_days=window_days)
        feat = aggregate_patient_features(
            patient_id=pid,
            prescriptions=precs,
            overlap_pairs=overlaps,
            ddi_matrix=ddi_matrix,
            dup_groups=dup_groups,
            age=age_map.get(pid),
            cyp_extractor=cyp_ext,
        )
        features_list.append(feat)

    if progress_cb:
        progress_cb(
            f"피처 완료 – 다재약물 환자 {len(features_list):,}명 / "
            f"전체 {total:,}명"
        )
    return features_list


def build_patient_features_from_parquet(
    parquet_paths: list,
    window_days: int = 90,
    poly_threshold: int = 5,
    patient_batch_size: int = 5000,
    memory_limit_mb: int = 0,
    progress_cb: Callable[[str], None] | None = None,
    progress_pct_cb: Callable[[float], None] | None = None,
    guard=None,
) -> list[PatientFeatures]:
    """
    Parquet 파일(s) -> 환자별 피처 계산.

    소량 데이터: 기존 일괄 로드 방식.
    대량 데이터: 디스크 기반 해시 파티셔닝으로 RAM 절약.
      Phase 1 – patient_id 컬럼만 읽어 행 수 파악
      Phase 2 – 소스 Parquet를 하나씩 읽어 환자ID 해시로 임시 파일 분배
      Phase 3 – 파티션 단위로 로드 → 피처 계산 → 결과 누적
    피크 메모리: max(소스 1개, 파티션 1개) + 참조 데이터

    Parameters
    ----------
    memory_limit_mb : 사용자 설정 RAM 한도(MB). 0이면 PROCESS_MEMORY_LIMIT_MB 사용.
    guard : MemoryGuard 또는 None. 배치 경계마다 RSS 체크.
    """
    import gc
    import shutil
    import tempfile
    from collections import defaultdict
    from hana_app.core.hana_etl import df_row_to_record
    from hana_app.core.memory_guard import get_guard
    _guard = get_guard(guard)

    mem_limit = memory_limit_mb if memory_limit_mb > 0 else PROCESS_MEMORY_LIMIT_MB
    # DuckDB에 할당할 메모리: 전체 한도의 50% (나머지는 pandas/피처계산용)
    duck_mem = max(256, mem_limit // 2)
    # 소량 데이터 임계값: RAM이 적으면 더 일찍 디스크 모드 전환
    SMALL_THRESHOLD = max(10_000, min(100_000, mem_limit * 12))

    ddi_matrix = _load_ddi_matrix()
    dup_groups = _load_dup_groups()
    age_map = _load_age_map()
    cyp_ext = _load_cyp_extractor()

    # ── Phase 1: 행 수 파악 (patient_id 컬럼만) ──────────────────
    if progress_cb:
        progress_cb(
            f"Parquet {len(parquet_paths)}개 파일 스캔 중... "
            f"(RAM 한도: {mem_limit:,} MB, DuckDB: {duck_mem:,} MB)"
        )

    total_rows = 0
    if _duckdb_available():
        with _duck_con(memory_limit_mb=duck_mem) as _con:
            for p in parquet_paths:
                _r = _con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{Path(p).as_posix()}')"
                ).fetchone()
                total_rows += _r[0]
    else:
        for p in parquet_paths:
            _tmp = pd.read_parquet(p, columns=["patient_id"])
            total_rows += len(_tmp)
            del _tmp
        gc.collect()

    if progress_cb:
        progress_cb(f"전체 레코드: {total_rows:,}건")

    # ── 소량 데이터: 기존 일괄 처리 ──────────────────────────────
    if total_rows <= SMALL_THRESHOLD:
        if progress_cb:
            progress_cb("소량 데이터 → 일괄 처리 모드")
        return _build_features_simple(
            parquet_paths, ddi_matrix, dup_groups,
            window_days, poly_threshold, patient_batch_size, progress_cb,
            progress_pct_cb, memory_limit_mb=mem_limit,
        )

    # ── Phase 2: 디스크 기반 해시 파티셔닝 ────────────────────────
    # 메모리가 적을수록 파티션을 많이 나눠 파티션당 크기를 줄임
    _rows_per_part = max(20_000, mem_limit * 25)  # ~100B/row 가정
    num_partitions = max(4, min(256, total_rows // _rows_per_part + 1))
    tmp_dir = Path(tempfile.mkdtemp(prefix="hana_feat_"))

    if progress_cb:
        progress_cb(
            f"대량 데이터 → 디스크 파티셔닝 "
            f"({num_partitions}개 파티션, 임시={tmp_dir.name})"
        )

    _use_duck = _duckdb_available()
    try:
        if _use_duck:
            # DuckDB: 전체 소스를 한번에 읽어 파티션 분배 — 메모리 초과 시 디스크 스필
            if progress_cb:
                progress_cb(
                    f"DuckDB 파티셔닝 중 ({num_partitions}개 파티션) "
                    f"— 메모리 초과 시 자동 디스크 스필..."
                )
            if progress_pct_cb:
                progress_pct_cb(0.10)
            _src_list = ", ".join(f"'{Path(p).as_posix()}'" for p in parquet_paths)
            _duck_tmp = tmp_dir / "_duck_tmp"
            with _duck_con(memory_limit_mb=duck_mem, tmp_dir=_duck_tmp) as _con:
                _con.execute(f"""
                    COPY (
                        SELECT *,
                               (abs(hash(patient_id::VARCHAR)) % {num_partitions})::INTEGER AS _part
                        FROM read_parquet([{_src_list}])
                    )
                    TO '{tmp_dir.as_posix()}'
                    (FORMAT PARQUET, PARTITION_BY (_part))
                """)
            if progress_cb:
                progress_cb("DuckDB 파티셔닝 완료")
            if progress_pct_cb:
                progress_pct_cb(0.50)
        else:
            # 판다스 폴백: 파일 1개씩 읽어 해시 파티셔닝
            logger.warning(
                "DuckDB 미설치 — pandas 폴백 파티셔닝. "
                "대용량 데이터 시 OOM 위험. `pip install duckdb` 권장."
            )
            for si, src_path in enumerate(parquet_paths):
                if progress_cb:
                    progress_cb(
                        f"파티셔닝 [{si+1}/{len(parquet_paths)}] "
                        f"{Path(src_path).name}..."
                    )
                if progress_pct_cb:
                    progress_pct_cb(0.05 + 0.45 * si / max(len(parquet_paths), 1))
                src_df = pd.read_parquet(src_path)
                hashes = pd.util.hash_array(
                    src_df["patient_id"].to_numpy().astype(str)
                )
                src_df["_part"] = np.abs(hashes) % num_partitions
                for part_id, part_df in src_df.groupby("_part"):
                    out = tmp_dir / f"p{int(part_id)}_s{si}.parquet"
                    part_df.drop(columns=["_part"]).to_parquet(out, index=False)
                del src_df, hashes
                gc.collect()

        # ── Phase 3: 파티션별 피처 계산 → 디스크 flush ────────────
        # features_list를 메모리에 쌓지 않고 배치마다 Parquet에 저장
        feat_out_dir = FEATURES_CACHE_DIR
        feat_out_dir.mkdir(parents=True, exist_ok=True)
        feat_parquet_paths: list[Path] = []
        _flush_buf: list[PatientFeatures] = []
        _flush_idx = 0
        _FLUSH_SIZE = max(1000, patient_batch_size)
        processed_patients = 0
        total_features = 0

        def _do_flush():
            nonlocal _flush_idx, _flush_buf, total_features
            if _flush_buf:
                p = _flush_features_to_parquet(_flush_buf, feat_out_dir, _flush_idx)
                feat_parquet_paths.append(p)
                total_features += len(_flush_buf)
                _flush_idx += 1
                _flush_buf = []
                gc.collect()

        for part_id in range(num_partitions):
            _duck_part_dir = tmp_dir / f"_part={part_id}"
            if _duck_part_dir.exists():
                part_files = sorted(_duck_part_dir.glob("*.parquet"))
            else:
                part_files = sorted(tmp_dir.glob(f"p{part_id}_s*.parquet"))
            if not part_files:
                continue

            _pf_glob = ", ".join(f"'{f.as_posix()}'" for f in part_files)

            if _use_duck:
                _duck_part_tmp = tmp_dir / "_duck_tmp"
                with _duck_con(memory_limit_mb=duck_mem, tmp_dir=_duck_part_tmp) as _pcon:
                    _part_pids = [
                        r[0] for r in _pcon.execute(
                            f"SELECT DISTINCT patient_id "
                            f"FROM read_parquet([{_pf_glob}])"
                        ).fetchall()
                    ]
                    for _bs in range(0, len(_part_pids), patient_batch_size):
                        _bid = _part_pids[_bs:_bs + patient_batch_size]
                        _pcon.execute(
                            "CREATE OR REPLACE TEMP TABLE _bid AS "
                            "SELECT unnest(?) AS patient_id",
                            [_bid],
                        )
                        _bdf = _pcon.execute(
                            f"SELECT r.* FROM read_parquet([{_pf_glob}]) r "
                            f"INNER JOIN _bid b ON r.patient_id = b.patient_id"
                        ).df()
                        by_patient: dict[str, list[PrescriptionRecord]] = defaultdict(list)
                        for row in _bdf.itertuples(index=False):
                            by_patient[row.patient_id].append(df_row_to_record(row))
                        del _bdf
                        for pid, precs in by_patient.items():
                            if len({p.wk_compn_cd for p in precs}) < poly_threshold:
                                continue
                            overlaps = calculate_overlaps_for_patient(
                                precs, window_days=window_days,
                            )
                            feat = aggregate_patient_features(
                                patient_id=pid,
                                prescriptions=precs,
                                overlap_pairs=overlaps,
                                ddi_matrix=ddi_matrix,
                                dup_groups=dup_groups,
                                age=age_map.get(pid),
                                cyp_extractor=cyp_ext,
                            )
                            _flush_buf.append(feat)
                            if len(_flush_buf) >= _FLUSH_SIZE:
                                _do_flush()
                        del by_patient
                        gc.collect()
                        _guard.check()
                        # 메모리 압박 시 배치 크기 동적 축소
                        patient_batch_size = _guard.suggest_batch_size(
                            patient_batch_size, row_bytes=800,
                        )
                processed_patients += len(_part_pids)
            else:
                try:
                    part_dfs = [pd.read_parquet(f) for f in part_files]
                except MemoryError:
                    logger.error(
                        "파티션 %d 로드 중 메모리 부족 — 파일 1개씩 순차 로드 시도",
                        part_id,
                    )
                    gc.collect()
                    part_dfs = []
                    for f in part_files:
                        try:
                            part_dfs.append(pd.read_parquet(f))
                        except MemoryError:
                            logger.warning("파일 %s 건너뜀 (메모리 부족)", f)
                            gc.collect()
                part_df = pd.concat(part_dfs, ignore_index=True) if part_dfs else pd.DataFrame()
                del part_dfs
                gc.collect()
                by_patient = defaultdict(list)
                for row in part_df.itertuples(index=False):
                    by_patient[row.patient_id].append(df_row_to_record(row))
                del part_df
                gc.collect()
                for pid, precs in by_patient.items():
                    if len({p.wk_compn_cd for p in precs}) < poly_threshold:
                        continue
                    overlaps = calculate_overlaps_for_patient(
                        precs, window_days=window_days,
                    )
                    feat = aggregate_patient_features(
                        patient_id=pid,
                        prescriptions=precs,
                        overlap_pairs=overlaps,
                        ddi_matrix=ddi_matrix,
                        dup_groups=dup_groups,
                        age=age_map.get(pid),
                        cyp_extractor=cyp_ext,
                    )
                    _flush_buf.append(feat)
                    if len(_flush_buf) >= _FLUSH_SIZE:
                        _do_flush()
                processed_patients += len(by_patient)
                del by_patient
                gc.collect()

            if progress_cb:
                progress_cb(
                    f"파티션 {part_id+1}/{num_partitions} 완료 "
                    f"({processed_patients:,}명 처리, 피처 {total_features:,}명 저장)"
                )
            if progress_pct_cb:
                progress_pct_cb(0.50 + 0.50 * (part_id + 1) / num_partitions)

        # 잔여 버퍼 flush
        _do_flush()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if progress_cb:
        progress_cb(
            f"피처 완료 – 다재약물 환자 {total_features:,}명 / "
            f"전체 {processed_patients:,}명 / "
            f"Parquet {len(feat_parquet_paths)}개 파일"
        )
    # 디스크 기반 반환: Parquet glob 경로
    return feat_parquet_paths


def _build_features_simple(
    parquet_paths: list,
    ddi_matrix,
    dup_groups,
    window_days: int,
    poly_threshold: int,
    patient_batch_size: int,
    progress_cb: Callable[[str], None] | None,
    progress_pct_cb: Callable[[float], None] | None = None,
    memory_limit_mb: int = 0,
) -> list[PatientFeatures]:
    """소량 데이터용 – 전체 로드 후 배치 처리. DuckDB 사용 시 디스크 스필."""
    import gc
    from collections import defaultdict
    from hana_app.core.hana_etl import df_row_to_record

    age_map = _load_age_map()
    cyp_ext = _load_cyp_extractor()
    mem_limit = memory_limit_mb if memory_limit_mb > 0 else PROCESS_MEMORY_LIMIT_MB
    duck_mem = max(256, mem_limit // 2)
    _use_duck = _duckdb_available() and len(parquet_paths) > 0
    _src_list = ", ".join(f"'{Path(p).as_posix()}'" for p in parquet_paths) if _use_duck else ""

    # 환자 ID 목록만 먼저 추출 (전체 데이터 로드 안 함)
    if _use_duck:
        with _duck_con(memory_limit_mb=duck_mem) as _con:
            patient_ids = [
                r[0] for r in _con.execute(
                    f"SELECT DISTINCT patient_id FROM read_parquet([{_src_list}])"
                ).fetchall()
            ]
    else:
        _id_set: set[str] = set()
        for p in parquet_paths:
            _id_set.update(pd.read_parquet(p, columns=["patient_id"])["patient_id"].unique())
            gc.collect()
        patient_ids = list(_id_set)
        del _id_set

    total_patients = len(patient_ids)

    # 디스크 기반 피처 저장
    feat_out_dir = FEATURES_CACHE_DIR
    feat_out_dir.mkdir(parents=True, exist_ok=True)
    feat_parquet_paths: list[Path] = []
    _flush_buf: list[PatientFeatures] = []
    _flush_idx = 0
    total_features = 0

    def _do_flush():
        nonlocal _flush_idx, _flush_buf, total_features
        if _flush_buf:
            p = _flush_features_to_parquet(_flush_buf, feat_out_dir, _flush_idx)
            feat_parquet_paths.append(p)
            total_features += len(_flush_buf)
            _flush_idx += 1
            _flush_buf = []
            gc.collect()

    if progress_cb:
        progress_cb(f"환자 {total_patients:,}명 → 배치 {patient_batch_size:,}명씩 처리 (전체 데이터 메모리 로드 없음)")

    for batch_start in range(0, total_patients, patient_batch_size):
        batch_ids = patient_ids[batch_start:batch_start + patient_batch_size]

        if progress_cb:
            pct = batch_start / max(total_patients, 1) * 100
            progress_cb(
                f"피처 계산 중... {batch_start:,}/{total_patients:,} "
                f"({pct:.0f}%)"
            )
        if progress_pct_cb:
            progress_pct_cb(batch_start / max(total_patients, 1))

        if _use_duck:
            with _duck_con(memory_limit_mb=duck_mem) as _con:
                _con.execute(
                    "CREATE OR REPLACE TEMP TABLE _bid AS "
                    "SELECT unnest(?) AS patient_id",
                    [batch_ids],
                )
                batch_df = _con.execute(
                    f"SELECT r.* FROM read_parquet([{_src_list}]) r "
                    f"INNER JOIN _bid b ON r.patient_id = b.patient_id"
                ).df()
        else:
            _batch_set = set(batch_ids)
            _parts = []
            for p in parquet_paths:
                _tmp = pd.read_parquet(p, columns=None)
                _parts.append(_tmp[_tmp["patient_id"].isin(_batch_set)])
                del _tmp
                gc.collect()
            batch_df = pd.concat(_parts, ignore_index=True) if _parts else pd.DataFrame()
            del _parts
            gc.collect()

        by_patient: dict[str, list[PrescriptionRecord]] = defaultdict(list)
        for row in batch_df.itertuples(index=False):
            by_patient[row.patient_id].append(df_row_to_record(row))
        del batch_df
        gc.collect()

        for pid, precs in by_patient.items():
            if len({p.wk_compn_cd for p in precs}) < poly_threshold:
                continue
            overlaps = calculate_overlaps_for_patient(
                precs, window_days=window_days,
            )
            feat = aggregate_patient_features(
                patient_id=pid,
                prescriptions=precs,
                overlap_pairs=overlaps,
                ddi_matrix=ddi_matrix,
                dup_groups=dup_groups,
                age=age_map.get(pid),
                cyp_extractor=cyp_ext,
            )
            _flush_buf.append(feat)

        del by_patient
        gc.collect()

        # 배치 끝마다 flush
        _do_flush()

    # 잔여 flush
    _do_flush()

    if progress_cb:
        progress_cb(
            f"피처 완료 – 다재약물 환자 {total_features:,}명 / "
            f"전체 {total_patients:,}명 / "
            f"Parquet {len(feat_parquet_paths)}개 파일"
        )
    return feat_parquet_paths


def _load_parquet_safe(candidates: list[Path]) -> pd.DataFrame | None:
    """Parquet 안전 로드. DuckDB가 있으면 디스크 스필로 OOM 방지."""
    for p in candidates:
        if p.exists():
            if _duckdb_available():
                try:
                    with _duck_con(memory_limit_mb=max(256, PROCESS_MEMORY_LIMIT_MB // 4)) as con:
                        return con.execute(
                            f"SELECT * FROM read_parquet('{p.as_posix()}')"
                        ).df()
                except Exception:
                    pass
            return pd.read_parquet(p)
    return None


def _load_ddi_matrix() -> pd.DataFrame | None:
    return _load_parquet_safe([
        ROOT / "data" / "processed" / "ddi_matrix_final.parquet",
        ROOT / "drugbank" / "ddi_matrix_final.parquet",
    ])


def _load_dup_groups() -> pd.DataFrame | None:
    return _load_parquet_safe([
        ROOT / "data" / "processed" / "efcy_duplicate_groups.parquet",
        ROOT / "drugbank" / "efcy_duplicate_groups.parquet",
    ])


AGE_MAP_PATH = ROOT / "data" / "raw" / "eligibility_ages.parquet"
DEMOGRAPHICS_PATH = ROOT / "data" / "raw" / "eligibility_demographics.parquet"
CYP_MATRIX_PATH = ROOT / "data" / "processed" / "cyp_matrix.parquet"
DRUG_INDEX_PATH = ROOT / "data" / "processed" / "drug_name_index.parquet"


def _load_cyp_extractor():
    """CYP 피처 추출기 로드. 데이터 파일 없으면 None 반환."""
    if CYP_MATRIX_PATH.exists() and DRUG_INDEX_PATH.exists():
        try:
            from scripts.features.cyp_features import CYPFeatureExtractor
            return CYPFeatureExtractor(
                cyp_matrix_path=CYP_MATRIX_PATH,
                drug_index_path=DRUG_INDEX_PATH,
            )
        except Exception as e:
            logger.warning("CYP 추출기 로드 실패: %s", e)
    return None


def _load_demographics() -> dict[str, dict]:
    """저장된 자격DB 인구통계(BYEAR·SEX_TYPE·RVSN_ADDR_CD) 로드.

    Returns
    -------
    dict[str, dict]
        {patient_id: {"byear": int, "sex_type": str, "addr_cd": str}}
    파일 없으면 빈 딕셔너리 반환.
    """
    if DEMOGRAPHICS_PATH.exists():
        try:
            df = pd.read_parquet(DEMOGRAPHICS_PATH)
            result: dict[str, dict] = {}
            for row in df.itertuples(index=False):
                pid = str(row.patient_id)
                result[pid] = {
                    "byear":    int(row.byear) if pd.notna(row.byear) else None,
                    "sex_type": str(row.sex_type) if pd.notna(row.sex_type) else None,
                    "addr_cd":  str(row.addr_cd)  if pd.notna(row.addr_cd)  else None,
                }
            return result
        except Exception as e:
            logger.warning("인구통계 매핑 로드 실패: %s", e)
    return {}


def save_demographics(
    demographics: dict[str, dict],
    path: Path | None = None,
    reference_year: int | None = None,
) -> Path:
    """인구통계 매핑을 Parquet로 저장 (ETL 추출 후 호출).

    Parameters
    ----------
    demographics : dict[str, dict]
        {patient_id: {"byear": int, "sex_type": str, "addr_cd": str}}
        fetch_eligibility_demographics 반환값을 그대로 전달.
    path : Path, optional
        저장 경로. 미지정 시 DEMOGRAPHICS_PATH 사용.
    reference_year : int, optional
        나이 계산 기준 연도. 지정 시 "age" 열도 함께 저장.
    """
    from datetime import date as _date

    out = path or DEMOGRAPHICS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    ref = reference_year or _date.today().year

    rows = []
    for pid, d in demographics.items():
        byear = d.get("byear")
        rows.append({
            "patient_id": pid,
            "byear":      byear,
            "age":        (ref - int(byear)) if byear is not None else None,
            "sex_type":   d.get("sex_type"),
            "addr_cd":    d.get("addr_cd"),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    logger.info("인구통계 매핑 저장: %s (%d명)", out, len(demographics))

    # 구버전 AGE_MAP_PATH 도 동시 갱신 (하위 호환)
    try:
        age_df = df[["patient_id", "age"]].dropna(subset=["age"])
        age_df = age_df.astype({"age": int})
        age_df.to_parquet(AGE_MAP_PATH, index=False)
    except Exception as e:
        logger.warning("나이 매핑 동기화 실패: %s", e)

    return out


def _load_age_map() -> dict[str, int]:
    """저장된 자격DB 나이 매핑 로드.

    DEMOGRAPHICS_PATH가 있으면 그 파일에서 age 열을 읽고,
    없으면 구버전 AGE_MAP_PATH로 폴백합니다.
    """
    for fpath in (DEMOGRAPHICS_PATH, AGE_MAP_PATH):
        if fpath.exists():
            try:
                df = pd.read_parquet(fpath, columns=["patient_id", "age"])
                df = df.dropna(subset=["age"])
                return dict(zip(df["patient_id"].astype(str), df["age"].astype(int)))
            except Exception as e:
                logger.warning("나이 매핑 로드 실패 (%s): %s", fpath.name, e)
    return {}


def save_age_map(age_map: dict[str, int], path: Path | None = None) -> Path:
    """나이 매핑을 Parquet로 저장 (ETL 추출 후 호출).

    .. deprecated::
        save_demographics 사용 권장.
    """
    out = path or AGE_MAP_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"patient_id": list(age_map.keys()), "age": list(age_map.values())})
    df.to_parquet(out, index=False)
    logger.info("나이 매핑 저장: %s (%d명)", out, len(age_map))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 모델 학습
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    df: pd.DataFrame | None = None,
    features_parquet: "str | Path | list[Path] | None" = None,
    model_name: str = "xgboost",
    target: str = "risk_binary",
    params: dict[str, Any] | None = None,
    test_size: float = 0.2,
    cv_folds: int = 5,
    sampling_size: int = 0,
    sampling_rounds: int = 1,
    threshold_optimization: bool = False,
    cost_sensitive: bool = False,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    progress_cb: Callable[[str], None] | None = None,
    progress_pct_cb: Callable[[float], None] | None = None,
    gpu_memory_fraction: float = GPU_MEMORY_FRACTION,
    memory_limit_mb: int = 0,
    feature_cols: list[str] | None = None,
    guard=None,
    features_df=None,   # ← 추가: 위험도 분포 요약 저장용
) -> dict[str, Any]:
    """
    DataFrame → 모델 학습 → 평가 결과 반환.

    Parameters
    ----------
    model_name : "xgboost" | "lightgbm" | "random_forest" | "logistic"
    target     : "risk_binary" (2분류) | "risk_label" (4분류)
    params     : 하이퍼파라미터 dict (None이면 기본값)
    feature_cols : 사용할 피처 컬럼 리스트 (None이면 기본 FEATURE_COLS)
    guard : MemoryGuard 또는 None. 학습 라운드 사이에서 RSS 체크.

    Returns
    -------
    result : {
        "model": fitted model,
        "metrics": {...},
        "feature_importance": pd.DataFrame,
        "classification_report": str,
        "cv_scores": list[float],
    }
    """
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        classification_report,
        roc_auc_score,
        accuracy_score,
        f1_score,
        average_precision_score,
        confusion_matrix,
    )
    from sklearn.preprocessing import label_binarize
    from hana_app.core.memory_guard import get_guard
    _guard = get_guard(guard)

    mem_limit = memory_limit_mb if memory_limit_mb > 0 else PROCESS_MEMORY_LIMIT_MB
    _feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS

    import gc as _gc

    # ── 다중 라운드 층화 샘플링 ──────────────────────────────────
    _rounds = max(1, sampling_rounds)
    _sample = sampling_size if sampling_size > 0 else 0

    if _rounds > 1 and _sample <= 0:
        _rounds = 1  # 샘플링 없으면 반복 무의미

    if _rounds > 1 and progress_cb:
        progress_cb(f"층화 샘플링 {_rounds}회 반복 학습 (각 {_sample:,}명)")

    _round_results: list[dict] = []

    for _round in range(_rounds):
        _guard.check()  # 라운드 시작 전 RAM 여유 확인
        _round_seed = 42 + _round * 7  # 라운드마다 다른 시드
        _round_label = f"[Round {_round+1}/{_rounds}] " if _rounds > 1 else ""

        if _rounds > 1 and progress_cb:
            progress_cb(f"{_round_label}시작 (seed={_round_seed})")

        # ── 데이터 로딩 (층화 샘플링 적용) ──────────────────────────
        _target_col = target

        if df is not None:
            # DataFrame 직접 전달
            if _sample > 0 and len(df) > _sample:
                if progress_cb:
                    progress_cb(f"{_round_label}층화 샘플링: {len(df):,}명 → {_sample:,}명 (seed={_round_seed})")
                _src_df = stratified_sample_from_df(df, _target_col, _sample, seed=_round_seed)
            else:
                _src_df = df
            X = _src_df[_feature_cols].copy()
            y = _src_df[target].copy()
            if _src_df is not df:
                del _src_df
            _gc.collect()
        elif features_parquet is not None:
            # 디스크 기반: DuckDB 층화 샘플링
            if _sample > 0:
                if progress_cb:
                    progress_cb(f"{_round_label}DuckDB 층화 샘플링: {_sample:,}명 추출 (seed={_round_seed})")
                _sampled = stratified_sample_from_parquet(
                    features_parquet, _target_col, _sample,
                    seed=_round_seed, memory_limit_mb=mem_limit,
                )
                X = _sampled[_feature_cols]
                y = _sampled[target]
                del _sampled
                _gc.collect()
            else:
                # 샘플링 없이 전체 로드 (소량 데이터)
                _cols = _feature_cols + [target]
                _cols_sql = ", ".join(_cols)
                if isinstance(features_parquet, list):
                    _src = ", ".join(f"'{Path(p).as_posix()}'" for p in features_parquet)
                    _src_expr = f"read_parquet([{_src}])"
                else:
                    _src_expr = f"read_parquet('{Path(features_parquet).as_posix()}')"
                duck_mem = max(256, mem_limit // 2)
                if progress_cb:
                    progress_cb(f"{_round_label}DuckDB로 피처 전체 로딩 중...")
                if _duckdb_available():
                    with _duck_con(memory_limit_mb=duck_mem) as _con:
                        _feat_df = _con.execute(f"SELECT {_cols_sql} FROM {_src_expr}").df()
                else:
                    if isinstance(features_parquet, list):
                        _parts = []
                        for p in features_parquet:
                            _parts.append(pd.read_parquet(p, columns=_cols))
                            if len(_parts) > 1:
                                _parts = [pd.concat(_parts, ignore_index=True)]
                                _gc.collect()
                        _feat_df = _parts[0] if _parts else pd.DataFrame()
                        del _parts
                        _gc.collect()
                    else:
                        _feat_df = pd.read_parquet(features_parquet, columns=_cols)
                X = _feat_df[_feature_cols]
                y = _feat_df[target]
                del _feat_df
                _gc.collect()
        else:
            raise ValueError("df 또는 features_parquet 중 하나를 전달해야 합니다.")

        # 메모리 한도가 낮으면 병렬 작업 수를 제한하여 OOM 방지
        # 실제 시스템 RAM도 함께 고려 (사용자가 한도를 높게 잡아도 물리 RAM 초과 방지)
        _sys_ram = _detect_system_ram_mb()
        _effective_mem = min(mem_limit, _sys_ram)
        _n_jobs = -1
        if _effective_mem <= 2048:
            _n_jobs = 1
        elif _effective_mem <= 4096:
            _n_jobs = max(1, min(2, _effective_mem // 2048))

        if progress_cb:
            progress_cb(
                f"{_round_label}학습 데이터: {len(X):,}건 | 특성: {len(_feature_cols)}개 | "
                f"RAM 한도: {mem_limit:,} MB | 병렬: {_n_jobs}"
            )
            progress_cb(f"타겟 분포:\n{y.value_counts().to_string()}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=_round_seed, stratify=y
        )
        # 원본 X, y 즉시 해제 (train/test split 완료 후 불필요)
        del X, y
        _gc.collect()
        _log_mem(f"{_round_label}train_test_split 후", progress_cb)

        # ── GPU 설정 및 메모리 제한 ──────────────────────────────────
        _use_gpu = _has_cuda()
        _gpu_guard = _GpuMemoryGuard(gpu_memory_fraction)
        _gpu_guard.__enter__()
        if _round == 0 and progress_cb:
            if _use_gpu and _gpu_guard.info:
                progress_cb(f"GPU 사용: {_gpu_guard.info}")
            elif _use_gpu:
                progress_cb(
                    f"GPU 감지됨 – torch/cupy 미설치로 메모리 {gpu_memory_fraction:.0%} "
                    "제한 미적용 (XGBoost 기본 설정으로 진행)"
                )
            else:
                progress_cb("CPU 모드 (CUDA GPU 없음)")

        model = _build_model(
            model_name, target, params, use_gpu=_use_gpu, n_jobs=_n_jobs,
            cost_sensitive=cost_sensitive, cost_fp=cost_fp, cost_fn=cost_fn,
        )

        # 교차검증
        if progress_cb:
            progress_cb(f"{_round_label}{model_name} {cv_folds}-fold CV 실행 중...")
        if progress_pct_cb:
            progress_pct_cb(0.1)

        scoring = "roc_auc" if target == "risk_binary" else "f1_macro"
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=_n_jobs)

        if progress_cb:
            progress_cb(f"{_round_label}CV 완료: {scoring} = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        if progress_pct_cb:
            progress_pct_cb(0.5)

        # train/test 크기 기록 (del 전)
        _train_size = len(X_train)
        _test_size = len(X_test)
        _y_classes = sorted(y_test.unique() if hasattr(y_test, 'unique') else np.unique(y_test))

        # 최종 학습 — XGBoost 4분류는 sample_weight 로 클래스 가중치 전달
        # (XGBClassifier 는 class_weights 파라미터 무시하므로 fit() 에 직접)
        if progress_pct_cb:
            progress_pct_cb(0.6)
        if model_name == "xgboost":
            _xgb_sw = _xgb_multiclass_sample_weight(
                target, y_train, cost_sensitive, cost_fp, cost_fn,
            )
            if _xgb_sw is not None:
                model.fit(X_train, y_train, sample_weight=_xgb_sw)
            else:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        # fit 완료 후 학습 데이터 해제 (CV도 끝남)
        del X_train, y_train
        _gc.collect()
        _log_mem(f"{_round_label}model.fit 후", progress_cb)
        if progress_pct_cb:
            progress_pct_cb(0.8)
        y_pred = model.predict(X_test)

        # 평가 — 불균형 데이터(공단 pilot 3,540:1 수준)에선 accuracy 는 오도하므로
        # F1-macro (동일 가중치), F1-weighted (지지도 가중치), PR-AUC 를 병기.
        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "cv_scores": cv_scores.tolist(),
            "train_size": _train_size,
            "test_size": _test_size,
        }

        # AUC + PR-AUC — 극단 불균형(예: 3,540:1)에선 ROC-AUC 보다 PR-AUC 가
        # 소수 클래스 식별 능력을 더 민감하게 반영.
        try:
            if target == "risk_binary":
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
                metrics["pr_auc"] = float(average_precision_score(y_test, y_proba))
                # ROC Curve 포인트 (최대 200점으로 다운샘플)
                try:
                    from sklearn.metrics import roc_curve as _roc_curve
                    _fpr, _tpr, _ = _roc_curve(y_test, y_proba)
                    _step = max(1, len(_fpr) // 200)
                    metrics["roc_curve"] = {
                        "fpr": _fpr[::_step].tolist(),
                        "tpr": _tpr[::_step].tolist(),
                    }
                except Exception:
                    pass
            else:
                y_proba = model.predict_proba(X_test)
                _y_bin = label_binarize(y_test, classes=_y_classes)
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(
                        _y_bin, y_proba, multi_class="ovr", average="macro",
                    )
                )
                # PR-AUC One-vs-Rest (macro) — 소수 클래스(Green) 민감도 측정
                metrics["pr_auc_ovr"] = float(
                    average_precision_score(_y_bin, y_proba, average="macro")
                )
        except Exception:
            pass

        # 분류 보고서
        class_names = {0: "Normal", 1: "위험"} if target == "risk_binary" else {
            v: k for k, v in RISK_LABEL_MAP.items()
        }
        target_names = [class_names.get(c, str(c)) for c in _y_classes]
        report = classification_report(y_test, y_pred, target_names=target_names)
        metrics["classification_report"] = report

        # 혼동 행렬
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        metrics["classes"] = _y_classes

        # Threshold Optimization (이진 분류 전용)
        optimal_threshold = 0.5
        if threshold_optimization and target == "risk_binary":
            if progress_cb:
                progress_cb(f"{_round_label}Threshold Optimization 실행 중...")
            optimal_threshold, _ = _optimize_threshold(
                model, X_test, y_test, cost_fp=cost_fp, cost_fn=cost_fn,
            )
            y_proba_opt = model.predict_proba(X_test)[:, 1]
            y_pred_opt = (y_proba_opt >= optimal_threshold).astype(int)
            metrics["optimal_threshold"] = optimal_threshold
            metrics["accuracy_at_optimal"] = float(accuracy_score(y_test, y_pred_opt))
            metrics["f1_at_optimal"] = float(f1_score(y_test, y_pred_opt, average="macro"))
            del y_proba_opt, y_pred_opt
            if progress_cb:
                progress_cb(
                    f"{_round_label}최적 임계값: {optimal_threshold:.2f} "
                    f"(F1={metrics['f1_at_optimal']:.4f})"
                )

        # 평가용 데이터 해제 (모든 메트릭 계산 완료)
        del X_test, y_test, y_pred
        _gc.collect()

        # 피처 중요도
        fi_df = _get_feature_importance(model, model_name, _feature_cols)

        if progress_pct_cb:
            progress_pct_cb(0.9)
        if progress_cb:
            progress_cb(f"{_round_label}모델 저장 중...")

        # 저장
        result = {
            "model": model,
            "model_name": model_name,
            "target": target,
            "metrics": metrics,
            "feature_importance": fi_df,
            "params": params or {},
            "feature_cols": _feature_cols,
            "gpu": _gpu_guard.info or "CPU",
            "memory_limit_mb": mem_limit,
            "sampling_round": _round + 1,
            "sampling_seed": _round_seed,
            "sampling_size": _sample,
            "features_df": features_df,
        }
        _save_result(result)
        _round_results.append(result)

        # GPU 해제 (라운드별)
        _gpu_guard.__exit__(None, None, None)

        if progress_cb:
            progress_cb(
                f"{_round_label}✅ 완료 Accuracy={metrics['accuracy']:.4f} | "
                f"F1={metrics['f1_macro']:.4f}"
            )

    # ── 라운드 종료: 최종 결과 선택 ──────────────────────────────
    if not _round_results:
        raise RuntimeError("모든 학습 라운드가 실패했습니다.")

    if _rounds == 1:
        best_result = _round_results[0]
    else:
        # 최고 F1 라운드 선택
        best_result = max(_round_results, key=lambda r: r["metrics"].get("f1_macro", 0))
        # 전체 라운드 통계 추가
        _all_f1 = [r["metrics"]["f1_macro"] for r in _round_results]
        _all_auc = [r["metrics"].get("roc_auc", r["metrics"].get("roc_auc_ovr", 0)) for r in _round_results]
        best_result["round_summary"] = {
            "total_rounds": _rounds,
            "best_round": best_result["sampling_round"],
            "f1_scores": _all_f1,
            "f1_mean": float(np.mean(_all_f1)),
            "f1_std": float(np.std(_all_f1)),
            "auc_scores": _all_auc,
            "auc_mean": float(np.mean(_all_auc)),
            "auc_std": float(np.std(_all_auc)),
        }
        if progress_cb:
            progress_cb(
                f"📊 {_rounds}회 샘플링 결과: "
                f"F1 평균={np.mean(_all_f1):.4f} ± {np.std(_all_f1):.4f} | "
                f"최고 Round {best_result['sampling_round']}"
            )
        # 비-최적 라운드의 모델 객체 해제 (메모리 절약)
        for r in _round_results:
            if r is not best_result:
                r.pop("model", None)
                r.pop("feature_importance", None)
        best_result["all_rounds"] = _round_results

    if progress_pct_cb:
        progress_pct_cb(1.0)
    if progress_cb:
        m = best_result["metrics"]
        progress_cb(
            f"✅ 학습 완료! Accuracy={m['accuracy']:.4f} | "
            f"F1={m['f1_macro']:.4f}"
        )

    return best_result


def _xgb_multiclass_sample_weight(target: str, y_train, cost_sensitive: bool,
                                   cost_fp: float, cost_fn: float):
    """XGBoost 4분류 학습용 sample_weight 배열.

    XGBoost 는 multiclass 에서 class_weights 파라미터를 무시하므로
    (실측 확인: "Parameters: { \"class_weights\" } are not used." 경고),
    sample_weight 를 fit() 에 직접 전달해야 가중치가 적용된다.

    Returns
    -------
    numpy.ndarray | None
        XGBoost 4분류에서만 가중치 배열, 그 외 None.
    """
    n_classes = 2 if target == "risk_binary" else 4
    if n_classes <= 2:
        return None

    from sklearn.utils.class_weight import compute_sample_weight

    if cost_sensitive:
        # RISK_LABEL_MAP: Normal=0, Green=1, Yellow=2, Red=3
        # 저위험(Normal/Green)→cost_fp 계열, 고위험(Yellow/Red)→cost_fn 계열
        weights_by_class = {0: cost_fp, 1: cost_fp * 1.5, 2: cost_fn * 0.7, 3: cost_fn}
        return compute_sample_weight(weights_by_class, y_train)
    return compute_sample_weight("balanced", y_train)


def _build_model(model_name: str, target: str, params: dict | None,
                 use_gpu: bool = False, n_jobs: int = -1,
                 cost_sensitive: bool = False, cost_fp: float = 1.0, cost_fn: float = 5.0):
    n_classes = 2 if target == "risk_binary" else 4
    # Cost-sensitive class weight 계산
    _cw = None
    _spw = None
    if cost_sensitive and n_classes == 2:
        _spw = cost_fn / max(cost_fp, 0.01)
        _cw = {0: cost_fp, 1: cost_fn}
    elif cost_sensitive and n_classes > 2:
        # Red=3 최고비용, Normal=0 최저비용
        _cw = {0: cost_fp, 1: cost_fp * 1.5, 2: cost_fn * 0.7, 3: cost_fn}

    if model_name == "xgboost":
        from xgboost import XGBClassifier
        default = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
            "random_state": 42,
            "n_jobs": n_jobs,
        }
        if _spw is not None:
            default["scale_pos_weight"] = _spw
        if use_gpu:
            default["device"] = "cuda"
            default.pop("n_jobs", None)
        if n_classes > 2:
            default["objective"] = "multi:softprob"
            default["num_class"] = n_classes
        return XGBClassifier(**(default | (params or {})))

    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        default = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": _cw or "balanced",
            "random_state": 42,
            "n_jobs": n_jobs,
            "verbose": -1,
        }
        if use_gpu:
            default["device_type"] = "cuda"
            default.pop("n_jobs", None)
        if n_classes > 2:
            default["objective"] = "multiclass"
            default["num_class"] = n_classes
        return LGBMClassifier(**(default | (params or {})))

    elif model_name == "catboost":
        # Windows DLL 충돌 방지: catboost보다 torch를 먼저 로드해야 함
        try:
            import torch  # noqa: F401
        except (ImportError, OSError):
            pass
        from catboost import CatBoostClassifier
        default = {
            "iterations": 200,
            "depth": 6,
            "learning_rate": 0.1,
            "loss_function": "Logloss" if n_classes == 2 else "MultiClass",
            "verbose": 0,
            "random_seed": 42,
            "task_type": "GPU" if use_gpu else "CPU",
            "thread_count": n_jobs if n_jobs > 0 else -1,
        }
        if _cw is not None:
            default["class_weights"] = list(_cw.values())
        return CatBoostClassifier(**(default | (params or {})))

    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        default = {
            "n_estimators": 200,
            "max_depth": 10,
            "class_weight": _cw or "balanced",
            "random_state": 42,
            "n_jobs": n_jobs,
        }
        return RandomForestClassifier(**(default | (params or {})))

    elif model_name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        default = {
            "max_iter": 500,
            "class_weight": _cw or "balanced",
            "random_state": 42,
            "n_jobs": n_jobs,
        }
        clf = LogisticRegression(**(default | (params or {})))
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    elif model_name == "stacking":
        return _build_stacking_model(
            target, params, use_gpu, n_jobs,
            cost_sensitive, cost_fp, cost_fn,
        )

    elif model_name in ("tabnet", "gnn", "temporal_transformer"):
        from hana_app.core.phase3_models import build_phase3_model
        return build_phase3_model(
            model_name, n_classes=n_classes, params=params,
            use_gpu=use_gpu, n_jobs=n_jobs,
        )

    raise ValueError(f"알 수 없는 모델: {model_name}")


def _build_stacking_model(
    target: str, params: dict | None,
    use_gpu: bool, n_jobs: int,
    cost_sensitive: bool = False, cost_fp: float = 1.0, cost_fn: float = 5.0,
):
    """Stacking Ensemble: XGBoost + LightGBM + RandomForest → LogisticRegression."""
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression

    base_names = (params or {}).get("base_models", ["xgboost", "lightgbm", "random_forest"])
    estimators = [
        (name, _build_model(name, target, None, use_gpu, n_jobs, cost_sensitive, cost_fp, cost_fn))
        for name in base_names
    ]
    final = LogisticRegression(max_iter=500, n_jobs=n_jobs)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final,
        cv=5,
        n_jobs=n_jobs,
        passthrough=False,
    )


def _optimize_threshold(
    model, X_test, y_test,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> tuple[float, dict]:
    """비용 최적화 임계값 탐색. 이진 분류 전용."""
    y_proba = model.predict_proba(X_test)[:, 1]
    best_threshold, best_cost = 0.5, float("inf")
    results = {}
    for t_int in range(10, 91):
        t = t_int / 100.0
        y_pred_t = (y_proba >= t).astype(int)
        fp = int(((y_pred_t == 1) & (y_test == 0)).sum())
        fn = int(((y_pred_t == 0) & (y_test == 1)).sum())
        tp = int(((y_pred_t == 1) & (y_test == 1)).sum())
        cost = fp * cost_fp + fn * cost_fn
        recall = tp / max(tp + fn, 1)
        if cost < best_cost:
            best_cost = cost
            best_threshold = t
        results[f"{t:.2f}"] = {"cost": cost, "fp": fp, "fn": fn, "recall": recall}
    return float(best_threshold), {"best_cost": best_cost, "thresholds": results}


def _get_feature_importance(model, model_name: str, feature_cols: list[str] | None = None) -> pd.DataFrame:
    _cols = feature_cols if feature_cols is not None else FEATURE_COLS
    try:
        if model_name == "logistic":
            coef = model.named_steps["clf"].coef_[0]
            fi = np.abs(coef)
        elif model_name == "stacking":
            # Stacking: final_estimator의 coef 사용
            final = model.final_estimator_
            if hasattr(final, "coef_"):
                fi = np.abs(final.coef_[0])
                # stacking meta-features 수 != feature_cols 수일 수 있음
                if len(fi) != len(_cols):
                    return pd.DataFrame({"feature": _cols, "importance": [0.0] * len(_cols)})
            else:
                return pd.DataFrame({"feature": _cols, "importance": [0.0] * len(_cols)})
        elif model_name == "catboost":
            fi = model.get_feature_importance()
        elif hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        elif hasattr(model, "feature_importances"):
            # TabNet 등
            fi = model.feature_importances
        else:
            return pd.DataFrame({"feature": _cols, "importance": [0.0] * len(_cols)})

        fi_arr = np.array(fi).flatten()
        if len(fi_arr) != len(_cols):
            return pd.DataFrame({"feature": _cols, "importance": [0.0] * len(_cols)})

        df = pd.DataFrame({"feature": _cols, "importance": fi_arr})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame({"feature": _cols, "importance": [0.0] * len(_cols)})


def _save_result(result: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    model_name = result["model_name"]

    # 모델 저장
    model_path = MODELS_DIR / f"{model_name}_{ts}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(result["model"], f)

    # 결과 저장 (JSON)
    meta = {
        k: v for k, v in result.items()
        if k not in ("model", "feature_importance", "features_df")
    }
    meta["model_path"] = str(model_path)
    meta["timestamp"] = ts
    if isinstance(result.get("feature_importance"), pd.DataFrame):
        meta["feature_importance"] = result["feature_importance"].to_dict("records")

    # 위험도 분포 요약 (features_df → 요약 통계만 JSON에 저장)
    _fdf = result.get("features_df")
    if _fdf is not None and not _fdf.empty:
        try:
            meta["risk_summary"] = _fdf["risk_level"].value_counts().to_dict()
            meta["drug_count_stats"] = {
                "mean": round(float(_fdf["drug_count"].mean()), 2),
                "max": int(_fdf["drug_count"].max()),
            }
            meta["ddi_means"] = {
                c: round(float(_fdf[c].mean()), 4)
                for c in ["ddi_contraindicated", "ddi_major", "ddi_moderate", "ddi_minor"]
                if c in _fdf.columns
            }
        except Exception:
            pass

    result_path = RESULTS_DIR / f"result_{model_name}_{ts}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    result["model_path"] = str(model_path)
    result["result_path"] = str(result_path)


def list_saved_results() -> list[dict]:
    """저장된 결과 목록 반환."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for p in sorted(RESULTS_DIR.glob("result_*.json"), reverse=True):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        data["_file"] = str(p)
        results.append(data)
    return results


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)
