"""
가명처리 (Pseudonymization)
SHA-256 기반 단방향 해시. SALT는 환경변수 ETL_PSEUDO_SALT 로 주입.
"""
from __future__ import annotations

import hashlib
import os
import warnings

import pandas as pd


_DEFAULT_SALT = "ddi_model_dev_salt_2024"
_SALT: str = os.environ.get("ETL_PSEUDO_SALT", "")


def _get_salt() -> str:
    if _SALT:
        return _SALT
    warnings.warn(
        "환경변수 ETL_PSEUDO_SALT 미설정. 개발용 기본 SALT 사용. 운영 환경에서는 반드시 설정하세요.",
        stacklevel=3,
    )
    return _DEFAULT_SALT


def hash_id(value: str, salt: str | None = None) -> str:
    """단일 ID 해시 (SHA-256 hex, 16자리 truncation)."""
    s = salt if salt is not None else _get_salt()
    raw = f"{s}:{value}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def pseudonymize_column(series: pd.Series, salt: str | None = None) -> pd.Series:
    """Series 전체 해시 변환. NaN은 NaN 유지."""
    s = salt if salt is not None else _get_salt()
    return series.where(series.isna(), series.astype(str).map(lambda v: hash_id(v, s)))


def pseudonymize_dataframe(
    df: pd.DataFrame,
    id_columns: list[str],
    salt: str | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    DataFrame 내 지정 컬럼들을 가명 처리.

    Parameters
    ----------
    df : 원본 DataFrame
    id_columns : 가명처리할 컬럼명 목록
    salt : SALT (None이면 환경변수/기본값 사용)
    inplace : True면 원본 수정, False면 복사본 반환
    """
    out = df if inplace else df.copy()
    s = salt if salt is not None else _get_salt()
    for col in id_columns:
        if col in out.columns:
            out[col] = pseudonymize_column(out[col], s)
    return out


# 표준 가명처리 대상 컬럼 (테이블별)
# 주의: NHIS 반출 데이터에서 INDI_DSCM_NO는 이미 가명화된 번호임.
# 이 매핑은 자체 DB에서 원본 ID가 있는 경우(원내 전처리 단계)에만 사용.
PSEUDO_COLUMNS: dict[str, list[str]] = {
    "T20": ["INDI_DSCM_NO", "MDCARE_SYM"],  # 개인식별번호, 요양기관기호
    "T30": ["INDI_DSCM_NO"],                 # T20 merge 전 원본 INDI_DSCM_NO 가명처리 필수
    "T40": ["INDI_DSCM_NO"],
    "T60": ["INDI_DSCM_NO"],
    "YOYANG": ["REPR_INDI_DSCM_NO", "MDCARE_SYM"],  # 대표자 개인식별번호, 요양기관기호(T20 조인 키)
    # 하위 호환
    "T50": ["MDCARE_SYM"],
}
