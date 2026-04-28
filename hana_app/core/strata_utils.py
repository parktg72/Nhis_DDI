"""층화 샘플링 공용 유틸 — 연령 band 정의의 single source of truth.

세 경로(HANA SQL `_AGE_CASE_SQL`, pandas `pd.cut`, SAS row-loop) 가 모두
동일 경계를 사용하도록 보장한다. 경계 변경 시 `hana_etl._AGE_CASE_SQL`
의 하드코딩 값과 `tests/test_hana_app/test_strata_consistency.py` 도
함께 갱신할 것.
"""
from __future__ import annotations

# (n+1) 길이 left-closed/right-open bins. pd.cut(..., right=False) 호환.
_DEFAULT_AGE_BINS: list[int] = [0, 20, 40, 60, 75, 200]
_DEFAULT_AGE_LABELS: list[str] = ["0-19", "20-39", "40-59", "60-74", "75+"]
_UNKNOWN_LABEL = "unknown"


def byear_to_age_band(byear: object, ref_year: int) -> str:
    """raw BYEAR 값에서 연령 band 레이블 반환 (단일값 분기).

    SAS streaming row-loop 처럼 pd.cut 호출이 부적합한 곳에서 사용.
    None/빈문자/파싱불가/음수 → 'unknown'.
    """
    if byear is None:
        return _UNKNOWN_LABEL
    s = str(byear).strip()
    if not s:
        return _UNKNOWN_LABEL
    try:
        age = int(ref_year) - int(float(s))
    except (TypeError, ValueError):
        return _UNKNOWN_LABEL
    if age < 0:
        return _UNKNOWN_LABEL
    for i, label in enumerate(_DEFAULT_AGE_LABELS):
        if age < _DEFAULT_AGE_BINS[i + 1]:
            return label
    return _UNKNOWN_LABEL
