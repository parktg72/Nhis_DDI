"""세 경로(HANA SQL / pd.cut / SAS row-loop)의 연령 band 정의 일치성 보증.

`strata_utils._DEFAULT_AGE_BINS` 가 single source of truth. 경계 변경 시
`hana_etl._AGE_CASE_SQL` 의 하드코딩 값도 같이 갱신해야 함 — 이 테스트가
잠금장치 역할.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from hana_app.core.hana_etl import _AGE_CASE_SQL
from hana_app.core.strata_utils import (
    _DEFAULT_AGE_BINS,
    _DEFAULT_AGE_LABELS,
    byear_to_age_band,
)


def test_hana_sql_breakpoints_match_default_bins():
    """`_AGE_CASE_SQL` 의 `< N` 분기점이 `_DEFAULT_AGE_BINS` 내부 경계와 일치."""
    breaks_in_sql = [int(m) for m in re.findall(r"<\s*(\d+)", _AGE_CASE_SQL)]
    # 첫 분기 `< 0` 은 음수 unknown 처리용. 나머지가 실제 band 경계.
    assert breaks_in_sql[0] == 0, "expected leading < 0 for negative-age unknown branch"
    assert breaks_in_sql[1:] == _DEFAULT_AGE_BINS[1:-1], (
        f"HANA SQL breaks {breaks_in_sql[1:]} != "
        f"_DEFAULT_AGE_BINS internal {_DEFAULT_AGE_BINS[1:-1]}"
    )


def test_hana_sql_labels_match_default_labels():
    for label in _DEFAULT_AGE_LABELS:
        assert f"'{label}'" in _AGE_CASE_SQL, f"label '{label}' missing from _AGE_CASE_SQL"


@pytest.mark.parametrize(
    "byear,ref_year,expected",
    [
        (2024, 2024, "0-19"),    # 0세
        (2005, 2024, "0-19"),    # 19세
        (2004, 2024, "20-39"),   # 20세 (band 경계)
        (1985, 2024, "20-39"),   # 39세
        (1984, 2024, "40-59"),   # 40세
        (1965, 2024, "40-59"),   # 59세
        (1964, 2024, "60-74"),   # 60세
        (1950, 2024, "60-74"),   # 74세
        (1949, 2024, "75+"),     # 75세
        (1900, 2024, "75+"),     # 124세 (NHIS 데이터 상한 내)
        (None, 2024, "unknown"),
        ("", 2024, "unknown"),
        ("abc", 2024, "unknown"),
        (2030, 2024, "unknown"), # 음수 나이
    ],
)
def test_byear_to_age_band_matches_pd_cut(byear, ref_year, expected):
    """`byear_to_age_band` 가 `pd.cut(_DEFAULT_AGE_BINS, right=False)` 와 동일 출력."""
    # 1) 단일값 분기 결과
    direct = byear_to_age_band(byear, ref_year)
    assert direct == expected

    # 2) pd.cut 경로 결과 — None/파싱 불가 케이스는 NaN → 'unknown' 로 보정 후 비교
    age_series = pd.Series(
        [ref_year - int(float(byear))] if (byear not in (None, "") and str(byear).strip().lstrip("-").replace(".", "", 1).isdigit()) else [pd.NA],
        dtype="Float64",
    )
    cut = pd.cut(
        age_series, bins=_DEFAULT_AGE_BINS, labels=_DEFAULT_AGE_LABELS, right=False
    ).astype(str).fillna("unknown")
    pd_path = cut.iloc[0]
    if expected == "unknown":
        # pd.cut: 음수/NaN/범위초과 → 'nan' 문자열 또는 NaN. 'unknown' 으로 정규화 후 일치 확인.
        assert pd_path in ("unknown", "nan")
    else:
        assert pd_path == expected, f"pd.cut path mismatch: {pd_path} != {expected}"
