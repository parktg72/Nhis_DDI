"""scripts.ops.build_edi_wk_map — EDI→WK 맵 빌더 검증."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ops import build_edi_wk_map as B


def _write_xlsx(tmp_path, rows) -> Path:
    p = tmp_path / "hira.xlsx"
    pd.DataFrame(rows).to_excel(p, index=False)
    return p


def test_normalize_edi():
    assert B.normalize_edi(660700010) == "660700010"   # int 9자리
    assert B.normalize_edi("660700010") == "660700010"
    assert B.normalize_edi(123.0) == "000000123"        # float .0 제거 + zfill
    assert B.normalize_edi("") is None
    assert B.normalize_edi("ABC") is None               # 비숫자
    assert B.normalize_edi(None) is None


def test_build_maps_edi_to_wk(tmp_path):
    xlsx = _write_xlsx(tmp_path, [
        {"제품코드": 660700010, "주성분코드": "421001ATB"},
        {"제품코드": 642902720, "주성분코드": "480600ATB"},
        # 다수 제품 → 한 주성분코드 (many-to-one, 정상)
        {"제품코드": 645302132, "주성분코드": "130830ASY"},
        {"제품코드": 645302135, "주성분코드": "130830ASY"},
    ])
    df, meta = B.build_edi_wk_map(xlsx)
    m = dict(zip(df["edi_code"], df["wk_compn_cd"]))
    assert m["660700010"] == "421001ATB"
    assert m["642902720"] == "480600ATB"
    assert meta["unique_edi"] == 4
    assert meta["unique_wk"] == 3
    assert "source_sha256" in meta
    assert "efmdc_clsf_no" in df.columns


def test_build_captures_efmdc(tmp_path):
    """분류(약효분류) 컬럼이 있으면 edi→efmdc 도 매핑 (dup_efmdc 정합용)."""
    xlsx = _write_xlsx(tmp_path, [
        {"제품코드": 660700010, "주성분코드": "421001ATB", "분류": 239},
        {"제품코드": 642902720, "주성분코드": "480600ATB", "분류": 114},
    ])
    df, meta = B.build_edi_wk_map(xlsx)
    e = dict(zip(df["edi_code"], df["efmdc_clsf_no"]))
    assert e["660700010"] == "239"
    assert e["642902720"] == "114"
    assert meta["edi_with_efmdc"] == 2


def test_build_rejects_edi_conflict(tmp_path):
    """한 edi 가 서로 다른 wk 로 충돌하면 함수 위반 → ValueError."""
    xlsx = _write_xlsx(tmp_path, [
        {"제품코드": 660700010, "주성분코드": "421001ATB"},
        {"제품코드": 660700010, "주성분코드": "999999XXX"},  # 동일 edi, 다른 wk
    ])
    with pytest.raises(ValueError, match="함수 위반"):
        B.build_edi_wk_map(xlsx)


def test_build_rejects_missing_columns(tmp_path):
    xlsx = _write_xlsx(tmp_path, [{"제품코드": 1, "딴컬럼": "x"}])
    with pytest.raises(ValueError, match="컬럼 없음"):
        B.build_edi_wk_map(xlsx)


def test_build_missing_file():
    with pytest.raises(FileNotFoundError):
        B.build_edi_wk_map("nonexistent_hira.xlsx")


def test_write_creates_parquet_and_meta(tmp_path):
    df = pd.DataFrame({"edi_code": ["000000001"], "wk_compn_cd": ["100000ATB"]})
    out = B.write_edi_wk_map(df, {"unique_edi": 1}, tmp_path / "sub" / "edi_to_wk.parquet")
    assert out.exists()
    assert out.with_suffix(".meta.json").exists()
    rt = pd.read_parquet(out)
    assert list(rt.columns) == ["edi_code", "wk_compn_cd"]
