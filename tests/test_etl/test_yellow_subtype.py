"""_assign_yellow_subtype 단위 테스트."""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.etl.models import PatientFeatures
from scripts.etl.prescription_aggregator import (
    _assign_risk_level,
    _assign_yellow_subtype,
)


def _make(**kwargs) -> PatientFeatures:
    base = dict(
        patient_id="P001",
        window_start=date(2026, 1, 1),
        window_end=date(2026, 3, 31),
    )
    base.update(kwargs)
    return PatientFeatures(**base)


def test_yellow_subtype_field_defaults_none():
    f = _make()
    assert f.yellow_subtype is None


def test_red_patient_has_no_subtype():
    f = _make(ddi_contraindicated=1)
    _assign_risk_level(f)
    _assign_yellow_subtype(f)
    assert f.risk_level == "Red"
    assert f.yellow_subtype is None


def test_normal_patient_has_no_subtype():
    f = _make()
    _assign_risk_level(f)
    _assign_yellow_subtype(f)
    assert f.yellow_subtype is None


def test_single_ddi_major_is_y_ddi_major():
    f = _make(ddi_major=1)
    _assign_risk_level(f)
    _assign_yellow_subtype(f)
    assert f.risk_level == "Yellow"
    assert f.yellow_subtype == "Y_DDI_MAJOR"


def test_single_ddi_moderate_is_y_ddi_mod():
    f = _make(ddi_moderate=2)
    _assign_risk_level(f)
    _assign_yellow_subtype(f)
    assert f.yellow_subtype == "Y_DDI_MOD"


def test_single_dup_is_y_dup():
    f = _make(dup_same_ingredient=1)
    _assign_risk_level(f)
    _assign_yellow_subtype(f)
    assert f.yellow_subtype == "Y_DUP"


def test_single_frag_is_y_frag():
    f = _make(institution_count=3)
    _assign_risk_level(f)
    _assign_yellow_subtype(f)
    assert f.yellow_subtype == "Y_FRAG"


def test_two_triggers_is_y_double():
    """DDI_MAJOR + DUP → Y_DOUBLE (yellow 요소 2개, Red 조건 미충족)."""
    f = _make(ddi_major=1, dup_same_ingredient=1)
    _assign_risk_level(f)
    _assign_yellow_subtype(f)
    assert f.risk_level == "Yellow"
    assert f.yellow_subtype == "Y_DOUBLE"


def test_three_triggers_is_y_triple():
    """DDI_MAJOR + DDI_MOD + FRAG → Y_TRIPLE (yellow 요소 3개)."""
    f = _make(ddi_major=1, ddi_moderate=2, institution_count=3)
    _assign_risk_level(f)
    _assign_yellow_subtype(f)
    assert f.yellow_subtype == "Y_TRIPLE"


def test_four_triggers_is_y_triple():
    """yellow 요소 4개(DDI_MAJOR+DDI_MOD+DUP+FRAG)도 Y_TRIPLE(3개+)."""
    f = _make(ddi_major=1, ddi_moderate=2, dup_same_ingredient=1, institution_count=3)
    _assign_risk_level(f)
    _assign_yellow_subtype(f)
    assert f.yellow_subtype == "Y_TRIPLE"


def test_count_label_excluded_when_red():
    """Red 조건 충족 시 계수 라벨 아닌 None (Red 가 흡수)."""
    f = _make(ddi_contraindicated=1, ddi_major=1, dup_same_ingredient=1)
    _assign_risk_level(f)
    _assign_yellow_subtype(f)
    assert f.risk_level == "Red"
    assert f.yellow_subtype is None


def test_edge_yellow_without_trigger_is_y_other(caplog):
    """규칙 드리프트 엣지: risk_level=Yellow 인데 trigger 가 0개 → Y_OTHER 로그."""
    import logging
    f = _make()
    f.risk_level = "Yellow"  # 의도적 오염
    with caplog.at_level(logging.WARNING):
        _assign_yellow_subtype(f)
    assert f.yellow_subtype == "Y_OTHER"
    assert any("yellow_without_trigger" in r.message for r in caplog.records)


def test_yellow_subtype_written_to_parquet(tmp_path):
    """feature_writer 결과를 parquet 으로 쓰고 다시 읽어도 yellow_subtype 이 보존되는지.

    None → NaN 변환은 pandas 의 기본 동작. 다운스트림 비교는 pd.isna() 사용 필요.
    """
    import pandas as pd
    from scripts.etl.feature_writer import features_to_df

    f1 = _make(patient_id="P001", ddi_major=1)
    _assign_risk_level(f1); _assign_yellow_subtype(f1)
    f2 = _make(patient_id="P002", ddi_contraindicated=1)
    _assign_risk_level(f2); _assign_yellow_subtype(f2)

    df = features_to_df([f1, f2])
    assert "yellow_subtype" in df.columns

    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    rt = pd.read_parquet(path)

    row1 = rt.loc[rt["patient_id"] == "P001"].iloc[0]
    assert row1["yellow_subtype"] == "Y_DDI_MAJOR"

    row2 = rt.loc[rt["patient_id"] == "P002"].iloc[0]
    assert pd.isna(row2["yellow_subtype"])   # None → NaN after parquet roundtrip


def test_ml_runner_row_has_yellow_subtype():
    """ml_runner._patient_features_to_row 가 yellow_subtype 을 포함하는지."""
    from hana_app.core.ml_runner import _patient_features_to_row

    f = _make(patient_id="P001", ddi_major=1)
    _assign_risk_level(f); _assign_yellow_subtype(f)
    row = _patient_features_to_row(f)
    assert row["yellow_subtype"] == "Y_DDI_MAJOR"

    f2 = _make(patient_id="P002", ddi_contraindicated=1)
    _assign_risk_level(f2); _assign_yellow_subtype(f2)
    row2 = _patient_features_to_row(f2)
    assert row2["yellow_subtype"] is None
