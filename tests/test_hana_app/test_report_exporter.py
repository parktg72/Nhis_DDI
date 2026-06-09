"""tests for hana_app.core.report_exporter"""
import io
import pytest
import pandas as pd

from hana_app.core.report_exporter import (
    build_csv_bytes,
    build_docx_bytes,
    DOCX_AVAILABLE,
    _build_reason,
    _effective_label,
)


def _row(**kw):
    defaults = {
        "patient_id": "P001",
        "risk_level": "Yellow",
        "yellow_subtype": "",
        "drug_count": 5,
        "ddi_contraindicated": 0,
        "ddi_major": 0,
        "ddi_moderate": 0,
        "ddi_minor": 0,
        "dup_same_ingredient": 0,
        "institution_count": 1,
        "has_high_risk_drug": 0,
        "has_renal_risk_drug": 0,
        "has_hepatic_risk_drug": 0,
        "triple_whammy": 0,
    }
    return {**defaults, **kw}


# ── CSV ─────────────────────────────────────────────────────────────────────

def test_csv_bytes_red_only():
    df = pd.DataFrame([
        _row(patient_id="P001", risk_level="Red", ddi_contraindicated=2),
        _row(patient_id="P002", risk_level="Green"),
    ])
    b = build_csv_bytes(df)
    assert b[:3] == b"\xef\xbb\xbf", "must start with UTF-8 BOM"
    text = b.decode("utf-8-sig")
    assert "P001" in text
    assert "P002" not in text
    assert "금기 DDI 2건" in text
    assert "즉각 개입" in text


def test_csv_bytes_all_labels():
    df = pd.DataFrame([
        _row(patient_id="R1", risk_level="Red", ddi_contraindicated=1),
        _row(patient_id="M1", risk_level="Yellow", yellow_subtype="Y_DDI_MAJOR", ddi_major=3),
        _row(patient_id="T1", risk_level="Yellow", yellow_subtype="Y_TRIPLE", drug_count=11),
        _row(patient_id="G1", risk_level="Green"),
    ])
    b = build_csv_bytes(df)
    text = b.decode("utf-8-sig")
    assert "R1" in text and "M1" in text and "T1" in text
    assert "G1" not in text
    assert "즉각 개입" in text
    assert "약사 전화" in text
    assert "문자 안내" in text
    assert "금기 DDI 1건" in text
    assert "중증 DDI 3건" in text


def test_csv_bytes_no_target_rows():
    df = pd.DataFrame([_row(patient_id="G1", risk_level="Green")])
    b = build_csv_bytes(df)
    text = b.decode("utf-8-sig")
    assert "G1" not in text
    # header only (환자ID column present)
    assert "환자ID" in text


def test_csv_utf8_bom():
    df = pd.DataFrame([_row(risk_level="Red")])
    b = build_csv_bytes(df)
    assert b[:3] == b"\xef\xbb\xbf"


def test_build_reason_red():
    r = _row(risk_level="Red", ddi_contraindicated=3)
    assert _build_reason(r) == "금기 DDI 3건"


def test_build_reason_major():
    r = _row(risk_level="Yellow", yellow_subtype="Y_DDI_MAJOR", ddi_major=5)
    assert _build_reason(r) == "중증 DDI 5건"


def test_build_reason_triple_multidrug():
    r = _row(risk_level="Yellow", yellow_subtype="Y_TRIPLE", drug_count=12)
    reason = _build_reason(r)
    assert "3중위험" in reason
    assert "다약제(12종)" in reason


# ── DOCX ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not DOCX_AVAILABLE, reason="python-docx not installed")
def test_docx_bytes_returns_bytes():
    last_result = {"model_name": "test_model", "metrics": {"f1_macro": 0.85}}
    b = build_docx_bytes(last_result)
    assert isinstance(b, bytes)
    assert b[:4] == b"PK\x03\x04", "must be a ZIP (DOCX magic)"


@pytest.mark.skipif(not DOCX_AVAILABLE, reason="python-docx not installed")
def test_docx_missing_ddi_means():
    last_result = {"model_name": "m", "metrics": {}}
    b = build_docx_bytes(last_result)
    assert b[:4] == b"PK\x03\x04"


@pytest.mark.skipif(not DOCX_AVAILABLE, reason="python-docx not installed")
def test_docx_empty_feature_importance():
    last_result = {"model_name": "m", "metrics": {}, "feature_importance": []}
    b = build_docx_bytes(last_result)
    assert b[:4] == b"PK\x03\x04"


@pytest.mark.skipif(not DOCX_AVAILABLE, reason="python-docx not installed")
def test_docx_with_features_df():
    df = pd.DataFrame([
        _row(patient_id="R1", risk_level="Red"),
        _row(patient_id="Y1", risk_level="Yellow", yellow_subtype="Y_DDI_MAJOR"),
        _row(patient_id="G1", risk_level="Green"),
    ])
    last_result = {
        "model_name": "hier",
        "metrics": {"f1_macro": 0.9},
        "ddi_means": {"ddi_major": 1.2, "ddi_contraindicated": 0.1},
        "drug_count_stats": {"mean": 6.5, "max": 14},
    }
    b = build_docx_bytes(last_result, df)
    assert b[:4] == b"PK\x03\x04"
