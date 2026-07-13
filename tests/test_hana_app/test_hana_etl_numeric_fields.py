from __future__ import annotations

import pandas as pd
import pytest

from hana_app.core.hana_etl import HANAExtractor

_T20_COLS = {
    "bill_no": "CMN_KEY",
    "patient_id": "INDI_DSCM_NO",
    "institution_id": "MDCARE_SYM",
    "start_date": "MDCARE_STRT_DT",
    "sex": "SEX_TYPE",
    "age_id": "SUJIN_POTM_AGE_ID",
    "institution_type": "YOYANG_CLSFC_CD",
}

_T60_COLS = {
    "bill_no": "CMN_KEY",
    "patient_id": "INDI_DSCM_NO",
    "institution_id": "MDCARE_SYM",
    "start_date": "MDCARE_STRT_DT",
    "drug_code": "GNL_NM_CD",
    "drug_code_alt": "RVSN_WK_COMPN_CD",
    "edi_code": "MCARE_DIV_CD",
    "dose_once": "MPRSC_TIME1_TUYAK_CPCT",
    "dose_freq": "MPRSC_DD1_TUYAK_CPCT",
    "total_days": "TOT_MCNT",
    "sick_code": "SICK_SYM1",
}


def _extractor() -> HANAExtractor:
    return HANAExtractor(
        conn=None,
        table_cfg={},
        col_cfg={"t20": _T20_COLS, "t60": _T60_COLS},
    )


def _t20_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CMN_KEY": ["41639422024041  0000300202408"],
            "SEX_TYPE": ["1"],
            "SUJIN_POTM_AGE_ID": [70],
            "YOYANG_CLSFC_CD": ["01"],
        }
    ).set_index("CMN_KEY")


def test_t60_to_records_reports_total_days_mapping_drift() -> None:
    """A CMN_KEY-shaped value in TOT_MCNT should fail with actionable context.

    The production traceback showed a 29-character CMN_KEY value being parsed as
    T60 total_days.  The generic ``int(...)`` error hides the likely config/copy
    drift, so the ETL should point operators at the T60 total_days mapping.
    """
    bad_key = "41639422024041  0000300202408"
    t60 = pd.DataFrame(
        {
            "CMN_KEY": [bad_key],
            "INDI_DSCM_NO": ["P001"],
            "MDCARE_STRT_DT": ["20240401"],
            "GNL_NM_CD": ["D0001"],
            "RVSN_WK_COMPN_CD": [""],
            "MCARE_DIV_CD": ["EDI0001"],
            "MPRSC_TIME1_TUYAK_CPCT": [1],
            "MPRSC_DD1_TUYAK_CPCT": [3],
            "TOT_MCNT": [bad_key],
            "SICK_SYM1": ["G30"],
            "MDCARE_SYM": ["INST001"],
        }
    )

    with pytest.raises(ValueError) as exc:
        _extractor()._t60_to_records(t60, _t20_index())

    msg = str(exc.value)
    assert "T60.TOT_MCNT" in msg
    assert "CMN_KEY-shaped" in msg
    assert "hana_config.json" in msg
