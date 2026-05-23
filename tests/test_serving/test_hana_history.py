from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from serving.hana_history import (
    HANAExtractorHistoryProvider,
    InMemoryHANAHistoryProvider,
    validate_history_frame,
)


def test_in_memory_history_provider_filters_patient_and_lookback() -> None:
    history = pd.DataFrame({
        "patient_id": ["P1", "P1", "P2"],
        "drug_code": ["D1", "D2", "D3"],
        "prescription_date": ["20260510", "20260401", "20260510"],
    })
    provider = InMemoryHANAHistoryProvider(history)

    result = provider.fetch_patient_history(
        "P1",
        reference_date=date(2026, 5, 11),
        lookback_days=30,
    )

    assert result["patient_id"].tolist() == ["P1"]
    assert result["drug_code"].tolist() == ["D1"]


def test_validate_history_frame_rejects_missing_required_column() -> None:
    history = pd.DataFrame({
        "patient_id": ["P1"],
        "prescription_date": ["20260510"],
    })

    with pytest.raises(ValueError, match="drug_code"):
        validate_history_frame(history)


def test_in_memory_history_provider_rejects_invalid_dates() -> None:
    history = pd.DataFrame({
        "patient_id": ["P1"],
        "drug_code": ["D1"],
        "prescription_date": ["2026-99-99"],
    })
    provider = InMemoryHANAHistoryProvider(history)

    with pytest.raises(ValueError, match="prescription_date"):
        provider.fetch_patient_history(
            "P1",
            reference_date=date(2026, 5, 11),
            lookback_days=30,
        )


def test_hana_extractor_history_provider_normalizes_t30_and_t60() -> None:
    class FakeExtractor:
        def __init__(self) -> None:
            self.calls = []
            self.cols = {
                "t30": {
                    "patient_id": "INDI_DSCM_NO",
                    "start_date": "MDCARE_STRT_DT",
                    "drug_code": "WK_COMPN_CD",
                    "drug_code_alt": "RVSN_WK_COMPN_CD",
                    "edi_code": "MCARE_DIV_CD",
                    "total_days": "TOT_MCNT",
                },
                "t60": {
                    "patient_id": "INDI_DSCM_NO",
                    "start_date": "MDCARE_STRT_DT",
                    "drug_code": "GNL_NM_CD",
                    "drug_code_alt": "RVSN_WK_COMPN_CD",
                    "edi_code": "MCARE_DIV_CD",
                    "total_days": "TOT_MCNT",
                },
            }

        def fetch_t30_by_date(self, start, end, patient_ids=None):
            self.calls.append(("t30", start, end, patient_ids))
            return pd.DataFrame({
                "INDI_DSCM_NO": ["P1"],
                "MDCARE_STRT_DT": ["20260510"],
                "WK_COMPN_CD": ["D30"],
                "RVSN_WK_COMPN_CD": ["D30_ALT"],
                "MCARE_DIV_CD": ["EDI30"],
                "TOT_MCNT": [7],
            })

        def fetch_t60_by_date(self, start, end, patient_ids=None):
            self.calls.append(("t60", start, end, patient_ids))
            return pd.DataFrame({
                "INDI_DSCM_NO": ["P1"],
                "MDCARE_STRT_DT": ["20260511"],
                "GNL_NM_CD": ["D60"],
                "RVSN_WK_COMPN_CD": ["D60_ALT"],
                "MCARE_DIV_CD": ["EDI60"],
                "TOT_MCNT": [14],
            })

    extractor = FakeExtractor()
    provider = HANAExtractorHistoryProvider(extractor)

    result = provider.fetch_patient_history(
        "P1",
        reference_date=date(2026, 5, 11),
        lookback_days=30,
    )

    assert extractor.calls == [
        ("t30", date(2026, 4, 12), date(2026, 5, 11), ["P1"]),
        ("t60", date(2026, 4, 12), date(2026, 5, 11), ["P1"]),
    ]
    assert result[["patient_id", "drug_code", "prescription_date", "source"]].to_dict("records") == [
        {
            "patient_id": "P1",
            "drug_code": "D30",
            "prescription_date": "20260510",
            "source": "T30",
        },
        {
            "patient_id": "P1",
            "drug_code": "D60",
            "prescription_date": "20260511",
            "source": "T60",
        },
    ]
