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
    # Fix B: drug_code 는 학습 vocab 네임스페이스인 EDI 코드(MCARE_DIV_CD)로 채운다.
    # WK_COMPN_CD("D30")/GNL_NM_CD("D60") 가 아니라 MCARE_DIV_CD("EDI30"/"EDI60").
    assert result[["patient_id", "drug_code", "prescription_date", "source"]].to_dict("records") == [
        {
            "patient_id": "P1",
            "drug_code": "EDI30",
            "prescription_date": "20260510",
            "source": "T30",
        },
        {
            "patient_id": "P1",
            "drug_code": "EDI60",
            "prescription_date": "20260511",
            "source": "T60",
        },
    ]
    # drug_code 와 edi_code 는 동일 EDI 네임스페이스(둘 다 MCARE_DIV_CD)여야 한다.
    assert result["drug_code"].tolist() == result["edi_code"].tolist()
    assert result["edi_code"].tolist() == ["EDI30", "EDI60"]


def test_hana_extractor_history_raises_when_edi_code_column_absent() -> None:
    # Fix B 트랩 가드: 비어있지 않은 t30/t60 프레임에 edi_code(MCARE_DIV_CD) 컬럼이 없으면
    # silent 빈 인코딩 대신 에러를 내야 한다(drug_code 가 EDI 네임스페이스이므로).
    class FakeExtractorNoEdi:
        def __init__(self) -> None:
            self.cols = {
                "t30": {
                    "patient_id": "INDI_DSCM_NO",
                    "start_date": "MDCARE_STRT_DT",
                    "drug_code": "WK_COMPN_CD",
                    "edi_code": "MCARE_DIV_CD",
                },
                "t60": {
                    "patient_id": "INDI_DSCM_NO",
                    "start_date": "MDCARE_STRT_DT",
                    "drug_code": "GNL_NM_CD",
                    "edi_code": "MCARE_DIV_CD",
                },
            }

        def fetch_t30_by_date(self, start, end, patient_ids=None):
            return pd.DataFrame({
                "INDI_DSCM_NO": ["P1"],
                "MDCARE_STRT_DT": ["20260510"],
                "WK_COMPN_CD": ["D30"],  # MCARE_DIV_CD 누락
            })

        def fetch_t60_by_date(self, start, end, patient_ids=None):
            return pd.DataFrame(columns=["INDI_DSCM_NO", "MDCARE_STRT_DT", "GNL_NM_CD"])

    provider = HANAExtractorHistoryProvider(FakeExtractorNoEdi())

    with pytest.raises(ValueError, match="missing columns"):
        provider.fetch_patient_history(
            "P1",
            reference_date=date(2026, 5, 11),
            lookback_days=30,
        )
