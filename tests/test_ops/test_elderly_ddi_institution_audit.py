from __future__ import annotations

from datetime import date

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _clear_ddi_lookup_cache_between_tests() -> None:
    # The reused MODE_11_hana DDI source caches matrix lookups by DataFrame id.
    # Unit tests create many short-lived matrices, so isolate tests from id reuse.
    from scripts.etl import prescription_aggregator

    prescription_aggregator._ddi_lookup_cache.clear()


class _FakeDrugMaster:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping

    def get_ddi_ids(self, wk_compn_cd: str) -> list[str]:
        return self.mapping.get(str(wk_compn_cd), [])


def test_pdf_text_code_extraction_preserves_trailing_star_display() -> None:
    from scripts.ops.elderly_ddi_institution_audit import (
        normalize_disease_codes_from_pdf_text,
    )

    text = "치매 F00*. 예시 F10.2*. 혈관성치매 F01 중복 F00*,"

    assert normalize_disease_codes_from_pdf_text(text) == ("F00*", "F10.2*", "F01")


DISEASE_CODES = ("F00*", "I63", "R25.1")


def test_disease_code_matching_normalizes_stars_dots_and_prefixes() -> None:
    from scripts.ops.elderly_ddi_institution_audit import patient_has_disease_code

    assert patient_has_disease_code(["F00.1"], DISEASE_CODES) is True
    assert patient_has_disease_code([" i63.9 "], DISEASE_CODES) is True
    assert patient_has_disease_code(["R25.1"], DISEASE_CODES) is True
    assert patient_has_disease_code(["R25.2", "J10"], DISEASE_CODES) is False


def test_select_target_patients_includes_age_65_and_under_65_with_disease_code() -> None:
    from scripts.ops.elderly_ddi_institution_audit import select_target_patients

    eligibility = pd.DataFrame(
        [
            {"patient_id": "P65", "age": 65},
            {"patient_id": "P64_DISEASE", "age": 64},
            {"patient_id": "P64_NO_DISEASE", "age": 64},
            {"patient_id": "P_BAD_AGE", "age": "unknown"},
            {"patient_id": None, "age": 70},
        ]
    )
    diagnoses = pd.DataFrame(
        [
            {"patient_id": "P64_DISEASE", "sick_code": "I63.9"},
            {"patient_id": "P64_NO_DISEASE", "sick_code": "J10"},
        ]
    )

    cohort, preflight = select_target_patients(
        eligibility,
        diagnoses,
        disease_codes=DISEASE_CODES,
        min_age=65,
    )

    assert cohort["patient_id"].tolist() == ["P65", "P64_DISEASE"]
    assert preflight.ok is True
    assert preflight.details["age_included_count"] == 1
    assert preflight.details["under_min_age_disease_code_included_count"] == 1
    assert preflight.details["under_min_age_without_disease_code_count"] == 1
    assert preflight.details["invalid_age_count"] == 1
    assert preflight.details["invalid_patient_id_count"] == 1
    assert "excluded_invalid_age" in preflight.warnings


def test_select_target_patients_without_diagnoses_is_provisional_age_only() -> None:
    from scripts.ops.elderly_ddi_institution_audit import select_target_patients

    eligibility = pd.DataFrame(
        [
            {"patient_id": "P65", "age": 65},
            {"patient_id": "P64", "age": 64},
        ]
    )

    cohort, preflight = select_target_patients(
        eligibility,
        diagnoses=None,
        disease_codes=DISEASE_CODES,
        min_age=65,
    )

    assert cohort["patient_id"].tolist() == ["P65"]
    assert preflight.ok is True
    assert "diagnosis_source_unavailable_under_min_age_not_included" in preflight.warnings
    assert preflight.details["diagnosis_source_status"] == "unavailable"
    assert preflight.details["under_min_age_disease_code_included_count"] == 0


def test_exclude_deceased_patients_requires_death_data_by_default() -> None:
    from scripts.ops.elderly_ddi_institution_audit import (
        DeathDataRequiredError,
        exclude_deceased_patients,
    )

    cohort = pd.DataFrame([{"patient_id": "P1", "age": 70}])

    with pytest.raises(DeathDataRequiredError):
        exclude_deceased_patients(cohort, None, audit_end=date(2024, 12, 31))


def test_exclude_deceased_patients_removes_deaths_on_or_before_audit_end() -> None:
    from scripts.ops.elderly_ddi_institution_audit import exclude_deceased_patients

    cohort = pd.DataFrame(
        [
            {"patient_id": "P_DEAD", "age": 70},
            {"patient_id": "P_ALIVE", "age": 71},
            {"patient_id": "P_FUTURE", "age": 72},
        ]
    )
    deaths = pd.DataFrame(
        [
            {"patient_id": "P_DEAD", "DTH_ASSMD_DT": 20241231},
            {"patient_id": "P_FUTURE", "DTH_ASSMD_DT": 20250101},
            {"patient_id": "P_NO_DATE", "DTH_ASSMD_DT": None},
        ]
    )

    filtered, preflight = exclude_deceased_patients(
        cohort,
        deaths,
        audit_end=date(2024, 12, 31),
    )

    assert filtered["patient_id"].tolist() == ["P_ALIVE", "P_FUTURE"]
    assert preflight.ok is True
    assert preflight.details["excluded_deceased_count"] == 1
    assert preflight.details["death_rows_with_parseable_date_count"] == 2


def test_exclude_deceased_patients_reports_unparseable_death_rows_without_excluding_by_default() -> None:
    from scripts.ops.elderly_ddi_institution_audit import exclude_deceased_patients

    cohort = pd.DataFrame([{"patient_id": "P_BAD_DEATH_DATE", "age": 70}])
    deaths = pd.DataFrame([{"patient_id": "P_BAD_DEATH_DATE", "DTH_ASSMD_DT": "not-a-date"}])

    filtered, preflight = exclude_deceased_patients(
        cohort,
        deaths,
        audit_end=date(2024, 12, 31),
    )

    assert filtered["patient_id"].tolist() == ["P_BAD_DEATH_DATE"]
    assert "death_rows_without_parseable_date" in preflight.warnings
    assert preflight.details["death_exclusion_status"] == "complete_with_warnings"
    assert preflight.details["death_rows_without_parseable_date_count"] == 1


def test_build_institution_name_map_filters_year_and_normalizes_fields() -> None:
    from scripts.ops.elderly_ddi_institution_audit import build_institution_name_map

    master = pd.DataFrame(
        [
            {"STD_YYYY": "2024", "MDCARE_SYM": " H001 ", "INST_NM": " Alpha Clinic "},
            {"STD_YYYY": "2023", "MDCARE_SYM": "H002", "INST_NM": "Old Year Clinic"},
            {"STD_YYYY": 2024.0, "MDCARE_SYM": "H003", "INST_NM": "Gamma Hospital"},
        ]
    )

    assert build_institution_name_map(master, std_year="2024") == {
        "H001": "Alpha Clinic",
        "H003": "Gamma Hospital",
    }


def test_build_institution_name_map_duplicate_same_year_keeps_last_row() -> None:
    from scripts.ops.elderly_ddi_institution_audit import build_institution_name_map

    master = pd.DataFrame(
        [
            {"STD_YYYY": "2024", "MDCARE_SYM": "H001", "INST_NM": "Old Name"},
            {"STD_YYYY": "2024", "MDCARE_SYM": "H001", "INST_NM": "New Name"},
            {"STD_YYYY": "2024", "MDCARE_SYM": "H002", "INST_NM": "Stable Name"},
        ]
    )

    assert build_institution_name_map(master, std_year="2024") == {
        "H001": "New Name",
        "H002": "Stable Name",
    }


def test_build_institution_name_map_duplicate_blank_last_removes_mapping() -> None:
    from scripts.ops.elderly_ddi_institution_audit import build_institution_name_map

    master = pd.DataFrame(
        [
            {"STD_YYYY": "2024", "MDCARE_SYM": "H001", "INST_NM": "Old Name"},
            {"STD_YYYY": "2024", "MDCARE_SYM": "H001", "INST_NM": " "},
        ]
    )

    assert build_institution_name_map(master, std_year="2024") == {}


def test_build_institution_name_map_requires_master_columns() -> None:
    from scripts.ops.elderly_ddi_institution_audit import build_institution_name_map

    with pytest.raises(ValueError, match="institution master"):
        build_institution_name_map(pd.DataFrame([{"STD_YYYY": "2024"}]), std_year="2024")


def test_attach_institution_names_preserves_unknown_ids_with_none_names() -> None:
    from scripts.ops.elderly_ddi_institution_audit import attach_institution_names

    events = pd.DataFrame(
        [
            {"institution_a_id": " H001 ", "institution_b_id": "UNKNOWN"},
            {"institution_a_id": None, "institution_b_id": "H002"},
        ]
    )

    named, preflight = attach_institution_names(
        events,
        {"H001": "Alpha Clinic", "H002": "Beta Hospital"},
        id_columns=("institution_a_id", "institution_b_id"),
    )

    assert named.to_dict("records") == [
        {
            "institution_a_id": "H001",
            "institution_b_id": "UNKNOWN",
            "institution_a_name": "Alpha Clinic",
            "institution_b_name": None,
        },
        {
            "institution_a_id": None,
            "institution_b_id": "H002",
            "institution_a_name": None,
            "institution_b_name": "Beta Hospital",
        },
    ]
    assert preflight.ok is True
    assert preflight.warnings == ("unmatched_institution_names",)
    assert preflight.details["unmatched_institution_name_count"] == 1
    assert preflight.details["unmatched_institution_ids"] == ["UNKNOWN"]


def _raw_record(
    wk: str,
    start: object,
    end: object,
    *,
    patient_id: str = "P1",
    institution_id: object = "H001",
    bill_no: str = "B1",
    edi_code: str = "E1",
    source: str = "T30",
) -> dict:
    return {
        "patient_id": patient_id,
        "institution_id": institution_id,
        "bill_no": bill_no,
        "wk_compn_cd": wk,
        "edi_code": edi_code,
        "gnl_nm_cd": None,
        "efmdc_clsf_no": None,
        "start_date": start,
        "end_date": end,
        "total_days": 1,
        "dose_once": 1.0,
        "dose_freq": 1,
        "sick_code": "I63",
        "sex": "1",
        "age_id": "065",
        "institution_type": "01",
        "source": source,
    }


def test_records_to_prescriptions_uses_local_raw_columns_and_skips_blank_wk() -> None:
    from scripts.ops.elderly_ddi_institution_audit import records_to_prescriptions

    records = pd.DataFrame(
        [
            _raw_record(" WK_A ", "2024-07-01", "2024-07-10", institution_id=" H001 "),
            _raw_record(" ", "2024-07-01", "2024-07-10", institution_id="H002"),
        ]
    )

    prescriptions = records_to_prescriptions(records)

    assert len(prescriptions) == 1
    assert prescriptions[0].patient_id == "P1"
    assert prescriptions[0].institution_id == "H001"
    assert prescriptions[0].wk_compn_cd == "WK_A"
    assert prescriptions[0].start_date == date(2024, 7, 1)
    assert prescriptions[0].end_date == date(2024, 7, 10)


def test_attributed_overlap_exact_boundary_preserves_cross_institution_pair() -> None:
    from scripts.ops.elderly_ddi_institution_audit import (
        calculate_attributed_overlaps_for_patient,
        records_to_prescriptions,
    )

    prescriptions = records_to_prescriptions(
        pd.DataFrame(
            [
                _raw_record("WK_A", date(2024, 7, 1), date(2024, 7, 10), institution_id="H001", source="T30"),
                _raw_record("WK_B", date(2024, 7, 4), date(2024, 7, 10), institution_id="H002", source="T60"),
            ]
        )
    )

    pairs = calculate_attributed_overlaps_for_patient(prescriptions, min_overlap=7)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.patient_id == "P1"
    assert {pair.drug_a_wk_compn, pair.drug_b_wk_compn} == {"WK_A", "WK_B"}
    assert {pair.institution_a_id, pair.institution_b_id} == {"H001", "H002"}
    assert {pair.source_a, pair.source_b} == {"T30", "T60"}
    assert pair.overlap_start == date(2024, 7, 4)
    assert pair.overlap_end == date(2024, 7, 10)
    assert pair.overlap_days == 7
    assert pair.same_institution is False


def test_attributed_overlap_six_day_overlap_is_below_threshold() -> None:
    from scripts.ops.elderly_ddi_institution_audit import (
        calculate_attributed_overlaps_for_patient,
        records_to_prescriptions,
    )

    prescriptions = records_to_prescriptions(
        pd.DataFrame(
            [
                _raw_record("WK_A", date(2024, 7, 1), date(2024, 7, 9)),
                _raw_record("WK_B", date(2024, 7, 4), date(2024, 7, 9)),
            ]
        )
    )

    assert calculate_attributed_overlaps_for_patient(prescriptions, min_overlap=7) == []


def test_attributed_overlap_excludes_same_wk_pairs() -> None:
    from scripts.ops.elderly_ddi_institution_audit import (
        calculate_attributed_overlaps_for_patient,
        records_to_prescriptions,
    )

    prescriptions = records_to_prescriptions(
        pd.DataFrame(
            [
                _raw_record("WK_A", date(2024, 7, 1), date(2024, 7, 20), institution_id="H001"),
                _raw_record("WK_A", date(2024, 7, 5), date(2024, 7, 20), institution_id="H002"),
            ]
        )
    )

    assert calculate_attributed_overlaps_for_patient(prescriptions) == []


def test_attributed_overlap_same_institution_true_and_missing_side_none() -> None:
    from scripts.ops.elderly_ddi_institution_audit import (
        calculate_attributed_overlaps_for_patient,
        records_to_prescriptions,
    )

    same_inst = records_to_prescriptions(
        pd.DataFrame(
            [
                _raw_record("WK_A", date(2024, 7, 1), date(2024, 7, 20), institution_id="H001"),
                _raw_record("WK_B", date(2024, 7, 5), date(2024, 7, 20), institution_id="H001"),
            ]
        )
    )
    missing_inst = records_to_prescriptions(
        pd.DataFrame(
            [
                _raw_record("WK_A", date(2024, 7, 1), date(2024, 7, 20), institution_id=None),
                _raw_record("WK_B", date(2024, 7, 5), date(2024, 7, 20), institution_id="H001"),
            ]
        )
    )

    assert calculate_attributed_overlaps_for_patient(same_inst)[0].same_institution is True
    assert calculate_attributed_overlaps_for_patient(missing_inst)[0].same_institution is None


def test_attributed_overlap_counts_older_long_running_rx_active_in_later_window() -> None:
    from scripts.ops.elderly_ddi_institution_audit import (
        calculate_attributed_overlaps_for_patient,
        records_to_prescriptions,
    )

    prescriptions = records_to_prescriptions(
        pd.DataFrame(
            [
                _raw_record("WK_LONG", date(2024, 7, 1), date(2024, 12, 31), institution_id="H001"),
                _raw_record("WK_NEW", date(2024, 10, 15), date(2024, 10, 30), institution_id="H002"),
            ]
        )
    )

    pairs = calculate_attributed_overlaps_for_patient(prescriptions)

    assert len(pairs) == 1
    assert {pairs[0].drug_a_wk_compn, pairs[0].drug_b_wk_compn} == {"WK_LONG", "WK_NEW"}
    assert pairs[0].overlap_start == date(2024, 10, 15)
    assert pairs[0].overlap_end == date(2024, 10, 30)


def test_attributed_overlap_keeps_repeated_wk_pair_occurrences() -> None:
    from scripts.ops.elderly_ddi_institution_audit import (
        calculate_attributed_overlaps_for_patient,
        records_to_prescriptions,
    )

    prescriptions = records_to_prescriptions(
        pd.DataFrame(
            [
                _raw_record("WK_A", date(2024, 7, 1), date(2024, 7, 20), institution_id="H001", bill_no="B1"),
                _raw_record("WK_B", date(2024, 7, 5), date(2024, 7, 20), institution_id="H002", bill_no="B2"),
                _raw_record("WK_A", date(2024, 8, 1), date(2024, 8, 20), institution_id="H001", bill_no="B3"),
                _raw_record("WK_B", date(2024, 8, 5), date(2024, 8, 20), institution_id="H002", bill_no="B4"),
            ]
        )
    )

    pairs = calculate_attributed_overlaps_for_patient(prescriptions)

    assert [(pair.overlap_start, pair.overlap_end) for pair in pairs] == [
        (date(2024, 7, 5), date(2024, 7, 20)),
        (date(2024, 8, 5), date(2024, 8, 20)),
    ]


def test_attributed_overlap_zip_dedup_keeps_swapped_institution_assignment() -> None:
    from scripts.ops.elderly_ddi_institution_audit import (
        calculate_attributed_overlaps_for_patient,
        records_to_prescriptions,
    )

    prescriptions = records_to_prescriptions(
        pd.DataFrame(
            [
                _raw_record("WK_A", date(2024, 7, 1), date(2024, 7, 20), institution_id="H001", bill_no="B1"),
                _raw_record("WK_B", date(2024, 7, 5), date(2024, 7, 20), institution_id="H002", bill_no="B2"),
                _raw_record("WK_A", date(2024, 7, 1), date(2024, 7, 20), institution_id="H002", bill_no="B3"),
                _raw_record("WK_B", date(2024, 7, 5), date(2024, 7, 20), institution_id="H001", bill_no="B4"),
            ]
        )
    )

    pairs = calculate_attributed_overlaps_for_patient(prescriptions)
    assignment_keys = {
        frozenset(((pair.drug_a_wk_compn, pair.institution_a_id), (pair.drug_b_wk_compn, pair.institution_b_id)))
        for pair in pairs
    }

    assert frozenset((("WK_A", "H001"), ("WK_B", "H002"))) in assignment_keys
    assert frozenset((("WK_A", "H002"), ("WK_B", "H001"))) in assignment_keys


def _attributed_pair(
    drug_a: str,
    drug_b: str,
    *,
    patient_id: str = "P1",
    institution_a_id: object = "H001",
    institution_b_id: object = "H002",
    start: date = date(2024, 7, 1),
    end: date = date(2024, 7, 10),
) -> object:
    from scripts.ops.elderly_ddi_institution_audit import AttributedOverlapPair

    return AttributedOverlapPair(
        patient_id=patient_id,
        drug_a_wk_compn=drug_a,
        drug_b_wk_compn=drug_b,
        drug_a_edi=f"E{drug_a}",
        drug_b_edi=f"E{drug_b}",
        institution_a_id=institution_a_id,
        institution_b_id=institution_b_id,
        source_a="T30",
        source_b="T60",
        overlap_start=start,
        overlap_end=end,
        overlap_days=(end - start).days + 1,
    )


def test_classify_attributed_ddi_pairs_includes_default_contraindicated_and_major() -> None:
    from scripts.ops.elderly_ddi_institution_audit import classify_attributed_ddi_pairs

    pairs = [
        _attributed_pair("WK_CONTRA_A", "WK_CONTRA_B", patient_id="P1"),
        _attributed_pair("WK_MAJOR_A", "WK_MAJOR_B", patient_id="P2"),
    ]
    drug_master = _FakeDrugMaster(
        {
            "WK_CONTRA_A": ["D001"],
            "WK_CONTRA_B": ["D002"],
            "WK_MAJOR_A": ["D003"],
            "WK_MAJOR_B": ["D004"],
        }
    )
    ddi_matrix = pd.DataFrame(
        [
            {"drug_a_id": "D001", "drug_b_id": "D002", "severity": "Contraindicated"},
            {"drug_a_id": "D003", "drug_b_id": "D004", "severity": "Major"},
        ]
    )

    classified = classify_attributed_ddi_pairs(pairs, ddi_matrix, drug_master)

    assert [(row.patient_id, row.severity) for row in classified] == [
        ("P1", "Contraindicated"),
        ("P2", "Major"),
    ]


def test_classify_attributed_ddi_pairs_excludes_default_moderate_minor_and_case_mismatch() -> None:
    from scripts.ops.elderly_ddi_institution_audit import classify_attributed_ddi_pairs

    pairs = [
        _attributed_pair("WK_MOD_A", "WK_MOD_B", patient_id="P_MOD"),
        _attributed_pair("WK_MINOR_A", "WK_MINOR_B", patient_id="P_MINOR"),
        _attributed_pair("WK_LOWER_A", "WK_LOWER_B", patient_id="P_LOWER"),
    ]
    drug_master = _FakeDrugMaster(
        {
            "WK_MOD_A": ["D010"],
            "WK_MOD_B": ["D011"],
            "WK_MINOR_A": ["D012"],
            "WK_MINOR_B": ["D013"],
            "WK_LOWER_A": ["D014"],
            "WK_LOWER_B": ["D015"],
        }
    )
    ddi_matrix = pd.DataFrame(
        [
            {"drug_a_id": "D010", "drug_b_id": "D011", "severity": "Moderate"},
            {"drug_a_id": "D012", "drug_b_id": "D013", "severity": "Minor"},
            {"drug_a_id": "D014", "drug_b_id": "D015", "severity": "major"},
        ]
    )

    assert classify_attributed_ddi_pairs(pairs, ddi_matrix, drug_master) == []


def test_classify_attributed_ddi_pairs_preserves_all_drugmaster_ids_and_uses_highest_duplicate_severity() -> None:
    from scripts.ops.elderly_ddi_institution_audit import classify_attributed_ddi_pairs

    pair = _attributed_pair("WK_COMBO", "WK_TARGET")
    drug_master = _FakeDrugMaster({"WK_COMBO": ["D_LOW", "D_HIGH"], "WK_TARGET": ["D_TARGET"]})
    ddi_matrix = pd.DataFrame(
        [
            {"drug_a_id": "D_LOW", "drug_b_id": "D_TARGET", "severity": "Minor"},
            {"drug_a_id": "D_HIGH", "drug_b_id": "D_TARGET", "severity": "Moderate"},
            {"drug_a_id": "D_HIGH", "drug_b_id": "D_TARGET", "severity": "Major"},
        ]
    )

    classified = classify_attributed_ddi_pairs([pair], ddi_matrix, drug_master)

    assert len(classified) == 1
    assert classified[0].severity == "Major"
    assert classified[0].drug_a_wk_compn == "WK_COMBO"
    assert classified[0].drug_b_wk_compn == "WK_TARGET"


def test_classify_attributed_ddi_pairs_preserves_overlap_and_institution_tri_state() -> None:
    from scripts.ops.elderly_ddi_institution_audit import classify_attributed_ddi_pairs

    pairs = [
        _attributed_pair("WK_A", "WK_B", patient_id="P_SAME", institution_a_id="H001", institution_b_id="H001"),
        _attributed_pair("WK_A", "WK_B", patient_id="P_CROSS", institution_a_id="H001", institution_b_id="H002"),
        _attributed_pair("WK_A", "WK_B", patient_id="P_UNKNOWN", institution_a_id=None, institution_b_id="H002"),
    ]
    drug_master = _FakeDrugMaster({"WK_A": ["D001"], "WK_B": ["D002"]})
    ddi_matrix = pd.DataFrame([{"drug_a_id": "D001", "drug_b_id": "D002", "severity": "Major"}])

    classified = classify_attributed_ddi_pairs(pairs, ddi_matrix, drug_master)

    assert [row.same_institution for row in classified] == [True, False, None]
    assert [(row.overlap_start, row.overlap_end, row.overlap_days) for row in classified] == [
        (date(2024, 7, 1), date(2024, 7, 10), 10),
        (date(2024, 7, 1), date(2024, 7, 10), 10),
        (date(2024, 7, 1), date(2024, 7, 10), 10),
    ]


def test_summarize_institution_ddi_severity_counts_cross_institution_pair_for_both_rows() -> None:
    from scripts.ops.elderly_ddi_institution_audit import (
        classify_attributed_ddi_pairs,
        summarize_institution_ddi_severity,
    )

    pairs = [_attributed_pair("WK_A", "WK_B", patient_id="P1", institution_a_id="H001", institution_b_id="H002")]
    drug_master = _FakeDrugMaster({"WK_A": ["D001"], "WK_B": ["D002"]})
    ddi_matrix = pd.DataFrame([{"drug_a_id": "D001", "drug_b_id": "D002", "severity": "Major"}])
    classified = classify_attributed_ddi_pairs(pairs, ddi_matrix, drug_master)

    summary, _preflight = summarize_institution_ddi_severity(
        classified,
        institution_names={"H001": "Alpha", "H002": "Beta"},
        death_exclusion_status="complete",
    )

    assert summary.to_dict("records") == [
        {
            "institution_id": "H001",
            "institution_name": "Alpha",
            "severity": "Major",
            "event_count": 1,
            "distinct_patient_count": 1,
            "same_institution_event_count": 0,
            "cross_institution_event_count": 1,
            "unknown_institution_event_count": 0,
            "unmatched_institution_name_count": 0,
        },
        {
            "institution_id": "H002",
            "institution_name": "Beta",
            "severity": "Major",
            "event_count": 1,
            "distinct_patient_count": 1,
            "same_institution_event_count": 0,
            "cross_institution_event_count": 1,
            "unknown_institution_event_count": 0,
            "unmatched_institution_name_count": 0,
        },
    ]


def test_summarize_institution_ddi_severity_counts_same_institution_pair_once() -> None:
    from scripts.ops.elderly_ddi_institution_audit import (
        classify_attributed_ddi_pairs,
        summarize_institution_ddi_severity,
    )

    pairs = [_attributed_pair("WK_A", "WK_B", patient_id="P1", institution_a_id="H001", institution_b_id="H001")]
    drug_master = _FakeDrugMaster({"WK_A": ["D001"], "WK_B": ["D002"]})
    ddi_matrix = pd.DataFrame([{"drug_a_id": "D001", "drug_b_id": "D002", "severity": "Contraindicated"}])
    classified = classify_attributed_ddi_pairs(pairs, ddi_matrix, drug_master)

    summary, _preflight = summarize_institution_ddi_severity(
        classified,
        institution_names={"H001": "Alpha"},
        death_exclusion_status="complete",
    )

    row = summary.iloc[0].to_dict()
    assert row["institution_id"] == "H001"
    assert row["severity"] == "Contraindicated"
    assert row["event_count"] == 1
    assert row["distinct_patient_count"] == 1
    assert row["same_institution_event_count"] == 1
    assert row["cross_institution_event_count"] == 0


def test_summarize_institution_ddi_severity_empty_schema_has_no_patient_or_drug_columns() -> None:
    from scripts.ops.elderly_ddi_institution_audit import (
        summarize_institution_ddi_severity,
    )

    summary, preflight = summarize_institution_ddi_severity([], death_exclusion_status="complete")

    assert summary.empty
    assert list(summary.columns) == [
        "institution_id",
        "institution_name",
        "severity",
        "event_count",
        "distinct_patient_count",
        "same_institution_event_count",
        "cross_institution_event_count",
        "unknown_institution_event_count",
        "unmatched_institution_name_count",
    ]
    assert "patient_id" not in summary.columns
    assert not any("drug" in column for column in summary.columns)
    assert preflight.details["event_count"] == 0


def test_summarize_institution_ddi_severity_reports_unmatched_name_count() -> None:
    from scripts.ops.elderly_ddi_institution_audit import (
        classify_attributed_ddi_pairs,
        summarize_institution_ddi_severity,
    )

    pairs = [_attributed_pair("WK_A", "WK_B", patient_id="P1", institution_a_id="H001", institution_b_id="H002")]
    drug_master = _FakeDrugMaster({"WK_A": ["D001"], "WK_B": ["D002"]})
    ddi_matrix = pd.DataFrame([{"drug_a_id": "D001", "drug_b_id": "D002", "severity": "Major"}])
    classified = classify_attributed_ddi_pairs(pairs, ddi_matrix, drug_master)

    summary, preflight = summarize_institution_ddi_severity(
        classified,
        institution_names={"H001": "Alpha"},
        death_exclusion_status="complete",
    )

    unmatched = summary.loc[summary["institution_id"] == "H002"].iloc[0].to_dict()
    assert unmatched["institution_name"] is None
    assert unmatched["unmatched_institution_name_count"] == 1
    assert preflight.warnings == ("unmatched_institution_names",)
    assert preflight.details["unmatched_institution_name_count"] == 1


def test_summarize_institution_ddi_severity_provisional_without_death_exclusion_is_not_final() -> None:
    from scripts.ops.elderly_ddi_institution_audit import (
        summarize_institution_ddi_severity,
    )

    _summary, preflight = summarize_institution_ddi_severity([], death_exclusion_status="unavailable")

    assert preflight.ok is False
    assert "death_exclusion_unavailable_provisional_only" in preflight.warnings
    assert preflight.details["death_exclusion_status"] == "unavailable"
    assert preflight.details["final_summary"] is False
