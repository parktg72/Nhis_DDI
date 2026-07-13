"""Ops-only helpers for elderly/disease-code DDI institution audits.

This module intentionally stays off the production training/serving path.  It
provides small pure functions that can be tested before wiring any real HANA or
Raw Parquet inputs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, Sequence

import pandas as pd

from scripts.etl.models import DrugOverlapPair, PrescriptionRecord
from scripts.etl.prescription_aggregator import ddi_pair_severities

DEFAULT_DISEASE_CODE_COLUMNS = (
    "sick_code",
    "SICK_CODE",
    "diagnosis_code",
    "DIAGNOSIS_CODE",
    "ICD_CODE",
    "MCEX_SICK_SYM",
    "MCEX_SICK_SYM1",
)

DEFAULT_INSTITUTION_DDI_SEVERITIES = ("Contraindicated", "Major")

INSTITUTION_DDI_SUMMARY_COLUMNS = (
    "institution_id",
    "institution_name",
    "severity",
    "event_count",
    "distinct_patient_count",
    "same_institution_event_count",
    "cross_institution_event_count",
    "unknown_institution_event_count",
    "unmatched_institution_name_count",
)


@dataclass(frozen=True)
class AuditPreflight:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    details: dict[str, object] = field(default_factory=dict)


class DeathDataRequiredError(ValueError):
    """Raised when final death exclusion is requested without death data."""


@dataclass(frozen=True)
class AttributedOverlapPair:
    patient_id: str
    drug_a_wk_compn: str
    drug_b_wk_compn: str
    drug_a_edi: str | None
    drug_b_edi: str | None
    institution_a_id: str | None
    institution_b_id: str | None
    source_a: str | None
    source_b: str | None
    overlap_start: date
    overlap_end: date
    overlap_days: int

    @property
    def same_institution(self) -> bool | None:
        if self.institution_a_id is None or self.institution_b_id is None:
            return None
        return self.institution_a_id == self.institution_b_id


@dataclass(frozen=True)
class ClassifiedAttributedDDIPair:
    patient_id: str
    severity: str
    drug_a_wk_compn: str
    drug_b_wk_compn: str
    drug_a_edi: str | None
    drug_b_edi: str | None
    institution_a_id: str | None
    institution_b_id: str | None
    source_a: str | None
    source_b: str | None
    overlap_start: date
    overlap_end: date
    overlap_days: int

    @property
    def same_institution(self) -> bool | None:
        if self.institution_a_id is None or self.institution_b_id is None:
            return None
        return self.institution_a_id == self.institution_b_id


def normalize_disease_codes_from_pdf_text(text: str) -> tuple[str, ...]:
    """Extract KCD-looking codes from PDF text.

    The source PDF contains codes such as F00*, I63, U23.4, and R25.1.
    Preserve the display form while de-duplicating in encounter order.
    """

    seen: set[str] = set()
    codes: list[str] = []
    pattern = re.compile(r"(?<![A-Z0-9.])([A-Z][0-9]{2}(?:\.[0-9A-Z]+)?)(\*)?(?![A-Z0-9])")
    for match in pattern.finditer(str(text).upper()):
        code = f"{match.group(1)}{match.group(2) or ''}".strip()
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return tuple(codes)


def patient_has_disease_code(
    diagnosis_codes: Iterable[object],
    disease_codes: Sequence[str],
) -> bool:
    """Return True if any diagnosis matches the elderly-disease codelist."""

    diagnosis_norms = [_normalize_kcd_code(value) for value in diagnosis_codes]
    diagnosis_norms = [value for value in diagnosis_norms if value]
    rules = [_disease_rule(code) for code in disease_codes]
    for diagnosis in diagnosis_norms:
        for source_norm, prefix_match in rules:
            if not source_norm:
                continue
            if diagnosis == source_norm:
                return True
            if prefix_match and diagnosis.startswith(source_norm):
                return True
    return False


def select_target_patients(
    eligibility: pd.DataFrame,
    diagnoses: pd.DataFrame | None,
    *,
    disease_codes: Sequence[str],
    min_age: int = 65,
    diagnosis_code_columns: Sequence[str] = DEFAULT_DISEASE_CODE_COLUMNS,
) -> tuple[pd.DataFrame, AuditPreflight]:
    """Select age>=min_age plus under-min-age patients with matching disease codes."""

    if "patient_id" not in eligibility.columns:
        raise ValueError("eligibility must contain patient_id")
    if "age" not in eligibility.columns:
        raise ValueError("eligibility must contain age")

    warnings: list[str] = []
    details: dict[str, object] = {}
    work = eligibility.copy()
    patient_ids = work["patient_id"].map(_string_or_none)
    ages = pd.to_numeric(work["age"], errors="coerce")
    valid_patient_id = patient_ids.notna()
    valid_age = ages.notna()

    invalid_patient_id_count = int((~valid_patient_id).sum())
    invalid_age_count = int((~valid_age & valid_patient_id).sum())
    if invalid_patient_id_count:
        warnings.append("excluded_invalid_patient_id")
    if invalid_age_count:
        warnings.append("excluded_invalid_age")

    valid = work.loc[valid_patient_id & valid_age].copy()
    valid["patient_id"] = patient_ids.loc[valid.index].astype(str)
    valid["age"] = ages.loc[valid.index]

    disease_patient_ids: set[str] = set()
    diagnosis_status = "unavailable"
    if diagnoses is None:
        warnings.append("diagnosis_source_unavailable_under_min_age_not_included")
    elif diagnoses.empty:
        diagnosis_status = "empty"
        warnings.append("diagnosis_source_empty_under_min_age_not_included")
    elif "patient_id" not in diagnoses.columns:
        diagnosis_status = "missing_patient_id"
        warnings.append("diagnosis_source_missing_patient_id_under_min_age_not_included")
    else:
        code_columns = [column for column in diagnosis_code_columns if column in diagnoses.columns]
        if not code_columns:
            diagnosis_status = "missing_code_column"
            warnings.append("diagnosis_source_missing_code_column_under_min_age_not_included")
        else:
            diagnosis_status = "available"
            disease_patient_ids = _patients_with_matching_diagnoses(
                diagnoses,
                code_columns=code_columns,
                disease_codes=disease_codes,
            )

    age_mask = valid["age"] >= min_age
    under_min_age = valid["age"] < min_age
    disease_mask = valid["patient_id"].isin(disease_patient_ids)
    cohort = valid.loc[age_mask | (under_min_age & disease_mask)].copy()

    details.update(
        {
            "min_age": min_age,
            "input_row_count": int(len(eligibility)),
            "invalid_patient_id_count": invalid_patient_id_count,
            "invalid_age_count": invalid_age_count,
            "valid_eligibility_count": int(len(valid)),
            "age_included_count": int(age_mask.sum()),
            "under_min_age_disease_code_included_count": int((under_min_age & disease_mask).sum()),
            "under_min_age_without_disease_code_count": int((under_min_age & ~disease_mask).sum()),
            "diagnosis_source_status": diagnosis_status,
            "matched_disease_code_patient_count": int(len(disease_patient_ids)),
            "selected_count": int(len(cohort)),
        }
    )
    return cohort.reset_index(drop=True), AuditPreflight(
        ok=True,
        warnings=tuple(dict.fromkeys(warnings)),
        details=details,
    )


def normalize_death_dates(
    deaths: pd.DataFrame,
    *,
    date_columns: Sequence[str] = ("DTH_ASSMD_DT", "DTH_HM_DT", "DTH_BFC_DT"),
) -> pd.DataFrame:
    """Return patient_id/death_date rows with parsed dates when possible."""

    if "patient_id" in deaths.columns:
        patient_col = "patient_id"
    elif "INDI_DSCM_NO" in deaths.columns:
        patient_col = "INDI_DSCM_NO"
    else:
        raise ValueError("deaths must contain patient_id or INDI_DSCM_NO")

    available_date_columns = [column for column in date_columns if column in deaths.columns]
    if not available_date_columns:
        raise ValueError("deaths must contain at least one configured death date column")

    normalized = pd.DataFrame({"patient_id": deaths[patient_col].map(_string_or_none)})
    parsed_dates = [_parse_death_date_column(deaths[column]) for column in available_date_columns]
    if len(parsed_dates) == 1:
        normalized["death_date"] = parsed_dates[0]
    else:
        # object-dtype date 컬럼을 그대로 min(axis=1)하면 date vs NaN(float) 비교로
        # TypeError — datetime64로 올려 min 후 date로 되돌린다 (NaT는 결측 유지)
        parsed_frame = pd.concat(
            [pd.to_datetime(series, errors="coerce") for series in parsed_dates], axis=1
        )
        min_dates = parsed_frame.min(axis=1)
        normalized["death_date"] = min_dates.dt.date.where(min_dates.notna(), None)
    return normalized.loc[normalized["patient_id"].notna()].reset_index(drop=True)


def exclude_deceased_patients(
    cohort: pd.DataFrame,
    deaths: pd.DataFrame | None,
    *,
    audit_end: date,
    death_policy: str = "require",
    exclude_unparseable_death_rows: bool = False,
    date_columns: Sequence[str] = ("DTH_ASSMD_DT", "DTH_HM_DT", "DTH_BFC_DT"),
) -> tuple[pd.DataFrame, AuditPreflight]:
    """Exclude patients with parsed death_date on or before audit_end."""

    if deaths is None:
        if death_policy == "require":
            raise DeathDataRequiredError("death data is required for final deceased-patient exclusion")
        if death_policy != "provisional_allow_missing":
            raise ValueError(f"unsupported death_policy: {death_policy}")
        return cohort.copy().reset_index(drop=True), AuditPreflight(
            ok=False,
            warnings=("death_data_unavailable_provisional_only",),
            details={"death_exclusion_status": "unavailable", "excluded_deceased_count": 0},
        )

    if "patient_id" not in cohort.columns:
        raise ValueError("cohort must contain patient_id")

    warnings: list[str] = []
    normalized = normalize_death_dates(deaths, date_columns=date_columns)
    parseable = normalized["death_date"].notna()
    unparseable_count = int((~parseable).sum())
    if unparseable_count:
        warnings.append("death_rows_without_parseable_date")

    parseable_deaths = normalized.loc[parseable].copy()
    if parseable_deaths.empty:
        deceased_ids: set[str] = set()
    else:
        deceased_ids = {
            str(row.patient_id)
            for row in parseable_deaths[["patient_id", "death_date"]].itertuples(index=False)
            if row.death_date <= audit_end
        }
    if exclude_unparseable_death_rows:
        deceased_ids.update(normalized.loc[~parseable, "patient_id"].astype(str))

    patient_ids = cohort["patient_id"].map(_string_or_none)
    keep = ~patient_ids.isin(deceased_ids)
    filtered = cohort.loc[keep].copy().reset_index(drop=True)
    status = "complete_with_warnings" if unparseable_count else "complete"
    return filtered, AuditPreflight(
        ok=True,
        warnings=tuple(warnings),
        details={
            "death_exclusion_status": status,
            "audit_end": audit_end.isoformat(),
            "death_input_row_count": int(len(deaths)),
            "death_rows_with_parseable_date_count": int(parseable.sum()),
            "death_rows_without_parseable_date_count": unparseable_count,
            "excluded_deceased_count": int((~keep).sum()),
            "remaining_count": int(len(filtered)),
        },
    )


def build_institution_name_map(
    yoyang: pd.DataFrame,
    *,
    std_year: str,
) -> dict[str, str]:
    """Build a deterministic institution-id -> institution-name map for one year.

    Duplicate same-year institution rows follow a last-row-wins policy.  If the
    final row for an institution has a blank name, the institution is left
    unmapped so downstream output preserves the ID with name ``None``.
    """

    required = {"STD_YYYY", "MDCARE_SYM", "INST_NM"}
    missing = sorted(required.difference(yoyang.columns))
    if missing:
        raise ValueError(f"institution master missing required columns: {missing}")

    target_year = _normalize_year(std_year)
    if target_year is None:
        raise ValueError(f"invalid std_year: {std_year!r}")

    work = yoyang.copy()
    work["_std_year"] = work["STD_YYYY"].map(_normalize_year)
    work["_institution_id"] = work["MDCARE_SYM"].map(_string_or_none)
    work["_institution_name"] = work["INST_NM"].map(_string_or_none)
    filtered = work.loc[
        (work["_std_year"] == target_year)
        & work["_institution_id"].notna()
    ]

    latest_names: dict[str, str | None] = {}
    for institution_id, institution_name in filtered[["_institution_id", "_institution_name"]].itertuples(
        index=False,
        name=None,
    ):
        latest_names[str(institution_id)] = _string_or_none(institution_name)
    return {
        institution_id: institution_name
        for institution_id, institution_name in latest_names.items()
        if institution_name is not None
    }


def attach_institution_names(
    events: pd.DataFrame,
    institution_names: dict[str, str],
    *,
    id_columns: Sequence[str] = ("institution_id",),
) -> tuple[pd.DataFrame, AuditPreflight]:
    """Attach institution name columns while preserving unknown institution IDs."""

    output = events.copy()
    unmatched_ids: set[str] = set()
    unmatched_count = 0
    normalized_map = {
        key: value
        for key, value in (
            (_string_or_none(raw_key), _string_or_none(raw_value))
            for raw_key, raw_value in institution_names.items()
        )
        if key is not None and value is not None
    }

    for id_column in id_columns:
        if id_column not in output.columns:
            raise ValueError(f"events missing institution id column: {id_column}")
        name_column = _institution_name_column(id_column)
        normalized_ids = output[id_column].map(_string_or_none)
        clean_ids = [_string_or_none(value) for value in normalized_ids]
        output[id_column] = pd.Series(clean_ids, index=output.index, dtype=object)
        names: list[str | None] = []
        for raw_institution_id in clean_ids:
            institution_id = _string_or_none(raw_institution_id)
            if institution_id is None:
                names.append(None)
            elif institution_id in normalized_map:
                names.append(normalized_map[institution_id])
            else:
                names.append(None)
                unmatched_ids.add(institution_id)
                unmatched_count += 1
        output[name_column] = pd.Series(names, index=output.index, dtype=object)

    warnings = ("unmatched_institution_names",) if unmatched_count else ()
    return output, AuditPreflight(
        ok=True,
        warnings=warnings,
        details={
            "event_row_count": int(len(output)),
            "institution_id_columns": list(id_columns),
            "unmatched_institution_name_count": int(unmatched_count),
            "unmatched_institution_ids": sorted(unmatched_ids),
        },
    )


def records_to_prescriptions(records: pd.DataFrame) -> list[PrescriptionRecord]:
    """Convert local Raw `records_YYYYMMDD.parquet` rows into prescriptions."""

    prescriptions: list[PrescriptionRecord] = []
    if records.empty:
        return prescriptions

    columns = tuple(records.columns)
    for row_values in records.itertuples(index=False, name=None):
        row = dict(zip(columns, row_values))
        patient_id = _string_or_none(row.get("patient_id"))
        wk_compn_cd = _string_or_none(row.get("wk_compn_cd"))
        start_date = _coerce_date(row.get("start_date"))
        end_date = _coerce_date(row.get("end_date"))
        if patient_id is None or wk_compn_cd is None or start_date is None:
            continue
        if end_date is None or end_date < start_date:
            # end < start(역전 행)는 신뢰 불가 end로 간주 — total_days로 재유도.
            # 역전 스팬을 그대로 두면 그 약물이 어떤 overlap에도 안 걸려 DDI 과소집계
            total_days = _coerce_positive_int(row.get("total_days"), default=1)
            end_date = start_date + timedelta(days=total_days - 1)
        else:
            total_days = max((end_date - start_date).days + 1, 1)
        prescriptions.append(
            PrescriptionRecord(
                patient_id=patient_id,
                institution_id=_string_or_none(row.get("institution_id")),
                bill_no=_string_or_none(row.get("bill_no")) or "",
                wk_compn_cd=wk_compn_cd,
                edi_code=_string_or_none(row.get("edi_code")),
                atc_code=None,
                gnl_nm_cd=_string_or_none(row.get("gnl_nm_cd")),
                efmdc_clsf_no=_string_or_none(row.get("efmdc_clsf_no")),
                drug_name=None,
                start_date=start_date,
                end_date=end_date,
                total_days=total_days,
                dose_once=_coerce_float(row.get("dose_once"), default=0.0),
                dose_freq=_coerce_positive_int(row.get("dose_freq"), default=1),
                sick_code=_string_or_none(row.get("sick_code")),
                sex=_string_or_none(row.get("sex")),
                age_id=_string_or_none(row.get("age_id")),
                institution_type=_string_or_none(row.get("institution_type")),
                source=_string_or_none(row.get("source")) or "Raw",
            )
        )
    return prescriptions


def calculate_attributed_overlaps_for_patient(
    prescriptions: list[PrescriptionRecord],
    *,
    window_days: int = 90,
    min_overlap: int = 7,
) -> list[AttributedOverlapPair]:
    """Calculate DDI-candidate overlaps while preserving both institutions."""

    if len(prescriptions) < 2:
        return []

    sorted_prescriptions = sorted(prescriptions, key=lambda record: record.start_date)
    pairs: list[AttributedOverlapPair] = []
    seen: set[tuple[tuple[tuple[str, str | None, str | None], ...], date, date]] = set()

    for anchor in sorted_prescriptions:
        window_start = anchor.start_date
        window_end = window_start + timedelta(days=window_days - 1)
        window_drugs = [
            prescription
            for prescription in sorted_prescriptions
            if prescription.start_date <= window_end and prescription.end_date >= window_start
        ]
        for left_index, left in enumerate(window_drugs):
            for right in window_drugs[left_index + 1:]:
                if left.wk_compn_cd == right.wk_compn_cd:
                    continue
                overlap_start = max(left.start_date, right.start_date)
                overlap_end = min(left.end_date, right.end_date)
                overlap_days = (overlap_end - overlap_start).days + 1
                if overlap_days < min_overlap:
                    continue
                dedup_key = _attributed_overlap_dedup_key(left, right, overlap_start, overlap_end)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                pairs.append(
                    AttributedOverlapPair(
                        patient_id=left.patient_id,
                        drug_a_wk_compn=left.wk_compn_cd,
                        drug_b_wk_compn=right.wk_compn_cd,
                        drug_a_edi=_string_or_none(left.edi_code),
                        drug_b_edi=_string_or_none(right.edi_code),
                        institution_a_id=_string_or_none(left.institution_id),
                        institution_b_id=_string_or_none(right.institution_id),
                        source_a=_string_or_none(left.source),
                        source_b=_string_or_none(right.source),
                        overlap_start=overlap_start,
                        overlap_end=overlap_end,
                        overlap_days=overlap_days,
                    )
                )
    return pairs


def classify_attributed_ddi_pairs(
    pairs: Iterable[AttributedOverlapPair],
    ddi_matrix: pd.DataFrame | None,
    drug_master: object | None,
    *,
    severity_allowlist: Sequence[str] = DEFAULT_INSTITUTION_DDI_SEVERITIES,
) -> list[ClassifiedAttributedDDIPair]:
    """Classify attributed overlaps through the existing DrugMaster/DDI path."""

    pair_list = list(pairs)
    if not pair_list or ddi_matrix is None or drug_master is None:
        return []

    adapted_pairs = [_attributed_to_drug_overlap_pair(pair) for pair in pair_list]
    original_by_id = {id(adapted): original for adapted, original in zip(adapted_pairs, pair_list)}
    allowed = set(severity_allowlist)
    classified: list[ClassifiedAttributedDDIPair] = []
    for adapted_pair, severity in ddi_pair_severities(adapted_pairs, ddi_matrix, drug_master):
        if severity not in allowed:
            continue
        classified.append(_classified_from_attributed_pair(original_by_id[id(adapted_pair)], severity))
    return classified


def summarize_institution_ddi_severity(
    classified_pairs: Iterable[ClassifiedAttributedDDIPair],
    *,
    institution_names: dict[str, str] | None = None,
    death_exclusion_status: str = "unavailable",
) -> tuple[pd.DataFrame, AuditPreflight]:
    """Aggregate classified DDI events by institution and severity."""

    normalized_names = _normalize_institution_name_lookup(institution_names)
    accumulators: dict[tuple[str, str], dict[str, object]] = {}
    classified_count = 0
    events_without_institution_count = 0

    for classified in classified_pairs:
        classified_count += 1
        sides = _summary_institution_sides(classified)
        if not sides:
            events_without_institution_count += 1
            continue
        for institution_id, bucket in sides:
            key = (institution_id, classified.severity)
            acc = accumulators.setdefault(
                key,
                {
                    "patients": set(),
                    "event_count": 0,
                    "same_institution_event_count": 0,
                    "cross_institution_event_count": 0,
                    "unknown_institution_event_count": 0,
                    "unmatched_institution_name_count": 0,
                },
            )
            acc["event_count"] = int(acc["event_count"]) + 1
            acc[f"{bucket}_institution_event_count"] = int(acc[f"{bucket}_institution_event_count"]) + 1
            patients = acc["patients"]
            if isinstance(patients, set):
                patients.add(classified.patient_id)
            if normalized_names.get(institution_id) is None:
                acc["unmatched_institution_name_count"] = int(acc["unmatched_institution_name_count"]) + 1

    records: list[dict[str, object]] = []
    for (institution_id, severity), acc in sorted(accumulators.items(), key=_summary_sort_key):
        patients = acc["patients"]
        records.append(
            {
                "institution_id": institution_id,
                "institution_name": normalized_names.get(institution_id),
                "severity": severity,
                "event_count": int(acc["event_count"]),
                "distinct_patient_count": len(patients) if isinstance(patients, set) else 0,
                "same_institution_event_count": int(acc["same_institution_event_count"]),
                "cross_institution_event_count": int(acc["cross_institution_event_count"]),
                "unknown_institution_event_count": int(acc["unknown_institution_event_count"]),
                "unmatched_institution_name_count": int(acc["unmatched_institution_name_count"]),
            }
        )

    summary = pd.DataFrame.from_records(records, columns=INSTITUTION_DDI_SUMMARY_COLUMNS)
    if not summary.empty:
        summary["institution_name"] = pd.Series(
            [None if pd.isna(value) else value for value in summary["institution_name"]],
            index=summary.index,
            dtype=object,
        )
    unmatched_name_count = int(summary["unmatched_institution_name_count"].sum()) if not summary.empty else 0
    warnings: list[str] = []
    if death_exclusion_status == "unavailable":
        warnings.append("death_exclusion_unavailable_provisional_only")
    if institution_names is None and not summary.empty:
        warnings.append("institution_name_source_unavailable_provisional_only")
    elif unmatched_name_count:
        warnings.append("unmatched_institution_names")
    if events_without_institution_count:
        warnings.append("events_without_institution_excluded_from_summary")

    final_summary = death_exclusion_status != "unavailable" and not (
        institution_names is None and not summary.empty
    )
    return summary, AuditPreflight(
        ok=final_summary,
        warnings=tuple(dict.fromkeys(warnings)),
        details={
            "death_exclusion_status": death_exclusion_status,
            "final_summary": final_summary,
            "classified_pair_count": classified_count,
            # event_count = 고유 이벤트 수. summary 합계는 cross-institution 이벤트가
            # 양쪽 기관 행에 각각 계상돼 최대 2배 과대 — 행 합계는 별도 키로 노출
            "event_count": int(classified_count),
            "attributed_event_row_total": int(summary["event_count"].sum()) if not summary.empty else 0,
            "institution_summary_row_count": int(len(summary)),
            "unmatched_institution_name_count": unmatched_name_count,
            "events_without_institution_count": events_without_institution_count,
        },
    )


def _attributed_overlap_dedup_key(
    left: PrescriptionRecord,
    right: PrescriptionRecord,
    overlap_start: date,
    overlap_end: date,
) -> tuple[tuple[tuple[str, str | None, str | None], ...], date, date]:
    sides = tuple(
        sorted(
            (
                (left.wk_compn_cd, _string_or_none(left.institution_id), _string_or_none(left.source)),
                (right.wk_compn_cd, _string_or_none(right.institution_id), _string_or_none(right.source)),
            )
        )
    )
    return sides, overlap_start, overlap_end


def _attributed_to_drug_overlap_pair(pair: AttributedOverlapPair) -> DrugOverlapPair:
    return DrugOverlapPair(
        patient_id=pair.patient_id,
        drug_a_wk_compn=pair.drug_a_wk_compn,
        drug_a_edi=pair.drug_a_edi,
        drug_a_atc=None,
        drug_a_name=None,
        drug_b_wk_compn=pair.drug_b_wk_compn,
        drug_b_edi=pair.drug_b_edi,
        drug_b_atc=None,
        drug_b_name=None,
        overlap_start=pair.overlap_start,
        overlap_end=pair.overlap_end,
        overlap_days=pair.overlap_days,
        window_start=pair.overlap_start,
        window_end=pair.overlap_end,
    )


def _classified_from_attributed_pair(
    pair: AttributedOverlapPair,
    severity: str,
) -> ClassifiedAttributedDDIPair:
    return ClassifiedAttributedDDIPair(
        patient_id=pair.patient_id,
        severity=severity,
        drug_a_wk_compn=pair.drug_a_wk_compn,
        drug_b_wk_compn=pair.drug_b_wk_compn,
        drug_a_edi=pair.drug_a_edi,
        drug_b_edi=pair.drug_b_edi,
        institution_a_id=pair.institution_a_id,
        institution_b_id=pair.institution_b_id,
        source_a=pair.source_a,
        source_b=pair.source_b,
        overlap_start=pair.overlap_start,
        overlap_end=pair.overlap_end,
        overlap_days=pair.overlap_days,
    )


def _summary_institution_sides(pair: ClassifiedAttributedDDIPair) -> list[tuple[str, str]]:
    institution_a_id = _string_or_none(pair.institution_a_id)
    institution_b_id = _string_or_none(pair.institution_b_id)
    if pair.same_institution is True:
        return [(institution_a_id, "same")] if institution_a_id is not None else []
    if pair.same_institution is False:
        return [
            (institution_id, "cross")
            for institution_id in dict.fromkeys((institution_a_id, institution_b_id))
            if institution_id is not None
        ]
    return [
        (institution_id, "unknown")
        for institution_id in dict.fromkeys((institution_a_id, institution_b_id))
        if institution_id is not None
    ]


def _normalize_institution_name_lookup(institution_names: dict[str, str] | None) -> dict[str, str]:
    if institution_names is None:
        return {}
    return {
        institution_id: institution_name
        for institution_id, institution_name in (
            (_string_or_none(raw_id), _string_or_none(raw_name))
            for raw_id, raw_name in institution_names.items()
        )
        if institution_id is not None and institution_name is not None
    }


def _summary_sort_key(item: tuple[tuple[str, str], dict[str, object]]) -> tuple[int, str, str]:
    (institution_id, severity), _acc = item
    severity_rank = {severity: index for index, severity in enumerate(DEFAULT_INSTITUTION_DDI_SEVERITIES)}
    return severity_rank.get(severity, len(severity_rank)), institution_id, severity


def _coerce_date(value: object) -> date | None:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"^(\d{8})\.0$", r"\1", text)
    if re.fullmatch(r"\d{8}", text):
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _coerce_positive_int(value: object, *, default: int) -> int:
    if pd.isna(value):
        return default
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    return max(int(parsed), 1)


def _coerce_float(value: object, *, default: float) -> float:
    if pd.isna(value):
        return default
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    return float(parsed)


def _parse_death_date_column(series: pd.Series) -> pd.Series:
    text = series.map(lambda value: "" if pd.isna(value) else str(value).strip())
    text = text.str.replace(r"^(\d{8})\.0$", r"\1", regex=True)
    yyyymmdd = text.str.fullmatch(r"\d{8}")
    parsed_default = pd.to_datetime(text.where(~yyyymmdd), errors="coerce").dt.date
    if not bool(yyyymmdd.any()):
        return parsed_default
    parsed_yyyymmdd = pd.to_datetime(
        text.where(yyyymmdd),
        format="%Y%m%d",
        errors="coerce",
    ).dt.date
    combined = parsed_default.where(~yyyymmdd, parsed_yyyymmdd)
    return combined.map(lambda value: value.date() if isinstance(value, pd.Timestamp) else value)


def _normalize_year(value: object) -> str | None:
    text = _string_or_none(value)
    if text is None:
        return None
    if re.fullmatch(r"\d{4}", text):
        return text
    if re.fullmatch(r"\d{4}\.0", text):
        return text[:4]
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    if float(numeric).is_integer() and 1000 <= int(numeric) <= 9999:
        return str(int(numeric))
    return None


def _institution_name_column(id_column: str) -> str:
    if id_column.endswith("_id"):
        return f"{id_column[:-3]}_name"
    return f"{id_column}_name"


def _normalize_kcd_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[.\s]", "", str(value).upper()).rstrip("*")


def _disease_rule(source_code: str) -> tuple[str, bool]:
    raw = str(source_code).strip().upper()
    normalized = _normalize_kcd_code(raw)
    prefix_match = raw.endswith("*") or len(normalized) == 3
    return normalized, prefix_match


def _string_or_none(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _patients_with_matching_diagnoses(
    diagnoses: pd.DataFrame,
    *,
    code_columns: Sequence[str],
    disease_codes: Sequence[str],
) -> set[str]:
    matched: set[str] = set()
    for _, row in diagnoses.iterrows():
        patient_id = _string_or_none(row.get("patient_id"))
        if patient_id is None:
            continue
        if patient_has_disease_code((row.get(column) for column in code_columns), disease_codes):
            matched.add(patient_id)
    return matched
