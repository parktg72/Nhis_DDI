"""Characterize serving fallbacks, request mutation, and vector alignment."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from serving.schemas import DrugItem, PredictRequest


def _request(
    drugs: list[DrugItem] | None = None,
    age: int | None = 65,
    sex: str | None = "M",
) -> PredictRequest:
    from serving.schemas import DrugItem, PredictRequest

    if drugs is None:
        drugs = [DrugItem(edi_code="A001", drug_name="aspirin", total_days=30)]
    return PredictRequest(
        patient_id="p1",
        drugs=drugs,
        patient_age=age,
        patient_sex=sex,
    )


def _assert_pydantic_validation_error(error: BaseException) -> None:
    error_type = type(error)
    assert error_type.__module__.startswith("pydantic")
    assert error_type.__name__ == "ValidationError"


def test_resource_absence_fallbacks():
    from serving.predictor import RequestFeatureBuilder

    _, features = RequestFeatureBuilder(
        ddi_matrix=None,
        cyp_extractor=None,
        code_standardizer=None,
    ).build(_request())

    assert {features[name] for name in (
        "ddi_contraindicated",
        "ddi_major",
        "ddi_moderate",
        "ddi_minor",
    )} == {0.0}
    assert {features[name] for name in (
        "cyp_risk_score",
        "cyp_high_risk_pairs",
        "cyp_max_enzyme_risk",
    )} == {0.0}
    assert features["dup_efmdc"] == 0.0
    assert features["drug_count_7d"] == features["drug_count"]


def test_legacy_rule_feature_fallback_can_be_nonzero():
    from serving.predictor import RequestFeatureBuilder
    from serving.schemas import DrugItem

    request = _request(
        drugs=[
            DrugItem(
                edi_code="A001",
                drug_name="warfarin",
                atc_code="B01AA03",
                total_days=30,
            )
        ]
    )
    _, features = RequestFeatureBuilder(code_standardizer=None).build(
        request,
        rule_features_active=False,
    )

    assert features["has_high_risk_drug"] == 1.0


def test_legacy_triple_whammy_fallback_can_be_nonzero():
    from serving.predictor import RequestFeatureBuilder
    from serving.schemas import DrugItem

    drugs = [
        DrugItem(edi_code="A1", atc_code="C09AA01", total_days=30),
        DrugItem(edi_code="A2", atc_code="C03DA01", total_days=30),
        DrugItem(edi_code="A3", atc_code="M01A", total_days=30),
    ]
    _, features = RequestFeatureBuilder(code_standardizer=None).build(
        _request(drugs=drugs),
        rule_features_active=False,
    )

    assert features["triple_whammy"] == 1.0


def test_request_construction_mutates_default_dates_and_strips_edi():
    from datetime import date

    from serving.schemas import DrugItem, PredictRequest

    today_before = date.today()
    drug = DrugItem(edi_code="  A001  ", total_days=30)
    request = PredictRequest(patient_id="p1", drugs=[drug])
    today_after = date.today()

    assert drug.edi_code == "A001"
    assert drug.start_date is not None
    assert request.reference_date is not None
    assert today_before <= drug.start_date <= today_after
    assert request.reference_date == drug.start_date


@pytest.mark.parametrize("days", [0, 366])
def test_drug_duration_domain_rejects_out_of_range(days: int):
    from serving.schemas import DrugItem

    with pytest.raises(ValueError) as error:
        DrugItem(edi_code="A001", total_days=days)
    _assert_pydantic_validation_error(error.value)


def test_request_domain_rejects_bad_age_sex_and_empty_drugs():
    from serving.schemas import DrugItem, PredictRequest

    with pytest.raises(ValueError) as empty_error:
        PredictRequest(patient_id="p1", drugs=[], patient_age=65)
    _assert_pydantic_validation_error(empty_error.value)
    with pytest.raises(ValueError) as age_error:
        PredictRequest(
            patient_id="p1",
            drugs=[DrugItem(edi_code="A001", total_days=30)],
            patient_age=121,
        )
    _assert_pydantic_validation_error(age_error.value)
    with pytest.raises(ValueError) as sex_error:
        PredictRequest(
            patient_id="p1",
            drugs=[DrugItem(edi_code="A001", total_days=30)],
            patient_sex="1",
        )
    _assert_pydantic_validation_error(sex_error.value)


def test_builder_without_standardizer_does_not_mutate_atc_code():
    from serving.predictor import RequestFeatureBuilder
    from serving.schemas import DrugItem

    drug = DrugItem(edi_code="A001", total_days=30)
    RequestFeatureBuilder(code_standardizer=None).build(_request(drugs=[drug]))
    assert drug.atc_code is None


def test_serving_defaults_and_all_feature_values_are_float():
    from serving.predictor import _BUILDER_KNOWN_COLS, RequestFeatureBuilder

    builder = RequestFeatureBuilder(code_standardizer=None)
    _, unknown = builder.build(_request(age=None, sex=None))
    _, male = builder.build(_request(age=50, sex="M"))
    _, female = builder.build(_request(age=50, sex="F"))

    assert unknown["age"] == 0.0
    assert unknown["sex_m"] == 0.5
    assert male["sex_m"] == 1.0
    assert female["sex_m"] == 0.0
    _, features = builder.build(
        _request(), feature_names=sorted(_BUILDER_KNOWN_COLS)
    )
    assert all(isinstance(value, float) for value in features.values())


def test_feature_vector_uses_bundle_name_order():
    from serving.predictor import RequestFeatureBuilder

    names = ["age", "drug_count", "ddi_major"]
    vector, features = RequestFeatureBuilder(code_standardizer=None).build(
        _request(), feature_names=names
    )

    assert vector.tolist() == [features[name] for name in names]
    assert all(math.isfinite(float(value)) for value in vector)
