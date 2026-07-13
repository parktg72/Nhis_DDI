"""Source-backed characterization tests for the four operational profiles."""
from __future__ import annotations


def test_tabular_builder_feature_membership_contract():
    from serving.predictor import (
        _BUILDER_KNOWN_COLS,
        _FEATURE_ALLOWED,
        _INTENTIONAL_FEATURE_ALLOWLIST,
    )

    expected = frozenset(
        {
            "drug_count",
            "institution_count",
            "age",
            "sex_m",
            "ddi_contraindicated",
            "ddi_major",
            "ddi_moderate",
            "ddi_minor",
            "avg_drug_duration",
            "long_term_drug_count",
            "dup_same_ingredient",
            "dup_atc5",
            "dup_atc4",
            "dup_atc3",
            "dup_efmdc",
            "has_high_risk_drug",
            "has_renal_risk_drug",
            "has_hepatic_risk_drug",
            "cyp_risk_score",
            "cyp_high_risk_pairs",
            "cyp_max_enzyme_risk",
            "triple_whammy",
            "qt_risk_count",
            "drug_count_7d",
        }
    )
    assert isinstance(_BUILDER_KNOWN_COLS, frozenset)
    assert _BUILDER_KNOWN_COLS == expected
    assert _INTENTIONAL_FEATURE_ALLOWLIST == frozenset()
    assert _FEATURE_ALLOWED == expected


def test_ui_and_etl_physical_feature_order_contracts():
    from hana_app.core.ml_runner import FEATURE_COLS
    from scripts.features.feature_engineer import ETL_NUMERIC_COLS

    assert FEATURE_COLS == [
        "drug_count",
        "drug_count_7d",
        "institution_count",
        "ddi_contraindicated",
        "ddi_major",
        "ddi_moderate",
        "ddi_minor",
        "triple_whammy",
        "qt_risk_count",
        "dup_same_ingredient",
        "dup_atc5",
        "dup_atc4",
        "dup_atc3",
        "dup_efmdc",
        "has_high_risk_drug",
        "has_renal_risk_drug",
        "has_hepatic_risk_drug",
        "cyp_risk_score",
        "cyp_max_enzyme_risk",
        "cyp_high_risk_pairs",
        "age",
        "sex_m",
    ]
    assert ETL_NUMERIC_COLS == [
        "drug_count",
        "drug_count_7d",
        "institution_count",
        "ddi_contraindicated",
        "ddi_major",
        "ddi_moderate",
        "ddi_minor",
        "triple_whammy",
        "qt_risk_count",
        "dup_same_ingredient",
        "dup_atc5",
        "dup_atc4",
        "dup_atc3",
        "age",
    ]


def test_ui_row_defaults_and_raw_hana_sex_domain():
    from datetime import date

    from hana_app.core.ml_runner import _patient_features_to_row
    from scripts.etl.models import PatientFeatures

    today = date.today()
    unknown = _patient_features_to_row(
        PatientFeatures(patient_id="p1", window_start=today, window_end=today)
    )
    male = _patient_features_to_row(
        PatientFeatures(
            patient_id="p1",
            window_start=today,
            window_end=today,
            sex="1",
            age=42,
        )
    )
    female = _patient_features_to_row(
        PatientFeatures(
            patient_id="p1",
            window_start=today,
            window_end=today,
            sex="2",
            age=43,
        )
    )

    assert unknown["age"] == -1
    assert unknown["sex_m"] == 0.5
    assert unknown["sex_type"] is None
    assert male["sex_m"] == 1.0
    assert male["sex_type"] == "1"
    assert female["sex_m"] == 0.0
    assert female["sex_type"] == "2"


def test_ui_risk_label_map_contract():
    from hana_app.core.ml_runner import RISK_LABEL_MAP

    assert RISK_LABEL_MAP == {
        "Red": 3,
        "Yellow": 2,
        "Green": 1,
        "Normal": 0,
    }


def test_hierarchical_label_and_action_contract():
    from hana_app.core.hierarchical_runner import (
        ACTION_BY_LABEL,
        RED_ACTION,
        STAGE2_LABELS,
        YELLOW_SUBTYPE_LABELS,
    )

    assert YELLOW_SUBTYPE_LABELS == (
        "Y_TRIPLE",
        "Y_DOUBLE",
        "Y_DDI_MAJOR",
        "Y_DDI_MOD",
        "Y_DUP",
        "Y_FRAG",
    )
    assert STAGE2_LABELS == YELLOW_SUBTYPE_LABELS + ("No_Alert",)
    assert ACTION_BY_LABEL == {
        "Y_DDI_MAJOR": "약사 전화",
        "Y_TRIPLE": "문자 안내",
        "Y_DOUBLE": "모니터링",
        "Y_DDI_MOD": "모니터링",
        "Y_DUP": "모니터링",
        "Y_FRAG": "모니터링",
        "No_Alert": "관여 안 함",
    }
    assert RED_ACTION == "즉각 개입"
    assert "Red" not in ACTION_BY_LABEL


def test_hierarchical_dispatch_returns_probability_mapping():
    import importlib

    from hana_app.core.hierarchical_runner import STAGE2_LABELS, _dispatch_result

    probabilities = importlib.import_module("numpy").array(
        [0.1, 0.1, 0.3, 0.2, 0.1, 0.1, 0.1]
    )
    result = _dispatch_result(
        p_red=0.5,
        stage2_probs=probabilities,
        stage2_labels=STAGE2_LABELS,
        tau_red=0.7,
        tau_review=0.3,
    )

    assert result["risk_level"] == "Y_DDI_MAJOR"
    assert result["red_suspect"] is True
    assert result["stage2_probs"] == dict(zip(STAGE2_LABELS, probabilities))

    red = _dispatch_result(
        p_red=0.7,
        stage2_probs=None,
        stage2_labels=STAGE2_LABELS,
        tau_red=0.7,
        tau_review=0.3,
    )
    assert red["stage2_probs"] is None
    assert red["red_suspect"] is False


def test_stage2_labels_map_to_four_level_risk_contract():
    from hana_app.core.hierarchical_runner import YELLOW_SUBTYPE_LABELS
    from serving.predictor import HybridPredictor
    from serving.schemas import RiskLevel

    assert HybridPredictor._stage2_label_to_risk("Red") == RiskLevel.RED
    for label in YELLOW_SUBTYPE_LABELS:
        assert HybridPredictor._stage2_label_to_risk(label) == RiskLevel.YELLOW
    assert HybridPredictor._stage2_label_to_risk("No_Alert") == RiskLevel.NORMAL


def test_tabular_risk_level_serialization_order_and_intervention_contract():
    from serving.schemas import INTERVENTION_MAP, RiskLevel

    assert [(level.value, level.order) for level in RiskLevel] == [
        ("Red", 3),
        ("Yellow", 2),
        ("Green", 1),
        ("Normal", 0),
    ]
    assert INTERVENTION_MAP == {
        RiskLevel.RED: "즉각 개입",
        RiskLevel.YELLOW: "복약 상담",
        RiskLevel.GREEN: "관여 안 함",
        RiskLevel.NORMAL: "관여 안 함",
    }


def test_semantic_versions_and_threshold_contract():
    from scripts.etl.prescription_aggregator import (
        DDI_FEATURE_SEMANTICS_VERSION,
        FEATURE_SEMANTICS_VERSION,
    )
    from serving.predictor import MLModel
    from serving.schemas import RiskLevel

    assert DDI_FEATURE_SEMANTICS_VERSION == "ddi.v2"
    assert FEATURE_SEMANTICS_VERSION == "rulefeat.v1"
    model = MLModel()
    model._threshold = 0.5
    assert model.classify(0.5) == RiskLevel.RED
    assert model.classify(0.3) == RiskLevel.YELLOW
    assert model.classify(0.15) == RiskLevel.GREEN
    assert model.classify(0.149) == RiskLevel.NORMAL


def test_feature_schema_lenient_sunset_default_contract():
    from datetime import date

    from serving.predictor import _FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT

    assert _FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT == date(2026, 8, 1)


def test_dl_history_dataset_and_bundle_contracts():
    from scripts.datasets.contracts import (
        DL_BUNDLE_REQUIRED_FILES,
        DL_DATASET_REQUIRED_COLUMNS,
        LOOKBACK_DAYS_DEFAULT,
        LOOKBACK_DAYS_MAX,
        LOOKBACK_DAYS_MIN,
        ML_DATASET_REQUIRED_COLUMNS,
    )
    from serving.dl_predictor import (
        _GRAPH_ARCHITECTURES,
        _SUPPORTED_ENCODING_STRATEGIES,
    )

    assert DL_BUNDLE_REQUIRED_FILES == (
        "model.pt",
        "model_config.json",
        "drug_vocab.json",
        "edge_index.pt",
        "feature_normalizer.pkl",
        "schema_version.json",
    )
    assert DL_DATASET_REQUIRED_COLUMNS == (
        "patient_id",
        "drug_code",
        "prescription_date",
    )
    assert ML_DATASET_REQUIRED_COLUMNS == (
        "patient_id",
        "drug_count",
        "drug_count_7d",
        "institution_count",
        "ddi_contraindicated",
        "ddi_major",
        "ddi_moderate",
        "ddi_minor",
        "risk_level",
    )
    assert _SUPPORTED_ENCODING_STRATEGIES == {"multi_hot"}
    assert _GRAPH_ARCHITECTURES == {"gat", "gcn"}
    assert (LOOKBACK_DAYS_MIN, LOOKBACK_DAYS_DEFAULT, LOOKBACK_DAYS_MAX) == (7, 365, 1825)


def test_dl_output_is_optional_but_primary_risk_is_required():
    from serving.schemas import PredictResponse

    fields = PredictResponse.model_fields
    assert fields["risk_level"].is_required() is True
    assert fields["dl_prediction"].is_required() is False
    assert fields["dl_error"].is_required() is False
