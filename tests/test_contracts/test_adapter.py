"""Tests for the test-only profile difference reporter."""
from __future__ import annotations

from tests.test_contracts.profile_diff_reporter import ProfileDiffReporter


def test_diff_reports_both_sides_and_shared_features():
    reporter = ProfileDiffReporter()
    reporter.register("left", ["shared", "left_only"])
    reporter.register("right", {"shared", "right_only"})

    diff = reporter.diff("left", "right")

    assert diff.profile_a == "left"
    assert diff.profile_b == "right"
    assert diff.only_in_a == frozenset({"left_only"})
    assert diff.only_in_b == frozenset({"right_only"})
    assert diff.shared == frozenset({"shared"})


def test_register_snapshots_mutable_input():
    features = {"age", "drug_count"}
    reporter = ProfileDiffReporter()
    reporter.register("snapshot", features)
    reporter.register("comparison", {"age"})

    features.add("later")

    diff = reporter.diff("snapshot", "comparison")
    assert diff.only_in_a == frozenset({"drug_count"})


def test_diff_tabular_vs_ui_experimental():
    from hana_app.core.ml_runner import FEATURE_COLS
    from serving.predictor import _BUILDER_KNOWN_COLS

    reporter = ProfileDiffReporter()
    reporter.register("tabular_binary", _BUILDER_KNOWN_COLS)
    reporter.register("ui_experimental", FEATURE_COLS)

    diff = reporter.diff("tabular_binary", "ui_experimental")

    assert diff.only_in_a == frozenset(
        {"avg_drug_duration", "long_term_drug_count"}
    )
    assert diff.only_in_b == frozenset()
    assert len(diff.shared) == 22


def test_diff_ui_experimental_vs_etl():
    from hana_app.core.ml_runner import FEATURE_COLS
    from scripts.features.feature_engineer import ETL_NUMERIC_COLS

    reporter = ProfileDiffReporter()
    reporter.register("ui_experimental", FEATURE_COLS)
    reporter.register("etl_numeric", ETL_NUMERIC_COLS)

    diff = reporter.diff("ui_experimental", "etl_numeric")

    assert diff.only_in_a == frozenset(
        {
            "dup_efmdc",
            "has_high_risk_drug",
            "has_renal_risk_drug",
            "has_hepatic_risk_drug",
            "cyp_risk_score",
            "cyp_max_enzyme_risk",
            "cyp_high_risk_pairs",
            "sex_m",
        }
    )
    assert diff.only_in_b == frozenset()
