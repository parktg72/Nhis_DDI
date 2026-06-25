"""Tests for persisted/current training result state handling."""

import json

import pandas as pd

from hana_app.core import ml_runner
from hana_app.core.ml_runner import _save_result, build_training_sequence, merge_train_results


def _result(model_name: str, f1: float = 0.8) -> dict:
    return {
        "model_name": model_name,
        "target": "risk_binary",
        "metrics": {"f1_macro": f1},
    }


def test_merge_train_results_preserves_current_ml_dl_when_hierarchical_finishes():
    existing = {
        "xgboost": _result("xgboost", 0.91),
        "tabnet": _result("tabnet", 0.86),
        "gnn": _result("gnn", 0.79),
    }
    hierarchical = {
        "hierarchical": {
            "model_name": "hierarchical",
            "target": "hierarchical",
            "metrics": {"f1_macro": 0.99},
        }
    }

    merged = merge_train_results(existing, hierarchical)

    assert list(merged) == ["xgboost", "tabnet", "gnn", "hierarchical"]
    assert merged["xgboost"]["metrics"]["f1_macro"] == 0.91
    assert merged["hierarchical"]["target"] == "hierarchical"


def test_save_result_mutates_result_with_timestamp_and_result_path(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_runner, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ml_runner, "MODELS_DIR", tmp_path / "models")
    result = {
        "model_name": "hierarchical",
        "target": "hierarchical",
        "metrics": {"f1_macro": 0.99},
        "features_df": pd.DataFrame([{"risk_level": "Yellow", "drug_count": 5}]),
    }

    _save_result(result)

    assert result["timestamp"]
    assert result["result_path"].endswith(f"result_hierarchical_{result['timestamp']}.json")
    saved = json.loads((tmp_path / "results" / f"result_hierarchical_{result['timestamp']}.json").read_text(encoding="utf-8"))
    assert saved["timestamp"] == result["timestamp"]


def test_save_result_preserves_existing_model_path_when_result_has_no_model_object(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_runner, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ml_runner, "MODELS_DIR", tmp_path / "models")
    result = {
        "model_name": "hierarchical",
        "target": "hierarchical",
        "model_path": str(tmp_path / "models" / "hierarchical" / "20260625_120000"),
        "metrics": {"f1_macro": 0.99},
    }

    _save_result(result)

    assert result["model_path"] == str(tmp_path / "models" / "hierarchical" / "20260625_120000")


def test_hierarchical_training_sequence_runs_ml_dl_comparison_before_hierarchy():
    plan = build_training_sequence(
        target="hierarchical",
        selected_models_p2=["xgboost", "lightgbm"],
        selected_models_p3=["tabnet", "gnn"],
    )

    assert plan == {
        "comparison_models": ["xgboost", "lightgbm", "tabnet", "gnn"],
        "comparison_target": "risk_binary",
        "run_hierarchical": True,
    }


def test_hierarchical_training_sequence_defaults_to_xgboost_if_user_selects_no_models():
    plan = build_training_sequence(
        target="hierarchical",
        selected_models_p2=[],
        selected_models_p3=[],
    )

    assert plan["comparison_models"] == ["xgboost"]
    assert plan["comparison_target"] == "risk_binary"
    assert plan["run_hierarchical"] is True


def test_non_hierarchical_training_sequence_keeps_user_target_and_no_hierarchy():
    plan = build_training_sequence(
        target="risk_label",
        selected_models_p2=["xgboost"],
        selected_models_p3=["tabnet"],
    )

    assert plan == {
        "comparison_models": ["xgboost", "tabnet"],
        "comparison_target": "risk_label",
        "run_hierarchical": False,
    }
