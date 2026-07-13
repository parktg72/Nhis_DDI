from __future__ import annotations

import json


def _write_json(path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_summary_includes_accepted_label_and_model(tmp_path) -> None:
    from scripts.ops.phase3_baseline_summary import build_phase3_baseline_summary

    linear_report = tmp_path / "linear.json"
    xgb_report = tmp_path / "xgb.json"
    ddi_report = tmp_path / "ddi.json"
    ddi_no_d000718_report = tmp_path / "ddi_no_d000718.json"
    future_audit = tmp_path / "future_audit.json"
    future_drug = tmp_path / "future_drug.json"
    future_aug = tmp_path / "future_aug.json"
    _write_json(linear_report, {
        "input_dim": 14705,
        "n_train_dataset": 67939,
        "n_val_dataset": 33879,
        "patient_overlap_count": 0,
        "train_label_positive_rate_pct": 23.9803,
        "val_label_positive_rate_pct": 22.2822,
        "train": {"val_auc": 0.84914, "val_pr_auc": 0.650496, "val_best_f1": 0.612092, "elapsed_sec": 37.451},
    })
    _write_json(xgb_report, {
        "train": {"val_auc": 0.752892, "val_pr_auc": 0.490353, "val_best_f1": 0.498509, "elapsed_sec": 106.53},
    })
    _write_json(ddi_report, {"label_positive_rate_pct": 5.8, "d_code_overlap_pct": 24.3644})
    _write_json(ddi_no_d000718_report, {"label_positive_rate_pct": 0.4, "overlap_positive_rate_pct": 0.2})
    _write_json(future_audit, {
        "n_evaluable": 42168,
        "label_positive_rate_pct": 11.9593,
        "censoring_rate_pct": 13.9522,
        "persistence_excluded_count": 16292,
        "persistence_rate_pct": 38.0615,
        "onset_type_note": "escalation-only",
    })
    _write_json(future_drug, {"train": {
        "val_auc": 0.63438,
        "val_pr_auc": 0.188207,
        "val_precision_at_top1_pct": 0.294118,
        "val_precision_at_top5_pct": 0.253555,
        "val_recall_at_top5_pct": 0.106046,
    }})
    _write_json(future_aug, {"train": {
        "val_auc": 0.64552,
        "val_pr_auc": 0.196427,
        "val_precision_at_top1_pct": 0.247059,
        "val_precision_at_top5_pct": 0.263033,
        "val_recall_at_top5_pct": 0.110010,
    }})

    summary = build_phase3_baseline_summary(
        linear_report_path=linear_report,
        xgboost_report_path=xgb_report,
        ddi_report_path=ddi_report,
        ddi_no_d000718_report_path=ddi_no_d000718_report,
        future_outcome_audit_path=future_audit,
        future_outcome_drug_only_report_path=future_drug,
        future_outcome_augmented_report_path=future_aug,
        raw_dir=tmp_path / "missing_raw",
        generated_at="2026-05-23T18:00:00+09:00",
    )

    assert summary["decision"]["accepted_label"] == "multi_institution_t6_aligned_patient_disjoint"
    assert summary["decision"]["accepted_model"] == "sparse_linear"
    assert summary["model_comparison"]["sparse_linear"]["val_auc"] == 0.84914
    assert summary["model_comparison"]["xgboost_quick"]["decision"] == "HELD"
    assert summary["feature_schema"]["input_dim"] == 14705
    assert summary["version"] == "3"
    assert summary["future_outcome_track"]["decision"] == "WEAK_FEASIBLE_RESEARCH_TRACK"
    assert summary["future_outcome_track"]["baseline_replacement"] is False
    assert summary["future_outcome_track"]["dataset"]["n_evaluable"] == 42168
    assert summary["future_outcome_track"]["models"]["augmented_oct_inst_count"]["inst_count_delta_auc"] == 0.01114
    assert "future_outcome_audit" in summary["input_data_manifest"]


def test_summary_rejected_ddi_includes_reason(tmp_path) -> None:
    from scripts.ops.phase3_baseline_summary import build_phase3_baseline_summary

    summary = build_phase3_baseline_summary(
        linear_report_path=tmp_path / "missing_linear.json",
        xgboost_report_path=tmp_path / "missing_xgb.json",
        ddi_report_path=tmp_path / "missing_ddi.json",
        ddi_no_d000718_report_path=tmp_path / "missing_ddi_no_d000718.json",
        raw_dir=tmp_path / "missing_raw",
        generated_at="2026-05-23T18:00:00+09:00",
    )

    ddi = next(item for item in summary["label_candidates"] if item["label"] == "ddi_contraindicated")
    assert ddi["decision"] == "REJECTED"
    assert "D000718" in ddi["reason"]
    note = summary["cohort_scale_note"]
    assert "153 daily record files" in note   # 07~11 same-window 코호트
    assert "patient-disjoint" in note          # aligned 60-day patient-disjoint 방법론


def test_summary_markdown_has_key_sections(tmp_path) -> None:
    from scripts.ops.phase3_baseline_summary import (
        build_phase3_baseline_summary,
        write_summary,
    )

    summary = build_phase3_baseline_summary(
        linear_report_path=tmp_path / "missing_linear.json",
        xgboost_report_path=tmp_path / "missing_xgb.json",
        ddi_report_path=tmp_path / "missing_ddi.json",
        ddi_no_d000718_report_path=tmp_path / "missing_ddi_no_d000718.json",
        raw_dir=tmp_path / "missing_raw",
        generated_at="2026-05-23T18:00:00+09:00",
    )
    json_path, md_path = write_summary(summary, tmp_path / "reports")

    assert json_path.exists()
    assert md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "# Phase 3 Baseline Summary" in md
    assert "## Decision" in md
    assert "## Model Comparison" in md
    assert "## Future Outcome Track" in md
    assert "## Rejected Or Held Candidates" in md
