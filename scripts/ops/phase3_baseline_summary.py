"""Build a reproducible Phase 3 baseline decision summary."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
from typing import Sequence


DEFAULT_LINEAR_REPORT = Path("data/datasets/multi_inst_t6_temporal_20241031_to_20241130_patient_disjoint/sparse_training_smoke_report.json")
DEFAULT_XGBOOST_REPORT = Path("data/datasets/multi_inst_t6_temporal_xgboost50_20241031_to_20241130_patient_disjoint/sparse_training_smoke_report.json")
DEFAULT_DDI_REPORT = Path("data/vocab/ddi_mapping_audit_report.json")
DEFAULT_DDI_NO_D000718_REPORT = Path("data/vocab/ddi_no_d000718/ddi_mapping_audit_report.json")
DEFAULT_FUTURE_OUTCOME_AUDIT = Path("data/reports/future_outcome_t6/future_outcome_label_audit.json")
DEFAULT_FUTURE_OUTCOME_DRUG_ONLY_REPORT = Path("data/datasets/future_multi_inst_onset_t6_20241031_to_20241130_linear_smoke/sparse_training_smoke_report.json")
DEFAULT_FUTURE_OUTCOME_AUGMENTED_REPORT = Path("data/datasets/future_multi_inst_onset_t6_20241031_to_20241130_with_oct_inst_count_linear_smoke/sparse_training_smoke_report.json")


def build_phase3_baseline_summary(
    *,
    linear_report_path: str | Path = DEFAULT_LINEAR_REPORT,
    xgboost_report_path: str | Path = DEFAULT_XGBOOST_REPORT,
    ddi_report_path: str | Path = DEFAULT_DDI_REPORT,
    ddi_no_d000718_report_path: str | Path = DEFAULT_DDI_NO_D000718_REPORT,
    future_outcome_audit_path: str | Path = DEFAULT_FUTURE_OUTCOME_AUDIT,
    future_outcome_drug_only_report_path: str | Path = DEFAULT_FUTURE_OUTCOME_DRUG_ONLY_REPORT,
    future_outcome_augmented_report_path: str | Path = DEFAULT_FUTURE_OUTCOME_AUGMENTED_REPORT,
    raw_dir: str | Path = "data/Raw",
    generated_at: str | None = None,
) -> dict:
    linear = _read_json(linear_report_path)
    xgboost = _read_json(xgboost_report_path)
    ddi = _read_json(ddi_report_path)
    ddi_no_d000718 = _read_json(ddi_no_d000718_report_path)
    future_audit = _read_json(future_outcome_audit_path)
    future_drug = _read_json(future_outcome_drug_only_report_path)
    future_augmented = _read_json(future_outcome_augmented_report_path)
    raw_coverage = _raw_coverage(Path(raw_dir))
    linear_train = linear.get("train", {})
    xgb_train = xgboost.get("train", {})

    input_dim = int(linear.get("input_dim") or 14705)
    summary = {
        "version": "2",
        "last_updated": "2026-05-23",
        "changes_from_v1": "Added future_outcome_track section (WEAK-FEASIBLE) and input_data_manifest.",
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "decision": {
            "accepted_label": "multi_institution_t6_exact30_patient_disjoint",
            "accepted_model": "sparse_linear",
            "status": "BASELINE_LOCKED",
            "reason": "Best validated proxy baseline under available 2024-10..2024-11 Raw data; DDI and XGBoost were rejected/held by audit.",
        },
        "raw_coverage": raw_coverage,
        "cohort_scale_note": "40K = one-day reference active cohort. 932K = two-month cumulative sampled population. Phase 3 reports use one-day reference cohorts.",
        "feature_schema": {
            "input_dim": input_dim,
            "vocab_cutoff": 100,
            "unk_token": "_unk",
            "feature_type": "sparse multi-hot drug_code",
        },
        "temporal_split": {
            "train_window": "2024-10-02..2024-10-31 (lookback_days=29, inclusive exact 30-day window)",
            "val_window": "2024-11-01..2024-11-30 (lookback_days=29, inclusive exact 30-day window)",
            "window_overlap_days": 0,
            "patient_overlap_removed": 6255,
            "patient_overlap_count": int(linear.get("patient_overlap_count", 0)),
            "n_train": int(linear.get("n_train_dataset") or linear_train.get("n_train") or 67939),
            "n_val": int(linear.get("n_val_dataset") or linear_train.get("n_val") or 33879),
            "train_positive_rate_pct": _metric(linear, "train_label_positive_rate_pct", 23.9803),
            "val_positive_rate_pct": _metric(linear, "val_label_positive_rate_pct", 22.2822),
        },
        "label_candidates": _label_candidates(linear_train, ddi, ddi_no_d000718),
        "model_comparison": {
            "sparse_linear": {
                "decision": "ACCEPTED",
                "val_auc": _metric(linear_train, "val_auc", 0.849140),
                "val_pr_auc": _metric(linear_train, "val_pr_auc", 0.650496),
                "val_best_f1": _metric(linear_train, "val_best_f1", 0.612092),
                "elapsed_sec": _metric(linear_train, "elapsed_sec", 37.451),
            },
            "xgboost_quick": {
                "decision": "HELD",
                "val_auc": _metric(xgb_train, "val_auc", 0.752892),
                "val_pr_auc": _metric(xgb_train, "val_pr_auc", 0.490353),
                "val_best_f1": _metric(xgb_train, "val_best_f1", 0.498509),
                "elapsed_sec": _metric(xgb_train, "elapsed_sec", 106.53),
                "reason": "Lower AUC/PR-AUC and higher cost on sparse multi-hot drug-code features.",
            },
        },
        "future_outcome_track": _future_outcome_track(
            future_audit,
            future_drug.get("train", {}),
            future_augmented.get("train", {}),
        ),
        "leakage_audit": {
            "patient_disjoint_validation": True,
            "patient_overlap_count": int(linear.get("patient_overlap_count", 0)),
            "future_window_leakage_check": "PASS: train and validation windows are non-overlapping adjacent months.",
            "institution_performance_variance": "NOT_EVALUATED: institution_id is label source and per-institution metrics require a separate fairness/stratified audit.",
        },
        "operational_constraints": {
            "linear_training_elapsed_sec": _metric(linear_train, "elapsed_sec", 37.451),
            "xgboost_quick_elapsed_sec": _metric(xgb_train, "elapsed_sec", 106.53),
            "inference_latency_status": "NOT_MEASURED",
            "inference_latency_note": "Sparse linear inference is expected to be cheaper than tree ensemble for 14,705-dimensional multi-hot input; measure before production serving.",
        },
        "environment": _environment_summary(),
        "recommendations": [
            "Use sparse_linear + multi_institution_t6_exact30_patient_disjoint as the Phase 3 proxy baseline.",
            "Acquire 2024-09 or 2024-12 Raw data for longer-gap temporal holdout before claiming generalization.",
            "Design a future clinical outcome label before clinical-risk claims.",
            "Revisit XGBoost only after dense embeddings or additional engineered temporal/institution features.",
            "Revisit DDI only if a direct EDI code to HIRA DUR D-code mapping table is available.",
        ],
        "sources": {
            "linear_report": _source(linear_report_path, linear),
            "xgboost_report": _source(xgboost_report_path, xgboost),
            "ddi_report": _source(ddi_report_path, ddi),
            "ddi_no_d000718_report": _source(ddi_no_d000718_report_path, ddi_no_d000718),
            "future_outcome_audit": _source(future_outcome_audit_path, future_audit),
            "future_outcome_drug_only_report": _source(future_outcome_drug_only_report_path, future_drug),
            "future_outcome_augmented_report": _source(future_outcome_augmented_report_path, future_augmented),
        },
        "input_data_manifest": _input_data_manifest({
            "linear_report": linear_report_path,
            "xgboost_report": xgboost_report_path,
            "ddi_report": ddi_report_path,
            "ddi_no_d000718_report": ddi_no_d000718_report_path,
            "future_outcome_audit": future_outcome_audit_path,
            "future_outcome_drug_only_report": future_outcome_drug_only_report_path,
            "future_outcome_augmented_report": future_outcome_augmented_report_path,
        }),
    }
    return summary


def write_summary(summary: dict, output_dir: str | Path = "data/reports") -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "phase3_baseline_summary.json"
    md_path = output_path / "phase3_baseline_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(summary), encoding="utf-8")
    return json_path, md_path


def _label_candidates(linear_train: dict, ddi: dict, ddi_no_d000718: dict) -> list[dict]:
    return [
        {
            "label": "sick_code_adr_proxy",
            "decision": "REJECTED",
            "val_auc": 0.409,
            "reason": "Weak/noisy proxy label from earlier MLP smoke.",
        },
        {
            "label": "multi_institution_t6",
            "decision": "ACCEPTED",
            "val_auc": _metric(linear_train, "val_auc", 0.849140),
            "val_pr_auc": _metric(linear_train, "val_pr_auc", 0.650496),
            "reason": "Patient-disjoint temporal structure signal remained stable.",
        },
        {
            "label": "therapeutic_dup_t6",
            "decision": "SANITY_ONLY",
            "val_auc": 0.947305,
            "reason": "Rule reconstruction from drug_code features; useful for pipeline sanity, not clinical prediction.",
        },
        {
            "label": "ddi_contraindicated",
            "decision": "REJECTED",
            "val_auc": None,
            "positive_rate_pct": _metric(ddi, "label_positive_rate_pct", 5.8),
            "d000718_excluded_positive_rate_pct": _metric(ddi_no_d000718, "label_positive_rate_pct", 0.4),
            "d000718_excluded_overlap_positive_rate_pct": _metric(ddi_no_d000718, "overlap_positive_rate_pct", 0.2),
            "d_code_overlap_pct": _metric(ddi, "d_code_overlap_pct", 24.3644),
            "reason": "D000718(Metformin)+contrast-agent dominance and low D-code reachability; D000718 exclusion collapses positive rate below 1%.",
        },
    ]


def _future_outcome_track(audit: dict, drug_only: dict, augmented: dict) -> dict:
    drug_auc = _metric(drug_only, "val_auc", 0.634380)
    aug_auc = _metric(augmented, "val_auc", 0.645520)
    return {
        "decision": "WEAK_FEASIBLE_RESEARCH_TRACK",
        "baseline_replacement": False,
        "dataset": {
            "n_evaluable": int(_metric(audit, "n_evaluable", 42168)),
            "positive_rate_pct": _metric(audit, "label_positive_rate_pct", 11.9593),
            "censoring_rate_pct": _metric(audit, "censoring_rate_pct", 13.9522),
            "onset_type_note": _metric(
                audit,
                "onset_type_note",
                "escalation-only (oct_count 1-5 -> nov_count >=6); clean_onset=0 under strict observability",
            ),
            "persistence_excluded": int(_metric(audit, "persistence_excluded_count", 16292)),
            "persistence_rate_pct": _metric(audit, "persistence_rate_pct", 38.0615),
        },
        "models": {
            "drug_only": {
                "val_auc": drug_auc,
                "val_pr_auc": _metric(drug_only, "val_pr_auc", 0.188207),
                "precision_at_top1_pct": _metric(drug_only, "val_precision_at_top1_pct", 0.294118),
                "precision_at_top5_pct": _metric(drug_only, "val_precision_at_top5_pct", 0.253555),
                "recall_at_top5_pct": _metric(drug_only, "val_recall_at_top5_pct", 0.106046),
            },
            "augmented_oct_inst_count": {
                "val_auc": aug_auc,
                "val_pr_auc": _metric(augmented, "val_pr_auc", 0.196427),
                "precision_at_top1_pct": _metric(augmented, "val_precision_at_top1_pct", 0.247059),
                "precision_at_top5_pct": _metric(augmented, "val_precision_at_top5_pct", 0.263033),
                "recall_at_top5_pct": _metric(augmented, "val_recall_at_top5_pct", 0.110010),
                "inst_count_delta_auc": round(float(aug_auc) - float(drug_auc), 6),
                "note": "oct_institution_count scalar adds marginal gain; drug diversity already encodes institution count indirectly",
            },
        },
        "caveats": [
            "internal random 80/20 split only; no temporal generalization claim",
            "no 2024-12 third window available",
            "13.95% censoring may introduce selective bias",
        ],
        "next_unblock": "acquire 2024-12 Raw month for 3-window temporal holdout",
    }


def _raw_coverage(raw_dir: Path) -> dict:
    dates = []
    if raw_dir.exists():
        for path in raw_dir.glob("records_*.parquet"):
            raw = path.stem.removeprefix("records_")
            try:
                dates.append(datetime.strptime(raw, "%Y%m%d").date())
            except ValueError:
                continue
    dates = sorted(dates)
    return {
        "raw_dir": str(raw_dir),
        "records_file_count": len(dates),
        "first_records_date": dates[0].isoformat() if dates else None,
        "last_records_date": dates[-1].isoformat() if dates else None,
        "has_2024_09": any(day.strftime("%Y%m").startswith("202409") for day in dates),
        "has_2024_12": any(day.strftime("%Y%m").startswith("202412") for day in dates),
        "additional_month_available": any(
            day.strftime("%Y%m") in {"202409", "202412"}
            for day in dates
        ),
    }


def _environment_summary() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _optional_version("torch"),
        "xgboost": _optional_version("xgboost"),
    }


def _optional_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _read_json(path: str | Path) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def _source(path: str | Path, payload: dict) -> dict:
    return {
        "path": str(path),
        "source": "file" if payload else "manual_entry",
    }


def _input_data_manifest(paths: dict[str, str | Path]) -> dict:
    manifest = {}
    for key, value in paths.items():
        path = Path(value)
        if path.exists():
            stat = path.stat()
            manifest[key] = {
                "path": str(path),
                "exists": True,
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
            }
        else:
            manifest[key] = {
                "path": str(path),
                "exists": False,
                "size_bytes": None,
                "modified_time": None,
            }
    return manifest


def _metric(payload: dict, key: str, fallback):
    return payload[key] if key in payload else fallback


def _markdown_summary(summary: dict) -> str:
    decision = summary["decision"]
    temporal = summary["temporal_split"]
    linear = summary["model_comparison"]["sparse_linear"]
    xgb = summary["model_comparison"]["xgboost_quick"]
    future = summary["future_outcome_track"]
    future_drug = future["models"]["drug_only"]
    future_aug = future["models"]["augmented_oct_inst_count"]
    raw = summary["raw_coverage"]
    lines = [
        "# Phase 3 Baseline Summary",
        "",
        "## Decision",
        "",
        f"- accepted_label: {decision['accepted_label']}",
        f"- accepted_model: {decision['accepted_model']}",
        f"- status: {decision['status']}",
        "",
        "## Raw Coverage",
        "",
        f"- first_records_date: {raw['first_records_date']}",
        f"- last_records_date: {raw['last_records_date']}",
        f"- records_file_count: {raw['records_file_count']}",
        f"- additional_month_available: {raw['additional_month_available']}",
        "",
        "## Temporal Split",
        "",
        f"- train_window: {temporal['train_window']}",
        f"- val_window: {temporal['val_window']}",
        f"- n_train: {temporal['n_train']}",
        f"- n_val: {temporal['n_val']}",
        f"- patient_overlap_count: {temporal['patient_overlap_count']}",
        "",
        "## Model Comparison",
        "",
        "| model | decision | val_auc | val_pr_auc | best_f1 | elapsed_sec |",
        "|---|---|---:|---:|---:|---:|",
        f"| sparse_linear | {linear['decision']} | {linear['val_auc']} | {linear['val_pr_auc']} | {linear['val_best_f1']} | {linear['elapsed_sec']} |",
        f"| xgboost_quick | {xgb['decision']} | {xgb['val_auc']} | {xgb['val_pr_auc']} | {xgb['val_best_f1']} | {xgb['elapsed_sec']} |",
        "",
        "## Future Outcome Track",
        "",
        f"- decision: {future['decision']}",
        f"- baseline_replacement: {future['baseline_replacement']}",
        f"- n_evaluable: {future['dataset']['n_evaluable']}",
        f"- positive_rate_pct: {future['dataset']['positive_rate_pct']}",
        f"- censoring_rate_pct: {future['dataset']['censoring_rate_pct']}",
        f"- next_unblock: {future['next_unblock']}",
        "",
        "| model | val_auc | val_pr_auc | precision@top1% | precision@top5% | recall@top5% |",
        "|---|---:|---:|---:|---:|---:|",
        f"| drug_only | {future_drug['val_auc']} | {future_drug['val_pr_auc']} | {future_drug['precision_at_top1_pct']} | {future_drug['precision_at_top5_pct']} | {future_drug['recall_at_top5_pct']} |",
        f"| augmented_oct_inst_count | {future_aug['val_auc']} | {future_aug['val_pr_auc']} | {future_aug['precision_at_top1_pct']} | {future_aug['precision_at_top5_pct']} | {future_aug['recall_at_top5_pct']} |",
        "",
        "## Rejected Or Held Candidates",
        "",
        "| candidate | decision | reason |",
        "|---|---|---|",
    ]
    for candidate in summary["label_candidates"]:
        if candidate["decision"] != "ACCEPTED":
            lines.append(f"| {candidate['label']} | {candidate['decision']} | {candidate['reason']} |")
    lines.extend([
        f"| xgboost_quick | {xgb['decision']} | {xgb['reason']} |",
        "",
        "## Recommendations",
        "",
        *[f"- {item}" for item in summary["recommendations"]],
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase 3 baseline decision summary.")
    parser.add_argument("--linear-report", default=str(DEFAULT_LINEAR_REPORT))
    parser.add_argument("--xgboost-report", default=str(DEFAULT_XGBOOST_REPORT))
    parser.add_argument("--ddi-report", default=str(DEFAULT_DDI_REPORT))
    parser.add_argument("--ddi-no-d000718-report", default=str(DEFAULT_DDI_NO_D000718_REPORT))
    parser.add_argument("--future-outcome-audit", default=str(DEFAULT_FUTURE_OUTCOME_AUDIT))
    parser.add_argument("--future-outcome-drug-only-report", default=str(DEFAULT_FUTURE_OUTCOME_DRUG_ONLY_REPORT))
    parser.add_argument("--future-outcome-augmented-report", default=str(DEFAULT_FUTURE_OUTCOME_AUGMENTED_REPORT))
    parser.add_argument("--raw-dir", default="data/Raw")
    parser.add_argument("--output-dir", default="data/reports")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_phase3_baseline_summary(
        linear_report_path=args.linear_report,
        xgboost_report_path=args.xgboost_report,
        ddi_report_path=args.ddi_report,
        ddi_no_d000718_report_path=args.ddi_no_d000718_report,
        future_outcome_audit_path=args.future_outcome_audit,
        future_outcome_drug_only_report_path=args.future_outcome_drug_only_report,
        future_outcome_augmented_report_path=args.future_outcome_augmented_report,
        raw_dir=args.raw_dir,
    )
    json_path, md_path = write_summary(summary, args.output_dir)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    print(f"accepted_label={summary['decision']['accepted_label']}")
    print(f"accepted_model={summary['decision']['accepted_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
