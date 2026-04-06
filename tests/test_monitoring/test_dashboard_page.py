"""Streamlit 대시보드 페이지 헬퍼 함수 테스트."""
from __future__ import annotations

import json
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestDashboardHelpers:
    def test_load_recent_metrics_returns_list(self, tmp_path):
        from hana_app.pages._monitoring_helpers import load_recent_metrics
        ts = datetime.now(timezone.utc).isoformat()
        path = tmp_path / "metrics.jsonl"
        _write_jsonl(path, [{"timestamp": ts, "patient_id": "P001", "risk_level": "RED", "disagree": False}])
        records = load_recent_metrics(path, hours=24)
        assert len(records) == 1
        assert records[0]["patient_id"] == "P001"

    def test_load_recent_metrics_file_missing_returns_empty(self, tmp_path):
        from hana_app.pages._monitoring_helpers import load_recent_metrics
        records = load_recent_metrics(tmp_path / "nonexistent.jsonl", hours=24)
        assert records == []

    def test_load_drift_report_returns_dict(self, tmp_path):
        from hana_app.pages._monitoring_helpers import load_drift_report
        partition = "2026-04-06"
        report = {
            "partition": partition,
            "n_drifted": 1,
            "summary": {"stable": 1, "warning": 0, "drift": 1},
            "features": [{"feature": "drug_count", "psi": 0.30, "status": "drift"}],
        }
        (tmp_path / f"drift_{partition}.json").write_text(json.dumps(report))
        loaded = load_drift_report(tmp_path, partition)
        assert loaded["n_drifted"] == 1
        assert loaded["features"][0]["status"] == "drift"

    def test_load_drift_report_missing_returns_none(self, tmp_path):
        from hana_app.pages._monitoring_helpers import load_drift_report
        result = load_drift_report(tmp_path, "2026-04-06")
        assert result is None

    def test_load_alerts_returns_list(self, tmp_path):
        from hana_app.pages._monitoring_helpers import load_alerts
        partition = "2026-04-06"
        alerts = [
            {"alert_type": "psi_drift", "severity": "CRITICAL", "message": "드리프트", "generated_at": "2026-04-06T00:00:00"},
        ]
        (tmp_path / f"alerts_{partition}.json").write_text(json.dumps(alerts))
        result = load_alerts(tmp_path, [partition])
        assert len(result) == 1
        assert result[0]["severity"] == "CRITICAL"

    def test_load_alerts_no_files_returns_empty(self, tmp_path):
        from hana_app.pages._monitoring_helpers import load_alerts
        result = load_alerts(tmp_path, ["2026-04-06"])
        assert result == []

    def test_compute_disagree_rate_zero_records(self):
        from hana_app.pages._monitoring_helpers import compute_disagree_rate
        assert compute_disagree_rate([]) == 0.0

    def test_compute_disagree_rate_correct(self):
        from hana_app.pages._monitoring_helpers import compute_disagree_rate
        records = [
            {"disagree": True},
            {"disagree": False},
            {"disagree": True},
            {"disagree": False},
        ]
        assert abs(compute_disagree_rate(records) - 0.5) < 1e-9

    def test_psi_status_label(self):
        from hana_app.pages._monitoring_helpers import psi_status_label
        assert psi_status_label(0.05) == "🟢 Stable"
        assert psi_status_label(0.15) == "🟡 Warning"
        assert psi_status_label(0.30) == "🔴 Drift"

    def test_get_recent_partitions(self, tmp_path):
        from hana_app.pages._monitoring_helpers import get_recent_partitions
        for p in ["2026-04-04", "2026-04-05", "2026-04-06"]:
            (tmp_path / f"drift_{p}.json").write_text("{}")
        result = get_recent_partitions(tmp_path, prefix="drift_", n=7)
        assert "2026-04-06" in result
        assert len(result) <= 7
