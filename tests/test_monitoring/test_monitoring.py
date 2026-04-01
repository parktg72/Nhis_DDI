"""
monitoring/ 단위 테스트

- TestMetrics        : Prometheus 메트릭 기록 (InMemory 폴백)
- TestDriftDetector  : PSI 계산, fit/detect/save/load
- TestModelMonitor   : 성능 추이, 서브그룹, 저하 감지
- TestAlertRules     : 알림 규칙 평가 (드리프트/성능/연속하락)
- TestDashboard      : Grafana 대시보드 JSON 무결성
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ─────────────────────────────────────────────────────────────────────────────
# TestMetrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_import_without_prometheus(self):
        """prometheus_client 없어도 import 가능."""
        from monitoring.metrics import record_prediction, record_psi, record_batch
        assert callable(record_prediction)

    def test_record_prediction_increments(self):
        from monitoring.metrics import PREDICTION_TOTAL
        before = PREDICTION_TOTAL.labels(risk_level="Red", source="api").get(
            risk_level="Red", source="api"
        ) if hasattr(PREDICTION_TOTAL, "get") else None
        from monitoring.metrics import record_prediction
        record_prediction("Red", "api", 42.0)
        # InMemory 폴백에서 카운트 증가 확인
        if hasattr(PREDICTION_TOTAL, "get"):
            after = PREDICTION_TOTAL.labels(risk_level="Red", source="api").get(
                risk_level="Red", source="api"
            )
            assert after > (before or 0)

    def test_record_prediction_disagree(self):
        from monitoring.metrics import RULE_ML_DISAGREE, record_prediction
        record_prediction("Red", "api", 10.0, rule_level="Red", ml_level="Yellow")
        if hasattr(RULE_ML_DISAGREE, "get"):
            count = RULE_ML_DISAGREE.get(rule_level="Red", ml_level="Yellow")
            assert count >= 1

    def test_record_no_disagree_when_same(self):
        from monitoring.metrics import RULE_ML_DISAGREE, record_prediction
        before = RULE_ML_DISAGREE.get(rule_level="Normal", ml_level="Normal") \
            if hasattr(RULE_ML_DISAGREE, "get") else 0
        record_prediction("Normal", "batch", 5.0, rule_level="Normal", ml_level="Normal")
        after = RULE_ML_DISAGREE.get(rule_level="Normal", ml_level="Normal") \
            if hasattr(RULE_ML_DISAGREE, "get") else 0
        assert after == before  # 동일 등급이면 불일치 카운트 증가 없음

    def test_record_batch_high_risk_rate(self):
        from monitoring.metrics import HIGH_RISK_RATE, record_batch
        record_batch(100, {"Red": 10, "Yellow": 20, "Green": 30, "Normal": 40}, "20260319")
        if hasattr(HIGH_RISK_RATE, "get"):
            rate = HIGH_RISK_RATE.get(partition="20260319")
            assert abs(rate - 0.10) < 1e-6

    def test_record_psi(self):
        from monitoring.metrics import PSI_SCORE, record_psi
        record_psi("drug_count", 0.15)
        if hasattr(PSI_SCORE, "get"):
            val = PSI_SCORE.get(feature_name="drug_count")
            assert abs(val - 0.15) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# TestDriftDetector
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftDetector:
    @pytest.fixture
    def reference_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "drug_count":   np.random.normal(7, 2, 1000).clip(1, 20).astype(int),
            "ddi_major":    np.random.poisson(1.5, 1000).clip(0, 10),
            "triple_whammy": np.random.binomial(1, 0.05, 1000),
            "sex":          np.random.choice(["M", "F"], 1000),
        })

    @pytest.fixture
    def stable_df(self, reference_df):
        """기준과 유사한 데이터 (PSI < 0.10 예상)."""
        np.random.seed(99)
        return pd.DataFrame({
            "drug_count":   np.random.normal(7.1, 2.1, 800).clip(1, 20).astype(int),
            "ddi_major":    np.random.poisson(1.5, 800).clip(0, 10),
            "triple_whammy": np.random.binomial(1, 0.05, 800),
            "sex":          np.random.choice(["M", "F"], 800),
        })

    @pytest.fixture
    def drifted_df(self):
        """크게 분포가 달라진 데이터 (PSI > 0.25 예상)."""
        np.random.seed(7)
        return pd.DataFrame({
            "drug_count":   np.random.normal(14, 3, 800).clip(1, 20).astype(int),  # 평균 2배
            "ddi_major":    np.random.poisson(5.0, 800).clip(0, 10),               # 3배 증가
            "triple_whammy": np.random.binomial(1, 0.40, 800),                    # 8배 증가
            "sex":          np.random.choice(["M", "F"], 800),
        })

    def test_psi_stable_near_zero(self, reference_df, stable_df):
        from monitoring.drift_detector import DriftDetector
        det = DriftDetector()
        det.fit(reference_df)
        report = det.detect(stable_df, partition="20260319")
        drug_result = next(r for r in report.feature_results if r.feature_name == "drug_count")
        assert drug_result.psi < 0.15, f"안정 데이터 PSI가 너무 높음: {drug_result.psi}"

    def test_psi_drifted_exceeds_threshold(self, reference_df, drifted_df):
        from monitoring.drift_detector import DriftDetector
        det = DriftDetector()
        det.fit(reference_df)
        report = det.detect(drifted_df, partition="20260319")
        drifted = [r for r in report.feature_results if r.is_drifted]
        assert len(drifted) >= 1, "드리프트 데이터에서 PSI > 0.25 피처가 없음"

    def test_trigger_retrain_on_2plus_drifted(self, reference_df, drifted_df):
        from monitoring.drift_detector import DriftDetector
        det = DriftDetector()
        det.fit(reference_df)
        report = det.detect(drifted_df, partition="20260319")
        if report.n_drifted >= 2:
            assert report.trigger_retrain is True

    def test_no_trigger_retrain_on_stable(self, reference_df, stable_df):
        from monitoring.drift_detector import DriftDetector
        det = DriftDetector()
        det.fit(reference_df)
        report = det.detect(stable_df, partition="20260319")
        # 안정 데이터는 재학습 트리거 없어야 함
        assert report.trigger_retrain is False

    def test_report_summary_keys(self, reference_df, stable_df):
        from monitoring.drift_detector import DriftDetector
        det = DriftDetector()
        det.fit(reference_df)
        report = det.detect(stable_df, partition="20260319")
        for key in ("total_features", "stable", "warning", "drift", "trigger_retrain"):
            assert key in report.summary

    def test_report_to_dict(self, reference_df, stable_df):
        from monitoring.drift_detector import DriftDetector
        det = DriftDetector()
        det.fit(reference_df)
        report = det.detect(stable_df, partition="20260319")
        d = report.to_dict()
        assert d["partition"] == "20260319"
        assert "features" in d
        assert isinstance(d["features"], list)

    def test_categorical_feature_psi(self):
        from monitoring.drift_detector import compute_psi_categorical
        ref = np.array(["M"] * 50 + ["F"] * 50)
        cur = np.array(["M"] * 50 + ["F"] * 50)  # 동일
        psi, _, _, _ = compute_psi_categorical(ref, cur)
        assert psi < 0.01

    def test_categorical_psi_drifted(self):
        from monitoring.drift_detector import compute_psi_categorical
        ref = np.array(["M"] * 50 + ["F"] * 50)
        cur = np.array(["M"] * 90 + ["F"] * 10)  # 크게 편향
        psi, _, _, _ = compute_psi_categorical(ref, cur)
        assert psi > 0.10

    def test_save_load(self, reference_df, tmp_path):
        from monitoring.drift_detector import DriftDetector
        det = DriftDetector()
        det.fit(reference_df)
        path = str(tmp_path / "drift_detector.pkl")
        det.save(path)
        loaded = DriftDetector.load(path)
        assert loaded._fitted
        assert set(loaded._reference.keys()) == set(det._reference.keys())

    def test_detect_without_fit_raises(self):
        from monitoring.drift_detector import DriftDetector
        det = DriftDetector()
        with pytest.raises(RuntimeError, match="fit"):
            det.detect(pd.DataFrame({"x": [1, 2, 3]}))

    def test_save_report_creates_json(self, reference_df, stable_df, tmp_path):
        from monitoring.drift_detector import DriftDetector
        det = DriftDetector()
        det.fit(reference_df)
        report = det.detect(stable_df, partition="20260319")
        path = det.save_report(report, str(tmp_path))
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["partition"] == "20260319"


# ─────────────────────────────────────────────────────────────────────────────
# TestModelMonitor
# ─────────────────────────────────────────────────────────────────────────────

class TestModelMonitor:
    @pytest.fixture
    def monitor(self, tmp_path):
        from monitoring.model_monitor import ModelPerformanceMonitor
        return ModelPerformanceMonitor(
            baseline_recall=0.90,
            baseline_precision=0.60,
            log_dir=str(tmp_path / "monitoring"),
        )

    def test_log_prediction_accumulates(self, monitor):
        for i in range(50):
            monitor.log_prediction(
                f"P{i:03d}",
                predicted="Red",
                actual="Red",
                age=70,
                sex="M",
            )
        snaps = monitor.compute_snapshot("20260319")
        assert any(s.subgroup == "all" for s in snaps)

    def test_recall_perfect(self, monitor):
        for i in range(50):
            monitor.log_prediction(f"P{i}", predicted="Red", actual="Red")
        snaps = monitor.compute_snapshot("20260319")
        all_snap = next(s for s in snaps if s.subgroup == "all")
        assert all_snap.recall == 1.0

    def test_recall_zero_when_all_miss(self, monitor):
        for i in range(50):
            monitor.log_prediction(f"P{i}", predicted="Normal", actual="Red")
        snaps = monitor.compute_snapshot("20260319")
        all_snap = next(s for s in snaps if s.subgroup == "all")
        assert all_snap.recall == 0.0

    def test_precision_formula(self, monitor):
        # 20 TP, 20 FP → precision = 0.5 (총 40개로 MIN_SAMPLES 초과)
        for i in range(20):
            monitor.log_prediction(f"TP{i}", predicted="Red", actual="Red")
        for i in range(20):
            monitor.log_prediction(f"FP{i}", predicted="Red", actual="Normal")
        snaps = monitor.compute_snapshot("20260319")
        all_snap = next(s for s in snaps if s.subgroup == "all")
        assert abs(all_snap.precision - 0.5) < 1e-6

    def test_subgroup_age_75plus(self, monitor):
        for i in range(40):
            monitor.log_prediction(f"P{i}", predicted="Red", actual="Red", age=80)
        snaps = monitor.compute_snapshot("20260319")
        assert any(s.subgroup == "age_75plus" for s in snaps)

    def test_subgroup_sex(self, monitor):
        for i in range(40):
            monitor.log_prediction(f"P{i}", predicted="Red", actual="Red", sex="F")
        snaps = monitor.compute_snapshot("20260319")
        assert any(s.subgroup == "female" for s in snaps)

    def test_degradation_recall_warning(self, monitor):
        # recall=0.85 → 기준 0.90 대비 0.05 하락 → WARNING
        for _ in range(85):
            monitor.log_prediction("p", predicted="Red", actual="Red")
        for _ in range(15):
            monitor.log_prediction("p", predicted="Normal", actual="Red")
        snaps = monitor.compute_snapshot("20260319")
        alerts = monitor.check_degradation(snaps)
        recall_alerts = [a for a in alerts if a.metric == "recall"]
        assert len(recall_alerts) >= 1
        assert any(a.severity in ("warning", "critical") for a in recall_alerts)

    def test_no_degradation_when_recall_ok(self, monitor):
        for _ in range(92):
            monitor.log_prediction("p", predicted="Red", actual="Red")
        for _ in range(8):
            monitor.log_prediction("p", predicted="Normal", actual="Red")
        snaps = monitor.compute_snapshot("20260319")
        alerts = monitor.check_degradation(snaps)
        recall_alerts = [a for a in alerts if a.metric == "recall"]
        assert len(recall_alerts) == 0

    def test_snapshot_saved_to_file(self, monitor, tmp_path):
        for i in range(50):
            monitor.log_prediction(f"P{i}", predicted="Red", actual="Red")
        monitor.compute_snapshot("20260319")
        log_dir = tmp_path / "monitoring"
        assert (log_dir / "performance_20260319.json").exists()

    def test_minimum_samples_filter(self, monitor):
        # 30 미만이면 스냅샷 미생성
        for i in range(10):
            monitor.log_prediction(f"P{i}", predicted="Red", actual="Red")
        snaps = monitor.compute_snapshot("20260319")
        assert len(snaps) == 0

    def test_recall_trend(self, monitor):
        for partition in ["20260301", "20260308", "20260315"]:
            for i in range(50):
                monitor.log_prediction(f"P{i}", predicted="Red", actual="Red")
            monitor.compute_snapshot(partition)
        trend = monitor.get_recall_trend(subgroup="all", last_n=3)
        assert len(trend) == 3


# ─────────────────────────────────────────────────────────────────────────────
# TestAlertRules
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertRules:
    @pytest.fixture
    def manager(self, tmp_path):
        from monitoring.alert_rules import AlertManager
        return AlertManager(log_dir=str(tmp_path / "alerts"))

    def _make_drift_report(self, n_drifted=0, n_warning=0):
        """테스트용 DriftReport 생성."""
        from monitoring.drift_detector import DriftReport, FeatureDriftResult

        results = []
        for i in range(n_drifted):
            results.append(FeatureDriftResult(
                feature_name=f"feat_drift_{i}",
                psi=0.30 + i * 0.05,
                status="drift",
                reference_dist=[0.1] * 10,
                current_dist=[0.1] * 10,
                bin_edges=list(range(11)),
            ))
        for i in range(n_warning):
            results.append(FeatureDriftResult(
                feature_name=f"feat_warn_{i}",
                psi=0.15,
                status="warning",
                reference_dist=[0.1] * 10,
                current_dist=[0.1] * 10,
                bin_edges=list(range(11)),
            ))
        return DriftReport(
            partition="20260319",
            generated_at="2026-03-19T00:00:00",
            feature_results=results,
        )

    def test_critical_alert_on_2plus_drifted(self, manager):
        from monitoring.alert_rules import AlertSeverity, AlertType
        report = self._make_drift_report(n_drifted=2)
        alerts = manager.evaluate_drift(report)
        assert any(a.severity == AlertSeverity.CRITICAL for a in alerts)
        assert any(a.alert_type == AlertType.PSI_DRIFT for a in alerts)

    def test_warning_alert_on_1_warning(self, manager):
        from monitoring.alert_rules import AlertSeverity
        report = self._make_drift_report(n_drifted=0, n_warning=1)
        alerts = manager.evaluate_drift(report)
        assert any(a.severity == AlertSeverity.WARNING for a in alerts)

    def test_no_alert_on_stable(self, manager):
        report = self._make_drift_report(n_drifted=0, n_warning=0)
        alerts = manager.evaluate_drift(report)
        assert len(alerts) == 0

    def test_consecutive_drop_trigger(self, manager):
        from monitoring.alert_rules import AlertType
        from monitoring.model_monitor import PerformanceSnapshot

        # 3회 연속 하락하는 히스토리
        history = [
            PerformanceSnapshot("20260301", "all", 100, 20, tp=92, fp=30, fn=8, tn=70),
            PerformanceSnapshot("20260308", "all", 100, 20, tp=90, fp=30, fn=10, tn=70),
            PerformanceSnapshot("20260315", "all", 100, 20, tp=87, fp=30, fn=13, tn=70),
        ]
        alerts = manager.evaluate_performance([], history)
        assert any(a.alert_type == AlertType.CONSECUTIVE_DROP for a in alerts)

    def test_no_consecutive_alert_on_fluctuating(self, manager):
        from monitoring.alert_rules import AlertType
        from monitoring.model_monitor import PerformanceSnapshot

        # 오르내리는 추이 — 연속 하락 아님
        history = [
            PerformanceSnapshot("20260301", "all", 100, 20, tp=90, fp=30, fn=10, tn=70),
            PerformanceSnapshot("20260308", "all", 100, 20, tp=93, fp=30, fn=7,  tn=70),
            PerformanceSnapshot("20260315", "all", 100, 20, tp=91, fp=30, fn=9,  tn=70),
        ]
        alerts = manager.evaluate_performance([], history)
        consec = [a for a in alerts if a.alert_type == AlertType.CONSECUTIVE_DROP]
        assert len(consec) == 0

    def test_ddi_db_update_alert(self, manager):
        from monitoring.alert_rules import AlertType
        alerts = manager.evaluate_ddi_db_update("20260319", n_new_rules=5)
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.DDI_DB_UPDATE

    def test_no_ddi_update_alert_on_zero(self, manager):
        alerts = manager.evaluate_ddi_db_update("20260319", n_new_rules=0)
        assert len(alerts) == 0

    def test_evaluate_all_combines(self, manager):
        report = self._make_drift_report(n_drifted=3)
        alerts = manager.evaluate_all(
            drift_report=report,
            partition="20260319",
            n_new_ddi_rules=3,
        )
        assert len(alerts) >= 2  # PSI CRITICAL + DDI_DB_UPDATE

    def test_save_alerts_creates_json(self, manager, tmp_path):
        from monitoring.alert_rules import Alert, AlertSeverity, AlertType
        alerts = [Alert(
            alert_type=AlertType.PSI_DRIFT,
            severity=AlertSeverity.CRITICAL,
            message="테스트 알림",
            partition="20260319",
        )]
        path = manager.save_alerts(alerts, "20260319")
        assert path is not None and os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["total"] == 1
        assert data["critical"] == 1

    def test_save_empty_alerts_returns_none(self, manager):
        result = manager.save_alerts([], "20260319")
        assert result is None

    def test_alert_to_dict(self):
        from monitoring.alert_rules import Alert, AlertSeverity, AlertType
        a = Alert(
            alert_type=AlertType.RECALL_DROP,
            severity=AlertSeverity.WARNING,
            message="Recall 하락",
            partition="20260319",
            detail={"drop": 0.05},
        )
        d = a.to_dict()
        assert d["alert_type"] == "recall_drop"
        assert d["severity"] == "WARNING"
        assert d["detail"]["drop"] == 0.05


# ─────────────────────────────────────────────────────────────────────────────
# TestDashboard
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboard:
    @pytest.fixture
    def dashboard(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "monitoring", "grafana", "dashboard.json"
        )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_dashboard_loadable(self, dashboard):
        assert dashboard is not None

    def test_panel_count(self, dashboard):
        assert len(dashboard["panels"]) == 19

    def test_panel_ids_unique(self, dashboard):
        ids = [p["id"] for p in dashboard["panels"]]
        assert len(ids) == len(set(ids))

    def test_panel_titles_nonempty(self, dashboard):
        for panel in dashboard["panels"]:
            assert panel.get("title", "").strip() != "", f"Panel id={panel['id']} 제목 없음"

    def test_required_panels_present(self, dashboard):
        titles = [p["title"] for p in dashboard["panels"]]
        required_keywords = [
            "위험등급", "Recall", "PSI", "불일치", "지연", "배치", "Precision"
        ]
        for kw in required_keywords:
            assert any(kw in t for t in titles), f"'{kw}' 패널 없음"

    def test_dashboard_uid(self, dashboard):
        assert dashboard["uid"] == "ddi-monitoring-v1"

    def test_timezone_seoul(self, dashboard):
        assert dashboard["timezone"] == "Asia/Seoul"

    def test_tags_include_ddi(self, dashboard):
        assert "ddi" in dashboard["tags"]
