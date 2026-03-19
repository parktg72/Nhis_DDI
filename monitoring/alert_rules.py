"""
재학습 트리거 및 알림 규칙

PROJECT_PLAN 5.6 재학습 트리거 정책:
  - PSI > 0.25 피처 2개 이상      → 긴급 재학습 (CRITICAL)
  - 고위험 Recall 3%p 이상 하락   → 재학습 검토 (WARNING)
  - 3개월 연속 성능 하락 추세     → 모델 전면 재검토 (CRITICAL)
  - DDI DB 메이저 업데이트        → Rule 갱신 + 재평가 (INFO)
  - 정기 재학습 (분기 1회)        → 자동 파이프라인 (SCHEDULED)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    PSI_DRIFT          = "psi_drift"
    RECALL_DROP        = "recall_drop"
    CONSECUTIVE_DROP   = "consecutive_drop"
    DDI_DB_UPDATE      = "ddi_db_update"
    SCHEDULED_RETRAIN  = "scheduled_retrain"


@dataclass
class Alert:
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    partition: str
    detail: dict = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "partition": self.partition,
            "detail": self.detail,
            "generated_at": self.generated_at,
            "resolved": self.resolved,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 알림 규칙 정의
# ─────────────────────────────────────────────────────────────────────────────

class AlertManager:
    """알림 규칙 평가 및 알림 발행.

    Usage:
        mgr = AlertManager(log_dir="data/monitoring/alerts")
        alerts = mgr.evaluate_drift(drift_report)
        alerts += mgr.evaluate_performance(snapshots, history)
        mgr.save_alerts(alerts, partition)
    """

    def __init__(
        self,
        log_dir: str = "data/monitoring/alerts",
        psi_critical: float = 0.25,
        psi_critical_count: int = 2,
        recall_drop_warning: float = 0.03,
        recall_drop_critical: float = 0.10,
        consecutive_drop_window: int = 3,
    ):
        self._log_dir = log_dir
        self._psi_critical = psi_critical
        self._psi_critical_count = psi_critical_count
        self._recall_drop_warning = recall_drop_warning
        self._recall_drop_critical = recall_drop_critical
        self._consecutive_drop_window = consecutive_drop_window

    def evaluate_drift(self, drift_report) -> list[Alert]:
        """PSI 드리프트 리포트 평가."""
        alerts = []
        drifted = [r for r in drift_report.feature_results if r.is_drifted]

        if len(drifted) >= self._psi_critical_count:
            alerts.append(Alert(
                alert_type=AlertType.PSI_DRIFT,
                severity=AlertSeverity.CRITICAL,
                message=(
                    f"피처 드리프트 감지 — PSI > {self._psi_critical} 피처 {len(drifted)}개. "
                    "긴급 재학습 필요."
                ),
                partition=drift_report.partition,
                detail={
                    "drifted_features": [r.feature_name for r in drifted],
                    "psi_values": {r.feature_name: round(r.psi, 4) for r in drifted},
                    "trigger_retrain": True,
                },
            ))
            logger.critical(
                "[%s] 긴급 재학습 트리거: %d개 피처 드리프트 (%s)",
                drift_report.partition, len(drifted),
                [r.feature_name for r in drifted],
            )
        elif drift_report.summary.get("warning", 0) > 0:
            warning_features = [
                r for r in drift_report.feature_results if r.status == "warning"
            ]
            alerts.append(Alert(
                alert_type=AlertType.PSI_DRIFT,
                severity=AlertSeverity.WARNING,
                message=f"피처 분포 변화 감지 — 주의 피처 {len(warning_features)}개.",
                partition=drift_report.partition,
                detail={
                    "warning_features": [r.feature_name for r in warning_features],
                    "psi_values": {r.feature_name: round(r.psi, 4) for r in warning_features},
                },
            ))

        return alerts

    def evaluate_performance(
        self,
        snapshots: list,
        history: Optional[list] = None,
    ) -> list[Alert]:
        """성능 스냅샷 평가."""
        alerts = []

        for snap in snapshots:
            if snap.subgroup != "all":
                continue  # 전체 서브그룹만 트리거 평가

            # Recall 하락
            if snap.recall < (1.0 - self._recall_drop_critical):
                pass  # 절대값 기준 — 상대 기준은 DegradationAlert에서 처리

        # 연속 하락 추세 감지
        if history and len(history) >= self._consecutive_drop_window:
            all_snaps = [s for s in history if s.subgroup == "all"]
            if len(all_snaps) >= self._consecutive_drop_window:
                recent = all_snaps[-self._consecutive_drop_window:]
                recalls = [s.recall for s in recent]
                if all(recalls[i] > recalls[i + 1] for i in range(len(recalls) - 1)):
                    alerts.append(Alert(
                        alert_type=AlertType.CONSECUTIVE_DROP,
                        severity=AlertSeverity.CRITICAL,
                        message=(
                            f"{self._consecutive_drop_window}회 연속 Recall 하락 추세 감지. "
                            "모델 전면 재검토 필요."
                        ),
                        partition=recent[-1].partition if recent else "",
                        detail={
                            "recall_trend": [round(r, 4) for r in recalls],
                            "partitions": [s.partition for s in recent],
                        },
                    ))
                    logger.critical(
                        "연속 성능 하락 감지 — Recall 추이: %s",
                        [round(r, 4) for r in recalls],
                    )

        return alerts

    def evaluate_ddi_db_update(self, partition: str, n_new_rules: int) -> list[Alert]:
        """DDI DB 업데이트 감지 알림."""
        if n_new_rules == 0:
            return []
        return [Alert(
            alert_type=AlertType.DDI_DB_UPDATE,
            severity=AlertSeverity.INFO,
            message=f"DDI DB 업데이트 감지 — 신규 규칙 {n_new_rules}건. Rule 갱신 및 재평가 필요.",
            partition=partition,
            detail={"n_new_rules": n_new_rules},
        )]

    def evaluate_all(
        self,
        drift_report=None,
        snapshots=None,
        history=None,
        partition: str = "",
        n_new_ddi_rules: int = 0,
    ) -> list[Alert]:
        """전체 알림 규칙 일괄 평가."""
        alerts: list[Alert] = []
        if drift_report is not None:
            alerts += self.evaluate_drift(drift_report)
        if snapshots is not None:
            alerts += self.evaluate_performance(snapshots, history)
        if n_new_ddi_rules > 0:
            alerts += self.evaluate_ddi_db_update(partition, n_new_ddi_rules)
        return alerts

    def save_alerts(self, alerts: list[Alert], partition: str) -> Optional[str]:
        """알림 JSON 저장."""
        if not alerts:
            return None
        os.makedirs(self._log_dir, exist_ok=True)
        path = os.path.join(self._log_dir, f"alerts_{partition}.json")
        payload = {
            "partition": partition,
            "generated_at": datetime.now().isoformat(),
            "total": len(alerts),
            "critical": sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL),
            "warning":  sum(1 for a in alerts if a.severity == AlertSeverity.WARNING),
            "alerts": [a.to_dict() for a in alerts],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("알림 %d건 저장: %s", len(alerts), path)
        return path
