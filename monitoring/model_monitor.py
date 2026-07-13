"""
모델 성능 모니터링

일별 예측 결과를 누적하여 Recall/Precision 추이를 계산하고
성능 저하(3%p 이상 하락)를 감지.

지원 서브그룹:
  - all           : 전체
  - age_75plus    : 75세 이상 고령
  - male / female : 성별
  - polypharmacy  : 10종 이상 약물 복용
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

RECALL_DROP_THRESHOLD    = 0.03   # 3%p 이상 하락 시 알림
PRECISION_DROP_THRESHOLD = 0.05   # 5%p 이상 하락 시 알림
MIN_SAMPLES_FOR_EVAL     = 30     # 최소 평가 샘플 수


@dataclass
class PerformanceSnapshot:
    """단일 시점 성능 스냅샷."""
    partition: str
    subgroup: str
    n_samples: int
    n_positive: int
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "partition": self.partition,
            "subgroup": self.subgroup,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp, "fp": self.fp,
            "fn": self.fn, "tn": self.tn,
        }


@dataclass
class DegradationAlert:
    """성능 저하 알림."""
    partition: str
    subgroup: str
    metric: str          # recall | precision
    baseline: float
    current: float
    drop: float
    severity: str        # warning | critical

    def to_dict(self) -> dict:
        return {
            "partition": self.partition,
            "subgroup": self.subgroup,
            "metric": self.metric,
            "baseline": round(self.baseline, 4),
            "current": round(self.current, 4),
            "drop": round(self.drop, 4),
            "severity": self.severity,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ModelPerformanceMonitor
# ─────────────────────────────────────────────────────────────────────────────

class ModelPerformanceMonitor:
    """일별 예측 결과를 누적하여 성능 추이를 추적.

    Usage:
        monitor = ModelPerformanceMonitor(baseline_recall=0.92)
        monitor.log_prediction("P001", predicted="Red", actual="Red",
                                age=78, sex="F", drug_count=12)
        snapshot = monitor.compute_snapshot(partition="20260319")
        alerts = monitor.check_degradation(snapshot)
    """

    def __init__(
        self,
        baseline_recall: float = 0.90,
        baseline_precision: float = 0.60,
        log_dir: str = "data/monitoring",
    ):
        self._baseline_recall    = baseline_recall
        self._baseline_precision = baseline_precision
        self._log_dir = log_dir
        # 버퍼: subgroup → list of (predicted_positive, actual_positive)
        self._buffer: Dict[str, list[Tuple[bool, bool]]] = defaultdict(list)
        self._history: list[PerformanceSnapshot] = []

    def log_prediction(
        self,
        patient_id: str,
        predicted: str,
        actual: Optional[str] = None,
        age: Optional[int] = None,
        sex: Optional[str] = None,
        drug_count: Optional[int] = None,
    ) -> None:
        """단건 예측 결과 기록 (actual이 없으면 지연 레이블 처리 대기)."""
        if actual is None:
            return  # 실제 레이블 없이는 성능 계산 불가

        pred_pos   = predicted in ("Red",)
        actual_pos = actual in ("Red",)

        # 전체
        self._buffer["all"].append((pred_pos, actual_pos))

        # 서브그룹
        if age is not None and age >= 75:
            self._buffer["age_75plus"].append((pred_pos, actual_pos))
        if sex == "M":
            self._buffer["male"].append((pred_pos, actual_pos))
        elif sex == "F":
            self._buffer["female"].append((pred_pos, actual_pos))
        if drug_count is not None and drug_count >= 10:
            self._buffer["polypharmacy"].append((pred_pos, actual_pos))

    def compute_snapshot(self, partition: str) -> list[PerformanceSnapshot]:
        """버퍼에 쌓인 예측으로 성능 스냅샷 생성 후 버퍼 초기화."""
        snapshots = []
        for subgroup, pairs in self._buffer.items():
            if len(pairs) < MIN_SAMPLES_FOR_EVAL:
                continue
            tp = sum(1 for p, a in pairs if p and a)
            fp = sum(1 for p, a in pairs if p and not a)
            fn = sum(1 for p, a in pairs if not p and a)
            tn = sum(1 for p, a in pairs if not p and not a)
            n_pos = tp + fn

            snap = PerformanceSnapshot(
                partition=partition,
                subgroup=subgroup,
                n_samples=len(pairs),
                n_positive=n_pos,
                tp=tp, fp=fp, fn=fn, tn=tn,
            )
            snapshots.append(snap)
            self._history.append(snap)

        self._buffer.clear()
        self._save_snapshots(snapshots, partition)
        return snapshots

    def check_degradation(
        self, snapshots: list[PerformanceSnapshot]
    ) -> list[DegradationAlert]:
        """성능 저하 알림 생성."""
        alerts = []
        for snap in snapshots:
            # Recall 하락 감지
            recall_drop = self._baseline_recall - snap.recall
            if recall_drop >= RECALL_DROP_THRESHOLD:
                severity = "critical" if recall_drop >= 0.10 else "warning"
                alerts.append(DegradationAlert(
                    partition=snap.partition,
                    subgroup=snap.subgroup,
                    metric="recall",
                    baseline=self._baseline_recall,
                    current=snap.recall,
                    drop=recall_drop,
                    severity=severity,
                ))
                logger.warning(
                    "[%s] %s Recall 하락: %.3f → %.3f (△%.3f) [%s]",
                    snap.partition, snap.subgroup,
                    self._baseline_recall, snap.recall, recall_drop, severity,
                )

            # Precision 하락 감지
            prec_drop = self._baseline_precision - snap.precision
            if prec_drop >= PRECISION_DROP_THRESHOLD:
                severity = "critical" if prec_drop >= 0.15 else "warning"
                alerts.append(DegradationAlert(
                    partition=snap.partition,
                    subgroup=snap.subgroup,
                    metric="precision",
                    baseline=self._baseline_precision,
                    current=snap.precision,
                    drop=prec_drop,
                    severity=severity,
                ))

        return alerts

    def get_recall_trend(
        self, subgroup: str = "all", last_n: int = 30
    ) -> list[dict]:
        """최근 N개 파티션의 Recall 추이."""
        snaps = [s for s in self._history if s.subgroup == subgroup]
        return [
            {"partition": s.partition, "recall": s.recall, "n_samples": s.n_samples}
            for s in snaps[-last_n:]
        ]

    def _save_snapshots(self, snapshots: list[PerformanceSnapshot], partition: str) -> None:
        """성능 스냅샷 JSON 저장."""
        if not snapshots:
            return
        os.makedirs(self._log_dir, exist_ok=True)
        path = os.path.join(self._log_dir, f"performance_{partition}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "partition": partition,
                    "generated_at": datetime.now().isoformat(),
                    "snapshots": [s.to_dict() for s in snapshots],
                },
                f, ensure_ascii=False, indent=2,
            )
