"""
PSI 기반 피처 드리프트 감지기

PSI (Population Stability Index):
  PSI = Σ (실제비율 - 기준비율) × ln(실제비율 / 기준비율)

해석 기준:
  PSI < 0.10  : 안정 (no significant change)
  0.10 ≤ PSI < 0.25 : 주의 (some change — investigate)
  PSI ≥ 0.25  : 드리프트 (major shift — retrain)

PROJECT_PLAN 재학습 트리거:
  주요 피처 2개 이상 PSI > 0.25 → 긴급 재학습
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# PSI 임계값
PSI_STABLE    = 0.10
PSI_WARNING   = 0.25  # 이상 → 드리프트
N_BINS        = 10    # 연속형 피처 구간 수
EPSILON       = 1e-6  # ln(0) 방지


@dataclass
class FeatureDriftResult:
    """단일 피처 PSI 결과."""
    feature_name: str
    psi: float
    status: str          # stable | warning | drift
    reference_dist: list[float]
    current_dist:   list[float]
    bin_edges:      list[float]

    @property
    def is_drifted(self) -> bool:
        return self.psi >= PSI_WARNING


@dataclass
class DriftReport:
    """전체 드리프트 리포트."""
    partition: str
    generated_at: str
    feature_results: list[FeatureDriftResult]
    n_drifted: int = 0
    trigger_retrain: bool = False   # 주요 피처 2개 이상 PSI > 0.25
    summary: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n_drifted = sum(1 for r in self.feature_results if r.is_drifted)
        self.trigger_retrain = self.n_drifted >= 2
        self.summary = {
            "total_features": len(self.feature_results),
            "stable":  sum(1 for r in self.feature_results if r.status == "stable"),
            "warning": sum(1 for r in self.feature_results if r.status == "warning"),
            "drift":   self.n_drifted,
            "trigger_retrain": self.trigger_retrain,
            "max_psi_feature": max(
                self.feature_results, key=lambda r: r.psi, default=None
            ) and max(self.feature_results, key=lambda r: r.psi).feature_name,
        }

    def to_dict(self) -> dict:
        return {
            "partition": self.partition,
            "generated_at": self.generated_at,
            "n_drifted": self.n_drifted,
            "trigger_retrain": self.trigger_retrain,
            "summary": self.summary,
            "features": [
                {
                    "feature": r.feature_name,
                    "psi": round(r.psi, 6),
                    "status": r.status,
                }
                for r in self.feature_results
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# PSI 계산 함수
# ─────────────────────────────────────────────────────────────────────────────

def _psi_status(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    elif psi < PSI_WARNING:
        return "warning"
    return "drift"


def compute_psi_continuous(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = N_BINS,
) -> tuple[float, list[float], list[float], list[float]]:
    """연속형 피처의 PSI 계산.

    Returns:
        (psi, ref_dist, cur_dist, bin_edges)
    """
    # 기준 데이터로 bin edges 결정
    inner_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    inner_edges = np.unique(inner_edges)  # 중복 제거 (uniform 데이터 대응)

    if len(inner_edges) < 2:
        return 0.0, [], [], list(inner_edges)

    bin_edges = np.concatenate([[-np.inf], inner_edges[1:-1], [np.inf]])
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)

    ref_dist = (ref_counts / max(ref_counts.sum(), 1)).tolist()
    cur_dist = (cur_counts / max(cur_counts.sum(), 1)).tolist()

    psi = 0.0
    for r, c in zip(ref_dist, cur_dist):
        r = max(r, EPSILON)
        c = max(c, EPSILON)
        psi += (c - r) * np.log(c / r)

    return float(psi), ref_dist, cur_dist, bin_edges.tolist()


def compute_psi_categorical(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[float, list[float], list[float], list]:
    """범주형 피처의 PSI 계산."""
    categories = sorted(set(reference) | set(current))

    def freq(arr):
        total = max(len(arr), 1)
        return [np.sum(arr == c) / total for c in categories]

    ref_dist = freq(reference)
    cur_dist = freq(current)

    psi = 0.0
    for r, c in zip(ref_dist, cur_dist):
        r = max(r, EPSILON)
        c = max(c, EPSILON)
        psi += (c - r) * np.log(c / r)

    return float(psi), ref_dist, cur_dist, categories


# ─────────────────────────────────────────────────────────────────────────────
# DriftDetector 클래스
# ─────────────────────────────────────────────────────────────────────────────

class DriftDetector:
    """PSI 기반 피처 드리프트 감지기.

    Usage:
        detector = DriftDetector()
        detector.fit(reference_df)
        report = detector.detect(current_df, partition="20260319")
        detector.save(path)
    """

    def __init__(self, n_bins: int = N_BINS, categorical_threshold: int = 20):
        self._n_bins = n_bins
        self._categorical_threshold = categorical_threshold  # 고유값 수 기준
        self._reference: dict[str, np.ndarray] = {}
        self._feature_types: dict[str, str] = {}  # continuous | categorical
        self._fitted = False

    def fit(self, df) -> "DriftDetector":
        """기준 분포 학습."""
        for col in df.columns:
            arr = df[col].dropna().values
            if len(arr) == 0:
                continue
            n_unique = len(np.unique(arr))
            ftype = "categorical" if (
                n_unique <= self._categorical_threshold or
                not np.issubdtype(arr.dtype, np.number)
            ) else "continuous"
            self._reference[col] = arr
            self._feature_types[col] = ftype
        self._fitted = True
        logger.info("DriftDetector fit 완료: %d 피처", len(self._reference))
        return self

    def detect(
        self,
        df,
        partition: str = "",
        features: Optional[list[str]] = None,
    ) -> DriftReport:
        """현재 데이터와 기준 분포를 비교하여 PSI 계산."""
        if not self._fitted:
            raise RuntimeError("detect() 전에 fit()을 먼저 호출하세요.")

        target_cols = features or [c for c in df.columns if c in self._reference]
        results: list[FeatureDriftResult] = []

        for col in target_cols:
            if col not in self._reference:
                continue
            cur_arr = df[col].dropna().values
            if len(cur_arr) == 0:
                continue

            ref_arr = self._reference[col]
            ftype = self._feature_types.get(col, "continuous")

            if ftype == "continuous":
                psi, ref_dist, cur_dist, bin_edges = compute_psi_continuous(
                    ref_arr, cur_arr, self._n_bins
                )
            else:
                psi, ref_dist, cur_dist, bin_edges = compute_psi_categorical(
                    ref_arr, cur_arr
                )

            results.append(FeatureDriftResult(
                feature_name=col,
                psi=psi,
                status=_psi_status(psi),
                reference_dist=ref_dist,
                current_dist=cur_dist,
                bin_edges=bin_edges,
            ))

        report = DriftReport(
            partition=partition,
            generated_at=datetime.now().isoformat(),
            feature_results=results,
        )

        if report.trigger_retrain:
            logger.warning(
                "재학습 트리거! PSI > %.2f 피처 %d개: %s",
                PSI_WARNING,
                report.n_drifted,
                [r.feature_name for r in results if r.is_drifted],
            )
        else:
            logger.info(
                "드리프트 감지 완료 — 안정:%d, 주의:%d, 드리프트:%d",
                report.summary.get("stable", 0),
                report.summary.get("warning", 0),
                report.n_drifted,
            )

        return report

    def save(self, path: str) -> None:
        """참조 분포 저장 (pickle)."""
        import pickle
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "reference": self._reference,
                "feature_types": self._feature_types,
                "n_bins": self._n_bins,
                "categorical_threshold": self._categorical_threshold,
            }, f)

    @classmethod
    def load(cls, path: str) -> "DriftDetector":
        """저장된 참조 분포 로드."""
        import pickle
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            raise RuntimeError(f"drift_reference.pkl 로드 실패: {path}") from exc
        detector = cls(
            n_bins=data["n_bins"],
            categorical_threshold=data["categorical_threshold"],
        )
        detector._reference = data["reference"]
        detector._feature_types = data["feature_types"]
        detector._fitted = True
        return detector

    def save_report(self, report: DriftReport, log_dir: str) -> str:
        """드리프트 리포트를 JSON으로 저장."""
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"drift_{report.partition}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        return path
