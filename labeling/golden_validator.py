"""
골든 데이터셋 품질 검증기

검증 항목:
  1. 완결성  : 필수 컬럼 존재, 레이블 누락률 < 5%
  2. 균형성  : 위험도별 분포 (목표 비율 ±8%)
  3. 일치율  : Rule 레이블 vs 전문가 레이블 일치율 ≥ 70%
  4. 신뢰도  : Cohen's Kappa ≥ 0.70 (다중 검토자 있을 때)
  5. 규모    : 목표 건수 (5,650건) 대비 달성률

검증 통과 기준 (모두 만족 시 골든 데이터셋으로 확정):
  - 레이블 누락률 < 5%
  - 일치율 ≥ 70%
  - 목표 건수 80% 이상 달성 (4,520건↑)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

GOLDEN_TARGET            = 5_650
MIN_COMPLETION_RATE      = 0.95   # 레이블 완결률 95% 이상
MIN_AGREEMENT_RATE       = 0.70   # Rule vs 전문가 일치율 70% 이상
MIN_KAPPA                = 0.70   # Cohen's Kappa 70% 이상
MIN_COVERAGE             = 0.80   # 목표 건수 대비 80% 이상
RISK_TOLERANCE           = 0.08   # 위험도 비율 허용 오차 ±8%

REQUIRED_COLUMNS = {
    "patient_id", "risk_level", "expert_label", "review_status",
}
LABEL_VALUES = {"Red", "Yellow", "Green", "Normal"}


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    value: float
    threshold: float
    message: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": round(self.value, 4),
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass
class GoldenValidationReport:
    """골든 데이터셋 검증 결과."""
    total_records: int
    reviewed_records: int
    checks: list[ValidationCheck]
    kappa: Optional[float]
    distribution: Dict[str, int]
    passed: bool = False

    def __post_init__(self):
        self.passed = all(c.passed for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "total_records":    self.total_records,
            "reviewed_records": self.reviewed_records,
            "passed":           self.passed,
            "kappa":            round(self.kappa, 4) if self.kappa is not None else None,
            "distribution":     self.distribution,
            "checks":           [c.to_dict() for c in self.checks],
        }

    def print_summary(self) -> None:
        status = "PASS" if self.passed else "FAIL"
        print(f"\n골든 데이터셋 검증 결과: [{status}]")
        print(f"  전체: {self.total_records}건 / 검토완료: {self.reviewed_records}건")
        if self.kappa is not None:
            print(f"  Cohen's Kappa: {self.kappa:.4f}")
        print("  위험도 분포:")
        for level, cnt in self.distribution.items():
            pct = cnt / max(self.total_records, 1) * 100
            print(f"    {level:8s}: {cnt:5d}건 ({pct:.1f}%)")
        print("  검증 항목:")
        for check in self.checks:
            mark = "✓" if check.passed else "✗"
            print(f"    [{mark}] {check.name}: {check.value:.4f} (기준: {check.threshold})")


# ─────────────────────────────────────────────────────────────────────────────
# Cohen's Kappa
# ─────────────────────────────────────────────────────────────────────────────

def compute_cohens_kappa(labels_a: list, labels_b: list) -> float:
    """두 검토자 간 Cohen's Kappa 계산.

    sklearn 없이 numpy만으로 구현.
    """
    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)

    # 관측 일치율
    p_o = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    # 기대 일치율
    p_e = 0.0
    for cat in categories:
        p_a = sum(1 for x in labels_a if x == cat) / n
        p_b = sum(1 for x in labels_b if x == cat) / n
        p_e += p_a * p_b

    if abs(1.0 - p_e) < 1e-9:
        return 1.0

    return (p_o - p_e) / (1.0 - p_e)


def compute_agreement_rate(labels_a: list, labels_b: list) -> float:
    """단순 일치율 (정확히 일치하는 비율)."""
    if not labels_a:
        return 0.0
    return sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)


# ─────────────────────────────────────────────────────────────────────────────
# GoldenValidator
# ─────────────────────────────────────────────────────────────────────────────

class GoldenValidator:
    """골든 데이터셋 품질 검증기.

    Usage:
        validator = GoldenValidator()
        report = validator.validate(golden_df)
        report.print_summary()
    """

    def __init__(
        self,
        target: int = GOLDEN_TARGET,
        min_completion: float = MIN_COMPLETION_RATE,
        min_agreement: float = MIN_AGREEMENT_RATE,
        min_kappa: float = MIN_KAPPA,
        min_coverage: float = MIN_COVERAGE,
        risk_ratios: Optional[Dict[str, float]] = None,
    ):
        self._target = target
        self._min_completion  = min_completion
        self._min_agreement   = min_agreement
        self._min_kappa       = min_kappa
        self._min_coverage    = min_coverage
        self._risk_ratios = risk_ratios or {
            "Red": 0.30, "Yellow": 0.30, "Green": 0.25, "Normal": 0.15,
        }

    def validate(
        self,
        df: pd.DataFrame,
        rule_label_col: str = "risk_level",
        expert_label_col: str = "expert_label",
        reviewer2_col: Optional[str] = None,
    ) -> GoldenValidationReport:
        """전체 검증 실행."""
        checks: list[ValidationCheck] = []
        reviewed = df[df["review_status"] == "reviewed"] if "review_status" in df.columns else df

        # 1. 필수 컬럼 검사
        checks.append(self._check_required_columns(df))

        # 2. 레이블 완결률
        checks.append(self._check_completion(reviewed, expert_label_col))

        # 3. 목표 규모 달성률
        checks.append(self._check_coverage(reviewed))

        # 4. 위험도 분포 균형성
        checks.append(self._check_distribution(reviewed, expert_label_col))

        # 5. Rule vs 전문가 일치율
        checks.append(self._check_agreement(reviewed, rule_label_col, expert_label_col))

        # 6. Cohen's Kappa (2차 검토자 있을 때)
        kappa = None
        if reviewer2_col and reviewer2_col in reviewed.columns:
            checks.append(self._check_kappa(reviewed, expert_label_col, reviewer2_col))
            labels_a = reviewed[expert_label_col].dropna().tolist()
            labels_b = reviewed[reviewer2_col].dropna().tolist()
            min_len = min(len(labels_a), len(labels_b))
            kappa = compute_cohens_kappa(labels_a[:min_len], labels_b[:min_len])

        # 레이블 분포
        dist: Dict[str, int] = {}
        if expert_label_col in reviewed.columns:
            dist = reviewed[expert_label_col].value_counts().to_dict()

        return GoldenValidationReport(
            total_records=len(df),
            reviewed_records=len(reviewed),
            checks=checks,
            kappa=kappa,
            distribution=dist,
        )

    # ── 개별 검증 메서드 ──────────────────────────────────────────────────────

    def _check_required_columns(self, df: pd.DataFrame) -> ValidationCheck:
        missing = REQUIRED_COLUMNS - set(df.columns)
        passed = len(missing) == 0
        return ValidationCheck(
            name="필수 컬럼 존재",
            passed=passed,
            value=float(len(df.columns)),
            threshold=float(len(REQUIRED_COLUMNS)),
            message="OK" if passed else f"누락 컬럼: {missing}",
        )

    def _check_completion(self, df: pd.DataFrame, col: str) -> ValidationCheck:
        if col not in df.columns or len(df) == 0:
            return ValidationCheck("레이블 완결률", False, 0.0, self._min_completion, "컬럼 없음")
        valid = df[col].isin(LABEL_VALUES).sum()
        rate = valid / len(df)
        return ValidationCheck(
            name="레이블 완결률",
            passed=rate >= self._min_completion,
            value=rate,
            threshold=self._min_completion,
            message=f"유효 레이블 {valid}/{len(df)}건",
        )

    def _check_coverage(self, df: pd.DataFrame) -> ValidationCheck:
        coverage = len(df) / self._target
        return ValidationCheck(
            name="목표 건수 달성률",
            passed=coverage >= self._min_coverage,
            value=coverage,
            threshold=self._min_coverage,
            message=f"{len(df)}/{self._target}건",
        )

    def _check_distribution(self, df: pd.DataFrame, col: str) -> ValidationCheck:
        if col not in df.columns or len(df) == 0:
            return ValidationCheck("위험도 분포 균형", False, 0.0, 1 - RISK_TOLERANCE, "컬럼 없음")
        dist = df[col].value_counts(normalize=True).to_dict()
        max_deviation = max(
            abs(dist.get(level, 0.0) - ratio)
            for level, ratio in self._risk_ratios.items()
        )
        passed = max_deviation <= RISK_TOLERANCE
        return ValidationCheck(
            name="위험도 분포 균형",
            passed=passed,
            value=1.0 - max_deviation,
            threshold=1.0 - RISK_TOLERANCE,
            message=f"최대 편차 {max_deviation:.4f} (허용 {RISK_TOLERANCE})",
        )

    def _check_agreement(self, df: pd.DataFrame, rule_col: str, expert_col: str) -> ValidationCheck:
        if rule_col not in df.columns or expert_col not in df.columns:
            return ValidationCheck("Rule-전문가 일치율", False, 0.0, self._min_agreement, "컬럼 없음")
        valid = df[df[expert_col].isin(LABEL_VALUES) & df[rule_col].isin(LABEL_VALUES)]
        if len(valid) == 0:
            return ValidationCheck("Rule-전문가 일치율", False, 0.0, self._min_agreement, "유효 데이터 없음")
        rate = compute_agreement_rate(
            valid[rule_col].tolist(), valid[expert_col].tolist()
        )
        return ValidationCheck(
            name="Rule-전문가 일치율",
            passed=rate >= self._min_agreement,
            value=rate,
            threshold=self._min_agreement,
            message=f"{len(valid)}건 비교",
        )

    def _check_kappa(self, df: pd.DataFrame, col_a: str, col_b: str) -> ValidationCheck:
        valid = df[df[col_a].isin(LABEL_VALUES) & df[col_b].isin(LABEL_VALUES)]
        if len(valid) == 0:
            return ValidationCheck("Cohen's Kappa", False, 0.0, self._min_kappa, "유효 데이터 없음")
        kappa = compute_cohens_kappa(valid[col_a].tolist(), valid[col_b].tolist())
        return ValidationCheck(
            name="Cohen's Kappa",
            passed=kappa >= self._min_kappa,
            value=kappa,
            threshold=self._min_kappa,
            message=f"{len(valid)}건 검토자 간 일치도",
        )
