"""
임상 전문가 검토용 샘플 추출기

PROJECT_PLAN 4.4 레이블링 전략 3차:
  임상 전문가 패널이 검토할 골든 데이터셋 샘플을 계층화 추출.
  목표: 5,650건

계층화 기준:
  - 위험도 (risk_level)    : Red/Yellow/Green/Normal
  - 연령대 (age_group)     : <65, 65-74, 75+
  - DDI 심각도 (ddi_tier)  : contraindicated / major / moderate_or_below
  - ADR 발생 여부 (has_adr): True / False

위험도별 목표 비율 (소수 등급 과표집):
  Red    : 30% (1,695건) — 임상적으로 가장 중요
  Yellow : 30% (1,695건)
  Green  : 25% (1,413건)
  Normal : 15%  (847건)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 골든 데이터셋 목표 규모
GOLDEN_TARGET = 5_650

# 위험도별 목표 비율
RISK_LEVEL_RATIOS: Dict[str, float] = {
    "Red":    0.30,
    "Yellow": 0.30,
    "Green":  0.25,
    "Normal": 0.15,
}

# 검토용 출력 컬럼 (PII 제외)
REVIEW_COLUMNS = [
    "patient_id", "risk_level", "rule_level", "adr_label", "confidence",
    "drug_count", "ddi_contraindicated", "ddi_major", "triple_whammy",
    "age_group", "sex", "cyp_risk_score",
    "top_risk_reasons",    # 전문가 검토 편의를 위한 요약
    "expert_label",        # 전문가 기입란 (초기 빈 값)
    "expert_comment",
    "review_status",       # pending | reviewed | disputed
]


@dataclass
class SampleSummary:
    """추출 결과 요약."""
    total: int
    by_risk_level: Dict[str, int]
    by_age_group: Dict[str, int]
    by_adr_label: Dict[str, int]
    coverage_rate: float           # 원본 대비 샘플 비율
    stratification_ok: bool        # 목표 비율 충족 여부

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "by_risk_level": self.by_risk_level,
            "by_age_group": self.by_age_group,
            "by_adr_label": self.by_adr_label,
            "coverage_rate": round(self.coverage_rate, 4),
            "stratification_ok": self.stratification_ok,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 계층화 샘플러
# ─────────────────────────────────────────────────────────────────────────────

class StratifiedSampler:
    """위험도 기반 계층화 샘플 추출기.

    Usage:
        sampler = StratifiedSampler(n_total=5650, seed=42)
        sample_df = sampler.extract(features_df)
        sampler.export_for_review(sample_df, "data/labeling/golden_review.csv")
    """

    def __init__(
        self,
        n_total: int = GOLDEN_TARGET,
        ratios: Dict[str, float] = None,
        seed: int = 42,
    ):
        self._n_total = n_total
        self._ratios = ratios or RISK_LEVEL_RATIOS
        self._seed = seed

        # 비율 합이 1.0인지 검증
        total_ratio = sum(self._ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"위험도 비율 합이 1.0이 아님: {total_ratio:.4f}")

    def extract(
        self,
        df: pd.DataFrame,
        risk_col: str = "risk_level",
        adr_col: Optional[str] = "adr_label",
    ) -> pd.DataFrame:
        """계층화 샘플 추출.

        각 위험도 계층 내에서 연령대/ADR 발생 여부를 고려한 균형 샘플링.
        원본 데이터가 목표 크기보다 작으면 전체 반환.
        """
        if risk_col not in df.columns:
            raise ValueError(f"'{risk_col}' 컬럼 없음")

        rng = np.random.default_rng(self._seed)
        sampled_parts: list[pd.DataFrame] = []

        for level, ratio in self._ratios.items():
            target_n = int(self._n_total * ratio)
            pool = df[df[risk_col] == level]

            if len(pool) == 0:
                logger.warning("위험도 '%s' 데이터 없음 — 해당 계층 건너뜀", level)
                continue

            if len(pool) <= target_n:
                # 데이터가 목표보다 적으면 전체 사용
                sampled_parts.append(pool.copy())
                logger.warning(
                    "'%s' 계층 데이터(%d건)가 목표(%d건)보다 적어 전체 사용",
                    level, len(pool), target_n,
                )
                continue

            # ADR 발생 여부로 내부 계층화 (가능한 경우)
            if adr_col and adr_col in df.columns:
                part = self._stratified_by_adr(pool, target_n, adr_col, rng)
            else:
                idx = rng.choice(len(pool), size=target_n, replace=False)
                part = pool.iloc[idx]

            sampled_parts.append(part)

        result = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
        result = result.sample(frac=1, random_state=self._seed).reset_index(drop=True)

        logger.info("계층화 샘플 추출 완료: %d건 (목표: %d건)", len(result), self._n_total)
        return result

    def _stratified_by_adr(
        self,
        pool: pd.DataFrame,
        target_n: int,
        adr_col: str,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """ADR 발생 여부로 내부 계층화 (ADR 발생 케이스 최소 20% 보장)."""
        adr_pos = pool[pool[adr_col] == 1]
        adr_neg = pool[pool[adr_col] == 0]

        n_pos = min(len(adr_pos), max(int(target_n * 0.20), len(adr_pos)))
        n_neg = target_n - n_pos

        parts = []
        if len(adr_pos) > 0:
            n_pos = min(n_pos, len(adr_pos))
            idx = rng.choice(len(adr_pos), size=n_pos, replace=False)
            parts.append(adr_pos.iloc[idx])

        if len(adr_neg) > 0:
            n_neg = min(n_neg, len(adr_neg))
            idx = rng.choice(len(adr_neg), size=n_neg, replace=False)
            parts.append(adr_neg.iloc[idx])

        return pd.concat(parts, ignore_index=True)

    def summarize(self, sample_df: pd.DataFrame, original_df: pd.DataFrame) -> SampleSummary:
        """추출 결과 요약 생성."""
        by_risk = sample_df["risk_level"].value_counts().to_dict() \
            if "risk_level" in sample_df.columns else {}
        by_age  = sample_df["age_group"].value_counts().to_dict() \
            if "age_group" in sample_df.columns else {}
        by_adr  = sample_df["adr_label"].value_counts().to_dict() \
            if "adr_label" in sample_df.columns else {}

        coverage = len(sample_df) / max(len(original_df), 1)

        # 목표 비율 대비 실제 비율 허용 오차 (±5%)
        strat_ok = True
        total = max(len(sample_df), 1)
        for level, target_ratio in self._ratios.items():
            actual = by_risk.get(level, 0) / total
            if abs(actual - target_ratio) > 0.08:
                strat_ok = False
                break

        return SampleSummary(
            total=len(sample_df),
            by_risk_level=by_risk,
            by_age_group=by_age,
            by_adr_label={str(k): v for k, v in by_adr.items()},
            coverage_rate=coverage,
            stratification_ok=strat_ok,
        )

    def export_for_review(
        self,
        sample_df: pd.DataFrame,
        out_path: str,
        fmt: str = "csv",
    ) -> str:
        """전문가 검토용 파일 내보내기 (CSV 또는 Excel).

        - 전문가 기입란(expert_label, expert_comment) 컬럼 추가
        - PII 제거 확인 (patient_id는 가명처리된 해시값)
        """
        export_df = sample_df.copy()

        # 전문가 기입 컬럼 초기화
        if "expert_label" not in export_df.columns:
            export_df["expert_label"] = ""
        if "expert_comment" not in export_df.columns:
            export_df["expert_comment"] = ""
        if "review_status" not in export_df.columns:
            export_df["review_status"] = "pending"

        # 존재하는 컬럼만 선택
        cols = [c for c in REVIEW_COLUMNS if c in export_df.columns]
        export_df = export_df[cols]

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        if fmt == "excel":
            export_df.to_excel(out_path, index=False, engine="openpyxl")
        else:
            export_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        logger.info("전문가 검토 파일 생성: %s (%d건)", out_path, len(export_df))
        return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 연령대 분류 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def assign_age_group(age: Optional[int]) -> str:
    """나이 → 연령대 문자열."""
    if age is None or (isinstance(age, float) and age != age):  # None or NaN
        return "unknown"
    if age < 65:
        return "lt65"
    if age < 75:
        return "65-74"
    return "75plus"


def add_age_group(df: pd.DataFrame, age_col: str = "patient_age") -> pd.DataFrame:
    """DataFrame에 age_group 컬럼 추가."""
    df = df.copy()
    if age_col in df.columns:
        df["age_group"] = df[age_col].apply(assign_age_group)
    else:
        df["age_group"] = "unknown"
    return df
