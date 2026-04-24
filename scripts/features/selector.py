"""
피처 선택 (Feature Selection)

단계:
  1. 분산 필터링: 분산 = 0인 피처 제거 (모든 환자가 동일한 값)
  2. 상관관계 필터링: Pearson 상관 > threshold인 쌍 중 하나 제거
  3. 중요도 기반 선택: 선택적 (XGBoost importance 기반, Phase 2에서 활용)

설계 원칙:
  - 필수 피처 (ddi_contraindicated, ddi_major, triple_whammy 등)는 보호
  - 선택 결과는 직렬화 가능
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 반드시 유지해야 하는 핵심 피처 (제거 불가)
PROTECTED_FEATURES = {
    "ddi_contraindicated",
    "ddi_major",
    "ddi_moderate",
    "ddi_minor",
    "triple_whammy",
    "qt_risk_count",
    "dup_same_ingredient",
    "drug_count",
    "institution_count",
    "cyp_risk_score",
    "cyp_high_risk_pairs",
}

# 피처 선택에서 제외할 메타 컬럼
META_COLS = {
    "patient_id", "window_start", "window_end",
    "risk_level", "risk_reasons", "age", "sex",
    "yellow_subtype",
}


class FeatureSelector:
    """
    분산·상관관계 기반 피처 선택기.

    Parameters
    ----------
    variance_threshold : 분산 이하 피처 제거 (기본 0: 상수 컬럼만 제거)
    correlation_threshold : Pearson 상관 절댓값 상한 (기본 0.95)
    """

    def __init__(
        self,
        variance_threshold: float = 0.0,
        correlation_threshold: float = 0.95,
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self._selected: list[str] = []
        self._removed_variance: list[str] = []
        self._removed_corr: list[str] = []
        self._fitted = False

    # ──────────────────────────────────────────────────────────────────────────
    # 학습 / 변환
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "FeatureSelector":
        """훈련 데이터로 선택 규칙 학습."""
        candidates = self._get_candidate_cols(df)

        # Step 1: 분산 필터링
        keep = []
        for col in candidates:
            var = df[col].astype(float).var()
            if var <= self.variance_threshold and col not in PROTECTED_FEATURES:
                self._removed_variance.append(col)
                logger.debug("분산 제거: %s (var=%.6f)", col, var)
            else:
                keep.append(col)

        # Step 2: 상관관계 필터링
        if len(keep) > 1:
            corr_matrix = df[keep].astype(float).corr().abs()
            to_remove: set[str] = set()
            for i in range(len(keep)):
                if keep[i] in to_remove:
                    continue
                for j in range(i + 1, len(keep)):
                    if keep[j] in to_remove:
                        continue
                    if corr_matrix.iloc[i, j] > self.correlation_threshold:
                        # 보호 피처가 아닌 쪽 제거
                        victim = keep[j] if keep[j] not in PROTECTED_FEATURES else keep[i]
                        if victim not in PROTECTED_FEATURES:
                            to_remove.add(victim)
                            logger.debug(
                                "상관 제거: %s (corr(%s, %s)=%.3f)",
                                victim, keep[i], keep[j], corr_matrix.iloc[i, j],
                            )
            self._removed_corr = list(to_remove)
            keep = [c for c in keep if c not in to_remove]

        self._selected = keep
        self._fitted = True
        logger.info(
            "FeatureSelector fit: %d → %d 피처 (분산제거 %d, 상관제거 %d)",
            len(candidates), len(self._selected),
            len(self._removed_variance), len(self._removed_corr),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """선택된 피처 컬럼만 반환 (메타 컬럼 포함)."""
        if not self._fitted:
            raise RuntimeError("fit() 먼저 호출하세요.")

        meta = [c for c in META_COLS if c in df.columns]
        selected_in_df = [c for c in self._selected if c in df.columns]
        return df[meta + selected_in_df]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ──────────────────────────────────────────────────────────────────────────
    # 직렬화
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path = "data/features/selector.pkl") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "selected": self._selected,
                "removed_variance": self._removed_variance,
                "removed_corr": self._removed_corr,
                "variance_threshold": self.variance_threshold,
                "correlation_threshold": self.correlation_threshold,
                "fitted": self._fitted,
            }, f)
        logger.info("피처 선택기 저장: %s (%d 피처)", path, len(self._selected))
        return path

    @classmethod
    def load(cls, path: str | Path = "data/features/selector.pkl") -> "FeatureSelector":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            variance_threshold=state["variance_threshold"],
            correlation_threshold=state["correlation_threshold"],
        )
        obj._selected = state["selected"]
        obj._removed_variance = state["removed_variance"]
        obj._removed_corr = state["removed_corr"]
        obj._fitted = state["fitted"]
        logger.info("피처 선택기 로드: %s (%d 피처)", path, len(obj._selected))
        return obj

    # ──────────────────────────────────────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────────────────────────────────────

    def _get_candidate_cols(self, df: pd.DataFrame) -> list[str]:
        """선택 대상 후보 컬럼 (메타 제외, 수치형만)."""
        result = []
        for col in df.columns:
            if col in META_COLS:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                result.append(col)
        return result

    @property
    def selected_features(self) -> list[str]:
        return list(self._selected)

    @property
    def n_features(self) -> int:
        return len(self._selected)

    def report(self) -> None:
        print(f"\n[피처 선택 결과]")
        print(f"  선택된 피처: {len(self._selected)}개")
        print(f"  분산 제거:   {len(self._removed_variance)}개 → {self._removed_variance}")
        print(f"  상관 제거:   {len(self._removed_corr)}개 → {self._removed_corr}")
