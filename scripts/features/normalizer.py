"""
피처 정규화 / 결측치 처리

전략:
  - 수치 피처: RobustScaler (이상값에 강건, 중앙값/IQR 기반)
  - 이진 피처 (0/1): 스케일링 불필요, 그대로 유지
  - 결측치: 중앙값(수치) / 최빈값(범주형)으로 대체
  - 스케일러 직렬화: joblib → data/features/scaler.joblib

이진 피처 목록 (스케일링 제외):
  triple_whammy, multi_institution_flag
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 이진 피처 (스케일링 제외)
BINARY_COLS = {"triple_whammy", "multi_institution_flag"}

# 범주형 피처 (인코딩 제외, 모델에 직접 사용 안함)
CATEGORICAL_COLS = {"patient_id", "risk_level", "sex", "risk_reasons",
                    "window_start", "window_end"}


class FeatureNormalizer:
    """
    피처 정규화기 (RobustScaler 기반).

    fit() → transform() 패턴 또는 fit_transform() 편의 메서드 사용.
    직렬화: save() / load() 로 재사용.
    """

    def __init__(self):
        self._medians: dict[str, float] = {}
        self._iqr: dict[str, float] = {}
        self._numeric_cols: list[str] = []
        self._fitted = False

    # ──────────────────────────────────────────────────────────────────────────
    # 학습 / 변환
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "FeatureNormalizer":
        """훈련 데이터로 스케일러 학습."""
        num_cols = self._get_numeric_cols(df)
        self._numeric_cols = num_cols

        for col in num_cols:
            series = df[col].dropna().astype(float)
            if len(series) == 0:
                self._medians[col] = 0.0
                self._iqr[col] = 1.0
                continue
            q75, q25 = np.percentile(series, [75, 25])
            iqr = q75 - q25
            self._medians[col] = float(np.median(series))
            self._iqr[col] = float(iqr) if iqr > 0 else 1.0

        self._fitted = True
        logger.info("FeatureNormalizer fit: %d 수치 피처", len(num_cols))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """피처 정규화 적용. fit() 먼저 호출 필요."""
        if not self._fitted:
            raise RuntimeError("fit() 먼저 호출하세요.")

        out = df.copy()

        # 결측치 중앙값 대체
        for col in self._numeric_cols:
            if col in out.columns:
                out[col] = out[col].fillna(self._medians.get(col, 0.0))

        # RobustScaler 적용 (이진 피처 제외)
        for col in self._numeric_cols:
            if col in out.columns and col not in BINARY_COLS:
                median = self._medians.get(col, 0.0)
                iqr = self._iqr.get(col, 1.0)
                out[col] = (out[col].astype(float) - median) / iqr

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """역변환 (스케일 복원)."""
        out = df.copy()
        for col in self._numeric_cols:
            if col in out.columns and col not in BINARY_COLS:
                median = self._medians.get(col, 0.0)
                iqr = self._iqr.get(col, 1.0)
                out[col] = out[col].astype(float) * iqr + median
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # 직렬화
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path = "data/features/scaler.pkl") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "medians": self._medians,
                "iqr": self._iqr,
                "numeric_cols": self._numeric_cols,
                "fitted": self._fitted,
            }, f)
        logger.info("스케일러 저장: %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path = "data/features/scaler.pkl") -> "FeatureNormalizer":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj._medians = state["medians"]
        obj._iqr = state["iqr"]
        obj._numeric_cols = state["numeric_cols"]
        obj._fitted = state["fitted"]
        logger.info("스케일러 로드: %s (%d 피처)", path, len(obj._numeric_cols))
        return obj

    # ──────────────────────────────────────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_numeric_cols(df: pd.DataFrame) -> list[str]:
        """스케일링 대상 수치 컬럼 추출."""
        exclude = CATEGORICAL_COLS
        num_cols = []
        for col in df.columns:
            if col in exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                num_cols.append(col)
        return num_cols

    @property
    def feature_names(self) -> list[str]:
        return list(self._numeric_cols)
