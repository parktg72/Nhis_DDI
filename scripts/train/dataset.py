"""
ML 훈련 데이터셋 준비

ml_features_{partition}.parquet 로드 → train/val/test 분할 → 클래스 불균형 처리

클래스 구성:
  - 이진: is_high_risk (Red=1 vs 나머지=0)  ← 주 타겟
  - 다중: risk_level (Red/Yellow/Green/Normal)  ← 보조 타겟

클래스 불균형 처리 전략:
  - scale_pos_weight (XGBoost): negative / positive 비율
  - class_weight (LightGBM): balanced
  - SMOTE는 폐쇄망에서 imbalanced-learn 의존성 문제로 미사용
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 메타/레이블 컬럼 (피처에서 제외)
NON_FEATURE_COLS = {
    "patient_id", "window_start", "window_end",
    "risk_level", "risk_reasons", "yellow_subtype", "sex_type",
    "is_high_risk",        # 이진 레이블
    "risk_level_encoded",  # 다중 레이블
}

RISK_ORDER = {"Normal": 0, "Green": 1, "Yellow": 2, "Red": 3}


class TrainDataset:
    """
    ML 훈련용 데이터셋.

    Attributes
    ----------
    X_train, X_val, X_test : 피처 행렬 (numpy)
    y_train, y_val, y_test : 이진 레이블
    y_multi_train, ...     : 다중 레이블 (0~3)
    feature_names          : 피처 컬럼명 목록
    pos_weight             : XGBoost scale_pos_weight 값
    """

    def __init__(
        self,
        X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
        y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
        y_multi_train: np.ndarray, y_multi_val: np.ndarray, y_multi_test: np.ndarray,
        feature_names: list[str],
        meta_train: pd.DataFrame, meta_val: pd.DataFrame, meta_test: pd.DataFrame,
    ):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.y_multi_train = y_multi_train
        self.y_multi_val = y_multi_val
        self.y_multi_test = y_multi_test
        self.feature_names = feature_names
        self.meta_train = meta_train
        self.meta_val = meta_val
        self.meta_test = meta_test

    @property
    def pos_weight(self) -> float:
        """XGBoost scale_pos_weight = neg / pos."""
        n_pos = int(self.y_train.sum())
        n_neg = len(self.y_train) - n_pos
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_val(self) -> int:
        return len(self.X_val)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def class_distribution(self) -> dict[str, dict[str, int]]:
        return {
            "train": {"pos": int(self.y_train.sum()), "neg": int((self.y_train == 0).sum())},
            "val":   {"pos": int(self.y_val.sum()),   "neg": int((self.y_val == 0).sum())},
            "test":  {"pos": int(self.y_test.sum()),  "neg": int((self.y_test == 0).sum())},
        }

    def print_summary(self) -> None:
        dist = self.class_distribution()
        print(f"\n{'='*55}")
        print(f"[데이터셋 요약]")
        print(f"{'='*55}")
        print(f"  피처 수: {self.n_features}")
        print(f"  {'분할':8s} {'전체':>8s} {'Red(1)':>8s} {'Non-Red(0)':>10s} {'Red비율':>8s}")
        for split in ["train", "val", "test"]:
            pos = dist[split]["pos"]
            neg = dist[split]["neg"]
            total = pos + neg
            ratio = pos / total * 100 if total else 0
            print(f"  {split:8s} {total:8d} {pos:8d} {neg:10d} {ratio:7.1f}%")
        print(f"  pos_weight (XGBoost): {self.pos_weight:.1f}")
        print(f"{'='*55}")


def load_dataset(
    partition: str,
    feature_base: str | Path = "data/features",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> TrainDataset:
    """
    ml_features_{partition}.parquet → TrainDataset 생성.

    Parameters
    ----------
    val_ratio   : 검증셋 비율
    test_ratio  : 테스트셋 비율
    random_state : 재현성 시드
    """
    path = Path(feature_base) / f"ml_features_{partition}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"ML 피처 파일 없음: {path}\n"
            f"먼저 scripts/features/feature_engineer.py 실행 필요"
        )

    df = pd.read_parquet(path)
    logger.info("ML 피처 로드: %d명 × %d컬럼", len(df), len(df.columns))

    return _split_dataset(df, val_ratio, test_ratio, random_state)


def load_dataset_from_df(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> TrainDataset:
    """DataFrame 직접 입력 (테스트/편의용)."""
    return _split_dataset(df, val_ratio, test_ratio, random_state)


def _split_dataset(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> TrainDataset:
    import pandas as pd
    rng = np.random.default_rng(random_state)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    df = df.copy()
    if "is_high_risk" not in df.columns:
        if "risk_level" in df.columns:
            df["is_high_risk"] = (df["risk_level"] == "Red").astype(int)
        else:
            raise ValueError("'is_high_risk' 또는 'risk_level' 컬럼 필요")
    if "risk_level" in df.columns:
        df["risk_level_encoded"] = df["risk_level"].map(RISK_ORDER).fillna(0).astype(int)
    else:
        df["risk_level_encoded"] = df["is_high_risk"] * 3

    y = df["is_high_risk"].values
    classes = np.unique(y)

    if len(classes) < 2:
        logger.warning("단일 클래스 — 층화 불가, 랜덤 분할")
        idx = rng.permutation(len(df))
        df = df.iloc[idx].reset_index(drop=True)
        n = len(df)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        train_df = df.iloc[:n - n_val - n_test]
        val_df = df.iloc[n - n_val - n_test:n - n_test]
        test_df = df.iloc[n - n_test:]
    else:
        split_parts = {"train": [], "val": [], "test": []}
        for cls in classes:
            cls_df = df[y == cls].copy()
            cls_idx = rng.permutation(len(cls_df))
            cls_df = cls_df.iloc[cls_idx].reset_index(drop=True)
            n_c = len(cls_df)
            n_c_test = max(1, int(n_c * test_ratio))
            n_c_val = max(1, int(n_c * val_ratio))
            n_c_train = n_c - n_c_val - n_c_test
            if n_c_train < 1:
                # Too few samples — put all in train, skip val/test for this class
                logger.warning("클래스 %s 샘플 부족 (%d개) — train만 배분", cls, n_c)
                split_parts["train"].append(cls_df)
                continue
            split_parts["train"].append(cls_df.iloc[:n_c_train])
            split_parts["val"].append(cls_df.iloc[n_c_train:n_c_train + n_c_val])
            split_parts["test"].append(cls_df.iloc[n_c_train + n_c_val:])

        train_df = pd.concat(split_parts["train"]).sample(frac=1, random_state=int(random_state)).reset_index(drop=True)
        val_df = pd.concat(split_parts["val"]).sample(frac=1, random_state=int(random_state)).reset_index(drop=True) if split_parts["val"] else pd.DataFrame(columns=df.columns)
        test_df = pd.concat(split_parts["test"]).sample(frac=1, random_state=int(random_state)).reset_index(drop=True) if split_parts["test"] else pd.DataFrame(columns=df.columns)

    def _arrays(sub):
        if len(sub) == 0:
            return np.empty((0, len(feature_cols))), np.empty(0, int), np.empty(0, int), pd.DataFrame()
        X = sub[feature_cols].astype(float).values
        y_bin = sub["is_high_risk"].values.astype(int)
        y_multi = sub["risk_level_encoded"].values.astype(int)
        meta_cols = [c for c in ["patient_id", "risk_level", "yellow_subtype", "window_start", "window_end"] if c in sub.columns]
        meta = sub[meta_cols].reset_index(drop=True)
        return X, y_bin, y_multi, meta

    X_tr, y_tr, ym_tr, m_tr = _arrays(train_df)
    X_va, y_va, ym_va, m_va = _arrays(val_df)
    X_te, y_te, ym_te, m_te = _arrays(test_df)

    ds = TrainDataset(X_train=X_tr, X_val=X_va, X_test=X_te,
                      y_train=y_tr, y_val=y_va, y_test=y_te,
                      y_multi_train=ym_tr, y_multi_val=ym_va, y_multi_test=ym_te,
                      feature_names=feature_cols,
                      meta_train=m_tr, meta_val=m_va, meta_test=m_te)
    ds.print_summary()
    return ds
