"""
하이퍼파라미터 정의 및 Optuna 탐색 공간

기본값은 의료 DDI 탐지 문제에 최적화:
  - Recall 우선 (Red 환자 미탐지 = 의료 위해)
  - 클래스 불균형 처리 (pos_weight, is_unbalance)
  - 과적합 방지 (reg_alpha, reg_lambda, min_child_weight)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# XGBoost 기본 하이퍼파라미터
# ─────────────────────────────────────────────────────────────────────────────

XGB_DEFAULT: dict[str, Any] = {
    "objective":        "binary:logistic",
    "eval_metric":      ["auc", "aucpr", "logloss"],
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 3,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,   # L1
    "reg_lambda":       1.0,   # L2
    "gamma":            0.1,   # 최소 분할 이득
    "tree_method":      "hist",
    "random_state":     42,
    "n_jobs":           -1,
    # scale_pos_weight: dataset.pos_weight 에서 동적으로 설정
}

# ─────────────────────────────────────────────────────────────────────────────
# LightGBM 기본 하이퍼파라미터
# ─────────────────────────────────────────────────────────────────────────────

LGB_DEFAULT: dict[str, Any] = {
    "objective":        "binary",
    "metric":           ["auc", "average_precision", "binary_logloss"],
    "n_estimators":     500,
    "learning_rate":    0.05,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "is_unbalance":     True,   # 클래스 불균형 자동 처리
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        -1,
}

# ─────────────────────────────────────────────────────────────────────────────
# Optuna 탐색 공간
# ─────────────────────────────────────────────────────────────────────────────

def xgb_search_space(trial: Any) -> dict[str, Any]:
    """XGBoost Optuna 탐색 공간."""
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
    }


def lgb_search_space(trial: Any) -> dict[str, Any]:
    """LightGBM Optuna 탐색 공간."""
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 훈련 설정
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """전체 훈련 설정."""
    model_type: str = "xgboost"          # "xgboost" | "lightgbm" | "ensemble"
    partition: str = ""                   # 데이터 파티션 (YYYYMM)
    feature_base: str = "data/features"
    model_dir: str = "models"
    experiment_name: str = "ddi_risk_model"

    # 훈련 제어
    early_stopping_rounds: int = 50
    recall_threshold: float = 0.90       # 최소 Recall 요건 (Red 환자)
    auc_threshold: float = 0.85          # 최소 AUC 요건
    probability_threshold: float = 0.5   # 분류 임계값 (최적화됨)

    # Optuna 하이퍼파라미터 탐색
    use_optuna: bool = False
    optuna_trials: int = 50
    optuna_timeout: int = 3600           # 초

    # 재현성
    random_state: int = 42

    # 하이퍼파라미터 (모델 타입에 따라 자동 선택)
    xgb_params: dict = field(default_factory=lambda: dict(XGB_DEFAULT))
    lgb_params: dict = field(default_factory=lambda: dict(LGB_DEFAULT))

    def get_model_params(self) -> dict:
        if self.model_type == "lightgbm":
            return self.lgb_params
        return self.xgb_params  # 기본값: xgboost
