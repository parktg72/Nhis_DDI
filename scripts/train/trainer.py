"""
XGBoost / LightGBM 모델 훈련

설계:
  - BaseTrainer 추상 클래스 → XGBoostTrainer / LGBMTrainer 구현
  - lazy import: xgboost/lightgbm 설치 여부에 따라 동적 로드
  - Early stopping: val AUC 기준
  - 모델 직렬화: joblib (pkl보다 NumPy 배열에 효율적)
  - 피처 중요도: gain 기반
"""
from __future__ import annotations

import logging
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .dataset import TrainDataset
from .evaluator import EvalResult, evaluate_all_splits

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 추상 기반 클래스
# ─────────────────────────────────────────────────────────────────────────────

class BaseTrainer(ABC):
    """모델 훈련 공통 인터페이스."""

    def __init__(self, params: dict[str, Any], config: Any):
        self.params = dict(params)
        self.config = config
        self.model = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.best_threshold_: float = 0.5
        self._trained = False

    @abstractmethod
    def fit(self, dataset: TrainDataset) -> "BaseTrainer":
        """모델 학습."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """양성 클래스 확률 반환."""
        ...

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """이진 예측. threshold=None이면 best_threshold_ 사용."""
        t = threshold if threshold is not None else self.best_threshold_
        return (self.predict_proba(X) >= t).astype(int)

    def evaluate(
        self,
        dataset: TrainDataset,
        min_recall: float = 0.90,
        min_auc: float = 0.85,
    ) -> dict[str, EvalResult]:
        if not self._trained:
            raise RuntimeError("fit() 먼저 호출하세요.")
        results = evaluate_all_splits(
            y_true_tr=dataset.y_train,
            y_prob_tr=self.predict_proba(dataset.X_train),
            y_true_va=dataset.y_val,
            y_prob_va=self.predict_proba(dataset.X_val),
            y_true_te=dataset.y_test,
            y_prob_te=self.predict_proba(dataset.X_test),
            min_recall=min_recall,
            min_auc=min_auc,
        )
        self.best_threshold_ = results["val"].threshold
        return results

    def save(self, path: str | Path) -> Path:
        import hashlib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "params": self.params,
            "feature_importances": self.feature_importances_,
            "best_threshold": self.best_threshold_,
            "trainer_class": self.__class__.__name__,
            **getattr(self, "_extra_meta", {}),
        }
        content = pickle.dumps(payload)
        path.write_bytes(content)
        sha256 = hashlib.sha256(content).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256}  {path.name}\n")
        logger.info("모델 저장: %s (sha256=%s…)", path, sha256[:16])
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseTrainer":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.model = state["model"]
        obj.params = state["params"]
        obj.feature_importances_ = state.get("feature_importances")
        obj.best_threshold_ = state.get("best_threshold", 0.5)
        obj._trained = True
        obj.config = None
        logger.info("모델 로드: %s", path)
        return obj

    def feature_importance_df(self, feature_names: list[str]) -> Any:
        """피처 중요도 DataFrame 반환 (gain 기준 정렬)."""
        try:
            import pandas as pd
        except ImportError:
            return None
        if self.feature_importances_ is None:
            return None
        import pandas as pd
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": self.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostTrainer(BaseTrainer):
    """XGBoost 이진 분류 훈련기."""

    def fit(self, dataset: TrainDataset) -> "XGBoostTrainer":
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "xgboost 미설치. packages_linux/py3XX/ 에서 설치 필요.\n"
                f"  pip install xgboost\n원래 오류: {e}"
            ) from e

        params = dict(self.params)
        params["scale_pos_weight"] = dataset.pos_weight
        # eval_metric 리스트 분리 (xgb.train 전달용)
        eval_metric = params.pop("eval_metric", ["auc"])

        t0 = time.perf_counter()
        logger.info(
            "XGBoost 훈련 시작: n_estimators=%d, lr=%.4f, pos_weight=%.1f",
            params.get("n_estimators", 500),
            params.get("learning_rate", 0.05),
            dataset.pos_weight,
        )

        model = xgb.XGBClassifier(
            **params,
            eval_metric=eval_metric,
            early_stopping_rounds=getattr(self.config, "early_stopping_rounds", 50),
            enable_categorical=False,
        )
        model.fit(
            dataset.X_train, dataset.y_train,
            eval_set=[(dataset.X_val, dataset.y_val)],
            verbose=50,
        )

        self.model = model
        self.feature_importances_ = model.feature_importances_
        self._trained = True
        logger.info("XGBoost 훈련 완료 (%.1fs)", time.perf_counter() - t0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM
# ─────────────────────────────────────────────────────────────────────────────

class LGBMTrainer(BaseTrainer):
    """LightGBM 이진 분류 훈련기."""

    def fit(self, dataset: TrainDataset) -> "LGBMTrainer":
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "lightgbm 미설치. packages_linux/py3XX/ 에서 설치 필요.\n"
                f"  pip install lightgbm\n원래 오류: {e}"
            ) from e

        params = dict(self.params)
        # LightGBM metric 리스트 분리
        metric = params.pop("metric", ["auc"])

        t0 = time.perf_counter()
        logger.info(
            "LightGBM 훈련 시작: n_estimators=%d, lr=%.4f",
            params.get("n_estimators", 500),
            params.get("learning_rate", 0.05),
        )

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=getattr(self.config, "early_stopping_rounds", 50),
                verbose=False,
            ),
            lgb.log_evaluation(period=50),
        ]

        model = lgb.LGBMClassifier(**params, metric=metric)
        model.fit(
            dataset.X_train, dataset.y_train,
            eval_set=[(dataset.X_val, dataset.y_val)],
            callbacks=callbacks,
            feature_name=dataset.feature_names,
        )

        self.model = model
        self.feature_importances_ = model.feature_importances_
        self._trained = True
        logger.info("LightGBM 훈련 완료 (%.1fs)", time.perf_counter() - t0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# 앙상블 (XGBoost + LightGBM 평균)
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleTrainer(BaseTrainer):
    """XGBoost + LightGBM 소프트 보팅 앙상블."""

    def __init__(
        self,
        xgb_params: dict,
        lgb_params: dict,
        config: Any,
        weights: tuple[float, float] = (0.5, 0.5),
    ):
        super().__init__(xgb_params, config)
        self.weights = weights
        self._xgb = XGBoostTrainer(xgb_params, config)
        self._lgb = LGBMTrainer(lgb_params, config)

    def fit(self, dataset: TrainDataset) -> "EnsembleTrainer":
        logger.info("앙상블 훈련: XGBoost + LightGBM")
        self._xgb.fit(dataset)
        self._lgb.fit(dataset)
        # 피처 중요도: 가중 평균
        if (self._xgb.feature_importances_ is not None
                and self._lgb.feature_importances_ is not None):
            w1, w2 = self.weights
            self.feature_importances_ = (
                w1 * self._xgb.feature_importances_
                + w2 * self._lgb.feature_importances_
            )
        self._trained = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        w1, w2 = self.weights
        p_xgb = self._xgb.predict_proba(X)
        p_lgb = self._lgb.predict_proba(X)
        return w1 * p_xgb + w2 * p_lgb

    def save(self, path: str | Path) -> Path:
        import hashlib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._xgb.save(path.with_suffix(".xgb.pkl"))
        self._lgb.save(path.with_suffix(".lgb.pkl"))
        payload = {
            "trainer_class": self.__class__.__name__,
            "weights": self.weights,
            "best_threshold": self.best_threshold_,
            "feature_importances": self.feature_importances_,
            **getattr(self, "_extra_meta", {}),
        }
        content = pickle.dumps(payload)
        path.write_bytes(content)
        sha256 = hashlib.sha256(content).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256}  {path.name}\n")
        logger.info("앙상블 모델 저장: %s (sha256=%s…)", path, sha256[:16])
        return path


# ─────────────────────────────────────────────────────────────────────────────
# 앙상블 3-way (XGBoost + LightGBM + GAT)
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleTrainer3Way(BaseTrainer):
    """XGBoost + LightGBM + GAT 소프트 보팅 앙상블 (3-way).

    가중치 최적화: calibration 스플릿에서 Recall >= 0.90 제약 하 AUC 최대화 (SLSQP).
    미지 약물 포함 요청: w_gat=0, 나머지 가중치 정규화.
    """

    def __init__(
        self,
        xgb_params: dict,
        lgb_params: dict,
        gat_params: dict,
        config: Any,
        weights: tuple = (1/3, 1/3, 1/3),
    ):
        super().__init__(xgb_params, config)
        self.weights = weights
        self._xgb = XGBoostTrainer(xgb_params, config)
        self._lgb = LGBMTrainer(lgb_params, config)
        from .gat_trainer import GATTrainer
        model_dir = getattr(config, "model_dir", "models")
        self._gat = GATTrainer(gat_params, config, model_dir=model_dir)

    def fit(self, dataset) -> "EnsembleTrainer3Way":
        """tabular 데이터용 fit — GAT는 별도 fit_gat() 호출 필요."""
        if not isinstance(dataset, TrainDataset):
            raise TypeError("EnsembleTrainer3Way.fit()은 TrainDataset 필요")
        logger.info("앙상블 3-way 훈련: XGBoost + LightGBM")
        self._xgb.fit(dataset)
        self._lgb.fit(dataset)
        if self._xgb.feature_importances_ is not None:
            w1, w2, _ = self.weights
            norm = w1 + w2 or 1.0
            self.feature_importances_ = (
                (w1 / norm) * self._xgb.feature_importances_
                + (w2 / norm) * self._lgb.feature_importances_
            )
        self._trained = True
        return self

    def fit_gat(self, gat_dataset) -> "EnsembleTrainer3Way":
        """GAT 서브모델 학습 (별도 호출)."""
        self._gat.fit_graph(gat_dataset)
        return self

    def optimize_weights(
        self,
        X_calib,
        y_calib,
        drug_pairs_calib,
        recall_threshold: float = 0.90,
    ) -> tuple:
        """Recall >= recall_threshold 제약 하에서 AUC 최대화 (SLSQP)."""
        from scipy.optimize import minimize
        from sklearn.metrics import roc_auc_score, recall_score

        p_xgb = self._xgb.predict_proba(X_calib)
        p_lgb = self._lgb.predict_proba(X_calib)
        p_gat_list = []
        for drug_a, drug_b in drug_pairs_calib:
            val = self._gat.predict_pair_proba(drug_a, drug_b)
            p_gat_list.append(val if val is not None else 0.5)
        p_gat = np.array(p_gat_list)

        def neg_auc(w):
            p = w[0] * p_xgb + w[1] * p_lgb + w[2] * p_gat
            try:
                return -roc_auc_score(y_calib, p)
            except Exception:
                return 0.0

        def recall_constraint(w):
            p = w[0] * p_xgb + w[1] * p_lgb + w[2] * p_gat
            pred = (p >= 0.5).astype(int)
            try:
                return recall_score(y_calib, pred, zero_division=0) - recall_threshold
            except Exception:
                return -1.0

        constraints = [
            {"type": "eq",   "fun": lambda w: w.sum() - 1.0},
            {"type": "ineq", "fun": recall_constraint},
        ]
        bounds = [(0.0, 1.0)] * 3
        x0 = np.array([1/3, 1/3, 1/3])

        result = minimize(neg_auc, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            self.weights = tuple(float(v) for v in result.x)
        else:
            logger.warning("가중치 최적화 실패 — 균등 가중치 유지: %s", result.message)
        logger.info("앙상블 최적 가중치: xgb=%.3f lgb=%.3f gat=%.3f", *self.weights)
        return self.weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """tabular-only 예측 (GAT 제외). serving에서는 predict_proba_with_gat 사용."""
        w1, w2, _ = self.weights
        norm = w1 + w2 or 1.0
        return (w1 / norm) * self._xgb.predict_proba(X) + (w2 / norm) * self._lgb.predict_proba(X)

    def predict_proba_with_gat(self, X: np.ndarray, drug_pairs) -> np.ndarray:
        """GAT 포함 예측."""
        w1, w2, w3 = self.weights
        p_xgb = self._xgb.predict_proba(X)
        p_lgb = self._lgb.predict_proba(X)

        results = np.zeros(len(X))
        for i, (drug_a, drug_b) in enumerate(drug_pairs):
            p_gat = self._gat.predict_pair_proba(drug_a, drug_b)
            if p_gat is None:
                norm = w1 + w2 or 1.0
                results[i] = (w1 / norm) * p_xgb[i] + (w2 / norm) * p_lgb[i]
            else:
                results[i] = w1 * p_xgb[i] + w2 * p_lgb[i] + w3 * p_gat
        return results

    def save(self, path) -> Path:
        import hashlib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._xgb.save(path.with_suffix(".xgb.pkl"))
        self._lgb.save(path.with_suffix(".lgb.pkl"))
        gat_path = path.parent / "gat_model.pt"
        if self._gat._trained:
            self._gat.save(gat_path)
        payload = {
            "trainer_class": self.__class__.__name__,
            "weights": self.weights,
            "best_threshold": getattr(self, "best_threshold_", 0.5),
            "feature_importances": getattr(self, "feature_importances_", None),
        }
        content = pickle.dumps(payload)
        path.write_bytes(content)
        sha256 = hashlib.sha256(content).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256}  {path.name}\n")
        logger.info("EnsembleTrainer3Way 저장: %s", path)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# 팩토리
# ─────────────────────────────────────────────────────────────────────────────

def build_trainer(config: Any) -> BaseTrainer:
    """TrainConfig에 따라 적절한 Trainer 반환."""
    from .hyperparams import TrainConfig
    if not isinstance(config, TrainConfig):
        raise TypeError(f"TrainConfig 필요, 받은 타입: {type(config)}")

    model_type = config.model_type.lower()
    if model_type == "xgboost":
        return XGBoostTrainer(config.xgb_params, config)
    elif model_type == "lightgbm":
        return LGBMTrainer(config.lgb_params, config)
    elif model_type == "ensemble":
        return EnsembleTrainer(config.xgb_params, config.lgb_params, config)
    elif model_type == "gat":
        from .gat_trainer import GATTrainer
        model_dir = getattr(config, "model_dir", "models")
        return GATTrainer(config.gat_params, config, model_dir=model_dir)
    elif model_type == "ensemble_gat":
        model_dir = getattr(config, "model_dir", "models")
        return EnsembleTrainer3Way(
            config.xgb_params, config.lgb_params, config.gat_params, config
        )
    else:
        raise ValueError(f"지원하지 않는 model_type: {model_type}")
