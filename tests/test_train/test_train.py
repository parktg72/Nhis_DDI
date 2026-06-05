"""
scripts/train 단위/통합 테스트
xgboost/lightgbm/sklearn 없이 numpy mock으로 실행 가능
"""
import pickle
import pytest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train.dataset import TrainDataset, load_dataset_from_df, RISK_ORDER
from scripts.train.evaluator import (
    compute_metrics, find_optimal_threshold, evaluate_all_splits,
    EvalResult, _numpy_auc_roc,
)
from scripts.train.hyperparams import TrainConfig, XGB_DEFAULT, LGB_DEFAULT
from scripts.train.experiment import ExperimentTracker
from scripts.train.pipeline import TrainResult


# ─────────────────────────────────────────────────────────────────────────────
# 공통 픽스처
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def binary_labels(rng):
    """불균형 이진 레이블 (15% Red)."""
    n = 200
    y = (rng.random(n) < 0.15).astype(int)
    return y


@pytest.fixture
def good_proba(binary_labels, rng):
    """실제 레이블과 높은 상관의 확률값 (AUC ≈ 0.9 이상)."""
    noise = rng.normal(0, 0.1, len(binary_labels))
    prob = np.clip(binary_labels * 0.7 + 0.15 + noise, 0.01, 0.99)
    return prob


@pytest.fixture
def random_proba(rng, binary_labels):
    """무작위 확률값 (AUC ≈ 0.5)."""
    return rng.random(len(binary_labels))


@pytest.fixture
def sample_ml_df(rng):
    """ml_features_{partition}.parquet 모사 DataFrame."""
    n = 150
    return pd.DataFrame({
        "patient_id":          [f"P{i:04d}" for i in range(n)],
        "drug_count":          rng.uniform(1, 12, n),
        "ddi_contraindicated": (rng.random(n) < 0.1).astype(float),
        "ddi_major":           rng.integers(0, 5, n).astype(float),
        "ddi_moderate":        rng.integers(0, 8, n).astype(float),
        "ddi_minor":           rng.integers(0, 10, n).astype(float),
        "triple_whammy":       (rng.random(n) < 0.05).astype(float),
        "qt_risk_count":       rng.integers(0, 4, n).astype(float),
        "institution_count":   rng.integers(1, 5, n).astype(float),
        "cyp_risk_score":      rng.uniform(0, 10, n),
        "age":                 rng.integers(30, 85, n).astype(float),
        "sex_male":            (rng.random(n) < 0.5).astype(float),
        "risk_level":          rng.choice(["Red","Yellow","Green","Normal"],
                                          n, p=[0.10, 0.20, 0.25, 0.45]),
        "is_high_risk":        None,  # 아래서 설정
        "window_start":        pd.Timestamp("2024-01-01"),
        "window_end":          pd.Timestamp("2024-03-31"),
    })


@pytest.fixture
def sample_ml_df_with_label(sample_ml_df):
    df = sample_ml_df.copy()
    df["is_high_risk"] = (df["risk_level"] == "Red").astype(int)
    return df


@pytest.fixture
def sample_dataset(sample_ml_df_with_label):
    return load_dataset_from_df(sample_ml_df_with_label, random_state=42)


# ─────────────────────────────────────────────────────────────────────────────
# TrainDataset 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainDataset:
    def test_split_sizes(self, sample_dataset):
        n = sample_dataset.n_train + sample_dataset.n_val + sample_dataset.n_test
        assert n == 150

    def test_feature_names_correct(self, sample_dataset):
        non_feat = {"patient_id", "risk_level", "is_high_risk",
                    "risk_level_encoded", "window_start", "window_end"}
        for col in sample_dataset.feature_names:
            assert col not in non_feat

    def test_pos_weight_positive(self, sample_dataset):
        assert sample_dataset.pos_weight > 0

    def test_pos_weight_reflects_imbalance(self, sample_dataset):
        """Red 10% → pos_weight ≈ 9."""
        assert sample_dataset.pos_weight > 2.0

    def test_class_distribution_sums(self, sample_dataset):
        dist = sample_dataset.class_distribution()
        for split in ["train", "val", "test"]:
            total = dist[split]["pos"] + dist[split]["neg"]
            n = getattr(sample_dataset, f"n_{split}")
            assert total == n

    def test_no_label_in_features(self, sample_dataset):
        feat_set = set(sample_dataset.feature_names)
        assert "is_high_risk" not in feat_set
        assert "risk_level" not in feat_set

    def test_x_shape(self, sample_dataset):
        assert sample_dataset.X_train.shape[1] == sample_dataset.n_features
        assert sample_dataset.X_val.shape[1] == sample_dataset.n_features

    def test_risk_level_encoded_range(self, sample_dataset):
        for arr in [sample_dataset.y_multi_train,
                    sample_dataset.y_multi_val,
                    sample_dataset.y_multi_test]:
            assert arr.min() >= 0
            assert arr.max() <= 3

    def test_load_from_df_without_is_high_risk(self, sample_ml_df):
        """is_high_risk 없어도 risk_level로 자동 생성."""
        df = sample_ml_df.copy()
        df = df.drop(columns=["is_high_risk"])
        ds = load_dataset_from_df(df)
        assert ds.y_train.max() <= 1

    def test_reproducible_split(self, sample_ml_df_with_label):
        ds1 = load_dataset_from_df(sample_ml_df_with_label, random_state=42)
        ds2 = load_dataset_from_df(sample_ml_df_with_label, random_state=42)
        np.testing.assert_array_equal(ds1.y_train, ds2.y_train)

    def test_meta_cols_include_yellow_subtype(self):
        """TrainDataset.meta_* 에 yellow_subtype 이 subgroup 조인 키로 포함되어야 함."""
        df = pd.DataFrame({
            "patient_id":    [f"P{i:04d}" for i in range(20)],
            "window_start":  ["2026-01-01"] * 20,
            "window_end":    ["2026-03-31"] * 20,
            "risk_level":    (["Red"] * 5 + ["Yellow"] * 10 + ["Normal"] * 5),
            "yellow_subtype": ([None] * 5 + ["Y_DDI_MAJOR"] * 5 + ["Y_TRIPLE"] * 5 + [None] * 5),
            "drug_count":    [5] * 20,
            "ddi_major":     [1] * 20,
        })
        ds = load_dataset_from_df(df, random_state=42)
        for split_name in ("train", "val", "test"):
            meta = getattr(ds, f"meta_{split_name}")
            if len(meta) > 0:
                assert "yellow_subtype" in meta.columns, (
                    f"meta_{split_name} 에 yellow_subtype 없음 (subgroup 분석 불가)"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluator:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        res = compute_metrics(y_true, y_prob, threshold=0.5)
        assert res.recall == 1.0
        assert res.precision == 1.0
        assert res.tp == 2
        assert res.fn == 0
        assert res.fp == 0

    def test_all_wrong_classifier(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.1, 0.1, 0.9, 0.9])
        res = compute_metrics(y_true, y_prob, threshold=0.5)
        assert res.recall == 0.0
        assert res.tp == 0
        assert res.fn == 2

    def test_auc_random_approx_half(self, random_proba, binary_labels):
        res = compute_metrics(binary_labels, random_proba, threshold=0.5)
        assert 0.3 < res.auc_roc < 0.7

    def test_auc_good_classifier_high(self, good_proba, binary_labels):
        res = compute_metrics(binary_labels, good_proba, threshold=0.3)
        assert res.auc_roc > 0.70

    def test_recall_threshold_priority(self, good_proba, binary_labels):
        """낮은 임계값 → Recall 증가."""
        res_low  = compute_metrics(binary_labels, good_proba, threshold=0.2)
        res_high = compute_metrics(binary_labels, good_proba, threshold=0.8)
        assert res_low.recall >= res_high.recall

    def test_passed_flags(self):
        res = EvalResult("val", auc_roc=0.90, recall=0.92)
        assert res.passed_auc
        assert res.passed_recall
        assert res.passed

    def test_failed_flags(self):
        res = EvalResult("val", auc_roc=0.80, recall=0.85)
        assert not res.passed_auc
        assert not res.passed_recall
        assert not res.passed

    def test_find_optimal_threshold_recall(self, good_proba, binary_labels):
        thresh, res = find_optimal_threshold(binary_labels, good_proba, min_recall=0.80)
        assert 0.0 < thresh < 1.0
        assert res.recall >= 0.79  # 허용 오차

    def test_f1_formula(self):
        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_prob = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.9])
        res = compute_metrics(y_true, y_prob, threshold=0.5)
        expected_f1 = 2 * res.precision * res.recall / (res.precision + res.recall + 1e-9)
        assert abs(res.f1 - expected_f1) < 0.001

    def test_numpy_auc_roc_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        auc = _numpy_auc_roc(y_true, y_prob)
        assert abs(auc - 1.0) < 0.05

    def test_numpy_auc_roc_random(self, rng):
        n = 100
        y_true = (rng.random(n) > 0.7).astype(int)
        y_prob = rng.random(n)
        auc = _numpy_auc_roc(y_true, y_prob)
        assert 0.0 <= auc <= 1.0

    def test_evaluate_all_splits_threshold_consistency(self, good_proba, binary_labels, rng):
        """모든 split에 동일 임계값 적용 검증."""
        n = len(binary_labels)
        half = n // 2
        results = evaluate_all_splits(
            y_true_tr=binary_labels[:half],
            y_prob_tr=good_proba[:half],
            y_true_va=binary_labels[half:],
            y_prob_va=good_proba[half:],
            y_true_te=binary_labels[:half],
            y_prob_te=good_proba[:half],
        )
        # val로 결정된 임계값이 train/test에도 동일 적용
        assert results["train"].threshold == results["val"].threshold == results["test"].threshold


# ─────────────────────────────────────────────────────────────────────────────
# TrainConfig 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainConfig:
    def test_default_model_type(self):
        cfg = TrainConfig()
        assert cfg.model_type == "xgboost"

    def test_get_model_params_xgb(self):
        cfg = TrainConfig(model_type="xgboost")
        params = cfg.get_model_params()
        assert "objective" in params
        assert params["objective"] == "binary:logistic"

    def test_get_model_params_lgb(self):
        cfg = TrainConfig(model_type="lightgbm")
        params = cfg.get_model_params()
        assert params["objective"] == "binary"
        assert params.get("is_unbalance") is True

    def test_recall_threshold_default(self):
        cfg = TrainConfig()
        assert cfg.recall_threshold == 0.90

    def test_xgb_default_immutable(self):
        """인스턴스 수정이 전역 기본값에 영향 없음."""
        cfg1 = TrainConfig()
        cfg1.xgb_params["n_estimators"] = 999
        cfg2 = TrainConfig()
        assert cfg2.xgb_params["n_estimators"] == XGB_DEFAULT["n_estimators"]


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentTracker 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentTracker:
    def test_log_metric_buffered(self, tmp_path):
        tracker = ExperimentTracker("test_exp")
        tracker.start_run("test_run")
        tracker.log_metric("auc", 0.87)
        assert tracker._log_buffer["metrics"]["auc"] == 0.87

    def test_log_params_buffered(self, tmp_path):
        tracker = ExperimentTracker("test_exp")
        tracker.start_run("test_run")
        tracker.log_params({"lr": 0.05, "n_est": 500})
        assert tracker._log_buffer["params"]["lr"] == 0.05

    def test_end_run_saves_json(self, tmp_path):
        log_dir = tmp_path / "mlruns" / "local"
        tracker = ExperimentTracker("test_exp")
        tracker.start_run("json_run")
        tracker.log_metric("recall", 0.91)
        tracker.end_run(log_dir=str(log_dir))
        files = list(log_dir.glob("*.json"))
        assert len(files) == 1
        import json
        data = json.loads(files[0].read_text())
        assert data["metrics"]["recall"] == 0.91

    def test_log_eval_result(self, tmp_path):
        tracker = ExperimentTracker("test_exp")
        tracker.start_run()
        res = EvalResult("val", auc_roc=0.88, recall=0.91, precision=0.60, f1=0.72)
        tracker.log_eval_result(res)
        metrics = tracker._log_buffer["metrics"]
        assert metrics["val_auc_roc"] == 0.88
        assert metrics["val_recall"] == 0.91


# ─────────────────────────────────────────────────────────────────────────────
# Mock Trainer 통합 테스트
# ─────────────────────────────────────────────────────────────────────────────

class MockTrainer:
    """xgboost/lightgbm 없이 동작하는 테스트용 Mock Trainer."""

    def __init__(self, recall_target: float = 0.92, auc_target: float = 0.88):
        self.recall_target = recall_target
        self.auc_target = auc_target
        self.feature_importances_ = None
        self.best_threshold_ = 0.3
        self._trained = False

    def fit(self, dataset: TrainDataset) -> "MockTrainer":
        self.feature_importances_ = np.ones(dataset.n_features) / dataset.n_features
        self._trained = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """레이블 기반 확률: 첫 번째 피처(drug_count)가 높으면 Red 확률 높음."""
        rng = np.random.default_rng(0)
        # 간단한 선형 조합 + 노이즈
        score = X[:, 0] / (X[:, 0].max() + 1e-9)
        noise = rng.normal(0, 0.05, len(X))
        return np.clip(score + noise, 0.01, 0.99)

    def evaluate(self, dataset: TrainDataset, min_recall: float = 0.90):
        from scripts.train.evaluator import evaluate_all_splits
        return evaluate_all_splits(
            y_true_tr=dataset.y_train,
            y_prob_tr=self.predict_proba(dataset.X_train),
            y_true_va=dataset.y_val,
            y_prob_va=self.predict_proba(dataset.X_val),
            y_true_te=dataset.y_test,
            y_prob_te=self.predict_proba(dataset.X_test),
            min_recall=min_recall,
        )

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"mock": True, "threshold": self.best_threshold_}, f)
        return path

    def feature_importance_df(self, feature_names):
        import pandas as pd
        return pd.DataFrame({
            "feature": feature_names,
            "importance": self.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


class TestMockTrainerIntegration:
    def test_mock_fit_and_evaluate(self, sample_dataset):
        trainer = MockTrainer()
        trainer.fit(sample_dataset)
        results = trainer.evaluate(sample_dataset)
        assert "train" in results
        assert "val" in results
        assert "test" in results
        for res in results.values():
            assert 0.0 <= res.recall <= 1.0
            assert 0.0 <= res.auc_roc <= 1.0

    def test_mock_save_load(self, sample_dataset, tmp_path):
        trainer = MockTrainer()
        trainer.fit(sample_dataset)
        path = trainer.save(tmp_path / "test_model.pkl")
        assert path.exists()
        with open(path, "rb") as f:
            state = pickle.load(f)
        assert state["mock"] is True

    def test_feature_importance_df(self, sample_dataset):
        trainer = MockTrainer()
        trainer.fit(sample_dataset)
        df = trainer.feature_importance_df(sample_dataset.feature_names)
        assert df is not None
        assert "feature" in df.columns
        assert "importance" in df.columns
        assert len(df) == sample_dataset.n_features

    def test_threshold_consistent_with_eval(self, sample_dataset):
        """find_optimal_threshold가 val AUC 기반 임계값 반환."""
        trainer = MockTrainer()
        trainer.fit(sample_dataset)
        results = trainer.evaluate(sample_dataset, min_recall=0.80)
        val_thresh = results["val"].threshold
        train_thresh = results["train"].threshold
        assert val_thresh == train_thresh

    def test_train_result_dataclass(self):
        result = TrainResult(partition="202401", model_type="xgboost")
        result.eval_results = {
            "val": EvalResult("val", auc_roc=0.88, recall=0.91),
        }
        result.passed = True
        assert result.val_recall == 0.91
        assert result.val_auc == 0.88
        assert result.passed
