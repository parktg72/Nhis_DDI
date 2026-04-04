"""scripts/train/ 파이프라인·트레이너 단위·통합 테스트.
xgboost/lightgbm 없이 MockTrainer로 실행 가능.
"""
import hashlib
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts.train.dataset import TrainDataset, load_dataset_from_df
from scripts.train.evaluator import EvalResult
from scripts.train.hyperparams import TrainConfig
from scripts.train.pipeline import TrainPipeline, TrainResult
from scripts.train.trainer import BaseTrainer, EnsembleTrainer, build_trainer


# ─── 공통 픽스처 ──────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_dataset(rng):
    """150행 이진 분류 DataFrame → TrainDataset."""
    import pandas as pd
    n = 150
    df = pd.DataFrame({
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
        "risk_level":          rng.choice(
            ["Red", "Yellow", "Green", "Normal"], n, p=[0.10, 0.20, 0.25, 0.45]
        ),
        "window_start":        "2024-01-01",
        "window_end":          "2024-03-31",
    })
    return load_dataset_from_df(df, random_state=42)


class MockTrainer(BaseTrainer):
    """xgboost/lightgbm 없이 동작하는 테스트용 Trainer."""

    def __init__(self, params=None, config=None):
        super().__init__(params or {}, config)

    def fit(self, dataset: TrainDataset) -> "MockTrainer":
        self.feature_importances_ = np.ones(dataset.n_features) / dataset.n_features
        self._trained = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(0)
        score = X[:, 0] / (X[:, 0].max() + 1e-9)
        return np.clip(score + rng.normal(0, 0.05, len(X)), 0.01, 0.99)


# ─── BaseTrainer save/load 테스트 ─────────────────────────────────────────────

class TestBaseTrainerSaveLoad:

    def test_save_creates_pkl_and_sha256(self, sample_dataset, tmp_path):
        """save() → pkl + sha256 사이드카 생성."""
        trainer = MockTrainer()
        trainer.fit(sample_dataset)
        path = trainer.save(tmp_path / "model.pkl")
        assert path.exists()
        sha_path = path.with_suffix(".pkl.sha256")
        assert sha_path.exists()

    def test_save_sha256_matches_file_content(self, sample_dataset, tmp_path):
        """sha256 파일의 해시값이 pkl 파일 내용과 일치."""
        trainer = MockTrainer()
        trainer.fit(sample_dataset)
        path = trainer.save(tmp_path / "model.pkl")
        content = path.read_bytes()
        actual_sha = hashlib.sha256(content).hexdigest()
        stored_sha = path.with_suffix(".pkl.sha256").read_text().strip().split()[0]
        assert actual_sha == stored_sha

    def test_load_restores_threshold(self, sample_dataset, tmp_path):
        """save 후 load → best_threshold_ 복원."""
        trainer = MockTrainer()
        trainer.fit(sample_dataset)
        trainer.evaluate(sample_dataset, min_recall=0.80)
        saved_threshold = trainer.best_threshold_
        path = trainer.save(tmp_path / "model.pkl")

        loaded = MockTrainer.load(path)
        assert loaded.best_threshold_ == saved_threshold

    def test_load_sets_trained_flag(self, sample_dataset, tmp_path):
        """load 후 _trained=True."""
        trainer = MockTrainer()
        trainer.fit(sample_dataset)
        path = trainer.save(tmp_path / "model.pkl")
        loaded = MockTrainer.load(path)
        assert loaded._trained is True

    def test_evaluate_before_fit_raises(self, sample_dataset):
        """fit() 전 evaluate() → RuntimeError."""
        trainer = MockTrainer()
        with pytest.raises(RuntimeError, match="fit"):
            trainer.evaluate(sample_dataset)


# ─── EnsembleTrainer.save 테스트 ──────────────────────────────────────────────

class MockXGBTrainer(MockTrainer):
    pass


class MockLGBTrainer(MockTrainer):
    pass


class TestEnsembleTrainerSave:

    @pytest.fixture
    def mock_ensemble(self, sample_dataset):
        """MockXGB + MockLGB를 가진 EnsembleTrainer 유사 객체."""
        ens = EnsembleTrainer.__new__(EnsembleTrainer)
        ens.weights = (0.5, 0.5)
        ens.best_threshold_ = 0.45
        ens.feature_importances_ = np.ones(sample_dataset.n_features) / sample_dataset.n_features
        ens._trained = True
        ens.params = {}
        ens.config = None
        # MockTrainer가 xgb/lgb 역할
        xgb = MockTrainer()
        xgb.fit(sample_dataset)
        lgb = MockTrainer()
        lgb.fit(sample_dataset)
        ens._xgb = xgb
        ens._lgb = lgb
        return ens

    def test_save_creates_three_pkl_files(self, mock_ensemble, tmp_path):
        """EnsembleTrainer.save → main.pkl + .xgb.pkl + .lgb.pkl."""
        path = mock_ensemble.save(tmp_path / "ens.pkl")
        assert (tmp_path / "ens.pkl").exists()
        assert (tmp_path / "ens.xgb.pkl").exists()
        assert (tmp_path / "ens.lgb.pkl").exists()

    def test_save_creates_three_sha256_files(self, mock_ensemble, tmp_path):
        """EnsembleTrainer.save → 3개 sha256 사이드카."""
        mock_ensemble.save(tmp_path / "ens.pkl")
        assert (tmp_path / "ens.pkl.sha256").exists()
        assert (tmp_path / "ens.xgb.pkl.sha256").exists()
        assert (tmp_path / "ens.lgb.pkl.sha256").exists()

    def test_save_all_sha256_match_content(self, mock_ensemble, tmp_path):
        """모든 sha256 파일이 대응 pkl 내용과 일치."""
        mock_ensemble.save(tmp_path / "ens.pkl")
        for name in ("ens.pkl", "ens.xgb.pkl", "ens.lgb.pkl"):
            pkl_path = tmp_path / name
            sha_path = pkl_path.with_suffix(pkl_path.suffix + ".sha256")
            content = pkl_path.read_bytes()
            actual = hashlib.sha256(content).hexdigest()
            stored = sha_path.read_text().strip().split()[0]
            assert actual == stored, f"{name} sha256 불일치"

    def test_save_main_payload_has_trainer_class(self, mock_ensemble, tmp_path):
        """메인 pkl payload에 trainer_class=EnsembleTrainer."""
        mock_ensemble.save(tmp_path / "ens.pkl")
        state = pickle.loads((tmp_path / "ens.pkl").read_bytes())
        assert state["trainer_class"] == "EnsembleTrainer"
        assert state["weights"] == (0.5, 0.5)


# ─── build_trainer 팩토리 테스트 ──────────────────────────────────────────────

class TestBuildTrainer:
    """build_trainer 팩토리 테스트.

    XGBoostTrainer/LGBMTrainer 인스턴스 생성은 실제 xgboost/lightgbm import를
    유발하지 않는다 (lazy import: fit() 호출 시에만 import). 따라서 이 테스트는
    xgboost/lightgbm 미설치 환경에서도 안전하게 실행된다.
    """

    @pytest.fixture
    def base_config(self):
        return TrainConfig(
            model_type="xgboost",
            partition="202401",
            feature_base="data/features",
            model_dir="models",
            use_optuna=False,
        )

    def test_build_xgboost_trainer(self, base_config):
        """model_type=xgboost → XGBoostTrainer 반환."""
        from scripts.train.trainer import XGBoostTrainer
        trainer = build_trainer(base_config)
        assert isinstance(trainer, XGBoostTrainer)

    def test_build_lightgbm_trainer(self, base_config):
        """model_type=lightgbm → LGBMTrainer 반환."""
        from scripts.train.trainer import LGBMTrainer
        config = TrainConfig(
            model_type="lightgbm", partition="202401",
            feature_base="data/features", model_dir="models",
        )
        trainer = build_trainer(config)
        assert isinstance(trainer, LGBMTrainer)

    def test_build_ensemble_trainer(self, base_config):
        """model_type=ensemble → EnsembleTrainer 반환."""
        config = TrainConfig(
            model_type="ensemble", partition="202401",
            feature_base="data/features", model_dir="models",
        )
        trainer = build_trainer(config)
        assert isinstance(trainer, EnsembleTrainer)

    def test_build_unknown_type_raises(self):
        """지원하지 않는 model_type → ValueError."""
        config = TrainConfig(
            model_type="svm", partition="202401",
            feature_base="data/features", model_dir="models",
        )
        with pytest.raises(ValueError, match="svm"):
            build_trainer(config)

    def test_build_wrong_config_type_raises(self):
        """TrainConfig 아닌 객체 전달 → TypeError."""
        with pytest.raises(TypeError, match="TrainConfig"):
            build_trainer({"model_type": "xgboost"})


# ─── TrainPipeline.run 통합 테스트 ────────────────────────────────────────────

class TestTrainPipelineRun:
    """TrainPipeline.run — xgboost 없이 MockTrainer monkey-patch로 실행."""

    @pytest.fixture
    def feature_base(self, tmp_path, rng):
        """tmp_path에 ml_features_202401.parquet + scaler.pkl + selector.pkl 생성."""
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_classif

        base = tmp_path / "features"
        base.mkdir()
        n = 150
        df = pd.DataFrame({
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
            "risk_level":          rng.choice(
                ["Red", "Yellow", "Green", "Normal"], n, p=[0.10, 0.20, 0.25, 0.45]
            ),
            "window_start":        "2024-01-01",
            "window_end":          "2024-03-31",
        })
        df.to_parquet(base / "ml_features_202401.parquet", index=False)
        feat_cols = [c for c in df.columns
                     if c not in {"patient_id", "risk_level", "window_start", "window_end"}]
        X = df[feat_cols].values
        y = (df["risk_level"] == "Red").astype(int).values
        scaler = StandardScaler().fit(X)
        (base / "scaler.pkl").write_bytes(pickle.dumps(scaler))
        sel = SelectKBest(f_classif, k="all").fit(X, y)
        (base / "selector.pkl").write_bytes(pickle.dumps(sel))
        return str(base)

    @pytest.fixture
    def pipeline_config(self, tmp_path, feature_base):
        return TrainConfig(
            model_type="xgboost",
            partition="202401",
            feature_base=feature_base,
            model_dir=str(tmp_path / "models"),
            use_optuna=False,
            recall_threshold=0.0,   # 낮은 임계값 → MockTrainer 통과
            auc_threshold=0.0,
        )

    def test_run_success_creates_model_file(self, pipeline_config):
        """MockTrainer → run() 성공 → passed=True + pkl 파일 생성.

        pipeline_config: recall_threshold=0.0, auc_threshold=0.0 →
        EvalResult.min_recall=0.0, min_auc=0.0 → 모든 점수 통과.
        """
        with patch("scripts.train.pipeline.build_trainer", return_value=MockTrainer()):
            pipeline = TrainPipeline(pipeline_config)
            result = pipeline.run()
        assert result.passed is True
        assert Path(result.model_path).exists()

    def test_run_stores_feature_meta(self, pipeline_config):
        """run() 결과 모델 pkl에 feature_names, artifact_version 포함."""
        with patch("scripts.train.pipeline.build_trainer", return_value=MockTrainer()):
            pipeline = TrainPipeline(pipeline_config)
            result = pipeline.run()
        state = pickle.loads(Path(result.model_path).read_bytes())
        assert "feature_names" in state
        assert state.get("artifact_version") == 2

    def test_run_dataset_missing_raises_gracefully(self, tmp_path):
        """데이터셋 파일 없으면 run() → passed=False, 파일 관련 오류 메시지 포함."""
        config = TrainConfig(
            model_type="xgboost", partition="missing",
            feature_base=str(tmp_path / "no_such_dir"),
            model_dir=str(tmp_path / "models"),
            use_optuna=False,
        )
        pipeline = TrainPipeline(config)
        result = pipeline.run()
        assert not result.passed
        assert len(result.errors) > 0
        # 오류 메시지에 파일 미존재 관련 키워드 포함 확인 (L-4)
        assert any(
            "없음" in e or "parquet" in e.lower() or "missing" in e.lower()
            for e in result.errors
        ), f"파일 관련 오류 메시지 없음: {result.errors}"

    def test_run_failed_recall_threshold(self, pipeline_config):
        """recall_threshold=1.01 (달성 불가) → passed=False.

        recall은 최대 1.0이므로 1.01은 항상 미달. 이로써 MockTrainer의
        predict_proba 출력과 무관하게 결정적으로 failed 상태를 검증한다.
        """
        config = TrainConfig(
            model_type="xgboost",
            partition=pipeline_config.partition,
            feature_base=pipeline_config.feature_base,
            model_dir=pipeline_config.model_dir,
            use_optuna=False,
            recall_threshold=1.01,  # recall ≤ 1.0 → 반드시 미달
            auc_threshold=0.0,
        )
        with patch("scripts.train.pipeline.build_trainer", return_value=MockTrainer()):
            result = TrainPipeline(config).run()
        assert not result.passed
