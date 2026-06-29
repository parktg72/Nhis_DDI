from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier

from hana_app.core import ml_runner
from hana_app.core.phase3_models import (
    GNNWrapper,
    TabNetWrapper,
    TemporalTransformerWrapper,
)


class _FakeLGBMClassifier(BaseEstimator, ClassifierMixin):
    created: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).created.append(kwargs)

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.75, 0.25], (len(X), 1))


class _FakeXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.75, 0.25], (len(X), 1))


class _FakeCatBoostClassifier(BaseEstimator, ClassifierMixin):
    fit_sample_weights: list[object] = []

    def __init__(
        self,
        iterations=200,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        verbose=0,
        random_seed=42,
        task_type="CPU",
        thread_count=-1,
        class_weights=None,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.verbose = verbose
        self.random_seed = random_seed
        self.task_type = task_type
        self.thread_count = thread_count
        # CatBoost currently normalizes/copies this constructor parameter; emulate
        # that sklearn clone-incompatibility so the regression catches it.
        self.class_weights = list(class_weights) if class_weights is not None else None

    def fit(self, X, y, sample_weight=None):
        type(self).fit_sample_weights.append(sample_weight)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.arange(len(X), dtype=int) % 2

    def predict_proba(self, X):
        return np.tile([0.75, 0.25], (len(X), 1))

    def get_feature_importance(self):
        return np.ones(2)


def _install_fake_module(monkeypatch, module_name: str, **attrs):
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def test_lightgbm_does_not_enable_cuda_just_because_gpu_exists(monkeypatch):
    _FakeLGBMClassifier.created = []
    _install_fake_module(monkeypatch, "lightgbm", LGBMClassifier=_FakeLGBMClassifier)

    model = ml_runner._build_model("lightgbm", "risk_binary", params={}, use_gpu=True)

    assert isinstance(model, _FakeLGBMClassifier)
    assert _FakeLGBMClassifier.created
    assert _FakeLGBMClassifier.created[-1].get("device_type") != "cuda"


def test_stacking_base_lightgbm_uses_cpu_when_gpu_is_only_nvidia_smi(monkeypatch):
    _FakeLGBMClassifier.created = []
    _install_fake_module(monkeypatch, "lightgbm", LGBMClassifier=_FakeLGBMClassifier)
    _install_fake_module(monkeypatch, "xgboost", XGBClassifier=_FakeXGBClassifier)

    model = ml_runner._build_model(
        "stacking",
        "risk_binary",
        params={"base_models": ["xgboost", "lightgbm", "random_forest"]},
        use_gpu=True,
    )

    assert [name for name, _ in model.estimators] == ["xgboost", "lightgbm", "random_forest"]
    assert _FakeLGBMClassifier.created
    assert _FakeLGBMClassifier.created[-1].get("device_type") != "cuda"


def test_catboost_cost_sensitive_training_is_clone_safe_and_uses_sample_weight(monkeypatch):
    _FakeCatBoostClassifier.fit_sample_weights = []
    _install_fake_module(monkeypatch, "catboost", CatBoostClassifier=_FakeCatBoostClassifier)
    monkeypatch.setattr(ml_runner, "_save_result", lambda result: None)
    monkeypatch.setattr(ml_runner, "_has_cuda", lambda: False)

    df = pd.DataFrame(
        {
            "drug_count": list(range(12)),
            "age": [40, 41, 42, 43, 44, 45, 60, 61, 62, 63, 64, 65],
            "risk_binary": [0, 1] * 6,
        }
    )

    built = ml_runner._build_model(
        "catboost",
        "risk_binary",
        params={},
        use_gpu=False,
        cost_sensitive=True,
        cost_fp=1.0,
        cost_fn=5.0,
    )
    clone(built)

    ml_runner.train_model(
        df=df,
        model_name="catboost",
        target="risk_binary",
        params={},
        test_size=0.25,
        cv_folds=3,
        sampling_size=0,
        cost_sensitive=True,
        cost_fp=1.0,
        cost_fn=5.0,
        feature_cols=["drug_count", "age"],
    )

    assert any(weight is not None for weight in _FakeCatBoostClassifier.fit_sample_weights)


def test_catboost_does_not_enable_gpu_just_because_nvidia_smi_exists(monkeypatch):
    _install_fake_module(monkeypatch, "catboost", CatBoostClassifier=_FakeCatBoostClassifier)

    model = ml_runner._build_model("catboost", "risk_binary", params={}, use_gpu=True)

    assert getattr(model, "task_type") == "CPU"


def test_phase3_wrappers_are_sklearn_classifiers_under_sklearn_18():
    for wrapper in [
        TabNetWrapper(max_epochs=1),
        GNNWrapper(max_epochs=1),
        TemporalTransformerWrapper(max_epochs=1),
    ]:
        assert is_classifier(wrapper)
        clone(wrapper)
