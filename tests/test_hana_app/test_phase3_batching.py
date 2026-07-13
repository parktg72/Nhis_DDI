from __future__ import annotations

import sys
import types

import numpy as np
from sklearn.base import clone


class _FakeTensor:
    def __init__(self, shape):
        if hasattr(shape, "shape"):
            self.shape = tuple(shape.shape)
        elif isinstance(shape, tuple):
            self.shape = shape
        else:
            self.shape = tuple(np.asarray(shape).shape)

    def __len__(self):
        return self.shape[0] if self.shape else 0

    def to(self, device):
        return self

    def unsqueeze(self, dim):
        shape = list(self.shape)
        if dim < 0:
            dim = len(shape) + dim + 1
        shape.insert(dim, 1)
        return _FakeTensor(tuple(shape))

    def mean(self, dim=None):
        if dim is None:
            return _FakeTensor((1,))
        shape = list(self.shape)
        if dim < 0:
            dim = len(shape) + dim
        if 0 <= dim < len(shape):
            shape.pop(dim)
        return _FakeTensor(tuple(shape))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        if len(self.shape) == 2:
            arr = np.ones(self.shape, dtype=np.float32)
            denom = max(self.shape[1], 1)
            return arr / denom
        return np.ones(self.shape or (1,), dtype=np.float32)


class _FakeParam(_FakeTensor):
    pass


class _FakeLoss:
    def backward(self):
        return None


def _install_fake_torch(monkeypatch):
    call_batch_sizes: list[int] = []

    torch_mod = types.ModuleType("torch")
    torch_mod.__path__ = []
    nn_mod = types.ModuleType("torch.nn")
    optim_mod = types.ModuleType("torch.optim")

    class Module:
        def to(self, device):
            return self

        def train(self):
            return self

        def eval(self):
            return self

        def parameters(self):
            return [_FakeParam((1, 1))]

        def __call__(self, *args, **kwargs):
            if hasattr(self, "forward"):
                return self.forward(*args, **kwargs)
            return args[0] if args else _FakeTensor((1,))

    class Linear(Module):
        def __init__(self, in_features, out_features):
            self.in_features = in_features
            self.out_features = out_features
            self.weight = _FakeParam((out_features, in_features))

        def __call__(self, x):
            call_batch_sizes.append(len(x))
            if len(x.shape) == 3:
                return _FakeTensor((x.shape[0], x.shape[1], self.out_features))
            return _FakeTensor((x.shape[0], self.out_features))

    class ReLU(Module):
        pass

    class Dropout(Module):
        def __init__(self, p=0.0):
            self.p = p

    class Sequential(Module):
        def __init__(self, *layers):
            self.layers = layers
            self.out_features = getattr(layers[-1], "out_features", 2) if layers else 2

        def __call__(self, x):
            call_batch_sizes.append(len(x))
            return _FakeTensor((len(x), self.out_features))

    class TransformerEncoderLayer(Module):
        def __init__(self, *args, **kwargs):
            pass

    class TransformerEncoder(Module):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            call_batch_sizes.append(len(x))
            return x

    class CrossEntropyLoss:
        def __call__(self, out, y):
            return _FakeLoss()

    class Adam:
        def __init__(self, params, lr=0.001):
            self.params = list(params)
            self.lr = lr

        def zero_grad(self):
            return None

        def step(self):
            return None

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            return None

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    nn_mod.Module = Module
    nn_mod.Linear = Linear
    nn_mod.ReLU = ReLU
    nn_mod.Dropout = Dropout
    nn_mod.Sequential = Sequential
    nn_mod.TransformerEncoderLayer = TransformerEncoderLayer
    nn_mod.TransformerEncoder = TransformerEncoder
    nn_mod.CrossEntropyLoss = CrossEntropyLoss
    optim_mod.Adam = Adam

    torch_mod.nn = nn_mod
    torch_mod.optim = optim_mod
    torch_mod.cuda = _Cuda
    torch_mod.float32 = "float32"
    torch_mod.int64 = "int64"
    torch_mod.long = "long"
    torch_mod.device = lambda name: name
    torch_mod.tensor = lambda arr, *args, **kwargs: _FakeTensor(np.asarray(arr))
    torch_mod.as_tensor = lambda arr, *args, **kwargs: _FakeTensor(np.asarray(arr))
    torch_mod.softmax = lambda logits, dim=1: _FakeTensor(logits.shape)
    torch_mod.no_grad = _NoGrad
    torch_mod.OutOfMemoryError = RuntimeError

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_mod)
    monkeypatch.setitem(sys.modules, "torch.optim", optim_mod)
    return call_batch_sizes


def test_temporal_transformer_trains_and_predicts_in_minibatches(monkeypatch):
    from hana_app.core.phase3_models import TemporalTransformerWrapper

    call_batch_sizes = _install_fake_torch(monkeypatch)
    model = TemporalTransformerWrapper(
        d_model=16,
        nhead=2,
        num_layers=1,
        max_epochs=2,
        batch_size=4,
        use_gpu=True,
    )

    X = np.ones((10, 3), dtype=np.float32)
    y = np.array([0, 1] * 5, dtype=np.int64)

    model.fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (10, 2)
    assert call_batch_sizes
    assert max(call_batch_sizes) <= 4
    assert clone(model).get_params()["batch_size"] == 4


def test_gnn_trains_and_predicts_in_minibatches(monkeypatch):
    from hana_app.core.phase3_models import GNNWrapper

    call_batch_sizes = _install_fake_torch(monkeypatch)
    model = GNNWrapper(hidden_dim=8, num_layers=1, max_epochs=2, batch_size=3, use_gpu=True)

    X = np.ones((8, 4), dtype=np.float32)
    y = np.array([0, 1] * 4, dtype=np.int64)

    model.fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (8, 2)
    assert call_batch_sizes
    assert max(call_batch_sizes) <= 3
    assert clone(model).get_params()["batch_size"] == 3


def test_phase3_factory_passes_batch_size_params():
    from hana_app.core.phase3_models import (
        GNNWrapper,
        TemporalTransformerWrapper,
        build_phase3_model,
    )

    gnn = build_phase3_model("gnn", params={"batch_size": 123})
    transformer = build_phase3_model("temporal_transformer", params={"batch_size": 456})

    assert isinstance(gnn, GNNWrapper)
    assert gnn.batch_size == 123
    assert isinstance(transformer, TemporalTransformerWrapper)
    assert transformer.batch_size == 456


def test_phase3_gpu_cross_validation_forces_single_worker():
    from hana_app.core import ml_runner

    assert ml_runner._effective_cv_n_jobs("temporal_transformer", -1, use_gpu=True) == 1
    assert ml_runner._effective_cv_n_jobs("gnn", 4, use_gpu=True) == 1
    assert ml_runner._effective_cv_n_jobs("tabnet", 2, use_gpu=True) == 1
    assert ml_runner._effective_cv_n_jobs("xgboost", -1, use_gpu=True) == -1
    assert ml_runner._effective_cv_n_jobs("temporal_transformer", 2, use_gpu=False) == 2


def test_train_model_uses_single_cv_worker_for_phase3_gpu(monkeypatch):
    import pandas as pd
    from sklearn import model_selection

    from hana_app.core import ml_runner

    observed = {}

    class DummyGpuGuard:
        info = "fake gpu"

        def __init__(self, fraction):
            self.fraction = fraction

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyModel:
        feature_importances_ = [0.5, 0.5]

        def fit(self, X, y, **kwargs):
            return self

        def predict(self, X):
            return [0] * len(X)

        def predict_proba(self, X):
            return [[0.8, 0.2] for _ in range(len(X))]

    def fake_cross_val_score(model, X, y, cv, scoring, n_jobs):
        observed["n_jobs"] = n_jobs
        return np.array([0.5] * cv)

    df = pd.DataFrame({
        "f1": list(range(20)),
        "f2": list(range(20, 40)),
        "risk_binary": [0, 1] * 10,
    })

    monkeypatch.setattr(ml_runner, "_has_cuda", lambda: True)
    monkeypatch.setattr(ml_runner, "_GpuMemoryGuard", DummyGpuGuard)
    monkeypatch.setattr(ml_runner, "_detect_system_ram_mb", lambda: 16384)
    monkeypatch.setattr(ml_runner, "_build_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(ml_runner, "_save_result", lambda result: None)
    monkeypatch.setattr(model_selection, "cross_val_score", fake_cross_val_score)

    ml_runner.train_model(
        df=df,
        model_name="temporal_transformer",
        target="risk_binary",
        feature_cols=["f1", "f2"],
        cv_folds=2,
        memory_limit_mb=8192,
    )

    assert observed["n_jobs"] == 1
