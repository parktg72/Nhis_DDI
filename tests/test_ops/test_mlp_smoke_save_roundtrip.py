"""WS2 회귀: _save_model_smoke 가 '학습된' 모델을 TorchScript 로 저장하는지 검증.

과거 결함: _save_model_smoke 가 새 untrained MultiHotMLP 의 state_dict 를 저장해
(1) 학습 가중치가 버려지고 (2) state_dict 라 DLModel(torch.jit.load 기대)이 로드 불가.
수정 후: train_mlp_smoke(return_model=True) 가 학습 모델을 반환하고 _save_model_smoke
가 그 모델을 torch.jit.trace 로 저장한다. 본 테스트는 torch 라운드트립으로 이를 못박는다.
"""
from __future__ import annotations

import numpy as np
import pytest


def _toy_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(100, 10)).astype("float32")
    y = np.array([1] * 10 + [0] * 90, dtype=np.int64)
    return X, y


def test_save_model_smoke_persists_trained_model_as_loadable_torchscript(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts.ops.mlp_smoke_train import (
        MultiHotMLP,
        _save_model_smoke,
        train_mlp_smoke,
    )

    X, y = _toy_data()
    result, model = train_mlp_smoke(
        X,
        y,
        hidden_dims=(8, 4),
        epochs=5,
        batch_size=16,
        seed=42,
        device="cpu",
        return_model=True,
    )
    assert result.n_positive_train > 0

    model.eval()
    sample = torch.tensor(X[:5], dtype=torch.float32)
    with torch.no_grad():
        trained_out = model(sample).detach().cpu().numpy()

    out_path = tmp_path / "model.pt"
    _save_model_smoke(model, X.shape[1], out_path, device="cpu")

    # DLModel 과 동일하게 torch.jit.load 로 로드 가능해야 한다(state_dict 면 실패).
    loaded = torch.jit.load(str(out_path))
    loaded.eval()
    with torch.no_grad():
        loaded_out = loaded(sample).detach().cpu().numpy()

    # 저장된 모델 == 학습된 모델: forward 출력이 동일해야 한다.
    np.testing.assert_allclose(loaded_out, trained_out, rtol=1e-5, atol=1e-6)

    # 회귀 가드: 새 untrained 모델(과거 버그가 저장하던 것)은 학습 모델과 다른 출력을 낸다.
    untrained = MultiHotMLP(input_dim=X.shape[1], hidden_dims=(8, 4))
    untrained.eval()
    with torch.no_grad():
        untrained_out = untrained(sample).detach().cpu().numpy()
    assert not np.allclose(loaded_out, untrained_out, rtol=1e-3, atol=1e-4)
