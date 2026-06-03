"""sparse-linear 익스포터 라운드트립/계약 테스트.

학습 nn.Linear(input_dim, 1)(sigmoid) 을 서빙 DL 번들(2-output, softmax)로
export 했을 때, 서빙이 돌려주는 probabilities["high"] 가 학습 sigmoid 점수와
정확히 일치함을 검증한다. 함정(가중치 미전이/인코더 불일치/argmax 혼동)을
분리해 잡기 위해 isolation 단언과 encoder-path 단언을 따로 둔다.

근거: 2026-06-02 ML/DL 리뷰 B0 (docs/reports/2026-06-02_ml_dl_and_diskfull_review.md).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from scripts.datasets.export_sparse_linear_bundle import (
    export_from_torch_linear,
    export_sparse_linear_bundle,
)
from scripts.ops.multihot_encoder import encode_patient_history
from serving.dl_predictor import DLModel


VOCAB = {"_unk": 0, "A": 1, "B": 2, "C": 3}
# 결정적 "학습" head — _unk/A/B/C 차원 가중치 + bias.
WEIGHT = [0.0, 1.5, -2.0, 0.5]
BIAS = -0.3


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _history(drug_codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "patient_id": ["P1"] * len(drug_codes),
        "drug_code": drug_codes,
        "prescription_date": ["20260510"] * len(drug_codes),
    })


def _export(tmp_path, **overrides):
    kwargs = dict(
        weight=WEIGHT,
        bias=BIAS,
        drug_vocab=VOCAB,
        run_id="test-sparse-linear",
        device="cpu",
    )
    kwargs.update(overrides)
    return export_sparse_linear_bundle(tmp_path, **kwargs)


# ── 라운드트립 parity ──────────────────────────────────────────────────────

def test_served_high_prob_equals_training_sigmoid(tmp_path):
    """서빙 probabilities['high'] == sigmoid(w·x + b). (argmax 가 아닌 명명 클래스 비교)"""
    _export(tmp_path)
    model = DLModel(runtime_lookback_days=365)
    assert model.load(tmp_path) is True

    history = _history(["A", "C"])  # x = [_unk=0, A=1, B=0, C=1]
    z = WEIGHT[1] * 1 + WEIGHT[3] * 1 + BIAS  # = 1.5 + 0.5 - 0.3 = 1.7
    expected_high = _sigmoid(z)

    out = model.predict(history)
    assert out["probabilities"]["high"] == pytest.approx(expected_high, rel=1e-5)
    assert out["probabilities"]["low"] == pytest.approx(1.0 - expected_high, rel=1e-5)
    # 확률 합 = 1 (softmax)
    assert sum(out["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)


def test_weight_transfer_isolation(tmp_path):
    """가중치가 실제로 전이됐는지를 인코더와 분리해 단언.

    동일 피처 벡터를 (1) 인메모리 학습 Linear(in,1)→sigmoid 와 (2) 리로드된
    2-output 번들→softmax[high] 에 직접 먹여 일치 확인. 인코더를 거치지 않으므로
    '가중치 미전이' 결함만 단독으로 잡힌다.
    """
    import torch

    trained = torch.nn.Linear(len(VOCAB), 1)
    with torch.no_grad():
        trained.weight[0] = torch.tensor(WEIGHT, dtype=trained.weight.dtype)
        trained.bias[0] = BIAS
    trained.eval()

    export_from_torch_linear(
        tmp_path, trained, VOCAB, run_id="iso", device="cpu",
    )
    model = DLModel(runtime_lookback_days=365)
    model.load(tmp_path)
    model._ensure_runtime_loaded()

    x = [0.0, 1.0, 1.0, 1.0]  # _unk off, A/B/C on — 인코더 우회
    with torch.no_grad():
        ref_logit = float(trained(torch.tensor([x], dtype=torch.float32))[0, 0])
        ref_high = _sigmoid(ref_logit)
        bundle_out = model._predict_forward(torch.tensor([x], dtype=torch.float32))
        probs = model._to_probabilities(torch, bundle_out)

    labels = model._model_config["output_labels"]
    high = probs[labels.index("high")]
    assert high == pytest.approx(ref_high, rel=1e-5)


def test_encoder_path_matches_expected_vector(tmp_path):
    """(b) 서빙 인코더가 기대 multi_hot 벡터를 만들고, 학습 인코더와도 일치."""
    _export(tmp_path)
    model = DLModel(runtime_lookback_days=365)
    model.load(tmp_path)
    model._ensure_runtime_loaded()

    history = _history(["A", "C"])
    features, known, unknown = model._encode_history(history)
    assert features == [0.0, 1.0, 0.0, 1.0]
    assert (known, unknown) == (2, 0)
    # 학습 인코더와 동등
    assert encode_patient_history(history, VOCAB).tolist() == features


def test_oov_drug_maps_to_unk_dimension(tmp_path):
    """미지 약물 → _unk(index 0) 차원 반영, high prob 가 그 가중치를 반영."""
    _export(tmp_path)
    model = DLModel(runtime_lookback_days=365)
    model.load(tmp_path)

    history = _history(["ZZZ_OOV"])  # x = [_unk=1, 0, 0, 0]
    z = WEIGHT[0] * 1 + BIAS  # _unk 가중치 0 → z = -0.3
    out = model.predict(history)
    assert out["unknown_drug_count"] == 1
    assert out["probabilities"]["high"] == pytest.approx(_sigmoid(z), rel=1e-5)


def test_bundle_reload_is_stable(tmp_path):
    """export → load → predict 를 두 번 해도 동일(결정적)."""
    _export(tmp_path)
    m1 = DLModel(runtime_lookback_days=365); m1.load(tmp_path)
    m2 = DLModel(runtime_lookback_days=365); m2.load(tmp_path)
    h = _history(["A", "B"])
    assert m1.predict(h)["probabilities"] == m2.predict(h)["probabilities"]


# ── 계약 검증 (오설정 차단) ────────────────────────────────────────────────

def test_export_rejects_vocab_without_unk(tmp_path):
    with pytest.raises(ValueError, match="_unk"):
        _export(tmp_path, drug_vocab={"A": 0, "B": 1, "C": 2, "D": 3})


def test_export_rejects_vocab_size_mismatch(tmp_path):
    # input_dim(=len(WEIGHT)=4) 과 vocab 크기(3) 불일치
    with pytest.raises(ValueError, match="input_dim"):
        _export(tmp_path, drug_vocab={"_unk": 0, "A": 1, "B": 2})


def test_export_rejects_noncontiguous_vocab(tmp_path):
    with pytest.raises(ValueError, match="contiguous"):
        _export(tmp_path, drug_vocab={"_unk": 0, "A": 1, "B": 2, "C": 5})


def test_export_rejects_multiclass_weight(tmp_path):
    # Linear(in, 2) 가중치(2행) 는 단일 head 가 아니므로 거부
    bad_weight = np.zeros((2, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="Linear"):
        _export(tmp_path, weight=bad_weight)


def test_export_accepts_2d_single_row_weight(tmp_path):
    # (1, input_dim) 형태(torch Linear.weight 원형) 도 허용
    manifest = _export(tmp_path, weight=np.asarray([WEIGHT], dtype=np.float64))
    assert manifest.exists()
