"""train/serve multi_hot 인코더 동등성 계약 테스트.

학습 인코더(scripts/ops/multihot_encoder.encode_patient_history)와 서빙 인코더
(serving.dl_predictor.DLModel._encode_history)가 동일 약물 이력에 대해 동일한
피처 벡터를 만들어야 한다. 특히 미지(OOV) 약물을 vocab["_unk"] 차원에 반영하는
정책이 양쪽에서 일치해야 silent train/serve skew 가 없다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.ops.multihot_encoder import encode_patient_history
from serving.dl_predictor import DLModel


def _serving_encode(vocab: dict[str, int], history_df: pd.DataFrame) -> list[float]:
    model = DLModel(runtime_lookback_days=365)
    model._model_config = {"encoding_strategy": "multi_hot", "input_dim": len(vocab)}
    model._drug_vocab = vocab
    features, _known, _unknown = model._encode_history(history_df)
    return features


def test_encoder_parity_with_unk_known_and_oov():
    # _unk 을 포함한 실 학습 vocab 형태
    vocab = {"D1": 0, "D2": 1, "D3": 2, "_unk": 3}
    history = pd.DataFrame({
        "patient_id": ["P1", "P1", "P1"],
        "drug_code": ["D2", "ZZZ_OOV", "D1"],  # ZZZ_OOV 는 미지 약물
        "prescription_date": ["20260510", "20260510", "20260511"],
    })

    train_vec = encode_patient_history(history, vocab)
    serve_vec = _serving_encode(vocab, history)

    # 미지 약물이 _unk 차원(index 3)을 1.0 으로 set 해야 한다.
    assert serve_vec[3] == 1.0
    assert serve_vec == train_vec.tolist()
    assert serve_vec == [1.0, 1.0, 0.0, 1.0]


def test_encoder_parity_all_known():
    vocab = {"D1": 0, "D2": 1, "_unk": 2}
    history = pd.DataFrame({
        "patient_id": ["P1", "P1"],
        "drug_code": ["D1", "D2"],
        "prescription_date": ["20260510", "20260511"],
    })

    train_vec = encode_patient_history(history, vocab)
    serve_vec = _serving_encode(vocab, history)

    assert serve_vec == train_vec.tolist()
    assert serve_vec[2] == 0.0  # 미지 약물 없으면 _unk 비활성


def test_serving_backcompat_vocab_without_unk_ignores_oov():
    # _unk 없는 구형/토이 번들: 미지 약물을 종전처럼 무시(하위호환)
    vocab = {"D1": 0, "D2": 1, "D3": 2}
    history = pd.DataFrame({
        "patient_id": ["P1", "P1", "P1"],
        "drug_code": ["D2", "UNKNOWN", "D1"],
        "prescription_date": ["20260510", "20260510", "20260511"],
    })

    serve_vec = _serving_encode(vocab, history)
    assert serve_vec == [1.0, 1.0, 0.0]


def test_encoder_parity_with_empty_and_nan_codes():
    # 학습 인코더는 NaN/None/pd.NA/빈 코드를 제거한다. 서빙도 동일해야 한다.
    vocab = {"D1": 0, "D2": 1, "_unk": 2}
    history = pd.DataFrame({
        "patient_id": ["P1", "P1", "P1", "P1", "P1", "P1"],
        "drug_code": ["D1", "", np.nan, None, pd.NA, "D2"],
        "prescription_date": ["20260510"] * 6,
    })

    train_vec = encode_patient_history(history, vocab)
    serve_vec = _serving_encode(vocab, history)

    assert serve_vec == train_vec.tolist()
    # 빈/NaN/None/pd.NA 는 미지 약물로 세지 않음 → _unk 비활성
    assert serve_vec[2] == 0.0
    assert serve_vec == [1.0, 1.0, 0.0]
