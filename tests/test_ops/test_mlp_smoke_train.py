from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


class _FakeProvider:
    def get_history_batch(self, patient_ids, reference_date: date, lookback_days: int = 60):
        del reference_date, lookback_days
        rows = []
        for patient_id in patient_ids:
            rows.append({"patient_id": patient_id, "drug_code": "D1"})
        return pd.DataFrame(rows)


def test_mlp_forward_shape() -> None:
    import torch
    from scripts.ops.mlp_smoke_train import MultiHotMLP

    model = MultiHotMLP(input_dim=10, hidden_dims=(8, 4))
    result = model(torch.zeros((3, 10), dtype=torch.float32))

    assert tuple(result.shape) == (3, 1)


def test_pos_weight_correct() -> None:
    from scripts.ops.mlp_smoke_train import compute_pos_weight

    y = np.array([1] * 10 + [0] * 90)

    assert compute_pos_weight(y) == 9.0


def test_pos_weight_zero_positive_falls_back_to_one() -> None:
    from scripts.ops.mlp_smoke_train import compute_pos_weight

    assert compute_pos_weight(np.zeros(10, dtype=np.int64)) == 1.0


def test_build_dataset_shape() -> None:
    from scripts.ops.mlp_smoke_train import build_dataset

    vocab = {"_unk": 0, "D1": 1}
    labels = {"P1": 1, "P2": 0}

    X, y = build_dataset(
        _FakeProvider(),
        vocab,
        ["P1", "P2"],
        labels,
        reference_date=date(2024, 11, 30),
    )

    assert X.shape == (2, len(vocab))
    assert y.tolist() == [1, 0]


def test_build_training_labels_supports_multi_institution_threshold() -> None:
    from scripts.ops.mlp_smoke_train import build_training_labels

    histories = pd.DataFrame([
        {"patient_id": "P1", "institution_id": "H001"},
        {"patient_id": "P1", "institution_id": "H002"},
        {"patient_id": "P1", "institution_id": "H003"},
        {"patient_id": "P2", "institution_id": "H001"},
        {"patient_id": "P2", "institution_id": "H002"},
    ])

    labels, metadata = build_training_labels(
        ["P1", "P2"],
        histories,
        label_source="multi_institution",
        multi_institution_threshold=3,
    )

    assert labels == {"P1": 1, "P2": 0}
    assert metadata["label_source"] == "multi_institution"
    assert metadata["multi_institution_threshold"] == 3


def test_build_training_labels_supports_therapeutic_duplication_threshold() -> None:
    from scripts.ops.mlp_smoke_train import build_training_labels

    histories = pd.DataFrame([
        {"patient_id": "P1", "efmdc_clsf_no": "114", "drug_code": "D1"},
        {"patient_id": "P1", "efmdc_clsf_no": "114", "drug_code": "D2"},
        {"patient_id": "P1", "efmdc_clsf_no": "396", "drug_code": "D3"},
        {"patient_id": "P1", "efmdc_clsf_no": "396", "drug_code": "D4"},
        {"patient_id": "P2", "efmdc_clsf_no": "114", "drug_code": "D1"},
        {"patient_id": "P2", "efmdc_clsf_no": "114", "drug_code": "D1"},
    ])

    labels, metadata = build_training_labels(
        ["P1", "P2"],
        histories,
        label_source="therapeutic_duplication",
        therapeutic_dup_threshold=2,
    )

    assert labels == {"P1": 1, "P2": 0}
    assert metadata["label_source"] == "therapeutic_duplication"
    assert metadata["therapeutic_dup_threshold"] == 2


def test_stratified_split_positives_in_both() -> None:
    from scripts.ops.mlp_smoke_train import stratified_train_val_indices

    y = np.array([1] * 10 + [0] * 90)
    train_idx, val_idx = stratified_train_val_indices(y, val_fraction=0.2, seed=42)

    assert int(y[train_idx].sum()) > 0
    assert int(y[val_idx].sum()) > 0
    assert len(train_idx) == 80
    assert len(val_idx) == 20


def test_stratified_split_single_positive_falls_back_without_error() -> None:
    from scripts.ops.mlp_smoke_train import stratified_train_val_indices

    y = np.array([1] + [0] * 19)
    train_idx, val_idx = stratified_train_val_indices(y, val_fraction=0.2, seed=42)

    assert len(train_idx) == 16
    assert len(val_idx) == 4


def test_mlp_smoke_train_minimal() -> None:
    from scripts.ops.mlp_smoke_train import train_mlp_smoke

    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 10)).astype("float32")
    y = np.array([1] * 10 + [0] * 90, dtype=np.int64)

    result = train_mlp_smoke(
        X,
        y,
        hidden_dims=(8, 4),
        epochs=5,
        batch_size=16,
        seed=42,
        device="cpu",
    )

    assert result.train_loss_final >= 0
    assert 0 <= result.val_auc <= 1
    assert result.n_positive_train > 0
    assert result.n_positive_val > 0


def test_result_fields_present() -> None:
    from scripts.ops.mlp_smoke_train import SmokeTrainResult

    result = SmokeTrainResult(
        train_loss_final=0.5,
        val_auc=0.6,
        val_precision=0.2,
        val_recall=0.3,
        n_train=80,
        n_val=20,
        n_positive_train=8,
        n_positive_val=2,
        elapsed_sec=1.2,
    )

    assert result.val_auc == 0.6
    assert 0 <= result.val_auc <= 1
    assert result.n_positive_train == 8


def test_training_is_seed_deterministic() -> None:
    from scripts.ops.mlp_smoke_train import train_mlp_smoke

    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 6)).astype("float32")
    y = np.array([1] * 12 + [0] * 48, dtype=np.int64)

    first = train_mlp_smoke(X, y, hidden_dims=(4,), epochs=3, batch_size=12, seed=123)
    second = train_mlp_smoke(X, y, hidden_dims=(4,), epochs=3, batch_size=12, seed=123)

    assert first.train_loss_final == second.train_loss_final
    assert first.val_auc == second.val_auc
