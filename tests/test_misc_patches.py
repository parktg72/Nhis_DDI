import numpy as np
import pandas as pd
import pytest


def test_stratified_split_has_red_in_all_splits():
    """Each split must contain at least 1 Red sample when dataset has enough Red."""
    from scripts.train.dataset import _split_dataset
    rng_data = np.random.default_rng(0)
    n = 300
    risk = ["Red"] * 20 + ["Green"] * 280
    df = pd.DataFrame({
        "patient_id": range(n),
        "drug_count": rng_data.integers(5, 15, n).astype(float),
        "age": rng_data.integers(40, 80, n).astype(float),
        "risk_level": risk,
    })
    ds = _split_dataset(df, val_ratio=0.15, test_ratio=0.15, random_state=42)
    assert ds.y_val.sum() > 0, "val split missing Red"
    assert ds.y_test.sum() > 0, "test split missing Red"


def test_psi_counts_overflow():
    """PSI must count values outside reference range."""
    from monitoring.drift_detector import compute_psi_continuous
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    cur = np.array([1.0, 2.0, 100.0, 200.0, 300.0] * 20)
    psi, _, _, _ = compute_psi_continuous(ref, cur, n_bins=5)
    assert psi > 0.25, f"PSI underestimated: {psi}"


def test_all_reasons_no_duplicate():
    """dup_reasons must not appear twice in risk_reasons."""
    from unittest.mock import patch, MagicMock
    import threading, time
    from serving.predictor import HybridPredictor
    from serving.schemas import RiskLevel, PredictRequest, DrugItem

    pred = HybridPredictor.__new__(HybridPredictor)
    pred._ml_lock = threading.RLock()
    pred._ml = MagicMock(loaded=False)
    pred._ddi_matrix = None
    pred._safety_net = None
    pred._dup_detector = None
    pred._builder = MagicMock()
    pred._start_time = time.time()

    with patch("serving.predictor._run_safety_net", return_value=(RiskLevel.NORMAL, [], [])), \
         patch("serving.predictor._run_duplicate_detector", return_value=(1, ["동일성분중복 1건"])):
        req = PredictRequest(
            patient_id="p1",
            drugs=[DrugItem(edi_code="A001", drug_name="aspirin", total_days=30)],
        )
        resp = pred.predict(req)

    count = resp.risk_reasons.count("동일성분중복 1건")
    assert count == 1, f"Duplicate reason appears {count} times"
