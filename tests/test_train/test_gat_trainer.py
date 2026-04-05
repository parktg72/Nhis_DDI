"""GAT 구성요소 유닛 테스트. torch/torch_geometric 미설치 시 skip."""
import pytest

torch = pytest.importorskip("torch", reason="PyTorch 미설치 — GAT 테스트 건너뜀")
pytest.importorskip("torch_geometric", reason="PyG 미설치 — GAT 테스트 건너뜀")

import numpy as np
import pandas as pd
from scripts.train.gat_dataset import GATDataset


class TestGATDataset:
    @pytest.fixture
    def prescription_df(self):
        return pd.DataFrame({
            "patient_id": ["P001", "P001", "P002", "P002"],
            "drug_code":  ["D001", "D002", "D002", "D003"],
            "prescription_date": ["2024-01-01"] * 4,
        })

    @pytest.fixture
    def ddi_df(self):
        return pd.DataFrame({
            "drug_a":   ["D001", "D002"],
            "drug_b":   ["D002", "D003"],
            "severity": ["contraindicated", "major"],
        })

    def test_gat_dataset_attributes(self, prescription_df, ddi_df):
        ds = GATDataset(prescription_df=prescription_df, ddi_df=ddi_df)
        assert ds.prescription_df is prescription_df
        assert ds.ddi_df is ddi_df
        assert ds.pairs_train is None
        assert ds.pairs_gat_val is None
        assert ds.pairs_calibration is None

    def test_unique_drugs(self, prescription_df, ddi_df):
        ds = GATDataset(prescription_df=prescription_df, ddi_df=ddi_df)
        assert set(ds.unique_drugs) == {"D001", "D002", "D003"}
