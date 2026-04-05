"""GAT 구성요소 유닛 테스트."""
import pytest
import numpy as np
import pandas as pd


class TestGATDataset:
    """GATDataset은 PyTorch 의존성 없음 — 항상 실행."""

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
        from scripts.train.gat_dataset import GATDataset
        ds = GATDataset(prescription_df=prescription_df, ddi_df=ddi_df)
        assert ds.prescription_df is prescription_df
        assert ds.ddi_df is ddi_df
        assert ds.pairs_train is None
        assert ds.pairs_gat_val is None
        assert ds.pairs_calibration is None

    def test_unique_drugs_sorted(self, prescription_df, ddi_df):
        from scripts.train.gat_dataset import GATDataset
        ds = GATDataset(prescription_df=prescription_df, ddi_df=ddi_df)
        # sorted 순서 검증 (set 비교 아님)
        assert ds.unique_drugs == ["D001", "D002", "D003"]

    def test_num_drugs(self, prescription_df, ddi_df):
        from scripts.train.gat_dataset import GATDataset
        ds = GATDataset(prescription_df=prescription_df, ddi_df=ddi_df)
        assert ds.num_drugs == 3


# Note: Later test classes (TestGraphBuilder, TestGATModel, etc.) will add their own
# pytest.importorskip guards at class or function level for PyTorch/torch_geometric deps.
