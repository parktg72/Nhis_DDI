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


class TestGraphBuilder:
    torch = pytest.importorskip("torch", reason="PyTorch 미설치")
    pyg = pytest.importorskip("torch_geometric", reason="PyG 미설치")

    @pytest.fixture
    def prescription_df(self):
        """3명 환자, 여러 약물 처방."""
        return pd.DataFrame({
            "patient_id": ["P1","P1","P1","P2","P2","P3"],
            "drug_code":  ["D1","D2","D3","D1","D2","D4"],
            "prescription_date": ["2024-01-01"]*6,
        })

    @pytest.fixture
    def ddi_df(self):
        return pd.DataFrame({
            "drug_a":   ["D1","D2"],
            "drug_b":   ["D2","D3"],
            "severity": ["contraindicated","major"],
        })

    def test_coprescription_pairs_created(self, prescription_df, ddi_df):
        """동일 patient_id + prescription_date → 엣지 생성."""
        from scripts.features.graph_builder import GraphBuilder
        builder = GraphBuilder()
        data = builder.build(prescription_df, ddi_df)
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] > 0

    def test_edge_weights_in_range(self, prescription_df, ddi_df):
        """엣지 가중치 [0, 1] 범위."""
        from scripts.features.graph_builder import GraphBuilder
        builder = GraphBuilder()
        data = builder.build(prescription_df, ddi_df)
        assert data.edge_weight is not None
        assert float(data.edge_weight.min()) >= 0.0
        assert float(data.edge_weight.max()) <= 1.0 + 1e-6

    def test_unknown_drug_returns_none_from_idx(self, prescription_df, ddi_df):
        """미지 약물 코드는 drug_to_idx에 없음 → get() returns None."""
        from scripts.features.graph_builder import GraphBuilder
        builder = GraphBuilder()
        builder.build(prescription_df, ddi_df)
        assert builder.drug_to_idx.get("UNKNOWN_DRUG") is None

    def test_isolated_node_warning(self, caplog):
        """고립 노드 비율 > 10% → WARNING 로그."""
        import logging
        from scripts.features.graph_builder import GraphBuilder
        # D1 단독 처방 → 고립 노드
        df = pd.DataFrame({
            "patient_id": ["P1"],
            "drug_code": ["D1"],
            "prescription_date": ["2024-01-01"],
        })
        ddi = pd.DataFrame({"drug_a": [], "drug_b": [], "severity": []})
        builder = GraphBuilder()
        with caplog.at_level(logging.WARNING):
            builder.build(df, ddi)
        assert any("고립 노드" in r.message for r in caplog.records)

    def test_save_creates_artifacts(self, prescription_df, ddi_df, tmp_path):
        """save() → gat_graph.pt + .sha256 + gat_graph_meta.json 생성."""
        from scripts.features.graph_builder import GraphBuilder
        builder = GraphBuilder()
        builder.build(prescription_df, ddi_df)
        builder.save(tmp_path)
        assert (tmp_path / "gat_graph.pt").exists()
        assert (tmp_path / "gat_graph.pt.sha256").exists()
        assert (tmp_path / "gat_graph_meta.json").exists()

    def test_load_verifies_sha256(self, prescription_df, ddi_df, tmp_path):
        """sha256 불일치 → RuntimeError."""
        from scripts.features.graph_builder import GraphBuilder
        builder = GraphBuilder()
        builder.build(prescription_df, ddi_df)
        builder.save(tmp_path)
        sha_path = tmp_path / "gat_graph.pt.sha256"
        sha_path.write_text("deadbeef  gat_graph.pt\n")
        with pytest.raises(RuntimeError, match="sha256"):
            GraphBuilder.load(tmp_path)

    def test_node_features_shape(self, prescription_df, ddi_df):
        """노드 피처 shape = [num_nodes, 3]."""
        from scripts.features.graph_builder import GraphBuilder
        builder = GraphBuilder()
        data = builder.build(prescription_df, ddi_df)
        num_nodes = len(builder.drug_to_idx)
        assert data.x.shape == (num_nodes, 3)

    def test_mean_degree_warning(self, caplog):
        """평균 노드 차수 < 5 → WARNING 로그."""
        import logging
        from scripts.features.graph_builder import GraphBuilder
        # 2명이 각 2종 약물 처방 → mean_degree 낮음
        df = pd.DataFrame({
            "patient_id": ["P1","P1","P2","P2"],
            "drug_code":  ["D1","D2","D3","D4"],
            "prescription_date": ["2024-01-01"]*4,
        })
        ddi = pd.DataFrame({"drug_a": [], "drug_b": [], "severity": []})
        builder = GraphBuilder()
        with caplog.at_level(logging.WARNING):
            builder.build(df, ddi)
        assert any("평균 노드 차수" in r.message for r in caplog.records)
