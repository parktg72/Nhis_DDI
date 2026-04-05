# tests/test_integration/test_gat_deploy.py
"""GAT 배포 체인 통합 테스트."""
import hashlib
import json
import pickle
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

torch = pytest.importorskip("torch", reason="PyTorch 미설치")
pytest.importorskip("torch_geometric", reason="PyG 미설치")

import numpy as np
import pandas as pd
from scripts.train.gat_dataset import GATDataset
from scripts.train.gat_trainer import GATTrainer
from scripts.features.graph_builder import GraphBuilder


# ── 공통 픽스처 ──────────────────────────────────────────────────────────────

@pytest.fixture
def small_prescription_df():
    rows = []
    for i in range(20):
        pid = f"P{i:03d}"
        for d in [f"D0{i%5+1}", f"D0{(i+1)%5+1}"]:
            rows.append({"patient_id": pid, "drug_code": d,
                          "prescription_date": "2024-01-01"})
    return pd.DataFrame(rows)

@pytest.fixture
def small_ddi_df():
    return pd.DataFrame({
        "drug_a":   ["D01","D02"],
        "drug_b":   ["D02","D03"],
        "severity": ["contraindicated","major"],
    })

@pytest.fixture
def trained_trainer(small_prescription_df, small_ddi_df, tmp_path):
    ds = GATDataset(prescription_df=small_prescription_df, ddi_df=small_ddi_df)
    trainer = GATTrainer(
        params={"hidden_dim":8,"heads":1,"out_dim":4,"epochs":2,"lr":0.01,"random_state":42},
        config=None, model_dir=tmp_path,
    )
    trainer.fit(ds)
    trainer.save(tmp_path / "gat_model.pt")
    return trainer, tmp_path


# ── 배포 체인 테스트 ──────────────────────────────────────────────────────────

class TestGATDeployChain:
    def test_gat_model_sha256_missing_raises(self, tmp_path):
        """gat_model.pt.sha256 누락 → load_gat() RuntimeError."""
        # gat_model.pt 생성 (sha256 없음)
        (tmp_path / "gat_model.pt").write_bytes(b"dummy")
        with pytest.raises(RuntimeError):
            GATTrainer.load_gat(tmp_path / "gat_model.pt")

    def test_gat_graph_sha256_mismatch_raises(self, trained_trainer):
        """gat_graph.pt sha256 조작 → load_gat() RuntimeError."""
        _, model_dir = trained_trainer
        sha_path = model_dir / "gat_graph.pt.sha256"
        sha_path.write_text("deadbeef  gat_graph.pt\n")
        with pytest.raises(RuntimeError, match="sha256"):
            GATTrainer.load_gat(model_dir / "gat_model.pt")

    def test_gat_model_sha256_mismatch_raises(self, trained_trainer):
        """gat_model.pt sha256 조작 → load_gat() RuntimeError."""
        _, model_dir = trained_trainer
        sha_path = model_dir / "gat_model.pt.sha256"
        sha_path.write_text("deadbeef  gat_model.pt\n")
        with pytest.raises(RuntimeError, match="sha256"):
            GATTrainer.load_gat(model_dir / "gat_model.pt")

    def test_save_creates_all_gat_artifacts(self, trained_trainer):
        """save() → 5개 아티팩트 모두 존재."""
        _, model_dir = trained_trainer
        for filename in [
            "gat_model.pt",
            "gat_model.pt.sha256",
            "gat_graph.pt",
            "gat_graph.pt.sha256",
            "gat_graph_meta.json",
        ]:
            assert (model_dir / filename).exists(), f"누락: {filename}"

    def test_meta_json_fields(self, trained_trainer):
        """gat_graph_meta.json에 필수 필드 존재."""
        _, model_dir = trained_trainer
        meta = json.loads((model_dir / "gat_graph_meta.json").read_text())
        for field in ["built_at", "num_nodes", "num_edges", "feature_dim"]:
            assert field in meta, f"메타 필드 누락: {field}"

    def test_load_roundtrip(self, trained_trainer):
        """save → load_gat → _trained=True."""
        _, model_dir = trained_trainer
        loaded = GATTrainer.load_gat(model_dir / "gat_model.pt")
        assert loaded._trained


# ── 미지 약물 앙상블 제외 테스트 ─────────────────────────────────────────────

class TestUnknownDrugExclusion:
    def test_unknown_drug_predict_pair_returns_none(self, trained_trainer):
        """미지 약물 → predict_pair_proba() None."""
        trainer, _ = trained_trainer
        result = trainer.predict_pair_proba("UNKNOWN_DRUG_XYZ", "D01")
        assert result is None

    def test_ensemble_excludes_gat_for_unknown(self):
        """EnsembleTrainer3Way: 미지 약물 → w_gat=0, 나머지 정규화."""
        from scripts.train.trainer import EnsembleTrainer3Way

        class FakeXGB:
            def predict_proba(self, X): return np.array([0.7])
        class FakeLGB:
            def predict_proba(self, X): return np.array([0.5])
        class FakeGAT:
            def predict_pair_proba(self, a, b): return None

        ens = EnsembleTrainer3Way.__new__(EnsembleTrainer3Way)
        ens._xgb = FakeXGB()
        ens._lgb = FakeLGB()
        ens._gat = FakeGAT()
        ens.weights = (0.3, 0.3, 0.4)
        ens._trained = True

        X = np.zeros((1, 3))
        result = ens.predict_proba_with_gat(X, [("UNKNOWN", "D01")])
        # w_gat=0 → (0.3*0.7 + 0.3*0.5) / 0.6 = 0.6
        assert abs(result[0] - 0.6) < 1e-5

    def test_weights_renormalize_sum_to_one(self):
        """미지 약물 제외 후 효과적 가중치 합 = 1.0."""
        from scripts.train.trainer import EnsembleTrainer3Way

        class FakeXGB:
            def predict_proba(self, X): return np.array([0.5])
        class FakeLGB:
            def predict_proba(self, X): return np.array([0.5])
        class FakeGAT:
            def predict_pair_proba(self, a, b): return None

        ens = EnsembleTrainer3Way.__new__(EnsembleTrainer3Way)
        ens._xgb = FakeXGB()
        ens._lgb = FakeLGB()
        ens._gat = FakeGAT()
        ens.weights = (0.4, 0.4, 0.2)
        ens._trained = True

        X = np.zeros((1, 3))
        # p_xgb=0.5, p_lgb=0.5, GAT 제외 → (0.4*0.5 + 0.4*0.5) / 0.8 = 0.5
        result = ens.predict_proba_with_gat(X, [("UNKNOWN", "D01")])
        assert abs(result[0] - 0.5) < 1e-5


# ── Path Traversal 테스트 ──────────────────────────────────────────────────────

class TestPathTraversal:
    def test_graph_builder_load_rejects_tampered_path(self, tmp_path):
        """sha256이 맞더라도 dummy 내용 파일 → torch.load 에서 실패."""
        content = b"dummy"
        sha = hashlib.sha256(content).hexdigest()
        (tmp_path / "gat_graph.pt").write_bytes(content)
        (tmp_path / "gat_graph.pt.sha256").write_text(f"{sha}  gat_graph.pt\n")
        # sha256은 통과하지만 torch.load가 실패해야 함
        with pytest.raises(Exception):
            GraphBuilder.load(tmp_path)
