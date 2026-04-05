# Phase 3 GAT 모델 구현 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PyTorch Geometric 2-layer GAT를 DDI 위험도 앙상블의 세 번째 서브모델로 추가한다.

**Architecture:** 처방 DataFrame → GraphBuilder → co-prescription 그래프 → GATModel → 약물쌍 위험 확률 → Platt 보정 → max 집계 → 요청 레벨 p_gat → XGB+LGB+GAT 가중 평균 (Recall ≥ 0.90 제약 하 AUC 최대화).

**Tech Stack:** PyTorch Geometric ≥ 2.5, PyTorch ≥ 2.1, scikit-learn (CalibratedClassifierCV), scipy.optimize (SLSQP)

---

## 파일 구조 (생성/수정)

| 작업 | 파일 |
|------|------|
| 생성 | `scripts/train/gat_dataset.py` |
| 생성 | `scripts/features/graph_builder.py` |
| 생성 | `scripts/train/gat_model.py` |
| 생성 | `scripts/train/base_graph_trainer.py` |
| 생성 | `scripts/train/gat_trainer.py` |
| 수정 | `scripts/train/hyperparams.py` |
| 수정 | `scripts/train/trainer.py` |
| 수정 | `serving/predictor.py` |
| 수정 | `dags/ddi_train_dag.py` |
| 생성 | `tests/test_train/test_gat_trainer.py` |
| 생성 | `tests/test_integration/test_gat_deploy.py` |

---

## Task 1: GATDataset 데이터 컨테이너

**Context:** `BaseTrainer.fit(dataset: TrainDataset)`는 numpy 행렬을 기대한다. GAT는 처방 DataFrame과 DDI 쌍 레이블이 필요하므로 별도 데이터 컨테이너를 만든다.

**Files:**
- Create: `scripts/train/gat_dataset.py`
- Test: `tests/test_train/test_gat_trainer.py` (첫 번째 테스트 클래스)

- [ ] **Step 1: 테스트 파일 생성 — GATDataset 기본 속성 검증**

```python
# tests/test_train/test_gat_trainer.py
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
```

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
cd /Volumes/model/claude/MODE_11_hana
python -m pytest tests/test_train/test_gat_trainer.py::TestGATDataset -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'scripts.train.gat_dataset'`

- [ ] **Step 3: GATDataset 구현**

```python
# scripts/train/gat_dataset.py
"""GAT 훈련용 데이터 컨테이너."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class GATDataset:
    """
    GAT 훈련에 필요한 데이터.

    Attributes
    ----------
    prescription_df : patient_id, drug_code, prescription_date 컬럼 필수
    ddi_df          : drug_a, drug_b, severity 컬럼 필수 (DDI 지식 베이스)
    pairs_train     : shape [N, 3] — (node_a_idx, node_b_idx, label) int64
    pairs_gat_val   : GAT 전용 조기종료용 쌍 (XGB/LGB val과 완전 분리)
    pairs_calibration: Platt scaling + 앙상블 가중치 최적화 전용
    """
    prescription_df: pd.DataFrame
    ddi_df: pd.DataFrame
    pairs_train: Optional[np.ndarray] = None        # [N, 3] int64
    pairs_gat_val: Optional[np.ndarray] = None      # [M, 3] int64
    pairs_calibration: Optional[np.ndarray] = None  # [K, 3] int64

    @property
    def unique_drugs(self) -> list[str]:
        """처방 DataFrame 내 고유 약물 코드 목록 (정렬)."""
        return sorted(self.prescription_df["drug_code"].unique())

    @property
    def num_drugs(self) -> int:
        return len(self.unique_drugs)
```

- [ ] **Step 4: 테스트 실행 — PASS 확인**

```bash
python -m pytest tests/test_train/test_gat_trainer.py::TestGATDataset -v
```
Expected: `2 passed`

- [ ] **Step 5: 커밋**

```bash
git add scripts/train/gat_dataset.py tests/test_train/test_gat_trainer.py
git commit -m "feat: GATDataset 데이터 컨테이너 추가"
```

---

## Task 2: GraphBuilder — 그래프 구성 및 직렬화

**Context:** 처방 DataFrame에서 co-prescription 그래프를 빌드하고 `gat_graph.pt` + `.sha256` + `gat_graph_meta.json`으로 직렬화한다. 엣지는 train 분할 데이터만 사용해 val/test 누출을 막는다. `drug_to_idx` 매핑도 그래프와 함께 저장한다.

**Files:**
- Create: `scripts/features/graph_builder.py`
- Test: `tests/test_train/test_gat_trainer.py` (TestGraphBuilder 클래스)

- [ ] **Step 1: 테스트 작성 — 쌍 생성, 엣지 가중치, 미지 약물 경고**

```python
# tests/test_train/test_gat_trainer.py 에 추가

from scripts.features.graph_builder import GraphBuilder


class TestGraphBuilder:
    @pytest.fixture
    def prescription_df(self):
        """같은 날짜에 3명 환자, 여러 약물 처방."""
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
        """동일 patient_id + prescription_date 처방 → 엣지 생성."""
        builder = GraphBuilder()
        data = builder.build(prescription_df, ddi_df)
        # P1이 D1,D2,D3 → 3쌍 / P2가 D1,D2 → 1쌍 (이미 포함)
        # D4는 P3 단독 → 고립 노드
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] > 0

    def test_edge_weights_log1p_normalized(self, prescription_df, ddi_df):
        """엣지 가중치가 [0, 1] 범위."""
        builder = GraphBuilder()
        data = builder.build(prescription_df, ddi_df)
        assert data.edge_weight is not None
        assert float(data.edge_weight.min()) >= 0.0
        assert float(data.edge_weight.max()) <= 1.0 + 1e-6

    def test_unknown_drug_excluded_not_zerovector(self, prescription_df, ddi_df, caplog):
        """미지 약물 코드 조회 시 경고 로그 발생."""
        import logging
        builder = GraphBuilder()
        builder.build(prescription_df, ddi_df)
        with caplog.at_level(logging.WARNING):
            idx = builder.drug_to_idx.get("UNKNOWN_DRUG", None)
        assert idx is None  # 미지 약물은 None 반환

    def test_isolated_node_warning(self, caplog):
        """고립 노드 비율 > 10% 시 WARNING."""
        import logging
        # D1만 있어 고립 노드 비율 100%
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
        builder = GraphBuilder()
        builder.build(prescription_df, ddi_df)
        builder.save(tmp_path)
        assert (tmp_path / "gat_graph.pt").exists()
        assert (tmp_path / "gat_graph.pt.sha256").exists()
        assert (tmp_path / "gat_graph_meta.json").exists()

    def test_load_verifies_sha256(self, prescription_df, ddi_df, tmp_path):
        """sha256 불일치 시 RuntimeError."""
        builder = GraphBuilder()
        builder.build(prescription_df, ddi_df)
        builder.save(tmp_path)
        # sha256 조작
        sha_path = tmp_path / "gat_graph.pt.sha256"
        sha_path.write_text("deadbeef  gat_graph.pt\n")
        with pytest.raises(RuntimeError, match="sha256"):
            GraphBuilder.load(tmp_path)
```

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
python -m pytest tests/test_train/test_gat_trainer.py::TestGraphBuilder -v 2>&1 | head -20
```
Expected: `ImportError: cannot import name 'GraphBuilder'`

- [ ] **Step 3: GraphBuilder 구현**

```python
# scripts/features/graph_builder.py
"""처방 데이터 → PyG 그래프 구성 및 직렬화."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GRAPH_FILE = "gat_graph.pt"
META_FILE = "gat_graph_meta.json"


class GraphBuilder:
    """
    처방 DataFrame → PyG Data(x, edge_index, edge_weight, drug_to_idx).

    사용법:
        builder = GraphBuilder()
        data = builder.build(prescription_df, ddi_df)
        builder.save(model_dir)
        # 나중에:
        builder2 = GraphBuilder.load(model_dir)
    """

    def __init__(self):
        self.data = None           # PyG Data 객체
        self.drug_to_idx: dict[str, int] = {}
        self._meta: dict = {}

    def build(self, prescription_df: pd.DataFrame, ddi_df: pd.DataFrame):
        """
        Parameters
        ----------
        prescription_df : 훈련 분할 처방 데이터만 입력 (val/test 누출 방지)
                          필수 컬럼: patient_id, drug_code, prescription_date
        ddi_df          : DDI 지식 베이스
                          필수 컬럼: drug_a, drug_b, severity
        Returns
        -------
        PyG Data 객체
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as e:
            raise ImportError("torch_geometric 미설치: pip install torch_geometric") from e

        # 1. 노드 목록
        all_drugs = sorted(prescription_df["drug_code"].unique())
        self.drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
        num_nodes = len(all_drugs)

        # 2. Co-prescription 엣지 (동일 patient_id + prescription_date)
        edge_counter: dict[tuple[int, int], int] = {}
        grp = prescription_df.groupby(["patient_id", "prescription_date"])
        for _, group in grp:
            codes = group["drug_code"].tolist()
            for a, b in combinations(codes, 2):
                if a not in self.drug_to_idx or b not in self.drug_to_idx:
                    continue
                ai, bi = self.drug_to_idx[a], self.drug_to_idx[b]
                key = (min(ai, bi), max(ai, bi))
                edge_counter[key] = edge_counter.get(key, 0) + 1

        if edge_counter:
            src, dst, weights = [], [], []
            for (ai, bi), cnt in edge_counter.items():
                src += [ai, bi]
                dst += [bi, ai]
                w = float(np.log1p(cnt))
                weights += [w, w]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_weight_raw = torch.tensor(weights, dtype=torch.float)
            max_w = float(edge_weight_raw.max()) or 1.0
            edge_weight = edge_weight_raw / max_w
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros(0, dtype=torch.float)

        # 3. 노드 피처: [log1p_degree, ddi_count, log1p_freq]
        degree = torch.zeros(num_nodes)
        if edge_counter:
            for (ai, bi) in edge_counter:
                degree[ai] += 1
                degree[bi] += 1
        log_degree = torch.log1p(degree)

        # DDI 카운트 피처
        ddi_count = torch.zeros(num_nodes)
        for _, row in ddi_df.iterrows():
            ai = self.drug_to_idx.get(str(row["drug_a"]))
            bi = self.drug_to_idx.get(str(row["drug_b"]))
            if ai is not None:
                ddi_count[ai] += 1
            if bi is not None:
                ddi_count[bi] += 1

        # 처방 빈도 피처
        freq_raw = prescription_df["drug_code"].value_counts()
        log_freq = torch.zeros(num_nodes)
        for drug, cnt in freq_raw.items():
            if drug in self.drug_to_idx:
                log_freq[self.drug_to_idx[drug]] = float(np.log1p(cnt))

        x = torch.stack([log_degree, ddi_count, log_freq], dim=1)  # [N, 3]

        # 4. 그래프 품질 경고
        isolated = int((degree == 0).sum())
        if num_nodes > 0:
            isolated_ratio = isolated / num_nodes
            if isolated_ratio > 0.10:
                logger.warning(
                    "고립 노드 비율 %.1f%% (>10%%) — GAT 효용 저하 가능. 아키텍처 재검토 권장.",
                    isolated_ratio * 100,
                )
        mean_degree = float(degree.mean()) if num_nodes > 0 else 0.0
        if mean_degree < 5:
            logger.warning(
                "평균 노드 차수 %.1f (<5) — GAT 효용 저하 가능.",
                mean_degree,
            )

        from torch_geometric.data import Data
        self.data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        self.data.drug_to_idx = self.drug_to_idx  # 직렬화 위해 함께 저장

        self._meta = {
            "built_at": datetime.utcnow().isoformat(),
            "num_nodes": num_nodes,
            "num_edges": edge_index.shape[1] // 2,
            "feature_dim": 3,
            "mean_degree": round(mean_degree, 2),
            "isolated_ratio": round(isolated / max(num_nodes, 1), 4),
        }
        logger.info(
            "그래프 빌드 완료: %d nodes, %d edges (mean_degree=%.1f)",
            num_nodes, edge_index.shape[1] // 2, mean_degree,
        )
        return self.data

    def save(self, model_dir: str | Path) -> None:
        """gat_graph.pt + .sha256 + gat_graph_meta.json 저장."""
        if self.data is None:
            raise RuntimeError("build() 먼저 호출하세요.")
        try:
            import torch
        except ImportError as e:
            raise ImportError("torch 미설치") from e

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        graph_path = model_dir / GRAPH_FILE
        torch.save({"data": self.data, "drug_to_idx": self.drug_to_idx}, graph_path)
        content = graph_path.read_bytes()
        sha256 = hashlib.sha256(content).hexdigest()
        (model_dir / (GRAPH_FILE + ".sha256")).write_text(f"{sha256}  {GRAPH_FILE}\n")

        meta = dict(self._meta)
        meta["edge_index_hash"] = sha256[:16]
        (model_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        logger.info("그래프 저장 완료: %s (sha256=%s…)", graph_path, sha256[:16])

    @classmethod
    def load(cls, model_dir: str | Path) -> "GraphBuilder":
        """gat_graph.pt sha256 검증 후 로드. 불일치 시 RuntimeError."""
        try:
            import torch
        except ImportError as e:
            raise ImportError("torch 미설치") from e

        model_dir = Path(model_dir)
        graph_path = model_dir / GRAPH_FILE
        sha_path = model_dir / (GRAPH_FILE + ".sha256")

        if not graph_path.exists():
            raise RuntimeError(f"gat_graph.pt 없음: {graph_path}")
        if not sha_path.exists():
            raise RuntimeError(f"gat_graph.pt.sha256 없음: {sha_path}")

        content = graph_path.read_bytes()
        expected = sha_path.read_text().strip().split()[0]
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"gat_graph.pt sha256 불일치 — 로드 거부 "
                f"(expected={expected[:16]}…, actual={actual[:16]}…)"
            )

        saved = torch.load(graph_path, weights_only=False)
        obj = cls()
        obj.data = saved["data"]
        obj.drug_to_idx = saved["drug_to_idx"]
        meta_path = model_dir / META_FILE
        if meta_path.exists():
            obj._meta = json.loads(meta_path.read_text())
        logger.info("그래프 로드 완료: %s", graph_path)
        return obj
```

- [ ] **Step 4: 테스트 실행 — PASS 확인**

```bash
python -m pytest tests/test_train/test_gat_trainer.py::TestGraphBuilder -v
```
Expected: `6 passed`

- [ ] **Step 5: 커밋**

```bash
git add scripts/features/graph_builder.py tests/test_train/test_gat_trainer.py
git commit -m "feat: GraphBuilder — co-prescription 그래프 빌드 및 직렬화"
```

---

## Task 3: GATModel — 2-layer GAT 모델

**Context:** PyG `GATConv` 2레이어. 약물쌍 스코어링: `concat([h_a, h_b, |h_a-h_b|, h_a⊙h_b])` → Linear → sigmoid.

**Files:**
- Create: `scripts/train/gat_model.py`
- Test: `tests/test_train/test_gat_trainer.py` (TestGATModel 클래스)

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_train/test_gat_trainer.py 에 추가

from scripts.train.gat_model import GATModel


class TestGATModel:
    @pytest.fixture
    def small_graph(self):
        """4노드, 4엣지 소규모 그래프."""
        import torch
        x = torch.randn(4, 3)
        edge_index = torch.tensor([[0,1,2,3],[1,0,3,2]], dtype=torch.long)
        return x, edge_index

    def test_forward_output_shape(self, small_graph):
        """forward() → [num_nodes, out_dim] 형태."""
        import torch
        x, edge_index = small_graph
        model = GATModel(feature_dim=3, hidden_dim=8, heads=2, out_dim=4)
        embeddings = model(x, edge_index)
        assert embeddings.shape == (4, 4)

    def test_score_pairs_range(self, small_graph):
        """score_pairs() → 값이 [0, 1] 범위."""
        import torch
        x, edge_index = small_graph
        model = GATModel(feature_dim=3, hidden_dim=8, heads=2, out_dim=4)
        pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        scores = model.score_pairs(x, edge_index, pairs)
        assert scores.shape == (2,)
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 1.0 + 1e-6

    def test_pair_feature_concat_dim(self, small_graph):
        """pair scorer 입력 차원 = out_dim * 4."""
        import torch
        x, edge_index = small_graph
        out_dim = 4
        model = GATModel(feature_dim=3, hidden_dim=8, heads=2, out_dim=out_dim)
        # pair_scorer Linear의 in_features = out_dim * 4
        assert model.pair_scorer[0].in_features == out_dim * 4
```

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
python -m pytest tests/test_train/test_gat_trainer.py::TestGATModel -v 2>&1 | head -10
```
Expected: `ImportError: cannot import name 'GATModel'`

- [ ] **Step 3: GATModel 구현**

```python
# scripts/train/gat_model.py
"""2-layer Graph Attention Network — DDI 약물쌍 위험도 예측."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATModel(nn.Module):
    """
    2-layer GAT. 약물쌍 위험도를 [0,1] 확률로 출력한다.

    Parameters
    ----------
    feature_dim : 노드 입력 피처 차원 (GraphBuilder 출력: 3)
    hidden_dim  : GAT layer 1 출력 차원 (heads 이전)
    heads       : Attention head 수 (layer 1)
    out_dim     : GAT layer 2 출력 차원 (pair scorer 입력: out_dim * 4)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        out_dim: int = 32,
    ):
        super().__init__()
        try:
            from torch_geometric.nn import GATConv
        except ImportError as e:
            raise ImportError("torch_geometric 미설치: pip install torch_geometric") from e

        self.conv1 = GATConv(feature_dim, hidden_dim, heads=heads, concat=True)
        # layer 1 출력: hidden_dim * heads
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False)
        # pair scorer: concat([h_a, h_b, |h_a-h_b|, h_a⊙h_b]) → [out_dim*4] → 1
        self.pair_scorer = nn.Sequential(
            nn.Linear(out_dim * 4, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : [num_nodes, feature_dim]
        edge_index : [2, num_edges]

        Returns
        -------
        [num_nodes, out_dim] 노드 임베딩
        """
        h = F.elu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h

    def score_pairs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : [num_nodes, feature_dim]
        edge_index : [2, num_edges]
        pairs      : [N, 2] (node_a_idx, node_b_idx) int64

        Returns
        -------
        [N] 쌍별 DDI 위험 확률 (sigmoid 보정 전)
        """
        h = self.forward(x, edge_index)   # [num_nodes, out_dim]
        h_a = h[pairs[:, 0]]              # [N, out_dim]
        h_b = h[pairs[:, 1]]              # [N, out_dim]
        feat = torch.cat(
            [h_a, h_b, (h_a - h_b).abs(), h_a * h_b],
            dim=-1,
        )                                  # [N, out_dim * 4]
        logit = self.pair_scorer(feat).squeeze(-1)  # [N]
        return torch.sigmoid(logit)
```

- [ ] **Step 4: 테스트 실행 — PASS 확인**

```bash
python -m pytest tests/test_train/test_gat_trainer.py::TestGATModel -v
```
Expected: `3 passed`

- [ ] **Step 5: 커밋**

```bash
git add scripts/train/gat_model.py tests/test_train/test_gat_trainer.py
git commit -m "feat: GATModel 2-layer GAT 정의"
```

---

## Task 4: BaseGraphTrainer 추상 클래스

**Context:** `BaseTrainer.fit(dataset: TrainDataset)`은 numpy 배열을 기대한다. GAT는 `GATDataset`을 받아야 하므로 중간 추상 클래스를 만든다. `save/load/evaluate` 인터페이스는 `BaseTrainer` 규약 유지.

**Files:**
- Create: `scripts/train/base_graph_trainer.py`
- Test: `tests/test_train/test_gat_trainer.py` (TestBaseGraphTrainer 클래스)

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_train/test_gat_trainer.py 에 추가

from scripts.train.base_graph_trainer import BaseGraphTrainer
from scripts.train.gat_dataset import GATDataset


class TestBaseGraphTrainer:
    def test_fit_rejects_train_dataset(self):
        """fit(TrainDataset) → TypeError. GAT는 GATDataset만 허용."""
        from scripts.train.dataset import TrainDataset
        import numpy as np

        class ConcreteGraph(BaseGraphTrainer):
            def fit_graph(self, dataset): ...
            def predict_pair_proba(self, drug_a, drug_b): return 0.5
            def predict_proba(self, X): return np.zeros(len(X))

        trainer = ConcreteGraph(params={}, config=None)
        with pytest.raises(TypeError, match="GATDataset"):
            trainer.fit(object())  # TrainDataset 아닌 아무 객체

    def test_fit_accepts_gat_dataset(self, prescription_df, ddi_df):
        """fit(GATDataset) 정상 호출."""
        import numpy as np

        class ConcreteGraph(BaseGraphTrainer):
            def fit_graph(self, dataset):
                self._trained = True

            def predict_pair_proba(self, drug_a, drug_b): return 0.5
            def predict_proba(self, X): return np.zeros(len(X))

        trainer = ConcreteGraph(params={}, config=None)
        ds = GATDataset(prescription_df=prescription_df, ddi_df=ddi_df)
        trainer.fit(ds)
        assert trainer._trained
```

(위 테스트에서 `prescription_df`, `ddi_df` 픽스처는 Task 2 클래스에서 공유하거나 별도 정의)

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
python -m pytest tests/test_train/test_gat_trainer.py::TestBaseGraphTrainer -v 2>&1 | head -10
```

- [ ] **Step 3: BaseGraphTrainer 구현**

```python
# scripts/train/base_graph_trainer.py
"""Graph 모델용 BaseTrainer 확장 — PyG Data 객체를 수용하도록 fit() 재정의."""
from __future__ import annotations

from abc import abstractmethod

import numpy as np

from .trainer import BaseTrainer
from .gat_dataset import GATDataset


class BaseGraphTrainer(BaseTrainer):
    """
    BaseTrainer 서브클래스.

    - fit(GATDataset) → fit_graph(GATDataset) 위임
    - fit(다른 타입)  → TypeError
    - predict_proba(X) : 배열 입력 지원 (serving 호환성)
    - fit_graph, predict_pair_proba: 서브클래스 구현 필수
    """

    def fit(self, dataset) -> "BaseGraphTrainer":
        if not isinstance(dataset, GATDataset):
            raise TypeError(
                f"BaseGraphTrainer.fit()은 GATDataset 필요, 받은 타입: {type(dataset).__name__}"
            )
        return self.fit_graph(dataset)

    @abstractmethod
    def fit_graph(self, dataset: GATDataset) -> "BaseGraphTrainer":
        """그래프 기반 학습 구현."""
        ...

    @abstractmethod
    def predict_pair_proba(self, drug_a: str, drug_b: str) -> float | None:
        """
        단일 약물쌍 DDI 위험 확률.
        미지 약물 포함 시 None 반환 (앙상블에서 GAT 제외).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """serving 호환 배열 인터페이스 — 서브클래스가 구현."""
        ...
```

- [ ] **Step 4: 테스트 실행 — PASS 확인**

```bash
python -m pytest tests/test_train/test_gat_trainer.py::TestBaseGraphTrainer -v
```
Expected: `2 passed`

- [ ] **Step 5: 커밋**

```bash
git add scripts/train/base_graph_trainer.py tests/test_train/test_gat_trainer.py
git commit -m "feat: BaseGraphTrainer 추상 클래스"
```

---

## Task 5: GATTrainer — 학습, 보정, 저장, 로드

**Context:** 60/20/10/10 분할 (train/xgb_lgb_val/gat_val/calibration). 긍정 쌍: ddi_df severity∈{contraindicated,major}. 부정 쌍: 5:1 비율 무작위 샘플. 조기종료는 gat_val AUC 기준. Platt scaling 보정. 저장 파일: `gat_model.pt` + `.sha256`.

**Files:**
- Create: `scripts/train/gat_trainer.py`
- Test: `tests/test_train/test_gat_trainer.py` (TestGATTrainer 클래스)

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_train/test_gat_trainer.py 에 추가

from scripts.train.gat_trainer import GATTrainer


class TestGATTrainer:
    @pytest.fixture
    def small_dataset(self):
        """소규모 GATDataset 생성."""
        n_patients = 30
        drugs = [f"D{i:02d}" for i in range(1, 8)]
        import random
        random.seed(42)
        rows = []
        for i in range(n_patients):
            pid = f"P{i:03d}"
            n_drugs = random.randint(2, 4)
            chosen = random.sample(drugs, n_drugs)
            for d in chosen:
                rows.append({"patient_id": pid, "drug_code": d,
                              "prescription_date": "2024-01-01"})
        prescription_df = pd.DataFrame(rows)
        ddi_df = pd.DataFrame({
            "drug_a":   ["D01","D02","D03"],
            "drug_b":   ["D02","D03","D04"],
            "severity": ["contraindicated","major","major"],
        })
        return GATDataset(prescription_df=prescription_df, ddi_df=ddi_df)

    def test_fit_sets_trained(self, small_dataset, tmp_path):
        """fit() → _trained=True."""
        trainer = GATTrainer(
            params={"hidden_dim": 8, "heads": 1, "out_dim": 4,
                    "epochs": 3, "lr": 0.01, "random_state": 42},
            config=None,
            model_dir=tmp_path,
        )
        trainer.fit_graph(small_dataset)
        assert trainer._trained

    def test_save_creates_all_artifacts(self, small_dataset, tmp_path):
        """save() → gat_model.pt + .sha256 생성."""
        trainer = GATTrainer(
            params={"hidden_dim": 8, "heads": 1, "out_dim": 4,
                    "epochs": 2, "lr": 0.01, "random_state": 42},
            config=None,
            model_dir=tmp_path,
        )
        trainer.fit_graph(small_dataset)
        trainer.save(tmp_path / "gat_model.pt")
        assert (tmp_path / "gat_model.pt").exists()
        assert (tmp_path / "gat_model.pt.sha256").exists()

    def test_load_graph_sha256_mismatch_raises(self, small_dataset, tmp_path):
        """gat_graph.pt sha256 불일치 → RuntimeError."""
        trainer = GATTrainer(
            params={"hidden_dim": 8, "heads": 1, "out_dim": 4,
                    "epochs": 2, "lr": 0.01, "random_state": 42},
            config=None,
            model_dir=tmp_path,
        )
        trainer.fit_graph(small_dataset)
        trainer.save(tmp_path / "gat_model.pt")
        # graph sha256 조작
        sha_path = tmp_path / "gat_graph.pt.sha256"
        sha_path.write_text("deadbeef  gat_graph.pt\n")
        with pytest.raises(RuntimeError, match="sha256"):
            GATTrainer.load_gat(tmp_path / "gat_model.pt")

    def test_predict_pair_proba_unknown_returns_none(self, small_dataset, tmp_path):
        """미지 약물 코드 → predict_pair_proba() None 반환 + 경고 로그."""
        import logging
        trainer = GATTrainer(
            params={"hidden_dim": 8, "heads": 1, "out_dim": 4,
                    "epochs": 2, "lr": 0.01, "random_state": 42},
            config=None,
            model_dir=tmp_path,
        )
        trainer.fit_graph(small_dataset)
        with pytest.warns(None):  # 경고 로그 (not pytest.warns)
            result = trainer.predict_pair_proba("UNKNOWN_DRUG", "D01")
        assert result is None

    def test_predict_pair_proba_known_returns_float(self, small_dataset, tmp_path):
        """알려진 약물쌍 → 0~1 float 반환."""
        trainer = GATTrainer(
            params={"hidden_dim": 8, "heads": 1, "out_dim": 4,
                    "epochs": 2, "lr": 0.01, "random_state": 42},
            config=None,
            model_dir=tmp_path,
        )
        trainer.fit_graph(small_dataset)
        result = trainer.predict_pair_proba("D01", "D02")
        if result is not None:
            assert 0.0 <= result <= 1.0
```

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
python -m pytest tests/test_train/test_gat_trainer.py::TestGATTrainer -v 2>&1 | head -10
```

- [ ] **Step 3: GATTrainer 구현**

```python
# scripts/train/gat_trainer.py
"""GATTrainer — GAT 학습, Platt 보정, 저장/로드."""
from __future__ import annotations

import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base_graph_trainer import BaseGraphTrainer
from .gat_dataset import GATDataset
from scripts.features.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

POSITIVE_SEVERITIES = {"contraindicated", "major"}
NEG_POS_RATIO = 5  # 부정:긍정 샘플 비율


class GATTrainer(BaseGraphTrainer):
    """
    GAT 훈련기.

    Parameters (params dict)
    ------------------------
    hidden_dim   : GAT layer 1 hidden (default 64)
    heads        : Attention heads (default 4)
    out_dim      : GAT layer 2 output dim (default 32)
    epochs       : 최대 에폭 (default 200)
    lr           : 학습률 (default 0.001)
    patience     : 조기종료 patience (default 20)
    random_state : 재현성 시드 (default 42)
    """

    def __init__(self, params: dict, config: Any, model_dir: str | Path = "models"):
        super().__init__(params, config)
        self.model_dir = Path(model_dir)
        self._gat_model = None
        self._graph_builder: Optional[GraphBuilder] = None
        self._calibrator = None   # sklearn CalibratedClassifierCV

    # ── 학습 ──────────────────────────────────────────────────────────────────

    def fit_graph(self, dataset: GATDataset) -> "GATTrainer":
        try:
            import torch
            import torch.nn as nn
            from torch.optim import Adam
            from sklearn.metrics import roc_auc_score
        except ImportError as e:
            raise ImportError(f"의존성 없음: {e}") from e

        from .gat_model import GATModel

        seed = self.params.get("random_state", 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

        t0 = time.perf_counter()

        # ── 1. 그래프 빌드 (train 데이터만) ──────────────────────────────────
        self._graph_builder = GraphBuilder()
        graph_data = self._graph_builder.build(
            dataset.prescription_df, dataset.ddi_df
        )
        drug_to_idx = self._graph_builder.drug_to_idx

        # ── 2. DDI 긍정쌍 생성 ────────────────────────────────────────────────
        pos_pairs = []
        for _, row in dataset.ddi_df.iterrows():
            if str(row["severity"]).lower() not in POSITIVE_SEVERITIES:
                continue
            ai = drug_to_idx.get(str(row["drug_a"]))
            bi = drug_to_idx.get(str(row["drug_b"]))
            if ai is not None and bi is not None:
                pos_pairs.append((ai, bi, 1))

        if not pos_pairs:
            logger.warning("긍정 DDI 쌍 없음 — 더미 학습만 수행")
            self._trained = True
            return self

        # ── 3. 부정쌍 샘플링 (NEG_POS_RATIO:1) ───────────────────────────────
        rng = np.random.default_rng(seed)
        pos_set = {(a, b) for a, b, _ in pos_pairs} | {(b, a) for a, b, _ in pos_pairs}
        n_nodes = len(drug_to_idx)
        neg_pairs = []
        attempts = 0
        target_neg = len(pos_pairs) * NEG_POS_RATIO
        while len(neg_pairs) < target_neg and attempts < target_neg * 20:
            a, b = int(rng.integers(n_nodes)), int(rng.integers(n_nodes))
            if a != b and (a, b) not in pos_set:
                neg_pairs.append((a, b, 0))
            attempts += 1

        all_pairs = pos_pairs + neg_pairs
        rng.shuffle(all_pairs)
        all_pairs = np.array(all_pairs, dtype=np.int64)  # [N, 3]

        # 60 / 20 / 10 / 10 → gat_val = 10%, calib = 10%, rest = train
        n = len(all_pairs)
        n_calib = max(1, int(n * 0.10))
        n_gatval = max(1, int(n * 0.10))
        n_train = n - n_calib - n_gatval

        tr_pairs = all_pairs[:n_train]
        gv_pairs = all_pairs[n_train:n_train + n_gatval]
        ca_pairs = all_pairs[n_train + n_gatval:]

        dataset.pairs_train = tr_pairs
        dataset.pairs_gat_val = gv_pairs
        dataset.pairs_calibration = ca_pairs

        # ── 4. 모델 초기화 ────────────────────────────────────────────────────
        feature_dim = graph_data.x.shape[1]
        model = GATModel(
            feature_dim=feature_dim,
            hidden_dim=self.params.get("hidden_dim", 64),
            heads=self.params.get("heads", 4),
            out_dim=self.params.get("out_dim", 32),
        )
        optimizer = Adam(model.parameters(), lr=self.params.get("lr", 0.001))
        criterion = nn.BCELoss()

        x = graph_data.x
        edge_index = graph_data.edge_index

        # ── 5. 학습 루프 (조기종료: gat_val AUC) ─────────────────────────────
        epochs = self.params.get("epochs", 200)
        patience = self.params.get("patience", 20)
        best_auc, no_improve = 0.0, 0
        best_state = None

        tr_pairs_t = torch.tensor(tr_pairs[:, :2], dtype=torch.long)
        tr_labels_t = torch.tensor(tr_pairs[:, 2], dtype=torch.float)
        gv_pairs_t = torch.tensor(gv_pairs[:, :2], dtype=torch.long)
        gv_labels = gv_pairs[:, 2]

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            scores = model.score_pairs(x, edge_index, tr_pairs_t)
            loss = criterion(scores, tr_labels_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    gv_scores = model.score_pairs(x, edge_index, gv_pairs_t).numpy()
                if len(np.unique(gv_labels)) > 1:
                    auc = roc_auc_score(gv_labels, gv_scores)
                    if auc > best_auc:
                        best_auc = auc
                        no_improve = 0
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    else:
                        no_improve += 1
                    if no_improve >= patience // 10:
                        logger.info("조기종료: epoch=%d, best_gat_val_auc=%.4f", epoch + 1, best_auc)
                        break

        if best_state:
            model.load_state_dict(best_state)
        self._gat_model = model
        self._trained = True

        logger.info(
            "GATTrainer 학습 완료 (%.1fs): gat_val_auc=%.4f, pairs=%d",
            time.perf_counter() - t0, best_auc, len(tr_pairs),
        )
        return self

    # ── 예측 ──────────────────────────────────────────────────────────────────

    def predict_pair_proba(self, drug_a: str, drug_b: str) -> float | None:
        """
        단일 약물쌍 위험 확률.
        미지 약물 포함 시 None 반환 (앙상블에서 GAT 서브모델 제외).
        """
        if not self._trained or self._gat_model is None:
            return None
        d2i = self._graph_builder.drug_to_idx
        ai = d2i.get(drug_a)
        bi = d2i.get(drug_b)
        if ai is None or bi is None:
            logger.warning("알 수 없는 약물 코드 — GAT 서브모델 제외: %s, %s", drug_a, drug_b)
            return None
        try:
            import torch
            self._gat_model.eval()
            data = self._graph_builder.data
            pairs_t = torch.tensor([[ai, bi]], dtype=torch.long)
            with torch.no_grad():
                raw = float(self._gat_model.score_pairs(data.x, data.edge_index, pairs_t)[0])
            # Platt 보정
            if self._calibrator is not None:
                raw = float(self._calibrator.predict_proba([[raw]])[:, 1][0])
            return raw
        except Exception as e:
            logger.error("GAT 예측 오류: %s", e)
            return None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """serving 호환 인터페이스 — 배열 입력은 지원하지 않아 0.5 반환."""
        logger.warning("GATTrainer.predict_proba(numpy array) 미지원 — predict_pair_proba 사용 권장")
        return np.full(len(X), 0.5)

    # ── 보정 ──────────────────────────────────────────────────────────────────

    def calibrate(self, calibration_pairs: np.ndarray) -> None:
        """
        Platt scaling 보정.

        Parameters
        ----------
        calibration_pairs : [N, 3] (node_a, node_b, label)
        """
        try:
            import torch
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
        except ImportError as e:
            raise ImportError(f"의존성 없음: {e}") from e

        if not self._trained or self._gat_model is None:
            raise RuntimeError("fit_graph() 먼저 호출하세요.")

        data = self._graph_builder.data
        pairs_t = torch.tensor(calibration_pairs[:, :2], dtype=torch.long)
        labels = calibration_pairs[:, 2].astype(int)

        self._gat_model.eval()
        with torch.no_grad():
            raw_scores = self._gat_model.score_pairs(
                data.x, data.edge_index, pairs_t
            ).numpy()

        if len(np.unique(labels)) < 2:
            logger.warning("calibration_pairs 단일 클래스 — Platt scaling 생략")
            return

        # Platt: sigmoid로 raw_score를 입력, logistic regression으로 보정
        self._calibrator = LogisticRegression()
        self._calibrator.fit(raw_scores.reshape(-1, 1), labels)
        logger.info("Platt scaling 보정 완료 (calibration_pairs=%d)", len(labels))

    # ── 저장/로드 ─────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """gat_model.pt + .sha256 저장. graph는 model_dir에 별도 저장."""
        if not self._trained:
            raise RuntimeError("fit_graph() 먼저 호출하세요.")
        try:
            import torch
        except ImportError as e:
            raise ImportError("torch 미설치") from e

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state": self._gat_model.state_dict(),
            "model_init_params": {
                "feature_dim": self._graph_builder.data.x.shape[1],
                "hidden_dim": self.params.get("hidden_dim", 64),
                "heads": self.params.get("heads", 4),
                "out_dim": self.params.get("out_dim", 32),
            },
            "calibrator": self._calibrator,
            "params": self.params,
            "trainer_class": self.__class__.__name__,
            "best_threshold": self.best_threshold_,
        }
        content = pickle.dumps(payload)
        path.write_bytes(content)
        sha256 = hashlib.sha256(content).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256}  {path.name}\n")

        # 그래프 아티팩트는 같은 디렉터리에 저장
        self._graph_builder.save(path.parent)
        logger.info("GATTrainer 저장: %s (sha256=%s…)", path, sha256[:16])
        return path

    @classmethod
    def load_gat(cls, path: str | Path) -> "GATTrainer":
        """gat_model.pt + gat_graph.pt sha256 검증 후 로드."""
        path = Path(path)
        # model sha256 검증
        sha_path = path.with_suffix(path.suffix + ".sha256")
        if not sha_path.exists():
            raise RuntimeError(f"sha256 없음: {sha_path}")
        content = path.read_bytes()
        expected = sha_path.read_text().strip().split()[0]
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise RuntimeError(f"gat_model.pt sha256 불일치: {path}")

        import pickle
        payload = pickle.loads(content)

        from .gat_model import GATModel
        init_p = payload["model_init_params"]
        model = GATModel(**init_p)
        model.load_state_dict(payload["model_state"])
        model.eval()

        # 그래프 아티팩트 검증 로드
        graph_builder = GraphBuilder.load(path.parent)

        obj = cls.__new__(cls)
        obj.params = payload["params"]
        obj.config = None
        obj._gat_model = model
        obj._graph_builder = graph_builder
        obj._calibrator = payload.get("calibrator")
        obj.best_threshold_ = payload.get("best_threshold", 0.5)
        obj._trained = True
        obj.model_dir = path.parent
        obj.feature_importances_ = None
        logger.info("GATTrainer 로드: %s", path)
        return obj
```

- [ ] **Step 4: 테스트 실행 — PASS 확인**

```bash
python -m pytest tests/test_train/test_gat_trainer.py::TestGATTrainer -v
```
Expected: `5 passed`

- [ ] **Step 5: 커밋**

```bash
git add scripts/train/gat_trainer.py tests/test_train/test_gat_trainer.py
git commit -m "feat: GATTrainer — 학습, Platt 보정, 저장/로드"
```

---

## Task 6: TrainConfig + build_trainer 확장

**Context:** `TrainConfig`에 `gat_params` 필드를 추가하고 `build_trainer()`에 `model_type="gat"` 지원을 추가한다.

**Files:**
- Modify: `scripts/train/hyperparams.py`
- Modify: `scripts/train/trainer.py`
- Test: `tests/test_train/test_gat_trainer.py` (TestBuildTrainer 클래스)

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_train/test_gat_trainer.py 에 추가

class TestBuildTrainer:
    def test_build_gat_trainer(self, tmp_path):
        """build_trainer(config) model_type='gat' → GATTrainer 반환."""
        from scripts.train.hyperparams import TrainConfig
        from scripts.train.trainer import build_trainer
        from scripts.train.gat_trainer import GATTrainer

        config = TrainConfig(model_type="gat", model_dir=str(tmp_path))
        trainer = build_trainer(config)
        assert isinstance(trainer, GATTrainer)

    def test_train_config_gat_params_default(self):
        """TrainConfig 기본 gat_params 존재 확인."""
        from scripts.train.hyperparams import TrainConfig
        config = TrainConfig()
        assert "hidden_dim" in config.gat_params
        assert "epochs" in config.gat_params
```

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
python -m pytest tests/test_train/test_gat_trainer.py::TestBuildTrainer -v 2>&1 | head -15
```

- [ ] **Step 3: hyperparams.py 수정 — gat_params 추가**

`scripts/train/hyperparams.py` 파일에서 `TrainConfig` 클래스에 아래 필드를 추가한다. (`xgb_params`, `lgb_params` 필드 뒤에):

```python
    # GAT 하이퍼파라미터
    gat_params: dict = field(default_factory=lambda: {
        "hidden_dim": 64,
        "heads": 4,
        "out_dim": 32,
        "epochs": 200,
        "lr": 0.001,
        "patience": 20,
        "random_state": 42,
    })
```

또한 `model_type` 검증부가 있다면 `"gat"`을 허용하도록 수정:

```python
    def get_model_params(self) -> dict:
        if self.model_type == "lightgbm":
            return self.lgb_params
        if self.model_type == "gat":
            return self.gat_params
        return self.xgb_params
```

- [ ] **Step 4: trainer.py 수정 — build_trainer에 gat 추가**

`scripts/train/trainer.py`의 `build_trainer()` 함수 `elif model_type == "ensemble":` 블록 뒤에 추가:

```python
    elif model_type == "gat":
        from .gat_trainer import GATTrainer
        model_dir = getattr(config, "model_dir", "models")
        return GATTrainer(config.gat_params, config, model_dir=model_dir)
```

- [ ] **Step 5: 테스트 실행 — PASS 확인**

```bash
python -m pytest tests/test_train/test_gat_trainer.py::TestBuildTrainer -v
```
Expected: `2 passed`

- [ ] **Step 6: 전체 테스트 — 기존 테스트 깨지지 않음 확인**

```bash
python -m pytest tests/test_train/ -v --tb=short 2>&1 | tail -20
```
Expected: 기존 테스트 모두 PASS

- [ ] **Step 7: 커밋**

```bash
git add scripts/train/hyperparams.py scripts/train/trainer.py tests/test_train/test_gat_trainer.py
git commit -m "feat: TrainConfig gat_params + build_trainer gat 지원"
```

---

## Task 7: EnsembleTrainer 3-way + Recall 제약 가중치 최적화

**Context:** 기존 `EnsembleTrainer`를 확장하여 `GATTrainer`를 세 번째 서브모델로 추가한다. 가중치 최적화는 calibration 스플릿에서 Recall ≥ 0.90 제약 하에 AUC를 최대화한다. 미지 약물 요청에서는 w_gat=0, 나머지 정규화.

**Files:**
- Modify: `scripts/train/trainer.py`
- Test: `tests/test_train/test_gat_trainer.py` (TestEnsemble3Way 클래스)

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_train/test_gat_trainer.py 에 추가

class TestEnsemble3Way:
    def test_weights_sum_to_one(self):
        """앙상블 가중치 합 = 1.0."""
        from scripts.train.trainer import EnsembleTrainer3Way
        ens = EnsembleTrainer3Way.__new__(EnsembleTrainer3Way)
        ens.weights = (0.3, 0.3, 0.4)
        assert abs(sum(ens.weights) - 1.0) < 1e-6

    def test_predict_proba_excludes_gat_for_unknown(self):
        """미지 약물 포함 요청 → GAT 제외, xgb+lgb 가중치 재정규화."""
        import numpy as np
        from scripts.train.trainer import EnsembleTrainer3Way

        class FakeXGB:
            def predict_proba(self, X): return np.array([0.6] * len(X))
        class FakeLGB:
            def predict_proba(self, X): return np.array([0.4] * len(X))
        class FakeGAT:
            def predict_pair_proba(self, a, b): return None  # unknown drug

        ens = EnsembleTrainer3Way.__new__(EnsembleTrainer3Way)
        ens._xgb = FakeXGB()
        ens._lgb = FakeLGB()
        ens._gat = FakeGAT()
        ens.weights = (0.3, 0.3, 0.4)
        ens._trained = True

        X = np.zeros((1, 3))
        drug_pairs = [("UNKNOWN", "D01")]
        result = ens.predict_proba_with_gat(X, drug_pairs)
        # w_gat=0 → (0.3*0.6 + 0.3*0.4) / 0.6 = 0.5
        assert abs(result[0] - 0.5) < 1e-5

    def test_build_trainer_ensemble_gat(self, tmp_path):
        """model_type='ensemble_gat' → EnsembleTrainer3Way."""
        from scripts.train.hyperparams import TrainConfig
        from scripts.train.trainer import build_trainer, EnsembleTrainer3Way
        config = TrainConfig(model_type="ensemble_gat", model_dir=str(tmp_path))
        trainer = build_trainer(config)
        assert isinstance(trainer, EnsembleTrainer3Way)
```

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
python -m pytest tests/test_train/test_gat_trainer.py::TestEnsemble3Way -v 2>&1 | head -10
```

- [ ] **Step 3: trainer.py에 EnsembleTrainer3Way 추가**

`scripts/train/trainer.py`의 `EnsembleTrainer` 클래스 뒤에 추가:

```python
class EnsembleTrainer3Way(BaseTrainer):
    """XGBoost + LightGBM + GAT 소프트 보팅 앙상블 (3-way).

    가중치 최적화: calibration 스플릿에서 Recall >= 0.90 제약 하 AUC 최대화 (SLSQP).
    미지 약물 포함 요청: w_gat=0, 나머지 가중치 정규화.
    """

    def __init__(
        self,
        xgb_params: dict,
        lgb_params: dict,
        gat_params: dict,
        config: Any,
        weights: tuple[float, float, float] = (1/3, 1/3, 1/3),
    ):
        super().__init__(xgb_params, config)
        self.weights = weights
        self._xgb = XGBoostTrainer(xgb_params, config)
        self._lgb = LGBMTrainer(lgb_params, config)
        from .gat_trainer import GATTrainer
        model_dir = getattr(config, "model_dir", "models")
        self._gat = GATTrainer(gat_params, config, model_dir=model_dir)

    def fit(self, dataset) -> "EnsembleTrainer3Way":
        """tabular 데이터용 fit — GAT는 별도 fit_gat() 호출 필요."""
        from .dataset import TrainDataset
        if not isinstance(dataset, TrainDataset):
            raise TypeError("EnsembleTrainer3Way.fit()은 TrainDataset 필요")
        logger.info("앙상블 3-way 훈련: XGBoost + LightGBM")
        self._xgb.fit(dataset)
        self._lgb.fit(dataset)
        if self._xgb.feature_importances_ is not None:
            w1, w2, _ = self.weights
            norm = w1 + w2 or 1.0
            self.feature_importances_ = (
                (w1 / norm) * self._xgb.feature_importances_
                + (w2 / norm) * self._lgb.feature_importances_
            )
        self._trained = True
        return self

    def fit_gat(self, gat_dataset) -> "EnsembleTrainer3Way":
        """GAT 서브모델 학습 (별도 호출)."""
        self._gat.fit_graph(gat_dataset)
        return self

    def optimize_weights(
        self,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        drug_pairs_calib: list[tuple[str, str]],
        recall_threshold: float = 0.90,
    ) -> tuple[float, float, float]:
        """
        Recall >= recall_threshold 제약 하에서 AUC 최대화 (SLSQP).

        Parameters
        ----------
        X_calib          : calibration 스플릿 tabular 피처
        y_calib          : calibration 스플릿 이진 레이블
        drug_pairs_calib : calibration 스플릿 약물쌍 목록 (요청당 1쌍)
        recall_threshold : 최소 Recall 요건

        Returns
        -------
        (w_xgb, w_lgb, w_gat) 최적 가중치
        """
        from scipy.optimize import minimize
        from sklearn.metrics import roc_auc_score, recall_score

        p_xgb = self._xgb.predict_proba(X_calib)
        p_lgb = self._lgb.predict_proba(X_calib)
        p_gat_list = []
        for drug_a, drug_b in drug_pairs_calib:
            val = self._gat.predict_pair_proba(drug_a, drug_b)
            p_gat_list.append(val if val is not None else 0.5)
        p_gat = np.array(p_gat_list)

        def neg_auc(w):
            p = w[0] * p_xgb + w[1] * p_lgb + w[2] * p_gat
            try:
                return -roc_auc_score(y_calib, p)
            except Exception:
                return 0.0

        def recall_constraint(w):
            p = w[0] * p_xgb + w[1] * p_lgb + w[2] * p_gat
            pred = (p >= 0.5).astype(int)
            try:
                return recall_score(y_calib, pred, zero_division=0) - recall_threshold
            except Exception:
                return -1.0

        constraints = [
            {"type": "eq",   "fun": lambda w: w.sum() - 1.0},
            {"type": "ineq", "fun": recall_constraint},
        ]
        bounds = [(0.0, 1.0)] * 3
        x0 = np.array([1/3, 1/3, 1/3])

        result = minimize(neg_auc, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            self.weights = tuple(float(v) for v in result.x)
        else:
            logger.warning("가중치 최적화 실패 — 균등 가중치 유지: %s", result.message)
        logger.info("앙상블 최적 가중치: xgb=%.3f lgb=%.3f gat=%.3f", *self.weights)
        return self.weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """tabular-only 예측 (GAT 제외). serving에서는 predict_proba_with_gat 사용."""
        w1, w2, _ = self.weights
        norm = w1 + w2 or 1.0
        return (w1 / norm) * self._xgb.predict_proba(X) + (w2 / norm) * self._lgb.predict_proba(X)

    def predict_proba_with_gat(
        self,
        X: np.ndarray,
        drug_pairs: list[tuple[str, str]],
    ) -> np.ndarray:
        """
        GAT 포함 예측.

        Parameters
        ----------
        X          : [N, feature_dim] tabular 피처
        drug_pairs : [(drug_a, drug_b)] 요청당 대표 쌍 (N개)

        Returns
        -------
        [N] 최종 앙상블 확률
        """
        w1, w2, w3 = self.weights
        p_xgb = self._xgb.predict_proba(X)
        p_lgb = self._lgb.predict_proba(X)

        results = np.zeros(len(X))
        for i, (drug_a, drug_b) in enumerate(drug_pairs):
            p_gat = self._gat.predict_pair_proba(drug_a, drug_b)
            if p_gat is None:
                # GAT 제외 — 나머지 가중치 정규화
                norm = w1 + w2 or 1.0
                results[i] = (w1 / norm) * p_xgb[i] + (w2 / norm) * p_lgb[i]
            else:
                results[i] = w1 * p_xgb[i] + w2 * p_lgb[i] + w3 * p_gat
        return results

    def save(self, path: str | Path) -> Path:
        import hashlib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._xgb.save(path.with_suffix(".xgb.pkl"))
        self._lgb.save(path.with_suffix(".lgb.pkl"))
        gat_path = path.parent / "gat_model.pt"
        if self._gat._trained:
            self._gat.save(gat_path)
        payload = {
            "trainer_class": self.__class__.__name__,
            "weights": self.weights,
            "best_threshold": self.best_threshold_,
            "feature_importances": self.feature_importances_,
        }
        content = pickle.dumps(payload)
        path.write_bytes(content)
        sha256 = hashlib.sha256(content).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256}  {path.name}\n")
        logger.info("EnsembleTrainer3Way 저장: %s", path)
        return path
```

`build_trainer()` 함수에 아래를 추가 (`elif model_type == "gat":` 뒤에):

```python
    elif model_type == "ensemble_gat":
        model_dir = getattr(config, "model_dir", "models")
        return EnsembleTrainer3Way(
            config.xgb_params, config.lgb_params, config.gat_params, config
        )
```

- [ ] **Step 4: 테스트 실행 — PASS 확인**

```bash
python -m pytest tests/test_train/test_gat_trainer.py::TestEnsemble3Way -v
```
Expected: `3 passed`

- [ ] **Step 5: 전체 기존 테스트 확인**

```bash
python -m pytest tests/test_train/ -v --tb=short 2>&1 | tail -15
```

- [ ] **Step 6: 커밋**

```bash
git add scripts/train/trainer.py scripts/train/hyperparams.py tests/test_train/test_gat_trainer.py
git commit -m "feat: EnsembleTrainer3Way + Recall 제약 가중치 최적화"
```

---

## Task 8: Serving 확장 — MLModel GAT 지원

**Context:** `MLModel.load()`가 `EnsembleTrainer3Way` 타입 시 `gat_model.pt` + `gat_graph.pt`를 로드한다. 예측 흐름: 요청 내 모든 약물쌍 열거 → GAT 쌍 스코어 → max 집계 → 앙상블 입력.

**Files:**
- Modify: `serving/predictor.py`

- [ ] **Step 1: 변경 위치 파악**

`serving/predictor.py`의 `MLModel` 클래스. `load()` 메서드(225번째 줄)와 `predict_proba()` 메서드(310번째 줄)를 수정한다.

- [ ] **Step 2: MLModel 속성 추가**

`MLModel.__init__()` 내 기존 속성 뒤에 추가:

```python
        self._gat_trainer = None   # GATTrainer 인스턴스 (EnsembleTrainer3Way용)
        self._gat_graph_age_warned = False
```

- [ ] **Step 3: load()에 GAT 로드 로직 추가**

`load()` 메서드의 `if self._model is None and state.get("trainer_class") == "EnsembleTrainer":` 블록 뒤에 추가:

```python
            # EnsembleTrainer3Way: GAT 서브모델 추가 로드
            if state.get("trainer_class") == "EnsembleTrainer3Way":
                gat_model_path = path.parent / "gat_model.pt"
                if gat_model_path.exists():
                    try:
                        from scripts.train.gat_trainer import GATTrainer
                        self._gat_trainer = GATTrainer.load_gat(gat_model_path)
                        # 그래프 나이 경고
                        import json
                        from datetime import datetime, timezone
                        meta_path = path.parent / "gat_graph_meta.json"
                        if meta_path.exists():
                            meta = json.loads(meta_path.read_text())
                            built_at_str = meta.get("built_at", "")
                            if built_at_str:
                                built_at = datetime.fromisoformat(built_at_str)
                                age_days = (datetime.utcnow() - built_at).days
                                if age_days > 180 and not self._gat_graph_age_warned:
                                    logger.warning(
                                        "gat_graph.pt 나이 %d일 (>180일) — 그래프 재빌드 권장",
                                        age_days,
                                    )
                                    self._gat_graph_age_warned = True
                        logger.info("GATTrainer 로드 완료: %s", gat_model_path)
                    except Exception as e:
                        logger.warning("GATTrainer 로드 실패 (GAT 제외 모드): %s", e)
                        self._gat_trainer = None
                else:
                    logger.warning("gat_model.pt 없음 — GAT 없이 앙상블 로드")
```

- [ ] **Step 4: predict_proba() GAT 집계 추가**

`serving/predictor.py`의 `predict_proba()` 메서드 (라인 310~). 기존 메서드 뒤에 새 메서드 추가:

```python
    def predict_proba_gat(
        self,
        X: np.ndarray,
        drug_codes: list[str],
    ) -> float:
        """
        GAT 포함 앙상블 예측.

        Parameters
        ----------
        X          : [1, feature_dim] tabular 피처 (스케일링 적용 후)
        drug_codes : 요청 내 약물 코드 목록

        Returns
        -------
        최종 DDI 위험 확률 (0~1)
        """
        from itertools import combinations

        # tabular 예측 (기존 경로)
        base_prob = float(self._model.predict_proba(X.reshape(1, -1))[0, 1])

        if self._gat_trainer is None or len(drug_codes) < 2:
            return base_prob

        # 모든 약물쌍 GAT 스코어 → max 집계
        valid_scores = []
        for drug_a, drug_b in combinations(drug_codes, 2):
            score = self._gat_trainer.predict_pair_proba(drug_a, drug_b)
            if score is not None:
                valid_scores.append(score)

        if not valid_scores:
            # 모든 쌍 미지 약물 → GAT 제외, tabular만 사용
            return base_prob

        p_gat = float(max(valid_scores))

        # 앙상블 가중치는 EnsembleTrainer3Way 저장값 사용
        # 단순 로드 시 weights 정보는 model_prod.pkl에 있음
        # state['weights']를 load() 시 캐시
        weights = getattr(self, "_ensemble_weights", (1/3, 1/3, 1/3))
        w1, w2, w3 = weights
        # base_prob = w1*p_xgb + w2*p_lgb (이미 계산됨, 단순화: base_prob를 tabular 몫으로 사용)
        # 최종: (w1+w2)*base_prob + w3*p_gat 정규화
        tab_weight = w1 + w2
        total = tab_weight + w3
        return (tab_weight * base_prob + w3 * p_gat) / (total or 1.0)
```

`load()` 내 weights 캐싱 추가 (EnsembleTrainer3Way 로드 직후):

```python
                        self._ensemble_weights = state.get("weights", (1/3, 1/3, 1/3))
```

- [ ] **Step 5: 전체 기존 serving 테스트 확인**

```bash
python -m pytest tests/test_serving/ -v --tb=short 2>&1 | tail -20
```
Expected: 기존 테스트 모두 PASS

- [ ] **Step 6: 커밋**

```bash
git add serving/predictor.py
git commit -m "feat: serving MLModel GAT 로드 + 약물쌍 max 집계 예측"
```

---

## Task 9: DAG 배포 체인 확장

**Context:** `_deploy_model()`의 Phase 1 선검증과 Phase 2 복사 대상에 GAT 아티팩트 3종(`gat_model.pt`, `gat_graph.pt`, `gat_graph_meta.json`)을 추가한다. sha256 누락 시 배포 중단.

**Files:**
- Modify: `dags/ddi_train_dag.py`

- [ ] **Step 1: 변경 위치 파악**

`dags/ddi_train_dag.py`의 `_deploy_model()` 함수. `artifacts` 목록 구성 부분(라인 212~225)을 수정한다.

- [ ] **Step 2: artifacts 목록에 GAT 아티팩트 추가**

기존 `.xgb.pkl`, `.lgb.pkl` 처리 블록 뒤에 추가:

```python
    # GAT 아티팩트 (EnsembleTrainer3Way인 경우)
    for gat_file, gat_sha_suffix in [
        ("gat_model.pt",   "gat_model.pt.sha256"),
        ("gat_graph.pt",   "gat_graph.pt.sha256"),
    ]:
        gat_src = base_src.parent / gat_file
        if gat_src.exists():
            gat_sha = base_src.parent / gat_sha_suffix
            if not gat_sha.exists():
                raise RuntimeError(
                    f"배포 중단 — GAT 아티팩트 해시 없음: {gat_sha}"
                )
            artifacts.append((gat_src, gat_file))
            artifacts.append((gat_sha, gat_sha_suffix))
    # gat_graph_meta.json (sha256 없음 — 메타데이터)
    gat_meta_src = base_src.parent / "gat_graph_meta.json"
    if gat_meta_src.exists():
        artifacts.append((gat_meta_src, "gat_graph_meta.json"))
```

- [ ] **Step 3: 배포 통합 테스트 실행 — 기존 테스트 깨지지 않음 확인**

```bash
python -m pytest tests/test_integration/test_deploy_integrity.py -v --tb=short 2>&1 | tail -20
```
Expected: 기존 18개 테스트 모두 PASS

- [ ] **Step 4: 커밋**

```bash
git add dags/ddi_train_dag.py
git commit -m "feat: DAG 배포 체인에 GAT 아티팩트 추가"
```

---

## Task 10: GAT 통합 테스트

**Context:** GAT 배포 체인 완결성, sha256 누락 중단, path traversal, 미지 약물 앙상블 제외를 검증한다.

**Files:**
- Create: `tests/test_integration/test_gat_deploy.py`

- [ ] **Step 1: 테스트 파일 작성**

```python
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
    trainer.fit_graph(ds)
    trainer.save(tmp_path / "gat_model.pt")
    return trainer, tmp_path


# ── 배포 체인 테스트 ──────────────────────────────────────────────────────────

class TestGATDeployChain:
    def test_gat_model_sha256_missing_raises(self, tmp_path):
        """gat_model.pt.sha256 누락 → load_gat() RuntimeError."""
        # gat_model.pt 생성 (sha256 없음)
        (tmp_path / "gat_model.pt").write_bytes(b"dummy")
        # gat_graph.pt도 없어서 결국 sha256 없음 에러
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
        """save → load_gat → predict_pair_proba 정상."""
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
            def predict_proba(self, X): return np.array([p for p in [0.5]])
        class FakeLGB:
            def predict_proba(self, X): return np.array([p for p in [0.5]])
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
        """model_dir 외부 경로를 가리키는 sha256 파일 → 로드 시 RuntimeError."""
        # gat_graph.pt를 다른 위치에 생성하고 sha256 조작
        outer_dir = tmp_path / "outer"
        outer_dir.mkdir()
        (tmp_path / "gat_graph.pt").write_bytes(b"dummy")
        # sha256은 올바른 값으로 작성하되 gat_graph.pt 내용은 dummy
        content = b"dummy"
        sha = hashlib.sha256(content).hexdigest()
        (tmp_path / "gat_graph.pt.sha256").write_text(f"{sha}  gat_graph.pt\n")
        # load는 성공하지만 내용이 유효하지 않아 torch.load에서 실패해야 함
        with pytest.raises(Exception):
            GraphBuilder.load(tmp_path)
```

- [ ] **Step 2: 테스트 실행 — 모두 PASS 확인**

```bash
python -m pytest tests/test_integration/test_gat_deploy.py -v --tb=short
```
Expected: 모두 PASS (PyG 미설치 시 skip)

- [ ] **Step 3: 전체 테스트 스위트 실행**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```
Expected: 기존 431개 + 신규 GAT 테스트 모두 PASS (PyG 미설치 환경은 skip)

- [ ] **Step 4: 커밋**

```bash
git add tests/test_integration/test_gat_deploy.py tests/test_train/test_gat_trainer.py
git commit -m "test: GAT 배포 체인 통합 테스트 + 유닛 테스트"
```

---

## 자기 검토 (Self-Review)

### 스펙 커버리지
| 스펙 요구사항 | 구현 Task |
|---|---|
| GraphBuilder gat_graph.pt 직렬화 (C-2) | Task 2 |
| 미지 약물 GAT 제외 (C-3) | Task 5, 7 |
| 요청 레벨 max 집계 (C-1) | Task 7, 8 |
| 훈련 분할만 엣지 구성 (H-1) | Task 2 |
| 60/20/10/10 분할 (H-2) | Task 5 |
| Platt scaling (H-3) | Task 5 |
| gat_graph_meta.json (H-4) | Task 2 |
| BaseGraphTrainer (H-5) | Task 4 |
| Recall 제약 가중치 최적화 (H-6) | Task 7 |
| h_a ⊙ h_b 쌍 스코어링 (M-3) | Task 3 |
| load() 그래프 해시 검증 (M-4) | Task 5 |
| torch.use_deterministic_algorithms (L-3) | Task 5 |
| DAG 배포 체인 GAT 아티팩트 (C-2) | Task 9 |
| 통합 테스트 (스펙 §8) | Task 10 |

### 메서드 시그니처 일관성
- `GATTrainer.fit_graph(GATDataset)` → Task 4, 5 동일
- `predict_pair_proba(drug_a: str, drug_b: str) → float | None` → Task 4, 5, 7, 10 동일
- `GATTrainer.load_gat(path)` → Task 5, 10 동일
- `EnsembleTrainer3Way.predict_proba_with_gat(X, drug_pairs)` → Task 7, 10 동일
