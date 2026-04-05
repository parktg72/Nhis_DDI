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
        self.data = None
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

        ddi_count = torch.zeros(num_nodes)
        for _, row in ddi_df.iterrows():
            ai = self.drug_to_idx.get(str(row["drug_a"]))
            bi = self.drug_to_idx.get(str(row["drug_b"]))
            if ai is not None:
                ddi_count[ai] += 1
            if bi is not None:
                ddi_count[bi] += 1

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
        self.data.drug_to_idx = self.drug_to_idx

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
