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
        [N] 쌍별 DDI 위험 확률 (sigmoid 출력)
        """
        # self() 호출로 PyTorch forward hook 활성화 (self.forward() 직접 호출 금지)
        h = self(x, edge_index)   # [num_nodes, out_dim]
        h_a = h[pairs[:, 0]]              # [N, out_dim]
        h_b = h[pairs[:, 1]]              # [N, out_dim]
        feat = torch.cat(
            [h_a, h_b, (h_a - h_b).abs(), h_a * h_b],
            dim=-1,
        )                                  # [N, out_dim * 4]
        logit = self.pair_scorer(feat).squeeze(-1)  # [N]
        return torch.sigmoid(logit)
