"""Graph 모델용 BaseTrainer 확장 — PyG Data 객체를 수용하도록 fit() 재정의."""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional

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
        """GATDataset만 허용. 다른 타입 → TypeError."""
        if not isinstance(dataset, GATDataset):
            raise TypeError(
                f"BaseGraphTrainer.fit()은 GATDataset 필요, "
                f"받은 타입: {type(dataset).__name__}"
            )
        return self.fit_graph(dataset)

    @abstractmethod
    def fit_graph(self, dataset: GATDataset) -> "BaseGraphTrainer":
        """그래프 기반 학습 구현."""
        ...

    @abstractmethod
    def predict_pair_proba(self, drug_a: str, drug_b: str) -> Optional[float]:
        """
        단일 약물쌍 DDI 위험 확률.
        미지 약물 포함 시 None 반환 (앙상블에서 GAT 제외).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """serving 호환 배열 인터페이스 — 서브클래스가 구현."""
        ...
