"""GAT 훈련용 데이터 컨테이너."""
from __future__ import annotations

from dataclasses import dataclass
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
    prescription_split: Optional[str] = None
    pairs_train: Optional[np.ndarray] = None        # [N, 3] int64
    pairs_gat_val: Optional[np.ndarray] = None      # [M, 3] int64
    pairs_calibration: Optional[np.ndarray] = None  # [K, 3] int64

    def __post_init__(self) -> None:
        """GraphBuilder가 train provenance를 강제 검증할 수 있도록 split 메타데이터를 보존."""
        if self.prescription_split is not None:
            self.prescription_df.attrs["split"] = str(self.prescription_split)

    @property
    def unique_drugs(self) -> list[str]:
        """처방 DataFrame 내 고유 약물 코드 목록 (정렬)."""
        return sorted(self.prescription_df["drug_code"].unique())

    @property
    def num_drugs(self) -> int:
        """고유 약물 코드 수."""
        return len(self.unique_drugs)
