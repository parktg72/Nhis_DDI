"""
CYP 효소 기반 약물상호작용 피처 추출

CYP(Cytochrome P450) 5개 주요 효소:
  CYP3A4, CYP2D6, CYP2C9, CYP2C19, CYP1A2

피처 유형:
  - substrate_count      : 환자 약물 중 각 CYP의 기질 수
  - strong_inhibitor_cnt : 강한 억제제 수 (기질과 동시 복용 시 고위험)
  - inhibitor_substrate_pairs : 억제제+기질 쌍 수 (실제 상호작용 위험)
  - cyp_risk_score       : CYP 전체 위험 점수 (가중합)
  - max_cyp_risk         : 가장 위험한 단일 CYP 위험도

위험 판단 기준:
  - Strong inhibitor + Substrate 동시 복용 → 혈중 농도 2배↑ 위험
  - CYP3A4: 약물 50% 이상 대사 → 가중치 최고
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CYP_ENZYMES = ["cyp3a4", "cyp2d6", "cyp2c9", "cyp2c19", "cyp1a2"]

# CYP 효소별 임상 중요도 가중치 (CYP3A4 > 나머지)
CYP_WEIGHTS = {
    "cyp3a4":  3.0,
    "cyp2d6":  2.0,
    "cyp2c9":  2.0,
    "cyp2c19": 1.5,
    "cyp1a2":  1.0,
}

# 최종 피처 컬럼명 목록
CYP_FEATURE_COLS = (
    [f"{e}_substrates" for e in CYP_ENZYMES]
    + [f"{e}_strong_inhibitors" for e in CYP_ENZYMES]
    + [f"{e}_inhibitor_substrate_pairs" for e in CYP_ENZYMES]
    + ["cyp_risk_score", "cyp_max_enzyme_risk", "cyp_high_risk_pairs"]
)


class CYPFeatureExtractor:
    """
    환자별 CYP 상호작용 피처 추출기.

    Parameters
    ----------
    cyp_matrix_path : cyp_matrix.parquet 경로
    drug_index_path : drug_name_index.parquet 경로 (ATC→DrugBank 매핑)
    """

    def __init__(
        self,
        cyp_matrix_path: str | Path = "data/processed/cyp_matrix.parquet",
        drug_index_path: str | Path = "data/processed/drug_name_index.parquet",
    ):
        self._cyp: pd.DataFrame = self._load_cyp(Path(cyp_matrix_path))
        self._atc_to_dbid: dict[str, str] = self._build_atc_map(Path(drug_index_path))
        # DrugBank ID → CYP 피처 행
        self._dbid_map: dict[str, dict] = {
            row["drugbank_id"]: row
            for _, row in self._cyp.iterrows()
            if pd.notna(row.get("drugbank_id"))
        }
        logger.info(
            "CYPFeatureExtractor: %d DrugBank 약물 로드, %d ATC 매핑",
            len(self._dbid_map), len(self._atc_to_dbid),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 초기화 헬퍼
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_cyp(path: Path) -> pd.DataFrame:
        if not path.exists():
            logger.warning("cyp_matrix.parquet 없음: %s", path)
            return pd.DataFrame()
        return pd.read_parquet(path)

    @staticmethod
    def _build_atc_map(path: Path) -> dict[str, str]:
        """atc_code → drugbank_id 매핑 딕셔너리."""
        if not path.exists():
            logger.warning("drug_name_index.parquet 없음: %s", path)
            return {}
        df = pd.read_parquet(path)
        mapping: dict[str, str] = {}
        for _, row in df.iterrows():
            dbid = str(row.get("drugbank_id", "")).strip()
            atc_raw = str(row.get("atc_codes", "")).strip()
            if not dbid or not atc_raw or atc_raw == "nan":
                continue
            # atc_codes가 쉼표 구분 또는 단일값
            for atc in atc_raw.split(","):
                atc = atc.strip()
                if atc:
                    mapping[atc] = dbid
        return mapping

    # ──────────────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────────────

    def extract(self, atc_codes: list[str]) -> dict[str, float]:
        """
        약물 ATC 코드 목록 → CYP 피처 딕셔너리.

        Parameters
        ----------
        atc_codes : 환자의 동시복용 약물 ATC 코드 목록
        """
        # ATC → DrugBank ID → CYP 행
        drug_rows = []
        for atc in atc_codes:
            atc = str(atc).strip()
            dbid = self._atc_to_dbid.get(atc)
            if dbid and dbid in self._dbid_map:
                drug_rows.append(self._dbid_map[dbid])

        features: dict[str, float] = {col: 0.0 for col in CYP_FEATURE_COLS}

        if not drug_rows:
            return features

        cyp_risk_score = 0.0
        max_enzyme_risk = 0.0
        total_high_risk_pairs = 0

        for enzyme in CYP_ENZYMES:
            substrates = [
                r for r in drug_rows
                if r.get(f"{enzyme}_substrate_count", 0) > 0
            ]
            strong_inhibitors = [
                r for r in drug_rows
                if r.get(f"{enzyme}_inhibitor_strong_count", 0) > 0
            ]
            inhibitors = [
                r for r in drug_rows
                if r.get(f"{enzyme}_inhibitor_count", 0) > 0
            ]

            n_sub = len(substrates)
            n_strong_inh = len(strong_inhibitors)
            n_inh = len(inhibitors)

            # 억제제+기질 쌍: 강한 억제제가 있고 기질도 있는 경우
            inh_sub_pairs = n_strong_inh * n_sub if n_sub > 0 and n_strong_inh > 0 else 0

            features[f"{enzyme}_substrates"] = float(n_sub)
            features[f"{enzyme}_strong_inhibitors"] = float(n_strong_inh)
            features[f"{enzyme}_inhibitor_substrate_pairs"] = float(inh_sub_pairs)

            # 효소별 위험 점수
            enzyme_risk = (
                n_strong_inh * 3.0   # 강한 억제제
                + n_inh * 1.0         # 일반 억제제
                + inh_sub_pairs * 2.0  # 억제제+기질 쌍 추가 가중
            ) * CYP_WEIGHTS[enzyme]

            cyp_risk_score += enzyme_risk
            max_enzyme_risk = max(max_enzyme_risk, enzyme_risk)
            total_high_risk_pairs += inh_sub_pairs

        features["cyp_risk_score"] = cyp_risk_score
        features["cyp_max_enzyme_risk"] = max_enzyme_risk
        features["cyp_high_risk_pairs"] = float(total_high_risk_pairs)

        return features

    def extract_batch(self, df: pd.DataFrame, atc_col: str = "atc_codes") -> pd.DataFrame:
        """
        환자별 ATC 코드 목록이 담긴 DataFrame → CYP 피처 DataFrame.
        df 에는 patient_id 와 atc_codes(리스트) 컬럼이 필요.
        """
        rows = []
        for _, row in df.iterrows():
            atc_list = row.get(atc_col, [])
            if isinstance(atc_list, str):
                atc_list = [a.strip() for a in atc_list.split(",") if a.strip()]
            feat = self.extract(atc_list or [])
            feat["patient_id"] = row.get("patient_id", "")
            rows.append(feat)

        if not rows:
            return pd.DataFrame(columns=["patient_id"] + CYP_FEATURE_COLS)

        result = pd.DataFrame(rows)
        # patient_id를 앞으로
        cols = ["patient_id"] + [c for c in result.columns if c != "patient_id"]
        return result[cols]
