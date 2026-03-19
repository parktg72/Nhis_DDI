"""
EDI 코드 → ATC 코드 표준화
DrugBank 파싱 결과(drug_name_index.parquet)를 매핑 테이블로 사용.
EDI 코드(건보 의약품코드) → ATC 코드 7자리 + 약품명 매핑.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CodeStandardizer:
    """
    EDI → ATC 코드 변환기.

    매핑 테이블 우선순위:
    1. drug_name_index.parquet (DrugBank 파싱 결과)
    2. 식약처 DUR parquet (dur_ddi_contraindicated.parquet 등)
    3. 사용자 정의 CSV (config/edi_atc_extra.csv)
    """

    def __init__(
        self,
        index_path: str | Path = "data/processed/drug_name_index.parquet",
        dur_path: str | Path | None = "data/dur/dur_ddi_contraindicated_std.parquet",
        extra_csv: str | Path | None = "config/edi_atc_extra.csv",
    ):
        self._map: dict[str, dict] = {}  # edi_code → {atc_code, drug_name}
        self._load_index(Path(index_path))
        if dur_path and Path(dur_path).exists():
            self._load_dur(Path(dur_path))
        if extra_csv and Path(extra_csv).exists():
            self._load_extra(Path(extra_csv))
        logger.info("CodeStandardizer 초기화: %d개 EDI 매핑 로드", len(self._map))

    # ──────────────────────────────────────────────────────────────────────────
    # 로딩 메서드
    # ──────────────────────────────────────────────────────────────────────────

    def _load_index(self, path: Path) -> None:
        if not path.exists():
            logger.warning("drug_name_index.parquet 없음: %s", path)
            return
        df = pd.read_parquet(path)
        # DrugBank 인덱스 컬럼: drug_id, name, atc_code (있는 경우)
        for _, row in df.iterrows():
            edi = str(row.get("drug_id", "")).strip()
            atc = str(row.get("atc_code", "")).strip() if pd.notna(row.get("atc_code")) else None
            name = str(row.get("name", "")).strip() if pd.notna(row.get("name")) else None
            if edi:
                self._map[edi] = {"atc_code": atc, "drug_name": name}

    def _load_dur(self, path: Path) -> None:
        """DUR 병용금기 표준화 데이터에서 성분코드 → ATC 보완 (없으면 code 그대로 사용)."""
        df = pd.read_parquet(path)
        for col_code, col_name in [
            ("drug_a_code", "drug_a_name"),
            ("drug_b_code", "drug_b_name"),
        ]:
            if col_code not in df.columns:
                continue
            sub = df[[col_code, col_name]].dropna(subset=[col_code]).drop_duplicates(col_code)
            for _, row in sub.iterrows():
                code = str(row[col_code]).strip()
                name = str(row[col_name]).strip() if pd.notna(row.get(col_name)) else None
                if code and code not in self._map:
                    self._map[code] = {"atc_code": None, "drug_name": name}

    def _load_extra(self, path: Path) -> None:
        """사용자 정의 EDI→ATC 매핑 CSV (edi_code, atc_code, drug_name)."""
        df = pd.read_csv(path, dtype=str)
        required = {"edi_code", "atc_code"}
        if not required.issubset(df.columns):
            logger.warning("extra CSV 컬럼 부족: %s", df.columns.tolist())
            return
        for _, row in df.iterrows():
            edi = str(row["edi_code"]).strip()
            atc = str(row["atc_code"]).strip() if pd.notna(row["atc_code"]) else None
            name = str(row.get("drug_name", "")).strip() or None
            if edi:
                self._map[edi] = {"atc_code": atc, "drug_name": name}

    # ──────────────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────────────

    def lookup(self, edi_code: str) -> tuple[Optional[str], Optional[str]]:
        """(atc_code, drug_name) 반환. 미매핑 시 (None, None)."""
        entry = self._map.get(str(edi_code).strip())
        if entry is None:
            return None, None
        return entry.get("atc_code"), entry.get("drug_name")

    def standardize(self, df: pd.DataFrame, edi_col: str = "EDI_CD") -> pd.DataFrame:
        """
        DataFrame에 atc_code, drug_name 컬럼 추가.
        원본 EDI_CD 컬럼은 보존.
        """
        if edi_col not in df.columns:
            raise ValueError(f"컬럼 '{edi_col}' 없음")

        atc_codes = []
        drug_names = []
        for code in df[edi_col].astype(str):
            atc, name = self.lookup(code)
            atc_codes.append(atc)
            drug_names.append(name)

        out = df.copy()
        out["atc_code"] = atc_codes
        out["drug_name"] = drug_names
        return out

    def unknown_rate(self, df: pd.DataFrame, edi_col: str = "EDI_CD") -> float:
        """ATC 매핑 불가 비율 계산 (품질 지표용)."""
        if edi_col not in df.columns or len(df) == 0:
            return 0.0
        known = df[edi_col].astype(str).isin(self._map)
        return float((~known).mean())

    @property
    def mapping_count(self) -> int:
        return len(self._map)
