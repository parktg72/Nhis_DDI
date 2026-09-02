"""
약물 코드 표준화.

기존: EDI(MCARE_DIV_CD) → ATC 코드 변환
신규: WK_COMPN_CD(주성분코드) → 성분명 → DDI ID 변환 (DrugMaster 통합)

DDI 분석 우선 경로 (주성분명 기반):
  WK_COMPN_CD → DrugMaster.get_components() → 성분명 목록
              → DrugMaster.get_ddi_ids()     → DDI 매트릭스 ID 목록

복합제 자동 처리:
  주성분코드 5-6번 == "00" → 성분명 2개 이상 반환
  각 성분에 대해 DDI 조회 수행
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .drug_master import DrugMaster

logger = logging.getLogger(__name__)


class CodeStandardizer:
    """
    약물 코드 → 성분명 / DDI ID 변환기.

    DrugMaster(HIRA 약제급여목록 기반)를 주요 조회 경로로 사용.
    EDI→ATC 매핑은 DrugBank 인덱스 기반 보조 경로로 유지.
    """

    def __init__(
        self,
        drug_master: DrugMaster | None = None,
        hira_xlsx: str | Path | None = None,
        ddi_matrix_path: str | Path = "data/processed/ddi_matrix_final.parquet",
        master_parquet: str | Path = "data/processed/hira_drug_master.parquet",
        # 레거시 EDI→ATC 경로 (DrugBank 기반)
        index_path: str | Path = "data/processed/drug_name_index.parquet",
        extra_csv: str | Path | None = "config/edi_atc_extra.csv",
        # EDI(제품코드)→WK(주성분코드) 브릿지 (Task B serving DDI parity)
        edi_wk_path: str | Path = "data/processed/edi_to_wk.parquet",
    ):
        # ── DrugMaster (성분명 기반 DDI) ────────────────────────────────────
        if drug_master is not None:
            self._master = drug_master
        elif Path(master_parquet).exists():
            self._master = DrugMaster.load_parquet(master_parquet, ddi_matrix_path)
        elif hira_xlsx and Path(hira_xlsx).exists():
            self._master = DrugMaster.from_files(hira_xlsx, ddi_matrix_path)
        else:
            logger.warning(
                "DrugMaster 초기화 실패: hira_xlsx=%s, master_parquet=%s",
                hira_xlsx, master_parquet,
            )
            self._master = DrugMaster()

        # ── 레거시: EDI → ATC 매핑 (DrugBank) ──────────────────────────────
        self._edi_map: dict[str, dict] = {}
        self._load_edi_index(Path(index_path))
        if extra_csv and Path(extra_csv).exists():
            self._load_extra(Path(extra_csv))

        # ── EDI(제품코드) → WK(주성분코드) + efmdc(약효분류) 브릿지 ─────────
        self._edi_wk: dict[str, str] = {}
        self._edi_efmdc: dict[str, str] = {}
        self._load_edi_wk(Path(edi_wk_path))

        logger.info(
            "CodeStandardizer 초기화: DrugMaster %d개 코드, EDI매핑 %d개, EDI→WK %d개",
            self._master.code_count,
            len(self._edi_map),
            len(self._edi_wk),
        )

    @staticmethod
    def _normalize_edi(value) -> str | None:
        """edi 를 9자리 문자열로 정규화 (build_edi_wk_map 과 동일 규칙)."""
        if value is None:
            return None
        s = str(value).strip()
        if s in ("", "nan", "None"):
            return None
        if s.endswith(".0"):
            s = s[:-2]
        if not s.isdigit():
            return None
        return s.zfill(9)

    def _load_edi_wk(self, path: Path) -> None:
        if not path.exists():
            logger.warning("edi→wk 맵 없음 — 서빙 DDI 미평가 위험: %s", path)
            return
        df = pd.read_parquet(path)
        if not {"edi_code", "wk_compn_cd"}.issubset(df.columns):
            logger.error("edi→wk 맵 컬럼 이상(%s): %s", path, list(df.columns))
            return
        for edi, wk in zip(df["edi_code"].astype(str), df["wk_compn_cd"].astype(str)):
            self._edi_wk[edi] = wk
        # efmdc(약효분류) 컬럼 — dup_efmdc 정합용 (구 맵엔 없을 수 있음)
        if "efmdc_clsf_no" in df.columns:
            for edi, ef in zip(df["edi_code"].astype(str), df["efmdc_clsf_no"]):
                if ef is not None and str(ef).strip() not in ("", "nan", "None"):
                    self._edi_efmdc[str(edi)] = str(ef).strip()

    def get_wk(self, edi_code: str) -> Optional[str]:
        """제품코드(EDI) → 주성분코드(WK). 미매핑 시 None."""
        norm = self._normalize_edi(edi_code)
        if norm is None:
            return None
        return self._edi_wk.get(norm)

    def get_efmdc(self, edi_code: str) -> Optional[str]:
        """제품코드(EDI) → 약효분류번호(efmdc_clsf_no). 미매핑 시 None."""
        norm = self._normalize_edi(edi_code)
        if norm is None:
            return None
        return self._edi_efmdc.get(norm)

    # ── 레거시 EDI 매핑 로딩 ─────────────────────────────────────────────────

    def _load_edi_index(self, path: Path) -> None:
        if not path.exists():
            return
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            edi = str(row.get("drug_id") or row.get("drugbank_id") or "").strip()
            atc = str(row.get("atc_codes") or row.get("atc_code") or "").strip() or None
            name = str(row.get("name") or row.get("drug_name") or "").strip() or None
            if edi:
                self._edi_map[edi] = {"atc_code": atc, "drug_name": name}

    def _load_extra(self, path: Path) -> None:
        df = pd.read_csv(path, dtype=str)
        if not {"edi_code", "atc_code"}.issubset(df.columns):
            return
        for _, row in df.iterrows():
            edi = str(row["edi_code"]).strip()
            atc = str(row["atc_code"]).strip() or None
            name = str(row.get("drug_name", "") or "").strip() or None
            if edi:
                self._edi_map[edi] = {"atc_code": atc, "drug_name": name}

    # ── 주성분코드 기반 공개 API (신규) ──────────────────────────────────────

    def get_components(self, wk_compn_cd: str) -> list[str]:
        """
        WK_COMPN_CD → 정규화 성분명 목록.

        복합제(5-6번=00): 여러 성분 반환
        단일제:           1개 성분 반환
        미등록:           빈 리스트
        """
        return self._master.get_components(wk_compn_cd)

    def get_ddi_ids(self, wk_compn_cd: str) -> list[str]:
        """
        WK_COMPN_CD → DDI 매트릭스 ID 목록.

        복합제는 각 성분의 DDI ID를 모두 반환.
        DDI 조회 시 이 ID 목록을 cross-product로 사용.
        """
        return self._master.get_ddi_ids(wk_compn_cd)

    def is_combination(self, wk_compn_cd: str) -> bool:
        """복합제 여부 (주성분코드 5-6번 == '00')."""
        return self._master.is_combination(wk_compn_cd)

    def expand_drug_count(self, wk_codes: list[str]) -> int:
        """
        복합제를 성분별로 전개한 고유 성분 수 계산.
        drug_count 피처 계산에 사용.
        """
        return len(self._master.expand_drug_count(wk_codes))

    # ── 레거시 EDI→ATC API (보조) ────────────────────────────────────────────

    def lookup_edi(self, edi_code: str) -> tuple[Optional[str], Optional[str]]:
        """EDI 코드 → (atc_code, drug_name). DrugBank 인덱스 기반."""
        entry = self._edi_map.get(str(edi_code).strip())
        if entry is None:
            return None, None
        return entry.get("atc_code"), entry.get("drug_name")

    def lookup_wk(self, wk_compn_cd: str) -> tuple[Optional[str], Optional[str]]:
        """WK_COMPN_CD 기반으로 ATC 코드 및 약물명 조회 (EDI 매핑 실패 시 폴백)."""
        if not wk_compn_cd:
            return None, None

        # 1. 주성분코드 → DDI ID(들) 조회
        ddi_ids = self._master.get_ddi_ids(wk_compn_cd)
        if not ddi_ids:
            return None, None

        # 2. 모든 ddi_id에 대해 매핑되는 ATC 및 약물명 수집
        #
        # RS1 — `_edi_map` 은 DrugBank ID 로 키잉된다(DB 19,844 · D 41). 그런데
        # 성분→DDI ID 인덱스가 참조하는 DUR D-코드는 314개이고 그중 275개가
        # `_edi_map` 에 없다. 종전에는 그 275개가 이름 없이 버려져 spironolactone·
        # amiodarone·methotrexate·clarithromycin·statin 같은 흔한 약물이 통째로
        # 무발화했다(실측: 하루치에 처방되고 성분 파싱도 되는데 해소 0건).
        # DDI 매트릭스가 그 ID 의 대표 성분명을 이미 갖고 있으므로 그것을 쓴다.
        # ATC 는 그 출처에 없으므로 채우지 않는다 — `standardize()` 의 `if atc_fb:`
        # 가드가 ATC 로만 폴백을 채택하므로 ETL 산출물은 바뀌지 않는다.
        atc_list = []
        name_list = []
        for dbid in ddi_ids:
            entry = self._edi_map.get(dbid)
            if entry:
                atc = entry.get("atc_code")
                name = entry.get("drug_name")
                if atc:
                    atc_list.append(atc)
                if name:
                    name_list.append(name)
                    continue
            fallback = self._master.name_for_ddi_id(dbid)
            if fallback:
                name_list.append(fallback)

        # ATC 유무로 반환 전체를 게이팅하지 않는다. 성분은 해소됐는데 그 인덱스
        # 엔트리에 ATC 가 없는 경우가 실제로 존재하며(실측: 하루치 고유 EDI 15,017개
        # 중 429건, 예: aspirin/DB D000452), 종전에는 확보된 약물명까지 함께 버려졌다.
        # 약물명은 이름 기반 Safety Net 규칙의 유일한 입력이므로 이 폐기는 곧 무발화다.
        if atc_list or name_list:
            return (",".join(atc_list) if atc_list else None,
                    name_list[0] if name_list else None)

        return None, None

    def lookup_wk_names(self, wk_compn_cd: str) -> list[str]:
        """WK_COMPN_CD 의 **모든** 성분명. 복합제에서 성분이 소실되지 않게 한다.

        RS2 — `lookup_wk` 는 `DrugItem.drug_name` 이 스칼라이므로 첫 성분만
        돌려준다. 복합제에서는 그것이 병용 성분일 수 있다(실측: 스타틴 복합제
        123건 중 해소에 성공한 75건 **전량**이 스타틴 아닌 이름으로 해소됐다 —
        `[rosuvastatin, losartan, amlodipine]` → `Losartan`). 이름 기반 규칙은
        목록을 입력으로 받으므로, 규칙 경로에는 전 성분을 넘긴다.

        `lookup_wk` 의 반환 계약은 건드리지 않는다.
        """
        if not wk_compn_cd:
            return []

        names: list[str] = []
        for dbid in self._master.get_ddi_ids(wk_compn_cd):
            entry = self._edi_map.get(dbid)
            name = entry.get("drug_name") if entry else None
            if not name:
                name = self._master.name_for_ddi_id(dbid)
            if name and name not in names:
                names.append(name)
        return names

    def standardize(self, df: pd.DataFrame, edi_col: str = "MCARE_DIV_CD") -> pd.DataFrame:
        """
        DataFrame에 atc_code, drug_name 컬럼 추가 (EDI→ATC 레거시 경로).
        원본 컬럼은 보존.
        """
        if edi_col not in df.columns:
            raise ValueError(f"컬럼 '{edi_col}' 없음")

        wk_col = None
        for col in ("WK_COMPN_CD", "RVSN_WK_COMPN_CD"):
            if col in df.columns:
                wk_col = col
                break

        atc_codes, drug_names = [], []
        if wk_col:
            for edi, wk in zip(df[edi_col].astype(str), df[wk_col].astype(str)):
                atc, name = self.lookup_edi(edi)
                if not atc or atc == "None" or atc == "nan":
                    atc_fb, name_fb = self.lookup_wk(wk)
                    if atc_fb:
                        atc, name = atc_fb, name_fb
                atc_codes.append(atc)
                drug_names.append(name)
        else:
            for code in df[edi_col].astype(str):
                atc, name = self.lookup_edi(code)
                atc_codes.append(atc)
                drug_names.append(name)

        out = df.copy()
        out["atc_code"] = atc_codes
        out["drug_name"] = drug_names
        return out

    def unknown_rate(self, df: pd.DataFrame, wk_col: str = "WK_COMPN_CD") -> float:
        """DDI 매핑 불가 비율 (품질 지표용)."""
        if wk_col not in df.columns or len(df) == 0:
            return 0.0
        mapped = df[wk_col].astype(str).apply(
            lambda c: bool(self._master.get_ddi_ids(c))
        )
        return float((~mapped).mean())

    @property
    def drug_master(self) -> DrugMaster:
        return self._master

    @property
    def mapping_count(self) -> int:
        return self._master.code_count
