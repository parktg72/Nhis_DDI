"""
HIRA 약제급여목록 기반 약물 마스터 테이블.

핵심 역할:
  WK_COMPN_CD(9자리 주성분코드) → 성분명 목록 → DDI 매트릭스 ID 변환

복합제 처리:
  - 주성분코드 5-6번 == "00" → 복합제 판정
  - 주성분명을 쉼표 구분으로 파싱 → 개별 성분명 추출
  - 각 성분명을 정규화 후 DDI 매트릭스의 drug_id와 매핑

DDI 조회 설계:
  1. WK_COMPN_CD → get_components() → [norm_name1, norm_name2, ...]
  2. norm_name   → get_ddi_id()    → drug_id (D-코드 or DB-코드)
  3. drug_id 쌍  → DDI 매트릭스   → severity
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 성분명 정규화: 염 · 수화물 접미사 제거
# ─────────────────────────────────────────────────────────────────────────────
_SALT_RE = re.compile(
    r"\s+(?:hydrochloride|hcl|sodium|potassium|magnesium|calcium|zinc|"
    r"aluminum|aluminium|sulfate|sulphate|acetate|tartrate|maleate|"
    r"fumarate|succinate|citrate|gluconate|mesylate|mesilate|tosylate|"
    r"besylate|bromide|phosphate|carbonate|bicarbonate|lactate|"
    r"monohydrate|dihydrate|trihydrate|hemihydrate|sesquihydrate|"
    r"tetrahydrate|pentahydrate|hexahydrate|anhydrous|anhydrate|hydrate|"
    r"monosodium|disodium|hemisulfate|hemifumarate|strontium|ammonium|"
    r"chloride|monohydrochloride|dihydrochloride|tromethamine|"
    r"trifenatate|propionate|valerate|dipropionate|pivalate|"
    r"dimethylsulfoxide|decanoate|enanthate|undecylenate)$",
    re.IGNORECASE,
)
_DOSE_RE = re.compile(
    r"\s+[\d][\d\.,]*\s*(?:mg|g|mL|mcg|%|IU|unit|U|mmol|mEq|ug|μg|μ)[^\,]*$",
    re.IGNORECASE,
)
# "(as BASENAME ...)" 패턴: base name 추출 우선
_AS_BASE_RE = re.compile(r"\(as\s+([a-zA-Z][a-zA-Z\s\-]+?)(?:\s+[\d\(].*?)?\)", re.IGNORECASE)
_PAREN_RE = re.compile(r"\s*\([^\)]*\)?", re.IGNORECASE)


def _normalize_name(raw: str) -> str:
    """
    성분명 정규화.
    예) "tramadol hydrochloride   37.5mg"            → "tramadol"
        "esomeprazole magnesium (as esomeprazole 20mg)" → "esomeprazole"
        "carbidopa hydrate (as carbidopa   25mg)"    → "carbidopa"
    """
    s = str(raw).strip()
    # "(as BASENAME ...)" 패턴에서 기본 성분명 추출
    m = _AS_BASE_RE.search(s)
    if m:
        base = m.group(1).strip()
        # base에도 salt/hydrate 접미사 제거 적용
        for _ in range(3):
            base2 = _SALT_RE.sub("", base.strip())
            if base2 == base:
                break
            base = base2
        return base.strip().lower()
    # 용량 제거
    s = _DOSE_RE.sub("", s)
    # 남은 괄호 제거
    s = _PAREN_RE.sub("", s)
    # 접미사 반복 제거 (hydrochloride, sodium, ...)
    for _ in range(3):
        s2 = _SALT_RE.sub("", s.strip())
        if s2 == s:
            break
        s = s2
    return s.strip().lower()


def _parse_ingredients(name_str: str) -> list[str]:
    """
    복합제 주성분명 파싱 → 정규화된 성분명 목록.
    쉼표로 구분된 각 성분에서 용량·염 제거.

    예) "ergotamine tartrate   1mg, caffeine anhydrous   0.1g"
        → ["ergotamine", "caffeine"]
    """
    # 쉼표 뒤에 알파벳/한글이 이어지는 위치에서 분리
    parts = re.split(r",\s*(?=[a-zA-Z가-힣\(])", str(name_str))
    results = []
    for p in parts:
        norm = _normalize_name(p)
        if norm and len(norm) > 2:
            results.append(norm)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# DrugMaster 클래스
# ─────────────────────────────────────────────────────────────────────────────
class DrugMaster:
    """
    HIRA 약제급여목록 + DDI 매트릭스 기반 약물 마스터.

    사용 예::
        master = DrugMaster.from_files(
            hira_xlsx="hira/약제급여목록및급여상한금액표.xlsx",
            ddi_matrix="data/processed/ddi_matrix_final.parquet",
        )
        # 복합제
        master.get_components("251800ATB")     # → ["ergotamine", "caffeine"]
        master.get_ddi_ids("251800ATB")        # → ["D000747", "D001266"]
        # 단일제
        master.get_components("A00100100")     # → ["simvastatin"]
        master.get_ddi_ids("A00100100")        # → ["D000027"]
    """

    def __init__(self) -> None:
        # ingr_code → 정규화 성분명 목록
        self._code_to_components: dict[str, list[str]] = {}
        # ingr_code → 원본 주성분명
        self._code_to_raw: dict[str, str] = {}
        # 복합제 여부
        self._code_is_combo: dict[str, bool] = {}
        # 정규화 성분명 → DDI ID (D-코드 or DB-코드)
        self._name_to_ddi_id: dict[str, str] = {}
        # DDI ID → 대표 성분명
        self._ddi_id_to_name: dict[str, str] = {}

    # ── 로딩 ─────────────────────────────────────────────────────────────────

    @classmethod
    def from_files(
        cls,
        hira_xlsx: str | Path = "hira/약제급여목록및급여상한금액표.xlsx",
        ddi_matrix_path: str | Path = "data/processed/ddi_matrix_final.parquet",
    ) -> "DrugMaster":
        master = cls()
        master._load_hira(Path(hira_xlsx))
        if Path(ddi_matrix_path).exists():
            master._build_ddi_name_index(Path(ddi_matrix_path))
        master._apply_synonyms()
        logger.info(
            "DrugMaster 로드 완료: %d개 주성분코드, %d개 DDI 성분명 인덱스",
            len(master._code_to_components),
            len(master._name_to_ddi_id),
        )
        return master

    def _load_hira(self, path: Path) -> None:
        if not path.exists():
            logger.warning("HIRA 약제급여목록 파일 없음: %s", path)
            return

        df = pd.read_excel(path, sheet_name=0, dtype=str)
        # 컬럼명 정규화 (줄바꿈 제거)
        df.columns = [c.replace("\n", "") for c in df.columns]

        code_col = "주성분코드"
        name_col = "주성분명"
        if code_col not in df.columns or name_col not in df.columns:
            logger.error("HIRA 파일 컬럼 없음: %s", list(df.columns))
            return

        df[code_col] = df[code_col].fillna("").str.strip()
        df[name_col] = df[name_col].fillna("").str.strip()

        # 주성분코드 기준 중복 제거 (같은 코드가 여러 제품)
        unique = df.drop_duplicates(subset=code_col)

        for _, row in unique.iterrows():
            code = row[code_col]
            raw_name = row[name_col]
            if not code:
                continue

            is_combo = len(code) >= 6 and code[4:6] == "00"
            components = _parse_ingredients(raw_name)

            self._code_to_components[code] = components
            self._code_to_raw[code] = raw_name
            self._code_is_combo[code] = is_combo

        n_combo = sum(v for v in self._code_is_combo.values())
        logger.info("HIRA 로드: 전체 %d개 (복합제 %d개)", len(self._code_to_components), n_combo)

    def _build_ddi_name_index(self, path: Path) -> None:
        """
        ddi_matrix_final.parquet의 약물명 → DDI ID 역방향 인덱스 구성.

        컬럼: drug_a_name, drug_b_name, drug_a_id, drug_b_id
        """
        df = pd.read_parquet(path)
        pairs = [("drug_a_name", "drug_a_id"), ("drug_b_name", "drug_b_id")]
        for name_col, id_col in pairs:
            if name_col not in df.columns or id_col not in df.columns:
                continue
            sub = df[[name_col, id_col]].dropna().drop_duplicates(name_col)
            for _, row in sub.iterrows():
                raw_name = str(row[name_col])
                drug_id = str(row[id_col]).strip()
                norm = _normalize_name(raw_name)
                if norm and drug_id:
                    # 먼저 등록된 것 유지 (DUR D-코드 우선)
                    if norm not in self._name_to_ddi_id:
                        self._name_to_ddi_id[norm] = drug_id
                    if drug_id not in self._ddi_id_to_name:
                        self._ddi_id_to_name[drug_id] = raw_name

    # ── 동의어 매핑 (HIRA 성분명 → DrugBank 표준명) ─────────────────────────
    # HIRA 급여목록의 성분명이 DrugBank 표준명과 다른 경우 매핑
    _SYNONYMS: dict[str, str] = {
        "diphenylhydantoin": "phenytoin",
        "sodium valproate": "valproic acid",
        "valproate semisodium": "valproic acid",
        "valproate": "valproic acid",
        "divalproex": "valproic acid",
        "levothyroxine sodium": "levothyroxine",
        "thyroxine": "levothyroxine",
        "acetylsalicylic acid": "aspirin",
        "salicylic acid acetyl": "aspirin",
        "paracetamol": "acetaminophen",
        "metamizole": "dipyrone",
        "methylprednisolone sodium succinate": "methylprednisolone",
        "prednisolone sodium phosphate": "prednisolone",
        "dexamethasone sodium phosphate": "dexamethasone",
        "thiamine": "vitamin b1",
        "pyridoxine": "vitamin b6",
        "cobalamin": "vitamin b12",
        "cyanocobalamin": "vitamin b12",
        "ergocalciferol": "vitamin d2",
        "cholecalciferol": "vitamin d3",
        "retinol": "vitamin a",
        "tocopherol": "vitamin e",
        "phytomenadione": "phytonadione",
        "glyceryl trinitrate": "nitroglycerin",
        "salbutamol": "albuterol",
        "frusemide": "furosemide",
        "ciclosporin": "cyclosporine",
        "nifedipine retard": "nifedipine",
        "glibenclamide": "glyburide",
        "bendroflumethiazide": "bendrofluazide",
        "co-amoxiclav": "amoxicillin",
        "sultamicillin": "ampicillin",
    }

    def _apply_synonyms(self) -> None:
        """동의어 테이블로 미매핑 성분명 → DDI ID 보완."""
        for code, components in self._code_to_components.items():
            for i, comp in enumerate(components):
                if comp not in self._name_to_ddi_id and comp in self._SYNONYMS:
                    synonym = self._SYNONYMS[comp]
                    if synonym in self._name_to_ddi_id:
                        self._name_to_ddi_id[comp] = self._name_to_ddi_id[synonym]

    def _repair_cached_components_from_raw_names(self) -> int:
        """
        오래된 parquet 캐시에 남은 성분 파싱 결과를 원본명 기준으로 보수적으로 보정.

        저장된 `components`보다 현재 parser로 `ingr_name_raw`를 다시 파싱한 결과가
        더 많은 DDI ID에 매핑될 때만 교체한다. 보호된 parquet artifact는 수정하지
        않고, 로드된 메모리 상태만 보정한다.
        """
        repaired = 0
        for code, raw_name in self._code_to_raw.items():
            reparsed = _parse_ingredients(raw_name)
            if not reparsed:
                continue
            current = self._code_to_components.get(code, [])
            current_ids = {ddi_id for comp in current if (ddi_id := self.get_ddi_id(comp))}
            reparsed_ids = {ddi_id for comp in reparsed if (ddi_id := self.get_ddi_id(comp))}
            reparsed_names = set(reparsed)
            current_components_are_represented = all(
                comp in reparsed_names
                or _normalize_name(comp) in reparsed_names
                or (self.get_ddi_id(comp) is not None and self.get_ddi_id(comp) in reparsed_ids)
                for comp in current
            )
            if not current_components_are_represented:
                continue
            if current_ids < reparsed_ids:
                self._code_to_components[code] = reparsed
                repaired += 1
        return repaired

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def is_combination(self, wk_compn_cd: str) -> bool:
        """복합제 여부 (5-6번 자리 == '00')."""
        code = str(wk_compn_cd).strip()
        if code in self._code_is_combo:
            return self._code_is_combo[code]
        return len(code) >= 6 and code[4:6] == "00"

    def get_components(self, wk_compn_cd: str) -> list[str]:
        """
        WK_COMPN_CD → 정규화된 성분명 목록.

        복합제: 2개 이상 반환 (예: ["ergotamine", "caffeine"])
        단일제: 1개 반환   (예: ["simvastatin"])
        미등록: 빈 리스트 []
        """
        code = str(wk_compn_cd).strip()
        return list(self._code_to_components.get(code, []))

    def get_raw_name(self, wk_compn_cd: str) -> Optional[str]:
        """원본 주성분명 반환."""
        return self._code_to_raw.get(str(wk_compn_cd).strip())

    def get_ddi_id(self, component_name: str) -> Optional[str]:
        """정규화된 성분명 → DDI 매트릭스 ID (D-코드 or DB-코드)."""
        return self._name_to_ddi_id.get(component_name.strip().lower())

    def get_ddi_ids(self, wk_compn_cd: str) -> list[str]:
        """
        WK_COMPN_CD → DDI 매트릭스 ID 목록.

        복합제의 경우 각 성분의 ID를 모두 반환.
        매핑 불가 성분은 제외.
        """
        components = self.get_components(wk_compn_cd)
        ids = []
        for name in components:
            ddi_id = self.get_ddi_id(name)
            if ddi_id:
                ids.append(ddi_id)
        return ids

    def get_component_id_pairs(self, wk_compn_cd: str) -> list[tuple[str, str]]:
        """(성분명, DDI_ID) 쌍 목록 반환."""
        components = self.get_components(wk_compn_cd)
        return [(n, self.get_ddi_id(n)) for n in components if self.get_ddi_id(n)]

    def expand_drug_count(self, wk_codes: list[str]) -> set[str]:
        """
        WK_COMPN_CD 목록 → 고유 성분명 집합 (복합제 전개).

        drug_count 피처 계산 시 복합제를 성분별로 분리하여 카운트.
        """
        unique_components: set[str] = set()
        for code in wk_codes:
            comps = self.get_components(code)
            if comps:
                unique_components.update(comps)
            else:
                # HIRA에 없는 코드는 코드 자체를 성분으로 취급
                unique_components.add(code)
        return unique_components

    @property
    def code_count(self) -> int:
        return len(self._code_to_components)

    @property
    def ddi_index_count(self) -> int:
        return len(self._name_to_ddi_id)

    # ── Parquet 저장/로드 ─────────────────────────────────────────────────────

    def save_parquet(
        self,
        path: str | Path = "data/processed/hira_drug_master.parquet",
    ) -> None:
        """
        HIRA 마스터 테이블을 parquet으로 저장.
        컬럼: ingr_code, is_combo, ingr_name_raw, components (리스트→문자열)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for code, comps in self._code_to_components.items():
            rows.append({
                "ingr_code":     code,
                "is_combo":      self._code_is_combo.get(code, False),
                "ingr_name_raw": self._code_to_raw.get(code, ""),
                "components":    "|".join(comps),   # 리스트 → 파이프 구분 문자열
                "ingr_count":    len(comps),
            })
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        logger.info("hira_drug_master.parquet 저장: %d행 → %s", len(df), path)

    @classmethod
    def load_parquet(
        cls,
        parquet_path: str | Path = "data/processed/hira_drug_master.parquet",
        ddi_matrix_path: str | Path = "data/processed/ddi_matrix_final.parquet",
    ) -> "DrugMaster":
        """
        저장된 parquet에서 빠르게 로드 (xlsx 파싱 없이).
        단, DDI 인덱스는 ddi_matrix에서 다시 구성.
        """
        master = cls()
        path = Path(parquet_path)
        if not path.exists():
            logger.warning("hira_drug_master.parquet 없음, xlsx에서 새로 빌드 필요")
            return master

        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            code = str(row["ingr_code"])
            raw_components = row["components"]
            if pd.isna(raw_components):
                comps = []
            else:
                comps = [c for c in str(raw_components).split("|") if c]
            master._code_to_components[code] = comps
            master._code_to_raw[code] = str(row.get("ingr_name_raw", ""))
            master._code_is_combo[code] = bool(row.get("is_combo", False))

        if Path(ddi_matrix_path).exists():
            master._build_ddi_name_index(Path(ddi_matrix_path))

        master._apply_synonyms()
        repaired = master._repair_cached_components_from_raw_names()
        if repaired:
            logger.info("DrugMaster cached components 보정: %d개 코드", repaired)

        logger.info("DrugMaster 로드 (parquet): %d개 코드", len(master._code_to_components))
        return master


# ─────────────────────────────────────────────────────────────────────────────
# 빌드 스크립트 (직접 실행 시)
# ─────────────────────────────────────────────────────────────────────────────
def build_and_save(
    hira_xlsx: str = "hira/약제급여목록및급여상한금액표.xlsx",
    ddi_matrix: str = "data/processed/ddi_matrix_final.parquet",
    out_parquet: str = "data/processed/hira_drug_master.parquet",
) -> DrugMaster:
    master = DrugMaster.from_files(hira_xlsx=hira_xlsx, ddi_matrix_path=ddi_matrix)
    master.save_parquet(out_parquet)

    # 매핑률 리포트
    total = master.code_count
    mapped = sum(
        1 for code in master._code_to_components
        if master.get_ddi_ids(code)
    )
    combo_total = sum(1 for v in master._code_is_combo.values() if v)
    combo_mapped = sum(
        1 for code, is_combo in master._code_is_combo.items()
        if is_combo and master.get_ddi_ids(code)
    )

    print(f"\n=== DrugMaster 빌드 결과 ===")
    print(f"전체 주성분코드:   {total:>6,}개")
    print(f"DDI 매핑 성공:    {mapped:>6,}개  ({mapped/total*100:.1f}%)")
    print(f"복합제:           {combo_total:>6,}개")
    print(f"복합제 DDI 매핑:  {combo_mapped:>6,}개  ({combo_mapped/combo_total*100:.1f}%)" if combo_total else "")
    print(f"DDI 인덱스 크기:  {master.ddi_index_count:>6,}개 성분명")
    print(f"저장 위치: {out_parquet}")
    return master


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).parent.parent.parent
    build_and_save(
        hira_xlsx=str(root / "hira/약제급여목록및급여상한금액표.xlsx"),
        ddi_matrix=str(root / "data/processed/ddi_matrix_final.parquet"),
        out_parquet=str(root / "data/processed/hira_drug_master.parquet"),
    )
