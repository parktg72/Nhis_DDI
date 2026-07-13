#!/usr/bin/env python3
"""
중복약물 탐지기

ATC 코드 기반 3단계 중복 탐지 + E1~E9 예외 허용 규칙 적용.

Level 1: ATC 5단계 동일 (완전 동일 성분) - 예외 없음
Level 2: ATC 4단계 동일 (동일 약리 소분류)
Level 3: ATC 3단계 동일 (동일 치료목적) - E1~E9 예외 적용

사용 예시:
  from rules.duplicate_detector import DuplicateDetector

  dd = DuplicateDetector()
  result = dd.detect(
      drugs=[
          {"name": "amlodipine", "atc": "C08CA01"},
          {"name": "nifedipine",  "atc": "C08CA05"},
          {"name": "aspirin",     "atc": "B01AC06"},
      ]
  )
  for dup in result.duplicates:
      print(dup)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

DEFAULT_RULES_PATH = Path(__file__).parent.parent / "config" / "drug_rules.yaml"
DEFAULT_EFCY_DUP_PATH = Path(__file__).parent.parent / "data" / "processed" / "efcy_duplicate_groups.parquet"


@dataclass
class DrugEntry:
    name: str
    atc_code: str            # ATC 5자리 코드 (예: C08CA01)
    edi_code: str = ""       # 건보 EDI 코드 (선택)

    @property
    def atc3(self) -> str:
        return self.atc_code[:4] if len(self.atc_code) >= 4 else self.atc_code

    @property
    def atc4(self) -> str:
        return self.atc_code[:5] if len(self.atc_code) >= 5 else self.atc_code

    @property
    def atc5(self) -> str:
        return self.atc_code[:7] if len(self.atc_code) >= 7 else self.atc_code


@dataclass
class DuplicatePair:
    drug_a: str
    drug_b: str
    atc_a: str
    atc_b: str
    level: int               # 1, 2, 3
    shared_atc: str          # 공통 ATC prefix
    exception_code: Optional[str]   # E1~E9 (예외 적용 시)
    is_allowed: bool         # 예외로 허용되면 True


@dataclass
class DuplicateResult:
    duplicate_level1_count: int = 0      # ATC 5단계 동일
    duplicate_level2_count: int = 0      # ATC 4단계 동일
    duplicate_level3_count: int = 0      # ATC 3단계 동일 (예외 적용 후)
    duplicates: list[DuplicatePair] = field(default_factory=list)
    exception_applied: list[str] = field(default_factory=list)   # 적용된 예외 코드

    @property
    def has_level1(self) -> bool:
        return self.duplicate_level1_count > 0

    @property
    def total_duplicates(self) -> int:
        return self.duplicate_level1_count + self.duplicate_level2_count + self.duplicate_level3_count

    def to_dict(self) -> dict:
        return {
            "duplicate_level1_count": self.duplicate_level1_count,
            "duplicate_level2_count": self.duplicate_level2_count,
            "duplicate_level3_count": self.duplicate_level3_count,
            "exception_applied": self.exception_applied,
        }


class DuplicateDetector:
    """
    ATC 코드 기반 중복약물 탐지기.

    ATC 코드가 없는 경우: 효능군중복(DUR) 데이터 보완 사용.
    """

    def __init__(
        self,
        rules_path: Path = DEFAULT_RULES_PATH,
        efcy_dup_path: Path = DEFAULT_EFCY_DUP_PATH,
    ):
        self._rules = self._load_rules(rules_path)
        self._exceptions = self._rules.get("duplicate_exceptions", {})
        self._efcy_groups = self._load_efcy_groups(efcy_dup_path)

    @staticmethod
    def _load_rules(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_efcy_groups(path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    def detect(self, drugs: list[dict]) -> DuplicateResult:
        """
        중복약물 탐지.

        Args:
            drugs: [{"name": "약물명", "atc": "ATC코드", "edi": "EDI코드(선택)"}, ...]

        Returns:
            DuplicateResult
        """
        result = DuplicateResult()

        entries = []
        for d in drugs:
            atc = str(d.get("atc", "")).strip().upper()
            if atc:
                entries.append(DrugEntry(
                    name=str(d.get("name", "")),
                    atc_code=atc,
                    edi_code=str(d.get("edi", "")),
                ))

        if len(entries) < 2:
            return result

        # 모든 약물 쌍 검사
        seen_pairs: set = set()

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a = entries[i]
                b = entries[j]

                pair_key = frozenset([a.atc_code, b.atc_code])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                dup = self._classify_pair(a, b)
                if dup is not None:
                    result.duplicates.append(dup)
                    if dup.level == 1:
                        result.duplicate_level1_count += 1
                    elif dup.level == 2:
                        result.duplicate_level2_count += 1
                    elif dup.level == 3 and not dup.is_allowed:
                        result.duplicate_level3_count += 1

                    if dup.exception_code and dup.exception_code not in result.exception_applied:
                        result.exception_applied.append(dup.exception_code)

        # 효능군중복 DUR 기반 보완 (ATC 코드 없는 경우)
        if not self._efcy_groups.empty:
            self._check_efcy_duplicates(drugs, entries, result)

        return result

    def _classify_pair(self, a: DrugEntry, b: DrugEntry) -> Optional[DuplicatePair]:
        """두 약물 간 중복 레벨 판정."""
        # Level 1: ATC 5단계 동일 (7자리)
        if a.atc5 and b.atc5 and a.atc5 == b.atc5 and len(a.atc5) >= 7:
            return DuplicatePair(
                drug_a=a.name, drug_b=b.name,
                atc_a=a.atc_code, atc_b=b.atc_code,
                level=1,
                shared_atc=a.atc5,
                exception_code=None,
                is_allowed=False,   # Level 1은 예외 없음
            )

        # Level 2: ATC 4단계 동일 (5자리)
        if a.atc4 and b.atc4 and a.atc4 == b.atc4 and len(a.atc4) >= 5:
            exc = self._find_exception(a.atc_code, level=2)
            return DuplicatePair(
                drug_a=a.name, drug_b=b.name,
                atc_a=a.atc_code, atc_b=b.atc_code,
                level=2,
                shared_atc=a.atc4,
                exception_code=exc,
                is_allowed=exc is not None,
            )

        # Level 3: ATC 3단계 동일 (4자리)
        if a.atc3 and b.atc3 and a.atc3 == b.atc3 and len(a.atc3) >= 4:
            exc = self._find_exception(a.atc_code, level=3)
            return DuplicatePair(
                drug_a=a.name, drug_b=b.name,
                atc_a=a.atc_code, atc_b=b.atc_code,
                level=3,
                shared_atc=a.atc3,
                exception_code=exc,
                is_allowed=exc is not None,
            )

        return None

    def _find_exception(self, atc_code: str, level: int) -> Optional[str]:
        """ATC 코드가 예외 허용 규칙에 해당하는지 확인."""
        for exc_code, exc_def in self._exceptions.items():
            for prefix in exc_def.get("atc_prefixes", []):
                prefix_str = str(prefix)
                if atc_code.startswith(prefix_str):
                    return exc_code
        return None

    def _check_efcy_duplicates(
        self,
        drugs: list[dict],
        entries: list[DrugEntry],
        result: DuplicateResult,
    ):
        """식약처 DUR 효능군중복 데이터 기반 추가 탐지."""
        if self._efcy_groups.empty:
            return

        # 약물코드(EDI) 기반으로 효능군 그룹 조회
        drug_edi_codes = {e.edi_code for e in entries if e.edi_code}
        if not drug_edi_codes:
            return

        # 효능군 그룹별로 중복 탐지
        class_col = "efcy_class_no" if "efcy_class_no" in self._efcy_groups.columns else None
        code_col = "drug_code" if "drug_code" in self._efcy_groups.columns else None

        if not class_col or not code_col:
            return

        # 처방 약물이 속한 효능군 조회
        drug_classes: dict[str, list[str]] = {}  # class_no → [drug_code, ...]
        for edi in drug_edi_codes:
            matches = self._efcy_groups[self._efcy_groups[code_col] == edi]
            for _, row in matches.iterrows():
                cls = str(row[class_col])
                drug_classes.setdefault(cls, []).append(edi)

        # 동일 효능군에 2개 이상인 경우 → Level 3 추가
        for cls, codes in drug_classes.items():
            if len(codes) >= 2:
                # 이미 ATC 기반으로 탐지된 쌍과 중복이 아닌지 확인
                dup_pair = DuplicatePair(
                    drug_a=codes[0], drug_b=codes[1],
                    atc_a="", atc_b="",
                    level=3,
                    shared_atc=f"효능군:{cls}",
                    exception_code=None,
                    is_allowed=False,
                )
                # DUR 기반이므로 예외 여부 보수적으로 판단
                result.duplicates.append(dup_pair)
                result.duplicate_level3_count += 1


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="중복약물 탐지기")
    parser.add_argument(
        "--drugs",
        required=True,
        help='JSON 형식 약물 목록 (예: \'[{"name":"amlodipine","atc":"C08CA01"},...]\')',
    )
    args = parser.parse_args()

    drugs = json.loads(args.drugs)
    dd = DuplicateDetector()
    result = dd.detect(drugs)

    print(f"[중복약물 탐지 결과]")
    print(f"  Level 1 (동일 성분): {result.duplicate_level1_count}건")
    print(f"  Level 2 (동일 약리 소분류): {result.duplicate_level2_count}건")
    print(f"  Level 3 (동일 치료목적, 예외 후): {result.duplicate_level3_count}건")
    print(f"  적용 예외: {result.exception_applied}")

    if result.duplicates:
        print("\n[상세 중복 목록]")
        for dup in result.duplicates:
            flag = "✅ 허용" if dup.is_allowed else "⚠️ 중복"
            exc = f" [{dup.exception_code}]" if dup.exception_code else ""
            print(f"  {flag} Level {dup.level}: {dup.drug_a} + {dup.drug_b} (ATC: {dup.shared_atc}){exc}")
