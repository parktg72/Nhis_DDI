"""Drug ontology helpers for DrugMaster-backed DDI lookup."""
from __future__ import annotations

from dataclasses import dataclass
import weakref

import pandas as pd

from .drug_master import DrugMaster
from .models import DrugOverlapPair

SEVERITY_ORDER = {"Contraindicated": 4, "Major": 3, "Moderate": 2, "Minor": 1}


def severity_rank(sev: str | None) -> int:
    """Return numeric rank for a DDI severity label."""
    return SEVERITY_ORDER.get(sev or "", 0)


def higher_severity(a: str | None, b: str | None) -> str | None:
    """Return the higher-ranked severity, preserving the selected label."""
    if severity_rank(b) > severity_rank(a):
        return b
    return a


@dataclass(frozen=True)
class DDIMatrixLookup:
    """Symmetric DDI severity lookup built from ddi_matrix_final columns."""

    pairs: dict[frozenset[str], str]

    @classmethod
    def from_matrix(cls, ddi_matrix: pd.DataFrame) -> "DDIMatrixLookup":
        ddi_lookup: dict[frozenset[str], str] = {}
        required = ("drug_a_id", "drug_b_id", "severity")
        if not all(c in ddi_matrix.columns for c in required):
            return cls(ddi_lookup)

        for row in ddi_matrix.itertuples(index=False):
            raw_a = getattr(row, "drug_a_id")
            raw_b = getattr(row, "drug_b_id")
            if pd.isna(raw_a) or pd.isna(raw_b):
                continue
            a_id = str(raw_a).strip()
            b_id = str(raw_b).strip()
            if not a_id or not b_id:
                continue
            raw_sev = getattr(row, "severity")
            if pd.isna(raw_sev):
                # NaN severity가 str()로 truthy "nan"이 되어 서빙 DDI 알림에
                # severity="nan"으로 노출되는 것 방지
                continue
            key = frozenset({a_id, b_id})
            new_sev = str(raw_sev)
            ddi_lookup[key] = higher_severity(ddi_lookup.get(key), new_sev) or new_sev
        return cls(ddi_lookup)

    def severity(self, id_a: str, id_b: str) -> str | None:
        a_id = str(id_a).strip()
        b_id = str(id_b).strip()
        if not a_id or not b_id:
            return None
        return self.pairs.get(frozenset({a_id, b_id}))

    def best_severity(self, ids_a: list[str], ids_b: list[str]) -> str | None:
        best: str | None = None
        for id_a in ids_a:
            for id_b in ids_b:
                best = higher_severity(best, self.severity(id_a, id_b))
        return best


@dataclass
class _LookupCacheEntry:
    frame_ref: weakref.ReferenceType[pd.DataFrame]
    lookup: DDIMatrixLookup


_lookup_cache: dict[int, _LookupCacheEntry] = {}


def clear_lookup_cache() -> None:
    _lookup_cache.clear()


def get_lookup(ddi_matrix: pd.DataFrame) -> DDIMatrixLookup:
    matrix_id = id(ddi_matrix)
    entry = _lookup_cache.get(matrix_id)
    if entry is not None and entry.frame_ref() is ddi_matrix:
        return entry.lookup

    lookup = DDIMatrixLookup.from_matrix(ddi_matrix)
    # 소멸 콜백으로 엔트리 자동 축출 — 없으면 죽은 DataFrame의 대형 lookup
    # (금기 매트릭스 ≈ 1.4M행)이 장수 프로세스에 무한 누적
    _lookup_cache[matrix_id] = _LookupCacheEntry(
        weakref.ref(ddi_matrix, lambda _r, _mid=matrix_id: _lookup_cache.pop(_mid, None)),
        lookup,
    )
    return lookup


class DrugOntology:
    """Facade combining DrugMaster component/ID resolution and DDI lookup."""

    def __init__(self, drug_master: DrugMaster, ddi_matrix: pd.DataFrame) -> None:
        self.drug_master = drug_master
        self.lookup = get_lookup(ddi_matrix)

    def components(self, wk_compn_cd: str) -> list[str]:
        return self.drug_master.get_components(wk_compn_cd)

    def ddi_ids(self, wk_compn_cd: str) -> list[str]:
        return self.drug_master.get_ddi_ids(wk_compn_cd)

    def pair_severity(self, wk_a: str, wk_b: str) -> str | None:
        ids_a = self.ddi_ids(wk_a)
        ids_b = self.ddi_ids(wk_b)
        if not (ids_a and ids_b):
            return None
        return self.lookup.best_severity(ids_a, ids_b)

    def pair_severities(
        self,
        pairs: list[DrugOverlapPair],
    ) -> list[tuple[DrugOverlapPair, str]]:
        out: list[tuple[DrugOverlapPair, str]] = []
        for pair in pairs:
            severity = self.pair_severity(pair.drug_a_wk_compn, pair.drug_b_wk_compn)
            if severity:
                out.append((pair, severity))
        return out
