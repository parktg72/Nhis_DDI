"""
처방 패턴 집계
T20+T30+T40+T50 조인 결과 → 환자별 90일 윈도우 피처 집계

집계 항목:
- 고유 약물 수 (drug_count)
- 처방 기관 수 (institution_count)
- 동시복용 피크 수 (max_concurrent)
- DDI 심각도별 카운트
- 중복약물 레벨별 카운트
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from .clinical_rules import collect_red_triggers, collect_yellow_triggers
from .drug_master import DrugMaster
from .models import DrugOverlapPair, PatientFeatures, PrescriptionRecord
from .overlap_calculator import calculate_overlaps_for_patient, get_concurrent_drug_count

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 고위험/신기능/간기능 저하 위험 약물 키워드 및 ATC prefix
# 단일 출처: rules/risk_drug_constants.py (Codex 2026-05-06 ISSUE-3 단일화).
# 기준: CLINICAL_STANDARDS_v1.0.md + drug_rules.yaml :123 high_risk_drugs.
# ─────────────────────────────────────────────────────────────────────────────

from rules.risk_drug_constants import (
    HIGH_RISK_KEYWORDS as _HIGH_RISK_KEYWORDS,
    HIGH_RISK_ATC_PREFIXES as _HIGH_RISK_ATC_PREFIXES,
    RENAL_RISK_KEYWORDS as _RENAL_RISK_KEYWORDS,
    RENAL_RISK_ATC_PREFIXES as _RENAL_RISK_ATC_PREFIXES,
    HEPATIC_RISK_KEYWORDS as _HEPATIC_RISK_KEYWORDS,
    HEPATIC_RISK_ATC_PREFIXES as _HEPATIC_RISK_ATC_PREFIXES,
)


def _check_risk_drugs(
    prescriptions: list[PrescriptionRecord],
    keywords: set[str],
    atc_prefixes: tuple[str, ...],
    drug_master: DrugMaster | None = None,
) -> bool:
    """처방 목록에서 특정 위험 약물 포함 여부 (성분 + 이름 + ATC 매칭).

    학습 records 는 drug_name/atc_code 가 없어(df_row_to_record) 이름·ATC 경로는
    프로덕션서 dead. → DrugMaster.get_components(wk) 성분명 키워드 매칭이 실 경로
    (Phase 2-3). 학습·서빙(향후 edi→wk) 공용 식별자(wk→components). 키워드는 기존
    risk_drug_constants 단일출처(새 정의 아님 — 식별자만 수정).
    """
    for p in prescriptions:
        if drug_master is not None and p.wk_compn_cd:
            for comp in drug_master.get_components(p.wk_compn_cd):
                c = comp.lower()
                if any(kw in c for kw in keywords):
                    return True
        name = (p.drug_name or "").lower()
        if name and any(kw in name for kw in keywords):
            return True
        atc = p.atc_code or ""
        if atc and atc.startswith(atc_prefixes):
            return True
    return False


def _fill_risk_drug_flags(
    features: PatientFeatures,
    prescriptions: list[PrescriptionRecord],
    drug_master: DrugMaster | None = None,
) -> None:
    """고위험/신기능/간기능 저하 위험 약물 포함 여부 플래그 설정 (성분 키워드, Phase 2-3)."""
    features.has_high_risk_drug = _check_risk_drugs(
        prescriptions, _HIGH_RISK_KEYWORDS, _HIGH_RISK_ATC_PREFIXES, drug_master,
    )
    features.has_renal_risk_drug = _check_risk_drugs(
        prescriptions, _RENAL_RISK_KEYWORDS, _RENAL_RISK_ATC_PREFIXES, drug_master,
    )
    features.has_hepatic_risk_drug = _check_risk_drugs(
        prescriptions, _HEPATIC_RISK_KEYWORDS, _HEPATIC_RISK_ATC_PREFIXES, drug_master,
    )


def _parse_date(s: str) -> Optional[date]:
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except (ValueError, TypeError):
        return None


def _calc_age(birth_year: str, window_end: date) -> Optional[int]:
    try:
        return window_end.year - int(birth_year)
    except (ValueError, TypeError):
        return None


def aggregate_patient_features(
    patient_id: str,
    prescriptions: list[PrescriptionRecord],
    overlap_pairs: list[DrugOverlapPair],
    ddi_matrix: pd.DataFrame | None,
    dup_groups: pd.DataFrame | None,
    age: int | None = None,
    sex: str | None = None,
    addr_cd: str | None = None,
    window_start: date | None = None,
    window_end: date | None = None,
    drug_master: DrugMaster | None = None,
    cyp_extractor=None,
) -> PatientFeatures:
    """
    단일 환자의 피처 벡터 계산.

    Parameters
    ----------
    ddi_matrix   : ddi_matrix_final.parquet (drug_a_id, drug_b_id, severity)
    dup_groups   : efcy_duplicate_groups.parquet (drug_code, efcy_class_no)
    drug_master  : DrugMaster 인스턴스 (복합제 성분 전개 및 DDI ID 매핑)
    """
    if not prescriptions:
        return PatientFeatures(
            patient_id=patient_id,
            window_start=window_start or date.today(),
            window_end=window_end or date.today(),
            age=age, sex=sex, addr_cd=addr_cd,
        )

    # 윈도우 결정 (처방 최소/최대 날짜)
    all_starts = [p.start_date for p in prescriptions]
    all_ends = [p.end_date for p in prescriptions]
    w_start = window_start or min(all_starts)
    # days=90: w_start 당일을 1일째로 계산하는 임상 관습 기준 (1~90일 = 90일 윈도우)
    # days=89: w_start를 0일째(포함)로 계산 시 동일한 90일 구간
    # CLINICAL_STANDARDS_v1.0.md §1.3 "90일 윈도우"는 시작일 포함 90일 의미 → days=89
    w_end = window_end or min(max(all_ends), w_start + timedelta(days=89))

    features = PatientFeatures(
        patient_id=patient_id,
        window_start=w_start,
        window_end=w_end,
        age=age,
        sex=sex,
        addr_cd=addr_cd,
    )

    # ── 기본 피처 ──────────────────────────────────────────────────────────
    # 고유 약물 수: 복합제는 성분별로 전개하여 카운트
    unique_wk = [p.wk_compn_cd for p in prescriptions]
    if drug_master is not None:
        features.drug_count = len(drug_master.expand_drug_count(unique_wk))
    else:
        features.drug_count = len(set(unique_wk))

    unique_insts = {p.institution_id for p in prescriptions if p.institution_id}
    features.institution_count = len(unique_insts)

    # 7일 내 동시 복용 수 (기준: 윈도우 종료일)
    features.drug_count_7d = get_concurrent_drug_count(prescriptions, w_end)

    # ── DDI 피처 ────────────────────────────────────────────────────────────
    if ddi_matrix is not None and overlap_pairs:
        _fill_ddi_features(features, overlap_pairs, ddi_matrix, drug_master)

    # ── 중복약물 피처 ────────────────────────────────────────────────────────
    if dup_groups is not None:
        _fill_dup_features(features, prescriptions, dup_groups, drug_master)

    # ── Triple Whammy (ACEi/ARB+K이뇨제+NSAID, 성분 키워드) ───────────────────
    # 직전까지 미계산(항상 False)이던 것을 산출. Red 트리거(collect_red_triggers)에 들어가
    # 라벨이 바뀌므로 **다음 재학습부터 반영**(기존 배포모델엔 inert). 서빙 활성화는
    # 재학습·배포 후(현재 서빙 모델피처 triple_whammy 는 배포모델=0 과 맞춰 0 유지).
    features.triple_whammy = detect_triple_whammy(unique_wk, drug_master)

    # ── 고위험/신기능/간기능 약물 플래그 ─────────────────────────────────────
    _fill_risk_drug_flags(features, prescriptions, drug_master)

    # ── CYP450 피처 ────────────────────────────────────────────────────────
    if cyp_extractor is not None:
        atc_codes = [p.atc_code for p in prescriptions if p.atc_code]
        if atc_codes:
            cyp_feat = cyp_extractor.extract(atc_codes)
            features.cyp_risk_score = cyp_feat.get("cyp_risk_score", 0.0)
            features.cyp_max_enzyme_risk = cyp_feat.get("cyp_max_enzyme_risk", 0.0)
            features.cyp_high_risk_pairs = int(cyp_feat.get("cyp_high_risk_pairs", 0))

    # ── 위험도 결정 ──────────────────────────────────────────────────────────
    _assign_risk_level(features)
    _assign_yellow_subtype(features)

    return features


_ddi_lookup_cache: dict[int, dict[frozenset, str]] = {}
_SEVERITY_ORDER = {"Contraindicated": 4, "Major": 3, "Moderate": 2, "Minor": 1}

# DDI 피처 시맨틱 버전 (단일 출처). DDI 카운트의 **의미**가 바뀌면 올린다.
#   v2: WK_COMPN_CD→DrugMaster→DB-code, overlap 쌍 기준(edi→wk 브릿지로 서빙 정합, Task B).
#   v1/누락: 구 경로(ATC all-pairs 또는 drug_master 미전달 ddi=0) — 서빙과 비호환.
# 학습 번들 메타에 기록하고, 서빙 reload 가드가 현재 버전과 불일치/누락 번들을 거부한다
# (구 ddi=0 학습 모델이 실 DDI 계산 서빙에 로드되면 train/serve 스큐 — d201743 전례).
DDI_FEATURE_SEMANTICS_VERSION = "ddi.v2"


def _get_ddi_lookup(ddi_matrix: pd.DataFrame) -> dict[frozenset, str]:
    """DDI 매트릭스 → ID 쌍 기반 조회 딕셔너리 (캐시됨)."""
    matrix_id = id(ddi_matrix)
    if matrix_id in _ddi_lookup_cache:
        return _ddi_lookup_cache[matrix_id]

    ddi_lookup: dict[frozenset, str] = {}
    id_cols = ("drug_a_id", "drug_b_id")
    if all(c in ddi_matrix.columns for c in id_cols):
        for row in ddi_matrix.itertuples(index=False):
            a_id = str(row.drug_a_id).strip()
            b_id = str(row.drug_b_id).strip()
            if not a_id or not b_id:
                continue
            key = frozenset({a_id, b_id})
            new_sev = str(row.severity)
            existing = ddi_lookup.get(key)
            if existing is None or _SEVERITY_ORDER.get(new_sev, 0) > _SEVERITY_ORDER.get(existing, 0):
                ddi_lookup[key] = new_sev

    _ddi_lookup_cache[matrix_id] = ddi_lookup
    return ddi_lookup


def ddi_pair_severities(
    pairs: list[DrugOverlapPair],
    ddi_matrix: pd.DataFrame,
    drug_master: DrugMaster | None = None,
) -> list[tuple[DrugOverlapPair, str]]:
    """각 동시복용 쌍의 최고 DDI 심각도. (pair, severity) 리스트(미평가 쌍 제외).

    학습·서빙 **단일 출처**: DrugMaster WK_COMPN_CD → 성분명 → DDI ID, 복합제는 성분 ID
    쌍 cross-product 에서 최고 심각도. count_ddi_severities·서빙 DDI 피처·DDI 알림이 모두
    이 함수를 거쳐 train/serve 스큐를 차단한다(d201743 전례).
    """
    out: list[tuple[DrugOverlapPair, str]] = []
    if "severity" not in ddi_matrix.columns or drug_master is None:
        # ddi_lookup 은 drug_a_id/drug_b_id 기반 → drug_master 없으면 WK 직접조회 불가 → 미평가.
        return out

    ddi_lookup = _get_ddi_lookup(ddi_matrix)

    def _best_severity_for_pair(ids_a: list[str], ids_b: list[str]) -> str | None:
        best: str | None = None
        for ia in ids_a:
            for ib in ids_b:
                sev = ddi_lookup.get(frozenset({ia, ib}))
                if sev and _SEVERITY_ORDER.get(sev, 0) > _SEVERITY_ORDER.get(best or "", 0):
                    best = sev
        return best

    for pair in pairs:
        ids_a = drug_master.get_ddi_ids(pair.drug_a_wk_compn)
        ids_b = drug_master.get_ddi_ids(pair.drug_b_wk_compn)
        if not (ids_a and ids_b):
            continue
        severity = _best_severity_for_pair(ids_a, ids_b)
        if severity:
            out.append((pair, severity))
    return out


def count_ddi_severities(
    pairs: list[DrugOverlapPair],
    ddi_matrix: pd.DataFrame,
    drug_master: DrugMaster | None = None,
) -> dict[str, int]:
    """동시복용 쌍 → 심각도별 카운트 dict. ddi_pair_severities 위에 구성(단일 출처)."""
    counts = {"Contraindicated": 0, "Major": 0, "Moderate": 0, "Minor": 0}
    for _pair, sev in ddi_pair_severities(pairs, ddi_matrix, drug_master):
        if sev in counts:
            counts[sev] += 1
    return counts


def _fill_ddi_features(
    features: PatientFeatures,
    pairs: list[DrugOverlapPair],
    ddi_matrix: pd.DataFrame,
    drug_master: DrugMaster | None = None,
) -> None:
    """동시복용 쌍 × DDI → features.ddi_* 누적. 카운트는 count_ddi_severities(공용)."""
    counts = count_ddi_severities(pairs, ddi_matrix, drug_master)
    features.ddi_contraindicated += counts["Contraindicated"]
    features.ddi_major += counts["Major"]
    features.ddi_moderate += counts["Moderate"]
    features.ddi_minor += counts["Minor"]


def count_same_ingredient_dups(
    prescriptions: list[PrescriptionRecord],
    drug_master: DrugMaster | None = None,
) -> int:
    """동일 성분(복합제 전개) 2개 이상 처방 수 = dup_same_ingredient 의 성분 경로.

    학습(_fill_dup_features)·서빙 공용 단일출처. DrugMaster 로 각 WK_COMPN_CD 를 성분명
    집합으로 전개(복합제 포함) 후 성분별 2회+ 카운트. drug_master 없으면 WK 직접 비교.
    (학습 records 는 atc_code 가 없어 ATC fallback 은 프로덕션에서 미발동 — 본 함수가 곧 값.)
    """
    from collections import Counter
    wk_codes = [p.wk_compn_cd for p in prescriptions if p.wk_compn_cd]
    if len(wk_codes) < 2:
        return 0
    if drug_master is not None:
        all_comps: list[str] = []
        for code in wk_codes:
            comps = set(drug_master.get_components(code)) or {code}
            all_comps.extend(comps)
        cnt = Counter(all_comps)
        return sum(1 for c in cnt.values() if c >= 2)
    cnt_wk = Counter(wk_codes)
    return sum(1 for c in cnt_wk.values() if c >= 2)


# ── Triple Whammy 성분 클래스 (학습·서빙 공용 단일출처) ────────────────────────
# ACEi/ARB + K보존 이뇨제 + NSAID 동시 → 급성 신손상 위험(Red 트리거).
# wk→ATC 매핑이 부재(HIRA 무, DrugBank 크로스워크는 K이뇨제/일부 NSAID 미해석)라
# **성분명 키워드**로 판정(get_components). ACEi='...pril', ARB='...sartan' 접미사 +
# K이뇨제·NSAID 명시 목록. 키워드 완전성은 추후 보강(임상 큐레이션). 변경 시 재학습 필요.
_TW_ACEI_ARB_SUFFIX = ("pril", "sartan")
_TW_KSPARING = frozenset({
    "spironolactone", "eplerenone", "amiloride", "triamterene", "canrenone",
})
_TW_NSAID = frozenset({
    "ibuprofen", "dexibuprofen", "naproxen", "diclofenac", "aceclofenac",
    "celecoxib", "etoricoxib", "meloxicam", "lornoxicam", "piroxicam", "tenoxicam",
    "ketoprofen", "dexketoprofen", "loxoprofen", "zaltoprofen", "flurbiprofen",
    "etodolac", "nabumetone", "sulindac", "indomethacin", "mefenamic", "talniflumate",
    "nimesulide", "ketorolac", "pelubiprofen", "polmacoxib", "fenbufen", "nabumeton",
})


def _wk_ingredient_classes(wk: str, drug_master: DrugMaster) -> tuple[bool, bool, bool]:
    """단일 wk 의 성분명 → (ACEi/ARB, K이뇨제, NSAID) 클래스 보유 여부."""
    acei_arb = ksparing = nsaid = False
    for comp in drug_master.get_components(wk):
        c = comp.lower()
        if c.endswith(_TW_ACEI_ARB_SUFFIX):
            acei_arb = True
        if any(k in c for k in _TW_KSPARING):
            ksparing = True
        if any(n in c for n in _TW_NSAID):
            nsaid = True
    return acei_arb, ksparing, nsaid


def detect_triple_whammy(wk_codes: list[str], drug_master: DrugMaster | None = None) -> bool:
    """ACEi/ARB + K보존이뇨제 + NSAID 동시복용 여부 (학습·서빙 공용 단일출처).

    DrugMaster.get_components(wk) 성분명 키워드 기반(ATC 부재 대응). drug_master 없으면 False.
    학습(aggregate_patient_features)·서빙(edi→wk 후)이 동일 wk 집합으로 호출 → parity by construction.
    """
    if drug_master is None or not wk_codes:
        return False
    a = k = n = False
    for wk in wk_codes:
        if not wk:
            continue
        wa, wk_, wn = _wk_ingredient_classes(wk, drug_master)
        a = a or wa
        k = k or wk_
        n = n or wn
        if a and k and n:
            return True
    return a and k and n


def _fill_dup_features(
    features: PatientFeatures,
    prescriptions: list[PrescriptionRecord],
    dup_groups: pd.DataFrame,
    drug_master: DrugMaster | None = None,
) -> None:
    """
    중복약물 레벨 계산.

    레벨 우선순위:
      1. 성분명 동일 (DrugMaster 전개 기반) — 복합제 성분과 단일제 교차 검사 포함
      2. EFMDC_CLSF_NO 동일 (NHIS 약효분류) — 효능군 중복
      3. ATC prefix 레벨 (DrugBank ATC 코드 기반)
    """
    from collections import Counter

    # ── 1. 성분명 기반 동일성분 중복 (공용 단일출처 — 서빙도 호출) ──────────
    features.dup_same_ingredient = count_same_ingredient_dups(prescriptions, drug_master)

    # ── 2. EFMDC_CLSF_NO 기반 효능군 중복 ──────────────────────────────────
    efmdc_codes = [p.efmdc_clsf_no for p in prescriptions if p.efmdc_clsf_no]
    if len(efmdc_codes) >= 2:
        cnt_efmdc = Counter(efmdc_codes)
        features.dup_efmdc = sum(1 for c in cnt_efmdc.values() if c >= 2)

    # ── 3. ATC 코드 기반 레벨별 중복 (DrugBank 매핑 된 경우) ─────────────────
    atc_codes = [p.atc_code for p in prescriptions if p.atc_code]
    if len(atc_codes) < 2:
        return

    # ATC 5단계 (7자리 full code) — 동일 성분 2개 이상 (WK_COMPN_CD 미매핑 보완)
    cnt5 = Counter(atc_codes)
    if features.dup_same_ingredient == 0:
        features.dup_same_ingredient = sum(1 for c in cnt5.values() if c >= 2)
    features.dup_atc5 = sum(1 for c in cnt5.values() if c >= 2)

    # ATC 4단계 (5자리 prefix)
    cnt4: Counter = Counter()
    for code in atc_codes:
        if len(code) >= 5:
            cnt4[code[:5]] += 1
    features.dup_atc4 = sum(1 for c in cnt4.values() if c >= 2)

    # ATC 3단계 (4자리 prefix)
    cnt3: Counter = Counter()
    for code in atc_codes:
        if len(code) >= 4:
            cnt3[code[:4]] += 1
    features.dup_atc3 = sum(1 for c in cnt3.values() if c >= 2)


def _assign_risk_level(features: PatientFeatures) -> None:
    """위험도 판정 (CLINICAL_STANDARDS_v1.0).

    trigger 집합을 clinical_rules 에서 수집해 Red > Yellow > Green > Normal
    순으로 분기한다. 판정 이유(risk_reasons)는 trigger 이름으로 기록되어
    서빙 단계에서 동일 이름으로 설명 가능하다.
    """
    red_triggers = collect_red_triggers(features)
    if red_triggers:
        features.risk_level = "Red"
        features.risk_reasons = sorted(red_triggers)
        return

    yellow_triggers = collect_yellow_triggers(features)
    if yellow_triggers:
        features.risk_level = "Yellow"
        features.risk_reasons = sorted(yellow_triggers)
        return

    # TODO: Green 트리거도 collect_green_triggers 로 clinical_rules 에 이관 예정
    #       현재는 risk_reasons 형식이 Red/Yellow(토큰) vs Green(한국어 문구) 로 혼재.
    if features.ddi_minor >= 1:
        features.risk_level = "Green"
        features.risk_reasons = [f"Minor DDI {features.ddi_minor}건"]
        return

    if features.drug_count >= 5:
        features.risk_level = "Green"
        features.risk_reasons = [f"5종↑ ({features.drug_count}종)"]
        return

    features.risk_level = "Normal"
    features.risk_reasons = []


def _assign_yellow_subtype(features: PatientFeatures) -> None:
    """Yellow 세분화 (risk_level == 'Yellow' 인 환자 전용).

    계수(複合) 라벨이 단일 라벨보다 우선한다: **위험 차원** 개수로 분류한다
    (4대 위험 중 yellow 레벨 = 상호작용·중복·다기관 3차원; 금기는 Red).
    상호작용(DDI_MAJOR|DDI_MOD)은 한 차원으로 묶는다 → major+mod 동시발동도 1차원.
    3차원=Y_TRIPLE, 2차원=Y_DOUBLE, 1차원=해당 단일 라벨로 개입 강도를 구분.
    Red 조건이 충족된 환자는 _assign_risk_level 이 이미 Red 로 결정했으므로
    이 함수가 실행되어도 Yellow 가 아니기에 None.

    규칙 드리프트 엣지 (risk_level=Yellow 인데 trigger 집합이 빔) 는 RuntimeError
    대신 Y_OTHER 폴백 + 경고 로그로 처리 (ETL 파이프라인 중단 방지).
    Y_OTHER 분기는 yellow_subtype 만 설정하고 risk_reasons 는 건드리지 않는다
    (분류 책임만 가지며, 객체 수복은 호출자 몫).
    """
    if features.risk_level != "Yellow":
        features.yellow_subtype = None
        return

    triggers = collect_yellow_triggers(features)
    # 계수는 trigger 가 아니라 **위험 차원** 개수로 센다(4대 위험 기준).
    # 상호작용(DDI_MAJOR|DDI_MOD)은 한 차원으로 묶이므로 major+mod 동시발동도 1.
    interaction = "DDI_MAJOR" in triggers or "DDI_MOD" in triggers
    duplication = "DUP" in triggers
    multi_inst = "FRAG" in triggers
    dim_count = int(interaction) + int(duplication) + int(multi_inst)

    if dim_count >= 3:
        features.yellow_subtype = "Y_TRIPLE"
        return
    if dim_count == 2:
        features.yellow_subtype = "Y_DOUBLE"
        return
    if dim_count == 1:
        if interaction:
            # 상호작용 단일 차원 — major 가 mod 보다 우선(더 중한 신호).
            features.yellow_subtype = "Y_DDI_MAJOR" if "DDI_MAJOR" in triggers else "Y_DDI_MOD"
        elif duplication:
            features.yellow_subtype = "Y_DUP"
        else:  # multi_inst
            features.yellow_subtype = "Y_FRAG"
        return

    logging.getLogger(__name__).warning(
        "yellow_without_trigger patient_id=%s — Y_OTHER 로 격리 (규칙 드리프트 의심)",
        features.patient_id,
    )
    features.yellow_subtype = "Y_OTHER"


def aggregate_batch(
    df_prescriptions: pd.DataFrame,
    df_t40: pd.DataFrame | None,
    overlap_df: pd.DataFrame,
    ddi_matrix: pd.DataFrame | None,
    dup_groups: pd.DataFrame | None,
    drug_master: DrugMaster | None = None,
) -> list[PatientFeatures]:
    """
    전체 환자 배치 집계.
    overlap_df: calculate_overlaps_batch() 결과 DataFrame.
    """
    # 처방 레코드 → 환자별 그룹
    from .overlap_calculator import prescriptions_from_df
    all_records = prescriptions_from_df(df_prescriptions)
    patient_records: dict[str, list[PrescriptionRecord]] = {}
    for r in all_records:
        patient_records.setdefault(r.patient_id, []).append(r)

    # 환자 인구통계 — T20에 SEX_TYPE, SUJIN_POTM_AGE_ID가 포함됨
    # df_t40은 상병내역(진단)이므로 인구통계 없음; df_prescriptions(T20+T30 조인)에서 추출
    patient_demo: dict[str, dict] = {}
    if "INDI_DSCM_NO" in df_prescriptions.columns:
        demo_cols = [c for c in ["INDI_DSCM_NO", "SEX_TYPE", "SUJIN_POTM_AGE_ID"]
                     if c in df_prescriptions.columns]
        if len(demo_cols) > 1:
            demo_df = df_prescriptions[demo_cols].drop_duplicates("INDI_DSCM_NO")
            for row in demo_df.itertuples(index=False):
                pid = str(row.INDI_DSCM_NO)
                patient_demo[pid] = {
                    "sex": str(getattr(row, "SEX_TYPE", "")) or None,
                    "age_id": str(getattr(row, "SUJIN_POTM_AGE_ID", "")) or None,
                }

    # 동시복용 쌍 → 환자별 그룹
    patient_pairs: dict[str, list[DrugOverlapPair]] = {}
    if not overlap_df.empty:
        for row in overlap_df.itertuples(index=False):
            pid = str(row.patient_id)
            patient_pairs.setdefault(pid, []).append(DrugOverlapPair(
                patient_id=pid,
                drug_a_wk_compn=str(getattr(row, "drug_a_wk_compn", "") or ""),
                drug_a_edi=getattr(row, "drug_a_edi", None) or None,
                drug_a_atc=getattr(row, "drug_a_atc", None) or None,
                drug_a_name=getattr(row, "drug_a_name", None) or None,
                drug_b_wk_compn=str(getattr(row, "drug_b_wk_compn", "") or ""),
                drug_b_edi=getattr(row, "drug_b_edi", None) or None,
                drug_b_atc=getattr(row, "drug_b_atc", None) or None,
                drug_b_name=getattr(row, "drug_b_name", None) or None,
                overlap_start=row.overlap_start,
                overlap_end=row.overlap_end,
                overlap_days=int(row.overlap_days),
                window_start=row.window_start,
                window_end=row.window_end,
            ))

    all_features: list[PatientFeatures] = []
    for patient_id, prx_list in patient_records.items():
        demo = patient_demo.get(patient_id, {})
        # SEX_TYPE: "1"=남, "2"=여 → PatientFeatures.sex에 그대로 저장
        sex = demo.get("sex") or None
        age_id = demo.get("age_id") or None
        # SUJIN_POTM_AGE_ID: 10세 단위 연령범주 ID (3=30대, 4=40대, ..., 8=80대)
        # 하한값(age_id * 10)으로 파생: "7"(70대) → 70, "8"(80대) → 80.
        # ≥75 규칙은 70대 버킷(70~79)에 75~79세가 포함되므로 age >= 70 으로 적용.
        age: int | None = int(age_id) * 10 if age_id and age_id.isdigit() else None

        pairs = patient_pairs.get(patient_id, [])
        feat = aggregate_patient_features(
            patient_id=patient_id,
            prescriptions=prx_list,
            overlap_pairs=pairs,
            ddi_matrix=ddi_matrix,
            dup_groups=dup_groups,
            age=age,
            sex=sex,
            drug_master=drug_master,
        )
        feat.age_id = age_id
        all_features.append(feat)

    return all_features
