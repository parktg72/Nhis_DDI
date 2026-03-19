"""
샘플 데이터 팩토리
테스트 및 개발용 합성 청구 데이터 생성.
실제 환자 데이터 없이 ETL 파이프라인 전체를 검증할 수 있음.

생성 데이터:
- T20 (명세서): 환자×기관×날짜 조합
- T30 (처방약물): 다양한 DDI 시나리오 포함
- T40 (수진자): 나이/성별 분포
- T50 (요양기관): 기관 유형
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 약물 풀 (Top 10 DDI 시나리오 포함)
# ─────────────────────────────────────────────────────────────────────────────

DRUG_POOL = [
    # (edi_code, atc_code, drug_name, class)
    ("A001001", "B01AA03", "warfarin",       "anticoag"),
    ("A001002", "B01AF01", "rivaroxaban",    "anticoag"),
    ("A001003", "M01AE01", "ibuprofen",      "nsaid"),
    ("A001004", "M01AE02", "naproxen",       "nsaid"),
    ("A001005", "B01AC04", "clopidogrel",    "antiplatelet"),
    ("A001006", "A02BC01", "omeprazole",     "ppi"),
    ("A001007", "C09AA02", "enalapril",      "acei"),
    ("A001008", "C09CA01", "losartan",       "arb"),
    ("A001009", "C03DA01", "spironolactone", "k_sparing"),
    ("A001010", "C01AA05", "digoxin",        "digoxin"),
    ("A001011", "C01BD01", "amiodarone",     "antiarrhythmic"),
    ("A001012", "C08DA01", "verapamil",      "ccb"),
    ("A001013", "L01BA01", "methotrexate",   "antimetabolite"),
    ("A001014", "J01EA01", "trimethoprim",   "antibiotic"),
    ("A001015", "N06AB03", "fluoxetine",     "ssri"),
    ("A001016", "N06AG02", "phenelzine",     "maoi"),
    ("A001017", "N02CC01", "sumatriptan",    "triptan"),
    ("A001018", "N05AN01", "lithium",        "mood_stab"),
    ("A001019", "C10AA01", "simvastatin",    "statin"),
    ("A001020", "J01FA09", "clarithromycin", "macrolide"),
    # QT 연장 약물
    ("A001021", "N05AD01", "haloperidol",    "antipsych"),
    ("A001022", "J01MA02", "ciprofloxacin",  "fluoroq"),
    ("A001023", "P01BA01", "chloroquine",    "antimalarial"),
    # 일반 약물 (위험도 낮음)
    ("B001001", "C10AA04", "atorvastatin",   "statin"),
    ("B001002", "A10BA02", "metformin",      "antidiabetic"),
    ("B001003", "C07AB02", "metoprolol",     "beta_blocker"),
    ("B001004", "C03CA01", "furosemide",     "loop_diuretic"),
    ("B001005", "R06AX13", "loratadine",     "antihistamine"),
    ("B001006", "N02BE01", "acetaminophen",  "analgesic"),
    ("B001007", "A11CC05", "cholecalciferol","vitamin"),
]

# DDI 시나리오 (환자 유형별 처방 약물 집합)
DDI_SCENARIOS = {
    "red_contraindicated": [
        # TOP01: warfarin + ibuprofen (Contraindicated)
        ["A001001", "A001003", "B001002", "B001003"],
    ],
    "red_major_triple": [
        # Major DDI 3건 이상
        ["A001005", "A001006",   # clopidogrel + omeprazole (Major)
         "A001010", "A001011",   # digoxin + amiodarone (Major)
         "A001019", "A001020",   # simvastatin + clarithromycin (Major)
         "B001002"],
    ],
    "red_triple_whammy": [
        # TOP03: ACEi + K보존이뇨제 + NSAIDs
        ["A001007", "A001009", "A001003", "B001002"],
    ],
    "yellow_major": [
        # Major DDI 1건
        ["A001001", "A001003", "B001002", "B001004"],
    ],
    "yellow_dup": [
        # 동일성분 중복 (같은 ATC 2개)
        ["A001003", "A001004", "B001002", "B001005"],  # 2개 NSAID
    ],
    "green_minor": [
        # Minor DDI만
        ["B001001", "B001002", "B001003", "B001004", "B001005"],
    ],
    "normal": [
        # DDI 없음, 약물 4종 이하
        ["B001002", "B001003", "B001005"],
    ],
}


def _random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def make_t40(
    n_patients: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """T40: 수진자 인구통계."""
    rng = random.Random(seed)
    rows = []
    for i in range(n_patients):
        pid = f"P{i+1:06d}"
        birth_year = rng.randint(1940, 2000)
        sex = rng.choice(["1", "2"])
        rows.append({
            "BNFCR_PSEUDO": pid,
            "SEX_TP_CD":    sex,
            "BTH_YYYY":     str(birth_year),
        })
    return pd.DataFrame(rows)


def make_t50(n_institutions: int = 10, seed: int = 42) -> pd.DataFrame:
    """T50: 요양기관."""
    rng = random.Random(seed)
    inst_types = ["1", "3", "11", "21"]
    rows = []
    for i in range(n_institutions):
        rows.append({
            "INST_PSEUDO": f"INST{i+1:04d}",
            "CLNC_TP_CD":  rng.choice(inst_types),
        })
    return pd.DataFrame(rows)


def make_t20_t30(
    n_patients: int = 100,
    ref_date: date | None = None,
    scenario_weights: dict[str, float] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    T20 (명세서) + T30 (처방약물) 합성 데이터 생성.

    Parameters
    ----------
    n_patients : 환자 수
    ref_date : 기준일 (기본: 오늘)
    scenario_weights : 시나리오별 가중치 (기본: 균등)
    """
    rng = random.Random(seed)
    ref = ref_date or date.today()
    window_start = ref - timedelta(days=89)

    if scenario_weights is None:
        scenario_weights = {
            "red_contraindicated": 0.05,
            "red_major_triple":    0.05,
            "red_triple_whammy":   0.05,
            "yellow_major":        0.15,
            "yellow_dup":          0.10,
            "green_minor":         0.20,
            "normal":              0.40,
        }

    scenarios = list(scenario_weights.keys())
    weights = [scenario_weights[s] for s in scenarios]

    drug_map = {d[0]: d for d in DRUG_POOL}
    institutions = [f"INST{i+1:04d}" for i in range(10)]

    t20_rows = []
    t30_rows = []
    bill_seq = 1

    for i in range(n_patients):
        pid = f"P{i+1:06d}"
        scenario_key = rng.choices(scenarios, weights=weights, k=1)[0]
        drug_set_list = DDI_SCENARIOS[scenario_key]
        drug_set = rng.choice(drug_set_list)

        # 처방 1~3회
        n_bills = rng.randint(1, 3)
        for _ in range(n_bills):
            start_dt = _random_date(window_start, ref - timedelta(days=7))
            end_dt = start_dt + timedelta(days=rng.randint(6, 30))
            bill_no = f"BILL{bill_seq:08d}"
            bill_seq += 1
            inst = rng.choice(institutions)

            t20_rows.append({
                "MDCARE_BILL_NO": bill_no,
                "BNFCR_PSEUDO":  pid,
                "INST_PSEUDO":   inst,
                "MDCARE_STRT_DT": start_dt.strftime("%Y%m%d"),
                "MDCARE_END_DT":  end_dt.strftime("%Y%m%d"),
                "SICK_SYM":      rng.choice(["I10", "E11", "J45", "M54", "K21"]),
            })

            # 약물 할당 (1명세서에 2~4종)
            n_drugs = min(len(drug_set), rng.randint(2, 4))
            selected = rng.sample(drug_set, n_drugs)
            for edi in selected:
                drug_info = drug_map.get(edi)
                if not drug_info:
                    continue
                total_days = (end_dt - start_dt).days + 1
                t30_rows.append({
                    "MDCARE_BILL_NO":  bill_no,
                    "EDI_CD":          edi,
                    "DOSG_ONCE":       rng.choice([0.5, 1.0, 2.0]),
                    "DOSG_FREQ_DY":    rng.choice([1, 2, 3]),
                    "MEDTIME_FRQ_CNT": total_days,
                })

    return pd.DataFrame(t20_rows), pd.DataFrame(t30_rows)


def make_edi_atc_map() -> pd.DataFrame:
    """샘플용 EDI→ATC 매핑 DataFrame."""
    rows = []
    for edi, atc, name, _ in DRUG_POOL:
        rows.append({"drug_id": edi, "atc_code": atc, "name": name})
    return pd.DataFrame(rows)
