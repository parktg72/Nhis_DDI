"""
샘플 데이터 팩토리
테스트 및 개발용 합성 청구 데이터 생성.
실제 환자 데이터 없이 ETL 파이프라인 전체를 검증할 수 있음.

생성 데이터 (NHIS 실제 레이아웃 기준):
  T20 (진료명세서): 환자×기관×날짜 + 성별/연령 + 상병
  T30 (진료내역):  약물별 WK_COMPN_CD + MCARE_DIV_CD + 용법 (다양한 DDI 시나리오)
  T40 (상병내역):  CMN_KEY 기준 상병 목록
  요양기관:        기관 유형 + 지역
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 약물 풀 (Top 10 DDI 시나리오 포함)
# 형식: (mcare_div_cd, wk_compn_cd, efmdc_clsf_no, atc_code, drug_name, class)
#   mcare_div_cd  = NHIS EDI 코드 (진료분류코드, 9자리 시뮬레이션)
#   wk_compn_cd   = NHIS 주성분코드 (9자리)
#   efmdc_clsf_no = 약효분류번호 (5자리)
# ─────────────────────────────────────────────────────────────────────────────

DRUG_POOL = [
    # (mcare_div_cd, wk_compn_cd, efmdc_clsf_no, atc_code, drug_name, class)
    ("A00100100", "W000000001", "04140", "B01AA03", "warfarin",       "anticoag"),
    ("A00100200", "W000000002", "04140", "B01AF01", "rivaroxaban",    "anticoag"),
    ("A00100300", "W000000003", "01140", "M01AE01", "ibuprofen",      "nsaid"),
    ("A00100400", "W000000004", "01140", "M01AE02", "naproxen",       "nsaid"),
    ("A00100500", "W000000005", "04150", "B01AC04", "clopidogrel",    "antiplatelet"),
    ("A00100600", "W000000006", "23200", "A02BC01", "omeprazole",     "ppi"),
    ("A00100700", "W000000007", "21400", "C09AA02", "enalapril",      "acei"),
    ("A00100800", "W000000008", "21400", "C09CA01", "losartan",       "arb"),
    ("A00100900", "W000000009", "21300", "C03DA01", "spironolactone", "k_sparing"),
    ("A00101000", "W000000010", "21200", "C01AA05", "digoxin",        "digoxin"),
    ("A00101100", "W000000011", "21100", "C01BD01", "amiodarone",     "antiarrhythmic"),
    ("A00101200", "W000000012", "21130", "C08DA01", "verapamil",      "ccb"),
    ("A00101300", "W000000013", "42290", "L01BA01", "methotrexate",   "antimetabolite"),
    ("A00101400", "W000000014", "61310", "J01EA01", "trimethoprim",   "antibiotic"),
    ("A00101500", "W000000015", "11720", "N06AB03", "fluoxetine",     "ssri"),
    ("A00101600", "W000000016", "11740", "N06AG02", "phenelzine",     "maoi"),
    ("A00101700", "W000000017", "11200", "N02CC01", "sumatriptan",    "triptan"),
    ("A00101800", "W000000018", "11930", "N05AN01", "lithium",        "mood_stab"),
    ("A00101900", "W000000019", "21810", "C10AA01", "simvastatin",    "statin"),
    ("A00102000", "W000000020", "61520", "J01FA09", "clarithromycin", "macrolide"),
    # QT 연장 약물
    ("A00102100", "W000000021", "11790", "N05AD01", "haloperidol",    "antipsych"),
    ("A00102200", "W000000022", "61370", "J01MA02", "ciprofloxacin",  "fluoroq"),
    ("A00102300", "W000000023", "64100", "P01BA01", "chloroquine",    "antimalarial"),
    # 일반 약물 (위험도 낮음)
    ("B00100100", "W000000024", "21810", "C10AA04", "atorvastatin",   "statin"),
    ("B00100200", "W000000025", "39620", "A10BA02", "metformin",      "antidiabetic"),
    ("B00100300", "W000000026", "21220", "C07AB02", "metoprolol",     "beta_blocker"),
    ("B00100400", "W000000027", "21300", "C03CA01", "furosemide",     "loop_diuretic"),
    ("B00100500", "W000000028", "14100", "R06AX13", "loratadine",     "antihistamine"),
    ("B00100600", "W000000029", "11140", "N02BE01", "acetaminophen",  "analgesic"),
    ("B00100700", "W000000030", "31130", "A11CC05", "cholecalciferol","vitamin"),
]

# DDI 시나리오 (MCARE_DIV_CD 기준)
DDI_SCENARIOS = {
    "red_contraindicated": [
        # TOP01: warfarin + ibuprofen (Contraindicated)
        ["A00100100", "A00100300", "B00100200", "B00100300"],
    ],
    "red_major_triple": [
        # Major DDI 3건 이상
        ["A00100500", "A00100600",   # clopidogrel + omeprazole (Major)
         "A00101000", "A00101100",   # digoxin + amiodarone (Major)
         "A00101900", "A00102000",   # simvastatin + clarithromycin (Major)
         "B00100200"],
    ],
    "red_triple_whammy": [
        # TOP03: ACEi + K보존이뇨제 + NSAIDs
        ["A00100700", "A00100900", "A00100300", "B00100200"],
    ],
    "yellow_major": [
        # Major DDI 1건
        ["A00100100", "A00100300", "B00100200", "B00100400"],
    ],
    "yellow_dup": [
        # 동일성분 중복 (같은 WK_COMPN_CD 계열 2개)
        ["A00100300", "A00100400", "B00100200", "B00100500"],  # 2개 NSAID
    ],
    "green_minor": [
        # Minor DDI만
        ["B00100100", "B00100200", "B00100300", "B00100400", "B00100500"],
    ],
    "normal": [
        # DDI 없음, 약물 4종 이하
        ["B00100200", "B00100300", "B00100500"],
    ],
}


def _random_date(start: date, end: date, rng: random.Random) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def make_t40(
    bill_nos: list[str],
    t20: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """T40: 상병내역 (진단 코드 목록).

    T40은 T20(CMN_KEY)에 매달린 상병 내역 테이블이며,
    각 명세서(CMN_KEY)에 1개 이상의 ICD-10 코드를 생성.
    t20을 넘기면 INDI_DSCM_NO, MDCARE_STRT_DT 등 조인 컬럼 포함.
    """
    rng = random.Random(seed)
    icd10_pool = ["I10", "E11", "J45", "M54", "K21", "E78", "N18", "F32", "I25", "G40"]

    # T20에서 CMN_KEY → INDI_DSCM_NO, MDCARE_STRT_DT 매핑
    cmn_to_t20: dict[str, dict] = {}
    if t20 is not None:
        for row in t20.itertuples(index=False):
            cmn_to_t20[str(row.CMN_KEY)] = {
                "INDI_DSCM_NO":    str(row.INDI_DSCM_NO),
                "MDCARE_STRT_DT":  str(row.MDCARE_STRT_DT),
                "SEX_TYPE":        str(row.SEX_TYPE),
                "SUJIN_POTM_AGE_ID": str(row.SUJIN_POTM_AGE_ID),
            }

    rows = []
    for bill_no in bill_nos:
        n_sick = rng.randint(1, 3)
        t20_info = cmn_to_t20.get(bill_no, {})
        for seq, icd in enumerate(rng.sample(icd10_pool, min(n_sick, len(icd10_pool))), start=1):
            row: dict = {
                "CMN_KEY":          bill_no,
                "SICK_DESC_SEQ_NO": f"{seq:02d}",
                "MCEX_SICK_SYM":    icd,
                "SICK_CLSF_TYPE":   "1" if seq == 1 else "2",  # 1=주상병, 2=부상병
            }
            row.update(t20_info)  # INDI_DSCM_NO, MDCARE_STRT_DT, SEX_TYPE, SUJIN_POTM_AGE_ID
            rows.append(row)
    return pd.DataFrame(rows)


def make_yoyang(n_institutions: int = 10, seed: int = 42) -> pd.DataFrame:
    """요양기관 현황."""
    rng = random.Random(seed)
    # YOYANG_CLSFC_CD: 2자리 코드 (실제 NHIS 기준)
    inst_types = ["11", "05", "03", "21", "01"]  # 의원, 병원, 종합병원, 약국, 상급종합
    sgg_pool = ["11140", "11150", "11170", "11200", "21110", "26110", "28110"]
    rows = []
    for i in range(n_institutions):
        inst_cd = f"YY{i+1:06d}"
        rows.append({
            "STD_YYYY":               "2024",
            "MDCARE_SYM":             inst_cd,
            "YOYANG_CLSFC_CD":        rng.choice(inst_types),
            "YOYANG_DETAIL_CLSFC_CD": "00",
            "INST_NM":                f"테스트의원{i+1:04d}",
            "ADDR":                   f"서울시 테스트구 {i+1}로",
            "ADDR_SGG_CD":            rng.choice(sgg_pool),
            "T20_Y":                  "Y",
            "TMPCLS_Y":               "N",
        })
    return pd.DataFrame(rows)


# 하위 호환 별칭
def make_t50(n_institutions: int = 10, seed: int = 42) -> pd.DataFrame:
    return make_yoyang(n_institutions, seed)


def make_t20_t30(
    n_patients: int = 100,
    ref_date: date | None = None,
    scenario_weights: dict[str, float] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    T20 (진료명세서) + T30 (진료내역) 합성 데이터 생성.

    NHIS 실제 레이아웃 컬럼 사용:
      T20: CMN_KEY, INDI_DSCM_NO, MDCARE_SYM, MDCARE_STRT_DT,
           MDCARE_STRT_YYYYMM, SICK_SYM1, SEX_TYPE, SUJIN_POTM_AGE_ID,
           YOYANG_CLSFC_CD, TOT_PRSC_DD_CNT
      T30: CMN_KEY, INDI_DSCM_NO, WK_COMPN_CD, RVSN_WK_COMPN_CD,
           MCARE_DIV_CD, EFMDC_CLSF_NO, TIME1_MDCT_CPCT, DD1_MQTY_FREQ,
           TOT_MCNT, MDCARE_STRT_DT
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

    # MCARE_DIV_CD → 약물 정보 매핑
    drug_map = {d[0]: d for d in DRUG_POOL}
    institutions = [f"YY{i+1:06d}" for i in range(10)]
    inst_types = ["11", "05", "03", "21", "01"]
    sex_pool = ["1", "2"]
    # SUJIN_POTM_AGE_ID: 연령범주 ID (10세 단위 시뮬레이션)
    age_id_pool = [str(i) for i in range(3, 9)]  # 3=30대, ..., 8=80대
    icd10_pool = ["I10", "E11", "J45", "M54", "K21", "E78", "N18"]

    t20_rows = []
    t30_rows = []
    bill_seq = 1

    for i in range(n_patients):
        pid = f"PT{i+1:08d}"
        scenario_key = rng.choices(scenarios, weights=weights, k=1)[0]
        drug_set_list = DDI_SCENARIOS[scenario_key]
        drug_set = rng.choice(drug_set_list)
        sex = rng.choice(sex_pool)
        age_id = rng.choice(age_id_pool)
        inst = rng.choice(institutions)
        inst_type = inst_types[institutions.index(inst) % len(inst_types)]

        # 처방 1~3회
        n_bills = rng.randint(1, 3)
        for _ in range(n_bills):
            start_dt = _random_date(window_start, ref - timedelta(days=7), rng)
            total_days = rng.randint(7, 30)
            bill_no = f"CMN{bill_seq:020d}"
            bill_seq += 1

            t20_rows.append({
                "CMN_KEY":              bill_no,
                "INDI_DSCM_NO":         pid,
                "MDCARE_SYM":           inst,
                "MDCARE_STRT_DT":       start_dt.strftime("%Y%m%d"),
                "MDCARE_STRT_YYYYMM":   start_dt.strftime("%Y%m"),
                "HIRA_EXM_YYYYMM":      start_dt.strftime("%Y%m"),
                "SICK_SYM1":            rng.choice(icd10_pool),
                "SICK_SYM2":            "",
                "SEX_TYPE":             sex,
                "SUJIN_POTM_AGE_ID":    age_id,
                "YOYANG_CLSFC_CD":      inst_type,
                "TOT_PRSC_DD_CNT":      total_days,
                "MCARE_TP":             "1",
                "WMED_OTMED_TYPE":      "1",
                "FORM_CD":              "10",
                "PAY_YN":               "1",
            })

            # 약물 할당 (1명세서에 2~4종)
            n_drugs = min(len(drug_set), rng.randint(2, 4))
            selected = rng.sample(drug_set, n_drugs)
            for mcare_div_cd in selected:
                drug_info = drug_map.get(mcare_div_cd)
                if not drug_info:
                    continue
                _, wk_compn_cd, efmdc_clsf_no, atc_code, drug_name, _ = drug_info
                t30_rows.append({
                    "CMN_KEY":          bill_no,
                    "INDI_DSCM_NO":     pid,
                    "MDCARE_STRT_DT":   start_dt.strftime("%Y%m%d"),
                    "MDCARE_STRT_YYYYMM": start_dt.strftime("%Y%m"),
                    "WK_COMPN_CD":      wk_compn_cd,
                    "RVSN_WK_COMPN_CD": wk_compn_cd,
                    "MCARE_DIV_CD":     mcare_div_cd,
                    "MCARE_DIV_CD_NM":  drug_name,
                    "EFMDC_CLSF_NO":    efmdc_clsf_no,
                    "TIME1_MDCT_CPCT":  rng.choice([0.5, 1.0, 2.0]),
                    "DD1_MQTY_FREQ":    rng.choice([1.0, 2.0, 3.0]),
                    "TOT_MCNT":         total_days,
                    "DRUG_MDCT_CPCT":   rng.choice([0.5, 1.0, 2.0]) * total_days,
                    "UPRC":             rng.uniform(100, 5000),
                    "AMT":              rng.uniform(1000, 50000),
                    "SEX_TYPE":         sex,
                    "SUJIN_POTM_AGE_ID": age_id,
                    "PAY_YN":           "1",
                    "FORM_CD":          "10",
                })

    return pd.DataFrame(t20_rows), pd.DataFrame(t30_rows)


def make_edi_atc_map() -> pd.DataFrame:
    """샘플용 MCARE_DIV_CD→ATC 매핑 DataFrame."""
    rows = []
    for mcare_div_cd, wk_compn_cd, efmdc_clsf_no, atc_code, name, _ in DRUG_POOL:
        rows.append({
            "drug_id":      mcare_div_cd,
            "wk_compn_cd":  wk_compn_cd,
            "efmdc_clsf_no": efmdc_clsf_no,
            "atc_code":     atc_code,
            "name":         name,
        })
    return pd.DataFrame(rows)
