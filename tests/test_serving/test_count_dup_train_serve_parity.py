"""P2 SPEC: 학습 ↔ 서빙 drug_count / drug_count_7d / dup_same_ingredient PARITY.

Task B(DDI parity)와 동일 원칙 — 서빙이 edi→wk→DrugMaster 로 학습과 같은 식별자공간·
함수를 써야 한다. 이들은 **모든 모델이 쓰는 코어 피처**라 서빙 스큐가 전 예측에 영향.

학습(aggregate_patient_features, df_row_to_record 경로 = records 에 atc_code 없음):
- drug_count       = len(drug_master.expand_drug_count(unique_wk))  (복합제 성분 전개)
- drug_count_7d    = get_concurrent_drug_count(prescriptions, w_end)
- dup_same_ingredient = 성분명 Counter (atc_code 없으니 ATC fallback 미발동)

서빙(현재): drug_count=고유EDI, drug_count_7d=drug_count 복사, dup=ATC5 기반 → 스큐.
서빙 정합 후 본 테스트 xfail 제거.

drug_count_7d 윈도우 정합: 학습 w_end 를 serving reference_date 와 동일하게 맞춰 비교한다
(학습에 window_end 명시 전달).
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MASTER = ROOT / "data" / "processed" / "hira_drug_master.parquet"
DDIMTX = ROOT / "data" / "processed" / "ddi_matrix_final.parquet"
DRUGIDX = ROOT / "data" / "processed" / "drug_name_index.parquet"
EDIWK = ROOT / "data" / "processed" / "edi_to_wk.parquet"

A = ("421001ATB", "660700010")  # mosapride (단일)
B = ("480600ATB", "642902720")  # tramadol + APAP (복합제)
C = ("152301ATB", "670606240")  # eperisone (단일)

_REF = date(2024, 7, 15)
# (이름, [(wk, edi, start_date, total_days), ...])
SCENARIOS = [
    ("expand_combo",   [(*A, _REF, 30), (*B, _REF, 30)]),                 # 복합제 전개
    ("dup_same_drug",  [(*A, _REF, 30), (*A, _REF, 30)]),                 # 동일 성분 중복
    ("mixed_overlap",  [(*A, _REF, 30), (*B, _REF, 30), (*C, _REF, 30)]), # 3약물 동시
    ("partial_concurrent", [(*A, _REF, 5), (*B, _REF - timedelta(days=20), 10)]),  # 일부만 동시
]

_KEYS = ("drug_count", "drug_count_7d", "dup_same_ingredient",
         "dup_atc5", "dup_atc4", "dup_atc3")

_need_data = pytest.mark.skipif(
    not (MASTER.exists() and DDIMTX.exists() and EDIWK.exists()),
    reason="DrugMaster/EDI→WK 데이터 파일 없음(이 환경)",
)


def _training_feats(drugs) -> dict:
    import hana_app.core.ml_runner as M
    from scripts.etl.models import PrescriptionRecord
    from scripts.etl.overlap_calculator import calculate_overlaps_for_patient
    from scripts.etl.prescription_aggregator import aggregate_patient_features

    M._DRUG_MASTER_CACHE.update({"obj": None, "loaded": False})
    dm = M._load_drug_master()
    ddi_matrix = M._load_ddi_matrix()
    dup_groups = M._load_dup_groups()
    recs = [
        PrescriptionRecord(
            patient_id="P", institution_id="I1", bill_no=f"B{i}",
            wk_compn_cd=wk, edi_code=edi, start_date=sd,
            end_date=sd + timedelta(days=td - 1), total_days=td, source="T30",
        )
        for i, (wk, edi, sd, td) in enumerate(drugs)
    ]
    overlaps = calculate_overlaps_for_patient(recs, window_days=90)
    feat = aggregate_patient_features(
        patient_id="P", prescriptions=recs, overlap_pairs=overlaps,
        ddi_matrix=ddi_matrix, dup_groups=dup_groups, drug_master=dm,
        window_end=_REF,   # 서빙 reference_date 와 동일 기준일로 정합
    )
    return {k: int(getattr(feat, k)) for k in _KEYS}


def _serving_feats(drugs) -> dict:
    import hana_app.core.ml_runner as M
    from scripts.etl.code_standardizer import CodeStandardizer
    from serving.predictor import RequestFeatureBuilder
    from serving.schemas import DrugItem, PredictRequest

    ddi_matrix = M._load_ddi_matrix()
    std = CodeStandardizer(master_parquet=str(MASTER), ddi_matrix_path=str(DDIMTX),
                           index_path=str(DRUGIDX), extra_csv=None)
    builder = RequestFeatureBuilder(ddi_matrix=ddi_matrix, code_standardizer=std)
    req = PredictRequest(
        patient_id="P", patient_age=50, patient_sex="M", reference_date=_REF,
        drugs=[DrugItem(edi_code=edi, total_days=td, start_date=sd)
               for (_wk, edi, sd, td) in drugs],
    )
    _vec, feat = builder.build(req)
    return {k: int(feat[k]) for k in _KEYS}


@_need_data
@pytest.mark.parametrize("name,drugs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_serving_count_dup_parity(name, drugs):
    """SPEC: serving {drug_count, drug_count_7d, dup_same_ingredient} == training."""
    train = _training_feats(drugs)
    serve = _serving_feats(drugs)
    assert serve == train, f"[{name}] train/serve 스큐 — train={train} serve={serve}"
