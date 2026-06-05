"""Task B SPEC: 학습 ↔ 서빙 DDI 중증도 PARITY.

요구사항(목표): 동일 약물셋·동일 날짜에 대해 serving RequestFeatureBuilder 의
ddi_* 가 학습 피처빌더(aggregate_patient_features)의 ddi_* 와 일치해야 한다.
join key(약물 식별 공간)와 **쌍 정의(동시복용 overlap)** 가 다르면 train/serve
스큐가 난다 (CLAUDE.md d201743 전례).

현황(2026-06-05):
- 학습: WK_COMPN_CD → DrugMaster → DDI ID(DrugBank) → ddi_matrix. **overlap_pairs**
  (calculate_overlaps_for_patient: frozenset dedup + overlap_days>=7)에 대해서만 카운트.
- 서빙: edi_code 만 받고 EDI→DB-code 브릿지 부재 + _ddi_index 가 ATC 키 조건이라
  항상 빈 dict → ddi_*=0. + 날짜 무시 all-pairs.
- 브릿지: HIRA 제품코드(EDI)→주성분코드(WK) 100% 1:1 → serving 에서 edi→wk→
  DrugMaster→DB-code 로 학습과 동일 경로 구성 가능.

이 테스트는 serving 정합 구현의 SPEC. 구현 완료 시 parity 테스트 xfail 제거.

Fixture (records+matrix 실측):
- mosapride(wk 421001ATB / edi 660700010)
- tramadol+APAP(wk 480600ATB / edi 642902720)  → mosapride 와 Major
- eperisone(wk 152301ATB / edi 670606240)       → mosapride 와 Moderate
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
EDIWK = ROOT / "data" / "processed" / "edi_to_wk.parquet"  # Task B EDI→WK 브릿지

A = ("421001ATB", "660700010")  # mosapride
B = ("480600ATB", "642902720")  # tramadol + APAP  (A×B = Major)
C = ("152301ATB", "670606240")  # eperisone        (A×C = Moderate)

# 시나리오: (이름, [(wk, edi, start_date, total_days), ...], 기대 학습 ddi)
# 핵심: NO_OVERLAP 은 날짜가 안 겹쳐 학습은 DDI 0 → 서빙이 all-pairs(날짜무시)로
# 세면 스큐가 드러난다(= overlap semantics 정합 검증).
_D1 = date(2024, 7, 1)
_D2 = date(2024, 8, 1)
SCENARIOS = [
    ("major_overlap",    [(*A, _D1, 30), (*B, _D1, 30)], {"ddi_major": 1}),
    ("moderate_overlap", [(*A, _D1, 30), (*C, _D1, 30)], {"ddi_moderate": 1}),
    ("major_no_overlap", [(*A, _D1, 5),  (*B, _D2, 10)], {"ddi_major": 0}),
]

_need_data = pytest.mark.skipif(
    not (MASTER.exists() and DDIMTX.exists() and EDIWK.exists()),
    reason="DDI/DrugMaster/EDI→WK 데이터 파일 없음(이 환경)",
)


def _ddi_keys(d) -> dict:
    return {k: int(d.get(k, 0)) for k in
            ("ddi_contraindicated", "ddi_major", "ddi_moderate", "ddi_minor")}


def _training_ddi(drugs) -> dict:
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
            wk_compn_cd=wk, edi_code=edi, gnl_nm_cd=None, efmdc_clsf_no=None,
            start_date=sd, end_date=sd + timedelta(days=td - 1),
            total_days=td, dose_once=1.0, dose_freq=1, sick_code=None,
            sex="1", age_id="5", institution_type=None, source="T30",
        )
        for i, (wk, edi, sd, td) in enumerate(drugs)
    ]
    overlaps = calculate_overlaps_for_patient(recs, window_days=90)
    feat = aggregate_patient_features(
        patient_id="P", prescriptions=recs, overlap_pairs=overlaps,
        ddi_matrix=ddi_matrix, dup_groups=dup_groups, drug_master=dm,
    )
    return _ddi_keys({
        "ddi_contraindicated": feat.ddi_contraindicated, "ddi_major": feat.ddi_major,
        "ddi_moderate": feat.ddi_moderate, "ddi_minor": feat.ddi_minor,
    })


def _serving_ddi(drugs) -> dict:
    import hana_app.core.ml_runner as M
    from scripts.etl.code_standardizer import CodeStandardizer
    from serving.predictor import RequestFeatureBuilder
    from serving.schemas import DrugItem, PredictRequest

    ddi_matrix = M._load_ddi_matrix()
    std = CodeStandardizer(master_parquet=str(MASTER), ddi_matrix_path=str(DDIMTX),
                           index_path=str(DRUGIDX), extra_csv=None)
    builder = RequestFeatureBuilder(ddi_matrix=ddi_matrix, code_standardizer=std)
    req = PredictRequest(
        patient_id="P", patient_age=50, patient_sex="M",
        drugs=[DrugItem(edi_code=edi, total_days=td, start_date=sd)
               for (_wk, edi, sd, td) in drugs],
    )
    _vec, feat = builder.build(req)
    return _ddi_keys(feat)


@_need_data
@pytest.mark.parametrize("name,drugs,expected", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_training_ddi_matches_expected(name, drugs, expected):
    """전제: 학습 경로가 시나리오별 기대 DDI(중첩시 탐지·비중첩시 0)를 낸다."""
    train = _training_ddi(drugs)
    for k, v in expected.items():
        assert train[k] == v, f"[{name}] 학습 {k}={train[k]} != 기대 {v} (전체 {train})"


@_need_data
@pytest.mark.parametrize("name,drugs,expected", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_serving_ddi_parity_with_training(name, drugs, expected):
    """serving ddi_* == training ddi_* (동일 약물·날짜). Task B 정합 완료(2026-06-05).

    serving 이 edi→wk(code_standardizer.get_wk)→PrescriptionRecord 재구성→
    calculate_overlaps_for_patient→count_ddi_severities(학습과 동일 공용함수)를 호출하므로
    날짜 미겹침(major_no_overlap)도 overlap 0 → DDI 0 으로 학습과 일치(all-pairs 스큐 제거).
    """
    train = _training_ddi(drugs)
    serve = _serving_ddi(drugs)
    assert serve == train, f"[{name}] train/serve DDI 스큐 — train={train} serve={serve}"
