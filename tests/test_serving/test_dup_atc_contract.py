"""ATC dup feature contract — serving must mirror ETL (prescription_aggregator).

ETL contract (scripts/etl/prescription_aggregator.py:325-343):
  dup_atc5 = full 7-char ATC code 중복
  dup_atc4 = 5-char prefix 중복
  dup_atc3 = 4-char prefix 중복

Codex review 2026-05-06 회귀: serving 의 dup_atc{5,4,3} 가 한 칸씩 prefix 가
밀려 있어 A10BA02/A10BA03 처방 시 ETL 은 dup_atc5=0, dup_atc4=1 인데 serving 은
dup_atc5=1 로 학습-서빙 drift 발생.
"""
from serving.predictor import RequestFeatureBuilder
from serving.schemas import DrugItem, PredictRequest


def _build_feat(atc_codes: list[str]) -> dict:
    drugs = [
        DrugItem(
            edi_code=f"E{i:03d}",
            atc_code=code,
            drug_name=f"drug_{i}",
            total_days=30,
        )
        for i, code in enumerate(atc_codes)
    ]
    req = PredictRequest(patient_id="p1", drugs=drugs, patient_age=65)
    builder = RequestFeatureBuilder()
    _, feat = builder.build(req)
    return feat


def test_dup_atc5_exact_match():
    """동일 ATC 5단계(7자리) 2개 → dup_atc5 = 1, dup_same_ingredient = 1."""
    feat = _build_feat(["A10BA02", "A10BA02"])
    assert feat["dup_atc5"] == 1.0, f"dup_atc5 expected 1, got {feat['dup_atc5']}"
    assert feat["dup_same_ingredient"] == 1.0


def test_dup_atc4_same_prefix5():
    """5자리 prefix 동일, 전체 코드 다름 → dup_atc4 = 1, dup_atc5 = 0."""
    # A10BA02 vs A10BA03 — prefix 5자리 'A10BA' 동일
    feat = _build_feat(["A10BA02", "A10BA03"])
    assert feat["dup_atc5"] == 0.0, f"dup_atc5 should be 0, got {feat['dup_atc5']}"
    assert feat["dup_atc4"] == 1.0, f"dup_atc4 expected 1, got {feat['dup_atc4']}"


def test_dup_atc3_same_prefix4():
    """4자리 prefix 동일, 5자리 다름 → dup_atc3 = 1, dup_atc4 = 0."""
    # A10BA02 vs A10BB01 — prefix 4자리 'A10B' 동일, 5자리 다름
    feat = _build_feat(["A10BA02", "A10BB01"])
    assert feat["dup_atc4"] == 0.0, f"dup_atc4 should be 0, got {feat['dup_atc4']}"
    assert feat["dup_atc3"] == 1.0, f"dup_atc3 expected 1, got {feat['dup_atc3']}"


def test_completely_different_atc_no_dup():
    """ATC 가 완전히 다른 두 약물 → 모든 dup_atcN = 0."""
    feat = _build_feat(["A10BA02", "C09AA01"])
    assert feat["dup_atc5"] == 0.0
    assert feat["dup_atc4"] == 0.0
    assert feat["dup_atc3"] == 0.0


def test_no_label_crossover_codex_2026_05_06():
    """4단계 중복이 dup_atc5 로 잘못 올라가지 않음 — Codex 2026-05-06 회귀 가드."""
    feat = _build_feat(["A10BA02", "A10BA03"])
    # 이전 버그: dup_atc5 = 1 (cnt4=5-prefix 결과가 dup_atc5 에 잘못 할당됨)
    assert feat["dup_atc5"] == 0.0, "회귀: ATC 4단계 중복이 dup_atc5 에 잘못 할당됨"
    assert feat["dup_atc4"] == 1.0
