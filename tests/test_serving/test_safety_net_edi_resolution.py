"""EDI-only 요청에서 Rule Safety Net 이 발화하는지 검증 (보고서 §1.6.1 결함 #13).

`DrugItem` 은 `edi_code` 만 필수이고 `drug_name`/`atc_code` 는 Optional 이므로,
실 청구 파이프라인의 기본형 요청에는 약물명이 없다. 그런데 `SafetyNet.assess()` 는
약물명 목록을 받고 `DrugMatcher` 는 이름(또는 이름으로 찾은 drugbank_id)으로만
그룹 매칭을 한다. 따라서 EDI→약물명 해소가 SafetyNet 호출보다 **먼저** 수행되지
않으면 Top-10 규칙·QT 다중병용 판정·고위험약 판정이 전량 무발화한다.

이 테스트는 그 해소가 `predict()` 안에서 SafetyNet 호출 이전에 일어나는지 본다.
"""
from __future__ import annotations

import logging
import sys
import threading
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rules.safety_net import SafetyNet
from scripts.etl.code_standardizer import CodeStandardizer
from serving.predictor import HybridPredictor, RequestFeatureBuilder
from serving.schemas import DrugItem, PredictRequest

# 테스트 전용 EDI 코드 → 실제 성분명. TOP01(항응고제 + NSAIDs) 구성.
_EDI_WARFARIN = "900000001"
_EDI_IBUPROFEN = "900000002"

# TOP09(QT 연장 3종 이상) 구성 — 보고서 §1.6.1 결함 #14 의 도달성 절반.
_EDI_HALOPERIDOL = "900000011"
_EDI_LEVOFLOXACIN = "900000012"
_EDI_ONDANSETRON = "900000013"


@pytest.fixture
def standardizer(tmp_path):
    """EDI→(atc, name) 만 채운 실제 CodeStandardizer (DrugMaster/edi_wk 는 미사용)."""
    idx = tmp_path / "edi_index.parquet"
    pd.DataFrame(
        [
            {"drug_id": _EDI_WARFARIN, "atc_code": "B01AA03", "drug_name": "Warfarin"},
            {"drug_id": _EDI_IBUPROFEN, "atc_code": "M01AE01", "drug_name": "Ibuprofen"},
            {"drug_id": _EDI_HALOPERIDOL, "atc_code": "N05AD01", "drug_name": "Haloperidol"},
            {"drug_id": _EDI_LEVOFLOXACIN, "atc_code": "J01MA12", "drug_name": "Levofloxacin"},
            {"drug_id": _EDI_ONDANSETRON, "atc_code": "A04AA01", "drug_name": "Ondansetron"},
        ]
    ).to_parquet(idx)
    return CodeStandardizer(
        index_path=idx,
        extra_csv=None,
        master_parquet=tmp_path / "absent_master.parquet",
        ddi_matrix_path=tmp_path / "absent_ddi.parquet",
        edi_wk_path=tmp_path / "absent_edi_wk.parquet",
    )


@pytest.fixture
def predictor(standardizer, tmp_path):
    """ML/계층 미적재 + 실제 SafetyNet(Top-10 규칙만) HybridPredictor."""
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml = MagicMock()
    pred._ml.loaded = False
    pred._ml_lock = threading.Lock()
    pred._hier_lock = threading.RLock()
    pred._hierarchical = None
    pred._ddi_matrix = None
    pred._cyp = None
    pred._std = standardizer
    pred._builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=standardizer)
    # DDI 매트릭스·약물 인덱스 없이 Top-10 규칙 경로만 활성화 (결함의 대상 경로)
    pred._safety_net = SafetyNet(
        ddi_matrix_path=tmp_path / "absent_ddi.parquet",
        drug_index_path=tmp_path / "absent_index.parquet",
    )
    pred._dup_detector = None
    return pred


def _edi_only_request() -> PredictRequest:
    """약물명·ATC 없이 EDI 코드만 담긴 요청 — 실 청구 파이프라인의 기본형."""
    return PredictRequest(
        patient_id="P-EDI-ONLY",
        patient_age=72,
        drugs=[
            DrugItem(edi_code=_EDI_WARFARIN, total_days=30, start_date=date(2024, 7, 1)),
            DrugItem(edi_code=_EDI_IBUPROFEN, total_days=7, start_date=date(2024, 7, 1)),
        ],
    )


def _named_request() -> PredictRequest:
    """동일 처방에 약물명이 실린 요청 — 대조군."""
    return PredictRequest(
        patient_id="P-NAMED",
        patient_age=72,
        drugs=[
            DrugItem(edi_code=_EDI_WARFARIN, drug_name="Warfarin",
                     total_days=30, start_date=date(2024, 7, 1)),
            DrugItem(edi_code=_EDI_IBUPROFEN, drug_name="Ibuprofen",
                     total_days=7, start_date=date(2024, 7, 1)),
        ],
    )


def test_named_request_triggers_top10_anticoagulant_nsaid_rule(predictor):
    """대조군 — 약물명이 실리면 TOP01 이 발화한다(규칙·픽스처 자체는 정상)."""
    res = predictor.predict(_named_request())

    assert any("TOP01" in r for r in res.risk_reasons), (
        f"약물명 요청에서도 TOP01 미발화 — 픽스처 또는 규칙 자체 문제. "
        f"risk_reasons={res.risk_reasons}"
    )


def test_edi_only_request_triggers_top10_anticoagulant_nsaid_rule(predictor):
    """약물명 없이 EDI 만 담긴 요청에서도 TOP01(항응고제+NSAIDs)이 발화해야 한다."""
    res = predictor.predict(_edi_only_request())

    assert any("TOP01" in r for r in res.risk_reasons), (
        f"EDI-only 요청에서 TOP01 미발화 — risk_reasons={res.risk_reasons}"
    )


def test_edi_only_request_triggers_qt_multiple_rule(predictor):
    """EDI 만 담긴 요청에서도 TOP09(QT 연장 3종 이상)가 발화해야 한다.

    QT 규칙은 `safety_net.py` 에 실재하나 약물명 입력을 전제하므로, 해소 시점이
    늦으면 도달 자체가 불가능했다(보고서 §1.6.1 결함 #14).
    """
    req = PredictRequest(
        patient_id="P-QT",
        patient_age=82,
        drugs=[
            DrugItem(edi_code=_EDI_HALOPERIDOL, total_days=14, start_date=date(2024, 7, 1)),
            DrugItem(edi_code=_EDI_LEVOFLOXACIN, total_days=7, start_date=date(2024, 7, 1)),
            DrugItem(edi_code=_EDI_ONDANSETRON, total_days=5, start_date=date(2024, 7, 1)),
        ],
    )

    res = predictor.predict(req)

    assert any("TOP09" in r for r in res.risk_reasons), (
        f"EDI-only 요청에서 TOP09 미발화 — risk_reasons={res.risk_reasons}"
    )


def test_atc_supplied_without_name_still_resolves_and_fires(predictor):
    """`atc_code` 만 실리고 약물명이 없는 요청에서도 규칙이 발화해야 한다.

    `DrugItem` 은 두 필드가 서로 독립인 Optional 이므로 ATC 만 실린 요청은 적법하다.
    해소가 `atc_code` 유무로 게이팅되면 이 형태에서 약물명이 영원히 비고, 매칭기는
    이름 없는 경로가 없으므로 규칙이 침묵한다.
    """
    req = PredictRequest(
        patient_id="P-ATC-ONLY",
        patient_age=72,
        drugs=[
            DrugItem(edi_code=_EDI_WARFARIN, atc_code="B01AA03",
                     total_days=30, start_date=date(2024, 7, 1)),
            DrugItem(edi_code=_EDI_IBUPROFEN, atc_code="M01AE01",
                     total_days=7, start_date=date(2024, 7, 1)),
        ],
    )

    res = predictor.predict(req)

    assert [d.drug_name for d in req.drugs] == ["Warfarin", "Ibuprofen"], (
        f"ATC 가 실렸다는 이유로 약물명 해소가 건너뛰어졌다 — {[d.drug_name for d in req.drugs]}"
    )
    assert any("TOP01" in r for r in res.risk_reasons), (
        f"ATC 만 실린 요청에서 TOP01 미발화 — risk_reasons={res.risk_reasons}"
    )


def test_resolve_codes_preserves_explicitly_supplied_values(standardizer):
    """요청에 실린 ATC/약물명은 해소가 덮어쓰지 않으며, 두 번 호출해도 같다(멱등).

    `build()` 가 갖고 있던 기존 의미를 새 API 위치에 고정하는 특성화 테스트다.
    """
    builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=standardizer)
    drugs = [
        DrugItem(edi_code=_EDI_WARFARIN, atc_code="Z99ZZ99", drug_name="현장기재명",
                 total_days=30, start_date=date(2024, 7, 1)),
        DrugItem(edi_code=_EDI_IBUPROFEN, total_days=7, start_date=date(2024, 7, 1)),
    ]

    builder.resolve_codes(drugs)
    first = [(d.atc_code, d.drug_name) for d in drugs]
    builder.resolve_codes(drugs)

    assert first[0] == ("Z99ZZ99", "현장기재명"), "요청에 실린 값이 덮어써졌다"
    assert first[1] == ("M01AE01", "Ibuprofen"), "미기재 항목이 해소되지 않았다"
    assert [(d.atc_code, d.drug_name) for d in drugs] == first, "두 번째 호출이 결과를 바꿨다"


# ─────────────────────────────────────────────────────────────────────────────
# 실 데이터 경로 — 운영 EDI 코드는 `lookup_edi` 로 해소되지 않는다.
# `drug_name_index.parquet` 이 DrugBank ID 로 키잉되어 있고 `config/edi_atc_extra.csv`
# 는 존재하지 않으므로, 실 EDI 는 edi→wk 브릿지(`lookup_wk`)로만 이름이 나온다.
# ─────────────────────────────────────────────────────────────────────────────

_REAL_EDI_WARFARIN = "645600390"   # wk 249103ATB → Warfarin
_REAL_EDI_NAPROXEN = "053500020"   # wk 199501ATB → Naproxen


@pytest.fixture(scope="module")
def real_standardizer():
    """저장소의 실제 참조DB를 쓰는 CodeStandardizer (모듈 스코프 — 적재 비용이 크다)."""
    if not Path(ROOT / "data/processed/edi_to_wk.parquet").exists():
        pytest.skip("실 참조DB 없음 — 운영 경로 테스트 생략")
    return CodeStandardizer()


@pytest.fixture
def real_predictor(real_standardizer, tmp_path):
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml = MagicMock()
    pred._ml.loaded = False
    pred._ml_lock = threading.Lock()
    pred._hier_lock = threading.RLock()
    pred._hierarchical = None
    pred._ddi_matrix = None
    pred._cyp = None
    pred._std = real_standardizer
    pred._builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=real_standardizer)
    pred._safety_net = SafetyNet(
        ddi_matrix_path=tmp_path / "absent_ddi.parquet",
        drug_index_path=tmp_path / "absent_index.parquet",
    )
    pred._dup_detector = None
    return pred


def test_real_edi_codes_resolve_via_wk_bridge_and_fire(real_predictor):
    """실 청구 EDI 코드만 실린 요청에서 TOP01 이 발화해야 한다.

    이 두 코드는 `lookup_edi` 로는 (None, None) 이며 edi→wk 브릿지로만 이름이 나온다.
    합성 인덱스가 아니라 저장소의 실제 참조DB를 쓴다.
    """
    req = PredictRequest(
        patient_id="P-REAL-EDI",
        patient_age=72,
        drugs=[
            DrugItem(edi_code=_REAL_EDI_WARFARIN, total_days=30, start_date=date(2024, 7, 1)),
            DrugItem(edi_code=_REAL_EDI_NAPROXEN, total_days=7, start_date=date(2024, 7, 1)),
        ],
    )

    res = real_predictor.predict(req)

    assert any("TOP01" in r for r in res.risk_reasons), (
        f"실 EDI 요청에서 TOP01 미발화 — names={[d.drug_name for d in req.drugs]}, "
        f"risk_reasons={res.risk_reasons}"
    )


def test_wk_fallback_does_not_write_joined_atc(real_standardizer):
    """wk 폴백은 약물명만 채우고 `atc_code` 는 오염시키지 않아야 한다.

    `lookup_wk` 의 ATC 는 복합제에서 `|`·`,` 로 결합된 문자열이고, 소비처
    (`_detect_risk_flags`, `_build_ddi_alerts`)는 `startswith` 로 파싱하므로
    결합 문자열이 들어가면 첫 원소에만 우연히 매칭되는 조용한 오작동이 된다.
    """
    builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=real_standardizer)
    drugs = [DrugItem(edi_code=_REAL_EDI_WARFARIN, total_days=30, start_date=date(2024, 7, 1))]

    builder.resolve_codes(drugs)

    assert drugs[0].drug_name == "Warfarin", f"wk 폴백이 약물명을 채우지 않았다 — {drugs[0].drug_name}"
    atc = drugs[0].atc_code
    assert atc is None or ("|" not in atc and "," not in atc), (
        f"결합 ATC 문자열이 atc_code 에 기록되었다 — {atc!r}"
    )


def test_active_duplicate_detector_never_lowers_grade_on_edi_only(standardizer, tmp_path):
    """중복탐지를 실제로 켠 EDI-only 요청에서 등급이 내려가지 않아야 한다.

    Step 0 해소로 중복탐지(Step 2)도 처음으로 해소된 ATC 를 받게 되었다. 이 경로는
    직전 판까지 테스트가 없었다(픽스처가 `_dup_detector=None`). 이 시스템의 결합
    불변식은 단방향 상향(`RiskLevel.max`)이므로, 탐지기를 켜는 것이 등급을 낮추면
    안 된다.
    """
    from rules.duplicate_detector import DuplicateDetector

    def _build(dup_detector):
        pred = HybridPredictor.__new__(HybridPredictor)
        pred._start_time = 0.0
        pred._ml = MagicMock()
        pred._ml.loaded = False
        pred._ml_lock = threading.Lock()
        pred._hier_lock = threading.RLock()
        pred._hierarchical = None
        pred._ddi_matrix = None
        pred._cyp = None
        pred._std = standardizer
        pred._builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=standardizer)
        pred._safety_net = SafetyNet(
            ddi_matrix_path=tmp_path / "absent_ddi.parquet",
            drug_index_path=tmp_path / "absent_index.parquet",
        )
        pred._dup_detector = dup_detector
        return pred

    without = _build(None).predict(_edi_only_request())
    with_dup = _build(DuplicateDetector()).predict(_edi_only_request())

    assert with_dup.risk_level.order >= without.risk_level.order, (
        f"중복탐지 활성화가 등급을 낮췄다 — {without.risk_level} → {with_dup.risk_level}"
    )
    assert any("TOP01" in r for r in with_dup.risk_reasons), (
        f"중복탐지 활성 상태에서 TOP01 이 사라졌다 — risk_reasons={with_dup.risk_reasons}"
    )


def test_missing_lookup_wk_is_logged_not_silent(caplog):
    """`lookup_wk` 없는 표준화기에서 해소를 건너뛸 때 조용히 넘어가면 안 된다.

    이 조합은 실 EDI 에 대해 약물명 해소율을 0% 로 되돌리고, 그 상태에서도 API 는
    정상 응답을 낸다. 경보가 사라졌다는 신호가 어디에도 남지 않으면 운영에서
    무경보와 무위험을 구별할 수 없다.
    """
    class _NoLookupWk:
        """역사적 계약만 구현한 표준화기 — `lookup_wk` 없음."""
        def lookup_edi(self, edi):
            return (None, None)

        def get_wk(self, edi):
            return "WK-DUMMY"

    builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=_NoLookupWk())
    drugs = [DrugItem(edi_code="900000999", total_days=1, start_date=date(2024, 7, 1))]

    with caplog.at_level(logging.WARNING, logger="serving.predictor"):
        builder.resolve_codes(drugs)

    assert drugs[0].drug_name is None, "픽스처 전제가 깨졌다 — 이름이 해소되면 안 된다"
    assert any("lookup_wk" in r.message for r in caplog.records), (
        f"lookup_wk 부재가 로그로 드러나지 않았다 — records={[r.message for r in caplog.records]}"
    )
