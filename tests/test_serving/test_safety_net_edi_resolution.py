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
from serving.predictor import (
    EDI_NAME_RESOLUTION_ENV,
    _run_duplicate_detector,
    RISK_FLAG_ATC_ENV,
    HybridPredictor,
    RequestFeatureBuilder,
    _detect_risk_flags,
)
from serving.schemas import DrugItem, PredictRequest, RiskLevel

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
def predictor(standardizer, tmp_path, monkeypatch):
    """ML/계층 미적재 + 실제 SafetyNet(Top-10 규칙만) HybridPredictor.

    EDI 이름 해소는 기본 비활성이므로 기능을 보는 테스트에서는 켜 준다.
    기본값 동작은 별도 테스트가 검증한다.
    """
    monkeypatch.setenv(EDI_NAME_RESOLUTION_ENV, "1")
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
_REAL_EDI_ASPIRIN  = "054801360"   # wk 111001ATE → Aspirin (인덱스 엔트리에 ATC 없음)


@pytest.fixture(scope="module")
def real_standardizer():
    """저장소의 실제 참조DB를 쓰는 CodeStandardizer (모듈 스코프 — 적재 비용이 크다)."""
    if not Path(ROOT / "data/processed/edi_to_wk.parquet").exists():
        pytest.skip("실 참조DB 없음 — 운영 경로 테스트 생략")
    return CodeStandardizer()


@pytest.fixture
def real_predictor(real_standardizer, tmp_path, monkeypatch):
    monkeypatch.setenv(EDI_NAME_RESOLUTION_ENV, "1")
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


def test_wk_fallback_does_not_write_joined_atc(real_standardizer, monkeypatch):
    """wk 폴백은 약물명만 채우고 `atc_code` 는 오염시키지 않아야 한다.

    `lookup_wk` 의 ATC 는 복합제에서 `|`·`,` 로 결합된 문자열이고, 소비처
    (`_detect_risk_flags`, `_build_ddi_alerts`)는 `startswith` 로 파싱하므로
    결합 문자열이 들어가면 첫 원소에만 우연히 매칭되는 조용한 오작동이 된다.
    """
    monkeypatch.setenv(EDI_NAME_RESOLUTION_ENV, "1")
    builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=real_standardizer)
    drugs = [DrugItem(edi_code=_REAL_EDI_WARFARIN, total_days=30, start_date=date(2024, 7, 1))]

    builder.resolve_codes(drugs)

    assert drugs[0].drug_name == "Warfarin", f"wk 폴백이 약물명을 채우지 않았다 — {drugs[0].drug_name}"
    atc = drugs[0].atc_code
    assert atc is None or ("|" not in atc and "," not in atc), (
        f"결합 ATC 문자열이 atc_code 에 기록되었다 — {atc!r}"
    )


def test_duplicate_detector_actually_fires_and_never_lowers_grade(standardizer, tmp_path, monkeypatch):
    """중복탐지가 **실제로 발화하는** 입력에서 등급이 내려가지 않아야 한다.

    직전 판의 테스트는 TOP01 이 발화하는 요청을 썼기 때문에, 탐지기가 아무것도 하지
    않아도 통과했다(agy·codex-terra 공통 지적). 이번에는 세 가지를 바꿨다.

    1. TOP01 이 걸리지 않는 요청을 쓴다 — 등급 상승 요인이 중복 하나뿐이어야 한다.
    2. `dup_count > 0` 을 직접 단언한다 — 탐지기가 실제로 일했는지 확인한다.
    3. 그 상태에서 등급이 baseline 보다 낮아지지 않음을 확인한다.
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

    monkeypatch.setenv(EDI_NAME_RESOLUTION_ENV, "1")
    # 동일 ATC 두 건 — TOP01 대상이 아니며 중복만 성립한다.
    def _req():
        return PredictRequest(
            patient_id="P-DUP", patient_age=70,
            drugs=[
                DrugItem(edi_code="900000501", atc_code="L01FG01",
                         drug_name="Bevacizumab alpha", total_days=30,
                         start_date=date(2024, 7, 1)),
                DrugItem(edi_code="900000502", atc_code="L01FG01",
                         drug_name="Bevacizumab beta", total_days=30,
                         start_date=date(2024, 7, 1)),
            ],
        )

    # 탐지기가 실제로 일했는지 먼저 확인 — 이것이 없으면 아래 단언이 무의미하다
    drugs = _req().drugs
    dup_count, dup_reasons = _run_duplicate_detector(drugs, dd_instance=DuplicateDetector())
    assert dup_count > 0, (
        f"중복탐지기가 발화하지 않아 이 테스트는 아무것도 검증하지 못한다 — "
        f"count={dup_count}, reasons={dup_reasons}"
    )

    without = _build(None).predict(_req())
    with_dup = _build(DuplicateDetector()).predict(_req())

    assert with_dup.risk_level.order >= without.risk_level.order, (
        f"중복탐지 활성화가 등급을 낮췄다 — {without.risk_level} → {with_dup.risk_level}"
    )
    assert any("중복" in r for r in with_dup.risk_reasons), (
        f"중복 사유가 응답에 실리지 않았다 — {with_dup.risk_reasons}"
    )


def test_duplicate_detection_is_inert_for_real_edi_only_requests(real_standardizer, monkeypatch):
    """실 EDI-only 요청에서는 중복탐지가 동작하지 않는다 — 알려진 공백을 고정한다.

    `DuplicateDetector.detect()` 는 **ATC 가 있는 약물만** 엔트리로 만든다. 그런데
    `resolve_codes()` 의 wk 폴백은 약물명만 채우고 ATC 는 채우지 않으며(복합제에서
    파이프·콤마 결합 문자열이 되어 `startswith` 소비처를 조용히 오작동시키므로),
    `lookup_edi` 는 실 EDI 를 하나도 해소하지 못한다.

    따라서 같은 성분으로 해소되는 실 EDI 두 건을 넣어도 탐지 결과는 0이다. 이는
    플래그를 켜도 달라지지 않는다 — 리뷰에서 제기된 "해소된 복합 ATC 가 탐지기를
    깨뜨린다"는 위험이 **도달 불가**임을 뜻하기도 한다.
    """
    from rules.duplicate_detector import DuplicateDetector

    monkeypatch.setenv(EDI_NAME_RESOLUTION_ENV, "1")
    builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=real_standardizer)
    # 051500081 / 051500082 는 둘 다 Bevacizumab 으로 해소된다
    drugs = [DrugItem(edi_code=e, total_days=30, start_date=date(2024, 7, 1))
             for e in ("051500081", "051500082")]
    builder.resolve_codes(drugs)

    assert [d.drug_name for d in drugs] == ["Bevacizumab", "Bevacizumab"], (
        f"픽스처 전제 붕괴 — {[d.drug_name for d in drugs]}"
    )
    assert all(d.atc_code is None for d in drugs), "wk 폴백이 ATC 를 채웠다"

    dup_count, _ = _run_duplicate_detector(drugs, dd_instance=DuplicateDetector())

    assert dup_count == 0, (
        f"공백이 메워졌다면 이 테스트를 갱신하라 — count={dup_count}"
    )

def test_missing_lookup_wk_is_logged_not_silent(caplog, monkeypatch):
    """`lookup_wk` 없는 표준화기에서 해소를 건너뛸 때 조용히 넘어가면 안 된다.

    이 조합은 실 EDI 에 대해 약물명 해소율을 0% 로 되돌리고, 그 상태에서도 API 는
    정상 응답을 낸다. 경보가 사라졌다는 신호가 어디에도 남지 않으면 운영에서
    무경보와 무위험을 구별할 수 없다.
    """
    monkeypatch.setenv(EDI_NAME_RESOLUTION_ENV, "1")

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


def test_real_warfarin_aspirin_pair_fires_top01(real_predictor):
    """실 Warfarin + 실 Aspirin 처방에서 TOP01 이 발화해야 한다.

    `054801360` 은 성분 `aspirin`(DDI ID `D000452`)으로 해소되지만 그 인덱스
    엔트리에는 ATC 코드가 없다. `lookup_wk` 가 `if atc_list:` 로 반환을 게이팅하면
    확보된 약물명까지 함께 버려져, 78세 환자의 와파린+아스피린 병용이 `NORMAL` 로
    응답된다. 실측상 이 형태의 손실은 하루치 고유 EDI 15,017개 중 429건이다.
    """
    req = PredictRequest(
        patient_id="P-WARF-ASA",
        patient_age=78,
        drugs=[
            DrugItem(edi_code=_REAL_EDI_WARFARIN, total_days=30, start_date=date(2024, 7, 1)),
            DrugItem(edi_code=_REAL_EDI_ASPIRIN, total_days=14, start_date=date(2024, 7, 1)),
        ],
    )

    res = real_predictor.predict(req)

    assert [d.drug_name for d in req.drugs] == ["Warfarin", "Aspirin"], (
        f"ATC 없는 엔트리의 약물명이 폐기됐다 — {[d.drug_name for d in req.drugs]}"
    )
    assert any("TOP01" in r for r in res.risk_reasons), (
        f"와파린+아스피린 병용에서 TOP01 미발화 — risk_reasons={res.risk_reasons}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 신/간기능 위험 플래그 — 이름 키워드로는 잡히지 않고 ATC 로만 잡히는 약물이 있다.
# `_RENAL_RISK_KEYWORDS` 는 대표 NSAID 만 담고 있어 aceclofenac·loxoprofen·
# polmacoxib 같은 실제 처방 NSAID 를 놓친다. ATC 접두 `M01A` 는 이들을 포함한다.
# 실측: 하루치 고유 EDI 15,017개 중 신기능 위험약이 이름만으로 340건, ATC 까지
# 보면 548건(+293).
# ─────────────────────────────────────────────────────────────────────────────

_REAL_EDI_ACECLOFENAC = "052400690"   # wk 100901ATB → Aceclofenac, ATC M01AB16
# filler 는 신·간기능 신호가 이름·ATC 어느 쪽으로도 없는 약물만 쓴다.
# (naproxen 같은 키워드 약물을 섞으면 테스트가 엉뚱한 이유로 통과한다)
_REAL_EDI_FILLER = ["050400040", "051500021", "051500101", "051500121"]


def test_risk_flags_detect_nsaid_by_atc_when_name_keyword_misses(real_standardizer, monkeypatch):
    """플래그를 켜면 이름 키워드에 없는 NSAID 도 ATC 접두로 신기능 위험을 인지한다."""
    monkeypatch.setenv(EDI_NAME_RESOLUTION_ENV, "1")
    monkeypatch.setenv(RISK_FLAG_ATC_ENV, "1")
    builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=real_standardizer)
    drugs = [DrugItem(edi_code=_REAL_EDI_ACECLOFENAC, total_days=14,
                      start_date=date(2024, 7, 1))]
    builder.resolve_codes(drugs)

    assert drugs[0].drug_name == "Aceclofenac", f"픽스처 전제 붕괴 — {drugs[0].drug_name}"
    assert not any(k in "aceclofenac" for k in ("ibuprofen", "naproxen", "diclofenac")), (
        "이 약물이 키워드 목록에 들어왔다면 테스트 전제가 무의미해진다"
    )

    has_renal, _ = builder.risk_flags(drugs)

    assert has_renal is True, "ATC M01AB16(NSAID)을 신기능 위험으로 인지하지 못했다"


def test_elderly_polypharmacy_with_atc_only_nsaid_is_red(real_predictor, monkeypatch):
    """78세·5종·ATC로만 식별되는 NSAID → SafetyNet 고령+장기기능 Red 조건이 걸려야 한다.

    `_determine_risk_grade` 는 `age >= 75 and drug_count >= 5 and (renal or hepatic)`
    를 Red 로 판정한다. 플래그가 약물명 키워드로만 계산되면 이 조건이 실 청구
    NSAID 처방에서 성립하지 않는다.
    """
    monkeypatch.setenv(RISK_FLAG_ATC_ENV, "1")
    drugs = [DrugItem(edi_code=_REAL_EDI_ACECLOFENAC, total_days=14,
                      start_date=date(2024, 7, 1))]
    drugs += [DrugItem(edi_code=e, total_days=30, start_date=date(2024, 7, 1))
              for e in _REAL_EDI_FILLER]

    res = real_predictor.predict(
        PredictRequest(patient_id="P-ELDERLY-NSAID", patient_age=78, drugs=drugs)
    )

    assert res.risk_level == RiskLevel.RED, (
        f"고령 다제약물 + NSAID 가 Red 로 승격되지 않았다 — {res.risk_level}, "
        f"reasons={res.risk_reasons}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ATC 집합 경로는 기본 비활성이다. 이 경로를 켜면 고령 다제약물 환자의 즉각 개입
# 대상이 실측 27.0% → 37.5%(적격군 기준)로 늘어나므로, 운영 승인 없이 전량 켜지
# 않는다. 플래그가 꺼진 상태에서는 종전 `_detect_risk_flags` 와 결과가 같아야 한다.
# ─────────────────────────────────────────────────────────────────────────────


def test_atc_candidate_path_is_disabled_by_default(real_predictor, monkeypatch):
    """플래그 미설정 시 ATC 집합 경로가 꺼져 있어 등급이 상향되지 않는다."""
    monkeypatch.delenv(RISK_FLAG_ATC_ENV, raising=False)
    drugs = [DrugItem(edi_code=_REAL_EDI_ACECLOFENAC, total_days=14,
                      start_date=date(2024, 7, 1))]
    drugs += [DrugItem(edi_code=e, total_days=30, start_date=date(2024, 7, 1))
              for e in _REAL_EDI_FILLER]

    res = real_predictor.predict(
        PredictRequest(patient_id="P-DEFAULT-OFF", patient_age=78, drugs=drugs)
    )

    assert res.risk_level != RiskLevel.RED, (
        f"기본값에서 ATC 경로가 켜져 있다 — {res.risk_level}"
    )


def test_risk_flags_matches_legacy_path_when_flag_off(real_standardizer, monkeypatch):
    """플래그가 꺼지면 주입 경로와 폴백 `_detect_risk_flags` 의 결과가 같아야 한다.

    두 경로가 서로 다른 ATC 의미를 갖는 채로 공존하면, 같은 요청이 호출 방식에
    따라 다른 신·간 위험 판정을 받는다.
    """
    monkeypatch.delenv(RISK_FLAG_ATC_ENV, raising=False)
    builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=real_standardizer)

    for edis in ([_REAL_EDI_ACECLOFENAC], _REAL_EDI_FILLER,
                 [_REAL_EDI_WARFARIN, _REAL_EDI_ASPIRIN, _REAL_EDI_ACECLOFENAC]):
        drugs = [DrugItem(edi_code=e, total_days=14, start_date=date(2024, 7, 1))
                 for e in edis]
        builder.resolve_codes(drugs)
        assert builder.risk_flags(drugs) == _detect_risk_flags(drugs), (
            f"플래그 off 에서 두 경로가 갈렸다 — edis={edis}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# HTTP 라우트 경유 — 지금까지의 검증은 전부 `predict()` 수준이었다. `/predict` 와
# `/predict/batch` 가 같은 경로를 지나는지는 제출자 진술이었을 뿐 증거가 없었다.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def http_client(real_standardizer, tmp_path, monkeypatch):
    monkeypatch.setenv(EDI_NAME_RESOLUTION_ENV, "1")
    """실제 FastAPI 앱에 EDI 해소가 가능한 예측기를 꽂은 TestClient."""
    from fastapi.testclient import TestClient
    import serving.predictor as pred_module
    from serving.main import app

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

    with TestClient(app, raise_server_exceptions=False) as client:
        pred_module._predictor = pred
        yield client


def _http_body(patient_id: str, edis: list[str]) -> dict:
    return {
        "patient_id": patient_id,
        "patient_age": 72,
        "drugs": [{"edi_code": e, "total_days": 14, "start_date": "2024-07-01"} for e in edis],
    }


def test_http_predict_route_fires_top10_on_edi_only_payload(http_client):
    """`POST /predict` 에 EDI 만 담아 보내도 TOP01 이 발화해야 한다."""
    r = http_client.post("/predict", json=_http_body("P-HTTP", [_REAL_EDI_WARFARIN, _REAL_EDI_ASPIRIN]))

    assert r.status_code == 200, r.text
    reasons = r.json()["risk_reasons"]
    assert any("TOP01" in x for x in reasons), f"HTTP 경로에서 TOP01 미발화 — {reasons}"


def test_http_batch_route_shares_the_same_resolution_path(http_client):
    """`POST /predict/batch` 도 같은 해소 경로를 지나야 한다."""
    body = {"requests": [_http_body("P-B1", [_REAL_EDI_WARFARIN, _REAL_EDI_ASPIRIN])]}
    r = http_client.post("/predict/batch", json=body)

    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert len(results) == 1
    assert any("TOP01" in x for x in results[0]["risk_reasons"]), (
        f"배치 경로에서 TOP01 미발화 — {results[0]['risk_reasons']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 두 플래그는 서로 독립이며 **둘 다 기본 비활성**이다. 실측 기준(적격군 150명 표본,
# 배포 번들 적재, 최종 등급):
#   둘 다 off  → Red 2명   (main 과 동등)
#   이름 해소만 on → Red 56명
#   둘 다 on   → Red 70명
# 가장 큰 상향이 이름 해소 쪽에 있으므로 그것을 통제하지 않는 플래그는 안전장치가 아니다.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def predictor_no_flags(standardizer, tmp_path, monkeypatch):
    """플래그를 전부 끈 예측기 — 기본 배포 상태."""
    monkeypatch.delenv(EDI_NAME_RESOLUTION_ENV, raising=False)
    monkeypatch.delenv(RISK_FLAG_ATC_ENV, raising=False)
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
    pred._dup_detector = None
    return pred


def test_edi_name_resolution_is_disabled_by_default(predictor_no_flags):
    """기본값에서는 EDI-only 요청의 규칙이 발화하지 않는다 — main 과 동등한 동작.

    이 상향(적격군 표본 기준 Red 2 → 56)은 운영 용량 결정 대상이므로, 병합만으로
    켜져서는 안 된다.
    """
    res = predictor_no_flags.predict(_edi_only_request())

    assert res.risk_reasons == [], (
        f"기본값에서 규칙이 발화했다 — risk_reasons={res.risk_reasons}"
    )


def test_named_request_still_works_with_all_flags_off(predictor_no_flags):
    """대조군 — 요청에 약물명이 실려 있으면 플래그와 무관하게 종전대로 발화한다.

    플래그가 차단하는 것은 EDI→약물명 **해소**이지 규칙 자체가 아니다.
    """
    res = predictor_no_flags.predict(_named_request())

    assert any("TOP01" in r for r in res.risk_reasons), (
        f"약물명이 실린 요청까지 막혔다 — risk_reasons={res.risk_reasons}"
    )


def test_atc_flag_is_nested_under_name_resolution(standardizer, tmp_path, monkeypatch):
    """ATC 플래그만 켜도(이름 해소 off) 아무 효과가 없어야 한다.

    두 플래그는 독립이 아니라 **중첩**이다 — 이름 해소가 주 플래그이고 ATC 플래그는
    그 안에서만 동작한다. `atc_candidates()` 는 `resolve_codes()` 와 무관하게 자체적으로
    `get_wk` → `lookup_wk` 를 타므로, 중첩시키지 않으면 주 플래그가 꺼진 배포에서도
    신/간기능 Red 상향이 발생한다(실측: 적격군 150명 표본에서 Red 2 → 40).
    """
    monkeypatch.delenv(EDI_NAME_RESOLUTION_ENV, raising=False)
    monkeypatch.setenv(RISK_FLAG_ATC_ENV, "1")
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
    pred._dup_detector = None

    res = pred.predict(_edi_only_request())

    assert res.risk_reasons == [], (
        f"이름 해소가 꺼졌는데 ATC 플래그만으로 발화했다 — {res.risk_reasons}"
    )


def test_build_legacy_path_features_unchanged_when_flag_off(real_standardizer, monkeypatch):
    """플래그가 꺼지면 `build()` 의 legacy 경로 위험 피처가 main 과 같아야 한다.

    `build()` 는 `predict()` 와 무관하게 `resolve_codes()` 를 호출한다. wk 폴백이
    플래그 밖에 있으면, 플래그가 꺼진 배포에서도 legacy(비 rulefeat.v1) 번들의
    `has_*_risk_drug` 피처가 0 → 비영으로 바뀐다. 이 피처들은 모델 입력이며
    `has_hepatic_risk_drug` 는 배포 모델 Stage1 importance 2위다.

    실측: 실 EDI 처방 300명 기준 main 은 이름 해소 0/300·위험 피처 전량 0,
    폴백이 플래그 밖일 때는 이름 299/300 · hepatic 45 / renal 31 / high 7.
    """
    monkeypatch.delenv(EDI_NAME_RESOLUTION_ENV, raising=False)
    monkeypatch.delenv(RISK_FLAG_ATC_ENV, raising=False)
    builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=real_standardizer)

    # 실 EDI — lookup_edi 로는 해소되지 않고 wk 폴백으로만 이름이 나온다
    req = PredictRequest(
        patient_id="P-LEGACY",
        patient_age=70,
        drugs=[DrugItem(edi_code=e, total_days=30, start_date=date(2024, 7, 1))
               for e in (_REAL_EDI_WARFARIN, _REAL_EDI_ASPIRIN, _REAL_EDI_ACECLOFENAC)],
    )
    _, feat = builder.build(req, feature_names=None, rule_features_active=False)

    assert [d.drug_name for d in req.drugs] == [None, None, None], (
        f"플래그가 꺼졌는데 build() 가 wk 폴백으로 약물명을 채웠다 — "
        f"{[d.drug_name for d in req.drugs]}"
    )
    assert feat["has_renal_risk_drug"] == 0.0, "legacy 경로 신기능 피처가 바뀌었다"
    assert feat["has_hepatic_risk_drug"] == 0.0, "legacy 경로 간기능 피처가 바뀌었다"
    assert feat["has_high_risk_drug"] == 0.0, "legacy 경로 고위험약 피처가 바뀌었다"


def test_atc_flag_alone_does_not_raise_elderly_grade(real_predictor, monkeypatch):
    """B-only 상태에서 고령·5종 환자 등급이 오르지 않아야 한다.

    직전 판의 독립성 테스트는 72세·2약제를 썼기 때문에
    `age >= 75 and drug_count >= 5` 조건 자체를 태우지 못했고, 따라서 독립성을 주장할
    근거가 되지 못했다. 이 테스트는 그 조건을 실제로 태운다.
    """
    monkeypatch.delenv(EDI_NAME_RESOLUTION_ENV, raising=False)
    monkeypatch.setenv(RISK_FLAG_ATC_ENV, "1")
    drugs = [DrugItem(edi_code=_REAL_EDI_ACECLOFENAC, total_days=14,
                      start_date=date(2024, 7, 1))]
    drugs += [DrugItem(edi_code=e, total_days=30, start_date=date(2024, 7, 1))
              for e in _REAL_EDI_FILLER]

    res = real_predictor.predict(
        PredictRequest(patient_id="P-B-ONLY", patient_age=78, drugs=drugs)
    )

    assert res.risk_level != RiskLevel.RED, (
        f"주 플래그가 꺼졌는데 ATC 플래그만으로 Red 가 됐다 — {res.risk_level}"
    )


def test_atc_only_request_untouched_when_flag_off(standardizer, monkeypatch):
    """`atc_code` 는 있고 `drug_name` 이 없는 요청도 기본값에서 main 과 같아야 한다.

    main 의 해소 블록은 `if not d.atc_code:` 로 게이팅되어 ATC 가 실려 있으면 조회
    자체를 건너뛴다. 브랜치가 두 필드를 독립으로 해소하도록 바꾸면서, 주 플래그가
    꺼진 상태에서도 이 형태의 요청에 약물명이 채워지게 됐다. 그러면 legacy 경로의
    `has_*_risk_drug` 피처가 main 과 달라진다.

    이 경로는 4,000명 동등성 측정이 `DrugItem(edi_code=...)` 만 만들었기 때문에
    빠져 있었다.
    """
    monkeypatch.delenv(EDI_NAME_RESOLUTION_ENV, raising=False)
    monkeypatch.delenv(RISK_FLAG_ATC_ENV, raising=False)
    builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=standardizer)
    drugs = [
        DrugItem(edi_code=_EDI_WARFARIN, atc_code="Z99ZZ99",
                 total_days=30, start_date=date(2024, 7, 1)),
        DrugItem(edi_code=_EDI_IBUPROFEN, atc_code="Y88YY88",
                 total_days=7, start_date=date(2024, 7, 1)),
    ]

    builder.resolve_codes(drugs)

    assert [d.drug_name for d in drugs] == [None, None], (
        f"주 플래그가 꺼졌는데 ATC-only 요청에 약물명이 채워졌다 — "
        f"{[d.drug_name for d in drugs]}"
    )
    assert [d.atc_code for d in drugs] == ["Z99ZZ99", "Y88YY88"], "요청에 실린 ATC 가 덮어써졌다"


def _response_fingerprint(res):
    """등급뿐 아니라 사유·알림·subtype·개입까지 담은 비교용 지문."""
    return (
        res.risk_level,
        res.rule_level,
        res.yellow_subtype,
        res.intervention,
        res.action,
        tuple(sorted(res.risk_reasons)),
        tuple(sorted((a.drug_a, a.drug_b, a.severity, a.source) for a in res.ddi_alerts)),
    )


def test_b_only_response_is_identical_to_all_flags_off(real_standardizer, tmp_path, monkeypatch):
    """B-only 상태의 **응답 전체**가 기본값과 같아야 한다 — 등급만이 아니다.

    직전 판의 B-only 테스트는 `!= Red` 만 확인했으므로 Green→Yellow 상향, 사유·알림
    변화, 환자별 Red 교체를 검출하지 못했다. 기본값(둘 다 off)이 main 과 동등함은
    별도로 입증되어 있으므로, B-only 가 기본값과 응답까지 같으면 main 과도 같다.

    코호트에는 `age >= 75 and drug_count >= 5` 조건을 실제로 태우는 요청을 포함한다.
    """
    def _build():
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
        pred._builder = RequestFeatureBuilder(ddi_matrix=None,
                                              code_standardizer=real_standardizer)
        pred._safety_net = SafetyNet(
            ddi_matrix_path=tmp_path / "absent_ddi.parquet",
            drug_index_path=tmp_path / "absent_index.parquet",
        )
        pred._dup_detector = None
        return pred

    cases = [
        # 고령 + 5종 — 신/간기능 Red 조건을 실제로 태운다
        (78, [_REAL_EDI_ACECLOFENAC] + _REAL_EDI_FILLER),
        # 고령 + 5종 + TOP01 쌍
        (80, [_REAL_EDI_WARFARIN, _REAL_EDI_ASPIRIN] + _REAL_EDI_FILLER[:3]),
        # 비고령 — 조건 미충족 대조군
        (60, [_REAL_EDI_ACECLOFENAC] + _REAL_EDI_FILLER),
        # 고령이나 약제 수 부족 — 조건 미충족 대조군
        (80, [_REAL_EDI_ACECLOFENAC, _REAL_EDI_FILLER[0]]),
    ]

    for age, edis in cases:
        def _run():
            req = PredictRequest(
                patient_id="P", patient_age=age,
                drugs=[DrugItem(edi_code=e, total_days=30, start_date=date(2024, 7, 1))
                       for e in edis],
            )
            return _response_fingerprint(_build().predict(req))

        monkeypatch.delenv(EDI_NAME_RESOLUTION_ENV, raising=False)
        monkeypatch.delenv(RISK_FLAG_ATC_ENV, raising=False)
        baseline = _run()

        monkeypatch.setenv(RISK_FLAG_ATC_ENV, "1")   # 주 플래그는 여전히 off
        b_only = _run()

        assert b_only == baseline, (
            f"B-only 응답이 기본값과 다르다 (age={age}, drugs={len(edis)})\n"
            f"  기본값: {baseline}\n  B-only: {b_only}"
        )


def test_atc_candidate_failure_preserves_name_based_flags(monkeypatch):
    """ATC 후보 조회가 예외를 던져도 이름으로 잡힌 위험 신호는 살아야 한다.

    `_risk_flags_from()` 이 넓은 `try` 안에서 `codes |= atc_provider(d)` 를 실행하고
    어떤 예외든 `return False, False` 하면, 이름이나 요청 ATC 로 이미 위험약이
    확인된 환자까지 플래그가 전멸한다. 고령·다제약물 환자가 Red 대신 낮은 등급으로
    응답될 수 있는 침묵 실패다.
    """
    class _Exploding:
        """`lookup_wk` 가 터지는 표준화기 — 운영에서 참조DB 손상·네트워크 실패 상황."""
        def lookup_edi(self, edi):
            return (None, None)

        def get_wk(self, edi):
            return "WK-DUMMY"

        def lookup_wk(self, wk):
            raise RuntimeError("참조DB 조회 실패")

    monkeypatch.setenv(EDI_NAME_RESOLUTION_ENV, "1")
    monkeypatch.setenv(RISK_FLAG_ATC_ENV, "1")
    builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=_Exploding())

    # 이름만으로 신기능 위험이 확정되는 약물 (ibuprofen 은 _RENAL_RISK_KEYWORDS 소속)
    drugs = [DrugItem(edi_code="900000777", drug_name="Ibuprofen",
                      total_days=7, start_date=date(2024, 7, 1))]

    has_renal, has_hepatic = builder.risk_flags(drugs)

    assert has_renal is True, (
        "ATC 후보 조회 예외가 이름 기반 신기능 신호까지 덮었다"
    )
    assert has_hepatic is False


# ─────────────────────────────────────────────────────────────────────────────
# 플래그 관측성 — codex-terra 가 "기본값 off 가 배포에서 강제되는가"를 병합 조건으로
# 걸었다. 환경변수 주입 자체는 배포 거버넌스 문제라 코드가 답할 수 없지만, **실행 중인
# 인스턴스가 어떤 값으로 돌고 있는지**는 코드가 드러낼 수 있다. 그것이 없으면 운영자가
# 확인할 방법이 없다.
# ─────────────────────────────────────────────────────────────────────────────


def test_health_exposes_flag_state(http_client, monkeypatch):
    """`/health` 가 두 플래그의 현재 상태를 드러내야 한다."""
    monkeypatch.delenv(EDI_NAME_RESOLUTION_ENV, raising=False)
    monkeypatch.delenv(RISK_FLAG_ATC_ENV, raising=False)

    r = http_client.get("/health")

    assert r.status_code == 200, r.text
    body = r.json()
    assert "serving_flags" in body, f"플래그 상태가 노출되지 않는다 — keys={list(body)}"
    assert body["serving_flags"] == {
        EDI_NAME_RESOLUTION_ENV: False,
        RISK_FLAG_ATC_ENV: False,
    }, body["serving_flags"]


def test_health_reflects_enabled_flags(http_client, monkeypatch):
    """플래그를 켜면 `/health` 가 그대로 반영해야 한다 — 운영자가 오인하지 않도록."""
    monkeypatch.setenv(EDI_NAME_RESOLUTION_ENV, "1")
    monkeypatch.setenv(RISK_FLAG_ATC_ENV, "1")

    body = http_client.get("/health").json()

    assert body["serving_flags"] == {
        EDI_NAME_RESOLUTION_ENV: True,
        RISK_FLAG_ATC_ENV: True,
    }, body["serving_flags"]


# ─────────────────────────────────────────────────────────────────────────────
# 트립와이어 — "중복탐지 도달 불가"는 코드 구조가 아니라 **데이터 사실**에 기댄다.
#
# `lookup_edi` 가 실 청구 EDI 를 하나도 해소하지 못하기 때문에 `resolve_codes()` 가
# `d.atc_code` 를 채우지 않고, 그래서 중복탐지가 엔트리를 만들지 못한다. 참조DB 를
# 백필하면(예: `config/edi_atc_extra.csv` 신설, D-코드 이름 회수) 이 전제가 무너지고
# 다음 두 판단이 **동시에** 무효가 된다.
#
#   ① "중복탐지 관련 위험은 도달 불가"
#   ② 그 위에 선 `_run_duplicate_detector` fail-safe 미추가 결정
#
# 아무도 알아채지 못한 채 무효화되는 것을 막기 위해, 전제가 깨지는 순간 실패하는
# 검사를 둔다. fable-advisor 10차 지적 — "영구 결정이 아니라 트립와이어가 걸릴
# 때까지의 조건부 결정으로 기록되어야 한다".
# ─────────────────────────────────────────────────────────────────────────────


def test_tripwire_lookup_edi_still_resolves_no_real_edi(real_standardizer):
    """실 EDI 표본에서 `lookup_edi` 해소가 0건이어야 한다 — 아니면 전제가 바뀐 것이다.

    이 테스트가 실패하면 참조DB 가 백필된 것이고, 다음을 재검토해야 한다.
      - 중복탐지가 이제 ATC 를 받는가 → `_run_duplicate_detector` fail-safe 필요성
      - `resolve_codes()` 가 기본값에서 `atc_code` 를 채우기 시작하는가 → 기본값
        동등성 재측정(세 층 sha256)
    """
    import glob
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(ROOT / "data/Raw/records_*.parquet")))
    if not files:
        pytest.skip("실 Raw 데이터 없음 — 트립와이어 검사 생략")

    edis = sorted(set(
        pq.read_table(files[0], columns=["edi_code"]).column("edi_code").to_pylist()
    ))
    resolved = [e for e in edis if any(real_standardizer.lookup_edi(e))]

    assert not resolved, (
        f"`lookup_edi` 가 실 EDI 를 해소하기 시작했다 ({len(resolved)}/{len(edis)}건). "
        f"'중복탐지 도달 불가'와 fail-safe 미추가 판단의 전제가 무너졌다. "
        f"표본: {resolved[:5]}"
    )


def test_tripwire_resolve_codes_leaves_atc_empty_for_real_edi(real_standardizer, monkeypatch):
    """플래그를 켜도 실 EDI 요청의 `atc_code` 가 비어 있어야 한다.

    위 트립와이어의 서빙 쪽 짝이다. `atc_code` 가 채워지기 시작하면 중복탐지가
    엔트리를 만들고, 복합제의 결합 ATC 문자열이 탐지기에 도달하게 된다.
    """
    monkeypatch.setenv(EDI_NAME_RESOLUTION_ENV, "1")
    monkeypatch.setenv(RISK_FLAG_ATC_ENV, "1")
    builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=real_standardizer)
    drugs = [DrugItem(edi_code=e, total_days=30, start_date=date(2024, 7, 1))
             for e in (_REAL_EDI_WARFARIN, _REAL_EDI_ASPIRIN, _REAL_EDI_ACECLOFENAC,
                       "051500081", "051500082")]

    builder.resolve_codes(drugs)

    filled = [(d.edi_code, d.atc_code) for d in drugs if d.atc_code]
    assert not filled, (
        f"실 EDI 요청에 `atc_code` 가 채워졌다 — 중복탐지 도달 가능성이 생겼다: {filled}"
    )
