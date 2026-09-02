"""A3 — 실 EDI만 담은 HTTP 요청에서 Top-10 규칙이 발화하는지 잠근다.

`test_safety_net_edi_resolution.py` 는 해소 경로 자체와 TOP01·TOP09 를 다룬다.
이 파일은 나머지 **발화 가능 규칙(TOP02·06·07·08)** 을 HTTP 경로로 고정하고,
**미발화 4종(TOP03·04·05·10)** 을 잔여로 명시적으로 못 박는다. 중복하지 않는다.

분류 근거와 원인 규명은 `docs/plans/2026-09-02-a3-발화가능성-실측.md`.

여기의 EDI 코드는 모두 2024-07-01 하루치 청구 원본에 **실제로 등장**하는 값이다.
합성 코드가 아니므로 참조DB 연결이 끊기면 조용히 통과하지 않고 드러난다.

────────────────────────────────────────────────────────────────────────────
미발화 4종 테스트의 의미 (중요)

아래 `test_residual_*` 는 "발화하지 않음"을 통과 조건으로 삼는다. 이는 현재
상태를 승인하는 것이 아니라 **안전 공백을 잠가 두는 것**이다. 해소 결함
RS1·RS2·RS3 중 하나라도 고쳐지면 이 테스트들이 **실패한다.** 그때 할 일은
테스트를 지우는 것이 아니라 잔여 목록에서 해당 규칙을 빼고 발화 테스트로
승격시키는 것이다. 실패 메시지가 그것을 지시한다.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import sys
import threading
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rules.safety_net import SafetyNet
from scripts.etl.code_standardizer import CodeStandardizer
from serving.predictor import (
    EDI_NAME_RESOLUTION_ENV,
    HybridPredictor,
    RequestFeatureBuilder,
)

from tests.test_serving.test_safety_net_edi_resolution import _require_real_data

# ── 실 EDI 코드 (2024-07-01 하루치에 실제 등장) ────────────────────────────
# 발화 가능 규칙의 구성 약물
EDI_CLOPIDOGREL   = "642503760"  # wk 136901ATB → Clopidogrel
EDI_S_PANTOPRAZOLE = "650203940"  # wk 519202ATE → S-Pantoprazole (ppi_cyp2c19)
EDI_ESCITALOPRAM  = "651903320"  # wk 474801ATB → Escitalopram (ssri)
EDI_LINEZOLID     = "642404000"  # wk 412903ATB → Linezolid (maoi)
EDI_FROVATRIPTAN  = "644703310"  # wk 509501ATB → Frovatriptan (triptan)
EDI_LITHIUM       = "642200390"  # wk 184701ATB → Lithium carbonate
EDI_CELECOXIB     = "647304120"  # wk 347702ACH → Celecoxib (nsaids)

# 미발화 4종의 구성 약물 — 처방은 실제로 있으나 이름 해소가 실패한다
EDI_ENALAPRIL     = "660700970"  # wk 151601ATB → Enalapril (acei) — 해소됨
EDI_SPIRONOLACTONE = "669800160"  # wk 231101ATB → 해소 실패 (RS1)
EDI_DIGOXIN       = "640000090"  # wk 144801ATB → Digoxin — 해소됨
EDI_AMIODARONE    = "652101250"  # wk 107401ATB → 해소 실패 (RS1)
EDI_METHOTREXATE  = "642101470"  # wk 192101ATB → 해소 실패 (RS1)
EDI_TRIMETHOPRIM  = "643900680"  # wk 311500ATB → Trimethoprim — 해소됨
EDI_ATORVASTATIN  = "648101430"  # wk 111501ATB → 해소 실패 (RS1)
EDI_CLARITHROMYCIN = "669804360"  # wk 134901ATB → 해소 실패 (RS1)
# 복합제 — 성분에 스타틴이 있으나 첫 성분 이름만 반환된다 (RS2)
EDI_STATIN_COMBO  = "073400160"  # wk 472300ATB → 성분 [atorvastatin, amlodipine], 해소명 Amlodipine


@pytest.fixture(scope="module")
def real_standardizer():
    if not (ROOT / "data/processed/edi_to_wk.parquet").exists():
        _require_real_data("실 참조DB")
    return CodeStandardizer()


@pytest.fixture
def http_client(real_standardizer, tmp_path, monkeypatch):
    """실 참조DB + 플래그 ON 상태의 FastAPI TestClient."""
    monkeypatch.setenv(EDI_NAME_RESOLUTION_ENV, "1")
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
    # DDI 매트릭스·약물 인덱스를 비워 Top-10 규칙 경로만 남긴다 (A3 의 대상)
    pred._safety_net = SafetyNet(
        ddi_matrix_path=tmp_path / "absent_ddi.parquet",
        drug_index_path=tmp_path / "absent_index.parquet",
    )
    pred._dup_detector = None

    with TestClient(app, raise_server_exceptions=False) as client:
        pred_module._predictor = pred
        yield client


def _body(patient_id: str, edis: list[str]) -> dict:
    """약물명·ATC 없이 EDI 만 담은 요청 — 실 청구 파이프라인의 기본형."""
    return {
        "patient_id": patient_id,
        "patient_age": 72,
        "drugs": [
            {"edi_code": e, "total_days": 14, "start_date": "2024-07-01"} for e in edis
        ],
    }


def _reasons(client, patient_id: str, edis: list[str]) -> list[str]:
    r = client.post("/predict", json=_body(patient_id, edis))
    assert r.status_code == 200, r.text
    return r.json()["risk_reasons"]


# ─────────────────────────────────────────────────────────────────────────────
# 발화 가능 4종 — TOP01·TOP09 는 기존 파일이 다룬다
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "rule_id, edis, label",
    [
        ("TOP02", [EDI_CLOPIDOGREL, EDI_S_PANTOPRAZOLE], "clopidogrel + PPI(CYP2C19)"),
        ("TOP06", [EDI_ESCITALOPRAM, EDI_LINEZOLID], "SSRI + MAOI"),
        ("TOP07", [EDI_ESCITALOPRAM, EDI_FROVATRIPTAN], "SSRI + triptan"),
        ("TOP08", [EDI_LITHIUM, EDI_CELECOXIB], "lithium + NSAID"),
    ],
)
def test_edi_only_http_fires(http_client, rule_id, edis, label):
    """EDI 만 담긴 HTTP 요청에서 해당 규칙이 발화해야 한다."""
    reasons = _reasons(http_client, f"P-{rule_id}", edis)
    assert any(rule_id in x for x in reasons), (
        f"{rule_id}({label}) 가 EDI-only HTTP 경로에서 미발화 — reasons={reasons}"
    )


def test_batch_route_shares_the_same_resolution_path(http_client):
    """배치 경로도 같은 해소 경로를 지나야 한다 (단건만 고치는 회귀 방지)."""
    body = {"requests": [_body("P-BATCH", [EDI_ESCITALOPRAM, EDI_FROVATRIPTAN])]}
    r = http_client.post("/predict/batch", json=body)
    assert r.status_code == 200, r.text
    reasons = r.json()["results"][0]["risk_reasons"]
    assert any("TOP07" in x for x in reasons), f"배치 경로에서 TOP07 미발화 — {reasons}"


# ─────────────────────────────────────────────────────────────────────────────
# 미발화 4종 — 잔여. 발화하게 되면 이 테스트가 실패하며, 그것이 신호다.
# ─────────────────────────────────────────────────────────────────────────────

_PROMOTE = (
    "{rid} 가 발화했다. 해소 결함({defect})이 고쳐졌다는 뜻이다. "
    "이 테스트를 지우지 말고, 잔여 목록(docs/plans/2026-09-02-a3-발화가능성-실측.md §3)에서 "
    "{rid} 를 빼고 위 test_edi_only_http_fires 파라미터로 승격시킬 것."
)


@pytest.mark.parametrize(
    "rule_id, edis, defect, label",
    [
        ("TOP03", [EDI_ENALAPRIL, EDI_SPIRONOLACTONE, EDI_CELECOXIB], "RS1",
         "Triple Whammy — K보존이뇨제가 이름으로 해소되지 않는다"),
        ("TOP04", [EDI_DIGOXIN, EDI_AMIODARONE], "RS1",
         "digoxin + amiodarone — amiodarone 이 해소되지 않는다"),
        ("TOP05", [EDI_METHOTREXATE, EDI_TRIMETHOPRIM], "RS1",
         "methotrexate + trimethoprim — methotrexate 가 해소되지 않는다"),
        ("TOP10", [EDI_ATORVASTATIN, EDI_CLARITHROMYCIN], "RS1",
         "statin + macrolide — 양쪽 모두 해소되지 않는다"),
    ],
)
def test_residual_does_not_fire(http_client, rule_id, edis, defect, label):
    """잔여 4종은 현재 발화하지 않는다 — 안전 공백을 잠가 둔다."""
    reasons = _reasons(http_client, f"P-RES-{rule_id}", edis)
    assert not any(rule_id in x for x in reasons), (
        _PROMOTE.format(rid=rule_id, defect=defect) + f" (reasons={reasons}, {label})"
    )


@pytest.mark.parametrize(
    "edi, expect_resolved, who",
    [
        # 잔여 규칙의 "해소되는 쪽" — 경로가 살아 있음을 증명한다
        (EDI_ENALAPRIL, True, "TOP03 acei"),
        (EDI_CELECOXIB, True, "TOP03 nsaid"),
        (EDI_DIGOXIN, True, "TOP04 digoxin"),
        (EDI_TRIMETHOPRIM, True, "TOP05 trimethoprim"),
        # 잔여 규칙의 "해소되지 않는 쪽" — 미발화의 실제 원인 (RS1)
        (EDI_SPIRONOLACTONE, False, "TOP03 k보존이뇨제"),
        (EDI_AMIODARONE, False, "TOP04 amiodarone"),
        (EDI_METHOTREXATE, False, "TOP05 methotrexate"),
        (EDI_ATORVASTATIN, False, "TOP10 statin"),
        (EDI_CLARITHROMYCIN, False, "TOP10 macrolide"),
    ],
)
def test_residual_cause_is_resolution_failure(real_standardizer, edi, expect_resolved, who):
    """잔여의 원인이 **이름 해소 실패**임을 코드 단위로 고정한다.

    위 `test_residual_does_not_fire` 는 응답이 비어 있어도 통과할 수 있다.
    이 테스트가 그 구멍을 막는다 — 같은 요청 안에서 한쪽은 해소되고
    다른 쪽만 해소되지 않는다는 것이 미발화의 실제 원인이다.
    """
    _, name = real_standardizer.lookup_edi(edi)
    if not name:
        wk = real_standardizer.get_wk(edi)
        if wk:
            _, name = real_standardizer.lookup_wk(wk)

    if expect_resolved:
        assert name, (
            f"{who}({edi}) 가 해소되지 않았다 — 해소 경로 자체가 죽었을 수 있다. "
            "이 상태면 잔여 테스트의 통과는 근거가 없다."
        )
    else:
        assert not name, (
            f"{who}({edi}) 가 이제 {name!r} 로 해소된다 — RS1 이 고쳐졌다는 뜻이다. "
            "해당 규칙을 잔여 목록에서 빼고 발화 테스트로 승격시킬 것."
        )


def test_residual_rs2_composite_drops_the_statin_component(real_standardizer):
    """RS2 — 복합제가 첫 성분 이름만 반환해 스타틴 성분이 소실된다.

    이 코드의 성분에는 스타틴이 있는데 해소된 이름에는 없다. 하루치에서 이런
    사례가 75건이며, 스타틴이 해소에 성공한 건수 전량이 여기 해당한다.
    RS2 가 고쳐지면 이 테스트가 실패한다 — 그때 TOP10 잔여도 함께 재검토할 것.
    """
    wk = real_standardizer.get_wk(EDI_STATIN_COMBO)
    assert wk, "복합제 EDI 의 주성분코드를 찾지 못했다 — 이 픽스처의 전제가 깨졌다"

    components = real_standardizer._master.get_components(wk)
    assert any("statin" in c.lower() for c in components), (
        f"이 코드는 더 이상 스타틴 복합제가 아니다 — 픽스처를 갱신할 것. 성분={components}"
    )

    _, name = real_standardizer.lookup_wk(wk)
    assert name, "복합제가 아예 해소되지 않았다 — 이 픽스처의 전제가 깨졌다"
    assert "statin" not in name.lower(), (
        f"복합제가 스타틴 이름으로 해소됐다 — RS2 가 고쳐졌다는 뜻이다. "
        f"잔여 목록에서 TOP10 을 재검토할 것. (성분={components}, 해소명={name!r})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 기본 OFF 불변식 — A1 병합 조건
# ─────────────────────────────────────────────────────────────────────────────

def test_flag_off_by_default_keeps_top10_silent(real_standardizer, tmp_path, monkeypatch):
    """플래그를 켜지 않으면 EDI-only 요청에서 Top-10 은 발화하지 않는다.

    A1 병합 대상의 기본 배포 상태가 main 과 동등함을 고정한다. 이 테스트가
    실패하면 플래그가 기본 활성으로 바뀐 것이며, 개입 대상이 조용히 늘어난다.
    """
    monkeypatch.delenv(EDI_NAME_RESOLUTION_ENV, raising=False)
    from serving.schemas import DrugItem, PredictRequest

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

    req = PredictRequest(
        patient_id="P-FLAG-OFF",
        patient_age=72,
        drugs=[
            DrugItem(edi_code=EDI_ESCITALOPRAM, total_days=14, start_date=date(2024, 7, 1)),
            DrugItem(edi_code=EDI_FROVATRIPTAN, total_days=14, start_date=date(2024, 7, 1)),
        ],
    )
    res = pred.predict(req)

    assert not any("TOP07" in r for r in res.risk_reasons), (
        f"플래그 OFF 인데 TOP07 이 발화했다 — 기본 배포 상태가 바뀌었다. "
        f"reasons={res.risk_reasons}"
    )
