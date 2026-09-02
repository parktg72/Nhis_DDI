"""실 EDI만 담은 HTTP 요청에서 Top-10 규칙이 발화하는지 잠근다 (A3).

`test_safety_net_edi_resolution.py` 는 해소 경로 자체와 TOP01·TOP09 를 다룬다.
이 파일은 나머지 여덟 규칙을 HTTP 경로로 고정한다. 중복하지 않는다.

분류 근거는 `docs/plans/2026-09-02-a3-발화가능성-실측.md`.

여기의 EDI 코드는 모두 2024-07-01 하루치 청구 원본에 **실제로 등장**하는 값이다.
합성 코드가 아니므로 참조DB 연결이 끊기면 조용히 통과하지 않고 드러난다.

────────────────────────────────────────────────────────────────────────────
잔여 4종의 승격 (2026-09-02)

이 파일은 처음에 TOP03·04·05·10 을 "미발화 잔여" 로 못박고 있었다. 원인은
해소 결함 RS1(성분↔DDI 식별자 미연결)·RS2(복합제 첫 성분만 반환)·RS3(statin
ATC 목록 오류) 였고, 셋을 고치자 넷 다 발화하게 됐다(해소율 60.2% → 80.7%,
발화 6종 → 10종). 잔여 테스트가 설계대로 실패해 승격을 지시했고 그에 따랐다.

아래 `test_resolution_did_not_regress` 가 그 해소를 되돌아가지 못하게 잠근다.
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

# RS1~RS3 수정으로 승격된 네 규칙의 구성 약물 (종전에는 이름 해소가 실패했다)
EDI_ENALAPRIL     = "660700970"  # wk 151601ATB → Enalapril (acei) — 해소됨
EDI_SPIRONOLACTONE = "669800160"  # wk 231101ATB → Spironolactone (RS1 수정으로 해소)
EDI_DIGOXIN       = "640000090"  # wk 144801ATB → Digoxin — 해소됨
EDI_AMIODARONE    = "652101250"  # wk 107401ATB → Amiodarone (RS1 수정으로 해소)
EDI_METHOTREXATE  = "642101470"  # wk 192101ATB → Methotrexate (RS1 수정으로 해소)
EDI_TRIMETHOPRIM  = "643900680"  # wk 311500ATB → Trimethoprim — 해소됨
EDI_ATORVASTATIN  = "648101430"  # wk 111501ATB → Atorvastatin (RS1 수정으로 해소)
EDI_CLARITHROMYCIN = "669804360"  # wk 134901ATB → Clarithromycin (RS1 수정으로 해소)
# 복합제 — 성분에 스타틴이 있다. RS2 수정 전에는 첫 성분 이름만 반환됐다.
EDI_STATIN_COMBO  = "073400160"  # wk 472300ATB → 성분 [atorvastatin, amlodipine]


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
        # RS1~RS3 수정으로 승격된 네 규칙
        ("TOP03", [EDI_ENALAPRIL, EDI_SPIRONOLACTONE, EDI_CELECOXIB], "Triple Whammy"),
        ("TOP04", [EDI_DIGOXIN, EDI_AMIODARONE], "digoxin + amiodarone"),
        ("TOP05", [EDI_METHOTREXATE, EDI_TRIMETHOPRIM], "methotrexate + trimethoprim"),
        ("TOP10", [EDI_ATORVASTATIN, EDI_CLARITHROMYCIN], "statin + macrolide"),
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
# 해소 회귀 방지 — RS1·RS2·RS3 이 되돌아가면 여기서 먼저 실패한다
# ─────────────────────────────────────────────────────────────────────────────

_REGRESS = (
    "{who}({edi}) 가 이름으로 해소되지 않는다. {defect} 가 되돌아갔다는 뜻이며, "
    "그 규칙은 실 청구 요청에서 조용히 무발화한다. 해소 경로를 먼저 고칠 것."
)


@pytest.mark.parametrize(
    "edi, who, defect",
    [
        (EDI_ENALAPRIL, "TOP03 acei", "RS1"),
        (EDI_CELECOXIB, "TOP03 nsaid", "RS1"),
        (EDI_SPIRONOLACTONE, "TOP03 k보존이뇨제", "RS1"),
        (EDI_DIGOXIN, "TOP04 digoxin", "RS1"),
        (EDI_AMIODARONE, "TOP04 amiodarone", "RS1"),
        (EDI_METHOTREXATE, "TOP05 methotrexate", "RS1"),
        (EDI_TRIMETHOPRIM, "TOP05 trimethoprim", "RS1"),
        (EDI_ATORVASTATIN, "TOP10 statin", "RS1"),
        (EDI_CLARITHROMYCIN, "TOP10 macrolide", "RS1"),
    ],
)
def test_resolution_did_not_regress(real_standardizer, edi, who, defect):
    """규칙이 의존하는 약물이 실제로 이름을 얻는지 코드 단위로 고정한다.

    HTTP 발화 테스트는 응답이 비어도 통과할 수 있는 형태가 아니지만, 발화 실패의
    원인이 규칙인지 해소인지는 구분해 주지 않는다. 이쪽이 그 구분을 준다 —
    해소가 깨지면 여기가 먼저 실패한다.

    이 아홉 건은 모두 RS1(성분↔DDI 식별자 미연결) 때문에 해소되지 않던 약물이다.
    `_edi_map` 은 DrugBank ID 로 키잉되는데 DUR D-코드 314개 중 275개가 거기
    없었고, 그 275개가 통째로 무발화했다.
    """
    _, name = real_standardizer.lookup_edi(edi)
    if not name:
        wk = real_standardizer.get_wk(edi)
        if wk:
            _, name = real_standardizer.lookup_wk(wk)

    assert name, _REGRESS.format(who=who, edi=edi, defect=defect)


def test_rs2_composite_keeps_the_statin_component(real_standardizer):
    """RS2 — 복합제의 스타틴 성분이 규칙 입력에서 소실되지 않는다.

    `lookup_wk` 는 `DrugItem.drug_name` 이 스칼라라 첫 성분만 돌려준다. 그것은
    그대로 두고, 규칙 경로가 쓰는 `lookup_wk_names` 가 전 성분을 돌려준다.
    종전에는 스타틴 복합제 해소 성공 75건 **전량**이 병용 성분 이름으로 해소돼
    statin 그룹에 한 건도 들어오지 않았다.
    """
    wk = real_standardizer.get_wk(EDI_STATIN_COMBO)
    assert wk, "복합제 EDI 의 주성분코드를 찾지 못했다 — 이 픽스처의 전제가 깨졌다"

    components = real_standardizer._master.get_components(wk)
    assert any("statin" in c.lower() for c in components), (
        f"이 코드는 더 이상 스타틴 복합제가 아니다 — 픽스처를 갱신할 것. 성분={components}"
    )

    names = real_standardizer.lookup_wk_names(wk)
    assert any("statin" in n.lower() for n in names), (
        f"복합제의 스타틴 성분이 규칙 입력에서 빠졌다 — RS2 가 되돌아갔다. "
        f"성분={components}, 해소명={names}"
    )

    # 대표 이름(스칼라)은 종전 계약 그대로 — 첫 성분이며 스타틴이 아닐 수 있다
    _, primary = real_standardizer.lookup_wk(wk)
    assert primary in names, f"대표 이름이 전 성분 목록에 없다 — {primary!r} not in {names}"


def test_composite_components_reach_the_rules(http_client):
    """복합제만 담긴 EDI-only 요청에서 스타틴 규칙이 발화한다.

    `EDI_STATIN_COMBO` 의 대표 이름은 Amlodipine 이므로, 이 요청이 TOP10 을
    발화시키려면 전 성분이 규칙에 도달해야만 한다.
    """
    reasons = _reasons(http_client, "P-COMBO", [EDI_STATIN_COMBO, EDI_CLARITHROMYCIN])

    assert any("TOP10" in x for x in reasons), (
        f"복합제의 스타틴 성분이 규칙에 도달하지 않았다 — reasons={reasons}"
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
