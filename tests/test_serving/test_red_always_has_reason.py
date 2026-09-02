"""Red 등급에는 반드시 사유가 붙는다 — Round 15 재검토 조건 K2.

플래그 ON 재측정에서 실청구 상위 300명 중 33명이 **사유 0건인 채로 Red** 였다
(플래그 OFF 에서는 0명). 원인은 등급 산출부가 risk_grade 만 "Red" 로 바꾸고
사유 목록에 아무것도 남기지 않는 데 있었다. 규칙 경로(TOP01~TOP10)만 사유를
남기므로, 규칙이 발화하지 않고 등급 조건만 성립한 환자는 근거 없이 즉각 개입
대상이 된다.

이름 해소 플래그를 켜면 `_has_high_risk_drug` 가 참이 되어 이 경로가 본격적으로
열리므로, 플래그 활성(A4)의 차단 항목이기도 하다. A2 에서 고친 조기 반환과
같은 계열 — 등급은 그대로 두고 사유만 채운다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rules.safety_net import SafetyNet


@pytest.fixture(scope="module")
def sn(tmp_path_factory):
    """DDI 매트릭스·약물 인덱스 없이 규칙 경로만 — 등급 조건을 고립시킨다."""
    t = tmp_path_factory.mktemp("sn")
    return SafetyNet(ddi_matrix_path=t / "absent_ddi.parquet",
                     drug_index_path=t / "absent_index.parquet")


# 고위험약(마스터 목록)과 무해한 채움 약물
_HIGH_RISK = "Digoxin"
_FILLER = [f"Drug{i}" for i in range(1, 20)]


def _assess(sn, drugs, age=None, renal=False, hepatic=False, count=None):
    return sn.assess(
        drugs=drugs,
        patient_age=age,
        concurrent_drug_count=count if count is not None else len(drugs),
        has_renal_risk=renal,
        has_hepatic_risk=hepatic,
    )


def test_polypharmacy_high_risk_red_carries_a_reason(sn):
    """10종 이상 + 고위험약 → Red 이고 사유가 있어야 한다."""
    res = _assess(sn, [_HIGH_RISK] + _FILLER[:11])

    assert res.risk_grade == "Red", f"등급 조건이 성립하지 않았다 — {res.risk_grade}"
    assert res.triggered_rules, "Red 인데 사유가 0건이다 — 약사가 근거 없이 개입 지시를 받는다"
    assert any("GRADE_POLYPHARMACY_HIGH_RISK" in r for r in res.triggered_rules), (
        f"해당 조건의 사유가 없다 — {res.triggered_rules}"
    )


def test_elderly_organ_risk_red_carries_a_reason(sn):
    """75세 이상 + 5종 이상 + 신/간 위험 → Red 이고 사유가 있어야 한다."""
    res = _assess(sn, _FILLER[:5], age=78, renal=True)

    assert res.risk_grade == "Red", f"등급 조건이 성립하지 않았다 — {res.risk_grade}"
    assert any("GRADE_ELDERLY_ORGAN_RISK" in r for r in res.triggered_rules), (
        f"해당 조건의 사유가 없다 — {res.triggered_rules}"
    )


@pytest.mark.parametrize(
    "label, kwargs",
    [
        ("10종+고위험약", dict(drugs=[_HIGH_RISK] + _FILLER[:11])),
        ("75세+5종+신위험", dict(drugs=_FILLER[:5], age=78, renal=True)),
        ("75세+5종+간위험", dict(drugs=_FILLER[:5], age=80, hepatic=True)),
        ("Triple Whammy", dict(drugs=["Enalapril", "Spironolactone", "Celecoxib"])),
        ("QT 3종", dict(drugs=["Haloperidol", "Levofloxacin", "Ondansetron"])),
        ("고위험약+대량", dict(drugs=["Lithium carbonate"] + _FILLER[:15], age=80, renal=True)),
    ],
)
def test_every_red_path_leaves_a_reason(sn, label, kwargs):
    """Red 로 가는 모든 경로가 사유를 남긴다 — 이것이 불변식이다."""
    res = _assess(sn, **kwargs)

    if res.risk_grade != "Red":
        pytest.skip(f"{label}: 이 입력이 더 이상 Red 를 만들지 않는다 (등급 규칙 변경 확인 필요)")

    assert res.triggered_rules, (
        f"{label}: Red 인데 사유가 0건이다. 등급을 올리는 조건은 반드시 사유를 남겨야 한다 — "
        "그렇지 않으면 약사가 근거 없이 즉각 개입 지시를 받는다."
    )


def test_reasons_do_not_change_the_grade(sn):
    """사유 추가가 등급을 바꾸지 않는다 — A2 와 같은 불변식.

    Yellow·Green·Normal 로 남아야 할 입력이 사유 때문에 올라가면 안 된다.
    """
    normal = _assess(sn, ["Drug1", "Drug2"])
    assert normal.risk_grade in ("Normal", "Green"), (
        f"무해 입력의 등급이 올라갔다 — {normal.risk_grade}, {normal.triggered_rules}"
    )
    assert not any("GRADE_" in r for r in normal.triggered_rules), (
        f"조건이 성립하지 않았는데 등급 사유가 붙었다 — {normal.triggered_rules}"
    )


def test_grade_reason_is_not_duplicated(sn):
    """같은 조건이 두 번 평가돼도 사유는 한 번만 남는다."""
    res = _assess(sn, [_HIGH_RISK] + _FILLER[:11])
    poly = [r for r in res.triggered_rules if "GRADE_POLYPHARMACY_HIGH_RISK" in r]
    assert len(poly) == 1, f"사유가 중복 기록됐다 — {res.triggered_rules}"
