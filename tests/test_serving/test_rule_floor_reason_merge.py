"""major DDI 와 중증 트리거가 함께 있을 때 사유가 소실되지 않는지 고정.

`rule_floor` 는 `ddi_major>=1` 이면 즉시 `Y_DDI_MAJOR` 를 돌려주고 그 자리에서 끝냈다.
그래서 triple whammy 를 함께 가진 환자는 약사 전화는 받지만 급성신손상 3제 조합이
`risk_reasons` 에 실리지 않았다 — 등급은 맞고 설명만 사라지는 실패다.

subtype·action 위계는 그대로 둔다(major 가 severe 보다 상위). 바뀌는 것은 사유뿐이다.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving.predictor import RequestFeatureBuilder

_REF = date(2024, 7, 1)


def _builder(**feats):
    """`_rule_namespace` 만 대체한 빌더. 나머지 경로는 건드리지 않는다."""
    b = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=None)
    ns = SimpleNamespace(
        ddi_contraindicated=0, ddi_major=0, triple_whammy=False,
        drug_count=3, has_high_risk_drug=False,
        has_renal_risk_drug=False, has_hepatic_risk_drug=False, age=60,
    )
    for k, v in feats.items():
        setattr(ns, k, v)
    b._rule_namespace = lambda drugs, ref, patient_age=None: ns
    return b


def test_major_and_triple_whammy_keeps_subtype_and_merges_reason():
    """둘 다인 환자 — subtype 은 Y_DDI_MAJOR 유지, 사유에 SEV_TRIPLE_WHAMMY 추가."""
    sub, reasons = _builder(ddi_major=1, triple_whammy=True).rule_floor([], _REF)
    assert sub == "Y_DDI_MAJOR", "major 우선 위계는 바뀌면 안 된다"
    assert "DDI_MAJOR" in reasons
    assert "SEV_TRIPLE_WHAMMY" in reasons, "3제 조합 사유가 소실됐다"


def test_major3plus_and_elderly_organ_merges_reason():
    """major≥3 + 고령·장기위험 — DDI_MAJOR_3PLUS 와 SEV_ELDERLY_ORGAN 이 함께."""
    sub, reasons = _builder(
        ddi_major=3, age=80, drug_count=6, has_renal_risk_drug=True
    ).rule_floor([], _REF)
    assert sub == "Y_DDI_MAJOR"
    assert reasons == {"DDI_MAJOR_3PLUS", "SEV_ELDERLY_ORGAN"}


def test_major_alone_unchanged():
    """중증 트리거가 없으면 종전과 동일 — 사유가 늘지 않는다."""
    assert _builder(ddi_major=1).rule_floor([], _REF) == ("Y_DDI_MAJOR", {"DDI_MAJOR"})
    assert _builder(ddi_major=5).rule_floor([], _REF) == ("Y_DDI_MAJOR", {"DDI_MAJOR_3PLUS"})


def test_severe_alone_unchanged():
    """major 가 없으면 종전대로 Y_TRIPLE."""
    sub, reasons = _builder(triple_whammy=True).rule_floor([], _REF)
    assert sub == "Y_TRIPLE"
    assert reasons == {"SEV_TRIPLE_WHAMMY"}


def test_neither_unchanged():
    assert _builder().rule_floor([], _REF) == (None, set())


@pytest.mark.parametrize(
    "feats,expected_sub",
    [
        ({"ddi_major": 1, "triple_whammy": True}, "Y_DDI_MAJOR"),
        ({"ddi_major": 3, "triple_whammy": True}, "Y_DDI_MAJOR"),
        ({"ddi_major": 1, "drug_count": 12, "has_high_risk_drug": True}, "Y_DDI_MAJOR"),
        ({"triple_whammy": True}, "Y_TRIPLE"),
    ],
)
def test_subtype_invariant_across_combinations(feats, expected_sub):
    """어떤 조합에서도 subtype 위계는 종전과 같다 — action 도 subtype 에서만 나온다."""
    sub, _ = _builder(**feats).rule_floor([], _REF)
    assert sub == expected_sub
