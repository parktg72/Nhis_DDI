"""M-3 — 기준서 Red 7조건의 측정용 평가기.

기준서(CLINICAL_STANDARDS_v1.0 §4.1)의 Red 조건과 코드가 1:1 이 아니다
(개선계획 차단 요인 B-2). 이 모듈은 **기준서 문언 그대로** 를 평가해
"되돌리면 얼마가 되는가" 를 잴 수 있게 한다. 서빙·라벨 경로는 건드리지 않는다.

여기서 고정하는 것 중 가장 중요한 것은 **미측정을 미해당으로 읽지 않는 것**이다.
"""
from __future__ import annotations

import pytest

from scripts.ops.standards_red_conditions import (
    CONDITION_IDS,
    DEMOTED_CONDITION_IDS,
    FIRED,
    NOT_FIRED,
    UNMEASURABLE,
    evaluate,
    summarize,
)


class Row:
    """FEATURE_COLS 속성을 갖는 최소 행 — PatientFeatures·itertuples 와 같은 모양."""

    _DEFAULTS = dict(
        drug_count=0, ddi_contraindicated=0, ddi_major=0, ddi_moderate=0,
        triple_whammy=0, qt_risk_count=0, has_high_risk_drug=0,
        has_renal_risk_drug=0, has_hepatic_risk_drug=0, age=60,
    )

    def __init__(self, **kw):
        bad = set(kw) - set(self._DEFAULTS)
        assert not bad, f"알 수 없는 피처: {bad}"
        for k, v in {**self._DEFAULTS, **kw}.items():
            setattr(self, k, v)


def _status(row, cid):
    return evaluate(row)[cid].status


# ── 조건 1 금기 ───────────────────────────────────────────────────────────
def test_contraindicated_fires_at_one():
    assert _status(Row(ddi_contraindicated=1), "STD_CONTRAINDICATED") == FIRED
    assert _status(Row(ddi_contraindicated=0), "STD_CONTRAINDICATED") == NOT_FIRED


# ── 조건 2 Major 3건 이상 ─────────────────────────────────────────────────
def test_major_3plus_boundary():
    assert _status(Row(ddi_major=3), "STD_MAJOR_3PLUS") == FIRED
    assert _status(Row(ddi_major=2), "STD_MAJOR_3PLUS") == NOT_FIRED


# ── 조건 3 Triple Whammy ──────────────────────────────────────────────────
def test_triple_whammy():
    assert _status(Row(triple_whammy=1), "STD_TRIPLE_WHAMMY") == FIRED
    assert _status(Row(triple_whammy=0), "STD_TRIPLE_WHAMMY") == NOT_FIRED


# ── 조건 4 10종 + Major ───────────────────────────────────────────────────
def test_ten_drugs_plus_major_needs_both():
    assert _status(Row(drug_count=10, ddi_major=1), "STD_10DRUG_MAJOR") == FIRED
    assert _status(Row(drug_count=9, ddi_major=1), "STD_10DRUG_MAJOR") == NOT_FIRED
    assert _status(Row(drug_count=10, ddi_major=0), "STD_10DRUG_MAJOR") == NOT_FIRED


# ── 조건 5 고위험약 + Major ───────────────────────────────────────────────
def test_high_risk_plus_major_needs_both():
    assert _status(Row(has_high_risk_drug=1, ddi_major=1), "STD_HIGHRISK_MAJOR") == FIRED
    assert _status(Row(has_high_risk_drug=0, ddi_major=1), "STD_HIGHRISK_MAJOR") == NOT_FIRED
    assert _status(Row(has_high_risk_drug=1, ddi_major=0), "STD_HIGHRISK_MAJOR") == NOT_FIRED


# ── 조건 6 고령 + 다제 + 장기위험 ─────────────────────────────────────────
def test_elderly_organ_boundaries():
    base = dict(drug_count=5, has_renal_risk_drug=1)
    assert _status(Row(age=75, **base), "STD_ELDERLY_ORGAN") == FIRED
    assert _status(Row(age=74, **base), "STD_ELDERLY_ORGAN") == NOT_FIRED
    assert _status(Row(age=75, drug_count=4, has_renal_risk_drug=1),
                   "STD_ELDERLY_ORGAN") == NOT_FIRED
    assert _status(Row(age=75, drug_count=5), "STD_ELDERLY_ORGAN") == NOT_FIRED


def test_hepatic_risk_alone_also_satisfies_the_organ_arm():
    assert _status(Row(age=80, drug_count=5, has_hepatic_risk_drug=1),
                   "STD_ELDERLY_ORGAN") == FIRED


@pytest.mark.parametrize("missing_age", [None, float("nan")])
def test_missing_age_is_unmeasurable_not_a_quiet_no(missing_age):
    """pandas 는 결측을 NaN 으로 준다. `NaN >= 75` 는 조용히 False 다."""
    r = evaluate(Row(age=missing_age, drug_count=9, has_renal_risk_drug=1))["STD_ELDERLY_ORGAN"]

    assert r.status == UNMEASURABLE
    assert r.reason == "age_missing"


# ── 조건 7 QT ─────────────────────────────────────────────────────────────
def test_qt_is_always_unmeasurable_from_cohort_features():
    """`qt_risk_count` 는 ETL 에서 대입되지 않는다 (B-3). 0 은 '없음'이 아니다."""
    for row in (Row(qt_risk_count=0), Row(qt_risk_count=5)):
        r = evaluate(row)["STD_QT_3PLUS"]
        assert r.status == UNMEASURABLE
        assert r.reason == "qt_risk_count_never_assigned"


# ── 피처 부재 ─────────────────────────────────────────────────────────────
def test_absent_feature_is_unmeasurable_not_zero():
    class Partial:
        ddi_contraindicated = 0
        # ddi_major 없음

    r = evaluate(Partial())["STD_MAJOR_3PLUS"]
    assert r.status == UNMEASURABLE
    assert r.reason == "feature_absent:ddi_major"


# ── 집계 ──────────────────────────────────────────────────────────────────
def test_summary_reports_a_denominator_on_every_condition():
    rows = [Row(ddi_major=3), Row(ddi_major=1), Row()]

    s = summarize(rows)

    assert s["denominator"] == 3
    assert s["conditions"]["STD_MAJOR_3PLUS"] == {
        "fired": 1, "not_fired": 2, "unmeasurable": 0,
    }
    assert s["conditions"]["STD_QT_3PLUS"]["unmeasurable"] == 3


def test_any_demoted_prefers_unmeasurable_over_not_fired():
    """하나라도 발화하면 발화. 아니면 미측정이 있으면 미측정. 둘 다 아니면 미해당."""
    fired = Row(ddi_major=3)
    unknown = Row(age=None)
    plain = Row()

    s = summarize([fired, unknown, plain])

    assert s["any_demoted"] == {"fired": 1, "not_fired": 0, "unmeasurable": 2}


def test_contraindicated_is_not_counted_as_demoted():
    """조건 1 은 강등되지 않았다. 되돌릴 규모에 섞으면 숫자가 부풀려진다."""
    assert "STD_CONTRAINDICATED" in CONDITION_IDS
    assert "STD_CONTRAINDICATED" not in DEMOTED_CONDITION_IDS
    assert len(CONDITION_IDS) == 7
    assert len(DEMOTED_CONDITION_IDS) == 6


def test_summary_of_nothing_has_a_zero_denominator_not_a_crash():
    s = summarize([])
    assert s["denominator"] == 0
    assert s["conditions"]["STD_MAJOR_3PLUS"]["fired"] == 0


# ── 현행 라벨 경로와의 대조 ───────────────────────────────────────────────
def test_agrees_with_the_label_path_where_both_implement_the_condition():
    from scripts.etl.clinical_rules import (
        collect_red_triggers,
        collect_severe_immediate_triggers,
    )

    rows = [
        Row(ddi_contraindicated=1),
        Row(triple_whammy=1),
        Row(age=80, drug_count=6, has_renal_risk_drug=1),
        Row(ddi_major=2, drug_count=3),
    ]
    for row in rows:
        ev = evaluate(row)
        assert (ev["STD_CONTRAINDICATED"].status == FIRED) is bool(collect_red_triggers(row))
        sev = collect_severe_immediate_triggers(row)
        assert (ev["STD_TRIPLE_WHAMMY"].status == FIRED) is ("SEV_TRIPLE_WHAMMY" in sev)
        assert (ev["STD_ELDERLY_ORGAN"].status == FIRED) is ("SEV_ELDERLY_ORGAN" in sev)


def test_condition_four_is_absent_from_the_label_path():
    """기준서는 `10종 + Major`, 코드는 `10종 + 고위험약`. 다른 조건이다."""
    from scripts.etl.clinical_rules import collect_severe_immediate_triggers

    row = Row(drug_count=10, ddi_major=1, has_high_risk_drug=0)

    assert evaluate(row)["STD_10DRUG_MAJOR"].status == FIRED
    assert collect_severe_immediate_triggers(row) == set()


def test_condition_five_is_absent_from_the_label_path():
    from scripts.etl.clinical_rules import collect_severe_immediate_triggers

    row = Row(has_high_risk_drug=1, ddi_major=1, drug_count=3)

    assert evaluate(row)["STD_HIGHRISK_MAJOR"].status == FIRED
    assert collect_severe_immediate_triggers(row) == set()


def test_a_value_that_refuses_comparison_is_unmeasurable():
    """pandas NA 처럼 `!=` 가 불리언을 안 주는 값도 결측으로 센다."""

    class NAish:
        def __ne__(self, other):
            raise TypeError("ambiguous")

    r = evaluate(Row(age=NAish(), drug_count=9, has_renal_risk_drug=1))["STD_ELDERLY_ORGAN"]
    assert r.status == UNMEASURABLE


def test_runs_over_a_dataframe_the_way_m1_will_use_it():
    """M1 은 코호트 DataFrame 을 넘긴다. itertuples 행에서 그대로 돌아야 한다."""
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame([
        {**Row._DEFAULTS, "ddi_major": 3},
        {**Row._DEFAULTS, "drug_count": 10, "ddi_major": 1},
        {**Row._DEFAULTS, "age": None},
    ])

    s = summarize(df.itertuples(index=False))

    assert s["denominator"] == 3
    assert s["conditions"]["STD_MAJOR_3PLUS"]["fired"] == 1
    assert s["conditions"]["STD_10DRUG_MAJOR"]["fired"] == 1
    assert s["conditions"]["STD_ELDERLY_ORGAN"]["unmeasurable"] == 1
    assert s["unmeasurable_reasons"]["STD_ELDERLY_ORGAN"]["age_missing"] == 1


def test_one_unreadable_organ_arm_blocks_the_whole_or():
    """읽은 쪽이 False 라고 해서 위험약이 없는 것이 아니다 — 못 읽은 쪽에 있었을 수 있다."""

    class NoRenalColumn:
        age = 80
        drug_count = 6
        ddi_contraindicated = 0
        ddi_major = 0
        triple_whammy = 0
        has_high_risk_drug = 0
        has_hepatic_risk_drug = 0
        # has_renal_risk_drug 없음

    r = evaluate(NoRenalColumn())["STD_ELDERLY_ORGAN"]
    assert r.status == UNMEASURABLE
    assert r.reason == "feature_absent:has_renal_risk_drug"
