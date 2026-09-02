"""탐지 전용 모드 — 규칙층이 발화하되 개입 등급은 올리지 않는다.

EDI→약물명 해소를 켜면 Top-10 이 발화하지만 즉각 개입 대상이 함께 급증한다(실측 28배).
탐지는 지금 필요하고 개입 용량은 아직 결정되지 않았으므로, 둘을 한 플래그에 묶으면
인력이 부족하다는 이유로 탐지까지 못 켜게 된다.

이 모드는 **SafetyNet 등급의 승격만** 끊는다. 아래 셋은 끊지 않는다 — 끊으면 안전 후퇴다.
  · 결정적 Red 백스톱 (금기)
  · rule_floor subtype 하한
  · ML 등급
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_DRUGS = [("645600390", "Warfarin"), ("054801360", "Aspirin")]   # TOP01 교과서 사례


def _fresh(**env):
    """플래그는 임포트 시점이 아니라 호출 시점에 읽히지만, 모듈 상태를 확실히 분리한다."""
    for k, v in env.items():
        os.environ[k] = v
    for m in [k for k in list(sys.modules) if k.startswith(("serving", "rules"))]:
        del sys.modules[m]
    return importlib.import_module("serving.predictor")


def _predict(*, resolve=True, detect_only=False, ml_level=None,
             red_backstop=False, floor=None):
    mod = _fresh(**{
        "SERVING_ENABLE_EDI_NAME_RESOLUTION": "1" if resolve else "",
        mod_env(): "1" if detect_only else "",
    })
    from serving.schemas import DrugItem, PredictRequest
    p = mod.HybridPredictor.__new__(mod.HybridPredictor)
    p._start_time = 0.0
    p._ml = MagicMock(); p._ml.loaded = False
    p._ddi_matrix = None; p._cyp = None; p._std = None
    b = mod.RequestFeatureBuilder(ddi_matrix=None, code_standardizer=None)
    b.build = lambda req, **kw: (np.zeros(3), {})
    b.red_triggers = lambda d, r, a=None: ({"RED_CONTRAINDICATED"} if red_backstop else set())
    b.rule_floor = lambda d, r, a=None: ((floor, {"SEV_TRIPLE_WHAMMY"}) if floor else (None, set()))
    b.risk_flags = lambda d: (False, False)
    names = dict(_DRUGS)
    b.resolve_codes = lambda ds: [setattr(d, "drug_name", names.get(d.edi_code)) for d in ds]
    p._builder = b
    from rules.safety_net import SafetyNet
    p._safety_net = SafetyNet(); p._dup_detector = None
    p._ml_lock = threading.Lock(); p._hier_lock = threading.RLock()
    h = MagicMock()
    if ml_level:
        h.loaded = True; h.feature_cols = []; h.feature_semantics_version = "rulefeat.v1"
        h.predict_risk_single = lambda fv: {
            "risk_level": ml_level, "p_red": 0.1, "stage2_probs": None,
            "red_suspect": False, "action": "모니터링"}
    else:
        h.loaded = False
    p._hierarchical = h
    req = PredictRequest(patient_id="T", drugs=[
        DrugItem(edi_code=c, total_days=30, start_date=date(2024, 7, 1)) for c, _ in _DRUGS])
    return p.predict(req)


def mod_env():
    return "SERVING_RULE_DETECT_ONLY"


@pytest.fixture(autouse=True)
def _clean_env():
    keep = {k: os.environ.get(k) for k in
            ("SERVING_ENABLE_EDI_NAME_RESOLUTION", "SERVING_RULE_DETECT_ONLY")}
    yield
    for k, v in keep.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v


def test_flag_exposed_in_health_state():
    mod = _fresh(SERVING_RULE_DETECT_ONLY="")
    assert mod_env() in mod.serving_flag_state()


def test_off_keeps_current_escalation():
    """종전 동작 — 해소가 켜지면 등급이 올라간다."""
    r = _predict(resolve=True, detect_only=False)
    assert r.risk_level.value == "Red"
    assert r.rule_level.value == "Red"


def test_on_detects_without_escalating():
    """핵심 — 사유·알림은 남고 개입 등급만 종전대로."""
    r = _predict(resolve=True, detect_only=True)
    assert r.risk_level.value == "Normal", "개입 등급이 올라가면 용량 결정을 우회한 것"
    assert r.rule_level.value == "Red", "탐지 결과는 관측 가능해야 한다"
    assert any("TOP01" in x for x in r.risk_reasons)
    assert len(r.ddi_alerts) > 0


def test_on_does_not_suppress_red_backstop():
    """금기 백스톱은 끊지 않는다 — 끊으면 안전 후퇴다."""
    r = _predict(resolve=True, detect_only=True, red_backstop=True)
    assert r.risk_level.value == "Red"
    assert "RED_CONTRAINDICATED" in r.risk_reasons


def test_on_does_not_suppress_rule_floor():
    """rule_floor subtype 하한도 그대로 — 해소 플래그와 무관한 경로다."""
    r = _predict(resolve=True, detect_only=True, floor="Y_TRIPLE")
    assert r.risk_level.value == "Yellow"
    assert r.yellow_subtype == "Y_TRIPLE"


def test_on_does_not_suppress_ml():
    """ML 등급도 그대로 — max(rule, ml) 에서 rule 만 끊는다."""
    r = _predict(resolve=True, detect_only=True, ml_level="Red")
    assert r.risk_level.value == "Red"


def test_on_without_resolution_is_noop():
    """해소가 꺼져 있으면 애초에 발화가 없으므로 아무 차이가 없다."""
    a = _predict(resolve=False, detect_only=False)
    b = _predict(resolve=False, detect_only=True)
    assert a.risk_level == b.risk_level == "Normal" or a.risk_level.value == b.risk_level.value


# ─────────────────────────────────────────────────────────────────────────────
# 실 EDI 코호트 불변식 — 위 합성 케이스가 고정하는 원리를 실 청구로 확인한다.
#
# 실측(하루치 상위 300명, 2026-09-02):
#   OFF        최종 GREEN 205 / YELLOW 95 · rule_level 전량 GREEN · Top-10 탐지 0
#   해소만 ON  최종 GREEN 184 / YELLOW 48 / RED 68 · Top-10 탐지 42
#   탐지 전용  최종 NORMAL 205 / YELLOW 95 · rule_level GREEN 228/RED 68/YELLOW 4
#              · Top-10 탐지 42 (해소 ON 과 동일)
#
# 즉 탐지는 해소 ON 과 같고, 개입 산출물(action·yellow_subtype)은 OFF 와 같다.
# `risk_level` 라벨만 GREEN → NORMAL 로 바뀐다 — 개입 지시는 `action` 이 나르므로
# 약사 업무는 달라지지 않지만, 응답에 보이는 변화이므로 여기 명시해 둔다.
# ─────────────────────────────────────────────────────────────────────────────

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))


def _real_cohort_outcomes(monkeypatch, detect_only: bool, resolution: bool, n: int = 40):
    """실 EDI 요청 n 건의 (action, yellow_subtype, rule_level, 사유수) 목록."""
    import threading
    from datetime import date
    from unittest.mock import MagicMock

    import pandas as pd
    import pytest as _pytest

    from rules.safety_net import SafetyNet
    from scripts.etl.code_standardizer import CodeStandardizer
    from serving.predictor import HybridPredictor, RequestFeatureBuilder
    from serving.schemas import DrugItem, PredictRequest

    raw = _ROOT / "data" / "Raw" / "records_20240701.parquet"
    if not raw.exists() or not (_ROOT / "data/processed/edi_to_wk.parquet").exists():
        _pytest.skip("실 청구 데이터·참조DB 없음 — 생략")

    for env in ("SERVING_ENABLE_EDI_NAME_RESOLUTION", "SERVING_RISK_FLAG_ATC_CANDIDATES",
                mod_env()):
        monkeypatch.delenv(env, raising=False)
    if resolution:
        monkeypatch.setenv("SERVING_ENABLE_EDI_NAME_RESOLUTION", "1")
    if detect_only:
        monkeypatch.setenv(mod_env(), "1")

    df = pd.read_parquet(raw, columns=["patient_id", "edi_code", "total_days"])
    for c in ("patient_id", "edi_code"):
        df[c] = df[c].astype(str).str.strip()
    df = df[df["edi_code"].ne("") & df["edi_code"].ne("nan")]
    top = df.groupby("patient_id").size().sort_values(ascending=False).index[:n]

    std = CodeStandardizer()
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml = MagicMock()
    pred._ml.loaded = False
    pred._ml_lock = threading.Lock()
    pred._hier_lock = threading.RLock()
    pred._hierarchical = None
    pred._ddi_matrix = None
    pred._cyp = None
    pred._std = std
    pred._builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=std)
    pred._safety_net = SafetyNet(
        ddi_matrix_path=_ROOT / "data" / "_absent_ddi.parquet",
        drug_index_path=_ROOT / "data" / "_absent_index.parquet",
    )
    pred._dup_detector = None

    out = []
    for pid in top:
        sub = df[df["patient_id"] == pid].head(40)
        drugs = [
            DrugItem(edi_code=r.edi_code,
                     total_days=max(1, min(365, int(r.total_days or 1))),
                     start_date=date(2024, 7, 1))
            for r in sub.itertuples()
        ]
        res = pred.predict(PredictRequest(patient_id=pid, patient_age=72, drugs=drugs))
        out.append({
            "action": str(getattr(res, "action", None)),
            "subtype": str(getattr(res, "yellow_subtype", None)),
            "rule_level": str(getattr(res, "rule_level", None)),
            "reasons": sorted(res.risk_reasons or []),
        })
    return out


def test_real_cohort_intervention_output_matches_flag_off(monkeypatch):
    """탐지 전용에서 **개입 산출물이 플래그 OFF 와 같아야 한다.**

    `action` 이 약사에게 가는 지시를 나르고 `yellow_subtype` 이 그 종류를 정한다.
    탐지를 켜면서 이 둘이 달라지면 "개입은 종전대로" 라는 주장이 깨진다.
    """
    off = _real_cohort_outcomes(monkeypatch, detect_only=False, resolution=False)
    detect = _real_cohort_outcomes(monkeypatch, detect_only=True, resolution=True)

    off_pairs = [(r["action"], r["subtype"]) for r in off]
    det_pairs = [(r["action"], r["subtype"]) for r in detect]

    assert det_pairs == off_pairs, (
        "탐지 전용이 개입 산출물을 바꿨다 — 탐지만 켜는 것이 아니게 된다. "
        f"불일치 {sum(1 for a, b in zip(off_pairs, det_pairs) if a != b)}건"
    )


def test_real_cohort_detection_is_not_reduced(monkeypatch):
    """탐지 전용에서 **탐지가 줄지 않아야 한다.**

    개입을 끊으면서 탐지까지 줄어들면 플래그를 켤 이유가 없다.

    등호가 아니라 포함 관계로 본다. 실측에서 탐지 전용 쪽 사유가 오히려 늘어나는
    환자가 있었는데(40명 중 4명), 늘어난 것은 전부 `rule_floor` 의 `SEV_*` 였다.
    해소 ON 에서는 등급이 이미 Red 라 하한이 적용될 자리가 없고, 탐지 전용에서
    등급이 내려가며 하한이 작동해 사유가 붙는다. `rule_floor` 를 억제하지 않는
    것이 이 플래그의 조건이므로 이는 의도된 동작이다 — 줄어드는 것만 금지한다.
    """
    resolve = _real_cohort_outcomes(monkeypatch, detect_only=False, resolution=True)
    detect = _real_cohort_outcomes(monkeypatch, detect_only=True, resolution=True)

    assert [r["rule_level"] for r in detect] == [r["rule_level"] for r in resolve], (
        "탐지 전용이 rule_level 을 바꿨다 — 무엇이 탐지됐는지 관측할 수 없게 된다"
    )

    lost = [
        (i, sorted(set(a["reasons"]) - set(b["reasons"])))
        for i, (a, b) in enumerate(zip(resolve, detect))
        if set(a["reasons"]) - set(b["reasons"])
    ]
    assert not lost, f"탐지 전용에서 사유가 사라졌다 — 탐지가 줄었다: {lost[:3]}"
