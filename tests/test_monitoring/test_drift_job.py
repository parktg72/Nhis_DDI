"""M7 — PSI 산출·노출 작업 (PR #22 리뷰 반영).

고정하는 것 넷.

**① 컬럼을 손으로 나열하지 않는다.** 종전 배치 DAG 는 세 열을 하드코딩했다.
그중 하나는 예측 parquet 에 **쓰인 적이 없고** 하나는 기준 분포의 피처명이
아니어서, 실제로 계산되던 열은 `drug_count` 하나뿐이었다.

**② push 에는 보낼 것만 담는다.** 전역 레지스트리를 밀면 이 프로세스에 정의만
되고 값이 없는 서빙 메트릭과 기본 수집기가 0 인 채로 함께 게시된다.

**③ 일간 작업은 다시 채점하지 않는다.** 같은 데이터를 다시 재면 새 정보 없이
리포트를 덮어쓰고 배치 DAG 와 파일 경합을 만든다. 대신 데이터가 며칠 된
것인지를 숫자로 낸다.

**④ push 실패를 삼키지 않는다.** 관측하라고 구성해 놓고 나가지 않았는데
태스크가 초록불이면 운영자는 관측되고 있다고 믿는다.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

pd = pytest.importorskip("pandas")

import monitoring.drift_job as dj
from monitoring.drift_detector import DriftDetector
from monitoring.drift_job import (
    PushFailed,
    push_latest_psi,
    run_drift_job,
    select_psi_columns,
)


@pytest.fixture(autouse=True)
def _no_gateway(monkeypatch):
    monkeypatch.delenv("DDI_PUSHGATEWAY_URL", raising=False)


def _reference(tmp_path, cols):
    ref = pd.DataFrame({c: list(range(1, 51)) for c in cols})
    p = tmp_path / "drift_reference.pkl"
    DriftDetector().fit(ref).save(str(p))
    return p


def _predictions(tmp_path, cols, name="predictions_20260904.parquet"):
    p = tmp_path / name
    pd.DataFrame({c: [1, 2, 3, 4] for c in cols}).to_parquet(p, index=False)
    return p


# ── ① 컬럼 선택 ──────────────────────────────────────────────────────────
def test_columns_come_from_the_reference_not_a_hardcoded_list(tmp_path):
    ref_p = _reference(tmp_path, ["drug_count", "ddi_major", "age"])
    df = pd.DataFrame({
        "drug_count": [1, 2], "ddi_major": [0, 1], "age": [70, 80],
        "patient_id": ["a", "b"], "intervention": ["x", "y"],
    })

    assert set(select_psi_columns(df, DriftDetector.load(str(ref_p)))) == {
        "drug_count", "ddi_major", "age"}


def test_extra_columns_do_not_break_it(tmp_path):
    ref_p = _reference(tmp_path, ["drug_count"])
    df = pd.DataFrame({"drug_count": [1, 2, 3], "무관한열": ["a", "b", "c"]})

    assert select_psi_columns(df, DriftDetector.load(str(ref_p))) == ["drug_count"]


def test_no_overlap_returns_empty_not_a_crash(tmp_path):
    ref_p = _reference(tmp_path, ["drug_count"])

    assert select_psi_columns(pd.DataFrame({"intervention": ["x"]}),
                              DriftDetector.load(str(ref_p))) == []


# ── 채점 ─────────────────────────────────────────────────────────────────
def test_job_writes_a_report(tmp_path):
    ref_p = _reference(tmp_path, ["drug_count", "ddi_major"])
    pred = _predictions(tmp_path, ["drug_count", "ddi_major"])
    out = tmp_path / "monitoring"

    report = run_drift_job(pred, ref_p, out, partition="20260904")

    assert len(report.feature_results) == 2
    assert (out / "drift_20260904.json").exists()


def test_job_is_a_noop_when_the_reference_is_absent(tmp_path):
    pred = _predictions(tmp_path, ["drug_count"])
    assert run_drift_job(pred, tmp_path / "없음.pkl", tmp_path, partition="x") is None


def test_job_is_a_noop_when_predictions_are_absent(tmp_path):
    ref_p = _reference(tmp_path, ["drug_count"])
    assert run_drift_job(tmp_path / "없음.parquet", ref_p, tmp_path, partition="x") is None


# ── ② push 페이로드 ──────────────────────────────────────────────────────
def test_push_carries_only_psi_not_the_global_registry(tmp_path, monkeypatch):
    """DAG 워커의 전역 레지스트리에는 값 없는 서빙 메트릭과 기본 수집기가 있다."""
    sent = {}
    monkeypatch.setattr(dj, "push_metrics",
                        lambda url, job, registry: sent.update(url=url, job=job, reg=registry) or True)
    monkeypatch.setenv("DDI_PUSHGATEWAY_URL", "http://pushgw:9091")

    ref_p = _reference(tmp_path, ["drug_count", "ddi_major"])
    run_drift_job(_predictions(tmp_path, ["drug_count", "ddi_major"]),
                  ref_p, tmp_path / "m", partition="20260904")

    assert sent["job"] == "ddi_drift"
    names = {s.name for m in sent["reg"].collect() for s in m.samples}
    assert names == {"ddi_psi_score"}
    assert not any(n.startswith(("process_", "python_", "ddi_prediction", "ddi_batch"))
                   for n in names)


def test_without_a_gateway_it_warns_instead_of_silently_dropping(tmp_path, monkeypatch, caplog):
    import logging
    pushed = []
    monkeypatch.setattr(dj, "push_metrics", lambda *a, **k: pushed.append(a) or True)

    ref_p = _reference(tmp_path, ["drug_count"])
    with caplog.at_level(logging.WARNING, logger="monitoring.drift_job"):
        run_drift_job(_predictions(tmp_path, ["drug_count"]), ref_p, tmp_path / "m",
                      partition="20260904")

    assert pushed == []
    assert any("DDI_PUSHGATEWAY_URL" in r.message for r in caplog.records)


# ── ④ 실패 표면화 ────────────────────────────────────────────────────────
def test_a_configured_push_that_fails_raises(tmp_path, monkeypatch):
    """관측하라고 구성해 놓고 나가지 않았는데 태스크가 초록불이면 안 된다."""
    monkeypatch.setattr(dj, "push_metrics", lambda url, job, registry: False)
    monkeypatch.setenv("DDI_PUSHGATEWAY_URL", "http://pushgw:9091")

    ref_p = _reference(tmp_path, ["drug_count"])
    with pytest.raises(PushFailed):
        run_drift_job(_predictions(tmp_path, ["drug_count"]), ref_p, tmp_path / "m",
                      partition="20260904")


# ── ③ 일간 노출 — 재채점하지 않는다 ──────────────────────────────────────
def _report(dirpath, partition, generated_at, psi=0.3):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / f"drift_{partition}.json").write_text(json.dumps({
        "partition": partition, "generated_at": generated_at,
        "n_drifted": 1, "trigger_retrain": False, "summary": {},
        "features": [{"feature": "drug_count", "psi": psi, "status": "드리프트"}],
    }, ensure_ascii=False), encoding="utf-8")


def test_daily_push_does_not_rewrite_the_report(tmp_path, monkeypatch):
    """배치가 쓴 리포트는 불변 기록이다 — 최초 generated_at 이 살아 있어야 한다."""
    m = tmp_path / "monitoring"
    _report(m, "20260905", "2026-09-05T05:10:00")
    before = (m / "drift_20260905.json").read_text(encoding="utf-8")

    push_latest_psi(m, today=date(2026, 9, 7))

    assert (m / "drift_20260905.json").read_text(encoding="utf-8") == before
    assert list(m.glob("drift_*.json")) == [m / "drift_20260905.json"]


def test_daily_push_picks_the_newest_partition(tmp_path, monkeypatch):
    m = tmp_path / "monitoring"
    _report(m, "20260901", "2026-09-01T05:10:00")
    _report(m, "20260905", "2026-09-05T05:10:00")

    latest = push_latest_psi(m, today=date(2026, 9, 7))

    assert latest.name == "drift_20260905.json"


def test_daily_push_reports_how_old_the_data_is(tmp_path, monkeypatch):
    """오래된 parquet 을 heartbeat 로 쓰는 것이 위험하다 — 그 나이를 숫자로 낸다."""
    sent = {}
    monkeypatch.setattr(dj, "push_metrics",
                        lambda url, job, registry: sent.update(reg=registry) or True)
    monkeypatch.setenv("DDI_PUSHGATEWAY_URL", "http://pushgw:9091")
    m = tmp_path / "monitoring"
    _report(m, "20260905", "2026-09-05T05:10:00")

    push_latest_psi(m, today=date(2026, 9, 7))

    vals = {s.name: s.value for mf in sent["reg"].collect() for s in mf.samples}
    assert vals["ddi_psi_source_partition_age_days"] == 2.0
    assert vals["ddi_psi_score"] == 0.3


def test_daily_push_is_a_noop_when_there_is_no_report(tmp_path):
    assert push_latest_psi(tmp_path / "없음", today=date(2026, 9, 7)) is None
