"""M7 — PSI 산출 작업의 컬럼 선택과 일간 재산출.

두 가지를 고정한다.

**① 컬럼을 손으로 나열하지 않는다.** 종전 배치 DAG 는 `drug_count`·`ddi_count`·
`rule_triggered` 셋을 하드코딩했다. 그중 `rule_triggered` 는 예측 parquet 에
**쓰인 적이 없고**, `ddi_count` 는 기준 분포의 피처명이 아니다. 실제로 PSI 가
계산되던 열은 `drug_count` **하나뿐**이었다. 기준 분포와의 교집합에 맡긴다.

**② 기준 분포에 없는 열은 조용히 건너뛴다.** 예측 parquet 에 무엇이 더 실리든
깨지지 않아야 한다.
"""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from monitoring.drift_detector import DriftDetector
from monitoring.drift_job import run_drift_job, select_psi_columns


def _reference(tmp_path, cols):
    ref = pd.DataFrame({c: list(range(1, 51)) for c in cols})
    d = DriftDetector().fit(ref)
    p = tmp_path / "drift_reference.pkl"
    d.save(str(p))
    return p


def test_columns_come_from_the_reference_not_a_hardcoded_list(tmp_path):
    ref_p = _reference(tmp_path, ["drug_count", "ddi_major", "age"])
    df = pd.DataFrame({
        "drug_count": [1, 2], "ddi_major": [0, 1], "age": [70, 80],
        "patient_id": ["a", "b"], "intervention": ["x", "y"],   # 기준에 없는 열
    })

    cols = select_psi_columns(df, DriftDetector.load(str(ref_p)))

    assert set(cols) == {"drug_count", "ddi_major", "age"}


def test_extra_columns_do_not_break_it(tmp_path):
    ref_p = _reference(tmp_path, ["drug_count"])
    df = pd.DataFrame({"drug_count": [1, 2, 3], "무관한열": ["a", "b", "c"]})

    cols = select_psi_columns(df, DriftDetector.load(str(ref_p)))

    assert cols == ["drug_count"]


def test_no_overlap_returns_empty_not_a_crash(tmp_path):
    ref_p = _reference(tmp_path, ["drug_count"])
    df = pd.DataFrame({"intervention": ["x"]})

    assert select_psi_columns(df, DriftDetector.load(str(ref_p))) == []


def test_job_writes_a_report_and_returns_the_column_count(tmp_path):
    ref_p = _reference(tmp_path, ["drug_count", "ddi_major"])
    pred = tmp_path / "predictions_20260904.parquet"
    pd.DataFrame({"drug_count": [1, 2, 3, 4], "ddi_major": [0, 1, 1, 2],
                  "patient_id": list("abcd")}).to_parquet(pred, index=False)
    out = tmp_path / "monitoring"

    report = run_drift_job(pred, ref_p, out, partition="20260904")

    assert report is not None
    assert len(report.feature_results) == 2
    assert (out / "drift_20260904.json").exists()


def test_job_is_a_noop_when_the_reference_is_absent(tmp_path):
    pred = tmp_path / "predictions_20260904.parquet"
    pd.DataFrame({"drug_count": [1]}).to_parquet(pred, index=False)

    assert run_drift_job(pred, tmp_path / "없음.pkl", tmp_path, partition="20260904") is None


def test_job_is_a_noop_when_predictions_are_absent(tmp_path):
    ref_p = _reference(tmp_path, ["drug_count"])

    assert run_drift_job(tmp_path / "없음.parquet", ref_p, tmp_path, partition="x") is None


def _run(tmp_path, monkeypatch, cols=("drug_count", "ddi_major")):
    ref_p = _reference(tmp_path, list(cols))
    pred = tmp_path / "predictions_20260904.parquet"
    pd.DataFrame({c: [1, 2, 3, 4] for c in cols}).to_parquet(pred, index=False)
    return run_drift_job(pred, ref_p, tmp_path / "m", partition="20260904")


def test_psi_is_recorded_per_feature(tmp_path, monkeypatch):
    seen = []
    import monitoring.drift_job as dj
    monkeypatch.setattr(dj, "record_psi", lambda feature_name, psi_value: seen.append(feature_name))
    monkeypatch.delenv("DDI_PUSHGATEWAY_URL", raising=False)

    _run(tmp_path, monkeypatch)

    assert set(seen) == {"drug_count", "ddi_major"}


def test_psi_is_pushed_when_a_gateway_is_configured(tmp_path, monkeypatch):
    """이 코드는 DAG 워커에서 돈다. push 하지 않으면 서빙의 노출 경로에는
    영원히 나타나지 않는다 — 프로세스가 다르다."""
    pushed = []
    import monitoring.drift_job as dj
    monkeypatch.setattr(dj, "push_metrics", lambda url, job: pushed.append((url, job)))
    monkeypatch.setenv("DDI_PUSHGATEWAY_URL", "http://pushgw:9091")

    _run(tmp_path, monkeypatch)

    assert pushed == [("http://pushgw:9091", "ddi_drift")]


def test_without_a_gateway_it_warns_instead_of_silently_dropping(tmp_path, monkeypatch, caplog):
    """조용히 사라지면 운영자는 PSI 가 나가고 있다고 믿는다."""
    import logging

    import monitoring.drift_job as dj
    pushed = []
    monkeypatch.setattr(dj, "push_metrics", lambda url, job: pushed.append(url))
    monkeypatch.delenv("DDI_PUSHGATEWAY_URL", raising=False)

    with caplog.at_level(logging.WARNING, logger="monitoring.drift_job"):
        _run(tmp_path, monkeypatch)

    assert pushed == []
    assert any("DDI_PUSHGATEWAY_URL" in r.message for r in caplog.records)
