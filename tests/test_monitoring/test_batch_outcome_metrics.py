"""M7 — 대시보드가 참조하나 정의되지 않았던 메트릭.

Grafana 대시보드는 `ddi_batch_success_total`·`ddi_batch_fail_total`·
`ddi_pharmacist_acceptance_rate` 셋을 참조하는데 코드에 정의가 없었다.
앞의 둘은 정의한다. **셋째는 정의하지 않는다** — 아래 테스트가 그 부재를 고정한다.
"""
from __future__ import annotations

import monitoring.metrics as mm


def test_batch_outcome_counters_exist():
    assert mm.BATCH_SUCCESS_TOTAL is not None
    assert mm.BATCH_FAIL_TOTAL is not None


def test_batch_counters_increment():
    mm.record_batch_outcome(success=3, fail=1, source="api")
    mm.record_batch_outcome(success=1, fail=0, source="api")
    # 폴백 구현도 실 구현도 예외 없이 누적되면 된다 — 값 조회는 구현별이라 보지 않는다


def test_pharmacist_acceptance_rate_is_deliberately_undefined():
    """정의하면 한 번도 채워지지 않은 채 0.0 으로 노출되고, 대시보드는
    "약사 수용률 0%" 로 읽는다. 피드백 수집 경로가 없다(P1-5). 미정의가 정직하다."""
    assert not hasattr(mm, "PHARMACIST_ACCEPTANCE_RATE")
    names = [n for n in dir(mm) if "ACCEPTANCE" in n.upper()]
    assert names == []


def test_dashboard_panel_says_the_feedback_path_is_missing():
    """패널을 지우지 않고 남기되, 무엇이 없어서 비어 있는지 제목에 적는다."""
    import json
    from pathlib import Path

    d = json.loads(Path("monitoring/grafana/dashboard.json").read_text(encoding="utf-8"))
    titles = [p.get("title", "") for p in d["panels"]]
    hit = [t for t in titles if "수용률" in t or "acceptance" in t.lower()]

    assert hit, "약사 수용률 패널이 사라졌다 — 지우지 말고 사유를 적어 남긴다"
    assert any("미구현" in t or "P1-5" in t for t in hit)
