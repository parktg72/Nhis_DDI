# 모니터링 대시보드 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FastAPI 예측 라우터에 실시간 메트릭 기록을 연결하고, Airflow DAG에 드리프트 감지·알림 태스크를 추가하고, Streamlit 앱에 4-tab 모니터링 페이지를 신규 추가한다.

**Architecture:** `MetricsWriter`(filelock 기반 jsonl)가 API 예측마다 레코드를 append한다. 배치 DAG 완료 후 `DriftDetector`가 PSI를 계산하고, `AlertManager`가 PSI·불일치율 알림을 JSON으로 저장한다. Streamlit 6번 페이지가 두 데이터 소스를 읽어 4-tab 대시보드를 제공한다.

**Tech Stack:** FastAPI, Streamlit, filelock, pandas, Airflow PythonOperator, 기존 monitoring/ 모듈 (DriftDetector, AlertManager)

**Spec:** `docs/superpowers/specs/2026-04-06-monitoring-dashboard-design.md`

---

## 파일 맵

| 태스크 | 파일 | 동작 |
|--------|------|------|
| 1 | `config/settings.py` | 모니터링 경로 4개 상수 추가 |
| 2 | `monitoring/metrics_writer.py` (신규) | MetricsWriter + 싱글턴 |
| 2 | `tests/test_monitoring/test_metrics_writer.py` (신규) | MetricsWriter 유닛 테스트 |
| 3 | `monitoring/alert_rules.py` | AlertType.RULE_ML_DISAGREE + evaluate_rule_ml_disagree() |
| 3 | `tests/test_monitoring/test_monitoring.py` | TestAlertRulesDisagree 클래스 추가 |
| 4 | `serving/routers/metrics.py` (신규) | GET /metrics 엔드포인트 |
| 4 | `serving/main.py` | metrics 라우터 등록 + lifespan에 MetricsWriter 초기화 |
| 5 | `serving/routers/predict.py` | MetricsWriter.append() 연결 |
| 6 | `scripts/train/pipeline.py` | _save_drift_reference() + run()에서 호출 |
| 7 | `dags/ddi_batch_predict_dag.py` | _detect_drift, _generate_alerts, 태스크 체인 |
| 8 | `hana_app/pages/6_📊_모니터링.py` (신규) | Streamlit 4-tab 모니터링 페이지 |
| 9 | `tests/test_integration/test_monitoring_pipeline.py` (신규) | 통합 테스트 |

---

## Task 1: settings.py — 모니터링 경로 상수

**Files:**
- Modify: `config/settings.py:56-69` (기존 데이터 파생 경로 블록 끝에 추가)

- [ ] **Step 1: settings.py 끝에 모니터링 블록 추가**

`config/settings.py` 파일 끝에 다음을 추가한다:

```python
# ── 모니터링 ────────────────────────────────────────────────────────────────────
MONITORING_DIR             = Path(os.environ.get("DDI_MONITORING_DIR",        "/app/data/monitoring"))
METRICS_JSONL_PATH         = Path(os.environ.get("DDI_METRICS_JSONL_PATH",    "/app/data/monitoring/metrics_live.jsonl"))
DRIFT_REFERENCE_PATH       = Path(os.environ.get("DDI_DRIFT_REFERENCE_PATH",  "/app/models/current/drift_reference.pkl"))
METRICS_JSONL_LOCK_TIMEOUT = float(os.environ.get("DDI_METRICS_JSONL_LOCK_TIMEOUT", "5.0"))
```

- [ ] **Step 2: import 테스트 — settings에서 상수 가져오기**

```bash
python -c "from config.settings import MONITORING_DIR, METRICS_JSONL_PATH, DRIFT_REFERENCE_PATH, METRICS_JSONL_LOCK_TIMEOUT; print('OK', MONITORING_DIR)"
```

Expected: `OK /app/data/monitoring`

- [ ] **Step 3: commit**

```bash
git add config/settings.py
git commit -m "feat: settings.py 모니터링 경로 상수 4개 추가"
```

---

## Task 2: MetricsWriter — filelock 기반 jsonl 기록기

**Files:**
- Create: `monitoring/metrics_writer.py`
- Create: `tests/test_monitoring/test_metrics_writer.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_monitoring/test_metrics_writer.py` 신규 생성:

```python
"""MetricsWriter 유닛 테스트."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsWriterAppend:
    def test_append_writes_json_line(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path)
        record = {
            "timestamp": "2026-04-06T10:00:00.000000",
            "partition": "2026-04-06",
            "patient_id": "P001",
            "risk_level": "RED",
            "rule_level": "RED",
            "ml_level": "YELLOW",
            "disagree": True,
            "latency_ms": 12.3,
            "source": "api",
        }
        w.append(record)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert loaded["patient_id"] == "P001"
        assert loaded["disagree"] is True

    def test_append_multiple_lines(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path)
        for i in range(5):
            w.append({"patient_id": f"P{i:03d}", "ts": i})
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 5

    def test_append_creates_parent_dir(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "nested" / "dir" / "metrics.jsonl"
        w = MetricsWriter(path=path)
        w.append({"x": 1})
        assert path.exists()

    def test_concurrent_append_no_data_loss(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path)
        n_threads = 10
        n_records = 20

        def write_records(thread_id):
            for i in range(n_records):
                w.append({"thread": thread_id, "seq": i})

        threads = [threading.Thread(target=write_records, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = path.read_text().strip().splitlines()
        assert len(lines) == n_threads * n_records


class TestMetricsWriterReadRecent:
    def _make_record(self, hours_ago: float, **kwargs) -> dict:
        ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
        return {"timestamp": ts, "patient_id": "P001", **kwargs}

    def test_read_recent_returns_recent_records(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path)
        w.append(self._make_record(0.5, partition="2026-04-06"))   # 30min ago — recent
        w.append(self._make_record(25.0, partition="2026-04-05"))  # 25h ago — old
        records = w.read_recent(hours=24)
        assert len(records) == 1
        assert records[0]["partition"] == "2026-04-06"

    def test_read_recent_file_not_found_returns_empty(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "nonexistent.jsonl"
        w = MetricsWriter(path=path)
        records = w.read_recent(hours=24)
        assert records == []

    def test_read_recent_skips_corrupt_lines(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "metrics.jsonl"
        ts = datetime.now(timezone.utc).isoformat()
        path.write_text(
            '{"timestamp": "' + ts + '", "patient_id": "P001"}\n'
            "NOT_VALID_JSON\n"
            '{"timestamp": "' + ts + '", "patient_id": "P002"}\n'
        )
        w = MetricsWriter(path=path)
        records = w.read_recent(hours=1)
        assert len(records) == 2
        assert {r["patient_id"] for r in records} == {"P001", "P002"}

    def test_read_recent_lock_timeout_on_read_returns_empty(self, tmp_path):
        """읽기 중 락 타임아웃 → [] 반환 (예외 전파 없음)."""
        from monitoring.metrics_writer import MetricsWriter
        from filelock import FileLock
        path = tmp_path / "metrics.jsonl"
        ts = datetime.now(timezone.utc).isoformat()
        path.write_text('{"timestamp": "' + ts + '", "x": 1}\n')
        w = MetricsWriter(path=path, lock_timeout=0.001)
        # 락을 선점해서 타임아웃 유도
        lock = FileLock(str(path) + ".lock")
        with lock:
            records = w.read_recent(hours=1)
        assert records == []


class TestMetricsWriterSingleton:
    def test_init_and_get(self, tmp_path):
        from monitoring.metrics_writer import init_metrics_writer, get_metrics_writer
        path = tmp_path / "metrics.jsonl"
        init_metrics_writer(path=path)
        w = get_metrics_writer()
        assert w is not None

    def test_get_before_init_raises(self, monkeypatch):
        import monitoring.metrics_writer as mw
        monkeypatch.setattr(mw, "_writer", None)
        with pytest.raises(RuntimeError, match="초기화"):
            mw.get_metrics_writer()
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_monitoring/test_metrics_writer.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'monitoring.metrics_writer'`

- [ ] **Step 3: MetricsWriter 구현**

`monitoring/metrics_writer.py` 신규 생성:

```python
"""
filelock 기반 JSON Lines 메트릭 기록기.

멀티 워커(uvicorn --workers N) 환경에서 안전하게 metrics_live.jsonl에 기록한다.
fcntl 미사용 — filelock 라이브러리로 크로스 플랫폼 지원.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from filelock import FileLock, Timeout

from config import settings

logger = logging.getLogger(__name__)


class MetricsWriter:
    """JSON Lines 파일에 메트릭 레코드를 append하는 기록기.

    Args:
        path: jsonl 파일 경로 (기본: settings.METRICS_JSONL_PATH)
        lock_timeout: 락 획득 최대 대기 시간(초) (기본: settings.METRICS_JSONL_LOCK_TIMEOUT)
    """

    def __init__(
        self,
        path: Path = None,
        lock_timeout: float = None,
    ) -> None:
        self._path = Path(path) if path is not None else settings.METRICS_JSONL_PATH
        self._lock_timeout = lock_timeout if lock_timeout is not None else settings.METRICS_JSONL_LOCK_TIMEOUT
        self._lock = FileLock(str(self._path) + ".lock")

    def append(self, record: dict) -> None:
        """레코드 1건을 jsonl 파일 끝에 추가한다.

        filelock 획득 → 파일 끝에 JSON 줄 추가 → flush → lock 해제.
        lock_timeout 초과 시 filelock.Timeout 예외를 그대로 전파한다.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock.acquire(timeout=self._lock_timeout):
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    def read_recent(self, hours: int = 24) -> list[dict]:
        """최근 `hours` 시간 이내의 레코드만 반환한다.

        - 파일 없으면 [] 반환
        - 파싱 실패 줄은 skip + WARNING 로그
        - 읽기 중 lock_timeout 초과 시 [] 반환 + WARNING 로그
        """
        if not self._path.exists():
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        try:
            with self._lock.acquire(timeout=self._lock_timeout):
                lines = self._path.read_text(encoding="utf-8").splitlines()
        except Timeout:
            logger.warning("MetricsWriter.read_recent(): 락 타임아웃 — 빈 목록 반환")
            return []

        results: list[dict] = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                ts_str = record.get("timestamp", "")
                ts = datetime.fromisoformat(ts_str)
                # timezone-aware 비교
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    results.append(record)
            except Exception as e:
                logger.warning("metrics_live.jsonl 줄 %d 파싱 실패 (skip): %s", i + 1, e)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 싱글턴
# ─────────────────────────────────────────────────────────────────────────────

_writer: Optional[MetricsWriter] = None


def get_metrics_writer() -> MetricsWriter:
    global _writer
    if _writer is None:
        raise RuntimeError(
            "MetricsWriter가 초기화되지 않았습니다. lifespan에서 init_metrics_writer() 호출 필요"
        )
    return _writer


def init_metrics_writer(
    path: Path = None,
    lock_timeout: float = None,
) -> MetricsWriter:
    global _writer
    _writer = MetricsWriter(path=path, lock_timeout=lock_timeout)
    return _writer
```

- [ ] **Step 4: 테스트 실행**

```bash
pytest tests/test_monitoring/test_metrics_writer.py -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 5: 전체 테스트 회귀 확인**

```bash
pytest --tb=short -q 2>&1 | tail -5
```

Expected: 이전과 동일한 passed 수 (+ 새 테스트)

- [ ] **Step 6: commit**

```bash
git add monitoring/metrics_writer.py tests/test_monitoring/test_metrics_writer.py
git commit -m "feat: MetricsWriter — filelock 기반 jsonl 기록기 + 싱글턴"
```

---

## Task 3: AlertType.RULE_ML_DISAGREE — 불일치율 알림 규칙

**Files:**
- Modify: `monitoring/alert_rules.py`
- Modify: `tests/test_monitoring/test_monitoring.py` (클래스 추가)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_monitoring/test_monitoring.py` 파일 끝에 다음 클래스를 추가한다:

```python
class TestAlertRulesDisagree:
    @pytest.fixture
    def manager(self, tmp_path):
        from monitoring.alert_rules import AlertManager
        return AlertManager(log_dir=str(tmp_path / "alerts"))

    def test_no_alert_below_warning_threshold(self, manager):
        from monitoring.alert_rules import AlertType
        alerts = manager.evaluate_rule_ml_disagree(0.10, "2026-04-06")
        assert len(alerts) == 0

    def test_warning_at_warning_threshold(self, manager):
        from monitoring.alert_rules import AlertSeverity, AlertType
        alerts = manager.evaluate_rule_ml_disagree(0.20, "2026-04-06")
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert alerts[0].alert_type == AlertType.RULE_ML_DISAGREE
        assert alerts[0].partition == "2026-04-06"

    def test_critical_at_critical_threshold(self, manager):
        from monitoring.alert_rules import AlertSeverity, AlertType
        alerts = manager.evaluate_rule_ml_disagree(0.35, "2026-04-06")
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert alerts[0].alert_type == AlertType.RULE_ML_DISAGREE

    def test_critical_threshold_takes_priority_over_warning(self, manager):
        from monitoring.alert_rules import AlertSeverity
        # critical_threshold=0.30 이상이면 CRITICAL만 (WARNING도 아님)
        alerts = manager.evaluate_rule_ml_disagree(0.31, "2026-04-06")
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_detail_contains_disagree_rate(self, manager):
        alerts = manager.evaluate_rule_ml_disagree(0.20, "2026-04-06")
        assert "disagree_rate" in alerts[0].detail
        assert abs(alerts[0].detail["disagree_rate"] - 0.20) < 1e-9

    def test_evaluate_all_includes_disagree(self, manager):
        from monitoring.alert_rules import AlertType
        alerts = manager.evaluate_all(
            partition="2026-04-06",
            disagree_rate=0.35,
        )
        types = [a.alert_type for a in alerts]
        assert AlertType.RULE_ML_DISAGREE in types
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_monitoring/test_monitoring.py::TestAlertRulesDisagree -v 2>&1 | head -20
```

Expected: `AttributeError: 'AlertManager' object has no attribute 'evaluate_rule_ml_disagree'`

- [ ] **Step 3: AlertType enum에 RULE_ML_DISAGREE 추가**

`monitoring/alert_rules.py`의 `AlertType` 클래스에서:

```python
# 기존
class AlertType(str, Enum):
    PSI_DRIFT          = "psi_drift"
    RECALL_DROP        = "recall_drop"
    CONSECUTIVE_DROP   = "consecutive_drop"
    DDI_DB_UPDATE      = "ddi_db_update"
    SCHEDULED_RETRAIN  = "scheduled_retrain"
```

를 다음으로 교체:

```python
class AlertType(str, Enum):
    PSI_DRIFT          = "psi_drift"
    RECALL_DROP        = "recall_drop"
    CONSECUTIVE_DROP   = "consecutive_drop"
    DDI_DB_UPDATE      = "ddi_db_update"
    SCHEDULED_RETRAIN  = "scheduled_retrain"
    RULE_ML_DISAGREE   = "rule_ml_disagree"
```

- [ ] **Step 4: evaluate_rule_ml_disagree() 메서드 추가**

`AlertManager` 클래스에서 `evaluate_ddi_db_update()` 메서드 직전에 다음을 삽입한다:

```python
    def evaluate_rule_ml_disagree(
        self,
        disagree_rate: float,
        partition: str,
        warning_threshold: float = 0.15,
        critical_threshold: float = 0.30,
    ) -> list[Alert]:
        """Rule/ML 불일치율 평가.

        Args:
            disagree_rate: 불일치 비율 (0.0 ~ 1.0)
            partition: 파티션 문자열 (YYYY-MM-DD)
            warning_threshold: WARNING 임계값 (기본 15%)
            critical_threshold: CRITICAL 임계값 (기본 30%)

        Returns:
            알림 리스트 (0 ~ 1건)
        """
        if disagree_rate >= critical_threshold:
            return [Alert(
                alert_type=AlertType.RULE_ML_DISAGREE,
                severity=AlertSeverity.CRITICAL,
                message=(
                    f"Rule/ML 불일치율 {disagree_rate:.1%} "
                    f"(CRITICAL 임계값 {critical_threshold:.0%} 초과)"
                ),
                partition=partition,
                detail={"disagree_rate": disagree_rate, "threshold": critical_threshold},
            )]
        if disagree_rate >= warning_threshold:
            return [Alert(
                alert_type=AlertType.RULE_ML_DISAGREE,
                severity=AlertSeverity.WARNING,
                message=(
                    f"Rule/ML 불일치율 {disagree_rate:.1%} "
                    f"(WARNING 임계값 {warning_threshold:.0%} 초과)"
                ),
                partition=partition,
                detail={"disagree_rate": disagree_rate, "threshold": warning_threshold},
            )]
        return []
```

- [ ] **Step 5: evaluate_all()에 disagree_rate 파라미터 추가**

`evaluate_all()` 메서드 시그니처와 바디를 다음으로 교체한다:

```python
    def evaluate_all(
        self,
        drift_report=None,
        snapshots=None,
        history=None,
        partition: str = "",
        n_new_ddi_rules: int = 0,
        disagree_rate: float = 0.0,
    ) -> list[Alert]:
        """전체 알림 규칙 일괄 평가."""
        alerts: list[Alert] = []
        if drift_report is not None:
            alerts += self.evaluate_drift(drift_report)
        if snapshots is not None:
            alerts += self.evaluate_performance(snapshots, history)
        if n_new_ddi_rules > 0:
            alerts += self.evaluate_ddi_db_update(partition, n_new_ddi_rules)
        if disagree_rate > 0.0:
            alerts += self.evaluate_rule_ml_disagree(disagree_rate, partition)
        return alerts
```

- [ ] **Step 6: 테스트 실행**

```bash
pytest tests/test_monitoring/test_monitoring.py::TestAlertRulesDisagree -v
```

Expected: 6개 테스트 모두 PASS

- [ ] **Step 7: 기존 알림 테스트 회귀 확인**

```bash
pytest tests/test_monitoring/test_monitoring.py -v
```

Expected: 모두 PASS (기존 TestAlertRules + 신규 TestAlertRulesDisagree)

- [ ] **Step 8: commit**

```bash
git add monitoring/alert_rules.py tests/test_monitoring/test_monitoring.py
git commit -m "feat: AlertType.RULE_ML_DISAGREE + evaluate_rule_ml_disagree() 추가"
```

---

## Task 4: GET /metrics 엔드포인트 + MetricsWriter 초기화

**Files:**
- Create: `serving/routers/metrics.py`
- Modify: `serving/main.py`

- [ ] **Step 1: serving/routers/metrics.py 신규 생성**

```python
"""
메트릭 조회 엔드포인트

GET /metrics  - 최근 24시간 예측 메트릭 조회 (X-Admin-Key 인증 필수)
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from monitoring.metrics_writer import get_metrics_writer
from serving.routers.health import _require_admin

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])


class MetricsResponse(BaseModel):
    records: list[dict]
    count: int
    hours: int


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    hours: int = 24,
    _: None = Depends(_require_admin),
) -> MetricsResponse:
    """최근 N시간 예측 메트릭 조회.

    X-Admin-Key 헤더 인증 필수. ADMIN_API_KEY 미설정 시 503 반환.
    """
    try:
        records = get_metrics_writer().read_recent(hours=hours)
    except Exception:
        logger.warning("메트릭 읽기 실패 — 빈 목록 반환", exc_info=True)
        records = []
    return MetricsResponse(records=records, count=len(records), hours=hours)
```

- [ ] **Step 2: serving/main.py — metrics 라우터 등록 + MetricsWriter 초기화**

`serving/main.py`에서 아래 두 곳을 수정한다.

**2a. import 추가** (기존 `from serving.routers import health, predict` 줄 아래):

```python
from serving.routers import health, predict, metrics as metrics_router
from monitoring.metrics_writer import init_metrics_writer
```

**2b. lifespan 함수에서 `init_predictor` 호출 직후에 추가**:

```python
    init_predictor(
        model_path=_model_path,
        ddi_matrix_path=str(_settings.DDI_MATRIX_PATH),
        drug_index_path=str(_settings.DRUG_INDEX_PATH),
        cyp_matrix_path=str(_settings.CYP_MATRIX_PATH),
    )
    logger.info("예측기 초기화 완료")
    # ↓ 신규 추가
    init_metrics_writer(
        path=_settings.METRICS_JSONL_PATH,
        lock_timeout=_settings.METRICS_JSONL_LOCK_TIMEOUT,
    )
    logger.info("MetricsWriter 초기화 완료: %s", _settings.METRICS_JSONL_PATH)
```

**2c. 라우터 등록** (기존 `app.include_router(predict.router)` 줄 아래):

```python
app.include_router(predict.router)
app.include_router(metrics_router.router)  # ← 신규
```

**2d. docstring 환경변수 목록에 추가** (기존 `ADMIN_API_KEY` 줄 아래):

```python
  DDI_METRICS_JSONL_PATH : 메트릭 jsonl 파일 경로 (기본: /app/data/monitoring/metrics_live.jsonl)
  DDI_METRICS_JSONL_LOCK_TIMEOUT : 락 타임아웃(초) (기본: 5.0)
```

- [ ] **Step 3: import 오류 없이 앱 로드 확인**

```bash
python -c "from serving.main import app; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: commit**

```bash
git add serving/routers/metrics.py serving/main.py
git commit -m "feat: GET /metrics 엔드포인트 + MetricsWriter lifespan 초기화"
```

---

## Task 5: predict.py — MetricsWriter 연결

**Files:**
- Modify: `serving/routers/predict.py`

- [ ] **Step 1: 실패하는 통합 테스트 작성 (Task 9 일부 선행)**

`tests/test_integration/test_monitoring_pipeline.py` 신규 생성 (Task 9에서 나머지 추가):

```python
"""모니터링 파이프라인 통합 테스트."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────────────────────────────
# predict → MetricsWriter 기록 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictMetricsWiring:
    @pytest.fixture
    def client_with_writer(self, tmp_path, monkeypatch):
        """MetricsWriter를 tmp_path에 연결한 TestClient."""
        jsonl_path = tmp_path / "metrics.jsonl"
        monkeypatch.setenv("DDI_METRICS_JSONL_PATH", str(jsonl_path))
        monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")

        # predictor mock
        from unittest.mock import MagicMock, patch
        from serving.schemas import PredictResponse, RiskLevel
        mock_pred = MagicMock()
        mock_pred.predict.return_value = PredictResponse(
            patient_id="P001",
            risk_level=RiskLevel.RED,
            rule_level=RiskLevel.RED,
            ml_level=RiskLevel.YELLOW,
            drugs=["A", "B"],
            interactions=[],
            rule_triggered=True,
            explanation="테스트",
        )

        import importlib
        import config.settings as s
        importlib.reload(s)

        from monitoring.metrics_writer import init_metrics_writer
        init_metrics_writer(path=jsonl_path)

        with patch("serving.predictor.get_predictor", return_value=mock_pred):
            from serving.main import app
            client = TestClient(app, raise_server_exceptions=False)
            yield client, jsonl_path

    def test_predict_writes_to_jsonl(self, client_with_writer):
        client, jsonl_path = client_with_writer
        resp = client.post("/predict", json={
            "patient_id": "P001",
            "drug_codes": ["A001", "B002"],
        })
        # 예측 성공 여부와 무관하게 jsonl 기록 확인
        if jsonl_path.exists():
            lines = jsonl_path.read_text().strip().splitlines()
            assert len(lines) >= 1
            record = json.loads(lines[0])
            assert "patient_id" in record
            assert "risk_level" in record
            assert "latency_ms" in record

    def test_predict_metrics_writer_failure_does_not_break_response(self, tmp_path, monkeypatch):
        """MetricsWriter.append() 예외 → 정상 응답 반환 검증."""
        from unittest.mock import MagicMock, patch
        from serving.schemas import PredictResponse, RiskLevel

        mock_pred = MagicMock()
        mock_pred.predict.return_value = PredictResponse(
            patient_id="P001",
            risk_level=RiskLevel.GREEN,
            rule_level=RiskLevel.GREEN,
            ml_level=None,
            drugs=["A"],
            interactions=[],
            rule_triggered=False,
            explanation="",
        )

        mock_writer = MagicMock()
        mock_writer.append.side_effect = RuntimeError("disk full")

        import monitoring.metrics_writer as mw
        monkeypatch.setattr(mw, "_writer", mock_writer)

        with patch("serving.predictor.get_predictor", return_value=mock_pred):
            from serving.main import app
            client = TestClient(app)
            resp = client.post("/predict", json={
                "patient_id": "P001",
                "drug_codes": ["A001"],
            })
        # MetricsWriter 실패해도 예측 응답 정상
        assert resp.status_code == 200


class TestMetricsEndpoint:
    def test_get_metrics_without_admin_key_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ADMIN_API_KEY", "secret-key")
        from monitoring.metrics_writer import init_metrics_writer
        init_metrics_writer(path=tmp_path / "metrics.jsonl")
        from serving.main import app
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code in (401, 403, 422, 503)

    def test_get_metrics_with_correct_admin_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ADMIN_API_KEY", "secret-key")
        jsonl_path = tmp_path / "metrics.jsonl"
        from monitoring.metrics_writer import init_metrics_writer
        init_metrics_writer(path=jsonl_path)
        from serving.main import app
        client = TestClient(app)
        resp = client.get("/metrics", headers={"X-Admin-Key": "secret-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert "records" in data
        assert "count" in data
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_integration/test_monitoring_pipeline.py::TestPredictMetricsWiring::test_predict_writes_to_jsonl -v 2>&1 | head -15
```

Expected: jsonl 파일이 없거나 빈 상태 (MetricsWriter 미연결)

- [ ] **Step 3: predict.py 수정 — import 추가**

`serving/routers/predict.py` 상단 import 블록에 추가:

```python
import time
from datetime import datetime, timezone

from monitoring.metrics_writer import get_metrics_writer
```

(기존에 `import time`이 있으면 datetime import만 추가)

- [ ] **Step 4: predict() 함수 수정**

기존 `predict()` 함수를:

```python
@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        pred = get_predictor()
        return pred.predict(req)
    except Exception as e:
        logger.exception("예측 처리 중 오류 (patient_id=%s)", req.patient_id)
        raise HTTPException(status_code=500, detail="내부 서버 오류: 예측 처리 실패")
```

다음으로 교체:

```python
@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    단일 환자 위험도 예측.

    - Rule-based Safety Net (Top 10 DDI 100% 탐지)
    - ML 모델 (XGBoost/LightGBM, 로드된 경우)
    - 최종등급 = max(Rule, ML)
    """
    try:
        pred = get_predictor()
        t0 = time.perf_counter()
        result = pred.predict(req)
        latency_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        logger.exception("예측 처리 중 오류 (patient_id=%s)", req.patient_id)
        raise HTTPException(status_code=500, detail="내부 서버 오류: 예측 처리 실패")

    try:
        get_metrics_writer().append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "partition": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "patient_id": req.patient_id,
            "risk_level": result.risk_level.value,
            "rule_level": result.rule_level.value if result.rule_level else None,
            "ml_level": result.ml_level.value if result.ml_level else None,
            "disagree": (
                result.rule_level != result.ml_level
                if result.rule_level and result.ml_level else False
            ),
            "latency_ms": round(latency_ms, 1),
            "source": "api",
        })
    except Exception:
        logger.warning("메트릭 기록 실패 — 예측은 정상 반환", exc_info=True)

    return result
```

- [ ] **Step 5: predict_batch() 루프 내부에도 MetricsWriter 연결**

기존 배치 루프:

```python
    for single_req in req.requests:
        try:
            results.append(pred.predict(single_req))
        except Exception as e:
            logger.warning("배치 부분 실패 (patient_id=%s): %s", single_req.patient_id, e)
            warnings.append(f"{single_req.patient_id}: 예측 처리 실패")
```

다음으로 교체:

```python
    for single_req in req.requests:
        try:
            t_single = time.perf_counter()
            single_result = pred.predict(single_req)
            single_latency_ms = (time.perf_counter() - t_single) * 1000
            results.append(single_result)
            try:
                get_metrics_writer().append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "partition": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "patient_id": single_req.patient_id,
                    "risk_level": single_result.risk_level.value,
                    "rule_level": single_result.rule_level.value if single_result.rule_level else None,
                    "ml_level": single_result.ml_level.value if single_result.ml_level else None,
                    "disagree": (
                        single_result.rule_level != single_result.ml_level
                        if single_result.rule_level and single_result.ml_level else False
                    ),
                    "latency_ms": round(single_latency_ms, 1),
                    "source": "api",
                })
            except Exception:
                logger.warning("배치 메트릭 기록 실패 (patient_id=%s)", single_req.patient_id)
        except Exception as e:
            logger.warning("배치 부분 실패 (patient_id=%s): %s", single_req.patient_id, e)
            warnings.append(f"{single_req.patient_id}: 예측 처리 실패")
```

- [ ] **Step 6: 테스트 실행**

```bash
pytest tests/test_integration/test_monitoring_pipeline.py::TestPredictMetricsWiring -v
pytest tests/test_integration/test_monitoring_pipeline.py::TestMetricsEndpoint -v
```

Expected: PASS

- [ ] **Step 7: 전체 테스트 회귀 확인**

```bash
pytest --tb=short -q 2>&1 | tail -5
```

Expected: passed 증가, 새 실패 없음

- [ ] **Step 8: commit**

```bash
git add serving/routers/predict.py tests/test_integration/test_monitoring_pipeline.py
git commit -m "feat: predict.py MetricsWriter 연결 + 통합 테스트 기본 프레임"
```

---

## Task 6: pipeline.py — DriftDetector 기준 분포 저장

**Files:**
- Modify: `scripts/train/pipeline.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_integration/test_monitoring_pipeline.py`에 클래스 추가:

```python
class TestPipelineDriftReference:
    def test_save_drift_reference_creates_pkl(self, tmp_path):
        """_save_drift_reference() → drift_reference.pkl 생성 및 로드 가능."""
        import numpy as np
        import pandas as pd
        train_df = pd.DataFrame({
            "drug_count": np.random.randint(1, 20, 100),
            "ddi_count": np.random.randint(0, 5, 100),
            "label": np.random.randint(0, 2, 100),
        })
        drift_ref_path = tmp_path / "drift_reference.pkl"

        from scripts.train.pipeline import TrainingPipeline
        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        # optional 파라미터로 경로 직접 전달 (settings 패치 불필요)
        pipeline._save_drift_reference(train_df, drift_reference_path=drift_ref_path)

        assert drift_ref_path.exists()
        from monitoring.drift_detector import DriftDetector
        loaded = DriftDetector.load(str(drift_ref_path))
        assert loaded._fitted
        assert "drug_count" in loaded._reference
        assert "ddi_count" in loaded._reference
        # label 컬럼은 제외됨
        assert "label" not in loaded._reference
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_integration/test_monitoring_pipeline.py::TestPipelineDriftReference -v 2>&1 | head -10
```

Expected: `AttributeError: type object 'TrainingPipeline' has no attribute '_save_drift_reference'`

- [ ] **Step 3: _save_drift_reference() 메서드 추가**

`scripts/train/pipeline.py`의 `TrainingPipeline` 클래스에서 `_run_gat_training()` 메서드 직전에 삽입한다:

```python
    def _save_drift_reference(self, train_df, drift_reference_path=None) -> None:
        """DriftDetector 기준 분포를 학습 데이터(train split)로 fit 후 저장.

        label, patient_id, split 컬럼은 피처에서 제외한다.
        배포 이전에 호출되어야 한다.

        Args:
            train_df: 학습 분할 DataFrame
            drift_reference_path: 저장 경로 오버라이드 (테스트용; None이면 settings.DRIFT_REFERENCE_PATH 사용)
        """
        from monitoring.drift_detector import DriftDetector
        from config import settings as _s

        path = drift_reference_path or _s.DRIFT_REFERENCE_PATH
        exclude_cols = {"label", "patient_id", "split"}
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        if not feature_cols:
            logger.warning("DriftDetector fit 대상 피처가 없음 — 기준 분포 저장 건너뜀")
            return

        detector = DriftDetector()
        detector.fit(train_df[feature_cols])
        path.parent.mkdir(parents=True, exist_ok=True)
        detector.save(str(path))
        logger.info("DriftDetector 기준 분포 저장 완료: %s (%d 피처)", path, len(feature_cols))
```

- [ ] **Step 4: run() 메서드에서 _save_drift_reference() 호출**

`run()` 메서드에서 **Step 6 (모델 저장) 직후**, `result.model_path = str(model_path)` 줄 바로 다음에 추가:

```python
            result.model_path = str(model_path)
            tracker.log_artifact(model_path, "model")

            # ── Step 6b: DriftDetector 기준 분포 저장 ────────────────────
            try:
                self._save_drift_reference(dataset.train)
            except Exception:
                logger.warning("DriftDetector 기준 분포 저장 실패 — 학습은 계속", exc_info=True)
```

`dataset.train`이 학습 분할 DataFrame이다. 프로젝트에서 `dataset.train`의 실제 속성명을 확인해야 한다.

- [ ] **Step 4b: dataset 속성명 확인**

```bash
grep -n "class.*Dataset\|self\.train\|\.train\b\|train_df\|X_train" scripts/train/trainer.py scripts/train/pipeline.py | head -20
```

`dataset.X_train`, `dataset.train_df`, 또는 다른 이름일 수 있다. 확인 후 올바른 속성명으로 교체.

만약 `dataset.X_train`이고 feature names가 `dataset.feature_names`이면:

```python
import pandas as pd
train_df_for_drift = pd.DataFrame(dataset.X_train, columns=dataset.feature_names)
self._save_drift_reference(train_df_for_drift)
```

- [ ] **Step 5: 테스트 실행**

```bash
pytest tests/test_integration/test_monitoring_pipeline.py::TestPipelineDriftReference -v
```

Expected: PASS

- [ ] **Step 6: commit**

```bash
git add scripts/train/pipeline.py tests/test_integration/test_monitoring_pipeline.py
git commit -m "feat: pipeline.py _save_drift_reference() — 학습 후 DriftDetector 기준 분포 저장"
```

---

## Task 7: DAG — 드리프트 감지 + 알림 태스크

**Files:**
- Modify: `dags/ddi_batch_predict_dag.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_integration/test_monitoring_pipeline.py`에 클래스 추가:

```python
class TestDAGDriftAndAlerts:
    @pytest.fixture
    def setup_drift_env(self, tmp_path):
        """_detect_drift, _generate_alerts 테스트를 위한 환경 셋업."""
        import numpy as np
        import pandas as pd
        from monitoring.drift_detector import DriftDetector
        from monitoring.metrics_writer import MetricsWriter
        from datetime import datetime, timezone

        # drift_reference.pkl 생성
        ref_df = pd.DataFrame({
            "drug_count": np.random.randint(1, 20, 200),
            "ddi_count": np.random.randint(0, 5, 200),
        })
        detector = DriftDetector()
        detector.fit(ref_df)
        drift_ref_path = tmp_path / "drift_reference.pkl"
        detector.save(str(drift_ref_path))

        # predictions_{partition}.parquet 생성 (drug_count, ddi_count 포함)
        partition = "2026-04-06"
        pred_path = tmp_path / f"predictions_{partition}.parquet"
        pred_df = pd.DataFrame({
            "patient_id": [f"P{i:03d}" for i in range(50)],
            "risk_level": ["RED"] * 10 + ["YELLOW"] * 20 + ["GREEN"] * 20,
            "drug_count": np.random.randint(1, 20, 50),
            "ddi_count": np.random.randint(0, 5, 50),
            "rule_triggered": [True] * 10 + [False] * 40,
        })
        pred_df.to_parquet(pred_path, index=False)

        # metrics_live.jsonl 생성 (Rule/ML 불일치 20%)
        metrics_path = tmp_path / "metrics_live.jsonl"
        writer = MetricsWriter(path=metrics_path)
        for i in range(10):
            writer.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "partition": partition,
                "patient_id": f"P{i:03d}",
                "risk_level": "RED",
                "rule_level": "RED",
                "ml_level": "YELLOW" if i < 2 else "RED",
                "disagree": i < 2,
                "latency_ms": 10.0,
                "source": "api",
            })

        return {
            "tmp_path": tmp_path,
            "partition": partition,
            "drift_ref_path": drift_ref_path,
            "pred_path": pred_path,
            "metrics_path": metrics_path,
            "monitoring_dir": tmp_path,
        }

    def test_detect_drift_creates_drift_json(self, setup_drift_env, monkeypatch):
        env = setup_drift_env
        import config.settings as _s
        # _detect_drift 내부에서 `from config import settings as _s` 패턴으로 접근하므로
        # setattr으로 모듈 속성 패치 가능
        monkeypatch.setattr(_s, "DRIFT_REFERENCE_PATH", env["drift_ref_path"])
        monkeypatch.setattr(_s, "PREDICTIONS_DIR", env["tmp_path"])
        monkeypatch.setattr(_s, "MONITORING_DIR", env["monitoring_dir"])

        from dags.ddi_batch_predict_dag import _detect_drift
        _detect_drift(partition=env["partition"])

        drift_json = env["tmp_path"] / f"drift_{env['partition']}.json"
        assert drift_json.exists()
        import json
        data = json.loads(drift_json.read_text())
        assert "partition" in data
        assert data["partition"] == env["partition"]

    def test_generate_alerts_creates_alert_json(self, setup_drift_env, monkeypatch):
        env = setup_drift_env
        partition = env["partition"]
        import config.settings as _s
        monkeypatch.setattr(_s, "MONITORING_DIR", env["monitoring_dir"])
        monkeypatch.setattr(_s, "METRICS_JSONL_PATH", env["metrics_path"])

        # 먼저 drift JSON을 수동 생성
        import json
        from pathlib import Path
        drift_json = env["tmp_path"] / f"drift_{partition}.json"
        drift_json.write_text(json.dumps({
            "partition": partition,
            "generated_at": "2026-04-06T00:00:00",
            "n_drifted": 0,
            "trigger_retrain": False,
            "summary": {"total_features": 2, "stable": 2, "warning": 0, "drift": 0},
            "features": [
                {"feature": "drug_count", "psi": 0.05, "status": "stable"},
                {"feature": "ddi_count", "psi": 0.03, "status": "stable"},
            ],
        }))

        from dags.ddi_batch_predict_dag import _generate_alerts
        _generate_alerts(partition=partition)

        alert_json = env["tmp_path"] / f"alerts_{partition}.json"
        assert alert_json.exists()
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_integration/test_monitoring_pipeline.py::TestDAGDriftAndAlerts -v 2>&1 | head -15
```

Expected: `ImportError: cannot import name '_detect_drift' from 'dags.ddi_batch_predict_dag'`

- [ ] **Step 3: ddi_batch_predict_dag.py에 _detect_drift() 추가**

기존 `_cleanup_staging()` 함수 다음에 삽입:

```python
def _detect_drift(partition: str) -> None:
    """배치 예측 parquet에서 PSI 드리프트를 감지하고 JSON 리포트를 저장한다.

    settings 접근을 `from config import settings as _s` 패턴으로 처리해
    테스트에서 monkeypatch.setattr이 정상 작동한다.
    """
    import pandas as pd
    from monitoring.drift_detector import DriftDetector
    from config import settings as _s  # 모듈 참조 — monkeypatch 대응

    drift_ref = _s.DRIFT_REFERENCE_PATH
    predictions_dir = _s.PREDICTIONS_DIR
    monitoring_dir = _s.MONITORING_DIR

    # drift_reference.pkl 존재 확인
    if not drift_ref.exists():
        logger.warning(
            "drift_reference.pkl 없음 (%s) — 드리프트 감지 건너뜀 (학습 파이프라인을 먼저 실행하세요)",
            drift_ref,
        )
        return

    pred_path = predictions_dir / f"predictions_{partition}.parquet"
    if not pred_path.exists():
        logger.warning("예측 파일 없음 (%s) — 드리프트 감지 건너뜀", pred_path)
        return

    df = pd.read_parquet(pred_path)
    # 배치 parquet에서 이용 가능한 컬럼만 사용
    # (완전한 feature vector는 ETL 고도화 단계에서 추가 예정)
    available_cols = [c for c in ("drug_count", "ddi_count", "rule_triggered") if c in df.columns]
    if not available_cols:
        logger.warning(
            "PSI 계산 가능한 컬럼 없음 (partition=%s, 컬럼=%s) — 드리프트 감지 건너뜀",
            partition, list(df.columns),
        )
        return

    detector = DriftDetector.load(str(drift_ref))
    report = detector.detect(df[available_cols], partition=partition)

    monitoring_dir.mkdir(parents=True, exist_ok=True)
    detector.save_report(report, str(monitoring_dir))
    logger.info(
        "드리프트 감지 완료 (partition=%s): %d 피처 분석, %d 드리프트",
        partition, len(report.feature_results), report.n_drifted,
    )
```

- [ ] **Step 4: _generate_alerts() 추가**

`_detect_drift()` 다음에 삽입:

```python
def _generate_alerts(partition: str) -> None:
    """드리프트 리포트와 Rule/ML 불일치율을 기반으로 알림을 생성하고 JSON으로 저장한다."""
    import json
    from types import SimpleNamespace
    from monitoring.alert_rules import AlertManager, AlertType, Alert, AlertSeverity
    from monitoring.metrics_writer import MetricsWriter
    from config import settings as _s  # 모듈 참조 — monkeypatch 대응

    monitoring_dir = _s.MONITORING_DIR
    metrics_jsonl_path = _s.METRICS_JSONL_PATH

    mgr = AlertManager()
    alerts: list[Alert] = []

    # ── 드리프트 알림 ───────────────────────────────────────────────────────
    report_path = monitoring_dir / f"drift_{partition}.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            drift_data = json.load(f)
        # DriftReport를 duck-type 오브젝트로 재구성 (JSON ↔ DriftReport 변환 없이 evaluate_drift 재사용)
        feat_results = [
            SimpleNamespace(
                feature_name=feat["feature"],
                psi=feat["psi"],
                status=feat["status"],
                is_drifted=feat["status"] == "drift",
            )
            for feat in drift_data.get("features", [])
        ]
        drift_obj = SimpleNamespace(
            partition=drift_data.get("partition", partition),
            feature_results=feat_results,
            summary=drift_data.get("summary", {}),
        )
        alerts += mgr.evaluate_drift(drift_obj)
    else:
        logger.warning("드리프트 리포트 없음 (%s) — 드리프트 알림 건너뜀", report_path)

    # ── Rule/ML 불일치율 알림 ───────────────────────────────────────────────
    writer = MetricsWriter(path=metrics_jsonl_path)
    records = [r for r in writer.read_recent(hours=24) if r.get("partition") == partition]
    disagree_rate = 0.0
    if records:
        disagree_rate = sum(1 for r in records if r.get("disagree")) / len(records)
        logger.info(
            "Rule/ML 불일치율 (partition=%s): %.1f%% (%d / %d 건)",
            partition, disagree_rate * 100, sum(1 for r in records if r.get("disagree")), len(records),
        )
    alerts += mgr.evaluate_rule_ml_disagree(disagree_rate, partition)

    # ── 알림 저장 ───────────────────────────────────────────────────────────
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    alert_path = monitoring_dir / f"alerts_{partition}.json"
    with open(alert_path, "w", encoding="utf-8") as f:
        json.dump([a.to_dict() for a in alerts], f, ensure_ascii=False, indent=2)
    logger.info("알림 생성 완료 (partition=%s): %d건", partition, len(alerts))
```

- [ ] **Step 5: DAG 블록에 새 태스크 추가**

`ddi_batch_predict_dag.py`의 DAG `with` 블록에서 기존 태스크 정의들 다음에 추가:

```python
    t_detect_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=_detect_drift,
        op_kwargs={"partition": "{{ ds }}"},
    )

    t_generate_alerts = PythonOperator(
        task_id="generate_alerts",
        python_callable=_generate_alerts,
        op_kwargs={"partition": "{{ ds }}"},
    )
```

- [ ] **Step 6: 태스크 체인 수정**

기존 체인:

```python
    (
        start
        >> wait_features
        >> t_partition
        >> t_health
        >> t_load
        >> t_predict
        >> t_summary
        >> t_cleanup
        >> end
    )
```

를 다음으로 교체:

```python
    (
        start
        >> wait_features
        >> t_partition
        >> t_health
        >> t_load
        >> t_predict
        >> t_summary
        >> t_detect_drift
        >> t_generate_alerts
        >> t_cleanup
        >> end
    )
```

- [ ] **Step 7: DriftDetector import 확인**

DAG 파일 상단 import에 아무것도 추가하지 않는다. `_detect_drift`와 `_generate_alerts` 내부에서 함수 로컬로 import하므로 DAG 파싱 시 불필요한 의존성을 로드하지 않는다.

- [ ] **Step 8: 테스트 실행**

```bash
pytest tests/test_integration/test_monitoring_pipeline.py::TestDAGDriftAndAlerts -v
```

Expected: PASS

- [ ] **Step 9: 전체 테스트 회귀**

```bash
pytest --tb=short -q 2>&1 | tail -5
```

- [ ] **Step 10: commit**

```bash
git add dags/ddi_batch_predict_dag.py tests/test_integration/test_monitoring_pipeline.py
git commit -m "feat: DAG _detect_drift + _generate_alerts 태스크 추가"
```

---

## Task 8: Streamlit 모니터링 페이지

**Files:**
- Create: `hana_app/pages/6_📊_모니터링.py`

참고: 기존 Streamlit 페이지는 `hana_app/pages/`에 있으며, 공통 DB 연결 등을 `hana_app/` 내부 유틸로 처리한다. 이 페이지는 DB 없이 파일 기반 데이터만 사용한다.

- [ ] **Step 1: 데이터 로직 헬퍼 함수 테스트 작성**

`tests/test_monitoring/test_dashboard_page.py` 신규 생성:

```python
"""Streamlit 대시보드 페이지 헬퍼 함수 테스트."""
from __future__ import annotations

import json
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestDashboardHelpers:
    def test_load_recent_metrics_returns_list(self, tmp_path):
        from hana_app.pages._monitoring_helpers import load_recent_metrics
        ts = datetime.now(timezone.utc).isoformat()
        path = tmp_path / "metrics.jsonl"
        _write_jsonl(path, [{"timestamp": ts, "patient_id": "P001", "risk_level": "RED", "disagree": False}])
        records = load_recent_metrics(path, hours=24)
        assert len(records) == 1
        assert records[0]["patient_id"] == "P001"

    def test_load_recent_metrics_file_missing_returns_empty(self, tmp_path):
        from hana_app.pages._monitoring_helpers import load_recent_metrics
        records = load_recent_metrics(tmp_path / "nonexistent.jsonl", hours=24)
        assert records == []

    def test_load_drift_report_returns_dict(self, tmp_path):
        from hana_app.pages._monitoring_helpers import load_drift_report
        partition = "2026-04-06"
        report = {
            "partition": partition,
            "n_drifted": 1,
            "summary": {"stable": 1, "warning": 0, "drift": 1},
            "features": [{"feature": "drug_count", "psi": 0.30, "status": "drift"}],
        }
        (tmp_path / f"drift_{partition}.json").write_text(json.dumps(report))
        loaded = load_drift_report(tmp_path, partition)
        assert loaded["n_drifted"] == 1
        assert loaded["features"][0]["status"] == "drift"

    def test_load_drift_report_missing_returns_none(self, tmp_path):
        from hana_app.pages._monitoring_helpers import load_drift_report
        result = load_drift_report(tmp_path, "2026-04-06")
        assert result is None

    def test_load_alerts_returns_list(self, tmp_path):
        from hana_app.pages._monitoring_helpers import load_alerts
        partition = "2026-04-06"
        alerts = [
            {"alert_type": "psi_drift", "severity": "CRITICAL", "message": "드리프트", "generated_at": "2026-04-06T00:00:00"},
        ]
        (tmp_path / f"alerts_{partition}.json").write_text(json.dumps(alerts))
        result = load_alerts(tmp_path, [partition])
        assert len(result) == 1
        assert result[0]["severity"] == "CRITICAL"

    def test_load_alerts_no_files_returns_empty(self, tmp_path):
        from hana_app.pages._monitoring_helpers import load_alerts
        result = load_alerts(tmp_path, ["2026-04-06"])
        assert result == []

    def test_compute_disagree_rate_zero_records(self):
        from hana_app.pages._monitoring_helpers import compute_disagree_rate
        assert compute_disagree_rate([]) == 0.0

    def test_compute_disagree_rate_correct(self):
        from hana_app.pages._monitoring_helpers import compute_disagree_rate
        records = [
            {"disagree": True},
            {"disagree": False},
            {"disagree": True},
            {"disagree": False},
        ]
        assert abs(compute_disagree_rate(records) - 0.5) < 1e-9

    def test_psi_status_label(self):
        from hana_app.pages._monitoring_helpers import psi_status_label
        assert psi_status_label(0.05) == "🟢 Stable"
        assert psi_status_label(0.15) == "🟡 Warning"
        assert psi_status_label(0.30) == "🔴 Drift"

    def test_get_recent_partitions(self, tmp_path):
        from hana_app.pages._monitoring_helpers import get_recent_partitions
        for p in ["2026-04-04", "2026-04-05", "2026-04-06"]:
            (tmp_path / f"drift_{p}.json").write_text("{}")
        result = get_recent_partitions(tmp_path, prefix="drift_", n=7)
        assert "2026-04-06" in result
        assert len(result) <= 7
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_monitoring/test_dashboard_page.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'hana_app.pages._monitoring_helpers'`

- [ ] **Step 3: 헬퍼 모듈 생성**

`hana_app/pages/_monitoring_helpers.py` 신규 생성:

```python
"""Streamlit 모니터링 페이지용 데이터 로딩 헬퍼."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def load_recent_metrics(path: Path, hours: int = 24) -> list[dict]:
    """metrics_live.jsonl에서 최근 hours 시간 이내 레코드를 반환한다."""
    path = Path(path)
    if not path.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    results = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            ts_str = record.get("timestamp", "")
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                results.append(record)
        except Exception as e:
            logger.warning("metrics 줄 %d 파싱 실패 (skip): %s", i + 1, e)
    return results


def load_drift_report(monitoring_dir: Path, partition: str) -> dict | None:
    """drift_{partition}.json을 로드한다. 없으면 None 반환."""
    path = Path(monitoring_dir) / f"drift_{partition}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("drift 리포트 파싱 실패 (%s): %s", path, e)
        return None


def load_alerts(monitoring_dir: Path, partitions: list[str]) -> list[dict]:
    """최근 partitions의 alerts_*.json을 합쳐서 반환한다."""
    monitoring_dir = Path(monitoring_dir)
    all_alerts = []
    for p in partitions:
        path = monitoring_dir / f"alerts_{p}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                all_alerts.extend(data)
            elif isinstance(data, dict) and "alerts" in data:
                all_alerts.extend(data["alerts"])
        except Exception as e:
            logger.warning("알림 파일 파싱 실패 (%s): %s", path, e)
    return all_alerts


def compute_disagree_rate(records: list[dict]) -> float:
    """Rule/ML 불일치율을 계산한다."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.get("disagree")) / len(records)


def psi_status_label(psi: float) -> str:
    """PSI 값에 따른 상태 레이블 반환."""
    if psi < 0.10:
        return "🟢 Stable"
    elif psi < 0.25:
        return "🟡 Warning"
    return "🔴 Drift"


def get_recent_partitions(monitoring_dir: Path, prefix: str = "drift_", n: int = 7) -> list[str]:
    """monitoring_dir에서 prefix로 시작하는 최근 n개 파티션 날짜를 반환한다."""
    monitoring_dir = Path(monitoring_dir)
    partitions = []
    for f in monitoring_dir.glob(f"{prefix}*.json"):
        name = f.stem  # e.g., "drift_2026-04-06"
        date_part = name[len(prefix):]
        if len(date_part) == 10:  # YYYY-MM-DD
            partitions.append(date_part)
    return sorted(partitions, reverse=True)[:n]
```

- [ ] **Step 4: 헬퍼 테스트 통과 확인**

```bash
pytest tests/test_monitoring/test_dashboard_page.py -v
```

Expected: 모두 PASS

- [ ] **Step 5: Streamlit 페이지 구현**

`hana_app/pages/6_📊_모니터링.py` 신규 생성:

```python
"""
모니터링 대시보드 — Streamlit 6번 페이지

4-tab 구성:
  Tab 1: 실시간 예측 현황 (metrics_live.jsonl)
  Tab 2: 드리프트 감지 (drift_{partition}.json)
  Tab 3: 알림 이력 (alerts_{partition}.json)
  Tab 4: 시스템 상태
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from hana_app.pages._monitoring_helpers import (
    compute_disagree_rate,
    get_recent_partitions,
    load_alerts,
    load_drift_report,
    load_recent_metrics,
    psi_status_label,
)

# ─────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────────────────

METRICS_JSONL_PATH = Path(os.environ.get("DDI_METRICS_JSONL_PATH", "/app/data/monitoring/metrics_live.jsonl"))
MONITORING_DIR     = Path(os.environ.get("DDI_MONITORING_DIR",     "/app/data/monitoring"))
DRIFT_REFERENCE_PATH = Path(os.environ.get("DDI_DRIFT_REFERENCE_PATH", "/app/models/current/drift_reference.pkl"))
SERVING_URL        = os.environ.get("DDI_SERVING_URL", "http://localhost:8000")

# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="모니터링 대시보드", layout="wide")
st.title("📊 모니터링 대시보드")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 실시간 예측 현황",
    "🌊 드리프트 감지",
    "⚠️ 알림 이력",
    "🔧 시스템 상태",
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: 실시간 예측 현황
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.subheader("실시간 예측 현황 (최근 24시간)")

    records = load_recent_metrics(METRICS_JSONL_PATH, hours=24)

    if not records:
        st.info("아직 예측 데이터가 없습니다. API를 통해 예측을 실행하면 여기에 표시됩니다.")
    else:
        df = pd.DataFrame(records)

        # 지표 요약
        col1, col2, col3, col4 = st.columns(4)
        total = len(df)
        disagree_rate = compute_disagree_rate(records)

        # 데이터 기간 안내
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            n_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
            period_label = f"지난 {n_days}일" if n_days < 30 else "최근 30일"
        else:
            period_label = "최근 24시간"

        col1.metric("총 예측 건수", f"{total:,}", help=period_label)
        col2.metric(
            "Rule/ML 불일치율",
            f"{disagree_rate:.1%}",
            help="rule_level ≠ ml_level 비율"
        )
        if "risk_level" in df.columns:
            red_count = (df["risk_level"] == "RED").sum()
            col3.metric("고위험(RED) 비율", f"{red_count/total:.1%}" if total else "0.0%")
        if "latency_ms" in df.columns:
            col4.metric("평균 응답시간", f"{df['latency_ms'].mean():.1f}ms")

        # 위험도 분포
        if "risk_level" in df.columns:
            st.subheader("위험도 분포")
            dist = df["risk_level"].value_counts().reset_index()
            dist.columns = ["risk_level", "count"]
            st.bar_chart(dist.set_index("risk_level"))

        # 시간대별 예측 추이 (1시간 집계)
        if "timestamp" in df.columns:
            st.subheader("시간대별 예측 추이 (1시간 집계)")
            df_hourly = df.set_index("timestamp").resample("1h").size().reset_index()
            df_hourly.columns = ["timestamp", "count"]
            st.line_chart(df_hourly.set_index("timestamp"))

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: 드리프트 감지
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("드리프트 감지 (PSI)")

    partitions = get_recent_partitions(MONITORING_DIR, prefix="drift_", n=7)
    if not partitions:
        st.info("아직 드리프트 데이터가 없습니다. 배치 DAG를 실행하면 여기에 표시됩니다.")
    else:
        selected_partition = st.selectbox("파티션 선택", partitions, index=0)
        report = load_drift_report(MONITORING_DIR, selected_partition)
        if report is None:
            st.info(f"파티션 {selected_partition}의 드리프트 데이터가 없습니다.")
        else:
            summary = report.get("summary", {})
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("분석 피처 수", summary.get("total_features", "N/A"))
            col2.metric("🟢 Stable", summary.get("stable", 0))
            col3.metric("🟡 Warning", summary.get("warning", 0))
            col4.metric("🔴 Drift", summary.get("drift", 0))

            if report.get("trigger_retrain"):
                st.error("⚡ 긴급 재학습 트리거 — 드리프트 피처 2개 이상 감지")

            features = report.get("features", [])
            if features:
                feat_df = pd.DataFrame(features)
                feat_df["상태"] = feat_df["psi"].apply(psi_status_label)
                feat_df = feat_df.rename(columns={"feature": "피처", "psi": "PSI", "status": "status"})
                st.dataframe(
                    feat_df[["피처", "PSI", "상태"]].sort_values("PSI", ascending=False),
                    use_container_width=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: 알림 이력
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("알림 이력 (최근 7일)")

    alert_partitions = get_recent_partitions(MONITORING_DIR, prefix="alerts_", n=7)
    alerts = load_alerts(MONITORING_DIR, alert_partitions)

    if not alerts:
        st.success("✅ 정상 — 최근 7일 내 발생한 알림이 없습니다.")
    else:
        alert_df = pd.DataFrame(alerts)
        # severity 색상 강조
        severity_icon = {
            "CRITICAL": "🔴",
            "WARNING": "🟡",
            "INFO": "🔵",
        }
        if "severity" in alert_df.columns:
            alert_df["severity"] = alert_df["severity"].apply(
                lambda s: f"{severity_icon.get(s, '')} {s}"
            )
        display_cols = [c for c in ("generated_at", "alert_type", "severity", "message") if c in alert_df.columns]
        st.dataframe(alert_df[display_cols].sort_values("generated_at", ascending=False) if "generated_at" in alert_df.columns else alert_df[display_cols], use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: 시스템 상태
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.subheader("시스템 상태")

    # Serving API 헬스체크
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Serving API**")
        try:
            import requests
            resp = requests.get(f"{SERVING_URL}/health", timeout=3)
            if resp.status_code == 200:
                st.success(f"✅ 정상 ({SERVING_URL})")
                health_data = resp.json()
                st.json(health_data)
            else:
                st.error(f"❌ 응답 오류 (HTTP {resp.status_code})")
        except Exception as e:
            st.warning(f"⚠️ 연결 실패: {e}")

    with col2:
        st.markdown("**모니터링 파일 상태**")
        # metrics_live.jsonl
        if METRICS_JSONL_PATH.exists():
            stat = METRICS_JSONL_PATH.stat()
            st.write(f"📄 metrics_live.jsonl: {stat.st_size / 1024:.1f} KB")
            import datetime
            mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
            st.write(f"   마지막 수정: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning("metrics_live.jsonl 없음")

        # drift_reference.pkl
        if DRIFT_REFERENCE_PATH.exists():
            stat = DRIFT_REFERENCE_PATH.stat()
            import datetime
            mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
            age_days = (datetime.datetime.now() - mtime).days
            label = f"✅ drift_reference.pkl ({age_days}일 전)"
            if age_days > 180:
                st.warning(f"⚠️ {label} — 180일 이상 경과, 재학습 권고")
            else:
                st.success(label)
        else:
            st.error("❌ drift_reference.pkl 없음 — 학습 파이프라인 실행 필요")
```

- [ ] **Step 6: 헬퍼 테스트 재확인**

```bash
pytest tests/test_monitoring/test_dashboard_page.py -v
```

Expected: PASS

- [ ] **Step 7: Streamlit 문법 검사 (import 오류 확인)**

```bash
python -c "import ast; ast.parse(open('hana_app/pages/6_📊_모니터링.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 8: commit**

```bash
git add "hana_app/pages/6_📊_모니터링.py" hana_app/pages/_monitoring_helpers.py tests/test_monitoring/test_dashboard_page.py
git commit -m "feat: Streamlit 모니터링 페이지 (4-tab) + 헬퍼 모듈"
```

---

## Task 9: 최종 회귀 검증 및 커밋

**Files:**
- Modify: `tests/test_integration/test_monitoring_pipeline.py` (Task 6, 7 테스트 추가 후 최종 점검)

- [ ] **Step 1: 전체 테스트 실행**

```bash
pytest --tb=short -q 2>&1 | tail -10
```

Expected: 새 실패 없음

- [ ] **Step 2: 모니터링 관련 테스트만 별도 실행**

```bash
pytest tests/test_monitoring/ tests/test_integration/test_monitoring_pipeline.py -v
```

Expected: 모두 PASS

- [ ] **Step 3: settings 환경변수 오버라이드 스모크 테스트**

```bash
DDI_METRICS_JSONL_PATH=/tmp/test_metrics.jsonl python -c "
from config.settings import METRICS_JSONL_PATH
assert str(METRICS_JSONL_PATH) == '/tmp/test_metrics.jsonl', METRICS_JSONL_PATH
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: 최종 커밋**

```bash
git add -u
git commit -m "test: 모니터링 파이프라인 통합 테스트 완성"
```

---

## 구현 노트

### pipeline.py dataset.train 속성명
Task 6 Step 4b에서 `dataset.train` 속성명을 확인해야 한다. 프로젝트에서 사용하는 Dataset 클래스의 실제 train split 속성명:

```bash
grep -n "class.*Dataset\|self\.train\b\|\.train_df\|X_train" scripts/train/trainer.py | head -10
```

결과에 따라 `dataset.train`, `dataset.X_train`, `dataset.train_df` 중 맞는 것 사용.

### MetricsWriter Timeout 처리
- `append()`: Timeout 예외 그대로 전파 → 호출자(predict.py)가 try/except로 묵과
- `read_recent()`: Timeout 시 `[]` 반환 + WARNING 로그

### SimpleNamespace DriftReport 재구성
`_generate_alerts()`에서 `json.load()` 후 `SimpleNamespace`로 재구성하는 것은 `evaluate_drift()`의 duck typing을 활용한다. `feature_name`, `is_drifted`, `status` 속성이 `FeatureDriftResult`와 동일해야 한다.
