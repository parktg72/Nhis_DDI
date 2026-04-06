# 모니터링 대시보드 설계 문서

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 실시간 예측 메트릭 수집, 드리프트 감지, Rule/ML 불일치 알림을 기존 Streamlit 앱(hana_app/)에 모니터링 페이지로 추가하고, Grafana 대시보드와 Airflow DAG를 완성한다.

**Architecture:** FastAPI serving에서 예측 결과를 `metrics_live.jsonl`(filelock 보호)에 기록하고, DAG 배치 후처리에서 PSI/드리프트를 계산한다. Streamlit 6번 페이지가 두 데이터 소스를 읽어 4-tab 대시보드를 제공한다. Prometheus 서버는 사용하지 않는다(로컬 개발 환경).

**Tech Stack:** FastAPI, Streamlit, filelock, pandas, Airflow, 기존 monitoring/ 모듈

---

## 1. 파일 구조

| 파일 | 역할 | 상태 |
|------|------|------|
| `config/settings.py` | 모니터링 경로 상수 추가 | 수정 |
| `monitoring/metrics_writer.py` | filelock 기반 jsonl 기록기 | **신규** |
| `monitoring/alert_rules.py` | AlertType.RULE_ML_DISAGREE 추가 | 수정 |
| `serving/routers/metrics.py` | GET /metrics 엔드포인트 | **신규** |
| `serving/routers/predict.py` | record_prediction() 연결 | 수정 |
| `serving/main.py` | metrics 라우터 등록 | 수정 |
| `scripts/train/pipeline.py` | DriftDetector.fit() + save() 호출 | 수정 |
| `dags/ddi_batch_predict_dag.py` | t_detect_drift, t_generate_alerts 태스크 추가 | 수정 |
| `hana_app/pages/6_📊_모니터링.py` | Streamlit 모니터링 페이지 (4-tab) | **신규** |
| `tests/test_monitoring/test_metrics_writer.py` | MetricsWriter 유닛 테스트 | **신규** |
| `tests/test_monitoring/test_dashboard_page.py` | Streamlit 페이지 로직 테스트 | **신규** |
| `tests/test_integration/test_monitoring_pipeline.py` | 전체 파이프라인 통합 테스트 | **신규** |

---

## 2. 설정 (config/settings.py)

기존 경로 상수 블록 끝에 추가:

```python
# ── 모니터링 ────────────────────────────────────────────────────────────────────
MONITORING_DIR        = Path(os.environ.get("DDI_MONITORING_DIR",    "/app/data/monitoring"))
METRICS_JSONL_PATH    = Path(os.environ.get("DDI_METRICS_JSONL_PATH", "/app/data/monitoring/metrics_live.jsonl"))
DRIFT_REFERENCE_PATH  = Path(os.environ.get("DDI_DRIFT_REFERENCE_PATH", "/app/models/current/drift_reference.pkl"))
METRICS_JSONL_LOCK_TIMEOUT = float(os.environ.get("DDI_METRICS_JSONL_LOCK_TIMEOUT", "5.0"))  # seconds
```

---

## 3. metrics_live.jsonl 스키마

각 줄은 독립적인 JSON 객체(JSON Lines 형식):

```json
{
  "timestamp": "2026-04-06T10:23:45.123456",
  "partition": "2026-04-06",
  "patient_id": "P001",
  "risk_level": "RED",
  "rule_level": "RED",
  "ml_level": "YELLOW",
  "disagree": true,
  "latency_ms": 12.3,
  "source": "api"
}
```

**필드 규칙:**
- `timestamp`: ISO 8601, 밀리초 포함
- `partition`: `YYYY-MM-DD` (배치 DAG 파티션과 동일 포맷)
- `risk_level` / `rule_level` / `ml_level`: `RED | YELLOW | GREEN | NORMAL`
- `ml_level`: ML 모델 미로드 시 `null`
- `disagree`: `rule_level != ml_level` (둘 다 non-null일 때만 유효; 한쪽 null이면 `false`)
- `source`: API 경유 → `"api"`, 배치 DAG 경유 → `"batch"`

---

## 4. MetricsWriter (monitoring/metrics_writer.py)

### 설계 원칙
- **filelock**: `from filelock import FileLock` — 멀티 워커 안전 (fcntl 사용 금지)
- **append-only**: 기존 항목 수정 없음
- **rolling 24h**: `read_recent()` 메서드에서 timestamp 필터링; 파일 자체는 truncate하지 않음

### 공개 인터페이스

```python
from monitoring.metrics_writer import MetricsWriter

writer = MetricsWriter(
    path: Path = settings.METRICS_JSONL_PATH,
    lock_timeout: float = settings.METRICS_JSONL_LOCK_TIMEOUT,
)

writer.append(record: dict) -> None
# filelock 획득 → 파일 끝에 JSON 한 줄 추가 → flush → lock 해제
# lock_timeout 초과 시 Timeout 예외 (filelock.Timeout) 그대로 전파

writer.read_recent(hours: int = 24) -> list[dict]
# now - hours 이후 timestamp 레코드만 반환
# 파일 없으면 [] 반환
# 파싱 실패 줄 → skip + logger.warning (전체 실패하지 않음)
```

### 인스턴스화 전략
FastAPI lifespan에서 single instance 생성, `serving/predictor.py`의 `get_predictor()`와 동일한 패턴으로 `get_metrics_writer()` 제공.

---

## 5. serving/routers/predict.py 수정

예측 완료 후 `MetricsWriter.append()` 호출:

```python
# predict() 함수 내, return pred.predict(req) 전후에 추가
t0 = time.perf_counter()
result = pred.predict(req)
latency_ms = (time.perf_counter() - t0) * 1000

try:
    get_metrics_writer().append({
        "timestamp": datetime.utcnow().isoformat(),
        "partition": datetime.utcnow().strftime("%Y-%m-%d"),
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

**원칙:** 메트릭 기록 실패는 예측 응답을 중단시키지 않음 (try/except 필수).

`predict_batch()` 내부 루프에도 동일하게 적용 (`source="api"`).

---

## 6. serving/routers/metrics.py (신규)

```
GET /metrics
  Header: X-Admin-Key: <ADMIN_API_KEY>
  응답: {"records": [...], "count": N, "hours": 24}
  인증 실패: 403
  파일 없음: {"records": [], "count": 0, "hours": 24}
```

ADMIN_API_KEY 미설정(`""`) 시 모든 요청 거부 (기존 admin 엔드포인트와 동일 정책).

---

## 7. AlertType 확장 (monitoring/alert_rules.py)

### AlertType enum 추가 항목

```python
class AlertType(str, Enum):
    PSI_DRIFT          = "psi_drift"
    RECALL_DROP        = "recall_drop"
    CONSECUTIVE_DROP   = "consecutive_drop"
    DDI_DB_UPDATE      = "ddi_db_update"
    SCHEDULED_RETRAIN  = "scheduled_retrain"
    RULE_ML_DISAGREE   = "rule_ml_disagree"   # ← 신규
```

### AlertManager 신규 메서드

```python
def evaluate_rule_ml_disagree(
    self,
    disagree_rate: float,
    partition: str,
    warning_threshold: float = 0.15,
    critical_threshold: float = 0.30,
) -> list[Alert]:
    """Rule/ML 불일치율 평가."""
    if disagree_rate >= critical_threshold:
        return [Alert(
            alert_type=AlertType.RULE_ML_DISAGREE,
            severity=AlertSeverity.CRITICAL,
            message=f"Rule/ML 불일치율 {disagree_rate:.1%} (임계값 {critical_threshold:.0%} 초과)",
            partition=partition,
            detail={"disagree_rate": disagree_rate, "threshold": critical_threshold},
        )]
    if disagree_rate >= warning_threshold:
        return [Alert(
            alert_type=AlertType.RULE_ML_DISAGREE,
            severity=AlertSeverity.WARNING,
            message=f"Rule/ML 불일치율 {disagree_rate:.1%} (임계값 {warning_threshold:.0%} 초과)",
            partition=partition,
            detail={"disagree_rate": disagree_rate, "threshold": warning_threshold},
        )]
    return []
```

`evaluate_all()` 내부에서 `evaluate_rule_ml_disagree()` 호출 추가 (disagree_rate 파라미터 주입).

---

## 8. 학습 파이프라인 DriftDetector 통합 (scripts/train/pipeline.py)

학습 완료 후(`_run_training()` 또는 `run()` 내부 적절한 위치에):

```python
def _save_drift_reference(self, train_df) -> None:
    """DriftDetector 기준 분포를 학습 데이터로 fit 후 저장."""
    from monitoring.drift_detector import DriftDetector
    from config.settings import DRIFT_REFERENCE_PATH

    feature_cols = [c for c in train_df.columns if c not in ("label", "patient_id", "split")]
    detector = DriftDetector()
    detector.fit(train_df[feature_cols])
    DRIFT_REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    detector.save(str(DRIFT_REFERENCE_PATH))
    logger.info("DriftDetector 기준 분포 저장 완료: %s", DRIFT_REFERENCE_PATH)
```

호출 시점: `_run_training()` 완료 직후, 배포 이전.

---

## 9. DAG 배치 후처리 (dags/ddi_batch_predict_dag.py)

### 추가 태스크

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

### 태스크 순서

```
t_batch_predict >> t_detect_drift >> t_generate_alerts
```

### _detect_drift 구현

```python
def _detect_drift(partition: str) -> None:
    from monitoring.drift_detector import DriftDetector
    from config.settings import DRIFT_REFERENCE_PATH, PREDICTIONS_DIR
    import pandas as pd

    detector = DriftDetector.load(str(DRIFT_REFERENCE_PATH))

    pred_path = PREDICTIONS_DIR / f"predictions_{partition}.parquet"
    df = pd.read_parquet(pred_path)

    # 이용 가능한 컬럼만 사용 (feature vector 전체 없음 — ETL 단계에서 확장 예정)
    available_cols = [c for c in ("drug_count", "ddi_count", "rule_triggered") if c in df.columns]
    if not available_cols:
        logger.warning("PSI 계산 가능한 컬럼 없음 (partition=%s) — drift 감지 건너뜀", partition)
        return

    report = detector.detect(df[available_cols], partition=partition)
    detector.save_report(report, str(MONITORING_DIR))
    logger.info("드리프트 감지 완료 (partition=%s): %d 피처 분석", partition, len(report.feature_results))
```

**범위 결정:** 현재 배치 parquet에서 이용 가능한 컬럼(drug_count, ddi_count, rule_triggered)으로 제한. 완전한 feature vector 저장은 ETL 고도화 단계에서 추가.

### _generate_alerts 구현

```python
def _generate_alerts(partition: str) -> None:
    from monitoring.alert_rules import AlertManager
    from monitoring.metrics_writer import MetricsWriter
    from config.settings import MONITORING_DIR, METRICS_JSONL_PATH
    import json, pandas as pd

    # 드리프트 리포트 로드
    report_path = MONITORING_DIR / f"drift_{partition}.json"
    drift_report = None
    if report_path.exists():
        with open(report_path) as f:
            drift_report = json.load(f)

    # Rule/ML 불일치율 계산 (metrics_live.jsonl에서)
    writer = MetricsWriter(path=METRICS_JSONL_PATH)
    records = [r for r in writer.read_recent(hours=24) if r.get("partition") == partition]
    disagree_rate = 0.0
    if records:
        disagree_rate = sum(1 for r in records if r.get("disagree")) / len(records)

    mgr = AlertManager()
    alerts = []
    if drift_report:
        alerts += mgr.evaluate_drift(drift_report)
    alerts += mgr.evaluate_rule_ml_disagree(disagree_rate, partition)

    alert_path = MONITORING_DIR / f"alerts_{partition}.json"
    with open(alert_path, "w") as f:
        json.dump([a.to_dict() for a in alerts], f, ensure_ascii=False, indent=2)
    logger.info("알림 생성 완료 (partition=%s): %d건", partition, len(alerts))
```

---

## 10. Streamlit 모니터링 페이지 (hana_app/pages/6_📊_모니터링.py)

### 4-tab 구조

```python
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 실시간 예측 현황",
    "🌊 드리프트 감지",
    "⚠️ 알림 이력",
    "🔧 시스템 상태",
])
```

### Tab 1: 실시간 예측 현황
- **데이터 소스:** `MetricsWriter.read_recent(hours=24)`
- 24시간 예측 건수, 위험도 분포 (파이차트)
- Rule/ML 불일치율 (게이지 0-100%)
- 시간대별 예측 건수 추이 (선 그래프, 1시간 집계)
- 최근 30일 집계: 데이터 < 30일이면 "지난 N일 기준" 표시

### Tab 2: 드리프트 감지
- **데이터 소스:** `data/monitoring/drift_{partition}.json`
- 가장 최근 파티션의 PSI 값 테이블 (피처명, PSI, 상태)
- PSI 상태 기준: < 0.10 🟢 Stable, 0.10-0.25 🟡 Warning, ≥ 0.25 🔴 Drift
- 파티션 선택 드롭다운 (최근 7일)
- 드리프트 파일 없음 → "아직 드리프트 데이터가 없습니다" 메시지

### Tab 3: 알림 이력
- **데이터 소스:** `data/monitoring/alerts_{partition}.json` (최근 7일)
- 알림 테이블: generated_at, alert_type, severity, message
- severity별 색상: CRITICAL 빨강, WARNING 주황, INFO 파랑
- 알림 없음 → "정상 — 발생한 알림 없음" 메시지

### Tab 4: 시스템 상태
- Serving API 헬스체크 (GET /health)
- 모델 로드 상태 (GET /model/info 또는 캐시된 predictor 상태)
- metrics_live.jsonl 파일 크기, 마지막 수정 시각
- drift_reference.pkl 파일 존재 여부, 파일 나이

---

## 11. 에러 처리 원칙

| 상황 | 처리 |
|------|------|
| MetricsWriter.append() 실패 | try/except로 묵과, WARNING 로그, 예측 응답 중단 없음 |
| filelock Timeout | 예외 전파 (append는 묵과 가능; read는 [] 반환 후 WARNING) |
| jsonl 파싱 오류 줄 | skip + WARNING 로그 (전체 실패하지 않음) |
| DAG drift 파일 없음 | WARNING 로그 후 건너뜀 |
| Streamlit 데이터 없음 | 빈 상태 안내 메시지 표시 (st.info) |
| 30일 미만 데이터 | "지난 N일 기준" 레이블로 graceful fallback |

---

## 12. 테스트 전략

### 유닛 테스트 (tests/test_monitoring/test_metrics_writer.py)

```
- append() → 파일에 JSON Lines 형식 기록 검증
- append() 동시 호출 → 데이터 유실 없음 (threading 사용)
- read_recent(hours=1) → 오래된 레코드 필터링 검증
- 파일 없음 → read_recent() 빈 리스트 반환
- 파싱 실패 줄 포함 → 나머지 정상 레코드 반환
- lock_timeout 0.001초 → Timeout 예외 발생
```

### 유닛 테스트 (tests/test_monitoring/test_alert_rules.py 확장)

```
- evaluate_rule_ml_disagree(0.10, ...) → []
- evaluate_rule_ml_disagree(0.20, ...) → WARNING
- evaluate_rule_ml_disagree(0.35, ...) → CRITICAL
- evaluate_all() → RULE_ML_DISAGREE 포함 경로 검증
```

### 통합 테스트 (tests/test_integration/test_monitoring_pipeline.py)

```
- predict() → metrics_live.jsonl 기록 검증 (임시 파일 사용)
- predict() MetricsWriter 예외 → 정상 응답 반환 검증
- GET /metrics X-Admin-Key 올바름 → 200 + records
- GET /metrics X-Admin-Key 없음 → 403
- pipeline.py _save_drift_reference() → drift_reference.pkl 생성 + 로드 가능
- DAG _detect_drift() → drift_{partition}.json 생성
- DAG _generate_alerts() → alerts_{partition}.json 생성
```

---

## 13. 알려진 제약사항 및 향후 검토

| 항목 | 내용 |
|------|------|
| PSI 피처 제한 | 현재 drug_count, ddi_count, rule_triggered만 가능 — 완전한 feature vector는 ETL 고도화 단계에서 |
| Prometheus 미사용 | 로컬 개발 환경 기준 — 운영 배포 시 metrics.py 연결 별도 검토 |
| metrics_live.jsonl 용량 | 운영 트래픽에서 파일 truncation 정책 미포함 — 향후 rotation 추가 필요 |
| 성능 모니터링 | Ground truth 부재로 Recall/Precision 제외, Rule/ML 불일치율을 proxy로 사용 |
