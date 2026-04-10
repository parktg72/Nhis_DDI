# Page 4 (결과 분석) + Page 6 (모니터링) 개선 설계

작성일: 2026-04-10  
상태: 승인됨

---

## 1. 배경 및 목표

### 발견된 문제

**Page 4 — 결과 분석**

| 코드 | 위치 | 심각도 | 설명 |
|------|------|--------|------|
| P4-A | L91-92 | Low | Dead code — `fi_data`를 같은 key로 재할당 |
| P4-B | L223 | Medium | 세션 의존성 버그 — `features_df`가 세션에만 존재, 저장된 결과 로드 시 탭4 항상 빈 화면 |
| P4-C | 탭 구성 | Medium | ROC Curve 미구현 — docstring에 ROC 명시되나 차트 없음 |
| P4-D | 상단 | Low | page_guards 미적용 |
| P4-E | _save_result | Medium | 위험도 분포 요약이 JSON에 저장되지 않아 복원 불가 |

**Page 6 — 모니터링**

| 코드 | 위치 | 심각도 | 설명 |
|------|------|--------|------|
| P6-A | import | High | Docker 경로(`/app/data/monitoring`) — Windows에 존재하지 않음 |
| P6-B | import | High | Docker 경로(`/app/models/current/...`) — 동일 |
| P6-C | Tab4 | Medium | API 서버 health 호출 — 데스크탑 앱엔 API 서버 없음 |
| P6-D | 상단 | Low | page_guards 미적용 |
| P6-E | Tab4 | Medium | HANA 연결 상태 미확인 |
| P6-F | Tab4 | Low | 모델/결과 파일 현황 미표시 |

### 목표

1. Page 4 무결성 수정 (P4-A~E)
2. Page 6을 Docker/API 환경 기반에서 **HANA 데스크탑 앱 맥락**으로 완전 재정의
3. 사용자가 탭을 열지 않아도 전체 상태를 즉시 파악할 수 있도록 UX 개선

---

## 2. Page 4 수정 설계

### 2-1. P4-A: Dead code 제거

```python
# Before
fi_data = result.get("feature_importance")
if fi_data is None and source == "저장된 결과":
    fi_data = result.get("feature_importance")  # no-op

# After
fi_data = result.get("feature_importance")
```

### 2-2. P4-B + P4-E: features_df 저장/복원

`_save_result()`에서 features_df를 포함하는 대신, result dict에 **위험도 분포 요약**을 저장한다.
(features_df 전체를 JSON에 저장하면 수십만 행이 될 수 있으므로 요약만 저장)

저장 형태 (ml_runner.py `_save_result` 수정):
```python
# result dict에 risk_summary 추가
if "features_df" in result and result["features_df"] is not None:
    df = result["features_df"]
    meta["risk_summary"] = df["risk_level"].value_counts().to_dict()
    meta["drug_count_stats"] = {
        "mean": float(df["drug_count"].mean()),
        "max": int(df["drug_count"].max()),
        "hist": df["drug_count"].clip(upper=30).value_counts().sort_index().to_dict(),
    }
    meta["ddi_means"] = {
        c: float(df[c].mean())
        for c in ["ddi_contraindicated","ddi_major","ddi_moderate","ddi_minor"]
        if c in df.columns
    }
```

Page 4 탭4에서 복원:
```python
# session_state에 features_df 있으면 원본 사용
# 없으면 result["risk_summary"] 등 요약으로 차트 재구성
```

### 2-3. P4-C: ROC Curve 탭 추가

- 탭 목록에 "📉 ROC Curve" 추가
- `metrics["roc_curve"]` 키 존재 시 FPR/TPR 선 차트 표시
- 없으면 "ROC Curve 데이터가 없습니다 (이번 개선 이전에 저장된 결과에는 roc_curve가 포함되지 않습니다)" 안내
- ml_runner에서 `roc_curve` 데이터를 result에 추가 저장 (binary 분류 한정, multiclass는 생략)

### 2-4. P4-D: page_guards 적용

```python
from hana_app.core.config import load_config
from hana_app.core.page_guards import check_hana_validated, get_validation_error

cfg = load_config()
if not check_hana_validated(cfg):
    st.warning(get_validation_error(cfg))
    st.stop()
```

단, 결과 분석 페이지는 저장된 결과를 보는 용도이므로 **SAS 모드에선 guard 생략** (is_hana 체크).

---

## 3. Page 6 재설계

### 3-1. 전체 구조

```
┌─ 상태 요약 바 (항상 표시) ─────────────────────────────────────┐
│  🟢 HANA 연결됨  │  🟢 ETL 정상 (2026-04-09)  │  🟡 모델 없음  │  🟢 저장소 정상  │
└────────────────────────────────────────────────────────────────┘

[Tab 1: 🔌 HANA 연결 상태] [Tab 2: 📋 ETL 실행 이력] [Tab 3: 🤖 모델 학습 이력] [Tab 4: 💾 시스템 상태]
```

### 3-2. 상태 요약 바

페이지 최상단에 4개 `st.metric` 또는 컬러 뱃지로 전체 상태 요약 표시.  
각 상태는 독립적으로 계산되며 탭을 열지 않아도 파악 가능.

| 항목 | 🟢 정상 조건 | 🟡 경고 | 🔴 오류 |
|------|------------|--------|--------|
| HANA 연결 | `is_connected()` = True | validated 만료 | 연결 끊김 |
| ETL 이력 | session_state에 ETL 기록 있음 | — | 기록 없음 (앱 재시작 시 소실됨을 안내) |
| 모델 상태 | results/ 파일 ≥ 1개 | 최신 결과 7일 이상 | 파일 없음 |
| 저장소 | results/ + models/ 접근 가능 | — | 경로 없음 |

### 3-3. Tab 1: HANA 연결 상태

- 현재 연결 호스트·포트·사용자 표시
- `validated` 상태 + `validated_at` + `validated_host`
- **재연결 버튼**: 클릭 시 `conn.ensure_connected(hana_creds, session_state)` 호출
- 실시간 `is_connected()` 결과 표시
- TTL 캐시 잔여 시간 표시 (선택)

### 3-4. Tab 2: ETL 실행 이력

- **영속 로그 파일**: `hana_app/etl_log.jsonl` — 앱 재시작 후에도 이력 유지
- 레코드 구조: `{"ts": "2026-04-10T09:00:00", "period_from": "2023-01", "period_to": "2023-12", "row_count": 123456, "elapsed_sec": 42.1, "status": "ok"|"error", "error": ""}`
- Page 3 ETL 완료 시 `etl_log.jsonl`에 JSON 라인 append
- 표시: 최근 50건, 시각/기간/건수/소요시간/상태 컬럼
- 세션에 기록 없어도 파일에서 이력 로드하므로 재시작 후에도 표시됨

### 3-5. Tab 3: 모델 학습 이력

- `list_saved_results()` 기반 — `hana_app/results/result_*.json`
- 결과별: 시각, 모델명, Accuracy, F1, AUC, 학습 건수
- 최신 결과 강조 (첫 행 bold 또는 배경색)
- 결과 파일 삭제 버튼 (확인 다이얼로그)
- 성능 지표 추이 라인 차트 (날짜 x축)

### 3-6. Tab 4: 시스템 상태

- `hana_app/models/` 파일 목록 + 크기
- `hana_app/results/` 파일 목록 + 크기
- 총 디스크 사용량 (models + results 합산)
- config 파일(`hana_config.json`) 존재 여부

### 3-7. 제거 항목 (Docker/API 의존)

- `config.settings` import 전체 제거 → 로컬 경로 직접 사용
- `SERVING_URL /health` 호출 제거
- `METRICS_JSONL_PATH`, `DRIFT_REFERENCE_PATH`, `MONITORING_DIR` 제거
- `_monitoring_helpers` import 제거 (탭1-4 모두 새 구현)

---

## 4. ml_runner.py 수정 사항

- `_save_result()`: `risk_summary`, `drug_count_stats`, `ddi_means` 저장 추가
- `run_training()` 반환 result에 `roc_curve` 데이터 포함 (binary 한정)
- ETL 완료 후 `hana_app/etl_log.jsonl`에 JSON 라인 append (Page 3 연동, 영속)

---

## 5. 파일 변경 범위

| 파일 | 변경 유형 |
|------|---------|
| `hana_app/pages/4_📊_결과_분석.py` | 수정 — P4-A~E 픽스, ROC탭 추가, page_guards |
| `hana_app/pages/6_📊_모니터링.py` | 전면 재작성 — 상태바 + 4탭 신규 구현 |
| `hana_app/core/ml_runner.py` | 수정 — risk_summary/roc_curve 저장, etl_log 기록 |
| `hana_app/pages/_monitoring_helpers.py` | 유지 (Page 6에서는 미사용, 향후 참조용) |
| `hana_app/core/etl_logger.py` | 신규 — ETL 로그 append/read 헬퍼 |
| `tests/test_hana_app/test_etl_logger.py` | 신규 — etl_logger 단위 테스트 |
| `tests/test_hana_app/test_page6_status.py` | 신규 — 상태 요약 로직 단위 테스트 |

---

## 6. 구현 순서

1. `hana_app/core/etl_logger.py` 신규 작성 + 테스트
2. `hana_app/core/ml_runner.py` 수정 — risk_summary, roc_curve 저장 + etl_log append
3. `hana_app/pages/4_📊_결과_분석.py` 버그 수정 (P4-A~E + ROC탭)
4. `hana_app/pages/6_📊_모니터링.py` 전면 재작성 — 상태바 + 4탭
5. 테스트 작성 (test_etl_logger, test_page6_status)
