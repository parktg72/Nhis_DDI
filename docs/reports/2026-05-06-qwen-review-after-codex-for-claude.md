# Qwen Review Recheck After Codex Fixes - For Claude

작성일: 2026-05-06
대상: MODE_11_hana
목적: qwen3.6 의견이 Codex 수정 이후에도 반영 가치가 있는지 재검토한 결과 전달

## Context

Codex 수정 후 확인된 변경:

- `serving/predictor.py`
  - 온라인 `RequestFeatureBuilder`의 `dup_atc5/dup_atc4/dup_atc3` 계산을 ETL 계약과 맞춤.
- `serving/routers/health.py`, `serving/schemas.py`
  - 계층 모델만 로드된 경우에도 `/health.model_loaded=True`, `/model/info.model_type="hierarchical"`가 되도록 수정.
- `tests/test_serving/test_feature_contract.py`, `tests/test_dags/test_ddi_train_dag.py`
  - `/tmp/codex-review-fixes` 임시 import 제거.
- `hana_app/requirements.txt`, `packages_win/requirements.txt`, BAT 설치 스크립트
  - Python 3.12 constraints 적용 경로 추가.
- 새 테스트:
  - `tests/test_serving/test_dup_atc_contract.py`
  - `tests/test_serving/test_health_hierarchical.py`

검증:

```bash
.venv/bin/python -m pytest \
  tests/test_serving/test_dup_atc_contract.py \
  tests/test_serving/test_health_hierarchical.py \
  tests/test_serving/test_feature_contract.py \
  tests/test_dags/test_ddi_train_dag.py -q
```

결과:

```text
10 passed in 1.52s
```

## Verdict On Qwen3.6 Items

### ISSUE-1: 모든 `sys.path.insert(0, ...)` 제거

판정: 방향은 맞지만 `P0 즉시 전체 제거`는 과함.

현재 상태:

- `/tmp/codex-review-fixes` 임시 import는 제거됨.
- 하지만 Streamlit page, HANA core, Airflow DAG, serving predictor, 테스트에는 여전히 `sys.path.insert`가 다수 남아 있음.

권장:

- 임시 경로 삽입 제거는 완료된 것으로 봐도 됨.
- 전체 제거는 `pyproject.toml`/editable install/package화 작업으로 별도 P1/P2 마이그레이션 권장.
- 한 번에 제거하면 Streamlit pages와 Airflow `/app` 실행 경로가 깨질 수 있음.

### ISSUE-2: `ml_runner.py` 분리

판정: 맞지만 결함 수정 우선순위는 낮음.

현재 상태:

- `hana_app/core/ml_runner.py`는 약 2,119라인.
- 유지보수성 리스크는 명확함.

권장:

- P1보다는 P2가 적절.
- 먼저 기능 계약/테스트 안정화 후 샘플링, feature build, 학습 실행, artifact 저장, 유틸을 분리하는 것이 안전.

### ISSUE-3: 위험 약물 상수 단일화

판정: 맞고 아직 반영 필요.

현재 상태:

- `serving/predictor.py`
  - `_HIGH_RISK_KEYWORDS`
  - `_RENAL_RISK_KEYWORDS`
  - `_HEPATIC_RISK_KEYWORDS`
- `scripts/etl/prescription_aggregator.py`
  - 동일/유사 상수 존재.
- `rules/safety_net.py`
  - `_has_high_risk_drug()` 내부에 별도 high-risk keyword list 존재.

영향:

- ETL, serving feature, rule safety net 간 Red 판정/feature 값 drift 가능.

권장:

- 공용 모듈 예: `scripts/etl/risk_drug_constants.py` 또는 `rules/risk_drug_constants.py`.
- ETL/serving/rules가 같은 상수를 import하게 변경.
- 상수 동등성 테스트 추가.

### ISSUE-4: Magic numbers 통합 config module화

판정: 일부 맞음. 무차별 중앙화는 비권장.

현재 상태:

- 운영 튜닝값 일부는 `config/settings.py`, `scripts/train/hyperparams.py`, `hana_app/core/config.py`에 있음.
- 여전히 하드코딩된 값 존재:
  - serving ML band: threshold * 0.6, threshold * 0.3
  - `ml_prob > 0.3` reason 노출 조건
  - `long_term_drug_count >= 30`
  - batch limit 1000
  - HANA/date/window 관련 일부 값

권장:

- 임상 기준값, 운영 튜닝값, 테스트 fixture 값을 구분.
- 운영에서 실제 조정 가능한 값부터 외부화.
- 임상 기준값은 `CLINICAL_STANDARDS_v1.0` 또는 명확한 clinical constants 모듈로 관리.

### ISSUE-6: DDI matrix matching regex special char escaping

판정: 부분적으로 맞고 아직 수정 필요.

현재 상태:

- `rules/safety_net.py:_apply_matrix_ddi()`는 이미 `regex=False`라 안전.
- 그러나 `rules/safety_net.py:get_ddi_severity()`는 아직 아래처럼 기본 regex=True:

```python
self._ddi_matrix["_a_lower"].str.contains(a_lower, na=False)
self._ddi_matrix["_b_lower"].str.contains(b_lower, na=False)
```

영향:

- 약물명에 `+`, `(`, `)`, `[`, `]`, `.`, `*` 등이 포함되면 regex로 해석되어 오탐/누락 가능.

권장:

- `get_ddi_severity()`의 모든 `str.contains()`에 `regex=False` 추가.
- 특수문자 약물명 테스트 추가.

### ISSUE-7: HANA DATE type 명시 CAST

판정: 맞고 아직 반영 필요. 단, 타입 확인 후 적용해야 함.

현재 상태:

- `hana_app/core/hana_etl.py`의 date query는 여전히:

```sql
WHERE "{start_date_col}" BETWEEN ? AND ?
```

대상 위치:

- `fetch_t20_by_date`
- `fetch_t30_by_date`
- `fetch_t60_by_date`
- `fetch_t40_by_date`
- `find_patients_by_drug_codes`의 date branch

주의:

- 프로젝트 지침상 HANA 스키마/컬럼 타입을 임의 추측하면 안 됨.
- 컬럼이 `DATE`, `YYYYMMDD` string, numeric 중 무엇인지 환경별 확인 필요.

권장:

- table metadata에서 `DATA_TYPE_NAME` 확인 후 date predicate builder를 분기.
- 또는 config에 date column representation을 명시.
- SQL builder 단위 테스트 추가.

### ISSUE-9: batch endpoint + ETL SQL builder 테스트 추가

판정: 맞음.

현재 상태:

- batch endpoint smoke test는 있음.
- 하지만 부족한 케이스:
  - batch max 1000 boundary
  - 1001건 validation failure
  - 부분 실패 시 `warnings`/`total`/분포 카운트 계약
  - metrics writer 실패가 batch 응답을 깨지 않는지
  - HANA SQL builder의 date predicate, pid batching, placeholders 계약

권장:

- P2로 추가.
- 특히 HANA SQL은 실제 DB 없이 `conn.query_df` mock으로 SQL 문자열과 params를 검증 가능.

### ISSUE-10: 운영 하이퍼파라미터 외부화

판정: 맞음. 단, 선별 적용 권장.

현재 상태:

- 일부 학습 값은 `config/settings.py`와 `scripts/train/hyperparams.py`에 있음.
- serving/계층 dispatch/메트릭 reason 조건 등에는 하드코딩이 남아 있음.

권장:

- 운영 중 바꿀 가능성이 있는 값부터 외부화:
  - serving classification band
  - reason probability display threshold
  - batch max size
  - long-term prescription day threshold
  - HANA query batch size는 기존 공유 상수 변경 금지 원칙 준수 필요.

## Additional Finding For Claude

### constraints-py312.txt is ignored by Git

중요도: HIGH

현재 상태:

- `constraints-py312.txt` 파일은 로컬에 존재함.
- 하지만 `.gitignore:58`의 `*.txt` 규칙에 걸려 Git에 추적되지 않음.
- `git status --short --untracked-files=all`에도 `constraints-py312.txt`가 보이지 않음.

확인:

```bash
git check-ignore -v constraints-py312.txt
```

결과:

```text
.gitignore:58:*.txt constraints-py312.txt
```

영향:

- `hana_app/requirements.txt`, `packages_win/requirements.txt`, `install_312.bat`, `install_all.bat`, `packages_win/install.bat`가 모두 `constraints-py312.txt`를 참조함.
- 이 파일이 커밋에 포함되지 않으면 다른 환경/Windows 폐쇄망 설치에서 constraints 파일 없음으로 설치가 실패하거나 dev/prod parity가 깨짐.

권장:

- `.gitignore`에 예외 추가:

```gitignore
!constraints-py312.txt
```

- 또는 파일명을 `.txt`가 아닌 추적 가능한 이름으로 변경.
- 추가로 현재 constraints는 `xgboost==3.2.0`을 핀하고 있는데, 이전 전체 pytest 세그폴트가 Python 3.12 + XGBoost 3.2.0 환경에서 발생했으므로 full-suite 재검증 필요.

## Recommended Next Order

1. `constraints-py312.txt`가 Git에 포함되도록 `.gitignore` 예외 추가 또는 파일명 변경.
2. `rules/safety_net.py:get_ddi_severity()`에 `regex=False` 추가 + 특수문자 테스트.
3. 위험 약물 상수 단일화 + ETL/serving/rules 동등성 테스트.
4. HANA date predicate builder 도입. 실제 컬럼 타입 확인 또는 config 기반 분기.
5. batch endpoint boundary/partial failure/metrics 테스트와 HANA SQL builder 테스트 추가.
6. `sys.path.insert` 전체 제거는 package화 계획으로 별도 진행.
7. `ml_runner.py` 분리는 기능 안정화 후 P2로 진행.
