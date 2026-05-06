# Codex Review For Claude - 2026-05-06

MODE_11_hana 프로젝트 리뷰 결과입니다. 후속 수정 우선순위 판단에 사용하세요.

## Findings

### HIGH - 온라인 서빙 ATC 중복 피처가 배치 ETL 계약과 다름

- 위치: `serving/predictor.py:641-648`
- 비교 기준: `scripts/etl/prescription_aggregator.py:325-343`
- 문제:
  - 배치 ETL 계약은 `dup_atc5 = 전체 ATC 7자리`, `dup_atc4 = 5자리 prefix`, `dup_atc3 = 4자리 prefix`입니다.
  - 온라인 `RequestFeatureBuilder`는 `dup_atc5`에 5자리 prefix, `dup_atc4`에 4자리 prefix, `dup_atc3`에 3자리 prefix를 넣고 있습니다.
- 영향:
  - 동일 `A10BA02`/`A10BA03` 처방은 ETL에서는 `dup_atc5=0`, `dup_atc4=1`이어야 하지만 온라인 서빙에서는 `dup_atc5=1`이 됩니다.
  - 학습-서빙 feature drift로 단일 ML 확률과 계층 Stage 2 라벨이 왜곡될 수 있습니다.
- 제안:
  - `RequestFeatureBuilder`의 `dup_atc5/4/3` 계산을 ETL과 동일하게 수정하세요.
  - 온라인 feature contract 테스트에 ATC 5/4/3 레벨별 회귀 케이스를 추가하세요.

### MEDIUM - 계층 모델 모드에서 상태 API가 모델 미로드처럼 보임

- 위치: `serving/routers/health.py:54-58`, `serving/routers/health.py:74-78`
- 문제:
  - `/health`의 `model_loaded`가 `pred._ml.loaded`만 봅니다.
  - `/model/info`도 단일 ML 모델 정보만 반환합니다.
  - `HIERARCHICAL_MODEL_DIR`로 `HierarchicalPredictor`가 정상 로드되고 `_ml`이 비어 있는 구성에서는 `/predict`는 동작해도 `/health`는 `model_loaded=false`, `/model/info`는 `model_type=none`으로 보입니다.
- 영향:
  - 운영 모니터링, 배포 검증, readiness 판단이 정상 계층 모델을 미로드 상태로 오판할 수 있습니다.
- 제안:
  - 최소 수정은 `model_loaded = pred._ml.loaded or (pred._hierarchical is not None and pred._hierarchical.loaded)`입니다.
  - 더 나은 수정은 응답에 `model_mode`, `hierarchical_loaded`, `feature_count`를 노출하는 것입니다.

### MEDIUM - 의존성 상한/잠금 부족으로 검증 환경이 불안정함

- 위치: `hana_app/requirements.txt:13-24`, `packages_win/requirements.txt:10-52`
- 문제:
  - 핵심 네이티브 의존성인 `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `fastapi/starlette` 계열이 하한 중심입니다.
  - 현재 `.venv`는 Python 3.12.13, NumPy 2.4.4, XGBoost 3.2.0, scikit-learn 1.8.0, FastAPI 0.136.1, Starlette 1.0.0입니다.
  - `.venv/bin/python -m pytest` 전체 실행은 `tests/test_hana_app/test_hierarchical_cv.py` 진입 중 XGBoost 네이티브 세그폴트로 중단됐습니다.
  - 같은 파일 단독 실행은 통과했으므로 full-suite 순서, 리소스, 네이티브 라이브러리 조합에 민감한 상태입니다.
- 영향:
  - CI/운영 PC/개발 PC에서 테스트 결과가 달라질 수 있습니다.
  - 세그폴트는 Python 예외가 아니므로 테스트 프로세스 전체가 죽습니다.
- 제안:
  - Python 3.12용 lock 또는 constraints 파일을 추가하세요.
  - XGBoost/NumPy/scikit-learn 조합은 실제 Windows 폐쇄망 wheel 세트와 동일하게 고정하세요.
  - full-suite에서 계층 CV 테스트가 반복 통과하는지 확인하세요.

### LOW - 테스트가 임시 경로의 과거 소스를 import할 수 있음

- 위치: `tests/test_serving/test_feature_contract.py:51-53`
- 문제:
  - `/tmp/codex-review-fixes`를 `sys.path` 앞에 삽입합니다.
  - 해당 경로가 남아 있으면 현재 워크트리 코드가 아니라 과거 복사본을 import할 수 있습니다.
- 영향:
  - feature contract 테스트가 실제 소스와 분리되어 거짓 통과할 수 있습니다.
- 제안:
  - 임시 경로 삽입을 제거하고 현재 repo import만 사용하세요.

## Verification

- `python --version` => `3.11.13`
- `python -m pytest` => `692 passed, 18 skipped, 10 failed, 19 errors`
  - 프로젝트 지침과 맞지 않는 Python 3.11 환경입니다.
  - FastAPI/Starlette 불일치와 NumPy API 차이로 대량 실패가 발생했습니다.
- `.venv/bin/python --version` => `3.12.13`
- `.venv/bin/python -m pytest`
  - `tests/test_hana_app/test_hierarchical_cv.py` 부근에서 XGBoost 세그폴트로 중단됐습니다.
- `.venv/bin/python -m pytest tests/test_hana_app/test_hierarchical_cv.py -q`
  - `9 passed in 192.85s`
- `.venv/bin/python -m pytest tests/test_serving/test_serving.py tests/test_serving/test_hierarchical_serving.py tests/test_serving/test_predictor.py -q`
  - `61 passed in 42.03s`

## Suggested Order

1. `RequestFeatureBuilder`의 `dup_atc5/4/3` 계산을 ETL 계약과 맞추고 회귀 테스트를 추가합니다.
2. `/health`와 `/model/info`가 단일 ML/계층 모델 양쪽 상태를 반영하도록 고칩니다.
3. Python 3.12용 constraints/lock을 만들고 full-suite 세그폴트를 재현 및 제거합니다.
4. `/tmp/codex-review-fixes` `sys.path` 삽입을 제거합니다.
