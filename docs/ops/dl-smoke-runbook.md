# DL Smoke 검증 운영 절차서

**작성**: 2026-05-18  
**대상**: 폐쇄망 Windows 운영자 / serving 운영팀  
**범위**: CUDA 12.6 DL 환경 확인, smoke DL bundle hot-swap, 실제 DL bundle 배포 전 검증  
**주의**: 본 절차는 smoke 검증 기준이다. 실제 HANA DB 접속과 실제 학습 모델 운영 투입은 별도 승인 후 진행한다.

---

## 1. 전제 조건

| 항목 | 기준 |
|---|---|
| Python | Windows Python 3.12 |
| 가상환경 | `.venv_hana\` |
| GPU | NVIDIA GeForce GTX 1080 Ti 또는 CUDA 12.6 wheel set 호환 GPU |
| DL wheel | `torch==2.11.0+cu126`, PyG companion packages |
| 관리자 키 | `ADMIN_API_KEY` 환경변수 또는 배치 두 번째 인자 |
| 기본 모델 경로 | `<project>\models` |

설치/복구는 `install_312.bat`를 기준으로 한다. CUDA DL package 검증을 hard fail로 강제하려면 설치 전에 다음 환경변수를 설정한다.

```bat
set DDI_REQUIRE_CUDA_DL=1
install_312.bat venv
```

---

## 2. 1회성 CUDA 환경 확인

폐쇄망 PC에서 다음 명령이 성공해야 한다.

```bat
.venv_hana\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

기대 예:

```text
2.11.0+cu126 12.6 True NVIDIA GeForce GTX 1080 Ti
```

PyG companion package도 확인한다.

```bat
.venv_hana\Scripts\python.exe -c "import torch_geometric, pyg_lib, torch_scatter, torch_sparse, torch_cluster; print('PyG companion OK')"
```

---

## 3. Smoke API E2E 검증

### 3.1 서버 실행

첫 번째 콘솔에서 실행한다.

```bat
set ADMIN_API_KEY=<운영자_키>
run_api_smoke_dl.bat %ADMIN_API_KEY% 8765
```

확인 포인트:

- 서버가 현재 콘솔에서 blocking 방식으로 실행된다.
- `MODEL_DIR` 기본값은 `<project>\models`이다.
- `DDI_SMOKE_HISTORY_PROVIDER=1`이 설정된다.
- smoke 전용 provider 활성화 WARNING 로그가 보여야 한다.

### 3.2 reload/predict 검증

두 번째 콘솔에서 실행한다. `ADMIN_API_KEY`는 새 콘솔에서도 별도 설정이 필요하다.

```bat
set ADMIN_API_KEY=<운영자_키>
verify_smoke_dl.bat http://127.0.0.1:8765 %ADMIN_API_KEY% --require-dl-prediction
```

내부 순서:

```text
create smoke bundle -> validate bundle -> /admin/reload/dl -> /predict
```

성공 기준:

- `/admin/reload/dl` 응답 `status=ok`
- `dl_loaded=true`
- `/predict` 응답에 `dl_prediction` 존재
- smoke 기준 `dl_prediction.predicted_label=high`
- `dl_error=null`

검증 종료 후 첫 번째 콘솔에서 `Ctrl+C`로 서버를 종료한다.

---

## 4. Smoke bundle 단독 생성/검증

API 서버 없이 bundle 파일만 생성하려면 다음을 실행한다.

```bat
generate_smoke_dl_bundle.bat
```

기본 출력 경로:

```text
models\dl\smoke
```

다른 출력 경로가 필요하면 첫 번째 인자로 넘긴다.

```bat
generate_smoke_dl_bundle.bat C:\tmp\hana_smoke_bundle
```

bundle semantic validation만 실행하려면 다음을 사용한다.

```bat
.venv_hana\Scripts\python.exe -m scripts.ops.validate_dl_bundle models\dl\smoke
```

기대 출력:

```text
bundle_dir=models\dl\smoke
status=ok
```

---

## 5. 실제 DL bundle 교체 전 절차

학습된 HANA DL 산출물이 준비되면 smoke bundle과 분리된 버전 디렉터리에 복사한다.

```text
models\dl\<version>\
  MANIFEST.json
  model.pt
  model_config.json
  drug_vocab.json
  edge_index.pt
  feature_normalizer.pkl
  schema_version.json
```

서버에 reload 하기 전에 반드시 로컬 validation을 먼저 통과시킨다.

```bat
.venv_hana\Scripts\python.exe -m scripts.ops.validate_dl_bundle models\dl\<version> --check-model
```

성공 기준:

```text
status=ok
```

`--check-model`은 `torch.jit.load(model.pt)`를 CPU에서 시도한다. 이 단계는 HANA DB나 실제 환자 데이터에 접속하지 않는다.

---

## 6. 하루 parquet 샘플 provider 확인

`records_20241001.parquet`처럼 하루 단위로 추출한 parquet 샘플은 실제 HANA DB 접속 전 provider 계약 확인에만 사용한다.

확인 범위:

- `patient_id`, `edi_code`, `start_date` 컬럼 계약
- `edi_code -> drug_code` 정규화
- `start_date -> prescription_date(YYYYMMDD)` 정규화
- `(patient_id, drug_code, prescription_date)` 기준 중복 제거
- 환자별 history frame이 DL predictor로 전달되는지 확인

제외 범위:

- 실제 T30/T60 날짜 window 정확도
- 여러 기준일에 걸친 이력 변화
- HANA DB 조회 성능
- 운영 서버 자동 provider wiring

코드 진실은 `scripts/ops/parquet_history_provider.py`다. 이 provider는 ops 검증 도구이며 운영 서버 wiring에는 사용하지 않는다.

샘플 파일 구조와 provider 출력 계약을 PII 출력 없이 확인하려면 다음 배치를 사용한다.

```bat
inspect_parquet_history.bat C:\Users\ptg\Downloads\records_20241001.parquet
```

특정 환자를 내부 샘플로 조회하려면 `--patient-id`를 넘긴다. 이 값은 조회에만 사용되며 출력에는 표시하지 않는다.

```bat
inspect_parquet_history.bat C:\Users\ptg\Downloads\records_20241001.parquet <환자ID>
```

`output key duplicates` 항목은 `ParquetHistoryProvider`가 최종 제거할 행 수다. 출력 key `(patient_id, drug_code, prescription_date)` 기준이며, 원본 parquet의 `end_date`와 `source` 차이는 무시한다. 이 때문에 원본 key 중복 수보다 높게 나올 수 있다.

---

## 7. 운영 서버 적용 시 주의

Smoke 서버 배치는 운영 서버용이 아니다.

| 항목 | 운영 기준 |
|---|---|
| `DDI_SMOKE_HISTORY_PROVIDER=1` | smoke 전용. 운영 서버에서는 사용 금지 |
| `run_api_smoke_dl.bat` | smoke E2E 전용. 운영 기동 스크립트와 분리 |
| `--skip-validation` | 이미 `validate_dl_bundle --check-model`을 별도로 통과한 경우에만 사용 |
| `MODEL_DIR` | `/admin/reload/dl` bundle path boundary와 일치해야 함 |
| `ADMIN_API_KEY` | `/admin/reload/dl` 호출에 필수 |
| CUDA wheel | GTX 1080 Ti는 `cu128` 금지, `cu126` 고정 |

운영 서버가 이미 실행 중이고 실제 bundle validation을 별도 통과했다면, 예외적으로 다음처럼 validation을 건너뛸 수 있다.

```bat
verify_smoke_dl.bat http://127.0.0.1:8000 %ADMIN_API_KEY% --require-dl-prediction --skip-validation
```

단, `--skip-validation`은 smoke 검증 단축용 예외 플래그다. 최초 배포 전 검증을 대체하지 않는다.

---

## 8. HANA DB 3개월 자료 확보 gate

현재 smoke/ops tooling은 완료 상태로 본다. 다음 코드 작업은 폐쇄망 운영 PC에서 HANA DB에 실제로 연결해 약 3개월 처방 자료를 확보한 뒤 진행한다.

확보 전에는 실 학습 파이프라인, 운영 HANA provider 자동 wiring, DL 결과의 최종 `risk_level` 반영 코드를 추가하지 않는다. 임시 parquet provider와 smoke provider는 검증 도구일 뿐 운영 계약이 아니다.

접속 기준:

```text
HANA host: 10.1.67.115
HANA port: 30015
ID/PW: 운영자가 실행 시 별도 입력. 저장소와 문서에 기록하지 않음.
```

3개월 자료 확보 시 확인할 항목:

| 항목 | 확인 내용 |
|---|---|
| DB 접속 | `10.1.67.115:30015`, 사용자 입력 ID/PW, 인증 방식, 폐쇄망 접속 가능 여부 |
| 기간 | 기준일 기준 약 3개월 T30/T60 처방 이력 |
| 컬럼 | `patient_id`, EDI 코드, 처방 시작일/종료일, source 구분의 실제 컬럼명 |
| 샘플 | 운영 검증용 patient ID 최소 1건 이상 |
| 규모 | rows, unique patients, source counts, full/output-key duplicates |
| PII | 점검 출력에는 환자 ID 원문을 남기지 않음 |

자료 확보 후 진행 순서:

1. `inspect_parquet_history.bat` 또는 동등한 HANA dry-run으로 schema와 중복 규모를 확인한다.
2. 학습-서빙 계약을 고정한다: aggregate tabular, drug sequence multi-hot, graph/GAT 중 하나를 선택한다.
3. 선택한 계약에 맞춰 `drug_vocab`, `edge_index`, tensor schema, TorchScript export 포맷을 문서화한다.
4. Hermes가 고정된 계약 기준으로 학습 파이프라인과 TorchScript export를 구현한다.
5. Codex가 `validate_dl_bundle --check-model`, `/admin/reload/dl`, `/predict` smoke/E2E 회귀를 독립 검증한다.
6. 성능/임상 검토가 끝나기 전까지 DL 결과는 auxiliary 필드로만 유지한다.

---

## 9. 실패 시 triage

| 증상 | 우선 확인 |
|---|---|
| `ADMIN_API_KEY is required` | 환경변수 또는 배치 두 번째 인자 확인 |
| `bundle validation failed` | `model_config.json`, `drug_vocab.json`, `MANIFEST.json` hash/index/schema 확인 |
| `HTTP request failed ... connection refused` | API 서버 실행 여부, 포트 번호 확인 |
| `/admin/reload/dl` 400 | bundle 경로가 `MODEL_DIR` 밖인지, lookback/schema/hash 불일치인지 확인 |
| `/predict`에 `dl_prediction` 없음 | smoke 서버는 `DDI_SMOKE_HISTORY_PROVIDER=1`, 운영 서버는 실제 HANA history provider 연결 확인 |
| TorchScript load 실패 | `model.pt`가 TorchScript인지, 운영 torch/cu126 버전과 호환되는지 확인 |

---

## 10. 코드 진실 위치

| 항목 | 위치 |
|---|---|
| smoke bundle 생성 | `scripts/datasets/smoke_dl_bundle.py` |
| smoke bundle 생성 배치 | `generate_smoke_dl_bundle.bat` |
| API smoke 서버 배치 | `run_api_smoke_dl.bat` |
| reload/predict 검증 | `scripts/ops/verify_smoke_dl.py`, `verify_smoke_dl.bat` |
| bundle semantic validator | `scripts/ops/validate_dl_bundle.py` |
| parquet sample history provider | `scripts/ops/parquet_history_provider.py` |
| parquet sample inspector | `scripts/ops/inspect_parquet_history.py`, `inspect_parquet_history.bat` |
| DL serving runtime | `serving/dl_predictor.py` |
| smoke history provider | `scripts/ops/smoke_history_provider.py` |
| smoke provider opt-in wiring | `serving/main.py` |
| 회귀 테스트 | `tests/test_ops/test_verify_smoke_dl.py`, `tests/test_ops/test_validate_dl_bundle.py`, `tests/test_ops/test_parquet_history_provider.py`, `tests/test_ops/test_inspect_parquet_history.py`, `tests/test_datasets/test_smoke_dl_bundle.py`, `tests/test_serving/test_main_smoke_history_provider.py` |
