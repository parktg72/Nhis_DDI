# CUDA 12.6 DL 사전 준비 상태 및 다음 단계

작성일: 2026-05-18

## 결론

CUDA 12.8 호환성 검토는 종료되었다. 운영 GPU가 NVIDIA GeForce GTX 1080 Ti(Pascal, `sm_61`)이므로 PyTorch 2.11 `cu128` wheel은 부적합하고, 운영 DL 대상은 PyTorch 2.11 `cu126` wheel set으로 확정한다.

폐쇄망 Windows 환경에서 `torch==2.11.0+cu126`과 PyG companion package import 검증은 통과했다. 사용자 제공 운영 PC 검증 로그 기준으로 `torch.cuda.is_available()`가 `True`이고 GPU는 NVIDIA GeForce GTX 1080 Ti로 확인되었다.

## 최종 저장된 작업

- `discusiton.md`에 CUDA Packaging and F Slice, DL Inference Unit Slice, DL `/predict` Auxiliary Integration Slice가 기록되어 있다.
- `packages_win/requirements_cuda_cu126.txt`에 Windows x64 / Python 3.12 / CUDA 12.6 운영 DL wheel set이 고정되어 있다.
- `packages_win/download_cuda_cu126.bat`에 인터넷 가능 Windows PC에서 cu126 wheel set을 내려받아 `packages_win/py312`에 모으는 절차가 저장되어 있다.
- `install_312.bat`는 `requirements_cuda_cu126.txt`가 있으면 CUDA DL package 설치를 시도하고, `DDI_REQUIRE_CUDA_DL=1`이면 CUDA/PyG 검증 실패 시 hard fail 하도록 갱신되어 있다.
- `serving.dl_predictor.DLModel`은 import-time torch-free 구조를 유지하며, bundle manifest/hash/lookback validation 후 첫 `predict()`에서 torch runtime artifact를 lazy load한다. `model_config.json`의 선택 필드 `architecture`로 forward 계약을 구분한다. 기본/`linear`는 `model(features)`, `gat`/`gcn`은 `model(features, edge_index)`를 호출한다.
- `HybridPredictor`는 DL bundle hot-swap, DL history provider, `/predict` auxiliary DL result attachment를 지원한다.
- `/predict`의 최종 `risk_level`은 아직 Rule/ML 결과만 사용한다. DL 결과는 `dl_prediction` 또는 `dl_error` 보조 필드로만 붙는다.
- `scripts/datasets/smoke_dl_bundle.py`는 학습된 HANA 모델 확인 전 단계에서 사용할 TorchScript smoke DL bundle을 생성한다. 생성물은 `model.pt`, `model_config.json`, `drug_vocab.json`, `edge_index.pt`, `feature_normalizer.pkl`, `schema_version.json`, `MANIFEST.json`이며, smoke `model_config.json`에는 `architecture: "linear"`가 명시된다.
- `generate_smoke_dl_bundle.bat`는 폐쇄망 Windows PC에서 `.venv_hana`를 우선 사용해 기본 경로 `models\dl\smoke`에 smoke bundle을 생성한다. 다른 경로가 필요하면 첫 번째 인자로 출력 경로를 넘긴다.
- `scripts/ops/verify_smoke_dl.py`와 `verify_smoke_dl.bat`는 이미 실행 중인 API 서버를 대상으로 smoke bundle 생성, bundle semantic validation, `/admin/reload/dl`, `/predict` 호출을 순서대로 검증한다. `ADMIN_API_KEY`가 필요하며, `MODEL_DIR` 기본값은 프로젝트 `models`이다. 배치 파일의 세 번째/네 번째 인자로 `--require-dl-prediction`, `--skip-validation`을 넘길 수 있다.
- `run_api_smoke_dl.bat`는 현재 콘솔에서 blocking 방식으로 smoke API 서버를 실행한다. `.venv_hana`를 우선 감지하고 `PYTHONUTF8=1`, `DDI_SMOKE_HISTORY_PROVIDER=1`, 기본 `MODEL_DIR=<project>\models`를 설정한 뒤 `uvicorn serving.main:app --host 127.0.0.1 --port <PORT>`를 실행한다.
- `scripts/ops/smoke_history_provider.py`는 HANA 없이 smoke `/predict` DL 보조 추론 경로를 검증하기 위한 `SmokeHistoryProvider`를 제공한다. 반환 컬럼은 `patient_id`, `drug_code`, `prescription_date`이며 smoke vocab `D1`, `D2`를 사용한다.
- `serving/main.py`는 `DDI_SMOKE_HISTORY_PROVIDER=1|true|yes|on`일 때만 `SmokeHistoryProvider`를 `HybridPredictor`에 주입한다. 기본값은 OFF이며 활성화 시 서버 시작 로그에 WARNING을 남긴다.
- `scripts/ops/validate_dl_bundle.py`는 실제 학습 DL bundle 배포 전 semantic validation을 수행한다. manifest/hash 검증에 더해 `model_config.json`의 `architecture`, `input_dim`, `output_labels`, `drug_vocab.json` index 범위, `schema_version.json` sidecar 불일치 warning, 선택적 `--check-model` TorchScript CPU load를 확인한다.
- `docs/ops/dl-smoke-runbook.md`는 폐쇄망 Windows 운영자 기준 CUDA 확인, smoke API E2E, smoke bundle 단독 검증, 실제 DL bundle 교체 전 validation, 운영 주의사항, triage를 정리한 운영 절차서다.
- `scripts/ops/parquet_history_provider.py`는 하루 단위 parquet 샘플을 실제 HANA 접속 전 DL history provider 계약으로 읽는 ops 검증 도구다. 날짜 window 필터는 적용하지 않고, `edi_code -> drug_code`, `start_date -> prescription_date(YYYYMMDD)`로 정규화한 뒤 `(patient_id, drug_code, prescription_date)` 기준으로 중복 제거한다.
- `scripts/ops/inspect_parquet_history.py`는 하루 단위 parquet 샘플의 rows/cols, 필수 컬럼, source counts, 날짜 범위, unique patient 수, full/output-key duplicate 수, provider sample 결과를 환자 ID 출력 없이 점검하는 CLI다.
- `inspect_parquet_history.bat`는 폐쇄망 Windows 운영자용 parquet 점검 배치다. `.venv_hana`를 우선 사용하고, 첫 번째 인자 parquet path, 두 번째 인자 optional patient id를 받아 `scripts.ops.inspect_parquet_history`를 실행한다.

## 검증 근거

- 문서 기록:
  - `discusiton.md`: cu128 부적합, cu126 확정, 2026-05-15 폐쇄망 CUDA/PyG 검증 통과, DL inference/predict auxiliary slice 기록.
  - `docs/ops/dl-smoke-runbook.md`: 폐쇄망 Windows 운영자가 따라 할 smoke DL 검증 절차와 실제 bundle 교체 전 검증 절차 기록.
- 패키징 계약:
  - `packages_win/requirements_cuda_cu126.txt`: `torch==2.11.0+cu126`, `torch-geometric==2.7.0`, `pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster` 고정.
  - `packages_win/download_cuda_cu126.bat`: PyTorch cu126 index와 PyG `torch-2.11.0+cu126` wheel page 사용.
  - `tests/test_packaging/test_cuda_packaging.py`: CUDA requirement pin, downloader index, `install_312.bat` opt-in hard fail 검증.
- 운영 PC 검증:
  - `D:\claude\MODE_11_hana`: `torch 2.11.0+cu126`, `torch.version.cuda 12.6`, `torch.cuda.is_available() True`, `NVIDIA GeForce GTX 1080 Ti`.
  - `torch_geometric`, `pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster` import 검증 통과.
- 서빙/DL 계약:
  - `serving/dl_predictor.py`: lazy torch load, manifest 재검증, fixed-size drug vector encoding, TorchScript load, architecture 기반 forward dispatch.
  - `tests/test_serving/test_dl_predictor.py`: torch import 지연, bundle validation, hash mismatch, reload recovery, lazy runtime predict, `gat` architecture의 `edge_index` 전달 검증.
  - `tests/test_serving/test_admin_reload_dl.py`: `/admin/reload/dl`, path boundary, lookback mismatch, health/model info metadata 검증.
  - `tests/test_serving/test_predictor.py`: DL auxiliary prediction attach 및 DL failure degradation 검증.
  - `tests/test_datasets/test_smoke_dl_bundle.py`: smoke bundle 생성 진입점, manifest validation, `/admin/reload/dl`, 실제 `HybridPredictor.reload_dl()`, `DLModel.predict()` E2E 검증.
  - `tests/test_packaging/test_cuda_packaging.py`: `generate_smoke_dl_bundle.bat`의 `.venv_hana` 감지, `PYTHONUTF8=1`, 기본 `models\dl\smoke` 출력, `verify_smoke_dl.bat`의 `--require-dl-prediction`/`--skip-validation` 전달, `run_api_smoke_dl.bat`의 smoke env/uvicorn 실행 계약, `inspect_parquet_history.bat`의 parquet path/patient id 전달, CRLF 줄바꿈 검증.
  - `tests/test_ops/test_verify_smoke_dl.py`: 운영 검증 스크립트의 bundle 생성, reload 전 semantic validation, validation 실패 시 HTTP reload 차단, HTTP reload 요청, `/predict` 요청, `ADMIN_API_KEY` 누락 처리, `dl_prediction` optional/warn 정책 검증.
  - `tests/test_ops/test_smoke_history_provider.py`: smoke history provider의 DL history schema, unknown drug 옵션, `HybridPredictor`에 주입했을 때 `dl_prediction` 생성 경로 검증.
  - `tests/test_serving/test_main_smoke_history_provider.py`: `DDI_SMOKE_HISTORY_PROVIDER` 기본 OFF, 명시 ON provider 생성, lifespan `init_predictor(dl_history_provider=...)` 전달 검증.
  - `tests/test_ops/test_validate_dl_bundle.py`: operational DL bundle semantic validator의 정상 bundle, invalid input_dim report, out-of-range vocab index, duplicate output labels, unknown architecture warning, schema sidecar mismatch warning, CLI nonzero on error 검증.
  - `tests/test_ops/test_parquet_history_provider.py`: 하루 parquet 샘플 provider의 schema, patient filter, unknown patient empty schema, full/key dedup, YYYYMMDD date format, `HybridPredictor` DL prediction 연결 검증.
  - `tests/test_ops/test_inspect_parquet_history.py`: parquet sample inspector의 file stats, missing required column, full/output-key duplicate count, first/provided patient sample, CLI nonzero on invalid schema 검증.
- 최신 검증:
  - `cmd.exe /c "set PYTHONUTF8=1&& .venv_hana\Scripts\python.exe -m pytest tests\test_datasets\test_contracts.py tests\test_datasets\test_smoke_dl_bundle.py tests\test_serving\test_dl_predictor.py tests\test_serving\test_admin_reload_dl.py tests\test_serving\test_main_smoke_history_provider.py tests\test_packaging\test_cuda_packaging.py tests\test_ops\test_verify_smoke_dl.py tests\test_ops\test_smoke_history_provider.py tests\test_ops\test_validate_dl_bundle.py tests\test_ops\test_parquet_history_provider.py tests\test_ops\test_inspect_parquet_history.py -q"`: `60 passed`.
  - `cmd.exe /c "set PYTHONUTF8=1&& .venv_hana\Scripts\python.exe -m scripts.ops.inspect_parquet_history C:\Users\ptg\Downloads\records_20241001.parquet"`: `rows=589672`, `source counts: T30=445144, T60=144528`, `unique patients: 52261`, `full duplicates: 20181`, `output key duplicates: 216400`, provider sample schema OK.
  - `cmd.exe /c "inspect_parquet_history.bat C:\Users\ptg\Downloads\records_20241001.parquet"`: `.venv_hana` 사용, 위 parquet 점검 결과와 동일, `[OK] 점검 완료`.
  - `cmd.exe /c "set PYTHONUTF8=1&& .venv_hana\Scripts\python.exe -m scripts.ops.validate_dl_bundle models\dl\smoke"`: `status=ok`.
  - `cmd.exe /c "generate_smoke_dl_bundle.bat %TEMP%\hana_smoke_bundle_bat_codex"`: smoke bundle 생성 성공.
  - `cmd.exe /c "set ADMIN_API_KEY=smoke-test&& verify_smoke_dl.bat http://127.0.0.1:9 smoke-test --require-dl-prediction"`: 서버 미기동 상태에서 예상대로 연결 실패 및 `exit /b 1` 반환. 배치→Python 검증 모듈 호출 경로 확인.
  - `cmd.exe /c "set ADMIN_API_KEY=& run_api_smoke_dl.bat"`: `ADMIN_API_KEY` 누락 시 즉시 `exit /b 1` 반환.
  - 로컬 smoke 서버 E2E:
    - 서버: `cmd.exe /c "run_api_smoke_dl.bat smoke-test 8766"`
    - 검증: `cmd.exe /c "verify_smoke_dl.bat http://127.0.0.1:8766 smoke-test --require-dl-prediction"`: `exit 0`
    - 결과: `/admin/reload/dl` 200, `dl_loaded=true`, `dl_bundle_run_id=smoke-deploy`, `/predict` 200, `dl_prediction.predicted_label=high`, `known_drug_count=2`, `unknown_drug_count=0`, `dl_error=null`.
    - 종료: 서버 PID 26488 `taskkill /PID 26488 /F`.

## 현재 남은 상태

- smoke TorchScript bundle 생성과 hot-swap E2E는 확보되었다.
- 운영 HTTP 검증 진입점은 확보되었다. `run_api_smoke_dl.bat [ADMIN_API_KEY] [PORT]`로 smoke 서버를 띄우고, 별도 콘솔에서 `verify_smoke_dl.bat [API_URL] [ADMIN_API_KEY] --require-dl-prediction`으로 bundle validation, reload, `/predict`를 확인한다.
- HANA 없는 smoke history provider와 서버 opt-in wiring은 확보되었다. `DDI_SMOKE_HISTORY_PROVIDER=1`로 서버를 시작하면 smoke 환경에서 `dl_prediction` hard 검증이 가능하다.
- 하루 단위 parquet 샘플 provider 계약은 확보되었다. 이 provider는 실제 T30/T60 window 정확도 검증이 아니라 컬럼 계약, dedup, 환자별 `edi_code` history 추출, DL prediction 경로 확인용이다.
- 실제 학습 DL bundle의 semantic validation 진입점은 확보되었다. 학습 산출물이 준비되면 hot-swap 전에 `python -m scripts.ops.validate_dl_bundle <bundle> --check-model`을 먼저 실행한다.
- 학습된 HANA DL 모델 산출물은 아직 확인 전이다. `linear`, `gat`, `gcn` forward 계약은 serving에 준비되었지만, 실 모델이 추가 인자를 요구하는 경우에는 별도 계약 확장이 필요하다.
- DL 결과는 보조 정보로만 반환되며, 최종 위험도 판정에는 아직 반영하지 않는다.
- 운영 검증 스크립트는 기본적으로 `dl_prediction` 부재를 경고로 처리한다. `DDI_SMOKE_HISTORY_PROVIDER=1` 또는 실제 HANA provider가 연결된 환경에서는 `--require-dl-prediction`을 hard fail 조건으로 사용한다.
- 다음 실작업 gate는 폐쇄망 운영 PC를 HANA DB `10.1.67.115:30015`에 연결해 약 3개월 T30/T60 처방 자료를 확보하는 것이다. ID/PW는 사용자가 실행 시 별도 입력하며 저장소에 기록하지 않는다. 이 자료로 실제 컬럼명, 기간 window, source 분포, 중복 규모, 샘플 patient ID를 확인하기 전에는 실 학습 파이프라인, 운영 HANA provider wiring, DL `risk_level` fusion 코드를 추가하지 않는다.

## 다음 단계

1. 첫 번째 콘솔에서 `run_api_smoke_dl.bat <key> 8000`으로 smoke API 서버를 실행한다.
2. 두 번째 콘솔에서 `verify_smoke_dl.bat http://127.0.0.1:8000 <key> --require-dl-prediction`으로 smoke bundle 생성, hot-swap, `/predict dl_prediction` 응답을 확인한다.
3. 폐쇄망 운영 PC에서 HANA DB `10.1.67.115:30015`에 접속하고, ID/PW는 사용자 입력으로 받아 약 3개월 T30/T60 처방 자료를 확보한다.
4. 확보 자료를 PII 출력 없이 점검한다: rows, unique patients, source counts, 실제 컬럼명, full/output-key duplicates, 샘플 patient provider 결과.
5. 학습-서빙 계약을 확정한다: aggregate tabular, drug sequence multi-hot, graph/GAT 중 하나를 선택하고 `drug_vocab`, `edge_index`, tensor schema, TorchScript export 포맷을 고정한다.
6. 학습된 HANA DL 산출물이 확인되면 operational bundle 형식으로 변환하고, `python -m scripts.ops.validate_dl_bundle <bundle> --check-model`을 통과시킨다.
7. 실제 HANA history provider 자동 주입 경로를 결정하고, 여러 날짜 또는 DB window 조회가 가능한 실제 T30/T60 처방 이력을 운영 샘플 patient로 검증한다.
8. 충분한 성능/임상 검토 후에만 DL 결과를 최종 `risk_level`에 반영하는 별도 decision slice를 시작한다.

## Claude 협업 상태

Claude는 Obsidian `mode_11_hana_2026-05-18.md`에 CUDA126 폐쇄망 검증 성공과 smoke DL bundle 작업 결정을 기록했다. 구현 검토에서는 smoke 단계는 진행 가능하되, 실제 GNN/GAT bundle 교체 전 `edge_index` 전달 계약을 반드시 재검토하라고 지적했다.

Claude는 다음 진행 협의에서 `scripts/ops/verify_smoke_dl.py`와 `verify_smoke_dl.bat` 추가를 권장했다. 핵심 리스크는 `ADMIN_API_KEY`, `MODEL_DIR`, HANA history provider 미주입 시 `dl_prediction` 부재다.

Claude는 이후 협의에서 가장 작은 안전한 다음 단위로 `SmokeHistoryProvider`를 먼저 추가하고, 서버 wiring은 별도 슬라이스로 분리하라고 제안했다. 이 단위는 HANA 실제 연결/컬럼을 건드리지 않는다.

Claude와의 후속 협의에 따라 `DDI_SMOKE_HISTORY_PROVIDER` opt-in 서버 wiring을 추가했다. 기본 OFF, lazy import, WARNING 로그, `init_predictor(dl_history_provider=...)` 전달 방식이다.

Claude는 이후 운영자 UX 후속 단위로 `verify_smoke_dl.bat`에 `--require-dl-prediction` 전달만 먼저 추가하고, 서버 기동 배치는 포트 충돌/블로킹 프로세스 리스크가 있으므로 다음 슬라이스로 분리하라고 권장했다.

Claude는 다음 슬라이스에서 `run_api_smoke_dl.bat`를 새 창 `start` 방식보다 현재 콘솔 blocking 방식으로 만들 것을 권장했다. 환경변수 상속과 서버 로그 확인, Ctrl+C 종료가 더 명확하기 때문이다.

Claude는 architecture forward 계약 확보 후 다음 안전 단위로 실제 HANA provider wiring보다 DL bundle semantic validator를 먼저 추가하라고 권장했다. 실제 학습 모델이 아직 확인 전이므로, 배포 전 manifest/hash와 config/vocab/schema/TorchScript load 계약을 먼저 막는 쪽이 운영 리스크가 작다는 판단이다.

Claude는 semantic validator 추가 후 다음 단위로 `verify_smoke_dl.py`에 reload 전 validation을 통합하라고 권장했다. 이에 따라 smoke 운영 검증 흐름은 `create bundle -> validate bundle -> /admin/reload/dl -> /predict` 순서가 되었고, 예외 상황을 위해 `--skip-validation` 플래그를 제공한다.

Claude는 이후 실제 HANA DB 접속 전 가능한 다음 작업으로 `docs/ops/dl-smoke-runbook.md` 운영 절차서 작성을 권장했다. 운영자 관점에서 CUDA 확인, smoke E2E, 실제 bundle validation, smoke-only 환경변수 금지, 실패 triage를 한 문서에 모으는 범위다.

Claude는 하루 단위 `records_20241001.parquet` 샘플 확인 후, 실제 DB 접속 전 `scripts/ops/parquet_history_provider.py`를 ops 검증 도구로 추가하라고 권장했다. 하루 스냅샷이므로 T30/T60 window 정확도는 검증하지 않고, 출력 컬럼 계약과 중복 제거, 환자별 `edi_code` history 추출, `HybridPredictor` DL path만 검증한다.

Claude는 parquet provider 추가 후 다음 안전 작업으로 `scripts/ops/inspect_parquet_history.py` CLI를 권장했다. 이 CLI는 운영자가 하루 parquet 샘플을 PII 출력 없이 점검하고, provider 출력 계약과 중복 규모를 확인하는 dry-run 도구다.

Claude는 이후 Windows 운영자 UX를 위해 `inspect_parquet_history.bat` 배치 래퍼를 추가하라고 권장했다. `.gitignore`의 `*.bat` 무시 정책 때문에 `!inspect_parquet_history.bat` unignore도 함께 추가했다.

Hermes CLI는 이전 60초/30초 재시도와 이번 45초 협의 요청 모두 응답 없이 timeout되었다. 장기 실행 `hermes -z` 프로세스는 남기지 않았다.
