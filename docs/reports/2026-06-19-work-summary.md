# 2026-06-19 ~ 2026-06-23 작업 요약

## 범위
- Windows 폐쇄망 설치 스크립트(`install_all.bat`, `install_312.bat`) 오류 보정 및 정적 검증
- Phase 3 ML/DL 및 CUDA 12.6 PyTorch/PyG 폐쇄망 설치 계약 보강
- DOCX 보고서의 혼동행렬 오판 사유 분석 기능 추가
- 계층 분류 Stage 2 예측값 정규화 및 관련 지표/CV 회귀 보강
- `H:\mode_11_hana` Windows 폐쇄망 실행용 clean staging 생성 및 재점검

## 1. Windows 설치 스크립트 보정

### 원인
- `install_all.bat 312 venv` 실행 시 상단 한글 주석/문자열이 CMD 코드페이지 처리 전 깨져 `'를' is not recognized...` 형태로 파싱될 수 있음.
- 기존 `.venv`가 WSL/Linux venv 또는 손상된 venv이면 `.venv\Scripts\python.exe`가 없어 `The system cannot find the path specified` 및 `hdbcli 설치 실패`로 이어질 수 있음.
- HANA/ML 단계별 설치에 Python 3.12 constraint가 빠지면 중복 wheel 버전으로 설치 churn이나 resolver 경고가 발생할 수 있음.

### 조치
- `install_all.bat` 상단 헤더를 ASCII-only로 변경하고 `chcp 65001 >nul` 유지.
- `.venv\Scripts\python.exe` 존재 및 실행 가능 여부를 검증하도록 수정.
- Windows venv가 아니거나 실행 불가하면 `.venv`를 삭제 후 재생성하도록 수정.
- Python 3.12 지정 시 `py -3.12 -m venv` 우선, 실패 시 `python -m venv` fallback.
- `hdbcli`, `hana-ml` 런타임 의존성, `pydotplus`, DOCX/그래프 패키지에 `%CONSTRAINT%` 적용 및 명시 검증 추가.
- `install_312.bat`의 HANA/ML 단계별 설치에도 `%CONSTRAINT%` 적용 보강.
- `run_desktop.bat`에 `PYTHONUTF8=1` 추가.

## 2. Phase 3 ML/DL 및 CUDA 폐쇄망 설치 계약

### 조치
- `download_all.bat`가 Python 3.12일 때 `packages_win\download_cuda_cu126.bat`를 호출하도록 보강.
- `install_312.bat`와 `install_all.bat`가 `torch`, `pytorch-tabnet`을 명시 설치/검증하도록 보강.
- `DDI_REQUIRE_PHASE3_DL=1`, `DDI_REQUIRE_CUDA_DL=1` hard-fail 옵션을 유지/검증.
- `hana_app/requirements.txt`에 Phase 3 DL 학습 의존성을 반영.
- `tests/test_hana_app/test_windows_ml_dl_install_contract.py` 추가.

### CUDA wheel set 확인
- `packages_win/requirements_cuda_cu126.txt`
  - `torch==2.11.0+cu126`
  - `torch-geometric==2.7.0`
  - `pyg_lib==0.6.0+pt211cu126`
  - `torch_scatter==2.1.2+pt211cu126`
  - `torch_sparse==0.6.18+pt211cu126`
  - `torch_cluster==1.6.3+pt211cu126`
- `H:\mode_11_hana\packages_win\py312`에 위 CUDA/PyG wheel 전부 존재 확인.

## 3. DOCX 혼동행렬 오판 사유 분석

### 조치
- `hana_app/core/ml_runner.py`
  - 평가 세트에서 `y_true != y_pred`인 오판 사례를 `metrics["misclassified_cases"]`에 저장.
  - 저장 필드는 익명 case 번호, 실제 라벨, 예측 라벨, 예측 확신도, 안전 feature 요약으로 제한.
  - `patient_id`, 원자료 식별자, 성별, 연령, 원본 row index 등 직접/준식별자는 저장하지 않음.
- `hana_app/core/report_exporter.py`
  - 혼동행렬 아래 `5-1-1. 오판 사유 분석` 섹션 추가.
  - 오판 유형별 건수 및 익명 대표 사례 표 생성.
  - 사유는 안전 feature 기반 rule summary로 작성.
- 테스트 보강
  - 오판 요약이 식별자를 노출하지 않는지 검증.
  - DOCX section plan에 `misclassification_analysis`가 포함되는지 검증.
  - ML runner 오판 사례 builder가 식별자/준식별자를 제외하는지 검증.

### 주의
- 과거 저장 결과에는 `misclassified_cases`가 없으므로, 개별 오판 사유는 새 학습/평가 결과부터 DOCX에 포함됨.

## 4. 계층 분류 회귀 보강

### 조치
- Stage 2 XGBoost `predict()`가 Windows/버전에 따라 확률 행렬을 반환할 때 local class index로 정규화.
- 관련 hierarchical CV/metrics/runner 테스트 보강.
- Page3 계층 분류 안내 문구를 실제 동작에 맞게 수정.

## 5. `H:\mode_11_hana` clean staging 및 폐쇄망 재점검

### 조치
- 원본: `C:\model\mode_11_hana` (`/mnt/c/model/mode_11_hana`)
- 대상: `H:\mode_11_hana` (`/mnt/h/mode_11_hana`)
- 포함: 소스, 설정, BAT, tests, docs, `packages_win/py312`, `hana/py312`, 런타임 기준 `data/processed` 7개 파일.
- 제외: `.git`, agent metadata, venv류, pycache/pytest cache, build/dist, raw/patient data, generated datasets/reports/results/log/cache.

### 포함한 런타임 기준 파일
- `data/processed/ddi_matrix_final.parquet`
- `data/processed/efcy_duplicate_groups.parquet`
- `data/processed/drug_name_index.parquet`
- `data/processed/hira_drug_master.parquet`
- `data/processed/edi_to_wk.parquet`
- `data/processed/edi_to_wk.meta.json`
- `data/processed/cyp_matrix.parquet`

### 검증
- 대상 크기: 약 6.5G.
- 핵심 파일 byte-level match: 전부 MATCH.
- 제외 항목 부재: OK.
- `data/processed` 외 parquet 없음.
- 런타임 기준 parquet 읽기 성공.
- 앱 로더 기준:
  - DDI matrix 로드 OK `(1456206, 7)`
  - duplication groups 로드 OK `(404, 16)`
  - DrugMaster 로드 OK `4426` 코드
  - CYPFeatureExtractor 로드 OK
- 전체 `.bat` 17개: CRLF, `chcp 65001`, Python 호출 스크립트 `PYTHONUTF8=1` 확인.
- AGY HQ 폐쇄망/Windows 배포 관점 read-only 검토: PASS.

## Windows 실행 명령

Python 3.12가 없으면 먼저 실행:

```bat
H:\mode_11_hana\python\python-3.12.10-amd64.exe
```

설치:

```bat
cd /d H:\mode_11_hana
install_312.bat venv
```

설치 완료 후 웹앱 실행:

```bat
cd /d H:\mode_11_hana\hana_app
run.bat
```

CUDA/DL까지 반드시 hard-fail 검증:

```bat
cd /d H:\mode_11_hana
set DDI_REQUIRE_PHASE3_DL=1
set DDI_REQUIRE_CUDA_DL=1
install_312.bat venv
```

대용량 학습/피처 생성 전 권장:

```bat
set HANA_FEAT_TMP=D:\hana_tmp
```

여유 공간은 10GB 이상 권장.
