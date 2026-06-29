# 2026-06-25 자동 ML/DL→hierarchical 학습 및 H: 운영 staging 요약

## 범위

- DOCX 보고서 `5-5. 모델 비교`가 최근 `hierarchical` 행만 반복하거나, 방금 학습한 ML/DL 결과를 잃는 문제를 수정했다.
- 일반 사용자가 학습 순서를 알 필요 없도록 Page 3에서 `hierarchical` 학습 시 선택된 ML/DL 모델을 `risk_binary` 비교용으로 먼저 자동 학습하고, 이어서 hierarchical Stage 1/2를 실행하도록 변경했다.
- `H:\mode_11_hana`를 삭제 후 새로 만들고, Windows 설치·운영에 필요한 clean runtime tree를 전체 staging했다.

## 주요 변경

### Page 3 학습 순서

- `build_training_sequence()`를 추가해 학습 실행 계획을 명시화했다.
- `target == "hierarchical"`이면 자동 순서가 다음처럼 동작한다.
  1. 선택된 Phase 2 ML / Phase 3 DL 모델을 `risk_binary` 비교용으로 선행 학습
  2. hierarchical Stage 1/Stage 2 학습
  3. Page 4 및 DOCX `5-5. 모델 비교`에 방금 학습한 ML/DL + hierarchical 결과를 함께 표시
- 비교용 ML/DL 자동 선행 학습이 전부 실패하면 hierarchical 학습을 중단한다. 일부 모델만 성공하면 성공한 ML/DL 결과와 hierarchical을 계속 비교한다.

### 결과 상태/저장 로직

- `merge_train_results()`로 현재 세션 모델 비교 결과를 모델명 기준으로 병합한다.
- hierarchical 학습이 기존 `st.session_state.train_results`를 `{"hierarchical": ...}`로 덮어써 직전 ML/DL 결과를 잃던 문제를 수정했다.
- `_save_result()`가 저장 직후 in-memory result에 `timestamp`를 채워 `current-01` 대신 실제 학습 시각이 표에 표시되게 했다.
- `_save_result()`가 기존 `model_path`를 `"None"` 문자열로 덮지 않도록 보존 로직을 추가했다.

### DOCX/Page 4 모델 비교

- `5-5. 모델 비교`는 현재 세션 결과를 우선 사용하고, 저장 이력은 모델별 최신 1건만 뒤에 붙인다.
- 같은 모델의 오래된 `hierarchical` 이력이 여러 줄 반복되지 않는다.
- 표와 그래프는 `ML` / `DL` / `Hierarchical` 구분을 포함한다.

## 커밋

- `9903237cbeff2f0a744d87725f13ca93c490ce7c`
- 메시지: `fix: auto-run comparison before hierarchy`
- 포함 파일:
  - `hana_app/core/ml_runner.py`
  - `hana_app/core/report_exporter.py`
  - `hana_app/pages/3_🤖_모델_학습.py`
  - `hana_app/pages/4_📊_결과_분석.py`
  - `tests/test_hana_app/test_ml_runner_result_state.py`
  - `tests/test_hana_app/test_report_exporter.py`

## 검증 evidence

### 로컬 C: 작업 트리

- `.venv_wsl/bin/python -m pytest tests/test_hana_app/test_report_exporter.py tests/test_hana_app/test_ml_runner_result_state.py -q --disable-warnings`
  - `30 passed, 803 warnings`
- `.venv_wsl/bin/python -m py_compile ...`
  - exit 0
- `git diff --check ...`
  - exit 0
  - Page 3 CRLF 관련 Git warning만 있음. 실제 파일은 CRLF 유지 확인.

### H: clean staging

- 대상: `H:\mode_11_hana`
- 새 폴더 생성 후 전체 staging:
  - 파일 수: 1,326
  - regular files 전송: 1,232
  - 전송량: 7.97GB
  - 최종 대상 크기: 7.6GB
- 캐시/환경/생성물 제외 확인:
  - `.pytest_cache`, `__pycache__`, `*.pyc`, `.venv*`, `venv`, `.git`, agent metadata, `build`, `dist`, `mlruns`, `out`, `hana_app/results`, `data/Raw`, `data/raw`, `data/datasets`, `data/reports` 모두 없음.
- runtime reference data:
  - `data/processed` 필수 7개 파일 정확히 존재: `ddi_matrix_final.parquet`, `efcy_duplicate_groups.parquet`, `drug_name_index.parquet`, `hira_drug_master.parquet`, `edi_to_wk.parquet`, `edi_to_wk.meta.json`, `cyp_matrix.parquet`.
  - `data/drugbank`, `data/dur`, `data/vocab`, `hana_app/models` 존재.
- BAT 검증:
  - 17개 BAT 모두 CRLF 정상 및 `chcp 65001` 포함.
  - `BAT_NOT_CRLF_COUNT 0`, `BAT_MISSING_CHCP_COUNT 0`.
- Windows wheelhouse:
  - `packages_win/py312`: 220 wheels
  - `hana/py312`: 211 wheels
  - 필수 패키지 확인 OK: `torch`, `pytorch-tabnet`, `xgboost`, `lightgbm`, `catboost`, `streamlit`, `python-docx`, `lxml`, `matplotlib`, `Pillow`, `hdbcli`, `hana-ml`.
  - CUDA/PyG wheels 확인 OK: `torch==2.11.0+cu126`, `torch-geometric==2.7.0`, `pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster`; `cuda_missing_count 0`.
- H: 코드 검증:
  - 핵심 Python 파일 AST parse 모두 통과.
  - H: 대상 focused regression: `9 passed, 136 warnings`.
  - H: 대상 DOCX sanity: `['xgboost', 'tabnet', 'gnn', 'hierarchical']` 행 포함 및 `DOCX_SANITY_OK`.

## 운영 방법

Windows CMD:

```bat
cd /d H:\mode_11_hana
install_all.bat 312 venv
cd /d H:\mode_11_hana\hana_app
run.bat
```

GPU/CUDA DL을 반드시 강제하려면:

```bat
cd /d H:\mode_11_hana
set DDI_REQUIRE_PHASE3_DL=1
set DDI_REQUIRE_CUDA_DL=1
install_312.bat venv
```

## 남은 상태

- `main`은 `origin/main`보다 1커밋 앞서 있다.
- `9903237`은 아직 push하지 않았다.
- 이번 커밋 범위 외 기존 dirty/untracked 항목은 남겨두었다.
  - `.gitignore`
  - `AGENTS.md`
  - `CLAUDE.md`
  - `.agents/`
  - `.claude/`
  - `.understand-anything/`

## 주의

- WSL 검증 환경에는 `Malgun Gothic` 폰트가 없어 matplotlib 한글 glyph/font warning이 발생했다. Windows 환경에서는 보통 맑은 고딕이 있어 줄어들 가능성이 높다.
- CUDA wheel 파일 존재는 확인했지만, 실제 폐쇄망 PC의 NVIDIA driver/CUDA runtime readiness는 별도 확인해야 한다.
