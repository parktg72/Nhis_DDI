# CLAUDE.md — MODE_11_hana

전역 `~/CLAUDE.md` (모델 역할 매트릭스·워크플로우·advisor/superpowers/gstack 디시플린)
는 **그대로 적용**. 본 문서는 프로젝트 차이만 적는다.

## 정체성

처방 데이터(HANA) 기반 **부적절 처방 위험 예측 ML 파이프라인** — 운영형 서빙 시스템.
NHIS 코호트·논문 작업 아님.

피처 우선순위 기준 4대 위험: **금기 · 중복 · 상호작용 · 다기관**.
추가로 Yellow 라벨 세분화 + 계층(hierarchical) 분류기 운영.

## 환경 — **Python 3.12 강제 (dev/prod 패리티)**

- **운영 PC = Windows 폐쇄망 + Python 3.12** 가 정답. 로컬 dev 도 같은 3.12 로 맞춘다.
- 로컬: `.venv` (uv, Python 3.12). 전역 pyenv 는 fastapi/starlette 충돌로 금지.
- Windows 배포: `packages_win/py312` 오프라인 휠 (`cp312-cp312-win_amd64`) + `install_312.bat`.
- BAT 10개는 **CRLF + `chcp 65001`** 유지 (LF 저장 시 한글 깨짐).
- 기존 3.11 `.venv` 발견 시 **재생성 후 호환 검증** — 검증 통과까지 3.11 venv 백업 보존.

```bash
uv venv --python 3.12 .venv      # 신규/재생성
source .venv/bin/activate         # pytest / python 실행 전 필수
python --version                  # → Python 3.12.x 확인
```

## HANA 데이터 (임의 변경 금지)

| 도메인 | 스키마.테이블 |
|---|---|
| 처방 T20/T30/T40/T60 | `NHISBASE.HBMT_TBGJME20` 외 |
| 자격(인구) | `NHISBDA.HHDV_DSES_YY` |
| 요양기관 | `NHISBDA.HHRT_MCINST_YY` |

ETL:
- T30 IN-list 분할 `_PID_BATCH_T30=5_000`.
- 2M+ 추출 시 세션 만료 → `db.py` auto-reconnect (commit `bd8e9dc`).
- 사전 층화 샘플링은 `strata_utils._DEFAULT_AGE_BINS` **단일 출처**. 재정의 금지.

## 디렉터리 지도

| 경로 | 역할 |
|---|---|
| `hana/`, `hana_app/`, `hana_desktop/` | HANA 추출·데스크톱 앱 |
| `dags/` | Airflow DAG (ETL→feature→train) |
| `labeling/` | 4대 위험 + Yellow 세분화 |
| `serving/` (`routers/`) | FastAPI 추론 (계층 모델·`/reload`) |
| `lay_out/` | PyWebView UI (page1/2/3) |
| `tests/` | etl·features·labeling·train·serving·integration·dags·hana_app |
| `rules/`, `drugbank/`, `hira/` | 외부 룰 / 약물 사전 |
| `packages_win/py312/` | Windows 오프라인 휠 |
| `mlruns/` | MLflow 실험 |
| `docs/` | `PROJECT_PLAN`, `CLINICAL_STANDARDS_v1.0`, `QA_PLAN_v1.0`, `data_pipeline_architecture`, `web-user-guide` |

## 학습 ↔ 서빙 컬럼 정렬 (회귀 방지)

`RequestFeatureBuilder` 의 컬럼명·순서는 학습과 **완전 일치** (commit `d201743` 회귀 사례).
serving 변경 시: 학습 feature schema diff → `tests/test_serving` + `tests/test_features` →
`/reload` 후 sample payload sanity check.

## Page 3 학습 UI

hierarchical 타겟 옵션 통합 (commit `fc9bd8c`), `predict_risk` → `features_df` 채우기 완료.
UI 변경 시 PyWebView 데스크톱 모드에서 **실제 클릭 검증**.

## 다중 AI 협업 (전역 + 본 레포 우선순위)

- **Critical** (라벨 정의·학습/서빙 스키마·HANA 쿼리) → cross-family 필수 (Anthropic ↔ OpenAI).
- 일상 구현 (UI·작은 리팩터) → Sonnet 단독 가능.
- `/advisor` 는 plan + 마무리 두 번이 기본.

## 묻지 않고 하면 안 되는 것

- HANA 스키마·테이블·컬럼명 임의 추측/변경
- 공유 상수 변경 (`_PID_BATCH_T30`, `_DEFAULT_AGE_BINS` 등)
- 학습 feature 스키마 변경 (서빙 동시 수정 없으면 금지)
- `packages_win/py312/` 휠 추가·삭제
- BAT 파일 LF 저장
- `mlruns/` 직접 편집
- 결과 `.parquet` / `out/` 산출물 자동 커밋

## 커밋 컨벤션

스코프 prefix 유지: `feat(serving)`, `fix(serving)`, `chore(deploy)`, `feat(ui)` 등.
한 커밋 = 한 논리 변경, 메시지는 *why* 중심.

## 스타일

언어는 사용자 톤. 숫자 → 해석 → 주의 순.
