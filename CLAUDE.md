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

피처 빌드 임시 디스크 (`build_patient_features_from_parquet`):
- DuckDB `COPY ... PARTITION_BY` 가 전체 raw Parquet 를 임시 디렉터리에 통째 복제
  (피크 ≈ 소스의 ~2배). 기본 임시 경로가 시스템 드라이브면 디스크풀 IOException 위험.
- **여유 10GB+ 드라이브를 `HANA_FEAT_TMP` 로 지정** (예: `set HANA_FEAT_TMP=D:\hana_tmp`).
  우선순위: `HANA_FEAT_TMP` → `HANA_TMP_DIR` → `hana_config.json` → 시스템 temp.
- `_preflight_temp_space` 가 시작 전 필요량 vs 가용량을 점검해 부족 시 친절한
  `InsufficientDiskSpaceError`(대안 드라이브 안내)를 던진다. 미설정 시 raw IOException.

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
- **OpenCode** (dev-only 보조): read-only 코드 리뷰·리팩터 대안·UI/UX 아이디어·계획 second opinion. Direct CLI `opencode run`으로만 사용하며, additive only라 critical cross-family gate를 충족하지 않고 Windows 폐쇄망 production dependency가 될 수 없다.
  - 모델: `opencode-go` provider 우선 (기본 `opencode-go/qwen3.7-max`). go 한도 소진(429/quota) 시 zen provider 폴백 `--model opencode/glm-5` (수동 재시도 — `opencode run` one-shot은 자동 폴백 미작동).
- **L0 결정**: Hermes가 LO로 호출한 세션은 Hermes가 오케스트레이션 L0, 그 외 단독 세션은 Claude가 L0. L0가 과제 성격에 따라 codex/opencode/agy 역할·모델을 배정한다 (전역 `~/.claude/CLAUDE.md` 위임 라우팅 참조).
- `/advisor` 는 plan + 마무리 두 번이 기본.

### ⚠️ 메시지 전송 보류 원칙 (Message Transmission Deferral Rule)
- **절대 원칙**: 다른 에이전트가 백그라운드 작업이나 연산을 수행하고 있을 때 메시지를 즉시 전송하지 말고 보류할 것.
- **이유**: 작업 수행 중에 실시간 메시지가 끼어들면 동작 흐름이 단절되거나 Hallucination 및 비정상 정지가 일어날 위험이 높음.
- **대기/상태 판단**:
  - 상대 에이전트로부터 "작업 완료" 혹은 `<channel>` 리마인더 응답을 받은 직후가 "대기 상태"이므로 이때 송신 가능.
  - 미응답이 지속 중일 때는 "작업 진행 중"으로 간주하고, 보낼 메시지를 임시 보류 큐(메모)에 기록한 뒤 대기.

### ❄️ Future-onset Research Freeze (동결 2026-05-26 / Jan 2025 트리거 취소 2026-06-02)
- **상태**: `RESEARCH_TRACK_FROZEN` (해제 트리거 없음 — 무기한 보류/parked)
- **금지 가드레일**: Nov→Dec 홀드아웃 데이터셋(`data/datasets/future_mi_t6_20241130_to_20241231_with_inst_efmdc_demo_disjoint_octnov`)을 대상으로 하는 추가적인 모델·피처·하이퍼파라미터 튜닝 일체 금지 (검증셋 과적합 방지). 동결 이유(반복 ablation 과사용)는 유효하며, 별도 해제 트리거는 더 이상 계획되지 않는다.
- **확보된 Raw 데이터 (최종)**: `data/Raw/records_20240701.parquet` .. `records_20241231.parquet` 184개 일별 파일(2024-07~12, 6개월), eligibility 50만명. **이 6개월이 최종 데이터셋**이며 추가 월 확보 계획 없음. 훈련·컨텍스트는 2024-07~11, Nov→Dec는 동결 홀드아웃. same-window baseline·DL 운영화 등 freeze-safe 작업에 활용하되 Nov→Dec 홀드아웃 튜닝 금지는 유지.
- **데이터셋 범위 확정 (2026-06-02)**: 기존 Gate 5A(2025-01 Raw 확보 시 Dec→Jan unseen holdout 해제)와 Gate 5B는 **공식 취소/폐기**. 2025년 1월 데이터는 확보하지 않으며 관련 계획·참조를 제거한다. future-onset 연구 트랙은 무기한 보류.
- **허용 범위**: Nov→Dec 홀드아웃을 건드리지 않는 freeze-safe 작업(same-window sparse-linear baseline, DL 운영화 등)만 진행한다.

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
