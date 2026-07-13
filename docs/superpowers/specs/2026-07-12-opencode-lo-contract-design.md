# OpenCode 최종 LO 계약 기반 아키텍처 설계

**작성일:** 2026-07-12
**상태:** 사용자 승인됨 (사용자가 문서를 검토하고 승인함)
**적용 범위:** MODE_11_hana 프로젝트 전체 (serving, hana_app, dags, scripts, rules)

---

## 1. 개요

### 1.1 목적

본 문서는 MODE_11_hana 프로젝트의 다중 AI 에이전트 협업 체계와 코드 아키텍처를 계약(contract) 기반으로 정합한다. OpenCode를 최종 L0/LO 오케스트레이터로 확정하고, Claude/Codex/AGY 세 worker의 역할 라우팅, advisor panel 선행 게이트, Oracle 승격 조건, 그리고 profile별 계약 프레임워크를 정의한다. 또한 현행 아키텍처의 책임 혼재 문제를 단계적으로 해결하기 위한 Phase 0A/0B/1/2A/2B 설계를 제공한다.

### 1.2 설계 원칙

1. **OpenCode가 최종 LO**: 사용자 커뮤니케이션, 작업 분류, worker 라우팅, 승인, 통합, 검증을 OpenCode가 독점한다. worker는 증거만 반환하고, OpenCode가 성공 여부를 최종 판정한다.
2. **계약 기반 분리**: 하나의 범용 feature 목록이 아니라, 각 운영 프로파일(tabular_binary, hierarchical, ui_experimental, dl_history)이 독립 계약을 가진다. profile 간 피처 집합은 다를 수 있으며, 하나의 목록으로 평탄화(flatten)하거나 zero diff로 강제해서는 안 된다.
3. **동결 안전**: Nov->Dec 홀드아웃 튜닝, Gate 5A/5B 활성화, 2025-01 데이터 확보 등 연구 동결 트랙 위반 작업은 설계 단계에서 배제한다.
4. **점진적 개선**: 한 번에 전체를 재작성하지 않고, 정책/런타임 상태 매핑(Phase 0A)부터 계약 명세(Phase 2A)까지 단계적으로 진행한다.
5. **검증 후 보고**: 모든 변경 주장은 도구 출력 기반이다. "될 것이다"는 검증이 아니다.
6. **동작 동결**: Phase 2는 기존 동작을 codify/freeze할 뿐, 피처 순서 변경, 라벨 변경, 시맨틱 버전 변경, 아티팩트 마이그레이션, 재학습, predictor 분할을 수행하지 않는다.

### 1.3 용어 정의

| 용어 | 정의 |
|---|---|
| LO | Leading Orchestrator. 최종 결정권과 사용자 커뮤니케이션을 소유한다. |
| L0 | LO와 동의어. 본 프로젝트에서는 OpenCode가 L0/LO이다. |
| worker | OpenCode 산하 Claude/Codex/AGY 실행 단위. 독립 세션, self-escalation 금지. |
| advisor panel | Oracle 승격 전 Fable 5 + Codex GPT-5.6-Sol 독립 의견을 수집하는 필수 게이트. |
| Oracle | OpenCode Go 모델 기반 read-only 고위험 분석 도구. advisor panel 완료 후에만 고려. |
| profile | 운영 모델 운용 단위. tabular_binary, hierarchical, ui_experimental, dl_history. |
| contract | profile별 피처/라벨/버전/검증 규칙을 정의한 명세. |
| characterization test | 기존 동작의 현재 상태를 기록하는 회귀 테스트. 구조 변경 전에 작성하여 동결을 보장. |

---

## 2. 현행 아키텍처 분석

### 2.1 전체 구조도

```
+=====================================================================+
|                    MODE_11_hana 시스템 전체 구조                      |
+=====================================================================+

  [사용자 인터페이스]
  +-------------------+     +-------------------+
  | lay_out/          |     | hana_app/         |
  | PyWebView UI      |     | Streamlit App     |
  | (page1/2/3)       |     | pages/1~6         |
  +--------+----------+     +--------+----------+
           |                        |
           v                        v
  +--------+------------------------+----------+
  |          hana_app/core/                    |
  |  ml_runner.py    hierarchical_runner.py    |
  |  hana_etl.py     strata_utils.py            |
  |  (Page 3 학습 경로)                          |
  +--------+-----------------------------------+
           |
           | (피처/라벨/모델 산출)
           v
  +--------+-----------------------------------+
  |          scripts/                          |
  |  etl/    features/   train/    ops/        |
  |  (ETL+피처+학습+운영스크립트)               |
  +--------+-----------------------------------+
           |
           | (모델 산출물)
           v
  +--------+-----------------------------------+
  |          serving/  (FastAPI)               |
  |  main.py  predictor.py  schemas.py         |
  |  routers/  dl_predictor.py  hana_history.py |
  |  (추론 서비스)                               |
  +--------+-----------------------------------+
           |
           | (Airflow 스케줄)
           v
  +--------+-----------------------------------+
  |          dags/  (Airflow DAG)              |
  |  ddi_etl_dag.py   ddi_feature_dag.py       |
  |  ddi_train_dag.py ddi_batch_predict_dag.py |
  +-------------------------------------------+

  [외부 참조]
  +-------------------------------------------+
  |  rules/    drugbank/    hira/             |
  |  (규칙/약물사전/심평원 DUR)                 |
  +-------------------------------------------+
```

### 2.2 현행 문제점

본 설계가 해결하는 현행 아키텍처의 구조적 문제는 다음과 같다.

**P1: predictor.py 책임 혼재**
`serving/predictor.py`는 단일 파일에 HybridPredictor, MLModel, HierarchicalPredictor, RequestFeatureBuilder, rule bridge 함수, 위험 약물 판정 헬퍼가 전부 혼재한다. 또한 `hana_app.core.hierarchical_runner`에서 `predict_risk`, `ACTION_BY_LABEL`, `STAGE2_LABELS`를 import하여 serving이 hana_app에 런타임 의존한다. 이는 serving을 독립 배포하기 어렵게 만든다. 본 설계에서는 predictor 분할을 Phase 3 (out of scope)로 분류하고, 현 설계 주기에서는 계약 명세와 특성화 테스트로 준비한다.

**P2: 이중 학습 경로**
Page 3 UI(`hana_app/core/ml_runner.py`)와 Airflow DAG(`dags/ddi_train_dag.py` -> `scripts/train/pipeline.py`)가 서로 다른 학습 경로를 사용한다. `ml_runner.py`의 `FEATURE_COLS`, `feature_engineer.py`의 `ETL_NUMERIC_COLS`, `predictor.py`의 `_BUILDER_KNOWN_COLS`가 서로 다른 출처에서 피처 목록을 정의하고 있다. 이는 train/serve 스큐 위험의 근원이다.

**P3: 피처 정의 분산**
피처 이름, 순서, 기본값 의미가 네 곳 이상에 분산되어 있다:
- `serving/predictor.py` `_BUILDER_KNOWN_COLS` (서빙 기준)
- `hana_app/core/ml_runner.py` `FEATURE_COLS` (Page 3 학습 기준)
- `scripts/features/feature_engineer.py` `ETL_NUMERIC_COLS` (Airflow 피처 엔지니어링 기준)
- `scripts/datasets/contracts.py` `ML_DATASET_REQUIRED_COLUMNS` (데이터셋 계약 기준)

`dup_efmdc`는 `ml_runner.py`의 `FEATURE_COLS`에는 있지만 `feature_engineer.py`의 `ETL_NUMERIC_COLS`에는 없다. `sex_m`은 `feature_engineer.py`에 없고 `ml_runner.py`에만 있다. 이 분산은 commit `d201743` train/serve 회귀의 근본 원인이었다.

**P4: scripts/ops 혼재**
`scripts/ops/` 디렉터리가 재사용 가능한 라이브러리 코드와 일회성 명령 스크립트를 혼재하고 있다. import 경로가 불명확해진다. `scripts/ops` 재배치는 본 설계 범위 밖의 후속 작업이다.

**P5: 아키텍처 문서 비현실성**
`data_pipeline_architecture.md`는 Spark, HDFS, S3, Feast, Great Expectations, Grafana를 기술 스택으로 명시하지만, 실제 운영 환경은 Windows 폐쇄망 Python 3.12 단일 머신이다. 이 문서는 참고용 설계(aspire)이지 운영 현실이 아니다. 문서 정합과 HANA 코드 리팩터링은 각각 별도의 후속 작업이다.

### 2.3 현행 계약 버전 상수

코드에 이미 존재하는 시맨틱 버전 상수가 사실상 profile별 계약의 시초이다:

| 상수 | 값 | 위치 | 역할 |
|---|---|---|---|
| `DDI_FEATURE_SEMANTICS_VERSION` | `ddi.v2` | `scripts/etl/prescription_aggregator.py` | DDI 카운트 의미 버전. v2는 WK->DrugMaster->DB-code overlap 경로. |
| `FEATURE_SEMANTICS_VERSION` | `rulefeat.v1` | `scripts/etl/prescription_aggregator.py` | 룰 파생 피처(triple_whammy, 위험약물 플래그) 의미 버전. |
| `_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT` | `2026-08-01` | `serving/predictor.py` | lenient escape hatch sunset. 이 날짜 이후 strict 강제. |

본 설계는 이 상수들을 계약 프레임워크 내로 통합한다. Phase 2에서는 이 상수값을 변경하지 않고 codify만 수행한다.

---

## 3. OpenCode 최종 LO 거버넌스

### 3.1 거버넌스 구조도

```
+=====================================================================+
|                  OpenCode L0/LO 거버넌스 구조                         |
+=====================================================================+

                        +-----------------+
                        |    사용자       |
                        +--------+--------+
                                 |
                                 v
                         +--------+--------+
                         |   OpenCode LO   |
                         |  (최종 결정권)   |
                         +----+---+----+---+
                              |   |    |
               +--------------+   |    +--------------+
               |                  |                   |
               v                  v                   v
      +--------+--------+ +-------+-------+ +--------+--------+
      |  Claude Worker  | | Codex Worker  | |  AGY Worker    |
      |  (Anthropic)    | | (OpenAI/GPT)  | |  (Google/Gemini)|
      +--------+--------+ +-------+-------+ +--------+--------+
               |                  |                   |
               |  [필수 게이트]     |                   |
               v                  v                   |
      +--------+--------+ +-------+-------+            |
      |  Fable 5 Advisor| | Codex GPT-5.6 |            |
      |  (read-only)    | | -Sol Advisor |            |
      +--------+--------+ +-------+-------+            |
               |                  |                   |
               +--------+---------+                   |
                        |                             |
                        v                             |
               +--------+--------+                    |
               | OpenCode 합성    |                    |
               | (양측 의견 통합) |                    |
               +--------+--------+                    |
                        |                             |
                        | [조건 충족 시]               |
                        v                             |
               +--------+--------+                    |
               |    Oracle        |                    |
               | (Go 모델, read-  |                    |
               |  only, 고위험)   |                    |
               +--------+--------+                    |
                        |                             |
                        +-----------------------------+
                        |
                        v
               +--------+--------+
               |  최종 결정 및    |
               |  사용자 보고     |
               +-----------------+
```

### 3.2 OpenCode LO 권한

OpenCode LO가 독점하는 책임:

1. **사용자 커뮤니케이션**: 최종 사용자 응답은 OpenCode만 작성한다. worker는 OpenCode에게 결과를 반환하고 사용자에게 직접 메시지를 보내지 않는다.
2. **작업 분류 및 라우팅**: 과제 성격을 판단하여 적절한 worker에게 역할과 모델을 배정한다.
3. **승인 게이트**: 쓰기, 외부 부작용, 커밋, 푸시, PR, 머지, 배포는 OpenCode의 명시적 승인이 필요하다. 범위나 대상 파일이 변경되면 재승인이 필요하다.
4. **통합 및 검증**: worker 결과를 무비판 수용하지 않고, OpenCode가 직접 변경 파일을 확인하고 관련 테스트/타입체크/빌드를 실행하여 검증한다.
5. **충돌 해결**: worker 간 의견 충돌 시 OpenCode가 최종 판정한다.
6. **보고**: 우려, 간극, 불확실성을 명확히 보고한다.

### 3.3 worker 운영 규칙

- worker 세션은 독립적이다. 세션 전환 시 새 컨텍스트 요약을 받는다.
- worker self-escalation은 금지된다. OpenCode 승인 없이 다른 worker를 호출할 수 없다.
- worker는 증거(변경 파일, 실행 명령, 검증 상태, 위험, 권장 다음 단계)를 포함한 결과를 반환한다.
- 커밋, 푸시, PR, 머지, 릴리스, 배포는 명시적 요청이 없으면 불가하다.
- 메시지 전송 보류 원칙: 다른 에이전트가 작업 중일 때 메시지를 즉시 전송하지 않고 대기한다.

---

## 4. 역할 라우팅 (Claude/Codex/AGY)

### 4.1 라우팅 매트릭스

| 과제 성격 | 담당 worker | 모델 패밀리 | CLI |
|---|---|---|---|
| 요구사항 종합, 아키텍처, 라벨 의미, 스키마/동결 논리 검토, 최종 QA | Claude HQ | Anthropic | Claude Code CLI |
| 복잡한 구현, TDD, 기술 검증, train/serve 패리티 회귀, read-only 코드 리뷰 | Codex HQ | OpenAI/GPT | Codex CLI |
| 환경, DevOps, Windows 배포, Python 3.12 패리티, BAT/CRLF, 디스크 게이트, 병렬/대량/저위험 작업 | AGY HQ | Google/Gemini | AGY CLI |
| OpenCode 내부 agent/category, Oracle, visual/multimodal | OpenCode Go | opencode-go/* | OpenCode 내부 |

### 4.2 모델 패밀리 전용 CLI 원칙

GPT/OpenAI 작업은 Codex CLI를 통해서만, Gemini/Google 작업은 AGY CLI를 통해서만, Anthropic/Claude 작업은 Claude Code CLI를 통해서만 수행한다. OpenCode Go provider를 우회하여 직접 OpenCode GPT/Gemini/Claude/GitHub Copilot 모델로 라우팅하지 않는다.

### 4.3 재라우팅 규칙

레인 한도 소진 또는 장애 시 OpenCode LO가 위임 자체를 재라우팅한다. CLI 자체 폴백 체인이 없으므로, L0가 작업을 다른 패밀리 CLI로 재분류하여 위임한다.

| 소진/장애 레인 | 1차 재라우팅 | 2차 재라우팅 |
|---|---|---|
| Codex/GPT | Claude Code 또는 AGY CLI로 재분류 | OpenCode Go 내부 agent로 축소 수행 |
| AGY/Gemini | Codex 또는 Claude Code CLI로 재분류 | OpenCode Go 내부 agent로 축소 수행 |
| Claude/Anthropic | Codex 또는 AGY CLI로 재분류 | OpenCode Go 내부 agent로 축소 수행 |
| OpenCode Go 단일 모델 | 다른 opencode-go/* 모델로 수동 재시도 | Codex/AGY/Claude CLI로 재분류 |
| OpenCode Go 전체 장애 | L0 승인 paid Zen 수동 재시도 | Codex/AGY/Claude CLI로 재분류 |

재라우팅 시 critical 작업의 cross-family 게이트는 유지한다. 대체 레인 모델의 패밀리를 확인하고 게이트 구성이 깨지면 OpenCode가 사용자에게 보고한다.

### 4.4 Critical 변경 cross-family 게이트

라벨 정의, train/serve 스키마, HANA 쿼리 로직, 동결/게이트 정책 변경은 머지 전 cross-family 검토가 필수이다. OpenCode는 additive only이며 이 cross-family 요건을 충족하지 않는다. 즉, critical 변경은 Anthropic과 OpenAI 양측의 독립 검토가 필요하며, OpenCode Go만으로는 부족하다.

---

## 5. Advisor Panel 게이트

### 5.1 게이트 구조도

```
+=====================================================================+
|                Advisor Panel -> Oracle 승격 게이트                    |
+=====================================================================+

  [1단계: Advisor Panel (필수)]
  +-------------------------------------------+
  |  ask_advisor_panel                        |
  |                                           |
  |  +-----------------+  +----------------+ |
  |  | Claude Code     |  | Codex CLI      | |
  |  | Fable 5         |  | GPT-5.6-Sol    | |
  |  | (read-only)     |  | (read-only)    | |
  |  +--------+--------+  +--------+-------+ |
  |           |                    |         |
  |           v                    v         |
  |  +--------+--------+  +--------+-------+ |
  |  | 독립 의견 산출   |  | 독립 의견 산출  | |
  |  | (파일 쓰기 금지, |  | (파일 쓰기 금지,| |
  |  |  권한 상승 금지, |  |  세션 지속 금지,| |
  |  |  외부 부작용 금지)|  |  외부 부작용 금지)|
  |  +--------+--------+  +--------+-------+ |
  |           |                    |         |
  |           +--------+-----------+         |
  |                    |                     |
  |  [양측 모두 성공?]  |                     |
  |      YES           v  NO                 |
  |           +--------+--------+            |
  |           | OpenCode 합성    |            |
  |           | (양측 의견 통합) |            |
  |           +--------+--------+            |
  |                    |                     |
  |                    v                     |
  |           [panel 완료]                    |
  +-------------------------------------------+
                       |
                       v
  [2단계: Oracle (조건부)]
  +-------------------------------------------+
  |  Oracle 고려 조건 (모두 충족 시):          |
  |                                           |
  |  1. panel 완료 및 OpenCode 합성 완료       |
  |  2. 다음 중 하나:                          |
  |     a. 미해결 고위험 아키텍처/보안/성능/    |
  |        정확성 위험 존재                     |
  |     b. 반복적 구현 실패로 근본 원인 분석    |
  |        필요                                |
  |     c. 사용자 명시적 요청                  |
  |                                           |
  |  Oracle 특성:                             |
  |  - opencode-go/* 모델 사용 (기본)         |
  |  - read-only                               |
  |  - paid Zen opencode/* 모델은 L0 수동      |
  |    선택만 (Go 전체 장애 시, 자동 폴백 금지) |
  |  - free 모델 사용 금지                    |
  |  - worker escalation 경로 금지             |
  +-------------------------------------------+
```

### 5.2 advisor panel 규칙

`ask_advisor_panel`은 Oracle 이전에 필수이다. 패널은 Claude Code Fable 5와 Codex GPT-5.6-Sol의 독립 read-only 의견을 수집한다.

**필수 조건:**
- Fable 5 의견과 Codex 의견이 모두 성공해야 한다.
- OpenCode가 두 의견을 합성한다.
- 부분 성공 또는 실패는 패널을 차단(block)한다.

**advisor 제약:**
- 파일을 쓸 수 없다.
- 권한을 상승시킬 수 없다.
- 세션을 지속할 수 없다.
- 외부 부작용을 수행할 수 없다.

**현재 상태:** `ask_advisor_panel`은 현재 노출되거나 연결되어 있지 않다. 본 설계 작성 시점에 Fable 5와 Codex GPT-5.6-Sol에 대한 직접 CLI 의견이 본 설계의 방향을 형성하는 데 참고 자료로 사용되었으나, 이는 공식 advisor panel 게이트를 충족하지 않는다. 공식 패널이 성공하기 전까지 Oracle은 차단된다. 패널 호출 인터페이스 구축은 별도 작업이며, 본 설계 범위 밖이다.

### 5.3 Oracle 게이트 규칙

Oracle은 advisor panel 완료 및 OpenCode 합성 이후에만 고려한다. 차단된 패널은 Oracle로 승격할 수 없다.

Oracle 고려 조건 (다음 중 하나):
1. 미해결 고위험 아키텍처, 보안, 성능, 또는 정확성 위험
2. 반복적 구현 실패로 근본 원인 분석 필요
3. 사용자 명시적 요청

Oracle 모델 정책:
- 기본: `opencode-go/*` 모델
- paid Zen `opencode/*` 모델은 L0 수동 선택만. Go quota/rate-limit/provider 장애 시에만, 자동 폴백이 아닌 수동 재시도.
- free 모델 사용 금지
- Oracle 작업을 worker escalation 경로로 라우팅 금지

---

## 6. 공유 계약 프레임워크

### 6.1 계약 프레임워크 원칙

본 설계는 하나의 계약 프레임워크를 정의하고, 그 안에 profile별 독립 계약을 둔다. 각 profile은 자체 피처 집합, 라벨 공간, 임계값, 시맨틱 버전, 검증 규칙을 가진다. profile 간 피처 집합은 다를 수 있으며, 이 다름은 설계적 의도이다. profile별 계약을 하나의 범용 피처 목록으로 평탄화(flatten)하거나, 모든 출처 간 zero diff를 강제해서는 안 된다. Phase 2는 이 profile별 계약을 codify할 뿐, 피처를 재정렬하거나 병합하지 않는다.

### 6.2 profile별 계약 구조도

```
+=====================================================================+
|              Profile별 계약 프레임워크                                 |
+=====================================================================+

  +-------------------------------------------+
  |        SharedContractBase                 |
  |  (공통: 버전 상수, 검증 인터페이스)         |
  |  profile 간 평탄화 금지                    |
  +--------+----------+----------+------------+
           |          |          |
           v          v          v
  +--------+--+ +----+------+ +-+----------+ +------------+
  | tabular_   | | hierarchi | | ui_experi | | dl_history |
  | binary     | | cal       | | mental    | |            |
  +-----+------+ +-----+-----+ +-----+------+ +-----+------+
        |              |             |              |
        v              v             v              v
  +-----+------+ +-----+------+ +-----+------+ +-----+------+
  | 피처 목록   | | 피처 목록   | | UI 입력     | | bundle 파일 |
  | 임계값      | | 라벨 공간   | | 검증 규칙   | | lookback   |
  | 시맨틱 버전 | | 임계값      | | 학습 경로   | | 인코딩 전략 |
  | 검증 규칙   | | 시맨틱 버전 | | 안전장치    | | 검증 규칙   |
  +------------+ +------------+ +------------+ +------------+

  [공통 버전 상수 (변경 금지 - Phase 2 codify only)]
  +-------------------------------------------+
  | DDI_FEATURE_SEMANTICS_VERSION = "ddi.v2"  |
  | FEATURE_SEMANTICS_VERSION = "rulefeat.v1"  |
  | _FEATURE_SCHEMA_LENIENT_SUNSET = 2026-08-01|
  +-------------------------------------------+
```

### 6.3 profile별 계약 정의

#### 6.3.1 tabular_binary 계약

단일 ML 모델(XGBoost/LightGBM/Ensemble) 추론용 계약이다.

| 항목 | 내용 |
|---|---|
| 피처 출처 | `serving/predictor.py` `_BUILDER_KNOWN_COLS` |
| 피처 순서 | 모델 번들 `feature_names` 기준. `RequestFeatureBuilder.build()`가 정렬. |
| 임계값 | 모델 번들 `best_threshold`. `MLModel.classify()`가 Red/Yellow/Green/Normal 분류. |
| 시맨틱 버전 | `DDI_FEATURE_SEMANTICS_VERSION = "ddi.v2"`. 번들 메타 불일치 시 로드 거부. |
| 검증 | `_validate_feature_schema()`가 `feature_names ⊆ _FEATURE_ALLOWED` 검증. lenient는 sunset 전까지만. |
| 산출 경로 | `RequestFeatureBuilder.build()` -> `MLModel.predict_proba()` -> `MLModel.classify()` |
| 핫스왑 | `HybridPredictor.reload_model()` (스레드 안전) |

#### 6.3.2 hierarchical 계약

Stage 1 Red 이진 + Stage 2 Yellow 서브라벨 7-class 계층 분류기 계약이다.

| 항목 | 내용 |
|---|---|
| 피처 출처 | `stage_meta.json` `feature_cols` |
| 라벨 공간 | `STAGE2_LABELS = (Y_TRIPLE, Y_DOUBLE, Y_DDI_MAJOR, Y_DDI_MOD, Y_DUP, Y_FRAG, No_Alert)` |
| 라벨 정합 | 번들 메타 `stage2_labels`가 현재 `STAGE2_LABELS`와 정확히 일치해야 로드. 불일치 시 거부. |
| 인코더 정합 | `encoder.classes_`가 `STAGE2_LABELS`와 일치, `classes_present` 인덱스가 범위 내. |
| 임계값 | `stage_meta.json` `thresholds`: `tau_red`, `tau_review`. 2단 분기. |
| 시맨틱 버전 | `DDI_FEATURE_SEMANTICS_VERSION` + `FEATURE_SEMANTICS_VERSION`. 번들 메타 불일치 시 거부. |
| 검증 | `_validate_feature_schema()` + 라벨 공간 가드 + 인코더 정합 가드 + 해시 검증. |
| 산출 경로 | `RequestFeatureBuilder.build()` -> `HierarchicalPredictor.predict_risk_single()` -> `predict_risk()` |
| 개입 액션 | `ACTION_BY_LABEL`: Y_DDI_MAJOR=약사 전화, Y_TRIPLE=문자 안내, Y_DOUBLE/Y_DDI_MOD/Y_DUP/Y_FRAG=모니터링 |
| 핫스왑 | `HybridPredictor.reload_hierarchical()` (스레드 안전) |
| 백스톱 | `red_triggers()` (금기, RED_CONTRAINDICATED), `rule_floor()` (Y_DDI_MAJOR/Y_TRIPLE) |

#### 6.3.3 ui_experimental 계약

Page 3 Streamlit 학습 UI 경로 계약이다. 운영 모델이 아닌 실험/검증용이다.

| 항목 | 내용 |
|---|---|
| 피처 출처 | `hana_app/core/ml_runner.py` `FEATURE_COLS` |
| 학습 경로 | `ml_runner.py` -> `aggregate_patient_features()` -> `FeatureEngineer` -> trainer |
| 라벨 | `RISK_LABEL_MAP`: Red=3, Yellow=2, Green=1, Normal=0 |
| 검증 | UI 내 stratified sample, cross-validation, metrics 표시 |
| 안전장치 | `page_guards.py`, `memory_guard.py`로 메모리/시간 제한 |
| 운영 분리 | 운영 serving 번들과 직접 연결되지 않음. 별도 경로. |
| 위험 | `FEATURE_COLS`와 `_BUILDER_KNOWN_COLS` 간 불일치 가능. Phase 2A에서 명세하고 Phase 2B에서 특성화 테스트로 기록. |

#### 6.3.4 dl_history 계약

운영 DL bundle(그래프 신경망) 추론용 계약이다.

| 항목 | 내용 |
|---|---|
| bundle 필수 파일 | `model.pt`, `model_config.json`, `drug_vocab.json`, `edge_index.pt`, `feature_normalizer.pkl`, `schema_version.json` |
| manifest | `MANIFEST.json` (SHA-256 해시 검증) |
| 인코딩 전략 | `multi_hot`만 지원. `count`는 dead infra이므로 제거됨. |
| 그래프 아키텍처 | `gat`, `gcn` |
| lookback | `LOOKBACK_DAYS_DEFAULT=365`, min=7, max=1825. 런타임과 번들 불일치 시 `LookbackMismatchError`. |
| 검증 | `validate_dl_bundle_manifest()`, `validate_lookback_consistency()`, hash 검증 |
| 산출 경로 | `HANAHistoryProvider.fetch_patient_history()` -> `DLModel.predict()` |
| 핫스왑 | `HybridPredictor.reload_dl()` (스레드 안전) |
| 운영 영향 | 현재 최종 `risk_level` 결정에 반영되지 않음. 보조 결과만 반환. |

### 6.4 계약 버전 관리

각 계약은 시맨틱 버전 상수를 통해 번들 메타에 기록되고, 서빙 로드 시 가드가 현재 버전과 불일치/누락 번들을 거부한다. 이는 commit `d201743` train/serve 스큐 전례에 대한 구조적 방어이다. Phase 2는 이 상수값을 변경하지 않고 codify만 수행한다.

```
  [학습 시]                    [서빙 로드 시]
  +------------------+         +------------------+
  | 번들 메타에 버전  |         | 현재 코드 버전과 |
  | 스탬프:          |         | 비교:            |
  | ddi_feature_     |         | 불일치/누락 ->   |
  | semantics_version|         | 로드 거부        |
  | feature_semantics|         | (재학습 필요)    |
  | _version         |         +------------------+
  +------------------+
```

---

## 7. 단계별 설계

### 7.1 단계 개요

```
  Phase 0A              Phase 0B              Phase 1               Phase 2A              Phase 2B
  (정책/런타임          (실행 가능한          (최소 도구화)          (계약 명세)            (특성화 테스트 +
   상태 매핑)            계약 기준선)
  +--------+            +--------+            +--------+            +--------+            +--------+
  | 정책   |            | 계약   |            | pytest |            | profile|            | 동결   |
  | 런타임 |            | 기준선 |            | 기준/  |            | 별 계약|            | 동작   |
  | 상태   |            | 실행   |            | 발견   |            | 명세  |            | 기록   |
  | 매핑   |            |        |            | Ruff   |            | codify|            | 호환   |
  |        |            |        |            | check  |            |        |            | 어댑터 |
  +--------+            +--------+            +--------+            +--------+            +--------+
       |                     |                     |                     |                     |
       v                     v                     v                     v                     v
  [문서]                [문서+보고서]         [설정+체크]            [문서+명세]            [테스트+어댑터]
  (변경 없음)           (변경 없음)           (프로덕션 변경 없음)   (동작 변경 없음)      (동작 변경 없음)
```

Phase 3(predictor/domain 추출) 및 이후 모든 구현 작업은 본 설계 주기 밖의 후속 작업(future work)이다.

### 7.2 Phase 0A: 정책/런타임 상태 매핑

**목표:** 현행 코드의 정책 상태와 런타임 동작을 매핑한다. 코드 변경이 없다.

**작업:**
1. 각 profile(tabular_binary, hierarchical, ui_experimental, dl_history)의 피처 목록, 라벨 공간, 임계값, 시맨틱 버전, 검증 규칙을 문서화한다.
2. `_BUILDER_KNOWN_COLS`, `FEATURE_COLS`, `ETL_NUMERIC_COLS`, `ML_DATASET_REQUIRED_COLUMNS` 간 차이를 표로 정리한다. 이 차이는 설계적 의도일 수 있으므로, zero diff를 목표로 하지 않고 현재 상태를 있는 그대로 기록한다.
3. `STAGE2_LABELS`, `ACTION_BY_LABEL`, `INTERVENTION_MAP`의 현재 값을 기록한다.
4. `DDI_FEATURE_SEMANTICS_VERSION`, `FEATURE_SEMANTICS_VERSION`의 현재 값과 의미를 기록한다.
5. **배포된 번들의 피처 이름 검사**: 운영 모델 번들의 `feature_names`/`feature_cols`를 추출하여 현재 코드 상수와 비교한 상태를 기록한다.
6. **FEATURE_SCHEMA_LENIENT 런타임 환경/deadline 검사**: `FEATURE_SCHEMA_LENIENT` 환경 변수 현재 설정 여부, `FEATURE_SCHEMA_LENIENT_SUNSET_DATE` 환경 변수 설정 여부, 코드 default sunset(2026-08-01) 대비 현재 날짜의 관계를 기록한다.
7. **pickle/joblib 모듈 경로 검사**: 모델 번들 역직렬화 시 필요한 모듈 경로(pickle이 참조하는 클래스 경로)를 기록한다. 운영 환경과 dev 환경 간 모듈 경로 차이가 있는지 확인한다.
8. **피처 dtype/기본값 검사**: 각 피처의 dtype(float, bool 등)과 기본값(0.0, 0.5, False 등)을 기록한다. `sex_m`의 기본값 0.5, 위험약물 플래그의 기본값 False 등.
9. **필수 리소스 의미 검사**: DDI 매트릭스, CYP 매트릭스, 코드 표준화기, DrugMaster 등 서빙에 필요한 리소스 파일의 존재 여부와 로드 동작을 기록한다.
10. **reload/rollback 동작 검사**: `reload_model()`, `reload_hierarchical()`, `reload_dl()`의 현재 동작, 스레드 안전성, 실패 시 롤백 동작을 기록한다.
11. **물리적 DataFrame/Parquet 컬럼 순서 검사**: 학습 피처 Parquet 파일과 서빙 피처 DataFrame의 물리적 컬럼 순서를 기록한다. 논리적 피처 이름 집합이 같아도 물리적 순서가 다르면 스큐 위험이 있다.

**인수 기준:**
- 4개 profile별 계약 명세가 문서로 존재한다.
- 피처 목록 분산 차이표가 존재한다 (zero diff가 아닌 현재 상태 기록).
- 배포된 번들의 피처 이름, dtype, 기본값 상태가 기록된다.
- FEATURE_SCHEMA_LENIENT 런타임 환경/deadline 상태가 기록된다.
- pickle/joblib 모듈 경로 상태가 기록된다.
- 필수 리소스 의미 상태가 기록된다.
- reload/rollback 동작 상태가 기록된다.
- 물리적 DataFrame/Parquet 컬럼 순서가 기록된다.
- 코드 변경이 없다.

**담당:** Claude HQ (문서화, 논리 검토)

### 7.3 Phase 0B: 재현 가능한 계약 기준선 보고서

**목표:** Phase 0A에서 매핑한 상태를 재현 가능한 기준선(baseline) 보고서로 정리한다. 코드 변경이 없다. 새 검사 도구 구현은 Phase 1에서 다룬다.

**작업:**
1. Phase 0A에서 수집한 profile별 계약 상태를 재현 가능한 보고서로 정리한다. 기존 read-only 명령(python -c, grep, ast-grep 등)을 사용하여 현재 상태를 출력하고, 그 출력을 보고서에 기록한다. 새 스크립트를 작성하거나 커밋하지 않는다.
2. `serving/predictor.py`의 모든 import 문을 추출하여 의존성 그래프를 작성한다. 기존 도구(grep, ast-grep)를 사용한다.
3. `serving -> hana_app.core.hierarchical_runner` 의존성을 명시적으로 기록한다. `predict_risk`, `ACTION_BY_LABEL`, `STAGE2_LABELS`를 import 중임을 기록한다.
4. `serving -> scripts.etl.*`, `serving -> rules.*` 의존성을 기록한다.
5. 순환 의존성이 있는지 확인한다.
6. 배포된 번들에서 피처 이름, dtype, 기본값, 시맨틱 버전을 기존 도구로 추출한 결과를 보고서에 기록한다.

**산출물:** 의존성 그래프 문서 (ASCII) 및 재현 가능한 상태 기준선 보고서 (기존 read-only 명령 출력 기록)

```
  serving/predictor.py
  |
  +-- hana_app.core.hierarchical_runner
  |   +-- predict_risk, ACTION_BY_LABEL, STAGE2_LABELS
  |
  +-- scripts.etl.prescription_aggregator
  |   +-- count_ddi_severities, ddi_pair_severities
  |   +-- _fill_dup_features, detect_triple_whammy, detect_risk_drug
  |   +-- DDI_FEATURE_SEMANTICS_VERSION, FEATURE_SEMANTICS_VERSION
  |
  +-- scripts.etl.overlap_calculator
  |   +-- calculate_overlaps_for_patient, get_concurrent_drug_count
  |
  +-- scripts.etl.clinical_rules
  |   +-- collect_red_triggers, collect_severe_immediate_triggers
  |
  +-- scripts.etl.code_standardizer
  |   +-- CodeStandardizer
  |
  +-- scripts.etl.models
  |   +-- PrescriptionRecord, PatientFeatures
  |
  +-- scripts.features.cyp_features
  |   +-- CYPFeatureExtractor
  |
  +-- scripts.train.gat_trainer
  |   +-- GATTrainer (EnsembleTrainer3Way용)
  |
  +-- rules.safety_net
  |   +-- SafetyNet
  |
  +-- rules.duplicate_detector
  |   +-- DuplicateDetector
  |
  +-- rules.risk_drug_constants
      +-- HIGH_RISK_KEYWORDS, RENAL_RISK_KEYWORDS, HEPATIC_RISK_KEYWORDS
      +-- HIGH_RISK_ATC_PREFIXES, RENAL_RISK_ATC_PREFIXES, HEPATIC_RISK_ATC_PREFIXES
```

**인수 기준:**
- 의존성 그래프 문서가 존재한다.
- `serving -> hana_app` 의존성이 명시적으로 기록된다.
- 순환 의존성 여부가 확인된다.
- profile별 상태 기준선 보고서가 재현 가능하다 (기존 read-only 명령 출력 기록).
- 코드 변경이 없다. 새 스크립트를 작성하거나 커밋하지 않는다.

**담당:** Claude HQ (의존성 분석, 기준선 보고서 작성)

### 7.4 Phase 1: 최소 도구화 (minimal tooling only)

**목표:** 계약 위반을 자동 검출할 수 있는 최소한의 도구 구성을 추가한다. 프로덕션 코드 동작은 변경하지 않는다. 도구 권한(packaging metadata, build backend, editable install)을 변경하지 않는다. autofix를 수행하지 않는다.

**작업:**
1. **pytest 기준선/발견**: 현재 pytest 컬렉션과 패스 세트를 기준선으로 설정한다. Phase 1 이후 기존 테스트의 모든 node ID와 pass/fail 결과가 불변임을 확인한다. 명시적으로 승인된 새 Phase 1 테스트만 추가될 수 있다. marker config를 추가하여 계약 관련 테스트를 표시한다.
2. **Ruff check-only**: Ruff를 check-only 모드로 구성한다. autofix를 비활성화한다. 계약 관련 규칙(예: import 방향, 사용되지 않는 import)을 활성화하되, 위반을 보고만 하고 자동 수정하지 않는다.
3. **명시적 제외**: 도구 검사에서 제외할 경로를 명시적으로 설정한다. `packages_win/`, `mlruns/`, `out/`, `graphify-out/`, `.venv*` 등 보호 경로를 제외한다.
4. **의존성 제약 drift 검사**: `requirements.txt`/`constraints-py312.txt`와 실제 설치된 패키지 버전 간 drift를 검사하는 check-only 도구를 추가한다. Python 3.12 패리티 위반을 보고한다.
5. **FEATURE_SCHEMA_LENIENT sunset 모니터**: `FEATURE_SCHEMA_LENIENT` 환경 변수가 sunset(2026-08-01) 이후에 켜져 있는지 검사하는 check-only 도구를 추가한다.

**제약:**
- 프로덕션 코드(`serving/`, `hana_app/`, `scripts/`, `dags/`)의 동작을 변경하지 않는다.
- `pyproject.toml`의 `[project]`, dependencies, `[build-system]`/build backend, packaging authority, editable install 섹션은 변경하지 않는다.
- `pyproject.toml`의 `[tool.*]` 섹션만 추가/수정할 수 있다. 도구 구성 파일(`pytest.ini`, `ruff.toml`, `pyproject.toml`의 `[tool.*]` 섹션)만 추가/수정한다.
- autofix를 활성화하지 않는다.
- 기존 테스트의 모든 node ID와 pass/fail 결과가 불변이다. 명시적으로 승인된 새 Phase 1 테스트만 추가될 수 있다.

**인수 기준:**
- pytest 기준선이 설정되고, Phase 1 적용 후 기존 테스트의 모든 node ID와 pass/fail 결과가 불변이다. 명시적으로 승인된 새 Phase 1 테스트만 추가될 수 있다.
- Ruff check-only가 계약 관련 위반을 보고한다 (autofix 없음).
- 명시적 제외 경로가 설정된다.
- 의존성 제약 drift 검사가 Python 3.12 패리티 위반을 보고한다.
- sunset 모니터가 2026-08-01 이후 lenient 활성을 경고한다.
- 프로덕션 코드 동작 변경 없음.
- 기존 테스트의 모든 node ID와 pass/fail 결과가 불변이다. 명시적으로 승인된 새 Phase 1 테스트만 추가될 수 있다.

**담당:** Codex HQ (도구 구성, TDD)

### 7.5 Phase 2A: 계약 명세 (contract specification)

**목표:** Phase 0A/0B에서 매핑한 상태를 바탕으로, profile별 계약을 공식 명세로 codify한다. 기존 동작을 동결(freeze)하고 문서화할 뿐, 코드 구조를 변경하지 않는다.

**작업:**
1. 각 profile별 계약을 공식 명세 문서로 작성한다. 피처 목록, 라벨 공간, 임계값, 시맨틱 버전, 검증 규칙, dtype, 기본값, 물리적 컬럼 순서를 포함한다.
2. profile 간 피처 집합 차이를 명시적으로 기록한다. 이 차이를 제거하거나 평탄화하지 않는다. 각 profile이 다른 피처 집합을 가질 수 있음을 명세에 기록한다.
3. `_BUILDER_KNOWN_COLS`, `FEATURE_COLS`, `ETL_NUMERIC_COLS`, `ML_DATASET_REQUIRED_COLUMNS` 간 차이를 공식 표로 작성한다. 차이를 제거하지 않고, 현재 상태를 있는 그대로 명세한다.
4. `serving -> hana_app.core.hierarchical_runner` 의존성을 명세에 기록한다. 이 의존성을 제거하지 않는다. 의존성 제거는 Phase 3 (out of scope)에서 수행한다.
5. `predict_risk`, `ACTION_BY_LABEL`, `STAGE2_LABELS`를 serving 내부로 복사하거나 이동하지 않는다. 향후 Phase 3에서 순수 도메인 정책을 중립 공유 모듈로 이동하되, 호환성 import를 필요에 따라 보존한다는 방침을 명세에 기록한다.

**제약:**
- 피처 이름, 순서, 기본값 의미를 변경하지 않는다.
- `FEATURE_COLS`를 `_BUILDER_KNOWN_COLS`에 병합하지 않는다.
- 라벨 정의(Red/Yellow/Green/Normal, Yellow subtype)를 변경하지 않는다.
- 시맨틱 버전 상수(`DDI_FEATURE_SEMANTICS_VERSION`, `FEATURE_SEMANTICS_VERSION`)를 변경하지 않는다.
- 모델 번들 형식(pickle, joblib, stage_meta.json)을 변경하지 않는다.
- 아티팩트를 마이그레이션하지 않는다.
- 재학습하지 않는다.
- predictor.py를 분할하지 않는다.
- `predict_risk`, `ACTION_BY_LABEL`, `STAGE2_LABELS`를 복사/이동하지 않는다.
- 기존 API 엔드포인트 응답 형식을 변경하지 않는다.
- 프로덕션 코드 동작을 변경하지 않는다.

**인수 기준:**
- 4개 profile별 공식 계약 명세가 문서로 존재한다.
- profile 간 피처 집합 차이가 명시적으로 기록된다 (평탄화하지 않음).
- 의존성 그래프가 명세에 포함된다.
- Phase 3 방침(중립 공유 모듈 이동, 호환성 import 보존)이 명세에 기록된다.
- 코드 동작 변경 없음.
- 기존 테스트 전체 통과.

**담당:** Claude HQ (명세 작성, 논리 검토)

### 7.6 Phase 2B: 특성화 테스트 및 호환성 어댑터 (characterization tests and compatibility adapters)

**목표:** Phase 2A에서 명세한 계약의 현재 동작을 특성화 테스트(characterization test)로 기록하여, 향후 구조 변경(Phase 3) 시 동결이 보장되도록 한다. profile 간 차이를 수용하는 호환성 어댑터를 추가한다.

**작업:**
1. **특성화 테스트 작성**: 각 profile별로 현재 동작을 기록하는 회귀 테스트를 작성한다. 이 테스트는 "현재 동작이 무엇인지"를 기록하며, "무엇이어야 하는지"를 단정하지 않는다.
   - dtype/기본값 특성화: 각 피처의 dtype과 기본값을 기록하는 테스트
   - 리소스 의미 특성화: DDI 매트릭스, CYP 매트릭스, DrugMaster 등 리소스 부재 시 fallback 동작을 기록하는 테스트
   - 요청 변형 특성화: `PredictRequest`의 약물 목록 변형(빈 목록, 미매핑 EDI, 단일 약물 등)에 대한 현재 응답을 기록하는 테스트
   - reload/rollback 특성화: `reload_model()`, `reload_hierarchical()`, `reload_dl()`의 성공/실패 시 현재 동작을 기록하는 테스트
2. **호환성 어댑터**: profile 간 피처 집합 차이를 수용하는 읽기 전용 어댑터를 추가한다. 이 어댑터는 차이를 보고할 뿐, 차이를 제거하지 않는다. 각 profile이 자체 피처 집합을 유지하도록 보장한다.
3. **물리적 컬럼 순서 특성화**: 학습 피처 Parquet과 서빙 피처 DataFrame의 물리적 컬럼 순서를 기록하는 테스트를 작성한다.

**제약:**
- 피처 이름, 순서, 기본값 의미를 변경하지 않는다.
- `FEATURE_COLS`를 `_BUILDER_KNOWN_COLS`에 병합하지 않는다.
- 라벨 정의를 변경하지 않는다.
- 시맨틱 버전 상수를 변경하지 않는다.
- 모델 번들 형식을 변경하지 않는다.
- 아티팩트를 마이그레이션하지 않는다.
- 재학습하지 않는다.
- predictor.py를 분할하지 않는다.
- 프로덕션 코드 동작을 변경하지 않는다. 테스트와 읽기 전용 어댑터만 추가한다.
- Nov->Dec 홀드아웃 튜닝 금지. freeze-safe 작업만 수행.

**인수 기준:**
- 각 profile별 특성화 테스트가 존재한다.
- dtype/기본값, 리소스 의미, 요청 변형, reload/rollback 동작이 기록된다.
- 물리적 컬럼 순서가 기록된다.
- 호환성 어댑터가 profile 간 차이를 보고한다 (차이를 제거하지 않음).
- 기존 테스트의 모든 node ID와 pass/fail 결과가 불변이다. 새 특성화 테스트만 추가된다 (컬렉션은 증가).
- 프로덕션 코드 동작 변경 없음.

**담당:** Codex HQ (테스트 구현, TDD), Claude HQ (논리 검토)

### 7.7 단계별 의존성 및 순서

```
  Phase 0A (상태 매핑)
       |
       v
  Phase 0B (계약 기준선 보고서) -- Phase 1 (최소 도구화) -- Phase 2A (계약 명세)
                                                                     |
                                                                     v
                                                  Phase 2B (특성화 테스트)
                                                                     |
                                                                     v
                                                           [동결 동작 기록 완료]
                                                                     |
                                                                     v
                                                  [Phase 3 이후: future work, 본 설계 주기 밖]
```

Phase 0A는 0B보다 반드시 선행한다. 0B는 0A의 상태 매핑 결과를 소비한다. Phase 1은 0B 완료 후 수행한다. Phase 2A는 2B보다 반드시 선행한다. 2B는 2A의 계약 명세를 기반으로 특성화 테스트를 작성한다.

### 7.8 Phase 3 및 이후: future work (본 설계 주기 밖)

Phase 3 이후 작업은 본 설계 범위 밖이다. 참고로, 향후 고려할 수 있는 작업은 다음과 같다:

- **predictor/domain 추출**: `serving/predictor.py`의 책임을 분리하고 `hana_app.core` 의존성을 제거. 순수 도메인 정책(`predict_risk`, `ACTION_BY_LABEL`, `STAGE2_LABELS`)을 중립 공유 모듈로 이동하되, 호환성 import를 보존.
- **광역 엔진 통합**: tabular_binary와 hierarchical를 하나의 추론 엔진으로 통합. Fable 5와 Codex 양측이 NO-GO로 합의.
- **`scripts/ops/` 재배치**: 재사용 코드와 명령 스크립트 분리.
- **문서 정합**: `data_pipeline_architecture.md`의 비현실적 내용(Spark, HDFS 등)을 운영 현실(Windows 폐쇄망 Python 3.12)에 맞게 수정.
- **HANA 코드 리팩터링**: HANA 추출/ETL 코드 구조 개선. 문서 정합과는 별개의 작업이다.

이 작업들은 본 설계가 완료된 후 별도 설계 주기에서 다룬다.

---

## 8. 데이터 흐름 및 제어 흐름

### 8.1 추론 데이터 흐름 (serving)

```
  [요청] PredictRequest (환자 약물 목록)
    |
    v
  +---------------------------+
  | RequestFeatureBuilder     |
  | .build()                  |
  |                           |
  | 1. EDI -> ATC 보완        |
  |    (CodeStandardizer)     |
  | 2. DDI 카운트 산출        |
  |    (edi->wk->overlap)     |
  | 3. 중복약물 카운트        |
  |    (edi->wk->DrugMaster)  |
  | 4. CYP 피처 추출          |
  | 5. 위험약물 플래그        |
  |    (rulefeat.v1:          |
  |     edi->wk->components)  |
  | 6. 피처 벡터 정렬          |
  |    (feature_names 순서)   |
  +-----------+---------------+
              |
              v
  +-----------+---------------+
  | [profile 분기]             |
  |                           |
  | hierarchical?             |
  |   YES -> HierarchicalPred|  -> predict_risk_single()
  |   NO  -> MLModel          |  -> predict_proba() + classify()
  +-----------+---------------+
              |
              v
  +-----------+---------------+
  | Rule Safety Net           |
  | (SafetyNet + DupDetector) |
  | -> rule_level, ddi_alerts |
  +-----------+---------------+
              |
              v
  +-----------+---------------+
  | 결정적 백스톱              |
  | red_triggers()            |
  |   -> RED_CONTRAINDICATED  |
  | rule_floor()               |
  |   -> Y_DDI_MAJOR/Y_TRIPLE |
  +-----------+---------------+
              |
              v
  +-----------+---------------+
  | 최종 등급                  |
  | = max(Rule, ML, 백스톱)   |
  +-----------+---------------+
              |
              v
  [응답] PredictResponse
    (risk_level, rule_level, ml_level, ml_probability,
     yellow_subtype, stage2_probs, red_suspect, action,
     ddi_alerts, risk_reasons, intervention)
```

### 8.2 학습 데이터 흐름

```
  [Page 3 UI 경로]                    [Airflow DAG 경로]
  +-------------------+              +-------------------+
  | hana_app/core/    |              | dags/ddi_train_   |
  | ml_runner.py      |              | dag.py            |
  |                   |              |                   |
  | FEATURE_COLS      |              | config.settings   |
  | (ml_runner)       |              | -> FEATURES_DIR   |
  |                   |              |                   |
  | aggregate_patient |              | load_dataset()    |
  | _features()       |              | -> TrainDataset   |
  | -> PatientFeatures|              |                   |
  | -> FeatureEngineer|              | TrainPipeline     |
  | -> trainer        |              | -> build_trainer()|
  | -> joblib 저장    |              | -> model.pkl 저장 |
  +--------+----------+              +--------+----------+
           |                                  |
           | [Phase 2A: 차이 명세]            |
           | [Phase 2B: 특성화 테스트]        |
           +----------------+-----------------+
                            |
                            v
                   +--------+----------+
                   | profile별 계약    |
                   | 명세 (Phase 2A)   |
                   | 차이 보존 (평탄화  |
                   | 금지)              |
                   +-------------------+
```

### 8.3 제어 흐름: OpenCode LO 작업 라우팅

```
  [사용자 요청]
    |
    v
  +---------------------------+
  | OpenCode LO: 작업 분류    |
  |                           |
  | 1. 성격 판단              |
  |    - 구현? -> Codex       |
  |    - 검토? -> Claude      |
  |    - 환경? -> AGY         |
  |    - 병렬/대량? -> AGY    |
  |                           |
  | 2. critical 여부          |
  |    - 라벨/스키마/HANA/    |
  |      동결 정책 ->         |
  |      cross-family 필수    |
  |                           |
  | 3. 고위험 여부            |
  |    - advisor panel 필요? |
  |    - Oracle 필요?        |
  +-----------+---------------+
              |
              v
  +-----------+---------------+
  | worker 라우팅              |
  | (역할 + 모델 배정)         |
  +-----------+---------------+
              |
              v
  +-----------+---------------+
  | worker 실행               |
  | (독립 세션)               |
  | -> 증거 반환              |
  +-----------+---------------+
              |
              v
  +-----------+---------------+
  | OpenCode LO: 검증         |
  |                           |
  | 1. 변경 파일 확인         |
  | 2. 테스트/타입체크/빌드  |
  | 3. 결과 판정              |
  +-----------+---------------+
              |
              v
  +-----------+---------------+
  | [critical 변경?]          |
  |   YES -> cross-family     |
  |         검토 필수         |
  |   NO  -> OpenCode 승인    |
  +-----------+---------------+
              |
              v
  [사용자 보고]
```

---

## 9. FEATURE_SCHEMA_LENIENT 위험 (2026-08-01)

### 9.1 현황

`FEATURE_SCHEMA_LENIENT=1` 환경 변수는 학습 모델이 `RequestFeatureBuilder` 미산출 컬럼을 사용할 때, silent 0.0 fallback으로 degraded 로드를 허용하는 escape hatch이다. 코드 default sunset 날짜는 `2026-08-01`이다(`serving/predictor.py`).

sunset 이후:
- `FEATURE_SCHEMA_LENIENT=1`이 설정되어 있어도 lenient가 차단된다.
- strict 강제: 미허용 컬럼 발견 시 모델 로드 거부.
- `FEATURE_SCHEMA_LENIENT_SUNSET_DATE` 환경 변수로 sunset 날짜를 override할 수 있으나, 잘못된 형식이면 안전 측(lenient 차단)으로 동작한다.

### 9.2 위험 시나리오

```
  2026-08-01 이전:
  +---------------------------+
  | FEATURE_SCHEMA_LENIENT=1  |
  | -> lenient 허용           |
  | -> unknown 컬럼 0.0 fallback|
  | -> degraded 로드 (warning)|
  +---------------------------+

  2026-08-01 이후:
  +---------------------------+
  | FEATURE_SCHEMA_LENIENT=1  |
  | -> lenient 차단 (sunset)  |
  | -> strict 강제            |
  | -> unknown 컬럼 = 로드 거부|
  +---------------------------+
              |
              v
  [위험] 운영 모델이 unknown 컬럼을
         사용 중이면 2026-08-01 이후
         서버 시작 시 모델 로드 실패
         -> 서비스 중단
```

### 9.3 완화 방안

1. **Phase 0A 상태 매핑**: 배포된 번들의 `feature_names`가 `_FEATURE_ALLOWED`의 부분집합인지, `FEATURE_SCHEMA_LENIENT` 환경 변수 설정 여부, sunset deadline 대비 현재 날짜의 관계를 기록한다.
2. **Phase 1 sunset 모니터**: 2026-08-01 이후 `FEATURE_SCHEMA_LENIENT` 환경 변수가 켜져 있으면 경고하는 check-only 도구를 추가한다.
3. **Health endpoint 가시성**: `/health` 응답의 `schema_drift`, `feature_schema_lenient`, `feature_schema_lenient_allowed`, `feature_schema_lenient_sunset_date` 필드로 운영 상태를 모니터링한다.
4. **sunset 날짜 환경 변수**: 필요 시 `FEATURE_SCHEMA_LENIENT_SUNSET_DATE`로 sunset을 연장할 수 있으나, 근본 해결을 지연시키는 용도로 사용해서는 안 된다.

### 9.4 본 설계와의 관계

본 설계의 Phase 2는 `FEATURE_SCHEMA_LENIENT` 환경 변수를 제거하지 않는다. Phase 2A에서 현재 동작을 명세하고, Phase 2B에서 특성화 테스트로 기록할 뿐, 코드를 변경하지 않는다. 환경 변수 제거나 strict 강제 코드 단순화는 Phase 3 이후 별도 PR에서 다룬다.

---

## 10. 인수 기준

### 10.1 전체 인수 기준

| 기준 | 검증 방법 | 적용 단계 |
|---|---|---|
| OpenCode가 최종 LO로 문서화됨 | 본 설계 문서 존재 | Phase 0A |
| advisor panel 게이트가 정의됨 | 본 설계 문서 5장 존재 | Phase 0A |
| 4개 profile별 계약이 정의됨 | 본 설계 문서 6장 + 별도 명세 | Phase 0A |
| profile 간 피처 집합 차이가 평탄화되지 않음 | 명세에 차이 명시적 기록 | Phase 0A, 2A |
| 배포된 번들 피처 이름/dtype/기본값 상태 기록 | 상태 매핑 문서 | Phase 0A |
| FEATURE_SCHEMA_LENIENT 런타임 상태 기록 | 상태 매핑 문서 | Phase 0A |
| pickle/joblib 모듈 경로 상태 기록 | 상태 매핑 문서 | Phase 0A |
| reload/rollback 동작 상태 기록 | 상태 매핑 문서 | Phase 0A |
| 물리적 DataFrame/Parquet 컬럼 순서 기록 | 상태 매핑 문서 | Phase 0A |
| 의존성 그래프가 작성됨 | 의존성 매핑 문서 | Phase 0B |
| pytest 기준선 설정, 기존 node ID/pass/fail 불변 (승인된 새 테스트만 추가) | pytest 실행 비교 | Phase 1 |
| Ruff check-only가 위반을 보고 (autofix 없음) | Ruff 실행 | Phase 1 |
| 의존성 제약 drift 검사가 동작함 | drift 검사 도구 실행 | Phase 1 |
| sunset 모니터가 동작함 | 모니터 도구 실행 | Phase 1 |
| 프로덕션 코드 동작 변경 없음 | 테스트 전체 통과 | Phase 1, 2A, 2B |
| profile별 공식 계약 명세 존재 | 명세 문서 | Phase 2A |
| Phase 3 방침(중립 공유 모듈, 호환성 import) 명세 | 명세 문서 | Phase 2A |
| profile별 특성화 테스트 존재 | 테스트 실행 | Phase 2B |
| 호환성 어댑터가 차이를 보고 (제거하지 않음) | 어댑터 실행 | Phase 2B |
| 기존 테스트 전체 통과 | pytest 실행 | Phase 1, 2A, 2B |
| 기존 테스트 node ID/pass/fail 불변 (1, 2B는 승인된 새 테스트만 추가) | pytest 실행 비교 | Phase 1, 2B |

### 10.2 단계별 게이트

각 Phase는 다음 Phase로 진입하기 전에 OpenCode LO의 명시적 승인이 필요하다. 승인 조건은 해당 Phase의 인수 기준이 모두 충족되어야 한다.

---

## 11. 테스트 계획

### 11.1 Phase 1 테스트

| 테스트 | 대상 | 기대 결과 |
|---|---|---|
| pytest 기존 node ID/pass/fail 불변 | 전체 테스트 | Phase 1 적용 전후 기존 node ID와 pass/fail 결과 불변, 승인된 새 테스트만 추가 |
| Ruff check-only | 계약 관련 규칙 | 위반 보고 (autofix 없음) |
| 의존성 제약 drift 검사 | requirements/constraints vs 설치 패키지 | Python 3.12 패리티 위반 보고 |
| sunset 모니터 | 2026-08-01 전후 | sunset 이후 경고 출력 |

### 11.2 Phase 2A 테스트

| 테스트 | 대상 | 기대 결과 |
|---|---|---|
| 기존 테스트 전체 통과 | 전체 테스트 | 전체 통과 (코드 변경 없음) |
| 프로덕션 코드 동작 변경 없음 | serving, hana_app, scripts | 동일한 API 응답, 동일한 모델 로드 |

### 11.3 Phase 2B 테스트

| 테스트 | 대상 | 기대 결과 |
|---|---|---|
| dtype/기본값 특성화 | 각 피처의 dtype과 기본값 | 현재 값 기록 |
| 리소스 의미 특성화 | DDI/CYP/DrugMaster 부재 시 fallback | 현재 fallback 동작 기록 |
| 요청 변형 특성화 | 빈 목록, 미매핑 EDI, 단일 약물 | 현재 응답 기록 |
| reload/rollback 특성화 | reload 성공/실패 | 현재 동작 기록 |
| 물리적 컬럼 순서 특성화 | 학습 Parquet vs 서빙 DataFrame | 현재 순서 기록 |
| 호환성 어댑터 | profile 간 차이 | 차이 보고 (제거하지 않음) |
| 기존 테스트 node ID/pass/fail 불변 | 전체 테스트 | 기존 node ID와 pass/fail 결과 불변, 새 테스트만 추가 |

### 11.4 회귀 테스트 범위

모든 Phase에서 다음 회귀 테스트가 통과해야 한다:
- `tests/test_serving/test_feature_schema_strict.py`: 피처 스키마 strict 검증
- `tests/test_serving/test_feature_contract.py`: 피처 계약 검증
- `tests/test_serving/test_predict.py`: 추론 엔드투엔드
- `tests/test_features/`: 피처 산출 정합
- `tests/test_hana_app/`: Page 3 학습 UI

Phase 1은 기존 pytest 테스트의 모든 node ID와 pass/fail 결과가 불변임을 확인한다. 명시적으로 승인된 새 Phase 1 테스트만 추가될 수 있다. Phase 2B는 새 특성화 테스트를 추가하므로 컬렉션이 증가할 수 있으나, 기존 테스트의 모든 node ID와 pass/fail 결과는 불변이어야 한다. Phase 2B는 구조 변경 전에 dtype/기본값/리소스/fallback/요청 변형/reload/rollback 동작을 특성화 테스트로 기록한다.

---

## 12. 위험 및 비목표

### 12.1 위험

| 위험 | 확률 | 영향 | 완화 |
|---|---|---|---|
| 2026-08-01 sunset 이후 운영 모델 로드 실패 | 중 | 높음 | Phase 0A에서 배포 번들 상태 매핑. Phase 1 sunset 모니터로 사전 경고. |
| Phase 2B 특성화 테스트가 누락된 동작을 기록하지 못함 | 중 | 중간 | Phase 0A 상태 매핑 항목(피처 이름, dtype, 기본값, 리소스, reload/rollback, 컬럼 순서)을 테스트 체크리스트로 사용. |
| profile 간 차이를 실수로 평탄화 | 낮음 | 높음 | Phase 2A 명세에서 차이를 명시적으로 기록. Phase 2B 어댑터가 차이를 보고. 코드 변경 없음. |
| `ask_advisor_panel` 미구축으로 Oracle 차단 | 높음 | 중간 | 본 설계는 패널 없이 진행. Oracle이 필요한 고위험 결정은 패널 구축 후로 연기. |
| pickle/joblib 모듈 경로가 환경 간 다름 | 중 | 중간 | Phase 0A에서 모듈 경로 매핑. Phase 2B에서 특성화 테스트로 기록. |

### 12.2 비목표

본 설계가 명시적으로 다루지 않는 범위:

1. **Phase 3 predictor/domain 추출**: `serving/predictor.py` 분할, `hana_app.core` 의존성 제거, 순수 도메인 정책 중립 공유 모듈 이동은 본 설계 주기 밖의 future work이다. 향후 Phase 3에서 순수 도메인 정책을 중립 공유 모듈로 이동하되, 호환성 import를 필요에 따라 보존한다. `predict_risk`, `ACTION_BY_LABEL`, `STAGE2_LABELS`를 serving 내부로 복사하지 않는다.
2. **광역 엔진 통합**: tabular_binary와 hierarchical를 하나의 추론 엔진으로 통합하는 작업은 본 설계 범위 밖이다. Fable 5와 Codex 양측이 NO-GO로 합의했다.
3. **라벨 정의 변경**: Red/Yellow/Green/Normal 라벨 조건, Yellow subtype 정의는 변경하지 않는다. Phase 2는 기존 정의를 문서화할 뿐이다.
4. **피처/라벨/버전 변경**: Phase 2는 기존 profile별 계약을 codify할 뿐, 새 피처 추가, 라벨 변경, 버전 변경, 피처 재정렬, `FEATURE_COLS`를 `_BUILDER_KNOWN_COLS`에 병합을 수행하지 않는다.
5. **아티팩트 마이그레이션/재학습**: 모델 번들 마이그레이션, 재학습을 수행하지 않는다.
6. **predictor 분할**: `predictor.py`를 분할하지 않는다. Phase 3 (out of scope)에서 다룬다.
7. **Nov->Dec 홀드아웃 튜닝**: 연구 동결 트랙 위반 작업은 금지한다. `RESEARCH_TRACK_FROZEN`.
8. **Gate 5A/5B 활성화**: 공식 취소/폐기된 게이트를 활성화하지 않는다.
9. **2025-01 데이터 확보**: 최종 데이터셋은 2024-07~12 184개 일별 파일이며, 추가 월 확보 계획이 없다.
10. **`data_pipeline_architecture.md` 재작성**: 현행 아키텍처 문서의 비현실적 내용 수정은 별도 문서 정합 작업이다. HANA 코드 리팩터링과는 별개이다.
11. **`scripts/ops/` 재배치**: 재사용 코드와 명령 스크립트 분리는 후속 작업이다.
12. **`FEATURE_SCHEMA_LENIENT` 환경 변수 제거**: Phase 3 이후 별도 PR에서 다룬다.
13. **Hermes 사용**: 본 프로젝트에서는 Hermes를 사용하지 않는다. 별도 지시가 있을 때까지 suspended이다.
14. **packaging metadata/build backend/editable install 변경**: Phase 1은 도구 구성만 다루고, packaging metadata를 변경하지 않는다.

### 12.3 동결 안전 선언

본 설계의 모든 Phase는 freeze-safe 작업이다. Nov->Dec 홀드아웃 데이터셋을 대상으로 하는 모델, 피처, 하이퍼파라미터 튜닝, ablation을 수행하지 않는다. 기존 라벨 정의, 피처 의미, 모델 번들 형식을 변경하지 않으므로 동결 트랙 위반이 없다.

### 12.4 보호 경로

본 설계의 모든 Phase는 다음 보호 경로를 편집/삭제/커밋하지 않는다:
- `packages_win/py312/` (Windows 오프라인 휠)
- `mlruns/` (MLflow 실험)
- 생성된 `.parquet` 파일
- `out/` 산출물
- `graphify-out/`

---

## 13. 롤아웃 및 롤백

### 13.1 롤아웃 계획

```
  [Phase 0A/0B: 문서+보고서]
  +-------------------+
  | 코드 변경 없음    |
  | -> 즉시 적용      |
  | -> 롤백 불필요    |
  +-------------------+
           |
           v
  [Phase 1: 도구 구성]
  +-------------------+
  | 도구 구성만 추가  |
  | 프로덕션 동작 변경 |
  | 없음              |
  | -> 즉시 적용       |
  | -> 롤백: 도구 구성 |
  |    되돌리기        |
  +-------------------+
           |
           v
  [Phase 2A: 계약 명세]
  +-------------------+
  | 문서만 추가       |
  | 코드 동작 변경    |
  | 없음              |
  | -> 즉시 적용       |
  | -> 롤백: 문서 삭제 |
  +-------------------+
           |
           v
  [Phase 2B: 특성화 테스트]
  +-------------------+
  | 테스트+어댑터만   |
  | 추가              |
  | 프로덕션 동작 변경 |
  | 없음              |
  | -> 즉시 적용       |
  | -> 롤백: 테스트/   |
  |    어댑터 삭제     |
  +-------------------+
```

### 13.2 롤백 계획

| Phase | 롤백 방법 | 롤백 영향 |
|---|---|---|
| 0A/0B | 문서/보고서 삭제 | 없음 (코드 변경 없음) |
| 1 | 도구 구성 파일 되돌리기 | 없음 (프로덕션 동작 변경 없음) |
| 2A | 명세 문서 삭제 | 없음 (코드 동작 변경 없음) |
| 2B | 테스트/어댑터 파일 삭제 | 없음 (프로덕션 동작 변경 없음) |

### 13.3 롤백 트리거

다음 상황 발생 시 즉시 롤백한다:

1. Phase 1 적용 후 기존 테스트 node ID 또는 pass/fail 결과 변화 (승인되지 않은 새 테스트 추가 포함)
2. Phase 1 적용 후 프로덕션 코드 동작 변경 발견
3. Phase 2B 적용 후 기존 테스트 실패
4. 운영 환경(Windows 폐쇄망)에서 Python 3.12 호환성 실패

### 13.4 롤백 검증

롤백 후 다음을 확인한다:
- 기존 테스트 전체 통과
- API 응답 형식 정상
- 모델 번들 로드 성공
- 운영 환경 정상 동작

---

## 14. 참조

### 14.1 권위 출처

본 설계의 권위 출처는 다음과 같다. 본 설계와 출처가 충돌하면 출처가 우선한다.

| 출처 | 경로 | 역할 |
|---|---|---|
| 프로젝트 CLAUDE.md | `CLAUDE.md` | 프로젝트 차이 정의, 환경, 데이터, 다중 AI 협업 |
| AGENTS.md | `AGENTS.md` | 에이전트 규칙, hard gates, 트리거, 역할, 통신 |
| L0 오케스트레이션 템플릿 | `~/.config/opencode/templates/AGENTS.l0-orchestration.md` | OpenCode LO 운영 규칙, advisor panel, Oracle 게이트 |

### 14.2 관련 코드

| 코드 | 경로 | 관련성 |
|---|---|---|
| predictor.py | `serving/predictor.py` | Phase 0A 상태 매핑 대상, Phase 3 분할 대상 (out of scope) |
| ml_runner.py | `hana_app/core/ml_runner.py` | Phase 0A 상태 매핑 대상 (FEATURE_COLS) |
| feature_engineer.py | `scripts/features/feature_engineer.py` | Phase 0A 상태 매핑 대상 (ETL_NUMERIC_COLS) |
| contracts.py | `scripts/datasets/contracts.py` | Phase 0A 상태 매핑 대상 (ML_DATASET_REQUIRED_COLUMNS) |
| hierarchical_runner.py | `hana_app/core/hierarchical_runner.py` | Phase 0B 의존성 매핑 대상, Phase 3 이동 대상 (out of scope) |
| prescription_aggregator.py | `scripts/etl/prescription_aggregator.py` | 시맨틱 버전 상수 출처 |

### 14.3 관련 기존 설계

| 설계 | 경로 | 관련성 |
|---|---|---|
| Yellow 세분화 + 계층 분류 설계 | `docs/superpowers/specs/2026-04-17-yellow-subtype-hierarchical-design.md` | hierarchical 계약의 라벨 정의 출처 |
| 구조 개선 설계 | `docs/superpowers/specs/2026-04-03-structural-improvements-design.md` | 선행 구조 개선 |

---

## 15. 검토 이력

| 단계 | 검토자 | 결과 | 비고 |
|---|---|---|---|
| 초안 작성 | OpenCode LO | 작성 완료 | 2026-07-12 |
| 직접 CLI 의견: Fable 5 | Claude Code Fable 5 | GO (수정안 포함) | Phase 1 tooling-only, Phase 2 codify-only, Phase 4 NO-GO. 본 의견은 참고 자료이며 공식 advisor panel 게이트를 충족하지 않음. |
| 직접 CLI 의견: Codex | Codex GPT-5.6-Sol | GO (수정안 포함) | 동일 수정안. 본 의견은 참고 자료이며 공식 advisor panel 게이트를 충족하지 않음. |
| OpenCode 합성 | OpenCode LO | 문서 작성 | 양측 수정안 반영하여 본 문서 작성. 사용자 승인됨. |
| 사용자 문서 검토 | 사용자 | 승인 | 2026-07-12. Phase 1 테스트 컬렉션 모순 정정 포함. |

본 문서는 Fable 5와 Codex GPT-5.6-Sol에 대한 직접 CLI 의견을 참고하여 작성되었다. 이 직접 CLI 의견은 본 설계의 방향을 형성하는 데 참고 자료로 사용되었으나, 공식 `ask_advisor_panel` 게이트를 충족하지 않는다. `ask_advisor_panel`은 현재 노출되거나 연결되어 있지 않으며, 공식 패널이 성공하기 전까지 Oracle은 차단된다. 본 문서는 사용자가 검토하고 승인했다.

Phase 1은 최소 도구화(tooling-only)만, Phase 2는 기존 profile별 계약을 codify/freeze만 수행하고 피처/라벨/버전 변경, 피처 재정렬, 병합, 아티팩트 마이그레이션, 재학습, predictor 분할을 수행하지 않는다. Phase 3 predictor/domain 추출 및 이후 모든 구현 작업은 본 설계 주기 밖의 future work이다. 광역 엔진 통합은 NO-GO이다. OpenCode가 모든 최종 결정과 검증을 소유한다.