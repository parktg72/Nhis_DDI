# 국민건강보험공단 다재약물 위험도 분류 AI 모델 - 종합 프로젝트 계획서

**문서번호**: NHIS-POLY-PLAN-2026-001
**버전**: 1.0
**작성일**: 2026-03-05
**작성**: 약물전문가, 데이터엔지니어, 모델연구원, MLOps엔지니어, QA검증전문가 (팀 합의)
**상태**: 팀 내부 합의 완료

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [위험도 분류 체계](#2-위험도-분류-체계)
3. [데이터 아키텍처 및 파이프라인](#3-데이터-아키텍처-및-파이프라인)
4. [모델 개발 전략](#4-모델-개발-전략)
5. [인프라 및 MLOps 설계](#5-인프라-및-mlops-설계)
6. [품질 보증 및 검증 계획](#6-품질-보증-및-검증-계획)
7. [단계별 로드맵](#7-단계별-로드맵)
8. [팀 역할 및 책임](#8-팀-역할-및-책임)

---

## 1. 프로젝트 개요

### 1.1 목적

국민건강보험공단 가입자 중 **다재약물(Polypharmacy)**을 처방받는 환자를 위험도에 따라 자동 분류하여, 약사·의사의 개입 우선순위를 결정하고 약물상호작용(DDI) 및 중복약물로 인한 유해사례(ADR)를 선제적으로 예방한다.

### 1.2 핵심 과제

| 과제 | 설명 |
|------|------|
| 다재약물 환자 탐지 | 동시 5종 이상 처방 환자 자동 추출 |
| 약물상호작용(DDI) 탐지 | DDI 심각도별 분류 (Contraindicated → Minor) |
| 중복약물 탐지 | ATC 코드 3/4/5단계별 중복 처방 식별 |
| 위험도 분류 | 고위험/중위험/저위험/정상 4단계 자동 분류 |
| 개입 우선순위 결정 | 위험군별 차등 개입 수준 제안 |

### 1.3 대상 범위

- **대상 데이터**: 건강보험 청구 데이터 (T20 명세서, T30 처방전, T40 수진자, T50 요양기관)
- **대상 인구**: 전체 가입자 약 5,200만 명 중 동시 5종 이상 약물 복용 환자 (약 800만 명 추정)
- **관찰 기간**: 90일 슬라이딩 윈도우 (기본), 30일 보조 윈도우 병행

---

## 2. 위험도 분류 체계

### 2.1 다재약물 정의 (합의 기준)

| 구분 | 기준 | 비고 |
|------|------|------|
| 다재약물 (Polypharmacy) | 90일 내 **5종 이상** 만성질환 치료 약물 동시 복용 | ATC 5단계 고유 성분 기준 |
| 과다재약물 (Excessive Polypharmacy) | 90일 내 **10종 이상** 동시 복용 | |
| 카운트 제외 | PRN 약물, 단기처방(7일 이하), 외용제/점안제, 백신 | |

### 2.2 환자 위험도 4단계 분류

| 등급 | 명칭 | 분류 기준 | 개입 수준 |
|------|------|-----------|-----------|
| 🔴 Red | 고위험 | Contraindicated DDI 보유 **또는** Major DDI 3건 이상 **또는** Triple Whammy 해당 **또는** 10종 이상 + 고위험 약물 포함 **또는** 75세 이상 + 5종 이상 + 신기능/간기능 저하 약물 | 즉각 개입 (약사 직접 연락, 처방 재검토) |
| 🟡 Yellow | 중위험 | Major DDI 1~2건 **또는** Moderate DDI 2건 이상 **또는** 동일 성분 중복처방(Level 1) **또는** 3개 이상 의료기관 동시 처방 | 월 1회 정기 모니터링 |
| 🟢 Green | 저위험 | Minor DDI만 보유 **또는** 5종 이상이나 DDI 없음 **또는** ATC 3단계 허용 범위 중복만 존재 | 분기 1회 자동 안내 |
| ⚪ Normal | 정상 | 해당 없음 | 대상 외 |

### 2.3 DDI 심각도 분류 체계

| 등급 | 정의 | 예시 |
|------|------|------|
| **Contraindicated** | 절대 병용금기 | MAO억제제 + SSRI, 메토트렉세이트 + 트리메토프림 |
| **Major** | 생명위협 또는 영구 손상 가능 | 와파린 + NSAIDs, 디곡신 + 아미오다론 |
| **Moderate** | 기존 질환 악화 또는 추가 치료 필요 | ACE억제제 + 칼륨보존이뇨제, 스타틴 + 마크로라이드 |
| **Minor** | 경미한 영향, 임상적으로 대체로 무의미 | 제산제 + 철분제 (흡수 감소) |

### 2.4 반드시 탐지할 Top 10 DDI (100% 탐지율 목표)

| # | DDI 조합 | 위험 | 탐지 방식 |
|---|----------|------|-----------|
| 1 | Warfarin/DOAC + NSAIDs | 출혈 위험 급증 | Rule-based |
| 2 | Clopidogrel + Omeprazole/Esomeprazole | 항혈소판 효과 감소 (CYP2C19) | Rule-based |
| 3 | ACEi/ARB + K-sparing diuretics + NSAIDs | Triple Whammy (급성신부전) | Rule-based |
| 4 | Digoxin + Amiodarone/Verapamil | 디곡신 독성 | Rule-based |
| 5 | Methotrexate + Trimethoprim/SMX | 골수억제 | Rule-based |
| 6 | SSRI + MAO억제제 | 세로토닌 증후군 | Rule-based |
| 7 | SSRI + Triptan | 세로토닌 증후군 | Rule-based |
| 8 | Lithium + NSAIDs/이뇨제 | 리튬 독성 | Rule-based |
| 9 | QT 연장 약물 다중 병용 | 치명적 부정맥 | Rule-based |
| 10 | Statin + 마크로라이드 항생제 (Clarithromycin) | 횡문근융해 | Rule-based |

### 2.5 중복약물 탐지 기준 (3단계)

| Level | 기준 | 예시 | 예외 |
|-------|------|------|------|
| Level 1 | ATC 5단계 동일 (동일 성분) | 두 기관에서 amlodipine 처방 | 없음 |
| Level 2 | ATC 4단계 동일 (동일 약리 소분류) | amlodipine + nifedipine (둘 다 C08CA) | 항고혈압제 병용요법 |
| Level 3 | ATC 3단계 동일 (동일 치료목적) | ACE억제제 + ARB | 가이드라인 병용 허용군 제외 |

**중복 허용 예외 규칙 (E1~E5)**

| 코드 | 규칙 | ATC | 임상 근거 |
|------|------|-----|-----------|
| E1 | 항고혈압제 다제병용 | C02-C09 | 가이드라인 권장 병용 |
| E2 | 당뇨병 다제병용 | A10 | 단계적 병용 요법 |
| E3 | 흡입제 병용 (천식/COPD) | R03 | ICS+LABA+LAMA 표준 |
| E4 | 진통제 단계적 병용 | N02, M01 | WHO 진통제 사다리 |
| E5 | DAPT (이중항혈소판) | B01AC | 관상동맥 시술 후 표준 |

---

## 3. 데이터 아키텍처 및 파이프라인

### 3.1 핵심 데이터 테이블

| 테이블 | 주요 컬럼 | 용도 |
|--------|-----------|------|
| **T20 (명세서)** | 명세서번호, 수진자ID, 요양기관번호, 진료개시일, 주상병코드 | 청구 단위 기본 정보, 환자 질환 프로파일 |
| **T30 (처방전)** | 명세서번호, 약품코드(EDI), 1회투약량, 1일투약횟수, 총투약일수 | 약물 복용 정보 핵심 |
| **T40 (수진자)** | 수진자ID, 연령대, 성별, 보험유형 | 환자 특성 피처 |
| **T50 (요양기관)** | 요양기관번호, 종별코드, 진료과목 | 다기관 처방 탐지 |

### 3.2 필수 참조 테이블 (구축 필요)

| 테이블 | 출처 | 내용 |
|--------|------|------|
| EDI→ATC 매핑 | 식약처 의약품안전나라, HIRA 급여목록 | EDI 코드→ATC 5단계 전환 |
| DDI 매트릭스 | HIRA DUR (1순위) + DrugBank (보완) | 약물쌍별 심각도, 기전, 임상 효과 |
| 중복약물 그룹 테이블 | 약물전문가 정의 | ATC 3/4단계별 병용 허용 예외 플래그 |
| CYP450 효소 매핑 | DrugBank, 문헌 | 기질/억제제/유도제 분류 (강도별 25개 피처) |
| 고위험 약물 목록 | 약물전문가 정의 | Warfarin, MTX, Lithium 등 |

### 3.3 핵심 피처 목록

**그룹 A: 다재약물 기본 지표**
- `concurrent_drugs_90d` : 90일 윈도우 내 ATC 5단계 고유 약물 수
- `concurrent_drugs_30d` : 30일 윈도우 내 동시 복용 약물 수
- `max_daily_drugs` : 단일 날짜 기준 최대 동시 복용 약물 수
- `prescribing_institutions` : 동시 처방 요양기관 수

**그룹 B: DDI 지표**
- `ddi_contraindicated_count` : Contraindicated DDI 쌍 수
- `ddi_major_count` : Major DDI 쌍 수
- `ddi_moderate_count` : Moderate DDI 쌍 수
- `max_ddi_severity` : 최대 DDI 심각도
- `triple_whammy_flag` : Triple Whammy 해당 여부 (0/1)
- `top10_ddi_flags` : Top 10 DDI 각각 바이너리 플래그

**그룹 C: CYP450 피처 (강도별 25개)**
- `cyp3a4_substrate_count`, `cyp3a4_strong_inhibitor_count`, `cyp3a4_moderate_inhibitor_count`, `cyp3a4_inducer_count`, `cyp3a4_interaction_risk`
- CYP2D6, CYP2C9, CYP2C19, CYP1A2 각 동일 구조

**그룹 D: 중복약물 지표**
- `duplicate_level1_count` : 동일 성분 중복 수
- `duplicate_level2_count` : 동일 약리 소분류 중복 수
- `duplicate_level3_count` : 동일 치료목적 중복 수 (예외 적용 후)
- `complex_ddi_flags` : 복합 DDI 12패턴 각 플래그

**그룹 E: 임상 복합 지표**
- `serotonin_syndrome_risk` : 세로토닌 관련 약물 조합 수
- `qt_prolongation_drug_count` : QT 연장 약물 수
- `bleeding_risk_combo` : 항응고+항혈소판+NSAIDs 조합
- `cci_score` : 동반질환지수 (Charlson Comorbidity Index)
- `adr_proxy_score` : ADR 프록시 지표 (DDI별 관련 상병코드 매핑)

**그룹 F: 환자 특성**
- 연령대, 성별, 보험유형, 주상병 코드 그룹

### 3.4 데이터 파이프라인 흐름

```
[T20/T30/T40/T50] + [DDI DB] + [ATC 매핑]
          ↓
    Apache Spark ETL
  (스키마검증 → 가명처리 → 코드표준화 → 품질검사)
          ↓
    Delta Lake (Data Lake)
  raw/ → cleaned/ → features/
          ↓
    Feature Engineering (PySpark)
  Module A: 동시복용 계산 (overlap 알고리즘)
  Module B: DDI 탐지 (쌍 생성 → DB 조인)
  Module C: 중복약물 탐지 (Level 1/2/3)
  Module D: 처방 패턴 분석
  Module E: 환자 특성 집계
          ↓
    Feature Store (Parquet/Delta Lake)
          ↓
    모델 학습 / 배치 추론
```

**동시복용 판정 알고리즘**:
- 중첩일수 ≥ 7일: 동시복용으로 판정 (1~6일은 약물 전환으로 제외)
- 기준기간: 90일 윈도우 (primary), 30일 윈도우 (secondary)

**기술 스택**: Apache Spark 3.x (PySpark) + Apache Airflow + Delta Lake + Great Expectations

---

## 4. 모델 개발 전략

### 4.1 문제 정의

**다단계 위험도 분류 (Multi-class Classification)**

- 입력: 환자별 처방 정보 (약물 목록, 처방 패턴, 환자 특성)
- 출력: 위험도 등급 (Red/Yellow/Green/Normal) + SHAP 기반 위험 요인 설명
- 보조 태스크: DDI 쌍 탐지 (이진 분류), 중복약물 탐지

### 4.2 하이브리드 아키텍처 (팀 최종 합의)

```
입력 데이터
    ├── [Layer 1: Rule-based Safety Net]
    │       Top 10 DDI / Contraindicated DDI / Triple Whammy 명시적 규칙
    │       → Rule 등급 산출
    │
    └── [Layer 2: ML 모델]
            Phase별 모델 (XGBoost → GNN+Transformer 앙상블)
            → ML 등급 산출

최종 등급 = max(Rule 등급, ML 등급)
※ ML이 놓쳐도 Rule이 잡으면 고위험 분류 보장 (Safety Net 역할)
```

### 4.3 단계별 모델 개발 로드맵

| Phase | 기간 | 모델 | 성능 목표 |
|-------|------|------|-----------|
| **Phase 1** | 1~2개월 | Rule-based 베이스라인 | Top 10 DDI 100% 탐지, 골든 데이터셋 레이블링 착수 |
| **Phase 2** | 2~3개월 | XGBoost / LightGBM | 고위험 Recall ≥ 90%, AUC ≥ 0.85 |
| **Phase 3** | 3~4개월 | GNN + Transformer + 앙상블 | 고위험 Recall ≥ 95%, AUC ≥ 0.93 |

### 4.4 레이블링 전략

1. **1차**: Rule-based 출력 + 임상 전문가 검토 → 준지도학습 레이블
2. **2차**: 유해사례(ADR) 발생 여부를 후향적 레이블로 활용 (ADR 프록시 5종)
3. **3차**: 임상 전문가 패널 샘플 검토 → **골든 데이터셋** 구축 (5,650건 목표)

**ADR 프록시 지표 (후향적 레이블용)**

| DDI 조합 | 관련 상병코드 (ICD-10) |
|----------|------------------------|
| 와파린+NSAIDs | K92(위장관출혈), D68(응고장애), I60-I62(뇌출혈) |
| Triple Whammy | N17(급성신부전), E87(전해질이상) |
| 디곡신 독성 | I49(부정맥), R11(구역/구토) |
| 세로토닌 증후군 | G25(이상운동), R56(경련) |
| 저혈당 | E16(저혈당), R55(실신) |

### 4.5 클래스 불균형 대응

- 고위험군 약 5~10% 예상
- Focal Loss + 클래스 가중치 조정 병행
- Stratified K-Fold 교차검증
- 비용민감 학습 (Cost-sensitive): 고위험 미탐지 비용 >> 오탐지 비용

### 4.6 모델 해석가능성 (임상가 신뢰 확보)

- 전 모델 SHAP 값 산출
- 결과 예시: "이 환자가 고위험인 이유: DDI(와파린+아스피린) 1건 Contraindicated, 동시 처방 12종"
- 약사/의사용 대시보드에 위험 요인 설명 통합

### 4.7 핵심 성능 지표

| 지표 | Phase 2 목표 | Phase 3 목표 |
|------|-------------|-------------|
| 고위험 Recall | ≥ 90% | ≥ 95% |
| 고위험 Precision | ≥ 60% | ≥ 70% |
| 중위험 Recall | ≥ 80% | ≥ 85% |
| Macro F1-Score | ≥ 0.75 | ≥ 0.80 |
| Top 10 DDI 탐지율 | 100% | 100% |
| 임상 Critical Error | 0건 | 0건 |

---

## 5. 인프라 및 MLOps 설계

### 5.1 인프라 아키텍처

**온프레미스 우선 (건보 폐쇄망 보안 요구)**

| 구성요소 | 사양 |
|----------|------|
| GPU 서버 (GNN/Transformer 학습) | NVIDIA A100 **80GB** × 2~4장 |
| CPU 서버 (XGBoost/LightGBM) | 64코어+, 256GB RAM |
| 스토리지 | HDFS + Delta Lake (레이크하우스) |
| 네트워크 | 건보 내부망 폐쇄망, VPN 기반 관리자 접근 |

### 5.2 전체 기술 스택

```
[데이터 파이프라인]    Apache Airflow → PySpark → Delta Lake

[실험 관리]           MLflow (실험 추적, 모델 레지스트리, 아티팩트)
                      + Weights & Biases (GNN/Transformer 학습)

[CI/CD]               GitLab CI/CD (폐쇄망 호환)
                      → Docker Build → Harbor (컨테이너 레지스트리)

[오케스트레이션]       Kubernetes on-premise

[모델 서빙]           배치 추론: Spark + Airflow DAG (월 1회 전체 가입자)
                      API 서빙: FastAPI + Docker + K8s

[모니터링]            Grafana (19개 패널) + Prometheus
                      드리프트 감지: PSI 기반 자동 알림

[보안]                Apache Ranger / RBAC + 감사 로그 5년 WORM 보관
```

### 5.3 모델 서빙 방식

- **배치 추론 (기본)**: 월 1회 전체 800만 명 스코어링 → 결과 저장 및 리포팅
- **증분 배치**: 신규 처방 발생 시 해당 환자만 재스코어링 (Airflow 트리거)
- **배치 출력 스키마**: patient_id, risk_grade, ml_grade, rule_grade, top_risk_factors (SHAP), complex_ddi_flags, model_version, pipeline_run_id

### 5.4 CI/CD 파이프라인

```
code_quality → data_pipeline_test → feature_test → build
  → staging_deploy → model_perf_test → system_test
  → qa_approval → shadow_deploy (1개월) → production_deploy
```

### 5.5 모니터링 (Grafana 19개 패널)

| 핵심 패널 | 내용 |
|-----------|------|
| 위험등급별 분포 추이 | 일간 |
| 서브그룹별 Recall/Precision | 주간 |
| PSI 추이 (피처별) | 일간 |
| Critical/Major Error 건수 | 실시간 |
| Rule vs ML 불일치율 | 일간 |
| 약사 피드백 수용률 | 주간 |
| 배치 처리 시간/성공률 | 일간 |

### 5.6 재학습 트리거 정책

| 조건 | 임계값 | 대응 |
|------|--------|------|
| PSI > 0.25 (주요 피처 2개 이상) | 자동 | 긴급 재학습 파이프라인 |
| 고위험 Recall 3%p 이상 하락 | 알림 | 수동 재학습 검토 |
| 3개월 연속 성능 하락 추세 | 수동 | 모델 전면 재검토 |
| DDI DB 메이저 업데이트 | 72시간 | Rule 갱신 + 재평가 |
| 정기 재학습 | 분기 1회 | 자동 파이프라인 실행 |

---

## 6. 품질 보증 및 검증 계획

### 6.1 골든 데이터셋 (5,650건)

| 카테고리 | 건수 | 비율 |
|----------|------|------|
| 고위험 (Red) | 1,500 | 26.5% |
| 중위험 (Yellow) | 1,500 | 26.5% |
| 저위험 (Green) | 1,000 | 17.7% |
| 정상 (Normal) | 650 | 11.5% |
| 경계 케이스 | 500 | 8.8% |
| 중복약물 예외 케이스 | 500 | 8.8% |

- 리뷰어 자격: 약사 면허 + 경력 5년 이상 또는 임상약학 전문약사
- Cohen's Kappa 목표: 위험등급 ≥ 0.80, DDI 심각도 ≥ 0.85, 중복약물 ≥ 0.90

### 6.2 배포 게이트 (9개 Blocking 조건)

| # | 게이트 | 임계값 |
|---|--------|--------|
| 1 | Critical Error 수 | == 0 (Zero Tolerance) |
| 2 | Major Error Rate | ≤ 1% |
| 3 | 고위험 Recall | ≥ Phase별 목표 |
| 4 | 고위험 Precision | ≥ 60% |
| 5 | 중복약물 예외 처리 정확도 | ≥ 90% |
| 6 | Top 10 DDI 탐지율 | 100% |
| 7 | 서브그룹 Recall 편차 | ≤ 5%p |
| 8 | Regression Test | 전부 통과 |
| 9 | QA 최종 승인 | 약물전문가 + QA검증전문가 서명 |

### 6.3 파일럿 테스트 계획

| Phase | 유형 | 기간 | 목적 |
|-------|------|------|------|
| Phase A | 후향적 검증 | 4주 | 과거 데이터 기반 성능 검증 |
| Phase B | Shadow Deployment | 4주 | 실시간 병렬 운영 (영향 없음) |
| Phase C | 전향적 A/B Test | 8주 | 실제 개입 효과 측정 |

---

## 7. 단계별 로드맵

### Phase 1: Rule-based 베이스라인 (1~2개월)

- [ ] EDI→ATC 매핑 테이블 구축
- [ ] DDI 매트릭스 DB 구축 (HIRA DUR + DrugBank)
- [ ] 동시복용 기간 계산 알고리즘 구현 (overlap 알고리즘)
- [ ] Rule-based Safety Net 구현 (Top 10 DDI + Contraindicated DDI)
- [ ] 골든 데이터셋 레이블링 착수 (초기 1,000건)
- [ ] 데이터 파이프라인 기초 구축 (Spark + Airflow)
- [ ] 개발 환경 구성 (P0: 2주 내 완료 목표)

### Phase 2: 전통 ML 모델 (2~3개월)

- [ ] 전체 피처 엔지니어링 완료 (50+ 피처)
- [ ] XGBoost / LightGBM 기반 분류기 개발
- [ ] SHAP 기반 설명 생성
- [ ] 골든 데이터셋 3,000건 완성
- [ ] 성능 목표 달성 검증 (Recall ≥ 90%, AUC ≥ 0.85)
- [ ] 배포 게이트 1차 통과
- [ ] Shadow Deployment 시작

### Phase 3: 고도화 모델 (3~4개월)

- [ ] GNN (약물-약물 상호작용 그래프 임베딩) 개발
- [ ] Transformer (처방 시퀀스 모델) 개발
- [ ] 앙상블: Phase 2 + Phase 3 모델 결합
- [ ] 골든 데이터셋 5,650건 완성
- [ ] 성능 목표 달성 검증 (Recall ≥ 95%, AUC ≥ 0.93)
- [ ] 전향적 A/B 테스트 실시
- [ ] 정식 운영 배포

---

## 8. 팀 역할 및 책임 (RACI)

| 영역 | 약물전문가 | 데이터엔지니어 | 모델연구원 | MLOps엔지니어 | QA검증전문가 |
|------|-----------|--------------|-----------|--------------|-------------|
| DDI 기준 정의 | **R** | I | C | I | C |
| 데이터 파이프라인 구축 | C | **R** | C | C | I |
| 피처 엔지니어링 | C | **R** | C | I | I |
| 모델 개발 | C | I | **R** | C | C |
| 인프라/MLOps 구축 | I | C | C | **R** | I |
| 골든 데이터셋 검토 | **R** | I | C | I | **R** |
| QA 검증 | C | C | C | C | **R** |
| 배포 게이트 승인 | C | I | I | C | **R** |

> R=책임, C=협의, I=공유

---

## 부록: 주요 합의 사항 요약

| 항목 | 합의 내용 |
|------|-----------|
| 다재약물 기준기간 | 90일 (primary), 30일 (secondary 병행) |
| 동시복용 판정 역치 | 중첩일수 ≥ 7일 |
| DDI DB 우선순위 | HIRA DUR > DrugBank > 식약처 의약품안전나라 |
| 모델 아키텍처 | 하이브리드 (Rule-based Safety Net + ML) |
| GPU 사양 | NVIDIA A100 80GB × 2~4장 |
| 골든 데이터셋 규모 | 5,650건 (Pure Evaluation) + 5,000~7,000건 (Semi-Golden) |
| 고위험 Recall 목표 | Phase 2: ≥ 90%, Phase 3: ≥ 95% |
| Critical Error 기준 | Zero Tolerance |
| Shadow 운영 기간 | 1개월 |
| 감사 로그 보관 | 5년 (WORM 방식) |

---

*본 문서는 2026-03-05 팀 전체 합의 사항을 반영하여 작성되었습니다.*
