# 건강보험공단 다제약물 위험도 분류 모델 QA Plan v1.0

**문서번호**: NHIS-POLY-QA-2026-001
**버전**: 1.0
**작성일**: 2026-03-05
**작성자**: QA검증전문가
**상태**: 최종안 (팀 리뷰 대기)

---

## 목차

1. [개요](#1-개요)
2. [시스템 아키텍처 및 QA 범위](#2-시스템-아키텍처-및-qa-범위)
3. [Golden Dataset 설계](#3-golden-dataset-설계)
4. [성능 지표 및 목표](#4-성능-지표-및-목표)
5. [임상 오류 분류 체계](#5-임상-오류-분류-체계)
6. [검증 방법론](#6-검증-방법론)
7. [편향 및 공정성 검증](#7-편향-및-공정성-검증)
8. [파일럿 테스트 설계](#8-파일럿-테스트-설계)
9. [모니터링 및 드리프트 탐지](#9-모니터링-및-드리프트-탐지)
10. [배포 게이트 및 CI/CD](#10-배포-게이트-및-cicd)
11. [규제 준수](#11-규제-준수)
12. [감사 추적 및 로그 관리](#12-감사-추적-및-로그-관리)
13. [테스트 시나리오](#13-테스트-시나리오)
14. [비상 대응 절차](#14-비상-대응-절차)
15. [부록](#15-부록)

---

## 1. 개요

### 1.1 프로젝트 배경

국민건강보험공단은 다제약물(Polypharmacy) 복용 환자의 위험도를 분류하여 차등화된 개입을 적용함으로써 약물 부작용(ADR)을 예방하고 운영 효율성을 개선하고자 한다. 본 QA Plan은 이 분류 모델의 정확성, 안전성, 공정성을 보장하기 위한 종합 검증 프레임워크를 정의한다.

### 1.2 분류 체계

| 등급 | 명칭 | 기준 | 개입 수준 |
|------|------|------|-----------|
| Red | 고위험 | 금기 DDI, 3개 이상 Major DDI, Triple Whammy 등 | 즉시 약사 개입 (DUR 우선 검토) |
| Yellow | 중위험 | 1-2개 Major DDI, 동일 ATC-4 이상 중복 등 | 정기 모니터링 (월 1회 리뷰) |
| Green | 저위험 | Moderate DDI만, Minor 중복 등 | 자동 알림 (분기별 리뷰) |
| Normal | 정상 | 해당 없음 | 대상 외 |

### 1.3 적용 범위

- **대상 데이터**: 건강보험 청구 데이터 (T20 진료, T30 처방, T40 투약, T50 수진자)
- **대상 환자**: 동시 5개 이상 약물 복용 환자 (다제약물 기준)
- **모델 유형**: Hybrid Architecture (Rule-based Safety Net + ML Model)
- **DDI 데이터베이스**: DrugBank, HIRA DUR, Lexicomp (교차 검증)

### 1.4 Hybrid Architecture

```
입력 데이터 --> [Layer 1: Rule-based Safety Net] --> Rule 등급
          \--> [Layer 2: ML Model]              --> ML 등급
                                                       |
                              최종 등급 = max(Rule 등급, ML 등급)
```

- **Layer 1 (Rule-based)**: 금기 DDI, Top 10 필수 탐지 DDI, Triple Whammy 등 규칙 기반 탐지
- **Layer 2 (ML)**: Phase별 모델 (XGBoost/LightGBM -> GNN/Transformer/Ensemble)
- **합산 규칙**: max() 적용 -- ML이 놓치더라도 Rule이 잡으면 고위험 분류

> **참고**: Hybrid Architecture는 모델연구원의 최종 확인 대기 중. 기술적으로 합의 완료 상태.

---

## 2. 시스템 아키텍처 및 QA 범위

### 2.1 QA 검증 레이어

| 레이어 | 검증 대상 | 담당 |
|--------|-----------|------|
| 데이터 품질 | 원천 데이터 무결성, ATC 코드 매핑, DDI DB 버전 관리 | 데이터엔지니어 + QA |
| Feature 품질 | Feature 연산 정확도, Null 처리, 시간 윈도우 정합성 | 데이터엔지니어 + QA |
| 모델 성능 | 분류 정확도, 리콜, 정밀도, 서브그룹별 공정성 | 모델연구원 + QA |
| 임상 타당성 | 임상적 오류율, 약물전문가 피드백 반영 | 약물전문가 + QA |
| 시스템 안정성 | 배치 처리 성능, 장애 복구, 로그 무결성 | MLOps엔지니어 + QA |
| 규제 준수 | 개인정보보호, 감사 추적, SaMD 분류 | QA (주관) |

---

## 3. Golden Dataset 설계

### 3.1 구조

| 구분 | 용도 | 규모 | 어노테이션 방식 |
|------|------|------|-----------------|
| Pure Evaluation Set | 모델 성능 최종 평가 전용 (학습에 절대 사용 불가) | 5,650건 | 2인 독립 리뷰 + 3인 중재 |
| Semi-Golden Set | 학습 데이터 라벨 보정 및 추가 학습 | 5,000-7,000건 | 자동 계산 + 전문가 확인 |

### 3.2 Pure Evaluation Set 구성 (5,650건)

| 카테고리 | 건수 | 비율 |
|----------|------|------|
| Red (고위험) | 1,500 | 26.5% |
| Yellow (중위험) | 1,500 | 26.5% |
| Green (저위험) | 1,000 | 17.7% |
| Normal | 650 | 11.5% |
| Edge Cases (경계) | 500 | 8.8% |
| 중복약물 예외 케이스 | 500 | 8.8% |

### 3.3 어노테이션 프로세스

#### 리뷰어 자격 요건
- 약사 면허 소지자 (경력 5년 이상)
- DUR 업무 경험 2년 이상
- 또는 임상약학 전문약사 자격

#### 독립 리뷰 프로토콜
- **1,000건 (Pure Independent)**: 자동 계산 결과 없이 완전 독립 판단 (anchoring bias 방지)
- **4,650건 (Auto-assisted)**: 자동 계산된 결과를 제시하되, 전문가가 확인/수정

#### 불일치 해소
1. 2인 리뷰어 독립 판단
2. 불일치 시 3번째 상급 전문가 중재
3. Cohen's Kappa로 일치도 측정

#### Cohen's Kappa 목표

| 판단 유형 | 목표 Kappa | 비고 |
|-----------|-----------|------|
| 위험등급 분류 (4등급) | >= 0.80 | Red/Yellow/Green/Normal |
| DDI 심각도 판단 | >= 0.85 | 금기/Major/Moderate/Minor |
| 중복약물 해당 여부 | >= 0.90 | Yes/No 이진 판단 |
| 중복약물 예외 해당 여부 | >= 0.75 | 임상적 판단 필요 |
| 최종 개입 필요성 | >= 0.80 | 개입 필요/불필요 |

### 3.4 Golden Dataset 관리

- **저장**: Delta Lake 형식, 버전 관리
- **접근 제어**: 학습 파이프라인에서 Pure Evaluation Set 접근 차단 (RBAC)
- **갱신 주기**: 반기 1회 검토, 필요 시 추가
- **변경 이력**: 모든 수정 사항 Git 이력 + 감사 로그 기록

---

## 4. 성능 지표 및 목표

### 4.1 Phase별 핵심 KPI

| 지표 | Phase 2 (XGBoost) | Phase 3 (GNN/Ensemble) | 비고 |
|------|-------------------|----------------------|------|
| **High-risk Recall** | >= 90% | >= 95% | 가장 중요한 안전 지표 |
| **High-risk Precision** | >= 60% | >= 70% | Alert Fatigue 방지 |
| **Medium-risk Recall** | >= 80% | >= 85% | |
| **Overall Accuracy** | >= 80% | >= 85% | |
| **Macro F1-Score** | >= 0.75 | >= 0.80 | 클래스 균형 성능 |

### 4.2 임상 오류율 목표

| 오류 등급 | 목표 | Blocking |
|-----------|------|----------|
| Critical Error (CE1-CE5) | **0건 (Zero Tolerance)** | Yes |
| Major Error (ME1-ME5) | <= 1% | Yes |
| Moderate Error (MoE1-MoE3) | <= 5% | Warn only |

### 4.3 Top 10 필수 탐지 DDI

모델은 다음 10개 DDI 조합에 대해 **100% 탐지율**을 달성해야 한다:

| # | DDI 조합 | 위험 | 목표 |
|---|----------|------|------|
| 1 | Warfarin + NSAIDs | 출혈 위험 | 100% |
| 2 | Warfarin + Fluoroquinolone | INR 상승 | 100% |
| 3 | ACEi/ARB + K-sparing diuretics + NSAIDs | Triple Whammy (급성신부전) | 100% |
| 4 | Methotrexate + NSAIDs | MTX 독성 | 100% |
| 5 | SSRI + MAOi | 세로토닌 증후군 | 100% |
| 6 | SSRI + Triptan | 세로토닌 증후군 | 100% |
| 7 | Digoxin + Amiodarone | Digoxin 독성 | 100% |
| 8 | Lithium + NSAIDs | Lithium 독성 | 100% |
| 9 | QT 연장 약물 다중 병용 | 심장 부정맥 | 100% |
| 10 | Clopidogrel + PPI (Omeprazole) | 항혈소판 효과 감소 | 100% |

> Rule-based Safety Net (Layer 1)에서 이 10개를 명시적 규칙으로 보장한다.

### 4.4 서브그룹별 성능 기준

- 모든 서브그룹에서 High-risk Recall이 전체 평균 대비 **5%p 이상 하락하지 않아야** 함
- 서브그룹 정의: 연령대(65 미만/65-74/75+), 성별, 보험유형(건강보험/의료급여), 처방기관 수(1-2/3-4/5+), 동시복용 약물수(5-9/10-14/15+)

### 4.5 Alert Fatigue 관리

- 고위험 알림: 임상의 1인당 **하루 최대 5건** 이하 목표
- Precision-threshold 최적화를 파일럿 기간 중 수행

---

## 5. 임상 오류 분류 체계

### 5.1 Critical Error (CE) - Zero Tolerance

| 코드 | 정의 | 예시 |
|------|------|------|
| CE1 | 금기(Contraindicated) DDI를 미탐지 | Warfarin + Metronidazole 미감지 |
| CE2 | 고위험 환자를 Normal로 분류 | Triple Whammy 환자를 정상 분류 |
| CE3 | Top 10 DDI 미탐지 | Warfarin + NSAIDs 미감지 |
| CE4 | 동일 성분 중복처방 미탐지 (ATC-5 동일) | 같은 성분 2개 기관 처방 미감지 |
| CE5 | 금기 DDI를 Green/Normal로 분류 | 금기인데 저위험/정상 분류 |

### 5.2 Major Error (ME) - <= 1%

| 코드 | 정의 | 예시 |
|------|------|------|
| ME1 | Major DDI를 Green으로 분류 | Major 상호작용을 저위험 분류 |
| ME2 | Red 환자를 Green으로 2단계 이상 하향 분류 | 고위험을 저위험으로 |
| ME3 | 동일 치료군(ATC-4) 중복을 미탐지 | 같은 계열 약물 중복 미감지 |
| ME4 | 3개 이상 Major DDI 보유자를 Yellow로 분류 | 다수 Major DDI를 중위험으로 |
| ME5 | 고위험 약물(Warfarin, MTX, Lithium 등) 관련 DDI를 1단계 하향 | 고위험 약물 DDI 과소평가 |

### 5.3 Moderate Error (MoE) - <= 5%

| 코드 | 정의 | 예시 |
|------|------|------|
| MoE1 | Yellow와 Green 간 1단계 오분류 | 중위험/저위험 경계 오분류 |
| MoE2 | Moderate DDI 심각도 과소평가 | Moderate를 Minor로 판단 |
| MoE3 | 중복약물 예외 적용 오류 (E1-E5 해당) | 임상적 허용 중복을 위험으로 분류 |

---

## 6. 검증 방법론

### 6.1 데이터 분할 전략

| 방법 | 용도 | 설명 |
|------|------|------|
| **Temporal Split** | 최종 성능 평가 | 과거 데이터로 학습, 미래 데이터로 검증 (시간 누수 방지) |
| **K-Fold CV** | 하이퍼파라미터 튜닝 전용 | 5-fold, 학습 과정에서만 사용 |
| **Stratified Sampling** | 서브그룹 검증 | 연령/성별/보험유형별 층화 추출 |

### 6.2 교차 검증 체계

| 검증 유형 | 내용 | 주기 |
|-----------|------|------|
| DDI DB 교차 검증 | DrugBank vs HIRA DUR vs Lexicomp 결과 비교 | DDI DB 업데이트 시 |
| Rule vs ML 교차 검증 | Rule-based와 ML 결과 불일치 분석 | 매 배치 추론 시 |
| Golden Dataset 평가 | Pure Evaluation Set 기반 성능 측정 | 모델 업데이트 시 + 월 1회 |
| 외부 검증 | 다른 기관 데이터 또는 외부 DDI DB로 검증 | 반기 1회 |

### 6.3 ADR Proxy 지표 (장기 유효성)

실제 약물 부작용(ADR) 데이터가 부족하므로, 다음 proxy 지표로 장기 유효성을 추적한다:

| Proxy 지표 | 정의 | 기대 방향 |
|------------|------|-----------|
| 약물 관련 입원율 | 고위험 분류 환자의 약물 관련 입원 비율 | 개입 후 감소 |
| 급격한 약물 중단율 | 부작용 의심으로 인한 갑작스런 중단 | 감소 |
| ER 방문 후 약물 변경 | 응급실 방문 후 처방 변경 비율 | 감소 |

### 6.4 모델 Handoff QA 체크포인트

모델연구원이 새 모델을 제출할 때 QA가 확인하는 항목:

| 체크포인트 | 확인 내용 | 기준 |
|------------|-----------|------|
| CP1 | Golden Dataset 성능 리포트 제출 | Phase별 KPI 충족 |
| CP2 | 서브그룹별 성능 테이블 | 5%p 이상 하락 없음 |
| CP3 | SHAP 설명 품질 | Top 3 risk factor가 임상적으로 타당 |
| CP4 | Error Analysis 리포트 | CE/ME/MoE별 건수 및 원인 분석 |
| CP5 | 이전 버전 대비 Regression 없음 | 기존 통과 케이스 유지 |
| CP6 | Top 10 DDI 탐지율 | 100% |

---

## 7. 편향 및 공정성 검증

### 7.1 서브그룹 정의

| 차원 | 그룹 | 근거 |
|------|------|------|
| 연령 | <65, 65-74, 75+ | 고령자 약동학 차이 |
| 성별 | 남성, 여성 | 약물 대사 차이 |
| 보험유형 | 건강보험, 의료급여 | 의료 접근성 차이 |
| 처방기관 수 | 1-2, 3-4, 5+ | 다기관 처방 위험 |
| 동시복용 약물수 | 5-9, 10-14, 15+ | 약물 부담 수준 |

### 7.2 공정성 기준

```
모든 서브그룹에서:
  High-risk Recall >= 전체 평균 - 5%p
  High-risk Precision >= 전체 평균 - 10%p
  False Negative Rate 차이 <= 5%p (그룹 간)
```

### 7.3 편향 탐지 프로세스

1. 모델 업데이트 시 서브그룹별 성능 자동 산출
2. 기준 미달 시 자동 알림 (Prometheus + Grafana)
3. 편향 발견 시 원인 분석 (데이터 불균형 vs 모델 구조) 후 보정

---

## 8. 파일럿 테스트 설계

### 8.1 Phase 구분

| Phase | 유형 | 기간 | 목적 |
|-------|------|------|------|
| Phase A | 후향적(Retrospective) 검증 | 4주 | 과거 청구 데이터 기반 성능 검증 |
| Phase B | Shadow Deployment | 4주 | 실시간 병렬 운영, 기존 시스템에 영향 없음 |
| Phase C | 전향적(Prospective) A/B Test | 8주 | 실제 개입 효과 측정 |

### 8.2 Phase A: 후향적 검증

- 최근 6개월 청구 데이터에 모델 적용
- Golden Dataset 기반 성능 평가
- 서브그룹별 성능 검증
- 기존 DUR 시스템 결과와 비교 분석

### 8.3 Phase B: Shadow Deployment

- 1개월간 실제 운영 환경에서 병렬 실행
- 기존 시스템과 새 모델의 결과를 동시에 생성하되, **새 모델 결과는 의사결정에 사용하지 않음**
- 비교 지표: 일치율, 불일치 원인 분석, 처리 시간
- Shadow 기간 중 Critical Error 발생 시 즉시 원인 분석

### 8.4 Phase C: 전향적 A/B Test

#### 윤리적 설계

```
고위험(Red) 환자: 전원 개입군 배정 (윤리적 이유로 대조군 불가)
중위험(Yellow) 환자: A/B 무작위 배정
  - A군 (개입): 모델 기반 개입
  - B군 (대조): 기존 방식 유지
저위험(Green) 환자: 모니터링만
```

#### 측정 지표

| 지표 | 정의 | 목표 |
|------|------|------|
| NNS (Number Needed to Screen) | 1건의 실제 개입 필요 케이스를 찾기 위해 스크리닝 필요한 수 | 측정 (baseline 대비 감소) |
| 약사 수용률 | 모델 권고를 약사가 수용한 비율 | >= 70% |
| 알림 피로도 | 임상의 설문 기반 피로도 측정 | 수용 가능 수준 |
| 개입 후 ADR Proxy 변화 | 약물 관련 입원/ER 방문 변화 | 감소 추세 |
| 처리 시간 | 건당 리뷰 소요 시간 | 기존 대비 단축 |
| 고위험 알림 수 | 임상의 1인당 일일 고위험 알림 수 | <= 5건/일 |

---

## 9. 모니터링 및 드리프트 탐지

### 9.1 PSI (Population Stability Index) 기반 드리프트 탐지

| PSI 값 | 상태 | 대응 |
|--------|------|------|
| < 0.10 | 안정 | 정상 모니터링 |
| 0.10 - 0.20 | 주의 | 주간 리뷰, 원인 분석 |
| 0.20 - 0.25 | 경고 | 즉시 분석, 재학습 검토 |
| > 0.25 | 위험 | 자동 알림, 긴급 재학습 트리거 |

### 9.2 모니터링 대시보드 (Grafana)

| 패널 | 내용 | 갱신 주기 |
|------|------|-----------|
| 1 | 위험등급별 분포 추이 | 일간 |
| 2 | 서브그룹별 Recall/Precision | 주간 |
| 3 | PSI 추이 (Feature별) | 일간 |
| 4 | Critical/Major Error 건수 | 실시간 |
| 5 | Rule vs ML 불일치율 | 일간 |
| 6 | 약사 피드백 수용률 | 주간 |
| 7 | 배치 처리 시간/성공률 | 일간 |
| 8 | 고위험 알림 수 (임상의별) | 일간 |
| 9 | 모델 버전별 성능 비교 | 모델 업데이트 시 |

### 9.3 자동 재학습 트리거

| 조건 | 임계값 | 대응 |
|------|--------|------|
| PSI > 0.25 (주요 Feature 2개 이상) | 자동 | 긴급 재학습 파이프라인 실행 |
| High-risk Recall 하락 | < Phase별 목표 - 3%p | 알림 + 수동 재학습 검토 |
| 월간 성능 리포트 하락 추세 | 3개월 연속 하락 | 모델 전면 재검토 |
| DDI DB 메이저 업데이트 | DrugBank/HIRA DUR 업데이트 | Rule-based 규칙 갱신 + 재평가 |

---

## 10. 배포 게이트 및 CI/CD

### 10.1 배포 게이트 조건

모든 blocking 조건을 통과해야 프로덕션 배포 가능:

| # | 게이트 | 지표 | 임계값 | Blocking |
|---|--------|------|--------|----------|
| 1 | Critical Error | critical_error_count | == 0 | Yes |
| 2 | Major Error Rate | major_error_rate | <= 0.01 | Yes |
| 3 | Moderate Error Rate | moderate_error_rate | <= 0.05 | Warn |
| 4 | High-risk Recall | high_risk_recall | >= Phase별 목표 | Yes |
| 5 | High-risk Precision | high_risk_precision | >= 0.60 | Yes |
| 6 | 중복약물 예외 처리 정확도 | duplicate_exception_accuracy | >= 0.90 | Yes |
| 7 | Top 10 DDI 탐지율 | top10_ddi_recall | == 1.0 | Yes |
| 8 | 서브그룹 Recall 편차 | subgroup_recall_gap | <= 0.05 | Yes |
| 9 | Regression Test | regression_pass | == true | Yes |

### 10.2 CI/CD 파이프라인 단계

```
code_quality --> data_pipeline_test --> feature_test --> build
  --> staging_deploy --> model_perf_test --> system_test
  --> qa_approval --> shadow_deploy --> production_deploy
```

### 10.3 환경 구성

| 환경 | 용도 | 데이터 |
|------|------|--------|
| Dev | 개발 및 단위 테스트 | 샘플 데이터 (익명화) |
| Staging | 통합 테스트 및 성능 검증 | 프로덕션 미러 데이터 (가명화) |
| Production | 실제 운영 | 실제 청구 데이터 |

---

## 11. 규제 준수

### 11.1 SaMD 분류

본 시스템은 **비SaMD (내부 행정 지원 도구)**로 분류한다:

- **근거**: 최종 의사결정은 약사/의사가 수행하며, 모델은 우선순위 정렬/필터링 역할만 담당
- **조건**: 모델 결과가 자동으로 환자 치료에 적용되지 않음
- **대비**: IEC 62304 수준의 문서화를 선제적으로 준비하여, 향후 SaMD 분류 변경 시 대응 가능

### 11.2 개인정보보호

| 항목 | 조치 |
|------|------|
| 환자 식별 | SHA-256 기반 가명처리, 원본 ID 접근 제한 |
| 데이터 최소화 | 분석 필요 항목만 추출, 불필요 개인정보 배제 |
| 접근 제어 | RBAC 기반, 역할별 데이터 접근 범위 제한 |
| 동의 | 건강보험법 제47조 기반 공익 목적 데이터 활용 |
| 로그 | 모든 데이터 접근/조회 이력 기록 |

### 11.3 감사 대비 문서

| 문서 | 내용 | 상태 |
|------|------|------|
| 모델 카드 (Model Card) | 모델 목적, 성능, 한계, 사용 범위 | 모델 배포 시 작성 |
| 데이터 시트 (Datasheet) | 데이터 출처, 전처리, 편향 분석 | 데이터 파이프라인 확정 시 |
| QA 리포트 | 검증 결과, 오류 분석, 개선 이력 | 매 모델 업데이트 시 |
| 변경 이력 | 모든 모델/데이터/규칙 변경 기록 | 상시 |

---

## 12. 감사 추적 및 로그 관리

### 12.1 WORM (Write Once Read Many) 감사 로그

- **구현**: Hash-chain 기반 무결성 보장
- **내용**: 모든 예측 결과, 모델 버전, 입력 Feature, 의사결정 근거
- **검증**: 일일 hash-chain 무결성 검증 자동 실행

### 12.2 로그 보존 정책

| 데이터 유형 | 보존 기간 | 저장 방식 |
|------------|-----------|-----------|
| 예측 로그 (Prediction Log) | 5년 | Hot (6개월) -> Warm (2년) -> Cold (2.5년) |
| 감사 로그 (Audit Log) | 5년 | WORM 스토리지 |
| 모델 아티팩트 | 영구 (주요 버전) | MLflow + 외부 백업 |
| 임상 피드백 | 5년 | Delta Lake |
| Golden Dataset | 영구 | Delta Lake + Git 버전관리 |

### 12.3 재현성 번들 (Reproducibility Bundle)

모든 모델 버전에 대해 다음을 패키징하여 보관:

```
reproducibility_bundle/
  model_artifact/        # 학습된 모델 파일
  training_data_hash     # 학습 데이터 해시
  feature_pipeline_hash  # Feature 파이프라인 코드 해시
  hyperparameters.json   # 하이퍼파라미터
  dependencies.txt       # 패키지 버전
  evaluation_report.json # Golden Dataset 평가 결과
  config.yaml            # 전체 설정
```

---

## 13. 테스트 시나리오

### 13.1 데이터 파이프라인 테스트 (DT)

| ID | 시나리오 | 검증 내용 | 기대 결과 |
|----|----------|-----------|-----------|
| DT-01 | ATC 코드 매핑 정확도 | 전체 약물 ATC 매핑률 | >= 99% |
| DT-02 | DDI DB 로딩 검증 | DrugBank/HIRA DUR 로딩 건수 | 기대 건수 일치 |
| DT-03 | 결측치 처리 | Null/NaN 처리 로직 | 정의된 대로 처리 |
| DT-04 | 시간 윈도우 정합성 | 동시 처방 기간 계산 | +-1일 오차 허용 |
| DT-05 | 데이터 스키마 검증 | 입출력 스키마 일치 | Great Expectations 통과 |
| DT-06 | 중복 제거 | 동일 처방 중복 제거 | 중복 0건 |
| DT-07 | 대용량 처리 | 월간 전체 데이터 배치 처리 | SLA 내 완료 |

### 13.2 Feature 테스트 (FT)

| ID | 시나리오 | 검증 내용 | 기대 결과 |
|----|----------|-----------|-----------|
| FT-01 | 동시복용 약물수 계산 | 기간 중복 약물 정확 카운트 | 수동 계산과 일치 |
| FT-02 | DDI pair 탐지 | 알려진 DDI pair 정확 탐지 | 100% 탐지 |
| FT-03 | Triple Whammy 탐지 | ACEi/ARB + K-sparing + NSAIDs 조합 | 정확 탐지 |
| FT-04 | 중복약물 탐지 | ATC-3/4/5 레벨별 중복 | 레벨별 정확 탐지 |
| FT-05 | 중복약물 예외 처리 | E1-E5 예외 규칙 적용 | 예외 정확 적용 |
| FT-06 | 복합 Feature 계산 | serotonin_risk, qt_prolongation 등 | 수동 계산과 일치 |

### 13.3 모델 성능 테스트 (MT)

| ID | 시나리오 | 검증 내용 | 기대 결과 |
|----|----------|-----------|-----------|
| MT-01 | Golden Dataset 전체 성능 | Phase별 KPI | 목표 충족 |
| MT-02 | Top 10 DDI 탐지 | 10개 DDI 조합 탐지율 | 100% |
| MT-03 | 서브그룹별 성능 | 연령/성별/보험유형별 | 5%p 이내 편차 |
| MT-04 | Edge Case 성능 | 경계 케이스 정확도 | >= 70% |
| MT-05 | Critical Error 검증 | CE1-CE5 발생 여부 | 0건 |
| MT-06 | Regression Test | 이전 버전 통과 케이스 | 전체 유지 |
| MT-07 | SHAP 설명 품질 | Top risk factor 임상 타당성 | 약물전문가 확인 |

### 13.4 시스템 테스트 (ST)

| ID | 시나리오 | 검증 내용 | 기대 결과 |
|----|----------|-----------|-----------|
| ST-01 | 배치 추론 End-to-End | 입력->Feature->추론->출력 전체 흐름 | 정상 완료 |
| ST-02 | 장애 복구 | 중간 단계 실패 시 재시도 | 자동 복구 |
| ST-03 | 감사 로그 무결성 | Hash-chain 검증 | 무결성 확인 |
| ST-04 | 동시성 처리 | 대규모 배치 병렬 처리 | SLA 내 완료 |

---

## 14. 비상 대응 절차

### 14.1 Critical Error 발생 시

```
1. 자동 알림 (Slack/Email) -> QA + 약물전문가 + MLOps
2. 해당 예측 결과 즉시 플래그 처리
3. 4시간 이내 원인 분석 착수
4. 24시간 이내 임시 조치 (Rule-based Safety Net 강화 등)
5. 72시간 이내 근본 원인 해결 및 재배포
6. 사후 보고서 작성 (Postmortem)
```

### 14.2 긴급 모델 롤백

```
조건: Critical Error 2건 이상 또는 Major Error Rate > 3%
절차:
  1. 현재 모델 즉시 비활성화
  2. 직전 안정 버전으로 자동 롤백
  3. Rule-based Safety Net만 우선 운영
  4. 원인 분석 후 수정 버전 재배포
```

### 14.3 긴급 재학습 정책

| 트리거 | 대응 시간 | 절차 |
|--------|-----------|------|
| PSI > 0.25 (주요 Feature) | 48시간 | 자동 재학습 파이프라인 실행 |
| Critical Error 발생 | 24시간 | 수동 분석 + 긴급 패치 |
| DDI DB 긴급 업데이트 | 72시간 | Rule 갱신 + 재평가 |
| 규제 변경 | 2주 | 영향 분석 + 전체 재검증 |

---

## 15. 부록

### 15.1 중복약물 예외 규칙

| 코드 | 규칙 | ATC 코드 | 임상 근거 |
|------|------|----------|-----------|
| E1 | 항고혈압제 다제병용 | C02-C09 | 가이드라인 권장 병용 |
| E2 | 당뇨병 다제병용 | A10 | 단계적 병용 요법 |
| E3 | 흡입제 병용 (천식/COPD) | R03 | ICS+LABA+LAMA 표준 요법 |
| E4 | 진통제 단계적 병용 | N02, M01 | WHO 진통제 사다리 |
| E5 | DAPT (이중항혈소판) | B01AC | 관상동맥 시술 후 표준 |

### 15.2 고위험 약물 목록

| 약물 | ATC 코드 | 위험 요인 |
|------|----------|-----------|
| Warfarin | B01AA03 | 출혈, 다수 DDI |
| Methotrexate | L01BA01 | 독성, 신기능 영향 |
| Lithium | N05AN01 | 좁은 치료 범위 |
| Digoxin | C01AA05 | 좁은 치료 범위 |
| Amiodarone | C01BD01 | QT 연장, 다수 DDI |
| Phenytoin | N03AB02 | 좁은 치료 범위 |
| Cyclosporine | L04AD01 | 신독성, 다수 DDI |

### 15.3 복합 DDI Feature (ML 입력)

| Feature | 정의 | 관련 위험 |
|---------|------|-----------|
| triple_whammy_flag | ACEi/ARB + K-sparing diuretic + NSAIDs 동시 처방 | 급성신부전 |
| serotonin_syndrome_risk | 세로토닌 관련 약물 조합 수 | 세로토닌 증후군 |
| qt_prolongation_drug_count | QT 연장 관련 약물 수 | 심장 부정맥 |
| bleeding_risk_combo | 항응고제 + 항혈소판제 + NSAIDs 조합 | 출혈 |

### 15.4 OTC 약물 제한사항

청구 데이터에는 일반의약품(OTC) 정보가 포함되지 않는다. 이에 대한 대응:

- **Rule-based 경고 플래그**: 항응고제(Warfarin)/Lithium 처방 환자에 대해 "OTC NSAIDs 병용 주의" 경고 자동 부착
- **한계 고지**: 모델 결과 보고 시 OTC 정보 미포함 사실 명시
- **향후 계획**: DUR 연계를 통한 OTC 정보 확보 가능성 검토

### 15.5 Prediction Log Schema

```json
{
  "prediction_id": "uuid",
  "model_version": "v2.1.0",
  "model_type": "lightgbm",
  "timestamp": "ISO-8601",
  "patient_id_pseudonym": "sha256_hash",
  "input_features": {
    "concurrent_drugs": "int",
    "ddi_pairs": "int",
    "max_ddi_severity": "Contraindicated|Major|Moderate|Minor",
    "duplicate_drugs": "int",
    "prescribing_institutions": "int",
    "age_group": "string",
    "insurance_type": "NHI|MedAid"
  },
  "prediction": {
    "risk_grade": "high|medium|low|normal",
    "probability": {"high": "float", "medium": "float", "low": "float", "normal": "float"},
    "rule_grade": "high|medium|low|normal",
    "ml_grade": "high|medium|low|normal",
    "final_grade_source": "rule|ml|both",
    "top_risk_factors": [
      {"feature": "string", "shap_value": "float"}
    ]
  },
  "feature_snapshot_id": "string",
  "pipeline_run_id": "string"
}
```

### 15.6 Clinician Feedback Schema

```json
{
  "feedback_id": "uuid",
  "prediction_id": "uuid",
  "clinician_id": "pseudonymized_id",
  "clinician_role": "pharmacist|physician",
  "timestamp": "ISO-8601",
  "model_risk_grade": "high|medium|low|normal",
  "clinician_risk_grade": "high|medium|low|disagree",
  "action_taken": "agree|modify_grade|override|dismiss",
  "action_detail": "string",
  "clinical_comment": "string (free text)",
  "time_spent_seconds": "int"
}
```

---

## 변경 이력

| 버전 | 일자 | 변경 내용 | 작성자 |
|------|------|-----------|--------|
| 0.1 | 2026-02-XX | 초안 작성 | QA검증전문가 |
| 0.5 | 2026-02-XX | 팀 피드백 반영 (약물전문가, 모델연구원, MLOps엔지니어, 데이터엔지니어) | QA검증전문가 |
| 1.0 | 2026-03-05 | 최종안 수립 (팀 합의 사항 전체 반영) | QA검증전문가 |

---

**승인**

| 역할 | 이름 | 승인일 | 서명 |
|------|------|--------|------|
| QA검증전문가 | - | 2026-03-05 | (작성) |
| 약물전문가 | - | | (대기) |
| 모델연구원 | - | | (대기) |
| MLOps엔지니어 | - | | (대기) |
| 데이터엔지니어 | - | | (대기) |
| Team Lead | - | | (대기) |
