# Yellow 세분화 + 계층 분류 (A+C 결합) 설계

**작성일:** 2026-04-17
**목적:** Red(응급 개입) 탐지 정확도를 유지하면서 Yellow 내부를 임상 개입 수준별로 세분화하여, 3,540:1 극단 불균형 하에서도 실제 운영에 유의한 경보 체계를 제공한다.

---

## 1. 요구사항 요약

### 1.1 주요 목적
- **1차 목표**: Red 정탐(Precision/Recall)을 현재 이진 성능 수준 이상으로 유지
- **2차 목표**: Yellow 내부를 임상 개입 단계별로 나누어, 알림 채널을 차등 배분(약사 전화 vs 문자) → 운영팀의 개입 우선순위 확보
- **3차 목표**: 학습 라벨과 운영 액션의 1:1 연결로, 모델 출력이 곧 임상 개입 트리거가 되도록 설계

### 1.2 사용자 확정 결정 (Q1/Q2/Q3)
- **Q1.b**: Y_DDI 단일 카테고리가 아니라 Major / Moderate 을 **분리** → Y_DDI_MAJOR, Y_DDI_MOD
- **Q2.x**: Y_MIX 는 **별개 범주** — 2개 이상의 Yellow trigger 가 동시에 존재하면서 Red 조건에는 미달한 환자군
- **Q3**: Y_FRAG (3+기관 파편화) 개입은 **문자 알림**

### 1.3 현행 라벨 정의 (scripts/etl/prescription_aggregator.py:344-407)
Red 조건 (elif 우선순위 순, 첫 번째 매칭으로 확정):
1. `ddi_contraindicated ≥ 1`
2. `ddi_major ≥ 3`
3. `triple_whammy == True`
4. `drug_count ≥ 10 AND has_high_risk_drug`
5. `age ≥ 75 AND drug_count ≥ 5 AND (has_renal_risk_drug OR has_hepatic_risk_drug)`

Yellow 조건 (Red 조건 미충족 시):
- A: `ddi_major ≥ 1` (즉, 1~2건)
- B: `ddi_moderate ≥ 2`
- C: `dup_same_ingredient ≥ 1`
- D: `institution_count ≥ 3`

Green 조건: `ddi_minor ≥ 1` 또는 `drug_count ≥ 5`
Normal: 그 외

---

## 2. Yellow 세분화 라벨 정의

### 2.1 5개 카테고리

| 서브라벨 | 조건 (Red 조건 모두 미충족 전제) | 임상 개입 |
|---|---|---|
| **Y_MIX** | Yellow A/B/C/D trigger 중 **2개 이상**이 동시에 True | 약사 전화 (즉시) |
| **Y_DDI_MAJOR** | 단일 trigger: `ddi_major ≥ 1` (즉 1~2건) 만 True | 약사 전화 |
| **Y_DDI_MOD** | 단일 trigger: `ddi_moderate ≥ 2` 만 True | 문자 알림 |
| **Y_DUP** | 단일 trigger: `dup_same_ingredient ≥ 1` 만 True | 문서 + 문자 알림 |
| **Y_FRAG** | 단일 trigger: `institution_count ≥ 3` 만 True | 문자 알림 |

**핵심 설계:** 서브라벨은 "어떤 Yellow trigger 들이 발동했는가" 의 순수 함수이다. 순서(elif 우선순위)가 아니라 **trigger 집합**으로 정의된다. 이로써:
- Y_MIX 는 "2개 이상 trigger" 라는 명시적 집합 조건으로 판정 → 재현 가능
- 단일 trigger 환자는 해당 trigger 이름이 그대로 서브라벨이 됨 → 직관적
- Yellow 조건을 미래에 변경/추가해도 규칙이 독립적으로 분기됨

### 2.2 Y_MIX 의 명시적 Red-gate

Y_MIX 판정은 **반드시** Red 조건 미충족이 선행되어야 한다. 다시 말해:

```python
def classify_yellow_subtype(features):
    # 1) 호출 전제: risk_level == "Yellow"
    assert features.risk_level == "Yellow"

    # 2) Yellow trigger 집합 평가 (clinical_rules.collect_yellow_triggers 위임)
    triggers = collect_yellow_triggers(features)   # 공용 함수

    # 3) 2개 이상 → Y_MIX
    if len(triggers) >= 2:
        return "Y_MIX"

    # 4) 단일 trigger → 해당 서브라벨
    if triggers == {"DDI_MAJOR"}: return "Y_DDI_MAJOR"
    if triggers == {"DDI_MOD"}:   return "Y_DDI_MOD"
    if triggers == {"DUP"}:       return "Y_DUP"
    if triggers == {"FRAG"}:      return "Y_FRAG"

    # 5) 엣지 케이스 (규칙 드리프트/데이터 이상)
    #    ETL 이 멈추면 안 되므로 로그 + Y_OTHER 폴백 버킷에 격리
    logger.warning(
        "yellow_without_trigger patient_id=%s features=%s",
        features.patient_id, triggers
    )
    return "Y_OTHER"   # 모델 학습 라벨에서는 제외(또는 No_Alert 로 병합), 운영 검수 큐로
```

**Y_OTHER 처리 원칙:**
- ETL 단계에서 버킷으로 존재 — 전체 파이프라인 중단 방지
- 모델 학습 시점에서 `yellow_subtype == "Y_OTHER"` 인 샘플은 Stage 2 학습셋에서 **제외**하고 별도 감사 로그
- 운영 큐에서는 "규칙 검증 대기" 로 표시 → 임상팀 수동 판정
- Y_OTHER 가 통계적으로 증가하면 규칙 업데이트 트리거 (관측 기반 가드)

### 2.3 라벨 상수

```python
# 문자열 상수 (ordinal 의미 없음). 정수 인코딩은 LabelEncoder 가 학습 시 수행.
YELLOW_SUBTYPE_LABELS = (
    "Y_MIX",
    "Y_DDI_MAJOR",
    "Y_DDI_MOD",
    "Y_DUP",
    "Y_FRAG",
)
STAGE2_LABELS = YELLOW_SUBTYPE_LABELS + ("No_Alert",)  # Stage 2 6-class
```

**주의:** 서브라벨을 `{1, 2, 3, 4, 5}` 같은 숫자로 직접 표현하지 말 것. XGBoost 가 ordinal 패턴을 학습하려 시도할 수 있다. `LabelEncoder` 를 fit 하고 `classes_` 를 고정한 뒤 직렬화 경계에서만 숫자화한다. 추론 시 역변환은 동일 encoder 로 수행.

---

## 3. 계층 분류 아키텍처 (Hierarchical)

### 3.1 왜 계층화인가
- **Red vs 나머지** 는 극단 불균형 (수십:수만) 이지만 비교적 명확한 규칙 기반 → 이진 모델이 강점
- **Yellow 내부 5개 서브라벨** 은 서로 덜 불균형 (약 3:1 ~ 10:1 수준) → 다분류 모델이 feature interaction 을 학습하기 수월
- 단일 7-class (Red/Y_MIX/Y_DDI_MAJOR/Y_DDI_MOD/Y_DUP/Y_FRAG/Green/Normal) 모델은 Red 가 "소수 중의 소수" 가 되어 Red recall 이 떨어질 위험

### 3.2 Stage 1: Red 이진 탐지
- **입력**: 모든 환자
- **출력**: P(Red | x) ∈ [0, 1]
- **모델**: XGBClassifier, `scale_pos_weight = (non-Red) / Red` 로 이진 불균형 대응
- **손실**: `binary:logistic`
- **임계값**: validation PR-AUC 곡선에서 Recall ≥ 0.90 조건 하 최대 Precision 점
- **차단 로직**: `P(Red | x) ≥ τ_red` → 즉시 Red 분류, Stage 2 에 흘리지 않음

### 3.3 Stage 2: Yellow 서브라벨 다분류
- **입력**: Stage 1 에서 Red 로 분류되지 않은 환자 (`P(Red | x) < τ_red`)
- **출력**: 6-class softmax ({Y_MIX, Y_DDI_MAJOR, Y_DDI_MOD, Y_DUP, Y_FRAG, Green/Normal})
  - Green 과 Normal 은 본 스테이지에서 "개입 불필요" 단일 클래스로 통합 (`No_Alert`)
  - 따라서 실제 클래스 수 = 6
- **모델**: XGBClassifier, `objective="multi:softprob"`, `num_class=6`
- **손실 보정**: `sample_weight` 를 `compute_sample_weight("balanced", y_train)` 로 계산 (기존 `_xgb_multiclass_sample_weight` 재사용)
- **Cost-sensitive 옵션**: Y_MIX 와 Y_DDI_MAJOR 의 FN 비용을 더 높게 주는 cost_ratio 반영 가능

### 3.4 추론 파이프라인 (Inference) — **2단 임계값**

Stage 1 FN 영구 유실을 막기 위해 **2단 임계값** 적용:

- `τ_red`: "Red 확정" 상한 (예: 0.70)
- `τ_review`: "Red 의심" 하한 (예: 0.30), `τ_review < τ_red`

```
환자 feature x
  └─→ [Stage 1] P(Red | x)
         ├─ ≥ τ_red             → 최종 라벨 = Red,            액션 = 응급 개입
         ├─ τ_review ≤ P < τ_red → Stage 2 실행 + "Red 의심" 태그 병행
         │                          → Stage 2 출력 그대로 사용하되,
         │                            UI/운영 큐에서 "Red 의심" 표시로 우선순위 상승
         └─ < τ_review          → [Stage 2] multiclass (태그 없음)
                                    ├─ Y_MIX       → 약사 전화 (즉시)
                                    ├─ Y_DDI_MAJOR → 약사 전화
                                    ├─ Y_DDI_MOD   → 문자 알림
                                    ├─ Y_DUP       → 문서 + 문자
                                    ├─ Y_FRAG      → 문자 알림
                                    └─ No_Alert    → (알림 없음)
```

**설계 의도**: 순수 Stage 2 만 쓰면 Stage 1 이 놓친 Red 가 영구 유실된다. `τ_review` 구간의 환자는 Stage 2 라벨로 액션하되 "Red 의심" 태그로 운영팀 검수 큐에 들어가, 사후 피드백으로 Stage 1 을 지속 개선할 수 있다.

### 3.5 학습 파이프라인

1. **라벨 생성** (ETL 단계): `_assign_risk_level()` 이후 추가로 `_assign_yellow_subtype()` 실행
   - `features.yellow_subtype: Optional[str]` 필드 추가
   - `risk_level == "Yellow"` 일 때만 서브라벨 할당, 그 외는 `None`
2. **Stage 1 학습**: y = `(risk_level == "Red")`
3. **Stage 2 학습**: `risk_level != "Red"` 인 환자만 추출 → y = 6-class 라벨
4. **두 모델 독립 저장**: `models/risk_stage1_red.joblib`, `models/risk_stage2_yellow.joblib` + 각 SHA-256 메타

---

## 4. 불균형 처리 전략

### 4.1 Stage 1 (Red 이진)
- `scale_pos_weight = count(non-Red) / count(Red)`
- PR-AUC 중심 튜닝 (ROC-AUC 는 극단 불균형에서 낙관적 편향)
- Calibration 별도 검증 (Platt scaling 또는 Isotonic) — 임계값 튜닝 전제

### 4.2 Stage 2 (Yellow 다분류) — **추가 보강**
현재 `_xgb_multiclass_sample_weight()` 는 4-class 구설계에 특화됨. 6-class 로 확장 필요:

```python
def _stage2_sample_weight(y_train, cost_sensitive=False, cost_ratio_by_class=None):
    """Stage 2 6-class balanced sample_weight.

    cost_ratio_by_class 예:
      {"Y_MIX": 3.0, "Y_DDI_MAJOR": 2.5, "Y_DDI_MOD": 1.0,
       "Y_DUP": 1.0, "Y_FRAG": 0.8, "No_Alert": 0.5}
    """
    from sklearn.utils.class_weight import compute_sample_weight
    balanced = compute_sample_weight("balanced", y_train)
    if not cost_sensitive or cost_ratio_by_class is None:
        return balanced
    # cost_ratio 는 클래스명 → 배수, y_train 은 정수 인코딩이므로 디코딩 매핑 필요
    cost_mult = np.array([cost_ratio_by_class[LABEL_DECODE[int(c)]] for c in y_train])
    return balanced * cost_mult
```

**설계 원칙:** `compute_sample_weight("balanced")` 는 기본 깔개, cost_ratio 는 임상 비용 차등 (Y_MIX FN > Y_FRAG FN) 을 상위에 곱셈한다. 기존 4-class 함수와 동일 계약.

### 4.3 CV 전략
- `StratifiedKFold` 수동 반복 (이미 `train_model` 에 구현된 패턴 재사용)
- Stage 1 과 Stage 2 를 **독립적으로** CV: Stage 1 은 전체 데이터, Stage 2 는 non-Red 서브셋
- Stage 2 CV 에서 Red 누출 방지: train 과 valid 모두 `risk_level != "Red"` 필터 적용 후 KFold

---

## 5. 평가 지표

### 5.1 Stage 1
- **주지표**: PR-AUC, Recall@Precision=0.70
- **부지표**: ROC-AUC, F1, Brier score (calibration)
- **임계값**: Recall ≥ 0.90 필수, τ_red 를 PR 곡선에서 선택

### 5.2 Stage 2
- **주지표**: Macro F1 (5개 Yellow 서브라벨 + No_Alert 균등 가중)
- **부지표**:
  - 각 서브라벨별 Precision / Recall / F1 (임상팀 검수용)
  - Confusion matrix 6×6
  - Y_MIX / Y_DDI_MAJOR 재현율 별도 추적 (운영 우선순위 높음)

### 5.3 End-to-End
- **Red precision/recall** (Stage 1 단독 성능과 동일해야 함)
- **Y_MIX + Y_DDI_MAJOR 결합 recall** (약사 전화 대상자 포착률)
- **문자 알림 오발 비율** (No_Alert 인데 Y_* 로 분류된 비율)

---

## 6. 파일 변경 범위

### 6.1 ETL (라벨 생성)
- **scripts/etl/clinical_rules.py (신규)**: `collect_red_triggers`, `collect_yellow_triggers`, `CLINICAL_STANDARDS_VERSION`. ETL/학습 필터/서빙 3곳에서 공용 import (규칙 드리프트 방지)
- **scripts/etl/models.py**: `PatientFeatures` 에 `yellow_subtype: Optional[str] = None` 필드 추가
- **scripts/etl/prescription_aggregator.py**:
  - `_assign_risk_level()` 리팩터링 — `clinical_rules.collect_*_triggers()` 호출로 치환, elif cascade 제거
  - `_assign_yellow_subtype()` 신규 — `risk_level == "Yellow"` 환자에 대해서만 호출, trigger 집합으로 분기, Y_OTHER 폴백 포함

### 6.2 ML (학습)
- **hana_app/core/ml_runner.py**:
  - `train_model()` 확장: `target="risk_hierarchical"` 옵션 추가 → Stage 1 + Stage 2 를 한 번에 학습
  - `_stage2_sample_weight()` 신규 (위 4.2 참조)
  - 모델 저장 시 두 개의 .joblib 파일 + 각 SHA-256 메타
- **기존 `target="risk_binary"`, `target="risk_label"` 유지** — 하위 호환

### 6.3 Inference (서빙)
- **hana_app/core/predict_runner.py** (신규 또는 기존 확장):
  - `predict_risk(features)` — Stage 1 → 임계값 분기 → Stage 2 파이프라인
  - 반환: `{"risk_level": "Red"|"Yellow"|"No_Alert", "yellow_subtype": Optional[str], "action": str, "probs": {...}}`

### 6.4 UI
- **hana_app 기존 대시보드**: 결과 표에 `yellow_subtype` 컬럼 + 액션 컬럼 추가
- 환자 상세 페이지: Stage 1 / Stage 2 확률 동시 표시, 서브라벨 근거 (trigger 집합) 함께 노출

---

## 7. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| Red 누출 (Stage 1 에서 놓친 Red 가 Stage 2 에서 Y_* 로 분류) | τ_red 튜닝 시 Recall ≥ 0.90 고정. 운영 모니터링에서 Stage 2 예측된 Y_MIX 중 실제 Red 비율 추적 (drift alarm) |
| Y_MIX 라벨 오염 (규칙 변경 시 이전 라벨과 불일치) | ETL 라벨 생성 시점에 `CLINICAL_STANDARDS_VERSION` 태깅. 모델 메타에도 기록 |
| 서브라벨 분포 불균형 (예: Y_DDI_MOD 가 Y_MIX 의 20배) | Stage 2 sample_weight + cost_ratio. CV 에서 서브라벨별 recall 최저치 모니터링 |
| 계층 구조가 복잡해져 유지보수 부담 | Stage 1 / Stage 2 를 독립 함수로 분리. 테스트도 각각 독립 |
| 규칙 변경이 코드 3 곳에 퍼짐 (ETL/학습/서빙) | **Task 0 에서 `clinical_rules.py` 중앙화 선행.** `collect_red_triggers`, `collect_yellow_triggers`, `CLINICAL_STANDARDS_VERSION` 을 공용 import |
| Stage 1 FN 영구 유실 | **2단 임계값 (`τ_red`, `τ_review`) + "Red 의심" 태그** (3.4 참조). 운영 피드백으로 Stage 1 재학습 |
| Y_OTHER 엣지 케이스가 학습 데이터 오염 | Stage 2 학습셋에서 `yellow_subtype == "Y_OTHER"` 제외. Y_OTHER 증가율 모니터링 |

---

## 8. 구현 순서 제안

0. **[블로커] `scripts/etl/clinical_rules.py` 신규 — 규칙 중앙화**
   - `collect_red_triggers(features) -> set[str]`
   - `collect_yellow_triggers(features) -> set[str]`
   - `CLINICAL_STANDARDS_VERSION = "v1.0"` 상수
   - ETL / 학습 필터 / 서빙 3곳에서 공용 import
   - 단위 테스트: 각 trigger 규칙별 경계 케이스

1. **[블로커] `_assign_risk_level` 리팩터링 — trigger 집합 선수집**
   - 기존 elif cascade 를 `collect_red_triggers` + `collect_yellow_triggers` 호출로 치환
   - Red trigger 집합이 비지 않으면 `risk_level = "Red"`, `risk_reasons = list(red_triggers)`
   - Yellow 도 동일
   - Y_MIX 판정은 trigger 집합 크기로 분기 → cascade 와 논리 분리
   - 회귀 테스트: 기존 라벨 분포와 일치 확인

2. **라벨 생성 (ETL)**
   - `PatientFeatures.yellow_subtype: Optional[str] = None` 필드 추가
   - `_assign_yellow_subtype()` 구현 + 단위 테스트 (Y_OTHER 폴백 포함)

3. **Stage 1 학습 보강**
   - 기존 `target="risk_binary"` 코드 경로 재사용 + PR-AUC 튜닝
   - τ_red, τ_review 선택 로직 추가 (PR 곡선 + Recall ≥ 0.90 제약)

4. **Stage 2 학습 신규**
   - `_stage2_sample_weight()` 구현
   - non-Red 서브셋 추출 → 6-class 학습
   - **검증 TODO**: `stratified_sample_from_parquet` 이 (a) prefilter `risk_level != "Red"` + (b) `yellow_subtype` 6-class 층화의 조합을 지원하는지 확인, 미지원 시 래퍼 추가

5. **계층 predict 함수**
   - `predict_risk()` 신규 (또는 기존 predict 확장)
   - 2단 임계값 분기 + "Red 의심" 태그 지원

6. **평가 리포트**
   - End-to-end 지표 + 서브라벨별 세부 지표
   - Y_OTHER 유입 비율 감사 대시보드

7. **UI 연동**
   - 대시보드에 서브라벨 + 액션 + "Red 의심" 태그 컬럼

---

## 9. 확정 필요 사항 (후속 논의)

- τ_red 최종 임계값 — 데이터를 실제로 돌려본 뒤 PR 곡선 기반 결정
- cost_ratio_by_class 의 숫자 — 임상팀 합의 필요 (본 문서는 예시 값)
- Y_DUP 의 개입을 "문서 + 문자" 로 할지 "문서만" 으로 할지 — 운영 부담 확인 후 결정

---

**이 스펙이 확정되면:** `writing-plans` 스킬로 태스크 단위 구현 계획 작성 → 서브에이전트 드리븐 실행.
