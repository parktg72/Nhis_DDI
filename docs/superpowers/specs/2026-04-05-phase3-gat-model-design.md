# Phase 3 GAT 모델 설계 문서

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** PyTorch Geometric 기반 GAT 모델을 기존 XGBoost + LightGBM 앙상블에 세 번째 서브모델로 추가하여 DDI 위험도 분류 성능을 Phase 3 목표(Recall ≥ 95%, AUC ≥ 0.93)에 근접시킨다.

**Architecture:** NHIS 처방 데이터에서 약물 동시처방 그래프를 구성하고 2-layer GAT로 약물쌍 위험도를 학습한다. GAT 출력은 Platt scaling으로 보정 후 요청 레벨로 집계하여 기존 앙상블의 가중 평균에 편입한다. 기존 `BaseTrainer` / `EnsembleTrainer` / DAG 배포 인프라를 최대한 재사용한다.

**Tech Stack:** PyTorch Geometric (PyG), PyTorch, scipy.optimize, 기존 XGBoost/LightGBM 앙상블

---

## 1. 파일 구조

| 파일 | 역할 |
|------|------|
| `scripts/features/graph_builder.py` | 훈련 데이터 → PyG Data 객체 + 직렬화 |
| `scripts/train/base_graph_trainer.py` | BaseTrainer 확장: PyG Data 인터페이스 |
| `scripts/train/gat_model.py` | 2-layer GAT 모델 정의 |
| `scripts/train/gat_trainer.py` | GATTrainer: train/calibrate/save/load |
| `scripts/train/ensemble_trainer.py` | GATTrainer 서브모델 풀 추가, recall 제약 가중치 최적화 |
| `serving/predictor.py` | MLModel.load() GAT 지원, 쌍 집계, 동적 서브모델 제외 |
| `dags/ddi_train_dag.py` | 배포 파일 체인에 GAT 아티팩트 추가 |
| `tests/test_train/test_gat_trainer.py` | GraphBuilder, GATModel, GATTrainer 유닛 테스트 |
| `tests/test_integration/test_gat_deploy.py` | GAT 배포 체인 통합 테스트 |

---

## 2. 그래프 구성 (`scripts/features/graph_builder.py`)

### 입력/출력
- **입력:** 훈련 분할 처방 DataFrame만 (val/test 누출 방지 — H-1)
  - 필수 컬럼: `patient_id`, `drug_code`, `prescription_date`
- **출력:**
  - `Data(x, edge_index, edge_weight, drug_to_idx)` — PyG 객체
  - `gat_graph.pt` — 직렬화된 Data 객체
  - `gat_graph.pt.sha256` — 무결성 해시
  - `gat_graph_meta.json` — 빌드 메타데이터

### 그래프 구조
```
노드: 훈련 데이터 내 고유 drug_code
엣지: 동일 patient_id + prescription_date 내 동시처방 쌍
엣지 가중치: 쌍별 동시처방 빈도 → log1p 정규화
노드 피처: 기존 FeatureBuilder 수치 피처 (ATC 계층, 상호작용 카운트 등)
```

### gat_graph_meta.json 구조
```json
{
  "built_at": "2026-04-05T12:00:00",
  "num_nodes": 1234,
  "num_edges": 56789,
  "edge_index_hash": "sha256:...",
  "data_date_range": {"start": "2020-01-01", "end": "2024-12-31"},
  "feature_dim": 42
}
```

### 미지 약물 처리 (C-3)
- 추론 시 `drug_to_idx`에 없는 약물 코드 → 해당 약물쌍의 GAT 서브모델 **앙상블 제외**
- 제로 벡터 폴백 사용 금지 (의료 안전 사유)
- `logger.warning("알 수 없는 약물 코드 — GAT 서브모델 제외: %s", drug_code)` 필수

### 그래프 품질 경고
- 평균 노드 차수 < 5 → `logger.warning` (GAT 효용 저하 가능)
- 고립 노드 비율 > 10% → `logger.warning` (아키텍처 재검토 권고)

---

## 3. GAT 모델 (`scripts/train/gat_model.py`)

### 모델 구조
```python
class GATModel(nn.Module):
    # layer 1: GATConv(in_channels=feature_dim, out_channels=64, heads=4, concat=True)
    # layer 2: GATConv(in_channels=256, out_channels=32, heads=1, concat=False)
    # pair scorer: Linear(in=32*4, out=1) + sigmoid
    # 입력 feature: concat([h_a, h_b, |h_a - h_b|, h_a * h_b])  # M-3: 곱셈항 포함
```

### 재현성 (L-3)
- 학습 시작 전 `torch.use_deterministic_algorithms(True)` 설정
- 시드 고정: `torch.manual_seed`, `torch.cuda.manual_seed_all`

---

## 4. 트레이너 계층 구조

### `BaseGraphTrainer` (`scripts/train/base_graph_trainer.py`)
`BaseTrainer` 서브클래스. PyG Data 객체를 수용하도록 데이터 처리 메서드 오버라이드. `save/load/evaluate` 인터페이스는 `BaseTrainer` 규약 유지. (H-5)

### `GATTrainer` (`scripts/train/gat_trainer.py`)
`BaseGraphTrainer` 서브클래스.

**데이터 분할 전략 (H-2): 60/20/10/10**
```
train(60%)         → 그래프 엣지 구성 + 모든 서브모델 학습
val(20%)           → XGB/LGB 조기종료 (EarlyStopping)
gat_val(10%)       → GAT 전용 조기종료 (XGB/LGB val과 완전 분리)
calibration(10%)   → Platt scaling 보정 + 앙상블 가중치 최적화 전용
                     (어떤 서브모델의 학습/조기종료에도 사용 안 함)
```

**보정 (H-3):**
```python
def calibrate(self, calibration_data) -> None:
    # sklearn.calibration.CalibratedClassifierCV (Platt)
    # calibrator를 gat_model.pt에 함께 직렬화
```

**저장 파일 (C-2):**
```
gat_model.pt              ← 모델 가중치 + calibrator
gat_model.pt.sha256
gat_graph.pt              ← 훈련 그래프 (edge_index, x, drug_to_idx)
gat_graph.pt.sha256
gat_graph_meta.json
```

**`load()` 검증 (M-4):**
- `gat_graph.pt` 존재 여부 + sha256 일치 확인
- 불일치 또는 누락 시 `RuntimeError` 발생

---

## 5. EnsembleTrainer 통합 (`scripts/train/ensemble_trainer.py`)

### 서브모델 풀 확장
```python
submodels = {
    "xgb": XGBTrainer,
    "lgb": LGBTrainer,
    "gat": GATTrainer,  # 신규
}
```

### 가중치 최적화 (H-6)
```python
# 목적: calibration 스플릿 AUC 최대화
# 제약: calibration 스플릿 Recall >= 0.90
# 도구: scipy.optimize.minimize (SLSQP)
# 제약: w_xgb + w_lgb + w_gat = 1.0, 각 w >= 0.0
```

### 동적 서브모델 제외 (C-3)
- 미지 약물 포함 요청 → 해당 요청에서 GAT 제외
- 나머지 서브모델 가중치 정규화하여 합 = 1.0 유지

### 배포 파일 체인 (C-2)
```
model_prod.pkl + .sha256
xgb_model.pkl + .sha256
lgb_model.pkl + .sha256
gat_model.pt  + .sha256   ← 신규
gat_graph.pt  + .sha256   ← 신규
gat_graph_meta.json       ← 신규
```
sha256 누락 파일 1개라도 존재 시 배포 중단.

---

## 6. Serving 통합 (`serving/predictor.py`)

### MLModel.load() 확장
- `.pt` 파일: `torch.load()` + path traversal 검증 (기존 scaler/selector 동일 패턴)
- 서빙 시작 시 `gat_graph.pt` 로드 → 메모리 상주 (frozen training graph)
- `gat_graph_meta.json` 로드 → 그래프 나이 > 180일 시 `logger.warning`

### 예측 흐름 (C-1 — 요청 레벨 통일)
```
request: [drug_a, drug_b, drug_c, ...]
  → 모든 쌍 열거: (a,b), (a,c), (b,c), ...
  → 각 쌍: GAT forward → p_pair (보정 후)
  → 미지 약물 포함 쌍: GAT 제외 표시
  → 유효 쌍 p_pair → max 집계 → p_gat (요청 레벨)
  → p_final = w_xgb·p_xgb + w_lgb·p_lgb + w_gat·p_gat
     (미지 약물 있으면 w_gat=0, 나머지 정규화)
```

### 서빙 환경 (M-2)
- CPU 추론 우선 (GPU 선택적)
- `MLModel.load()` 에서 CPU/GPU 가용성 자동 감지, 로그 출력
- CPU 레이턴시 목표: < 100ms / 요청

---

## 7. DAG 통합 (`dags/ddi_train_dag.py`)

- `_deploy_model` 파일 체인 목록에 GAT 아티팩트 3종 추가
- 롤백 시 GAT 아티팩트도 함께 복구
- sha256 누락 → 기존 정책과 동일하게 배포 중단 및 `RuntimeError`

---

## 8. 테스트 전략

### 유닛 테스트 (`tests/test_train/test_gat_trainer.py`)
```
- GraphBuilder:
    - 동시처방 쌍 생성 정확도 (동일 날짜 처방만 엣지)
    - 엣지 가중치 log1p 정규화 검증
    - train split 전용 입력 → val pair 엣지 미포함 검증 (H-1)
    - 고립 노드 비율 임계값 경고
- GATModel:
    - forward 출력 shape [num_pairs, 1], 값 ∈ [0, 1]
    - concat([h_a, h_b, |h_a-h_b|, h_a⊙h_b]) 피처 검증
- GATTrainer:
    - 60/20/10/10 분할 비율 검증
    - calibrate() → calibration 스플릿 ECE 개선 검증
    - save() → gat_model.pt + gat_graph.pt + 2개 sha256 + meta.json 생성
    - load() → gat_graph.pt sha256 불일치 시 RuntimeError
    - 미지 약물 코드 → 경고 로그 + 쌍 스코어 None 반환
```

### 통합 테스트 (`tests/test_integration/test_gat_deploy.py`)
```
- 배포 체인: gat_model.pt sha256 누락 시 배포 중단
- 배포 체인: gat_graph.pt sha256 누락 시 배포 중단
- MLModel.load(): gat_graph.pt sha256 불일치 시 RuntimeError
- path traversal: model_dir 외부 .pt 경로 거부
- 동적 서브모델 제외: 미지 약물 → w_gat=0, 나머지 정규화 후 합=1.0
- 롤백: GAT 아티팩트 모두 복구 검증
```

### 성능 테스트 (`tests/test_train/test_pipeline.py` 확장)
```
- GAT 단독 AUC ≥ 0.80 (최소 기여 기준)
- 앙상블 Recall ≥ 0.90, AUC ≥ 0.85 (Phase 2 기준 유지)
- recall 제약 하 가중치 최적화: w 합 = 1.0, 각 w ≥ 0.0 검증
```

---

## 9. 알려진 제약사항 및 향후 검토

| 항목 | 내용 |
|------|------|
| 그래프 규모 | NHIS 실데이터로 `|V|`, `|E|` 프로파일링 후 subgraph sampling 여부 결정 |
| 트랜스덕티브 한계 | 신규 약물 진입 시 그래프 재빌드 필요 — 정기 재학습 DAG로 대응 |
| GPU 선택적 | CPU 레이턴시 100ms 초과 시 GPU serving 전환 검토 |
| Phase 3 목표 | Recall ≥ 95%, AUC ≥ 0.93은 Transformer 서브모델 추가 후 재평가 |
