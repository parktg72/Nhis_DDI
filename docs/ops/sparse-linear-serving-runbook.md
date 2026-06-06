# sparse_linear(다기관) 서빙 운영 — 재현 → B0 export → 배포 runbook

> phase3 인가 모델 `sparse_linear`(label=`multi_institution_t6_aligned_patient_disjoint`,
> val AUC **0.844954**, input_dim **15,273** drug-vocab)을 운영 서빙(DL 경로)에 올리는 절차.
> 가중치가 미저장돼 있어 **재현 → 가중치 저장 → B0 export → reload_dl** 로 배포한다.

## 0. 배경 / 갭

- 인가 모델은 same-window(2024-08~09 train, lookback 60d) **다기관 예측 ML**. freeze-safe
  (Nov→Dec future-onset 홀드아웃 아님; val은 holdout-disjoint).
- 서빙 갭: serving은 tabular ~25컬럼만 처리(sparse/CSR 코드 없음). sparse_linear는 별도
  **DL 경로**(`serving/dl_predictor.DLModel`, drug-only multi-hot)로만 서빙 가능.
- 인가 모델 **가중치는 디스크에 없음**(report만). 단 X_csr 행렬·config(seed42/ep20/batch2048)
  보존 → **결정적 재현 가능**(2026-06-06 실측 val_auc 0.844954 정확 재현).

## 1. 재현 + 가중치 저장 (freeze-safe — 동일 config, 신규 튜닝 아님)

```bash
.venv_hana/Scripts/python.exe -m scripts.ops.sparse_training_smoke \
  --train-dataset-dir data/datasets/multi_inst_t6_20240930_l60 \
  --val-dataset-dir   data/datasets/multi_inst_t6_20241130_l60_disjoint_train_l60_disjoint_frozen \
  --model linear --epochs 20 --batch-size 2048 --seed 42 \
  --output-dir cache/sparse_repro_perfect \
  --save-model-path cache/sparse_repro_perfect/sparse_linear.pt
```
→ report `val_auc=0.844954` 확인(인가 일치) + `sparse_linear.pt`(nn.Linear(15273,1) state_dict) 저장.

> ⚠️ **인가 데이터셋 정확히**: train=`multi_inst_t6_20240930_l60`, val=`..._disjoint_train_l60_disjoint_frozen`
> (n_val=15,596). 다른 val 디렉터리(`..._disjoint_train_l60`, n_val=16,633)는 0.84123 — 인가본 아님.
> 출처: `data/datasets/multi_inst_t6_temporal_20240930_to_20241130_l60_aligned_perfect/sparse_training_smoke_report.json`.

## 2. B0 export → DL 번들

```python
import json, torch, torch.nn as nn
from scripts.datasets.export_sparse_linear_bundle import export_from_torch_linear
vocab = json.load(open("data/vocab/drug_vocab.json"))          # 현재 15,273, _unk@0 (== input_dim)
model = nn.Linear(len(vocab), 1); model.load_state_dict(torch.load("cache/sparse_repro_perfect/sparse_linear.pt"))
export_from_torch_linear(
    "hana_app/models/dl/sparse_linear_multi_inst_t6_0930_perfect",
    model, vocab, run_id="sparse_linear_multi_inst_t6_aligned_patient_disjoint_perfect_repro20260606")
```
→ DL 번들(`MANIFEST.json`·`model.pt`·`drug_vocab.json`(15,273)·`model_config.json` 등). `Linear(in,1)`을
`Linear(in,2)` `softmax([0,z])[1]=sigmoid(z)`로 재구성(설계 등가, 테스트 잠금).

## 3. 서빙 검증 (parity)

```python
from serving.dl_predictor import DLModel; import pandas as pd
dl=DLModel(); assert dl.load("hana_app/models/dl/sparse_linear_multi_inst_t6_0930_perfect")
hist=pd.DataFrame({"patient_id":["P","P","P"],"prescription_date":["2024-09-01"]*3,"drug_code":[<코드3개>]})
print(dl.predict(hist)["probabilities"]["high"])   # == 원 Linear sigmoid(z)
```
2026-06-06 실측: DL `high`=0.435172 == Linear sigmoid 0.435172 (정확 등가). `vocab len==input_dim`,
`_unk` 매핑·OOV 경로는 [[serving DDI parity]] 인코더 정합과 동일 규약.

## 4. 배포

- DL 번들은 **startup env 미연동** — `HybridPredictor.reload_dl(<bundle_dir>)`(admin/runtime)로 로드.
  (계층=`HIERARCHICAL_MODEL_DIR`, tabular=`MODEL_PATH`는 startup env. DL은 reload_dl만.)
- `/predict` 응답에서 DL 출력은 **`dl_prediction` 보조 필드**(다기관 위험 low/high)로 노출.
  **주 `risk_level`은 hierarchical**(별 모델). DL 번들은 gitignored 배포 아티팩트(비커밋).

## 5. 남은 결정 (별도 cross-family)

- **DL 출력의 임상 위험 노출**(`dl_prediction`을 proxy 위험 라벨로 격상)은 별도 cross-family
  사인오프 — 본 절차는 "서빙 가능·parity 검증"까지. ([[app-feature-builder-missing-drugmaster-ddi-zero]])
- (선택) `DL_MODEL_DIR` startup env 추가 시 reload_dl 없이 부팅 로드 가능 — 서빙 변경이라 별도.
