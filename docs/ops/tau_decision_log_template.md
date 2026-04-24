# τ (τ_red / τ_review) 결정 로그

> 계층 분류 모델의 2단 임계값을 실데이터로 확정하는 단계에서 **폐쇄망 작업자** 가
> 사용하는 의사결정 기록 양식. 재학습 시마다 새 사본을 만들고, 결정 근거·근거
> 테이블·최종값을 남겨 Git 에 commit 한다.
>
> Task 4 (실데이터 τ 튜닝) 의 산출물 — 완성본은 `docs/ops/tau_decision_log_YYYY-MM-DD.md`
> 형태로 저장.

---

## 0. 메타

| 항목 | 값 |
|---|---|
| 작성일 | YYYY-MM-DD |
| 작성자 | (이름/역할) |
| 모델 버전 | stage_meta.json 의 `clinical_standards_version` |
| 데이터 스냅샷 | NHIS 기준일 / 환자 수 / 분할 방식 |
| 검증 세트 크기 | N 환자 (양성 K 명) |

---

## 1. 입력 아티팩트

- `y_true.npy` 경로: `____________________________________________________`
- `y_proba.npy` 경로: `____________________________________________________`
- 추출 방법 (아래 스니펫 복붙 후 경로 수정):

```python
import joblib, numpy as np, pandas as pd
stage1 = joblib.load("models/hierarchical/stage1_red.joblib")
val = pd.read_parquet("data/validation_features.parquet")
# feature_cols 는 stage_meta.json["feature_cols"] 와 동일 순서여야 함
FEATURE_COLS = [...]
X = val[FEATURE_COLS].to_numpy()
y = val["is_red"].astype(int).to_numpy()
np.save("y_true.npy", y)
np.save("y_proba.npy", stage1.predict_proba(X)[:, 1])
```

---

## 2. 민감도 스윕 실행

```bash
python scripts/train/tau_sensitivity.py \
  --y-true y_true.npy \
  --y-proba y_proba.npy \
  --recall-floors 0.85,0.88,0.90,0.92,0.95 \
  --review-recall-target 0.98 \
  --output-dir reports/tau_sensitivity/$(date +%F)/
```

산출물:

- `reports/tau_sensitivity/<date>/tau_report.json`
- `reports/tau_sensitivity/<date>/tau_report.md`

---

## 3. 스윕 결과 (tau_report.md 붙여넣기)

<!-- 여기에 tau_report.md 의 표를 복사 -->

```
(여기에 마크다운 테이블 붙여넣기)
```

---

## 4. 결정

### 4.1 `recall_floor` 선정

- **선택값**: `0.__`
- **이유** (하나 이상):
  - [ ] 임상 가이드라인: __________________________ 에서 Red recall ≥ 0.__ 요구
  - [ ] `fallback=YES` 인 값 후보에서 제외
  - [ ] `red_leakage_%` 가 허용 상한 (__%) 이하
  - [ ] `stage2_traffic_%` 가 운영 처리 용량 범위 내
- **기각한 후보와 이유**:
  - 0.__ → __________________________
  - 0.__ → __________________________

### 4.2 `review_recall_target` 선정

- **선택값**: `0.__`
- **이유**:
  - review band 크기가 일일 수동 검토 가능 건수 이하 (약사 N명 × M건/시)
  - Stage 1 FN 영구 유실 (`red_lost_clean_stage2`) 을 Y 건 이하로 유지

### 4.3 `cost_ratio_by_class` 선정 (Stage 2 가중치)

비용 분석 가정:

| 채널 | 단가 | 적용 라벨 |
|---|---|---|
| 약사 전화 (즉시) | ₩____ / 건 | Y_MIX |
| 약사 전화 | ₩____ / 건 | Y_DDI_MAJOR |
| 문자 알림 | ₩____ / 건 | Y_DDI_MOD / Y_DUP / Y_FRAG |
| Red 누출 피해 (가정치) | ₩____ / 건 | (recall 상한 논거) |

- **선택 비율** (`cost_ratio_by_class`):
  ```python
  cost_ratio_by_class = {
      "Y_MIX":       _,
      "Y_DDI_MAJOR": _,
      "Y_DDI_MOD":   _,
      "Y_DUP":       _,
      "Y_FRAG":      _,
      "No_Alert":    _,
  }
  ```
- **근거**: __________________________

---

## 5. 재학습 및 검증

1. `train_hierarchical(..., recall_floor=__, review_recall_target=__, cost_sensitive=True, cost_ratio_by_class={...})` 실행
2. `stage_meta.json` 의 `thresholds` 확인
3. Stage 1 PR-AUC / Stage 2 Macro F1 / 클래스별 P·R·F1 확인
4. 이전 모델과 confusion matrix 비교

---

## 6. 최종 서명

| 역할 | 이름 | 서명 | 일자 |
|---|---|---|---|
| 데이터 분석 | | | |
| 임상 감수 | | | |
| 운영 승인 | | | |

---

## 7. 배포 이후 모니터링 포인트

- `Y_OTHER` 증가율 (rule 드리프트 시그널)
- Stage 2 에서 예측된 Y_MIX 중 실제 Red 비율 (Stage 1 FN 재발견 지표)
- `red_suspect=True` cohort 의 사후 Red 확인율
- 월 1회 본 결정 로그 재검토 (데이터 이동 / 임상 가이드 개정 감시)
