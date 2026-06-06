# Model Card — sparse_linear (다기관 multi_institution_t6)

> 2026-06-06 cross-family(Claude LO ↔ Codex) 거버넌스 리뷰의 산출물.
> **승인 범위 = AUXILIARY-ONLY**(정보 필드). 임상 위험 구동(risk_level/intervention) **미승인**.

## 식별

| 항목 | 값 |
|---|---|
| run_id | `sparse_linear_multi_inst_t6_aligned_patient_disjoint_perfect_repro20260606` |
| label | `multi_institution_t6_aligned_patient_disjoint` (same-window) |
| label semantics | `institution_count >= 3` (= `clinical_rules.py` FRAG 룰과 동일 임계, same-window) |
| 아키텍처 | `nn.Linear(15273, 1)` → B0 export 시 `Linear(15273, 2)` (`softmax([0,z])[1]=sigmoid(z)`) |
| 인코딩 | drug-only multi-hot (`drug_code`=EDI/MCARE_DIV_CD → vocab, `_unk@0`) |
| vocab | `data/vocab/drug_vocab.json` (len 15,273, `_unk@0`, == input_dim) |
| lookback | 60일 |
| train window | 2024-08-01 .. 2024-09-30 (reference 2024-09-30), n_train 47,834 |
| val | n_val 15,596 (train·frozen-holdout 모두 disjoint) |
| **val AUC** | **0.844954** (2026-06-06 정확 재현 실측) |
| val_best_threshold | F1-max @ 학습 prevalence (**임상 임계 아님** — 랭킹 점수로만 사용) |
| freeze | same-window, freeze-safe (Nov→Dec future-onset 홀드아웃 아님) |
| 출처 | `data/datasets/multi_inst_t6_temporal_20240930_to_20241130_l60_aligned_perfect/sparse_training_smoke_report.json` |

## 승인 범위 (Decision A, cross-family 2026-06-06)

- **현재 사용**: `/predict` 응답의 `dl_prediction`(Optional) **정보 필드만**. 주 `risk_level`/
  `intervention`에 **영향 없음**(serving/predictor.py:1350-1353은 rule/hierarchical만; schemas.py:155-158 명시).
- `high` 확률은 **proxy propensity**(이력 약물패턴 기반 다기관 성향)이지 검증된 임상 위험 아님.
- **위험 구동(C) 미승인.** 별도 임상 알림(B)도 아래 선결조건 충족 전 미승인.

### Decision A 근거 (각 단독 충분)
1. phase3 `CLINICAL_REVIEW_AUTHORIZED`(심사 개시) ≠ 위험 구동 배포 인가.
2. same-window 라벨: `MULTI_INSTITUTION_THRESHOLD=3`이 FRAG 룰 미러 → 서빙이 요청에서
   `institution_count`를 직접 계산(predictor.py:966). DL의 시계열 예측 우위 없음.
3. `val_best_threshold`=F1-max(학습 prevalence) → 임상 임계 미설정·캘리브레이션 미검증.

## 운영 모니터링

- **`unknown_drug_count` / total** = OOV율 = silent-skew 조기경보(학습 vocab은 2024-08~09 시점).
  임계(예: >30%) 초과 시 score 신뢰도 낮음 → 표시 억제/플래그(임계는 운영 결정 필요).
- 실패 시 `dl_error`(non-blocking, predictor.py:1338/1346-1348) — graceful bypass 유지(적절).

## 승인 경로 (단계별 선결조건)

### B — 별도 "다기관 성향" 임상 알림 (risk_level 불변)
- 임상팀의 명시적 사인오프("history-based drug-pattern fragmentation propensity", **임상위험 아님** 정직 명명)
- shadow-mode 비교(DL high vs 룰 Y_FRAG 일치율)
- 본 모델 카드 + OOV 임계 정의

### C — risk_level/intervention 구동 (근시일 비권장)
- B 전체 + **prospective temporal validation**(same-window 아님) + 임상 캘리브레이션 스터디
  + subgroup/fairness 감사(age·sex·institution type·OOV) + 책임 프로토콜 + 규제/법률 검토

## 학습↔서빙 일관성 (검증 포인트)

- `drug_code`=EDI(`hana_history.py:162-178`) → `_encode_history`(`dl_predictor.py:286`)가 vocab 조회 — 식별자공간 정합.
- 권장 추가검증(B 진행 시): exact parity 샘플 테스트, lookback 경계 parity, vocab hash/input_dim/`_unk` 일관성,
  OOV율 vs 학습 메타, score 분포 drift, multi-hot dedup 동작.

## 변경 이력
- 2026-06-06: 재현(0.844954)→B0 export→DL 서빙 parity 실증(DL high=0.435172==Linear sigmoid). 절차: `docs/ops/sparse-linear-serving-runbook.md`. cross-family Decision A(auxiliary-only) 확정.
