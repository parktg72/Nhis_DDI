# Serving DDI — EDI→WK 커버리지 & 미매핑 degraded 운영 가이드

> Task B(serving DDI train/serve parity)의 P1 후속. 서빙 DDI 의 EDI→WK 브릿지 누락
> 동작과 그 임상중요도, 운영 모니터링·맵 갱신 절차를 정리한다.

## 1. 배경 — 왜 degraded 가 생기나

- **학습**은 NHIS 레코드의 `wk_compn_cd`(주성분코드)를 직접 가진다 → `DrugMaster.get_ddi_ids`
  로 DB-code(DrugBank id) 쌍을 만들어 동시복용(overlap) 기준 DDI 를 카운트한다.
- **서빙**은 요청에서 `edi_code`(제품코드)만 받는다. 따라서 `edi→wk` 브릿지가 필요하다:
  `data/processed/edi_to_wk.parquet`(HIRA 약제급여목록 xlsx 의 제품코드→주성분코드,
  `scripts/ops/build_edi_wk_map.py` 가 생성).
- 이 맵에 **없는 edi 는 서빙이 wk 를 못 얻어 DDI 평가에서 제외**한다(`RequestFeatureBuilder
  ._build_ddi_records` 가 `continue`). 이때 1회 경고 로그를 남긴다. **원칙: "미매핑 ≠ DDI
  음성"** — 누락은 "DDI 없음"이 아니라 "평가 불가(degraded)"다.

## 2. 임상중요도 측정 결과

측정 스크립트: `python -m scripts.ops.measure_edi_wk_coverage`
(raw records 는 edi+wk 를 동시 보유하므로 ground-truth 로 누락 영향을 정량화한다.)

**기준 데이터: `records_20240701.parquet`** (1일, 환자 50,061명 / 처방 505,404행, 쌍단위 5,000명 샘플)

| 지표 | 값 |
|---|---|
| EDI 맵 커버리지 | **96.0%** (14,419 / 15,017 unique edi) |
| 미매핑 edi | 598 |
| 미매핑 중 DDI-capable(records-wk 기준) | 48.7% (291) |
| 처방행 미매핑 비율 | 2.25% |
| 처방행 미매핑-DDI 비율 | 1.10% |
| 미매핑-DDI 약물에 노출된 환자 | 6.15% (3,079 / 50,061) |

**쌍 단위 — 서빙이 실제로 놓치는 DDI 이벤트 (5,000명 샘플, full[records-wk] vs serving[map]):**

| Severity | full | serving | 누락 | 누락률 |
|---|---|---|---|---|
| **Contraindicated** | 7 | 7 | **0** | **0.0%** |
| **Major** | 443 | 420 | 23 | **5.2%** |
| Moderate | 7,978 | 7,633 | 345 | 4.3% |
| Minor | 60 | 58 | 2 | 3.3% |

### 해석 (운영 판단)

- ✅ **금기(Contraindicated) = 0% 누락**. 가장 위험한 등급은 서빙이 전부 포착한다.
- ⚠️ **Major ≈ 5% 누락** (~20건 중 1건). Moderate ≈ 4%. degraded 가 **저중증 쪽으로
  치우쳐** 임상 안전망의 핵심(금기)은 유지된다.
- 누락은 **위양성이 아니라 위음성(놓침)** 방향이다 → 서빙이 "DDI 없음"으로 과신하지 않도록
  degraded 표시/경고가 중요(아래 §3).

> 주의: 1일 데이터 기준. 월 단위 재측정 시 수치 변동 가능. 맵 갱신 후 반드시 재측정한다.

## 3. 운영 모니터링

- 서빙 로그의 `DDI: edi→wk 미매핑 N/M 건 — DDI 평가서 제외(degraded, 미매핑≠음성)` 경고
  **비율을 모니터링**한다. 급증은 (a) 신약/신제품코드 유입 (b) HIRA 맵 노후화 신호.
- degraded 가 발생한 응답은 "DDI 안전"으로 해석하면 안 된다. 임상 UI/운영팀에 **"일부 약물
  DDI 평가 제외됨"** 을 노출하는 것을 권장(후속 과제).

## 4. 맵 갱신 절차 (커버리지 개선)

1. 최신 HIRA 약제급여목록 xlsx 를 `hira/약제급여목록및급여상한금액표.xlsx` 로 교체.
2. 재빌드: `python -m scripts.ops.build_edi_wk_map`
   (제품코드→주성분코드, edi 9자리 정규화, edi→wk 함수성 검증=충돌 시 빌드 실패).
3. **재측정**: `python -m scripts.ops.measure_edi_wk_coverage` 로 커버리지·누락률 확인.
4. `data/processed/edi_to_wk.parquet` 배포(참조 아티팩트, 커밋 추적됨).

> 커버리지를 더 높이려면 records 자체의 edi→wk(코호트 100%)로 맵을 보강하는 방안도 있으나,
> 서빙 배포 안정성을 위해 **권위 출처(HIRA 급여목록)** 를 기본으로 한다(Codex Q6③ 합의).

## 5. 알려진 한계 / 후속

- 미매핑 Major ~5% 는 맵 갱신으로 줄일 수 있으나 0 이 되긴 어렵다(비급여·신제품 등).
- 본 측정은 약물-개체/쌍 단위. `drug_count`/`dup_same_ingredient` 등 다른 피처의
  train/serve 정합은 **별도 과제**(Task B 범위 밖, Codex 리뷰 P2).
- 관련: [[serving DDI parity]] (Task B), `tests/test_serving/test_ddi_train_serve_parity.py`.
