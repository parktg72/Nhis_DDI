# 폐쇄망 ML/DL 3증상 수정 설계

작성 2026-07-06. 근본원인은 정적추적 + ground-truth 검증으로 **확정**. 적용은
폐쇄망 재현/검증 통과 후(Iron Law), 학습 스키마 건드리는 항목은 cross-family 게이트 후.

전국민 확대(질환별/샘플링 추출) 전 ①③ 반드시 반영 — 코호트 커지면 결측·비다제 비율이
달라져 왜곡 폭이 커진다.

---

## 증상 ① 성별 여성 치우침

### 근본원인 (확정)
- `hana_app/core/ml_runner.py:459` — `"sex_m": 1 if f.sex == "1" else 0`. 비대칭 인코딩:
  정확히 문자열 `"1"`만 남(male), 여("2")·불명("U")·None·결측 전부 `0`.
- `hana_app/core/ml_runner.py:870/1239/1293` — `sex=demo.get("sex_type")`. demographics 결측 →
  `None` → sex_m=0 전원.
- `hana_app/core/ml_runner.py:~1572` — 데모 read 시 `str(row.sex_type)`. parquet가 float 저장이면
  `str(1.0)=="1.0" != "1"` → **남자도 sex_m=0**.
- `hana_app/core/report_exporter.py:1194` — `sex_f = total_n - sex_m`. 여를 뺄셈으로 추론 →
  결측·불명 전부 여로 계상. 데모 결측이면 리포트 "남 0명 / 여 100%".

### 수정 설계

**(1-a) 리포트側 — SAFE (report-only, 스키마 무관, 우선 적용 가능)**
- `report_exporter.py` 성별 집계를 뺄셈 폐기 → 실제 값 3-way 카운트:
  남 = `sex_m==1`, 여 = 명시적 여 지표, 불명 = 그 외.
- features_df에 여/불명 지표가 없으면 **불명 버킷을 별도 행으로 노출**
  (`["성별 — 불명", "K명"]`) — 결측을 여로 숨기지 말 것. 왜곡이 리포트에 드러나게.
- 회귀 테스트: sex_m 전부 0인 df → "남 0 / 여 0 / 불명 100%" (여 100% 아님).

**(1-b) 피처側 — CRITICAL (학습↔서빙 동시 + cross-family 게이트 + 재학습)**
- demographics 로드에 `_normalize_sex(raw) -> "1" | "2" | None` 헬퍼 도입:
  `"1"/"2"`, `"M"/"F"`, float `1.0/2.0`, 공백/None 모두 정규화. float-string 버그
  (`str(1.0)`) 근본 해소.
- train↔serve 결측 처리 정렬: 서빙 `serving/predictor.py:1029`는 결측 sex→**0.5**(중립),
  학습은 hard 1/0 → **불일치 존재**. 학습도 결측→0.5(또는 명시적 sex_unknown 피처)로
  맞추면 스키마/값 변경 → **재학습 필수 + `RequestFeatureBuilder` 컬럼 parity 확인**.
  → 결정 필요: (A) sex_m 유지 + 결측 0.5 정렬(최소변경), (B) sex_male/sex_female/sex_unknown
  3-way 명시 피처(권장, 결측을 여로 오염 안 시킴). 둘 다 재학습.
- RAW 모드 fail-loud: `pages/3_🤖_모델_학습.py` `_ensure_demographics_from_raw` 미실행/
  `DEMOGRAPHICS_PATH` 결측이면 **조용한 sex=0 대신 명시적 에러**(프로젝트 fail-closed 이토스).

### 폐쇄망 검증 (수정 전)
1. `DEMOGRAPHICS_PATH`(`data/raw/eligibility_demographics.parquet`) 존재?
2. `features_df["sex_m"].value_counts()` — 거의 0이면 확정.
3. `pd.read_parquet(DEMOGRAPHICS_PATH)["sex_type"].dtype` + `.value_counts()` — 문자열 "1"/"2"인가 float인가.
4. 원본 `SELECT SEX_TYPE, COUNT(*) FROM ... GROUP BY SEX_TYPE` — 진짜 코호트 분포 확인(실제 여초과 배제).

---

## 증상 ② review/red 임계값 1.00인데 Red 의심

### 근본원인 (확정) — "1.00"은 반올림 착시, 동작은 정상
- prod `hana_app/models/hierarchical/retrain_prod_0711_hierarchy_cur/stage_meta.json`:
  **`tau_red=0.9999722`, `tau_review=0.9999712`** (1.00 아님).
- UI/로그가 `:.2f`/`:.3f`로 반올림: `pages/4_📊_결과_분석.py:108-109`,
  `pages/3_🤖_모델_학습.py:2040-2041,2204`, `hierarchical_runner.py:749`.
- rulefeat.v1 stage-1 과확신 → `p_red≈0.99998` → `hierarchical_runner.py:724` `p_red>=tau_review`
  → "Red 의심"(`predictor.py:1485`) 정상. `:709` `p_red>=tau_red` → 확정 Red.
  **확률 1.0 불필요, 0.99997만 넘으면 됨.**
- 최종 Red는 모델점수와 무관한 **금기 결정적 백스톱**(`predictor.py:1432-1445`)에서도 독립 발생.
- 숨은 red flag: `hierarchical_runner.py:324-325` `tau_review=tau_red-1e-6` 강제. PR커브 붕괴
  (거의 완벽분리=과확신) 신호. review 밴드 폭 **1e-6** → 사실상 아무것도 그 구간에 안 들어옴 =
  score 기반 review 큐가 死.

### 수정 설계

**(2-a) 표시 — SAFE cosmetic (우선 적용)**
- tau_red/tau_review 포맷 `:.2f`/`:.3f` → **`:.6f`**(또는 지수표기). 위 4개 사이트.
  운영자가 0.999972로 보게 → 착시 해소.

**(2-b) 모델링 — 코드 fix 아님, 표면화/결정 필요 (labeling·cross-family)**
- tau가 천장에 붙고 밴드 1e-6 = stage-1이 rulefeat.v1(라벨 준-동어반복 피처)에 과확신.
  선택지: (i) stage-1 확률 보정(isotonic/Platt) 후 재선택, (ii) rulefeat.v1 누수 재검토,
  (iii) Red를 룰/백스톱 주도로 인정하고 score-review 밴드 死 문서화.
- 로드 로그에 **full precision tau + 밴드폭 경고**(폭<1e-3이면 "PR curve collapsed —
  score review band inert") 추가. SAFE.

### 폐쇄망 검증
- 배포된 **active** 모델 stage_meta.json을 반올림 없이 열어 tau 확인. ⚠️ 구 모델들은
  tau_red가 0.0086 등 완전 다름 → 어느 모델이 실제 서빙 중인지부터 확정.

---

## 증상 ③ 분석대상 N ≠ 추출 대상자 N

### 근본원인 (확정) — 의미론차(대부분 정상) + 리포트 내부불일치(실버그)
- 추출 N: `hana_etl.py:1197` distinct patient, **poly 필터 없음**.
- 분석 N: `report_exporter.py:1177` dedup + **다제 poly_threshold=5 필터**
  (`ml_runner.py:858/1226/1280/1434` 성분<5명 제거) 통과분.
- 지배 갭 = 다제필터(설계상 정상, 추출=전원 vs 분석=다제만).
- 실버그(내부불일치): 같은 리포트가 총N 2방식 — §1 `total=len(features_df)` raw
  (`report_exporter.py:920`) vs §7 `total_n` dedup(`:1177`). patient_id 중복(재사용·concat
  데이터셋) 있으면 두 총N 불일치. id-less 행은 dedup이 보존(`:563-567`) → 카운트 방식차 추가.

### 수정 설계

**(3-a) 의미 투명화 — SAFE**
- 리포트 §7 + Page3/4에 **깔때기 명시**: "추출 대상자 N(전체) → 다제(≥5성분) 통과 M →
  분석대상 M". drop 수·사유 한 줄. 갭이 정상임을 운영자가 알게.
- 라벨을 "추출 대상자 N" / "분석대상 N(다제)"로 구분 표기.

**(3-b) 내부 총N 기준 통일 — SAFE report-only 버그 fix**
- `analysis_subject_df`를 한 번 계산해 §1·§7·CSV·Page3/4 **모두 동일 기준** 사용.
  distinct 환자수는 `nunique(patient_id)` 명시.
- 회귀 테스트: patient_id 중복 있는 features_df → §1 총N == §7 총N.

### 폐쇄망 검증
- `features_df['patient_id'].nunique()` vs `len(features_df)` — 다르면 중복존재 = 실버그 활성.
- 추출 unique_patients vs 다제통과수 vs len(features_df) vs dedup 후 — 깔때기 각 단계 수치 대조.

---

## 적용 순서 & 게이트

| 우선 | 항목 | 위험 | 게이트 | 검증 |
|---|---|---|---|---|
| 1 | 2-a 표시정밀도, 3-a 깔때기, 3-b 총N통일, 1-a 리포트 성별 3-way | SAFE (report/UI) | 없음 | Windows .venv pytest + 실클릭 |
| 2 | 2-b 로드로그 밴드경고 | SAFE | 없음 | 단위테스트 |
| 3 | 1-b sex 정규화+결측정렬+fail-loud | **CRITICAL** | 학습↔서빙 동시 + cross-family(Anthropic↔OpenAI) + 재학습 + `/reload` sanity | 폐쇄망 재현→feature schema diff→parity test |
| 4 | 2-b 확률보정/rulefeat 누수 | **CRITICAL** labeling | cross-family + 홀드아웃 주의(freeze 가드) | 별도 |

- 1단계(SAFE)는 폐쇄망 검증 없이도 코드상 옳음이 자명 → 먼저 적용 가능. 단 WSL은
  `.venv`가 Windows라 pytest 불가 → Windows env에서 러너 실행 필수(프로젝트 규약).
- 3·4단계는 학습 피처/라벨 = critical. **반드시** 폐쇄망 재현으로 가설 확정 후,
  학습·서빙 동시수정 + cross-family 게이트. sex_m 단독 변경 금지(서빙 parity 회귀 위험).

## 미결 결정 (사용자)
- 1-b: sex 인코딩 (A) sex_m+결측0.5 최소변경 vs (B) 3-way 명시피처. → 재학습 범위 결정.
- 2-b: score-review 밴드 死 수용 vs stage-1 재보정. → labeling 트랙(freeze 가드 확인).
