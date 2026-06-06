# MODE_11_hana 종합 검토 리포트 — ML/DL 개선 + 디스크풀 에러 RCA

- 작성일: 2026-06-02
- 데이터 기준: `data/Raw/records_20240701~20241231.parquet` 184개(2024-07~12, 6개월) + eligibility 50만명. **이 6개월이 최종 데이터셋.** 훈련=07~11, **Nov→Dec=동결 홀드아웃**.
- 방법: orient(직접 확인) → read-only 에이전트 2개 병렬 검토(hermes-worker, general-purpose) → 핵심 버그 직접 재검증.
- 가드레일: RESEARCH_TRACK_FROZEN(Nov→Dec 홀드아웃 튜닝 금지), 학습↔서빙 스키마 정렬, 라벨 정의/HANA/스키마 변경은 cross-family.

---

## A. 디스크풀 에러 RCA (`mode_11_error.txt`)

### 결론: **코드 버그가 아니라 운영 이슈 + 미커밋 방어로직**

증상: `IOException: Could not write file ".../Temp/hana_feat_xxx/_part=48/data_0.parquet" (디스크 공간 부족)` — DuckDB `COPY ... PARTITION_BY` 중.

근본 원인(검증됨):
- 에러를 낸 배포본은 `D:\claude\MODE_11_hana`(별도 체크아웃), temp가 시스템 C: 기본 경로(`AppData/Local/Temp`)로 떨어졌고 그 드라이브가 가득 참.
- 떨어진 건 raw DuckDB `IOException`이지 레포의 친절한 `InsufficientDiskSpaceError`가 **아님** → 방어로직 `_preflight_temp_space`가 실행되지 않음.
- 검증: `git show HEAD:hana_app/core/ml_runner.py | grep -c _preflight_temp_space` → **0**, 워킹트리 → 2. 즉 **방어로직이 워킹트리에만 있고 미커밋**. 배포본=구버전.
- `HANA_FEAT_TMP`/`HANA_TMP_DIR`는 BAT·docs·runbook 어디에도 안내 없음 → 운영자가 temp를 넉넉한 드라이브로 돌릴 방법을 모름.

풋프린트:
- `data/Raw` parquet 총 ~0.99 GB. DuckDB COPY 피크 temp ≈ 소스의 ~2배(버퍼+파티션 동시 기록).
- pandas 폴백 경로(`ml_runner.py:1041-1050`)는 **파일 단위 스트리밍**이라 피크 temp ≈ 소스 1파일. DuckDB 경로만 한 번에 전체 복제.
- `_preflight_temp_space`의 `headroom=10.0`은 실측 ~2배 대비 보수적이나 의도적(공유 드라이브/느린 디스크/단편화 대비, 조기·친절 실패 우선). **유지 권장.**

### 조치 (우선순위)
- **P0 (안전, freeze-safe)**: 워킹트리의 `_preflight_temp_space` + `_resolve_feat_tmp_base` + `InsufficientDiskSpaceError` + 호출부(`ml_runner.py:984-988`)를 **커밋**. 효과: raw IOException → "약 N GB 필요 / X·Y·Z 드라이브 여유" 친절 메시지.
- **P1 (운영 안내)**: `HANA_FEAT_TMP` 사용법을 runbook + BAT(`install_312.bat`/`run.bat`, CRLF+chcp 65001 유지) + CLAUDE.md ETL 절에 명시. 예: `set HANA_FEAT_TMP=D:\hana_tmp`(여유 10GB+).
- **P2 (구조, 후순위)**: DuckDB COPY를 소스 파일 단위 증분 COPY로 분할해 temp 풋프린트 축소. preflight로 급한 불은 꺼졌으므로 대코호트(>5GB) 상시화 시 재검토.

---

## B. ML/DL freeze-safe 개선 후보

### B0 (P0) — sparse-linear baseline 운영화: 익스포터 + OOV 정합 + 계약 테스트

**핵심**: 메모리상 "sparse_linear(AUC 0.845)은 /predict 서빙 불가"로 기록됐으나, 실제로 **서빙 기계장치는 이미 존재·테스트됨**. 빠진 건 "훈련 가중치→번들" 익스포터 하나.
- 서빙 로더 `serving/dl_predictor.py`(`DLModel`) + 핫스왑 `serving/predictor.py:1044 reload_dl()` + `tests/test_serving/test_admin_reload_dl.py` 已 wired & tested.
- 번들 원형 `scripts/datasets/smoke_dl_bundle.py`: `nn.Linear`→`jit.trace`→multi_hot 번들 생성기 존재. 단 가중치 하드코딩 랜덤 + vocab 3-drug 토이. 아키텍처는 훈련 linear와 동일.
- 즉 P0 = "서빙 경로 신설"이 아니라 **smoke 번들러를 실모델 익스포터로 일반화**.

**🔴 함께 고쳐야 할 확정 결함 — OOV 인코딩 train/serve drift (직접 확인)**
- 학습: `scripts/ops/multihot_encoder.py:30-32` → 미지 약물 `vector[unk_index]=1.0`.
- 서빙: `serving/dl_predictor.py:283-284` → 미지 약물 `unknown_count+=1; continue` (`_unk` 차원 **절대 set 안 함**).
- 결과: `_unk` 활성으로 학습된 모델을 `_unk`=0으로 서빙 → silent skew. tabular 쪽 `_validate_feature_schema`가 막는 부류의 사고인데 **DL 경로엔 등가 가드 없음**.
- 조치: 익스포터가 OOV 정책을 번들 메타로 명시 + `_encode_history`가 `_unk` set + **훈련 인코더 == 서빙 인코더 동등성 계약 테스트**.

**프레이밍(오버셀 방지)**: `serving/predictor.py:1159 final_level = max(rule_level, ml_level)` — `dl_prediction`은 별도 응답 필드, 최종 등급에 합산 안 됨. 따라서 이 운영화 = **주 등급 옆 shadow/aux 스코어**(위험 모델 교체 아님). 이게 freeze-safe인 이유.

**🚩 Critical (cross-family 필요)**: `multi_institution`/`therapeutic_duplication`은 **proxy 라벨**(`build_sparse_training_dataset.py:178,191`). proxy 스코어를 환자 "위험" 출력으로 서빙 = 라벨 정의 결정 → 서빙 노출 전 cross-family 리뷰 필수. 홀드아웃: same-window 학습 한정, `*_disjoint_octnov` 튜닝 금지.

- 임팩트 높음 / 난이도 중 / 리스크 낮음(분리 트랙).

### B1 (P1) — mlp_smoke 모델 저장 결함 (직접 확인)
- `scripts/ops/mlp_smoke_train.py:386-391 _save_model_smoke`가 학습 모델이 아니라 **새 untrained `MultiHotMLP`의 state_dict**를 저장. `train_mlp_smoke`가 학습한 모델은 함수 로컬이라 버려짐. 게다가 `state_dict`라 `DLModel`(TorchScript 기대)이 로드 불가.
- 조치: `train_mlp_smoke`가 모델 반환 → `jit.trace`로 저장(B0 익스포터와 경로 공유). 가드레일 저촉 없음.

### B1 (P1) — hierarchical CV `Y_OTHER` 학습/평가 라벨 불일치
- 평가 `hana_app/core/hierarchical_cv.py:46-54`는 `Y_OTHER`를 No_Alert로 폴백. 학습 `hierarchical_runner.py:53`은 `Y_OTHER`를 학습셋에서 제외(ValueError). → 학습 분포에 없는 샘플이 평가에 No_Alert로 유입돼 Stage2 메트릭 왜곡.
- 조치: 평가에서도 `Y_OTHER` mask out(학습과 동일) 또는 별도 버킷 집계. 결정 전 의도 확인 필요.
- (참고: 이 항목은 에이전트 보고 기반, fix 착수 시 재검증 권장.)

### B2 (P2) — 피처 효율 / 테스트 갭
- `multihot_encoder.encode_batch`(`:59`) 전체 그룹 dict 물질화 — 대코호트 메모리 비효율(smoke 무방, 운영 배치 시 청크).
- 테스트 갭: (a) 훈련↔서빙 multi_hot **인코더 동등성** 계약 테스트 부재(B0 OOV drift 잡을 테스트 없음), (b) 실모델 익스포터 round-trip(export→validate→reload→predict).
- sparse 트랙은 tabular `FEATURE_COLS`(`ml_runner.py:50`)와 별개 컨트랙트라 `RequestFeatureBuilder` 정렬 리스크 없음(의도적 격리, `sparse_research.py:3-5`). 양호.

---

## C. 권장 실행 순서 (게이팅)

1. **즉시 안전(승인 시 바로)**: A.P0 preflight 커밋, A.P1 runbook/BAT/CLAUDE.md HANA_FEAT_TMP 안내, B1 mlp_smoke 저장 결함 수정.
2. **검증 후**: B1 hierarchical Y_OTHER(의도 확인 후), B2 인코더 동등성 테스트 추가.
3. **cross-family 사인오프 후에만**: B0의 익스포터+OOV 정합을 **서빙 노출**까지 가는 부분(proxy 라벨→위험 출력). 단, OOV 정합 자체(인코더 버그 수정 + 계약 테스트)는 서빙 노출과 분리해 먼저 진행 가능.

검증 근거 파일: `hana_app/core/ml_runner.py`(preflight 미커밋·COPY), `serving/dl_predictor.py:272-292`(OOV), `scripts/ops/multihot_encoder.py:30-32`(학습 _unk), `scripts/ops/mlp_smoke_train.py:386-391`(untrained 저장), `scripts/datasets/smoke_dl_bundle.py`(익스포터 원형), `serving/predictor.py:1159`(dl=aux).

---

## D. 실행 결과 (2026-06-02, 사용자 승인 3개 워크스트림)

### ✅ WS3 — OOV 인코더 train/serve 정합 + 계약 테스트 (완료·검증·cross-family 실시)
- `serving/dl_predictor.py:_encode_history`: 미지 약물을 vocab `_unk` 차원에 반영(학습 인코더와 정합). `_unk` 없는 구형/토이 번들은 종전대로 무시(하위호환). `pd.isna`로 None/np.nan/pd.NA skip + 빈 코드 skip(학습 `dropna` 정합).
- 신규 `tests/test_serving/test_dl_encoder_parity.py` 4건: 학습 `encode_patient_history` == 서빙 `_encode_history` 동등성(known/OOV/all-known/None·NaN·pd.NA·empty).
- 검증: `pytest test_dl_encoder_parity.py test_dl_predictor.py` → **11 passed**(신규 4 + 기존 7, 하위호환 유지).
- **cross-family 검증 완료(codex-bridge, CLAUDE.md 필수 룰)**: multi_hot 정합 **OK 확인**. 단 아래 E절의 추가 발견(P1/P2/P3)을 surface. 특히 **P2(HANA 코드 네임스페이스)는 본 수정 범위 밖의 잠재적 상위 결함**.

### ✅ WS2 — mlp_smoke 저장 결함 (코드 수정 완료, torch env 검증 대기)
- `scripts/ops/mlp_smoke_train.py`: `train_mlp_smoke(..., return_model=True)`로 **학습된 모델** 반환(기존 호출부/테스트 하위호환). `_save_model_smoke(model, input_dim, path)`가 untrained 새 모델 대신 학습 모델을 **`torch.jit.trace`→TorchScript**로 저장(`DLModel`이 `torch.jit.load` 기대, dl_predictor.py:201). `run_raw_training_smoke`가 학습 모델을 저장에 전달.
- 검증: `py_compile` OK, 다른 `_save_model_smoke` 호출부 없음. **torch 미설치 로컬 env라 torch 실행 테스트는 배포/CUDA env에서 수행 필요.**

### ✅ WS1 — 디스크 P1 문서화 (완료) / P0 커밋 (사용자 확인 대기)
- P1: `CLAUDE.md` ETL 절에 `HANA_FEAT_TMP` 안내 추가. 신규 `docs/ops/feature-build-temp-disk-runbook.md`(증상·원인·조치·우선순위·BAT 안내). BAT는 CRLF(LF 금지) 가드레일상 직접 미편집 — runbook에 CRLF 보존 안내로 제시.
- P0(preflight 커밋): `_preflight_temp_space`는 워킹트리에 존재하나, `ml_runner.py` 미커밋 diff(+166)에 memory guard/progress_cb 등 **별도 in-flight 변경 ~14 hunk가 섞임** → 파일 통째 커밋은 무관 작업을 엮으므로 자동 커밋 보류. **분리 커밋 여부는 사용자 결정 필요.**

### 미커밋 상태
모든 변경은 워킹트리에만 있음(자동 커밋 안 함). 로컬 env 의존성 누락(`filelock`, `torch` 미설치)으로 일부 무관 서빙 테스트는 ERROR/실패(사전 존재, 내 변경 무관). torch 휠은 `packages_win/py312`에 win_amd64만 있어 Linux형 로컬 bash env에서 설치 불가 → torch 라운드트립은 Windows env 필요.

---

## E. cross-family(codex) 추가 발견 — 본 수정 범위 밖, 사용자 결정 필요

| P | 발견 | 심각도 | 근거 |
|---|---|---|---|
| **P2** | **확정(코드 검증)**: 학습 vocab = `edi_code` = **MCARE_DIV_CD**(EDI 약품코드, `full_cohort_history_loader.py:110` **및** `multi_day_parquet_provider.py:150` 양쪽 provider 모두 + `config.py:42,63`). 서빙 DL 인코더는 `history_df["drug_code"]` 조회 = **WK_COMPN_CD(T30)/GNL_NM_CD(T60)**(주성분/일반명코드, `hana_history.py:159,169` + `config.py:40,61`). **서로 다른 약물 코드체계** → 서빙이 주성분코드를 EDI vocab 에 조회 → 거의 전부 OOV. DL 서빙(sparse/multihot)이 상위 계약에서 깨짐. | **확정·高** | cross-family 검증 완료. **수정 후보**: 서빙 프레임이 `edi_code`(MCARE_DIV_CD)를 별도 컬럼으로 이미 보유(`hana_history.py:180`) → DL 인코더가 `drug_code` 대신 `edi_code` 조회하도록 정렬 가능. 단 학습↔서빙 계약·라벨 정의 → 사용자+cross-family 결정 필요. |
| **P1** | `serving/dl_predictor.py:_load_drug_vocab`가 `_unk` 존재를 검증하지 않음. 운영 vocab(`build_drug_vocab.py:47`)은 항상 `_unk` 포함하나 smoke 번들(`smoke_dl_bundle.py`)은 미포함 → OOV가 조용히 드롭(로그 없음). | 중(구체 경로 존재) | 하드닝: `_load_drug_vocab`에 `_unk` 체크 + smoke 번들에 `_unk` 추가. 단 `_unk` 필수화 시 하위호환 테스트 갱신 필요(설계 결정). |
| **P3** | `_SUPPORTED_ENCODING_STRATEGIES`의 `"count"`는 학습측 인코더가 전무(dead infra). 오설정 번들이 조용히 수용됨. | 낮음(수동 오설정 필요) | `dl_predictor.py:29` allowlist를 `{multi_hot}`로 좁히거나 schema_version 가드. |

> P2는 본 OOV 수정으로 해결되지 않는 **별개의 상위 계약 문제**다. edi_code ↔ WK_COMPN_CD/GNL_NM_CD 가 동일 네임스페이스인지 사용자 확인 전까지 DL 서빙 운영화는 보류 권장.
