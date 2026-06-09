# Page 4 결과 다운로드 기능 설계

**날짜:** 2026-06-09  
**범위:** `hana_app/pages/4_📊_결과_분석.py` + `hana_app/core/report_exporter.py`  
**상태:** 승인됨

---

## 1. 목표

Page 4 (결과분석) 하단에 두 개의 다운로드 버튼 추가:
- **DOCX 보고서**: 개입 위계 분포 + 모델 성능 + 피처 중요도 + DDI 통계
- **대상자 CSV**: Red / Y_DDI_MAJOR / Y_TRIPLE / Y_DOUBLE 대상자 + 한국어 사유

데이터 출처: `st.session_state["features_df"]` + `st.session_state["last_result"]` (배치 학습 예측 결과)  
저장 방식: `st.download_button` — 메모리(BytesIO) → OS 저장 다이얼로그 (PyWebView 호환)

---

## 2. 아키텍처

### 신규 파일

```
hana_app/core/report_exporter.py
```

공개 함수 2개:

```python
def build_csv_bytes(features_df: pd.DataFrame) -> bytes
def build_docx_bytes(last_result: dict, features_df: pd.DataFrame) -> bytes
```

### 변경 파일

```
hana_app/pages/4_📊_결과_분석.py
```

기존 콘텐츠 하단에 다운로드 섹션 추가. 기존 로직 무변경.

### 데이터 흐름

```
st.session_state["features_df"]  ──► build_csv_bytes()  ──► st.download_button (CSV)
st.session_state["last_result"]  ──►
                                      build_docx_bytes() ──► st.download_button (DOCX)
st.session_state["features_df"]  ──►
```

---

## 3. CSV 명세

### 필터 조건

```python
target_labels = {"Red", "Y_DDI_MAJOR", "Y_TRIPLE", "Y_DOUBLE"}
# risk_level == "Red" OR yellow_subtype in {Y_DDI_MAJOR, Y_TRIPLE, Y_DOUBLE}
```

### 출력 컬럼

| 컬럼명 | 원본 필드 | 설명 |
|--------|-----------|------|
| 환자ID | `patient_id` (실컬럼명 구현 시 확인) | 식별자 |
| 개입조치 | `risk_level` + `yellow_subtype` → 매핑 | 한국어 |
| 위험라벨 | `yellow_subtype` / `risk_level` | 영문 원본 |
| 사유 | 규칙 기반 생성 | 한국어 (B등급) |
| 다약제수 | `drug_count` | |
| 중증DDI건수 | `ddi_major` | |
| 금기DDI건수 | `ddi_contraindicated` | |
| 중복처방수 | `dup_count` | |
| 다기관수 | `institution_count` | |

### 개입조치 한국어 매핑

| 라벨 | 개입조치 |
|------|----------|
| Red | 즉각 개입 |
| Y_DDI_MAJOR | 약사 전화 |
| Y_TRIPLE | 문자 안내 |
| Y_DOUBLE | 모니터링 |

### 한국어 사유 생성 규칙 (rule category, B등급)

```python
def _build_reason(row) -> str:
    label = row.get("yellow_subtype") or row.get("risk_level")

    if label == "Red" or row.get("risk_level") == "Red":
        n = int(row.get("ddi_contraindicated", 0))
        return f"금기 DDI {n}건"

    if label == "Y_DDI_MAJOR":
        n = int(row.get("ddi_major", 0))
        return f"중증 DDI {n}건"

    # Y_TRIPLE / Y_DOUBLE: 충족 차원 조합 나열
    dims = []
    drug_count = row.get("drug_count", 0)
    if drug_count >= 10:
        dims.append(f"다약제({int(drug_count)}종)")
    if row.get("has_high_risk") or row.get("has_renal") or row.get("has_hepatic"):
        dims.append("고위험약물/장기부전")
    inst = row.get("institution_count", 0)
    # MULTI_INSTITUTION_THRESHOLD (serving/schemas.py 상수) 임포트해서 사용
    if inst >= MULTI_INSTITUTION_THRESHOLD:
        dims.append(f"다기관({int(inst)}개)")

    prefix = "3중위험" if label == "Y_TRIPLE" else "2중위험"
    return f"{prefix} — " + "+".join(dims) if dims else prefix
```

### 파일명 & 인코딩

- 파일명: `대상자_위험분류_YYYYMMDD_HHMMSS.csv`
- 인코딩: UTF-8 BOM (`utf-8-sig`) — Excel 한글 깨짐 방지

---

## 4. DOCX 명세

### 라이브러리

`python-docx` (`.venv_hana`에 기설치)

### 섹션 구성

| # | 섹션 | 데이터 출처 |
|---|------|------------|
| 1 | 표지 (보고서명, 생성일시, 모델명) | `last_result["model_name"]`, datetime.now() |
| 2 | 개입 위계 분포 표 | `features_df.groupby` 집계 |
| 3 | 모델 성능 지표 표 | `last_result["metrics"]` |
| 4 | 피처 중요도 Top 15 표 | `last_result["feature_importance"]` |
| 5 | DDI / 다약제 통계 표 | `last_result["ddi_means"]`, `last_result["drug_count_stats"]` |
| 6 | 분석 메모 | 고정 텍스트 |

#### 섹션 2 — 개입 위계 분포 표

| 개입조치 | 라벨 | 건수 | 비율 |
|----------|------|------|------|
| 즉각 개입 | Red | N | x.xx% |
| 약사 전화 | Y_DDI_MAJOR | N | x.xx% |
| 문자 안내 | Y_TRIPLE | N | x.xx% |
| 모니터링 | Y_DOUBLE·Y_DDI_MOD·Y_DUP·Y_FRAG | N | x.xx% |
| 관여 안함 | No_Alert·Green·Normal | N | x.xx% |

#### 섹션 6 — 분석 메모 (고정)

> 본 보고서는 MODE_11_hana 배치 학습 예측 결과 기준입니다.  
> 서빙 실시간 예측과 수치 차이가 있을 수 있습니다.

### 파일명

`위험예측_보고서_YYYYMMDD_HHMMSS.docx`

---

## 5. UI 배치

Page 4 기존 콘텐츠 최하단, `st.divider()` 후:

```python
st.divider()
st.subheader("📥 결과 다운로드")

col1, col2 = st.columns(2)
has_data = (features_df is not None) and (last_result is not None)

with col1:
    # DOCX 버튼
with col2:
    # CSV 버튼

if not has_data:
    st.caption("학습 결과가 로드된 경우에만 활성화됩니다.")
```

**성능:** `build_docx_bytes` / `build_csv_bytes`는 `@st.cache_data`로 래핑 — 리렌더 시 중복 생성 방지.

---

## 6. 에러 처리

| 상황 | 처리 |
|------|------|
| `features_df` / `last_result` None | 버튼 disabled |
| CSV 대상자 0건 | `st.warning("추출 대상 없음")` + CSV 버튼 disabled |
| `ddi_means` / `drug_count_stats` 키 누락 | 해당 DOCX 섹션 "데이터 없음" 표시, 생성 계속 |
| `feature_importance` 비어있음 | 섹션 4 생략 |
| `python-docx` import 실패 | `st.error("docx 라이브러리 미설치")` |

---

## 7. 테스트

파일: `tests/test_ops/test_report_exporter.py`

| 테스트 | 검증 내용 |
|--------|-----------|
| `test_csv_bytes_red_only` | Red만 있는 df → 1행, 사유 "금기 DDI N건" |
| `test_csv_bytes_all_labels` | 4개 라벨 혼합 → 각 행 사유·개입조치 정합 |
| `test_csv_bytes_no_target_rows` | Green만 → 빈 결과 (헤더만) |
| `test_csv_utf8_bom` | 첫 3바이트 == `b'\xef\xbb\xbf'` |
| `test_docx_bytes_returns_bytes` | 반환타입 `bytes`, ZIP magic number (`PK\x03\x04`) |
| `test_docx_missing_ddi_means` | `ddi_means` 키 없어도 예외 없이 생성 |
| `test_docx_empty_feature_importance` | 피처 중요도 없어도 생성 |

---

## 8. 범위 외 (이번 작업 제외)

- 실시간 서빙 API 응답 기반 export (개별 환자 예측 결과)
- 보고서 레이아웃 커스터마이징 UI
- 이메일/공유 기능
- `gen_result_docx.py` CLI 도구 변경
