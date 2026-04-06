# serving/middleware.py + schemas.py 테스트 보강 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `RequestLoggingMiddleware` 미처리 예외→500 경로 테스트, 나머지 스키마 엣지케이스(DrugItem 필드 경계값, PredictRequest 선택 필드 검증) 테스트 추가.

**Architecture:** `tests/test_serving/test_middleware_schemas.py` (신규). FastAPI `TestClient`와 `starlette.testclient` 직접 사용. 기존 `test_serving.py`의 스키마 테스트와 중복되지 않도록 미커버 케이스만 추가.

**Tech Stack:** pytest, fastapi, starlette, httpx

---

### Task 1: `RequestLoggingMiddleware` 미처리 예외 → 500 JSON 응답 테스트

**Files:**
- Create: `tests/test_serving/test_middleware_schemas.py`

- [ ] **Step 1: 테스트 파일 생성**

```python
"""serving/middleware.py + schemas.py 미커버 엣지케이스 테스트."""
import pytest
from datetime import date
from fastapi import FastAPI
from fastapi.testclient import TestClient

from serving.middleware import RequestLoggingMiddleware
from serving.schemas import (
    DrugItem, PredictRequest, RiskLevel, Severity,
)


# ─── 미들웨어 테스트 ──────────────────────────────────────────────────────────

def _make_crash_app() -> FastAPI:
    """특정 경로에서 예외를 발생시키는 최소 FastAPI 앱."""
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ok")
    def ok():
        return {"status": "ok"}

    @app.get("/crash")
    def crash():
        raise RuntimeError("의도적 충돌")

    return app


@pytest.fixture
def crash_client():
    with TestClient(_make_crash_app(), raise_server_exceptions=False) as c:
        yield c


class TestRequestLoggingMiddleware:

    def test_ok_request_returns_200(self, crash_client):
        """정상 요청 → 200."""
        r = crash_client.get("/ok")
        assert r.status_code == 200

    def test_unhandled_exception_returns_500(self, crash_client):
        """처리되지 않은 예외 → 500 JSON."""
        r = crash_client.get("/crash")
        assert r.status_code == 500

    def test_500_response_has_request_id_in_body(self, crash_client):
        """500 응답 body에 request_id 포함."""
        r = crash_client.get("/crash")
        body = r.json()
        assert "request_id" in body
        assert body["request_id"]  # 비어있지 않음

    def test_500_response_has_detail(self, crash_client):
        """500 응답 body에 detail 포함."""
        r = crash_client.get("/crash")
        body = r.json()
        assert "detail" in body

    def test_x_request_id_header_on_ok(self, crash_client):
        """정상 응답 헤더에 X-Request-ID 포함."""
        r = crash_client.get("/ok")
        assert "x-request-id" in r.headers
        assert len(r.headers["x-request-id"]) > 0

    def test_x_request_id_header_on_error(self, crash_client):
        """500 응답 헤더에도 X-Request-ID 포함."""
        r = crash_client.get("/crash")
        # 미들웨어가 예외를 JSONResponse로 변환 후 헤더 주입
        # 참고: 미들웨어 내에서 헤더를 주입하므로 500에도 있어야 함
        assert "x-request-id" in r.headers

    def test_x_elapsed_ms_header_present(self, crash_client):
        """X-Elapsed-Ms 헤더가 정상 응답에 포함."""
        r = crash_client.get("/ok")
        assert "x-elapsed-ms" in r.headers
        elapsed = float(r.headers["x-elapsed-ms"])
        assert elapsed >= 0.0
```

- [ ] **Step 2: 테스트 실행 — RED/GREEN 확인**

```bash
python3 -m pytest tests/test_serving/test_middleware_schemas.py::TestRequestLoggingMiddleware -v
```
Expected: 현재 미들웨어 구현에 따라 일부 FAIL 가능.

`test_x_request_id_header_on_error`가 실패하면 미들웨어 구현에서 JSONResponse 생성 후 헤더 주입 여부 확인.

현재 `serving/middleware.py:33-50`:
```python
try:
    response = await call_next(request)
except Exception as exc:
    ...
    response = JSONResponse(status_code=500, content={...})   # ← 헤더 없음

elapsed_ms = ...
response.headers["X-Request-ID"] = request_id   # ← 모든 응답에 주입됨
response.headers["X-Elapsed-Ms"] = f"{elapsed_ms:.1f}"
```
JSONResponse도 `finally` 블록 이후 헤더가 주입되므로 모든 테스트 PASS 예상.

- [ ] **Step 3: 테스트 실행 — GREEN 확인**

```bash
python3 -m pytest tests/test_serving/test_middleware_schemas.py::TestRequestLoggingMiddleware -v
```
Expected: `7 passed`

- [ ] **Step 4: 커밋**

```bash
git add tests/test_serving/test_middleware_schemas.py
git commit -m "test: RequestLoggingMiddleware 500 경로·헤더 테스트 7건"
```

---

### Task 2: `DrugItem` + `PredictRequest` 미커버 스키마 엣지케이스 테스트

**Files:**
- Modify: `tests/test_serving/test_middleware_schemas.py`

참고: `test_serving.py`에 이미 존재하는 테스트:
- `test_drug_item_validation_empty_edi` — edi_code 빈 문자열
- `test_drug_item_total_days_ge1` — total_days >= 1
- `test_drug_item_total_days_le365` — total_days <= 365
- `test_predict_request_empty_drugs` — 빈 약물 목록
- `test_predict_request_default_date_set` — reference_date 자동 설정

아래는 아직 없는 케이스만 추가.

- [ ] **Step 1: 미커버 스키마 테스트 추가 (같은 파일에 이어서)**

```python
# ─── 스키마 엣지케이스 테스트 ─────────────────────────────────────────────────

class TestDrugItemEdgeCases:

    def test_drug_item_dose_once_must_be_positive(self):
        """dose_once <= 0 → ValidationError."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            DrugItem(edi_code="A001", total_days=7, dose_once=0.0)

    def test_drug_item_dose_freq_max_10(self):
        """dose_freq > 10 → ValidationError."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            DrugItem(edi_code="A001", total_days=7, dose_freq=11)

    def test_drug_item_dose_freq_min_1(self):
        """dose_freq < 1 → ValidationError."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            DrugItem(edi_code="A001", total_days=7, dose_freq=0)

    def test_drug_item_optional_fields_default(self):
        """atc_code, drug_name, start_date, institution_id 모두 None 허용."""
        d = DrugItem(edi_code="A001", total_days=7)
        assert d.atc_code is None
        assert d.drug_name is None
        assert d.start_date is None


class TestPredictRequestEdgeCases:

    def test_single_drug_allowed(self):
        """약물 1개도 유효."""
        req = PredictRequest(
            patient_id="TEST001",
            drugs=[DrugItem(edi_code="A001", total_days=7)],
        )
        assert len(req.drugs) == 1

    def test_reference_date_none_replaced_by_today(self):
        """reference_date=None → model_validator가 date.today()로 대체."""
        from datetime import date
        req = PredictRequest(
            patient_id="TEST001",
            drugs=[DrugItem(edi_code="A001", total_days=7)],
            reference_date=None,
        )
        # set_default_dates() model_validator가 None을 date.today()로 교체한다
        assert req.reference_date == date.today()

    def test_patient_age_none_is_valid(self):
        """patient_age=None 허용."""
        req = PredictRequest(
            patient_id="TEST001",
            drugs=[DrugItem(edi_code="A001", total_days=7)],
            patient_age=None,
        )
        assert req.patient_age is None

    def test_patient_sex_invalid_raises(self):
        """patient_sex M/F 외 값 → ValidationError (Field pattern='^[MF]$' 검증)."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            PredictRequest(
                patient_id="TEST001",
                drugs=[DrugItem(edi_code="A001", total_days=7)],
                patient_sex="X",  # M 또는 F만 허용
            )


class TestRiskLevelEdgeCases:

    def test_max_with_equal_levels(self):
        """동일 등급 max → 동일 등급 반환."""
        assert RiskLevel.max(RiskLevel.RED, RiskLevel.RED) == RiskLevel.RED
        assert RiskLevel.max(RiskLevel.NORMAL, RiskLevel.NORMAL) == RiskLevel.NORMAL

    def test_max_all_combinations(self):
        """Red > Yellow > Green > Normal 순서 보장."""
        levels = [RiskLevel.RED, RiskLevel.YELLOW, RiskLevel.GREEN, RiskLevel.NORMAL]
        for i in range(len(levels)):
            for j in range(len(levels)):
                result = RiskLevel.max(levels[i], levels[j])
                expected = levels[min(i, j)]  # 낮은 인덱스 = 높은 등급
                assert result == expected, f"max({levels[i]}, {levels[j]}) = {result}, expected {expected}"
```

참고: `patient_sex` 검증은 이미 `Field(pattern="^[MF]$")`로 구현됨.
별도 `@field_validator` 추가 불필요 — `test_patient_sex_invalid_raises`는 기존 스키마로 바로 GREEN.

- [ ] **Step 2: 테스트 실행 — GREEN 확인**

```bash
python3 -m pytest tests/test_serving/test_middleware_schemas.py -v
```
Expected: `7 + 10 = 17 passed`

- [ ] **Step 3: 전체 테스트 통과 확인**

```bash
python3 -m pytest --tb=short -q
```
Expected: 기존 포함 모두 PASS

- [ ] **Step 4: 커밋**

```bash
git add tests/test_serving/test_middleware_schemas.py
git commit -m "test: DrugItem·PredictRequest·RiskLevel 스키마 엣지케이스 테스트 10건"
```
