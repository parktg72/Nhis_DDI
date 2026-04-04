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
