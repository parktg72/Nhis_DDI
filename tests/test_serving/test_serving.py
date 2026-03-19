"""
serving/ 단위/통합 테스트
FastAPI TestClient로 실제 HTTP 요청 테스트
"""
import pytest
from datetime import date
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from serving.schemas import (
    DDIAlert, DrugItem, PredictRequest, PredictResponse,
    RiskLevel, Severity, INTERVENTION_MAP,
)
from serving.predictor import HybridPredictor, RequestFeatureBuilder, _check_triple_whammy, _count_qt_drugs


# ─────────────────────────────────────────────────────────────────────────────
# 공통 픽스처
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def warfarin_nsaid_drugs():
    """Contraindicated: warfarin + ibuprofen."""
    return [
        DrugItem(edi_code="A001001", atc_code="B01AA03", drug_name="warfarin",
                 total_days=30, start_date=date(2024, 1, 1)),
        DrugItem(edi_code="A001003", atc_code="M01AE01", drug_name="ibuprofen",
                 total_days=7, start_date=date(2024, 1, 5)),
    ]


@pytest.fixture
def triple_whammy_drugs():
    """Triple Whammy: ACEi + K보존이뇨제 + NSAIDs."""
    return [
        DrugItem(edi_code="A001007", atc_code="C09AA02", drug_name="enalapril",
                 total_days=30, start_date=date(2024, 1, 1)),
        DrugItem(edi_code="A001009", atc_code="C03DA01", drug_name="spironolactone",
                 total_days=30, start_date=date(2024, 1, 1)),
        DrugItem(edi_code="A001003", atc_code="M01AE01", drug_name="ibuprofen",
                 total_days=7, start_date=date(2024, 1, 1)),
    ]


@pytest.fixture
def normal_drugs():
    """Normal: 단순 약물 2종."""
    return [
        DrugItem(edi_code="B001002", atc_code="A10BA02", drug_name="metformin",
                 total_days=30, start_date=date(2024, 1, 1)),
        DrugItem(edi_code="B001003", atc_code="C07AB02", drug_name="metoprolol",
                 total_days=30, start_date=date(2024, 1, 1)),
    ]


@pytest.fixture
def mock_predictor():
    """Safety Net/ML 없이 동작하는 Mock HybridPredictor."""
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml = MagicMock()
    pred._ml.loaded = False
    pred._ddi_matrix = None
    pred._cyp = None
    pred._std = None
    pred._builder = RequestFeatureBuilder(ddi_matrix=None, cyp_extractor=None, code_standardizer=None)
    return pred


@pytest.fixture
def app_client(mock_predictor):
    """FastAPI TestClient with mocked predictor."""
    from serving.main import app
    import serving.predictor as pred_module
    pred_module._predictor = mock_predictor

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ─────────────────────────────────────────────────────────────────────────────
# 스키마 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemas:
    def test_risk_level_order(self):
        assert RiskLevel.RED.order > RiskLevel.YELLOW.order
        assert RiskLevel.YELLOW.order > RiskLevel.GREEN.order
        assert RiskLevel.GREEN.order > RiskLevel.NORMAL.order

    def test_risk_level_max(self):
        assert RiskLevel.max(RiskLevel.RED, RiskLevel.NORMAL) == RiskLevel.RED
        assert RiskLevel.max(RiskLevel.GREEN, RiskLevel.YELLOW) == RiskLevel.YELLOW
        assert RiskLevel.max(RiskLevel.NORMAL, RiskLevel.NORMAL) == RiskLevel.NORMAL

    def test_drug_item_validation_empty_edi(self):
        with pytest.raises(Exception):
            DrugItem(edi_code="", total_days=7)

    def test_drug_item_total_days_ge1(self):
        with pytest.raises(Exception):
            DrugItem(edi_code="A001", total_days=0)

    def test_drug_item_total_days_le365(self):
        with pytest.raises(Exception):
            DrugItem(edi_code="A001", total_days=366)

    def test_predict_request_empty_drugs(self):
        with pytest.raises(Exception):
            PredictRequest(patient_id="P001", drugs=[])

    def test_predict_request_default_date_set(self):
        req = PredictRequest(
            patient_id="P001",
            drugs=[DrugItem(edi_code="A001", total_days=7)],
        )
        assert req.reference_date == date.today()
        assert req.drugs[0].start_date == date.today()

    def test_severity_enum_values(self):
        assert Severity.CONTRAINDICATED == "Contraindicated"
        assert Severity.MAJOR == "Major"

    def test_intervention_map_complete(self):
        for level in RiskLevel:
            assert level in INTERVENTION_MAP


# ─────────────────────────────────────────────────────────────────────────────
# 예측 로직 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionLogic:
    def test_triple_whammy_detected(self, triple_whammy_drugs):
        atcs = [d.atc_code for d in triple_whammy_drugs]
        assert _check_triple_whammy(atcs) is True

    def test_triple_whammy_not_detected_missing_component(self):
        atcs = ["C09AA02", "C03DA01"]  # ACEi + K보존이뇨제 (NSAIDs 없음)
        assert _check_triple_whammy(atcs) is False

    def test_qt_drug_count(self):
        atcs = ["N05AD01", "J01MA02", "A10BA02"]  # 2개 QT약물
        assert _count_qt_drugs(atcs) == 2

    def test_qt_drug_count_none(self):
        atcs = ["A10BA02", "C07AB02"]
        assert _count_qt_drugs(atcs) == 0

    def test_feature_builder_basic(self, normal_drugs):
        builder = RequestFeatureBuilder()
        req = PredictRequest(patient_id="P001", drugs=normal_drugs)
        vec, feat = builder.build(req)
        assert feat["drug_count"] == 2.0
        assert vec.ndim == 1
        assert len(vec) > 0

    def test_feature_builder_ddi_counts(self, warfarin_nsaid_drugs):
        """ATC 기반 DDI 카운트 계산."""
        import pandas as pd
        # 간이 DDI 매트릭스
        ddi_df = pd.DataFrame([{
            "drug_a_atc": "B01AA03", "drug_b_atc": "M01AE01",
            "severity": "Contraindicated",
            "drug_a_name": "warfarin", "drug_b_name": "ibuprofen",
        }])
        builder = RequestFeatureBuilder(ddi_matrix=ddi_df)
        req = PredictRequest(patient_id="P001", drugs=warfarin_nsaid_drugs)
        vec, feat = builder.build(req)
        assert feat["ddi_contraindicated"] == 1.0

    def test_ml_classify_threshold(self):
        from serving.predictor import MLModel
        ml = MLModel()
        ml._threshold = 0.5
        assert ml.classify(0.9) == RiskLevel.RED
        assert ml.classify(0.1) == RiskLevel.NORMAL

    def test_hybrid_rule_only_mode(self, mock_predictor, normal_drugs):
        """ML 없이 Rule만으로 예측."""
        req = PredictRequest(patient_id="P001", drugs=normal_drugs)
        with patch("serving.predictor._run_safety_net") as mock_sn, \
             patch("serving.predictor._run_duplicate_detector") as mock_dup:
            mock_sn.return_value = (RiskLevel.NORMAL, [], [])
            mock_dup.return_value = (0, [])
            result = mock_predictor.predict(req)
        assert result.patient_id == "P001"
        assert result.rule_level == RiskLevel.NORMAL
        assert result.ml_level is None

    def test_hybrid_ml_upgrades_level(self, mock_predictor, normal_drugs):
        """ML이 Rule보다 높은 등급 → 최종등급 = ML."""
        mock_predictor._ml.loaded = True
        mock_predictor._ml.predict_proba = MagicMock(return_value=0.85)
        mock_predictor._ml.classify = MagicMock(return_value=RiskLevel.RED)
        mock_predictor._ml._threshold = 0.5

        req = PredictRequest(patient_id="P001", drugs=normal_drugs)
        with patch("serving.predictor._run_safety_net") as mock_sn, \
             patch("serving.predictor._run_duplicate_detector") as mock_dup:
            mock_sn.return_value = (RiskLevel.NORMAL, [], [])
            mock_dup.return_value = (0, [])
            result = mock_predictor.predict(req)

        assert result.risk_level == RiskLevel.RED
        assert result.rule_level == RiskLevel.NORMAL
        assert result.ml_level == RiskLevel.RED

    def test_hybrid_rule_overrides_ml(self, mock_predictor, warfarin_nsaid_drugs):
        """Rule이 더 높을 때 Rule이 최종등급."""
        mock_predictor._ml.loaded = True
        mock_predictor._ml.predict_proba = MagicMock(return_value=0.1)
        mock_predictor._ml.classify = MagicMock(return_value=RiskLevel.NORMAL)

        req = PredictRequest(patient_id="P001", drugs=warfarin_nsaid_drugs)
        with patch("serving.predictor._run_safety_net") as mock_sn, \
             patch("serving.predictor._run_duplicate_detector") as mock_dup:
            mock_sn.return_value = (RiskLevel.RED, ["Contraindicated DDI"], [])
            mock_dup.return_value = (0, [])
            result = mock_predictor.predict(req)

        assert result.risk_level == RiskLevel.RED
        assert result.rule_level == RiskLevel.RED


# ─────────────────────────────────────────────────────────────────────────────
# API 엔드포인트 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIEndpoints:
    def test_root(self, app_client):
        resp = app_client.get("/")
        assert resp.status_code == 200
        assert "DDI" in resp.json()["service"]

    def test_health_ok(self, app_client):
        resp = app_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "degraded")
        assert "version" in body

    def test_model_info(self, app_client):
        resp = app_client.get("/model/info")
        assert resp.status_code == 200
        body = resp.json()
        assert "model_type" in body

    def test_predict_single(self, app_client, normal_drugs):
        with patch("serving.predictor._run_safety_net") as mock_sn, \
             patch("serving.predictor._run_duplicate_detector") as mock_dup:
            mock_sn.return_value = (RiskLevel.NORMAL, [], [])
            mock_dup.return_value = (0, [])

            payload = {
                "patient_id": "P000001",
                "drugs": [
                    {"edi_code": "B001002", "atc_code": "A10BA02",
                     "drug_name": "metformin", "total_days": 30},
                    {"edi_code": "B001003", "atc_code": "C07AB02",
                     "drug_name": "metoprolol", "total_days": 30},
                ],
                "patient_age": 65,
                "patient_sex": "M",
            }
            resp = app_client.post("/predict", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["patient_id"] == "P000001"
        assert body["risk_level"] in ["Red", "Yellow", "Green", "Normal"]
        assert "intervention" in body
        assert "ddi_alerts" in body

    def test_predict_response_schema(self, app_client):
        with patch("serving.predictor._run_safety_net") as mock_sn, \
             patch("serving.predictor._run_duplicate_detector") as mock_dup:
            mock_sn.return_value = (RiskLevel.GREEN, ["5종↑"], [])
            mock_dup.return_value = (0, [])

            payload = {
                "patient_id": "P000002",
                "drugs": [{"edi_code": f"D{i:03d}", "total_days": 30} for i in range(5)],
            }
            resp = app_client.post("/predict", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        required_fields = {
            "patient_id", "risk_level", "rule_level",
            "drug_count", "ddi_alerts", "risk_reasons",
            "intervention", "reference_date",
        }
        assert required_fields.issubset(body.keys())

    def test_predict_invalid_empty_drugs(self, app_client):
        payload = {"patient_id": "P001", "drugs": []}
        resp = app_client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_invalid_sex(self, app_client):
        payload = {
            "patient_id": "P001",
            "drugs": [{"edi_code": "D001", "total_days": 7}],
            "patient_sex": "X",  # 유효하지 않은 성별
        }
        resp = app_client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_batch(self, app_client):
        with patch("serving.predictor._run_safety_net") as mock_sn, \
             patch("serving.predictor._run_duplicate_detector") as mock_dup:
            mock_sn.return_value = (RiskLevel.NORMAL, [], [])
            mock_dup.return_value = (0, [])

            payload = {
                "requests": [
                    {
                        "patient_id": f"P{i:04d}",
                        "drugs": [{"edi_code": "B001002", "total_days": 30}],
                    }
                    for i in range(5)
                ]
            }
            resp = app_client.post("/predict/batch", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 5
        total_dist = body["red_count"] + body["yellow_count"] + body["green_count"] + body["normal_count"]
        assert total_dist == 5
        assert "elapsed_ms" in body

    def test_request_id_header(self, app_client):
        resp = app_client.get("/health")
        assert "x-request-id" in resp.headers

    def test_elapsed_ms_header(self, app_client):
        resp = app_client.get("/health")
        assert "x-elapsed-ms" in resp.headers

    def test_docs_available(self, app_client):
        resp = app_client.get("/docs")
        assert resp.status_code == 200
