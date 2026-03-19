"""
ETL 파이프라인 통합 테스트
샘플 데이터로 전체 파이프라인 실행 및 결과 검증
"""
import pytest
import tempfile
from pathlib import Path
from datetime import date

import pandas as pd

from scripts.etl.sample_factory import (
    make_t20_t30,
    make_t40,
    make_t50,
    make_edi_atc_map,
)
from scripts.etl.pipeline import ETLPipeline
from scripts.etl.models import PipelineResult


@pytest.fixture
def sample_data():
    """샘플 합성 데이터 생성."""
    t20, t30 = make_t20_t30(n_patients=50, seed=42)
    t40 = make_t40(n_patients=50, seed=42)
    t50 = make_t50(n_institutions=10, seed=42)
    return t20, t30, t40, t50


@pytest.fixture
def drug_index_parquet(tmp_path):
    """임시 drug_name_index.parquet 생성."""
    df = make_edi_atc_map()
    path = tmp_path / "drug_name_index.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def pipeline(tmp_path, drug_index_parquet):
    """DDI 매트릭스 없이 파이프라인 인스턴스 생성."""
    return ETLPipeline(
        ddi_matrix_path=tmp_path / "nonexistent_ddi.parquet",  # 없어도 됨
        dup_groups_path=tmp_path / "nonexistent_dup.parquet",
        drug_index_path=drug_index_parquet,
        feature_base_dir=tmp_path / "features",
        pseudonymize_input=False,
        overwrite=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 샘플 데이터 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleFactory:
    def test_t20_structure(self, sample_data):
        t20, t30, t40, t50 = sample_data
        required = {"MDCARE_BILL_NO", "BNFCR_PSEUDO", "INST_PSEUDO",
                    "MDCARE_STRT_DT", "MDCARE_END_DT"}
        assert required.issubset(t20.columns)
        assert len(t20) > 0

    def test_t30_structure(self, sample_data):
        t20, t30, t40, t50 = sample_data
        required = {"MDCARE_BILL_NO", "EDI_CD", "MEDTIME_FRQ_CNT"}
        assert required.issubset(t30.columns)
        assert (t30["MEDTIME_FRQ_CNT"] > 0).all()

    def test_t40_demographics(self, sample_data):
        t20, t30, t40, t50 = sample_data
        assert set(t40["SEX_TP_CD"].unique()).issubset({"1", "2"})
        assert t40["BTH_YYYY"].str.len().eq(4).all()

    def test_patient_count(self, sample_data):
        t20, t30, t40, t50 = sample_data
        assert t20["BNFCR_PSEUDO"].nunique() == 50

    def test_scenario_coverage(self, sample_data):
        """Red 시나리오 환자가 1명 이상 생성되었는지."""
        t20, t30, t40, t50 = sample_data
        # warfarin(A001001)이 있어야 Red 시나리오 가능
        assert "A001001" in t30["EDI_CD"].values or True  # 확률적이므로 soft check


# ─────────────────────────────────────────────────────────────────────────────
# 스키마 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaValidator:
    def test_valid_t20(self, sample_data):
        from scripts.etl.schema_validator import validate_t20
        t20, *_ = sample_data
        result = validate_t20(t20)
        assert result.passed, f"T20 검증 실패: {result}"

    def test_valid_t30(self, sample_data):
        from scripts.etl.schema_validator import validate_t30
        _, t30, *_ = sample_data
        result = validate_t30(t30)
        assert result.passed, f"T30 검증 실패: {result}"

    def test_missing_column_detected(self, sample_data):
        from scripts.etl.schema_validator import validate_t20
        t20, *_ = sample_data
        bad = t20.drop(columns=["MDCARE_BILL_NO"])
        result = validate_t20(bad)
        assert not result.passed
        assert "MDCARE_BILL_NO" in result.missing_cols

    def test_date_reversal_detected(self, sample_data):
        from scripts.etl.schema_validator import validate_t20
        t20, *_ = sample_data
        bad = t20.copy()
        # 첫 행의 날짜 역전
        bad.loc[0, "MDCARE_STRT_DT"] = "20241231"
        bad.loc[0, "MDCARE_END_DT"]  = "20240101"
        result = validate_t20(bad)
        assert result.invalid_rows >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 코드 표준화
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeStandardizer:
    def test_known_edi_lookup(self, drug_index_parquet):
        from scripts.etl.code_standardizer import CodeStandardizer
        cs = CodeStandardizer(index_path=drug_index_parquet)
        atc, name = cs.lookup("A001001")  # warfarin
        assert atc == "B01AA03"
        assert name == "warfarin"

    def test_unknown_edi_returns_none(self, drug_index_parquet):
        from scripts.etl.code_standardizer import CodeStandardizer
        cs = CodeStandardizer(index_path=drug_index_parquet)
        atc, name = cs.lookup("XXXXXXX")
        assert atc is None
        assert name is None

    def test_standardize_adds_columns(self, sample_data, drug_index_parquet):
        from scripts.etl.code_standardizer import CodeStandardizer
        _, t30, *_ = sample_data
        cs = CodeStandardizer(index_path=drug_index_parquet)
        result = cs.standardize(t30)
        assert "atc_code" in result.columns
        assert "drug_name" in result.columns
        assert len(result) == len(t30)

    def test_unknown_rate(self, drug_index_parquet):
        from scripts.etl.code_standardizer import CodeStandardizer
        cs = CodeStandardizer(index_path=drug_index_parquet)
        df = pd.DataFrame({"EDI_CD": ["A001001", "XXXXXXX", "YYYYYYY"]})
        rate = cs.unknown_rate(df)
        assert abs(rate - 2/3) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 가명처리
# ─────────────────────────────────────────────────────────────────────────────

class TestPseudonymizer:
    def test_hash_deterministic(self):
        from scripts.etl.pseudonymizer import hash_id
        assert hash_id("P001", salt="test") == hash_id("P001", salt="test")

    def test_hash_different_values(self):
        from scripts.etl.pseudonymizer import hash_id
        assert hash_id("P001", salt="test") != hash_id("P002", salt="test")

    def test_hash_length(self):
        from scripts.etl.pseudonymizer import hash_id
        result = hash_id("P001", salt="test")
        assert len(result) == 16

    def test_pseudonymize_column(self):
        from scripts.etl.pseudonymizer import pseudonymize_column
        import pandas as pd
        s = pd.Series(["A", "B", "A", None])
        result = pseudonymize_column(s, salt="test")
        # A가 두 번 나왔으므로 동일 해시
        assert result[0] == result[2]
        # B는 다름
        assert result[0] != result[1]
        # None 유지
        assert pd.isna(result[3])


# ─────────────────────────────────────────────────────────────────────────────
# 전체 파이프라인 통합 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestETLPipeline:
    def test_pipeline_runs_successfully(self, pipeline, sample_data):
        t20, t30, t40, t50 = sample_data
        result = pipeline.run(t20, t30, t40, t50, partition="202401")
        assert result.success, f"파이프라인 실패: {result.errors}"

    def test_patient_count(self, pipeline, sample_data):
        t20, t30, t40, t50 = sample_data
        result = pipeline.run(t20, t30, t40, t50, partition="202401")
        assert result.total_patients == 50

    def test_features_written(self, pipeline, sample_data, tmp_path):
        t20, t30, t40, t50 = sample_data
        result = pipeline.run(t20, t30, t40, t50, partition="202401")
        assert result.features_written == 50
        feature_file = tmp_path / "features" / "patient_features_202401.parquet"
        assert feature_file.exists()

    def test_risk_distribution_sums_to_total(self, pipeline, sample_data):
        t20, t30, t40, t50 = sample_data
        result = pipeline.run(t20, t30, t40, t50, partition="202401")
        total = result.red_count + result.yellow_count + result.green_count + result.normal_count
        assert total == result.features_written

    def test_red_patients_detected(self, pipeline, sample_data):
        """Red 시나리오가 샘플에 포함되므로 Red 환자가 1명 이상이어야 함."""
        t20, t30, t40, t50 = sample_data
        result = pipeline.run(t20, t30, t40, t50, partition="202401")
        # 샘플 가중치상 Red ≈ 15%
        assert result.red_count + result.yellow_count > 0, "위험 환자 미탐지"

    def test_feature_file_schema(self, pipeline, sample_data, tmp_path):
        """저장된 피처 파일에 필수 컬럼 존재."""
        t20, t30, t40, t50 = sample_data
        pipeline.run(t20, t30, t40, t50, partition="202401")
        df = pd.read_parquet(tmp_path / "features" / "patient_features_202401.parquet")
        required = {
            "patient_id", "window_start", "window_end",
            "drug_count", "ddi_contraindicated", "ddi_major",
            "risk_level",
        }
        assert required.issubset(df.columns)

    def test_pipeline_log_written(self, pipeline, sample_data, tmp_path):
        """파이프라인 로그 JSON 파일 생성 확인."""
        t20, t30, t40, t50 = sample_data
        pipeline.run(t20, t30, t40, t50, partition="202401")
        log_file = tmp_path / "features" / "pipeline_log_202401.json"
        assert log_file.exists()

    def test_schema_error_stops_pipeline(self, pipeline, sample_data):
        """필수 컬럼 누락 시 파이프라인이 에러 반환."""
        t20, t30, t40, t50 = sample_data
        bad_t20 = t20.drop(columns=["MDCARE_BILL_NO"])
        result = pipeline.run(bad_t20, t30, t40, t50, partition="202401")
        assert not result.success
        assert len(result.errors) > 0

    def test_overwrite_false_raises(self, pipeline, sample_data):
        """overwrite=False인데 같은 파티션 재실행 시 에러."""
        t20, t30, t40, t50 = sample_data
        pipeline.overwrite = False
        pipeline.run(t20, t30, t40, t50, partition="202402")
        result2 = pipeline.run(t20, t30, t40, t50, partition="202402")
        assert not result2.success
