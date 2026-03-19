"""
scripts/features 단위/통합 테스트
"""
import math
import pytest
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from scripts.features.cyp_features import CYPFeatureExtractor, CYP_FEATURE_COLS
from scripts.features.temporal_features import extract_temporal, TEMPORAL_FEATURE_COLS
from scripts.features.normalizer import FeatureNormalizer
from scripts.features.selector import FeatureSelector, PROTECTED_FEATURES
from scripts.features.feature_engineer import FeatureEngineer, BINARY_LABEL_COL
from scripts.etl.models import PrescriptionRecord


# ─────────────────────────────────────────────────────────────────────────────
# 픽스처
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mini_cyp_df():
    """최소 cyp_matrix DataFrame."""
    return pd.DataFrame([
        {
            "drugbank_id": "DB00001", "drug_name": "warfarin",
            "cyp3a4_substrate_count": 0, "cyp3a4_inhibitor_strong_count": 0,
            "cyp3a4_inhibitor_moderate_count": 0, "cyp3a4_inhibitor_count": 0,
            "cyp3a4_inducer_count": 0, "cyp3a4_interaction_risk": 0,
            "cyp2d6_substrate_count": 0, "cyp2d6_inhibitor_strong_count": 0,
            "cyp2d6_inhibitor_moderate_count": 0, "cyp2d6_inhibitor_count": 0,
            "cyp2d6_inducer_count": 0, "cyp2d6_interaction_risk": 0,
            "cyp2c9_substrate_count": 1, "cyp2c9_inhibitor_strong_count": 0,
            "cyp2c9_inhibitor_moderate_count": 0, "cyp2c9_inhibitor_count": 0,
            "cyp2c9_inducer_count": 0, "cyp2c9_interaction_risk": 1,
            "cyp2c19_substrate_count": 0, "cyp2c19_inhibitor_strong_count": 0,
            "cyp2c19_inhibitor_moderate_count": 0, "cyp2c19_inhibitor_count": 0,
            "cyp2c19_inducer_count": 0, "cyp2c19_interaction_risk": 0,
            "cyp1a2_substrate_count": 0, "cyp1a2_inhibitor_strong_count": 0,
            "cyp1a2_inhibitor_moderate_count": 0, "cyp1a2_inhibitor_count": 0,
            "cyp1a2_inducer_count": 0, "cyp1a2_interaction_risk": 0,
        },
        {
            "drugbank_id": "DB00002", "drug_name": "fluconazole",
            "cyp3a4_substrate_count": 0, "cyp3a4_inhibitor_strong_count": 1,
            "cyp3a4_inhibitor_moderate_count": 0, "cyp3a4_inhibitor_count": 1,
            "cyp3a4_inducer_count": 0, "cyp3a4_interaction_risk": 3,
            "cyp2d6_substrate_count": 0, "cyp2d6_inhibitor_strong_count": 0,
            "cyp2d6_inhibitor_moderate_count": 0, "cyp2d6_inhibitor_count": 0,
            "cyp2d6_inducer_count": 0, "cyp2d6_interaction_risk": 0,
            "cyp2c9_substrate_count": 0, "cyp2c9_inhibitor_strong_count": 1,
            "cyp2c9_inhibitor_moderate_count": 0, "cyp2c9_inhibitor_count": 1,
            "cyp2c9_inducer_count": 0, "cyp2c9_interaction_risk": 3,
            "cyp2c19_substrate_count": 0, "cyp2c19_inhibitor_strong_count": 0,
            "cyp2c19_inhibitor_moderate_count": 0, "cyp2c19_inhibitor_count": 0,
            "cyp2c19_inducer_count": 0, "cyp2c19_interaction_risk": 0,
            "cyp1a2_substrate_count": 0, "cyp1a2_inhibitor_strong_count": 0,
            "cyp1a2_inhibitor_moderate_count": 0, "cyp1a2_inhibitor_count": 0,
            "cyp1a2_inducer_count": 0, "cyp1a2_interaction_risk": 0,
        },
    ])


@pytest.fixture
def mini_drug_index():
    """ATC → DrugBank 매핑."""
    return pd.DataFrame([
        {"drug_name": "warfarin",    "drug_name_lower": "warfarin",
         "drugbank_id": "DB00001",   "atc_codes": "B01AA03", "groups": "approved"},
        {"drug_name": "fluconazole", "drug_name_lower": "fluconazole",
         "drugbank_id": "DB00002",   "atc_codes": "J02AC01", "groups": "approved"},
        {"drug_name": "ibuprofen",   "drug_name_lower": "ibuprofen",
         "drugbank_id": "DB00003",   "atc_codes": "M01AE01", "groups": "approved"},
    ])


@pytest.fixture
def cyp_extractor(mini_cyp_df, mini_drug_index, tmp_path):
    cyp_path = tmp_path / "cyp_matrix.parquet"
    idx_path = tmp_path / "drug_index.parquet"
    mini_cyp_df.to_parquet(cyp_path, index=False)
    mini_drug_index.to_parquet(idx_path, index=False)
    return CYPFeatureExtractor(cyp_matrix_path=cyp_path, drug_index_path=idx_path)


@pytest.fixture
def sample_prescriptions():
    base = date(2024, 1, 1)
    return [
        PrescriptionRecord(
            patient_id="P001", institution_id="INST01", bill_no="B001",
            edi_code="A001", atc_code="B01AA03", drug_name="warfarin",
            start_date=base, end_date=base + timedelta(days=29),
            total_days=30, dose_once=1.0, dose_freq=1,
        ),
        PrescriptionRecord(
            patient_id="P001", institution_id="INST02", bill_no="B002",
            edi_code="A002", atc_code="M01AE01", drug_name="ibuprofen",
            start_date=base + timedelta(days=5), end_date=base + timedelta(days=19),
            total_days=15, dose_once=1.0, dose_freq=3,
        ),
        PrescriptionRecord(
            patient_id="P001", institution_id="INST01", bill_no="B003",
            edi_code="A003", atc_code="J02AC01", drug_name="fluconazole",
            start_date=base + timedelta(days=50), end_date=base + timedelta(days=79),
            total_days=30, dose_once=1.0, dose_freq=1,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CYP 피처 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestCYPFeatures:
    def test_feature_cols_complete(self, cyp_extractor):
        feat = cyp_extractor.extract(["B01AA03"])
        for col in CYP_FEATURE_COLS:
            assert col in feat, f"피처 누락: {col}"

    def test_empty_atc_returns_zeros(self, cyp_extractor):
        feat = cyp_extractor.extract([])
        assert feat["cyp_risk_score"] == 0.0
        assert feat["cyp_high_risk_pairs"] == 0.0

    def test_warfarin_is_cyp2c9_substrate(self, cyp_extractor):
        """warfarin(B01AA03)은 CYP2C9 기질."""
        feat = cyp_extractor.extract(["B01AA03"])
        assert feat["cyp2c9_substrates"] == 1.0

    def test_fluconazole_is_cyp2c9_strong_inhibitor(self, cyp_extractor):
        """fluconazole(J02AC01)은 CYP2C9 강한 억제제."""
        feat = cyp_extractor.extract(["J02AC01"])
        assert feat["cyp2c9_strong_inhibitors"] == 1.0

    def test_warfarin_fluconazole_pair_detected(self, cyp_extractor):
        """warfarin(기질) + fluconazole(억제제) → CYP2C9 위험 쌍 탐지."""
        feat = cyp_extractor.extract(["B01AA03", "J02AC01"])
        assert feat["cyp2c9_inhibitor_substrate_pairs"] >= 1.0
        assert feat["cyp_risk_score"] > 0.0
        assert feat["cyp_high_risk_pairs"] >= 1.0

    def test_no_interaction_unknown_drug(self, cyp_extractor):
        """매핑 불가 ATC → 피처 0."""
        feat = cyp_extractor.extract(["XXXXXXX"])
        assert feat["cyp_risk_score"] == 0.0

    def test_risk_score_higher_with_more_pairs(self, cyp_extractor):
        """억제제+기질 쌍이 많을수록 위험 점수 증가."""
        single = cyp_extractor.extract(["B01AA03"])["cyp_risk_score"]
        pair = cyp_extractor.extract(["B01AA03", "J02AC01"])["cyp_risk_score"]
        assert pair > single


# ─────────────────────────────────────────────────────────────────────────────
# 시계열 피처 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalFeatures:
    def test_feature_cols_complete(self, sample_prescriptions):
        feat = extract_temporal(sample_prescriptions)
        for col in TEMPORAL_FEATURE_COLS:
            assert col in feat, f"피처 누락: {col}"

    def test_empty_prescriptions(self):
        feat = extract_temporal([])
        assert feat["drug_count_early"] == 0.0
        assert feat["drug_count_late"] == 0.0

    def test_drug_count_split(self, sample_prescriptions):
        """전반/후반 분할 정확성."""
        feat = extract_temporal(sample_prescriptions)
        # warfarin(1/1), ibuprofen(1/6) → 전반 / fluconazole(2/20) → 후반
        assert feat["drug_count_early"] == 2.0
        assert feat["drug_count_late"] == 1.0
        assert feat["drug_trend"] == -1.0

    def test_new_drug_in_late(self, sample_prescriptions):
        """후반에 새로 등장한 약물 카운트."""
        feat = extract_temporal(sample_prescriptions)
        assert feat["new_drug_in_late"] == 1.0  # fluconazole 신규

    def test_multi_institution_flag(self, sample_prescriptions):
        """2개 기관 → multi_institution_flag=0."""
        feat = extract_temporal(sample_prescriptions)
        assert feat["multi_institution_flag"] == 0.0

    def test_institution_entropy_positive(self, sample_prescriptions):
        """2개 기관 → 엔트로피 > 0."""
        feat = extract_temporal(sample_prescriptions)
        assert feat["institution_entropy"] > 0.0

    def test_long_term_drug_count(self, sample_prescriptions):
        """30일↑ 처방: warfarin(30일), fluconazole(30일)."""
        feat = extract_temporal(sample_prescriptions)
        assert feat["long_term_drug_count"] == 2.0

    def test_chronic_drug_ratio(self, sample_prescriptions):
        """3개 중 2개 장기처방 → ratio ≈ 0.667."""
        feat = extract_temporal(sample_prescriptions)
        assert abs(feat["chronic_drug_ratio"] - 2/3) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 정규화 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureNormalizer:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "patient_id":   ["P1", "P2", "P3", "P4", "P5"],
            "drug_count":   [1.0, 3.0, 5.0, 7.0, 9.0],
            "ddi_major":    [0.0, 0.0, 1.0, 2.0, 3.0],
            "triple_whammy":[0.0, 0.0, 0.0, 1.0, 1.0],  # 이진
        })

    def test_fit_transform_shape(self, sample_df):
        norm = FeatureNormalizer()
        out = norm.fit_transform(sample_df)
        assert out.shape == sample_df.shape

    def test_patient_id_unchanged(self, sample_df):
        norm = FeatureNormalizer()
        out = norm.fit_transform(sample_df)
        assert list(out["patient_id"]) == list(sample_df["patient_id"])

    def test_median_centered(self, sample_df):
        """정규화 후 중앙값 ≈ 0."""
        norm = FeatureNormalizer()
        out = norm.fit_transform(sample_df)
        assert abs(out["drug_count"].median()) < 0.1

    def test_binary_col_unchanged(self, sample_df):
        """이진 피처는 스케일링 없이 그대로."""
        norm = FeatureNormalizer()
        out = norm.fit_transform(sample_df)
        assert set(out["triple_whammy"].unique()) == {0.0, 1.0}

    def test_missing_values_filled(self):
        df = pd.DataFrame({
            "drug_count": [1.0, None, 3.0, None, 5.0],
        })
        norm = FeatureNormalizer()
        out = norm.fit_transform(df)
        assert out["drug_count"].isna().sum() == 0

    def test_save_load(self, sample_df, tmp_path):
        norm = FeatureNormalizer()
        norm.fit(sample_df)
        path = tmp_path / "scaler.pkl"
        norm.save(path)
        norm2 = FeatureNormalizer.load(path)
        out1 = norm.transform(sample_df)
        out2 = norm2.transform(sample_df)
        pd.testing.assert_frame_equal(out1, out2)

    def test_transform_without_fit_raises(self, sample_df):
        norm = FeatureNormalizer()
        with pytest.raises(RuntimeError):
            norm.transform(sample_df)


# ─────────────────────────────────────────────────────────────────────────────
# 피처 선택 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureSelector:
    @pytest.fixture
    def sample_df(self):
        rng = np.random.default_rng(42)
        n = 50
        base = pd.DataFrame({
            "patient_id": [f"P{i}" for i in range(n)],
            "drug_count": rng.integers(1, 15, n).astype(float),
            "ddi_contraindicated": rng.integers(0, 3, n).astype(float),
            "ddi_major": rng.integers(0, 5, n).astype(float),
            "const_col": [0.0] * n,                # 상수 → 제거 대상
            "triple_whammy": rng.integers(0, 2, n).astype(float),
            "risk_level": rng.choice(["Red","Yellow","Green","Normal"], n),
        })
        # 완전 중복 컬럼 (ddi_major의 복사본)
        base["ddi_major_dup"] = base["ddi_major"].copy()
        return base

    def test_constant_column_removed(self, sample_df):
        sel = FeatureSelector(variance_threshold=0.0)
        sel.fit(sample_df)
        assert "const_col" not in sel.selected_features

    def test_correlated_column_removed(self, sample_df):
        sel = FeatureSelector(correlation_threshold=0.95)
        sel.fit(sample_df)
        # ddi_major와 ddi_major_dup 중 하나 제거
        has_both = ("ddi_major" in sel.selected_features and
                    "ddi_major_dup" in sel.selected_features)
        assert not has_both

    def test_protected_features_preserved(self, sample_df):
        sel = FeatureSelector()
        out = sel.fit_transform(sample_df)
        for feat in ["ddi_contraindicated", "ddi_major", "triple_whammy"]:
            if feat in sample_df.columns:
                assert feat in out.columns, f"보호 피처 제거됨: {feat}"

    def test_meta_cols_preserved(self, sample_df):
        sel = FeatureSelector()
        out = sel.fit_transform(sample_df)
        assert "patient_id" in out.columns
        assert "risk_level" in out.columns

    def test_transform_without_fit_raises(self, sample_df):
        sel = FeatureSelector()
        with pytest.raises(RuntimeError):
            sel.transform(sample_df)

    def test_save_load(self, sample_df, tmp_path):
        sel = FeatureSelector()
        sel.fit(sample_df)
        path = tmp_path / "selector.pkl"
        sel.save(path)
        sel2 = FeatureSelector.load(path)
        assert sel.selected_features == sel2.selected_features


# ─────────────────────────────────────────────────────────────────────────────
# FeatureEngineer 통합 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineer:
    @pytest.fixture
    def etl_feature_parquet(self, tmp_path):
        """ETL 피처 파일 모사."""
        rng = np.random.default_rng(42)
        n = 30
        df = pd.DataFrame({
            "patient_id":          [f"P{i:04d}" for i in range(n)],
            "window_start":        [date(2024, 1, 1)] * n,
            "window_end":          [date(2024, 3, 31)] * n,
            "drug_count":          rng.integers(1, 12, n).astype(float),
            "drug_count_7d":       rng.integers(1, 8, n).astype(float),
            "institution_count":   rng.integers(1, 5, n).astype(float),
            "ddi_contraindicated": rng.integers(0, 2, n).astype(float),
            "ddi_major":           rng.integers(0, 4, n).astype(float),
            "ddi_moderate":        rng.integers(0, 6, n).astype(float),
            "ddi_minor":           rng.integers(0, 8, n).astype(float),
            "triple_whammy":       rng.integers(0, 2, n).astype(float),
            "qt_risk_count":       rng.integers(0, 4, n).astype(float),
            "dup_same_ingredient": rng.integers(0, 2, n).astype(float),
            "dup_atc5":            rng.integers(0, 3, n).astype(float),
            "dup_atc4":            rng.integers(0, 5, n).astype(float),
            "dup_atc3":            rng.integers(0, 7, n).astype(float),
            "age":                 rng.integers(30, 85, n).astype(float),
            "sex":                 rng.choice(["M", "F"], n),
            "risk_level":          rng.choice(["Red","Yellow","Green","Normal"], n),
            "risk_reasons":        [""] * n,
        })
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        df.to_parquet(feat_dir / "patient_features_202401.parquet", index=False)
        return feat_dir

    def test_run_produces_output(self, etl_feature_parquet, tmp_path):
        eng = FeatureEngineer(
            cyp_extractor=None,
            feature_base=etl_feature_parquet,
            fit_mode=True,
        )
        result = eng.run("202401")
        assert len(result) == 30
        assert "patient_id" in result.columns

    def test_binary_label_created(self, etl_feature_parquet):
        eng = FeatureEngineer(
            cyp_extractor=None,
            feature_base=etl_feature_parquet,
            fit_mode=True,
        )
        result = eng.run("202401")
        assert BINARY_LABEL_COL in result.columns
        assert set(result[BINARY_LABEL_COL].unique()).issubset({0, 1})

    def test_output_file_saved(self, etl_feature_parquet):
        eng = FeatureEngineer(
            cyp_extractor=None,
            feature_base=etl_feature_parquet,
            fit_mode=True,
        )
        eng.run("202401")
        assert (etl_feature_parquet / "ml_features_202401.parquet").exists()
        assert (etl_feature_parquet / "scaler.pkl").exists()
        assert (etl_feature_parquet / "selector.pkl").exists()

    def test_protected_features_in_output(self, etl_feature_parquet):
        eng = FeatureEngineer(
            cyp_extractor=None,
            feature_base=etl_feature_parquet,
            fit_mode=True,
        )
        result = eng.run("202401")
        assert "ddi_contraindicated" in result.columns
        assert "ddi_major" in result.columns

    def test_sex_encoded(self, etl_feature_parquet):
        """sex 컬럼이 sex_male(0/1)로 인코딩."""
        eng = FeatureEngineer(
            cyp_extractor=None,
            feature_base=etl_feature_parquet,
            fit_mode=True,
        )
        result = eng.run("202401")
        assert "sex" not in result.columns
        assert "sex_male" in result.columns
