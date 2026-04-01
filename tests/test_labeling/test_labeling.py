"""
labeling/ 단위/통합 테스트

- TestADRLabeler       : ADR 프록시 레이블 생성 (5종 ADR 시나리오)
- TestStratifiedSampler: 계층화 샘플 추출, 비율 검증, 파일 내보내기
- TestGoldenValidator  : 골든 데이터셋 검증 (완결률/일치율/Kappa/분포)
- TestLabelingPipeline : 3단계 파이프라인 통합
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ─────────────────────────────────────────────────────────────────────────────
# 공통 픽스처
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def features_df():
    """10종 위험도별 PatientFeatures 샘플."""
    np.random.seed(42)
    n = 200
    levels = (["Red"] * 60 + ["Yellow"] * 60 + ["Green"] * 50 + ["Normal"] * 30)
    return pd.DataFrame({
        "patient_id":         [f"P{i:04d}" for i in range(n)],
        "risk_level":         levels,
        "drug_count":         np.random.randint(5, 15, n),
        "ddi_contraindicated": np.random.randint(0, 3, n),
        "ddi_major":          np.random.randint(0, 5, n),
        "triple_whammy":      np.random.randint(0, 2, n),
        "cyp_risk_score":     np.random.uniform(0, 5, n),
        "patient_age":        np.random.randint(40, 90, n),
        "sex":                np.random.choice(["M", "F"], n),
        "atc_codes":          [["B01AA03", "M01AE01"]] * 60
                              + [["C09AA02", "C03DA01"]] * 60
                              + [["A10BA02"]] * 50
                              + [["C07AB02"]] * 30,
    })


@pytest.fixture
def diagnosis_df():
    """ICD-10 상병코드 샘플 (Red 환자 절반에 출혈 코드 배정)."""
    rows = []
    for i in range(30):  # P0000~P0029 : 출혈 ADR
        rows.append({"patient_id": f"P{i:04d}", "icd10_code": "K92.1", "days_after_rx": 30})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TestADRLabeler
# ─────────────────────────────────────────────────────────────────────────────

class TestADRLabeler:
    def test_bleeding_adr_detected(self):
        from labeling.adr_labeler import ADRLabeler, ADRType
        labeler = ADRLabeler()
        result = labeler.label(
            patient_id="P001",
            atc_codes=["B01AA03", "M01AE01"],          # 와파린 + NSAIDs
            diagnosis_codes=[("K92.1", 30)],           # 위장관 출혈
            rule_risk_level="Red",
        )
        assert result.label == 1
        assert any(e.adr_type == ADRType.BLEEDING for e in result.adr_evidences)

    def test_acute_kidney_adr(self):
        from labeling.adr_labeler import ADRLabeler, ADRType
        labeler = ADRLabeler()
        result = labeler.label(
            patient_id="P002",
            atc_codes=["C09AA02", "C03DA01", "M01AE01"],  # Triple Whammy
            diagnosis_codes=[("N17.0", 20)],               # 급성신부전
            rule_risk_level="Red",
        )
        assert result.label == 1
        assert any(e.adr_type == ADRType.ACUTE_KIDNEY for e in result.adr_evidences)

    def test_digoxin_toxicity_adr(self):
        from labeling.adr_labeler import ADRLabeler, ADRType
        labeler = ADRLabeler()
        result = labeler.label(
            patient_id="P003",
            atc_codes=["C01AA05", "C01BD01"],  # 디곡신 + 아미오다론
            diagnosis_codes=[("I49.0", 15)],   # 부정맥
            rule_risk_level="Yellow",
        )
        assert result.label == 1
        assert any(e.adr_type == ADRType.DIGOXIN_TOXICITY for e in result.adr_evidences)

    def test_serotonin_syndrome_adr(self):
        from labeling.adr_labeler import ADRLabeler, ADRType
        labeler = ADRLabeler()
        result = labeler.label(
            patient_id="P004",
            atc_codes=["N06AB06", "N06AF03"],  # SSRI + MAOi
            diagnosis_codes=[("G25.3", 10)],   # 이상운동
            rule_risk_level="Red",
        )
        assert result.label == 1
        assert any(e.adr_type == ADRType.SEROTONIN for e in result.adr_evidences)

    def test_hypoglycemia_adr(self):
        from labeling.adr_labeler import ADRLabeler, ADRType
        labeler = ADRLabeler()
        result = labeler.label(
            patient_id="P005",
            atc_codes=["A10AB01", "A10BB01"],  # 인슐린 + 설포닐우레아
            diagnosis_codes=[("E16.0", 5)],    # 저혈당
            rule_risk_level="Yellow",
        )
        assert result.label == 1
        assert any(e.adr_type == ADRType.HYPOGLYCEMIA for e in result.adr_evidences)

    def test_no_adr_without_icd10(self):
        from labeling.adr_labeler import ADRLabeler
        labeler = ADRLabeler()
        result = labeler.label(
            patient_id="P006",
            atc_codes=["B01AA03", "M01AE01"],
            diagnosis_codes=[],
            rule_risk_level="Red",
        )
        assert result.label == 0
        assert len(result.adr_evidences) == 0

    def test_outside_lookback_ignored(self):
        """lookback 기간(90일) 초과 진단코드는 무시."""
        from labeling.adr_labeler import ADRLabeler
        labeler = ADRLabeler(lookback_days=90)
        result = labeler.label(
            patient_id="P007",
            atc_codes=["B01AA03", "M01AE01"],
            diagnosis_codes=[("K92.1", 100)],  # 100일 후 — lookback 초과
        )
        assert result.label == 0

    def test_confidence_high_with_atc(self):
        from labeling.adr_labeler import ADRLabeler, CONFIDENCE_HIGH
        labeler = ADRLabeler()
        result = labeler.label(
            patient_id="P008",
            atc_codes=["B01AA03", "M01AE01"],
            diagnosis_codes=[("K92.1", 30)],
        )
        assert result.confidence == CONFIDENCE_HIGH

    def test_confidence_medium_without_atc(self):
        from labeling.adr_labeler import ADRLabeler, CONFIDENCE_MEDIUM
        labeler = ADRLabeler()
        result = labeler.label(
            patient_id="P009",
            atc_codes=["A10BA02"],       # 관련 없는 ATC
            diagnosis_codes=[("K92.1", 30)],
        )
        assert result.confidence == CONFIDENCE_MEDIUM

    def test_final_label_red_on_adr(self):
        from labeling.adr_labeler import ADRLabeler
        labeler = ADRLabeler()
        result = labeler.label(
            patient_id="P010",
            atc_codes=["B01AA03", "M01AE01"],
            diagnosis_codes=[("K92.1", 30)],
            rule_risk_level="Yellow",  # Rule은 Yellow지만 ADR 발생
        )
        assert result.final_label == "Red"

    def test_adr_score_positive(self):
        from labeling.adr_labeler import ADRLabeler
        labeler = ADRLabeler()
        result = labeler.label(
            patient_id="P011",
            atc_codes=["B01AA03", "M01AE01"],
            diagnosis_codes=[("K92.1", 30)],
        )
        assert result.adr_score > 0

    def test_label_batch(self, features_df, diagnosis_df):
        from labeling.adr_labeler import ADRLabeler
        import ast

        labeler = ADRLabeler()
        # diagnosis_df를 features_df 형식으로 변환
        diag_map = {}
        for pid, grp in diagnosis_df.groupby("patient_id"):
            diag_map[str(pid)] = list(zip(grp["icd10_code"], grp["days_after_rx"]))

        df = features_df.copy()
        df["diagnosis_codes"] = df["patient_id"].map(lambda p: diag_map.get(p, []))
        results = labeler.label_batch(df)
        assert len(results) == len(features_df)
        # P0000~P0029에 출혈 코드 배정 → 일부 ADR=1
        adr_pos = sum(r.label for r in results)
        assert adr_pos >= 20  # 최소 20건 이상 ADR 탐지


# ─────────────────────────────────────────────────────────────────────────────
# TestStratifiedSampler
# ─────────────────────────────────────────────────────────────────────────────

class TestStratifiedSampler:
    def test_extract_respects_total(self, features_df):
        from labeling.sample_extractor import StratifiedSampler
        sampler = StratifiedSampler(n_total=100)
        sample = sampler.extract(features_df)
        assert len(sample) <= 100

    def test_risk_level_distribution_balanced(self, features_df):
        from labeling.sample_extractor import StratifiedSampler, RISK_LEVEL_RATIOS
        sampler = StratifiedSampler(n_total=100)
        sample = sampler.extract(features_df)
        dist = sample["risk_level"].value_counts(normalize=True)
        for level, ratio in RISK_LEVEL_RATIOS.items():
            if level in dist:
                assert abs(dist[level] - ratio) < 0.15, \
                    f"{level} 비율 편차: {abs(dist[level] - ratio):.3f}"

    def test_all_risk_levels_present(self, features_df):
        from labeling.sample_extractor import StratifiedSampler
        sampler = StratifiedSampler(n_total=100)
        sample = sampler.extract(features_df)
        assert set(sample["risk_level"].unique()) == {"Red", "Yellow", "Green", "Normal"}

    def test_small_pool_uses_all(self):
        """목표보다 데이터가 적으면 전체 반환."""
        from labeling.sample_extractor import StratifiedSampler
        df = pd.DataFrame({
            "patient_id": [f"P{i}" for i in range(10)],
            "risk_level": ["Red"] * 10,
        })
        sampler = StratifiedSampler(n_total=100)
        sample = sampler.extract(df)
        assert len(sample) == 10

    def test_reproducible_with_same_seed(self, features_df):
        from labeling.sample_extractor import StratifiedSampler
        s1 = StratifiedSampler(n_total=100, seed=42).extract(features_df)
        s2 = StratifiedSampler(n_total=100, seed=42).extract(features_df)
        pd.testing.assert_frame_equal(s1.reset_index(drop=True), s2.reset_index(drop=True))

    def test_different_seed_different_result(self, features_df):
        from labeling.sample_extractor import StratifiedSampler
        s1 = StratifiedSampler(n_total=100, seed=42).extract(features_df)
        s2 = StratifiedSampler(n_total=100, seed=99).extract(features_df)
        assert not s1["patient_id"].equals(s2["patient_id"])

    def test_invalid_ratio_raises(self):
        from labeling.sample_extractor import StratifiedSampler
        with pytest.raises(ValueError, match="1.0"):
            StratifiedSampler(ratios={"Red": 0.5, "Yellow": 0.3})  # 합 != 1.0

    def test_export_csv(self, features_df, tmp_path):
        from labeling.sample_extractor import StratifiedSampler
        sampler = StratifiedSampler(n_total=100)
        sample = sampler.extract(features_df)
        sample["expert_label"] = ""
        sample["review_status"] = "pending"
        out = str(tmp_path / "golden.csv")
        sampler.export_for_review(sample, out)
        assert os.path.exists(out)
        loaded = pd.read_csv(out)
        assert "expert_label" in loaded.columns
        assert "review_status" in loaded.columns

    def test_summarize_coverage(self, features_df):
        from labeling.sample_extractor import StratifiedSampler
        sampler = StratifiedSampler(n_total=100)
        sample = sampler.extract(features_df)
        summary = sampler.summarize(sample, features_df)
        assert summary.total == len(sample)
        assert 0 < summary.coverage_rate <= 1.0

    def test_add_age_group(self):
        from labeling.sample_extractor import add_age_group
        df = pd.DataFrame({"patient_age": [30, 65, 75, None]})
        df = add_age_group(df)
        assert df.loc[0, "age_group"] == "lt65"
        assert df.loc[1, "age_group"] == "65-74"
        assert df.loc[2, "age_group"] == "75plus"
        assert df.loc[3, "age_group"] == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# TestGoldenValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestGoldenValidator:
    @pytest.fixture
    def golden_df(self):
        """검증을 통과할 수 있는 골든 데이터셋 샘플."""
        n = 500
        risk = (["Red"] * 150 + ["Yellow"] * 150 + ["Green"] * 125 + ["Normal"] * 75)
        return pd.DataFrame({
            "patient_id":    [f"P{i:04d}" for i in range(n)],
            "risk_level":    risk,
            "expert_label":  risk,          # Rule과 일치 (일치율 100%)
            "review_status": ["reviewed"] * n,
        })

    def test_passes_with_good_data(self, golden_df):
        from labeling.golden_validator import GoldenValidator
        validator = GoldenValidator(target=500, min_coverage=0.80)
        report = validator.validate(golden_df)
        assert report.passed

    def test_fails_on_missing_columns(self):
        from labeling.golden_validator import GoldenValidator
        df = pd.DataFrame({"patient_id": ["P001"], "risk_level": ["Red"]})
        validator = GoldenValidator()
        report = validator.validate(df)
        col_check = next(c for c in report.checks if c.name == "필수 컬럼 존재")
        assert col_check.passed is False

    def test_fails_on_low_completion(self, golden_df):
        from labeling.golden_validator import GoldenValidator
        df = golden_df.copy()
        df.loc[:50, "expert_label"] = ""   # 50건 레이블 제거 → 완결률 하락
        validator = GoldenValidator(target=500, min_coverage=0.80)
        report = validator.validate(df)
        completion = next(c for c in report.checks if c.name == "레이블 완결률")
        assert not completion.passed

    def test_fails_on_low_coverage(self):
        from labeling.golden_validator import GoldenValidator
        n = 100  # 목표 5,650 대비 1.7% — 80% 미달
        df = pd.DataFrame({
            "patient_id":    [f"P{i}" for i in range(n)],
            "risk_level":    ["Red"] * n,
            "expert_label":  ["Red"] * n,
            "review_status": ["reviewed"] * n,
        })
        validator = GoldenValidator(target=5_650)
        report = validator.validate(df)
        coverage_check = next(c for c in report.checks if c.name == "목표 건수 달성률")
        assert coverage_check.passed is False

    def test_fails_on_low_agreement(self, golden_df):
        from labeling.golden_validator import GoldenValidator
        df = golden_df.copy()
        # 전문가가 절반을 다르게 레이블 → 일치율 50%
        df.loc[:249, "expert_label"] = "Normal"
        validator = GoldenValidator(target=500, min_coverage=0.80)
        report = validator.validate(df)
        agree = next(c for c in report.checks if "일치율" in c.name)
        assert agree.passed is False

    def test_cohen_kappa_perfect(self):
        from labeling.golden_validator import compute_cohens_kappa
        labels = ["Red", "Yellow", "Green", "Normal"] * 25
        kappa = compute_cohens_kappa(labels, labels)
        assert kappa == pytest.approx(1.0)

    def test_cohen_kappa_random(self):
        from labeling.golden_validator import compute_cohens_kappa
        np.random.seed(42)
        a = np.random.choice(["Red", "Yellow", "Green", "Normal"], 200).tolist()
        b = np.random.choice(["Red", "Yellow", "Green", "Normal"], 200).tolist()
        kappa = compute_cohens_kappa(a, b)
        assert -1.0 <= kappa <= 1.0

    def test_cohen_kappa_empty(self):
        from labeling.golden_validator import compute_cohens_kappa
        assert compute_cohens_kappa([], []) == 0.0

    def test_agreement_rate_perfect(self):
        from labeling.golden_validator import compute_agreement_rate
        labels = ["Red", "Yellow", "Green"]
        assert compute_agreement_rate(labels, labels) == pytest.approx(1.0)

    def test_agreement_rate_zero(self):
        from labeling.golden_validator import compute_agreement_rate
        a = ["Red", "Yellow"]
        b = ["Normal", "Green"]
        assert compute_agreement_rate(a, b) == pytest.approx(0.0)

    def test_with_second_reviewer(self, golden_df):
        from labeling.golden_validator import GoldenValidator
        df = golden_df.copy()
        df["reviewer2_label"] = df["expert_label"]   # 완전 일치
        validator = GoldenValidator(target=500, min_coverage=0.80)
        report = validator.validate(df, reviewer2_col="reviewer2_label")
        assert report.kappa == pytest.approx(1.0)

    def test_print_summary(self, golden_df, capsys):
        from labeling.golden_validator import GoldenValidator
        validator = GoldenValidator(target=500, min_coverage=0.80)
        report = validator.validate(golden_df)
        report.print_summary()
        out = capsys.readouterr().out
        assert "PASS" in out or "FAIL" in out


# ─────────────────────────────────────────────────────────────────────────────
# TestLabelingPipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelingPipeline:
    def test_stage1_adds_rule_label(self, features_df):
        from labeling.pipeline import LabelingPipeline
        pipeline = LabelingPipeline()
        df = pipeline.stage1_rule_labels(features_df)
        assert "rule_label" in df.columns
        assert "is_high_risk_rule" in df.columns
        assert df["rule_label"].equals(df["risk_level"])

    def test_stage1_missing_risk_level_raises(self):
        from labeling.pipeline import LabelingPipeline
        pipeline = LabelingPipeline()
        df = pd.DataFrame({"patient_id": ["P001"], "drug_count": [5]})
        with pytest.raises(ValueError, match="risk_level"):
            pipeline.stage1_rule_labels(df)

    def test_stage2_no_diagnosis_df(self, features_df):
        from labeling.pipeline import LabelingPipeline
        pipeline = LabelingPipeline()
        df = pipeline.stage1_rule_labels(features_df)
        result = pipeline.stage2_adr_labels(df, diagnosis_df=None)
        assert "adr_label" in result.columns
        assert result["adr_label"].sum() == 0  # 진단 데이터 없으면 ADR 없음

    def test_stage2_with_diagnosis(self, features_df, diagnosis_df):
        from labeling.pipeline import LabelingPipeline
        pipeline = LabelingPipeline()
        df = pipeline.stage1_rule_labels(features_df)
        result = pipeline.stage2_adr_labels(df, diagnosis_df)
        assert result["adr_label"].sum() > 0

    def test_stage3_creates_golden_sample(self, features_df, tmp_path):
        from labeling.pipeline import LabelingPipeline
        pipeline = LabelingPipeline(output_dir=str(tmp_path), golden_target=100)
        df = pipeline.stage1_rule_labels(features_df)
        df = pipeline.stage2_adr_labels(df)
        golden, report = pipeline.stage3_golden_sample(df, "20260319")
        assert len(golden) <= 100
        assert "expert_label" in golden.columns
        assert "review_status" in golden.columns
        assert all(golden["review_status"] == "pending")

    def test_full_pipeline_run(self, features_df, diagnosis_df, tmp_path):
        from labeling.pipeline import run_labeling
        result = run_labeling(
            features_df=features_df,
            diagnosis_df=diagnosis_df,
            partition="20260319",
            output_dir=str(tmp_path),
        )
        assert result.n_total == len(features_df)
        assert result.n_adr_positive >= 0
        assert result.n_golden_sample > 0
        assert os.path.exists(result.labels_path)
        assert os.path.exists(result.golden_path)
        assert os.path.exists(result.validation_path)

    def test_validation_json_structure(self, features_df, tmp_path):
        from labeling.pipeline import run_labeling
        result = run_labeling(
            features_df=features_df,
            partition="20260319",
            output_dir=str(tmp_path),
        )
        with open(result.validation_path, encoding="utf-8") as f:
            val = json.load(f)
        assert "total_records" in val
        assert "checks" in val
        assert isinstance(val["checks"], list)

    def test_labels_parquet_saved(self, features_df, tmp_path):
        from labeling.pipeline import run_labeling
        result = run_labeling(
            features_df=features_df,
            partition="20260319",
            output_dir=str(tmp_path),
        )
        df = pd.read_parquet(result.labels_path)
        assert len(df) == len(features_df)
        assert "rule_label" in df.columns

    def test_adr_rate_property(self, features_df, diagnosis_df, tmp_path):
        from labeling.pipeline import run_labeling
        result = run_labeling(
            features_df=features_df,
            diagnosis_df=diagnosis_df,
            partition="20260319",
            output_dir=str(tmp_path),
        )
        assert 0.0 <= result.adr_rate <= 1.0
