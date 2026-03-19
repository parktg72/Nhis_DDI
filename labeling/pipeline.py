"""
레이블링 파이프라인 통합

PROJECT_PLAN 4.4 레이블링 3단계 전략 통합 실행:

  Stage 1 — Rule 레이블
    scripts/etl/ 결과에서 rule_level(risk_level) 사용.
    모든 환자에 대한 1차 레이블.

  Stage 2 — ADR 프록시 레이블 (후향적)
    ICD-10 상병코드 기반 ADR 발생 여부로 레이블 보강.
    ADR 발생 환자 → Red 레이블 보강.

  Stage 3 — 골든 샘플 추출 및 검증
    계층화 샘플링 → 전문가 검토 파일 생성 → 품질 검증.

출력:
  data/labeling/labels_{partition}.parquet     : 전체 레이블
  data/labeling/golden_{partition}.csv         : 전문가 검토용 샘플
  data/labeling/validation_{partition}.json    : 검증 리포트
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from labeling.adr_labeler import ADRLabeler, LabelResult
from labeling.sample_extractor import StratifiedSampler, add_age_group
from labeling.golden_validator import GoldenValidator, GoldenValidationReport

logger = logging.getLogger(__name__)


@dataclass
class LabelingResult:
    """파이프라인 실행 결과."""
    partition: str
    n_total: int
    n_adr_positive: int
    n_golden_sample: int
    validation_passed: bool
    labels_path: str
    golden_path: str
    validation_path: str
    elapsed_sec: float

    @property
    def adr_rate(self) -> float:
        return self.n_adr_positive / max(self.n_total, 1)


class LabelingPipeline:
    """3단계 레이블링 파이프라인.

    Usage:
        pipeline = LabelingPipeline(output_dir="data/labeling")
        result = pipeline.run(
            features_df=features_df,
            diagnosis_df=diagnosis_df,
            partition="20260319",
        )
    """

    def __init__(
        self,
        output_dir: str = "data/labeling",
        golden_target: int = 5_650,
        adr_lookback_days: int = 90,
        seed: int = 42,
    ):
        self._output_dir = output_dir
        self._labeler   = ADRLabeler(lookback_days=adr_lookback_days)
        self._sampler   = StratifiedSampler(n_total=golden_target, seed=seed)
        self._validator = GoldenValidator(target=golden_target)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1 — Rule 레이블 확인
    # ─────────────────────────────────────────────────────────────────────────

    def stage1_rule_labels(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """ETL 결과의 risk_level을 rule_label로 매핑."""
        df = features_df.copy()
        if "risk_level" not in df.columns:
            raise ValueError("'risk_level' 컬럼 없음 — ETL 결과 확인 필요")

        df["rule_label"] = df["risk_level"]
        df["is_high_risk_rule"] = (df["risk_level"] == "Red").astype(int)

        dist = df["risk_level"].value_counts().to_dict()
        logger.info("Stage 1 Rule 레이블 — 분포: %s", dist)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 — ADR 프록시 레이블 병합
    # ─────────────────────────────────────────────────────────────────────────

    def stage2_adr_labels(
        self,
        features_df: pd.DataFrame,
        diagnosis_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """ADR 프록시 레이블 생성 및 features_df에 병합.

        diagnosis_df 없으면 ADR 레이블 없이 진행 (rule_label만 사용).
        diagnosis_df 필수 컬럼: patient_id, icd10_code, days_after_rx
        """
        df = features_df.copy()

        if diagnosis_df is None or diagnosis_df.empty:
            logger.info("Stage 2 ADR 레이블 — diagnosis_df 없음, rule_label 사용")
            df["adr_label"]      = 0
            df["adr_score"]      = 0.0
            df["adr_confidence"] = "LOW"
            df["final_label"]    = df.get("risk_level", pd.Series("Normal", index=df.index))
            return df

        # 환자별 진단코드 딕셔너리 구축
        diag_map: dict = {}
        for pid, grp in diagnosis_df.groupby("patient_id"):
            pairs = list(zip(grp["icd10_code"], grp["days_after_rx"]))
            diag_map[str(pid)] = pairs

        results: list[LabelResult] = []
        for _, row in df.iterrows():
            pid = str(row["patient_id"])
            atc_codes  = list(row.get("atc_codes", []) or [])
            diag_codes = diag_map.get(pid, [])
            rule_level = row.get("risk_level", None)

            res = self._labeler.label(
                patient_id=pid,
                atc_codes=atc_codes,
                diagnosis_codes=diag_codes,
                rule_risk_level=rule_level,
            )
            results.append(res)

        adr_df = pd.DataFrame([r.to_dict() for r in results])
        adr_df = adr_df.rename(columns={"label": "adr_label", "confidence": "adr_confidence"})

        df = df.merge(
            adr_df[["patient_id", "adr_label", "adr_score", "adr_confidence", "final_label"]],
            on="patient_id", how="left",
        )

        n_adr = int(df["adr_label"].sum())
        logger.info("Stage 2 ADR 레이블 — ADR 발생: %d건 (%.1f%%)",
                    n_adr, n_adr / max(len(df), 1) * 100)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3 — 골든 샘플 추출 및 검증
    # ─────────────────────────────────────────────────────────────────────────

    def stage3_golden_sample(
        self,
        df: pd.DataFrame,
        partition: str,
    ) -> tuple[pd.DataFrame, GoldenValidationReport]:
        """계층화 샘플 추출 → 전문가 검토 파일 생성 → 초기 검증."""
        # 연령대 컬럼 추가
        df = add_age_group(df, age_col="patient_age")

        # 샘플 추출 (adr_label 기반 내부 계층화)
        adr_col = "adr_label" if "adr_label" in df.columns else None
        sample = self._sampler.extract(df, risk_col="risk_level", adr_col=adr_col)

        # expert_label 초기화 (rule_label 복사 — 전문가 수정 전 기본값)
        sample["expert_label"]   = sample.get("rule_label", sample.get("risk_level", ""))
        sample["expert_comment"] = ""
        sample["review_status"]  = "pending"

        # 검증 (이 시점에선 전문가 검토 전이므로 rule_label 기준 검증)
        report = self._validator.validate(
            sample,
            rule_label_col="risk_level",
            expert_label_col="expert_label",
        )

        summary = self._sampler.summarize(sample, df)
        logger.info("Stage 3 골든 샘플 — %s", summary.to_dict())
        return sample, report

    # ─────────────────────────────────────────────────────────────────────────
    # 전체 파이프라인
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        features_df: pd.DataFrame,
        diagnosis_df: Optional[pd.DataFrame] = None,
        partition: str = "",
    ) -> LabelingResult:
        """3단계 레이블링 파이프라인 실행."""
        import time
        t0 = time.perf_counter()
        os.makedirs(self._output_dir, exist_ok=True)

        partition = partition or datetime.now().strftime("%Y%m%d")
        logger.info("레이블링 파이프라인 시작 — 파티션: %s", partition)

        # Stage 1
        df = self.stage1_rule_labels(features_df)

        # Stage 2
        df = self.stage2_adr_labels(df, diagnosis_df)

        # 전체 레이블 저장
        labels_path = os.path.join(self._output_dir, f"labels_{partition}.parquet")
        df.to_parquet(labels_path, index=False)

        # Stage 3
        golden_df, validation = self.stage3_golden_sample(df, partition)

        golden_path = os.path.join(self._output_dir, f"golden_{partition}.csv")
        self._sampler.export_for_review(golden_df, golden_path)

        val_path = os.path.join(self._output_dir, f"validation_{partition}.json")
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(validation.to_dict(), f, ensure_ascii=False, indent=2)

        elapsed = time.perf_counter() - t0
        n_adr = int(df.get("adr_label", pd.Series([0])).sum()) if "adr_label" in df.columns else 0

        result = LabelingResult(
            partition=partition,
            n_total=len(df),
            n_adr_positive=n_adr,
            n_golden_sample=len(golden_df),
            validation_passed=validation.passed,
            labels_path=labels_path,
            golden_path=golden_path,
            validation_path=val_path,
            elapsed_sec=elapsed,
        )

        logger.info(
            "레이블링 완료 — 전체:%d 골든:%d ADR:%d(%.1f%%) 검증:%s (%.1fs)",
            result.n_total, result.n_golden_sample,
            result.n_adr_positive, result.adr_rate * 100,
            "PASS" if result.validation_passed else "FAIL",
            elapsed,
        )
        return result


def run_labeling(
    features_df: pd.DataFrame,
    diagnosis_df: Optional[pd.DataFrame] = None,
    partition: str = "",
    output_dir: str = "data/labeling",
) -> LabelingResult:
    """레이블링 파이프라인 편의 함수."""
    return LabelingPipeline(output_dir=output_dir).run(
        features_df=features_df,
        diagnosis_df=diagnosis_df,
        partition=partition,
    )
