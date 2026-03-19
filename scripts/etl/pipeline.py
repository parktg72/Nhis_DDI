"""
ETL 파이프라인 오케스트레이터
T20/T30/T40/T50 → 피처 스토어까지 전체 흐름 실행

실행 흐름:
  1. 스키마 검증 (schema_validator)
  2. 가명처리 (pseudonymizer) ← 원본 데이터가 미가명 상태인 경우
  3. EDI→ATC 코드 표준화 (code_standardizer)
  4. 데이터 품질 검사 (quality_checker)
  5. T20+T30 조인 → 처방 레코드
  6. 동시복용 계산 (overlap_calculator)
  7. 피처 집계 (prescription_aggregator)
  8. 피처 저장 (feature_writer)
"""
from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from .code_standardizer import CodeStandardizer
from .feature_writer import (
    print_feature_summary,
    write_features,
    write_overlap_pairs,
    write_pipeline_log,
)
from .models import PipelineResult
from .overlap_calculator import calculate_overlaps_batch
from .prescription_aggregator import aggregate_batch
from .quality_checker import check_all, print_quality_summary
from .schema_validator import validate_all

logger = logging.getLogger(__name__)


class ETLPipeline:
    """
    DDI 모델 ETL 파이프라인.

    Parameters
    ----------
    ddi_matrix_path : DDI 매트릭스 parquet 경로
    dup_groups_path : 효능군 중복 그룹 parquet 경로
    drug_index_path : drug_name_index parquet 경로
    feature_base_dir : 피처 저장 기본 디렉토리
    pseudonymize_input : True면 입력 데이터에 가명처리 적용
    overwrite : 기존 피처 파일 덮어쓰기
    """

    def __init__(
        self,
        ddi_matrix_path: str | Path = "data/processed/ddi_matrix_final.parquet",
        dup_groups_path: str | Path = "data/processed/efcy_duplicate_groups.parquet",
        drug_index_path: str | Path = "data/processed/drug_name_index.parquet",
        feature_base_dir: str | Path = "data/features",
        pseudonymize_input: bool = False,
        overwrite: bool = False,
    ):
        self.feature_base_dir = Path(feature_base_dir)
        self.pseudonymize_input = pseudonymize_input
        self.overwrite = overwrite

        # DDI 매트릭스 로드
        self.ddi_matrix: Optional[pd.DataFrame] = None
        if Path(ddi_matrix_path).exists():
            self.ddi_matrix = pd.read_parquet(ddi_matrix_path)
            logger.info("DDI 매트릭스 로드: %d행", len(self.ddi_matrix))
        else:
            logger.warning("DDI 매트릭스 없음: %s", ddi_matrix_path)

        # 효능군 중복 그룹 로드
        self.dup_groups: Optional[pd.DataFrame] = None
        if Path(dup_groups_path).exists():
            self.dup_groups = pd.read_parquet(dup_groups_path)
            logger.info("효능군 중복 그룹 로드: %d행", len(self.dup_groups))
        else:
            logger.warning("효능군 중복 그룹 없음: %s", dup_groups_path)

        # 코드 표준화기
        self.standardizer = CodeStandardizer(
            index_path=drug_index_path,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 메인 실행
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        t20: pd.DataFrame,
        t30: pd.DataFrame,
        t40: Optional[pd.DataFrame] = None,
        t50: Optional[pd.DataFrame] = None,
        partition: Optional[str] = None,
    ) -> PipelineResult:
        """
        전체 ETL 파이프라인 실행.

        Parameters
        ----------
        t20, t30, t40, t50 : 청구 데이터 DataFrame
        partition : 파티션 식별자 (YYYYMM). None이면 오늘 날짜 사용.
        """
        t0 = time.perf_counter()
        if partition is None:
            partition = date.today().strftime("%Y%m")

        result = PipelineResult(partition=partition)

        # ── Step 1: 스키마 검증 ─────────────────────────────────────────────
        logger.info("[Step 1] 스키마 검증")
        val_results = validate_all(t20, t30, t40, t50)
        for tbl, vr in val_results.items():
            if not vr.passed:
                msg = f"{tbl} 스키마 검증 실패: missing={vr.missing_cols}, null={vr.null_violations}"
                logger.error(msg)
                result.errors.append(msg)

        if result.errors:
            result.elapsed_seconds = time.perf_counter() - t0
            return result

        # ── Step 2: 가명처리 (옵션) ──────────────────────────────────────────
        if self.pseudonymize_input:
            logger.info("[Step 2] 가명처리")
            from .pseudonymizer import pseudonymize_dataframe, PSEUDO_COLUMNS
            t20 = pseudonymize_dataframe(t20, PSEUDO_COLUMNS.get("T20", []))
            if t40 is not None:
                t40 = pseudonymize_dataframe(t40, PSEUDO_COLUMNS.get("T40", []))
            if t50 is not None:
                t50 = pseudonymize_dataframe(t50, PSEUDO_COLUMNS.get("T50", []))
        else:
            logger.info("[Step 2] 가명처리 스킵 (pseudonymize_input=False)")

        # ── Step 3: EDI→ATC 코드 표준화 ──────────────────────────────────────
        logger.info("[Step 3] 코드 표준화 (EDI→ATC)")
        t30_std = self.standardizer.standardize(t30, edi_col="EDI_CD")
        edi_unknown = self.standardizer.unknown_rate(t30, edi_col="EDI_CD")
        logger.info("  EDI 미매핑율: %.1f%%", edi_unknown * 100)

        # ── Step 4: 품질 검사 ────────────────────────────────────────────────
        logger.info("[Step 4] 데이터 품질 검사")
        quality_reports = check_all(t20, t30_std, edi_unknown_rate=edi_unknown)
        print_quality_summary(quality_reports)

        for tbl, qr in quality_reports.items():
            if not qr.passed:
                for w in qr.warnings:
                    logger.warning("[품질경고] %s: %s", tbl, w)

        # ── Step 5: T20+T30 조인 ────────────────────────────────────────────
        logger.info("[Step 5] T20+T30 조인")
        merged = self._join_t20_t30(t20, t30_std, t40, t50)
        result.total_prescriptions = len(t20)
        result.total_drug_items = len(t30_std)
        result.total_patients = merged["BNFCR_PSEUDO"].nunique()
        logger.info("  환자 %d명, 처방약물 %d건", result.total_patients, result.total_drug_items)

        # ── Step 6: 동시복용 계산 ─────────────────────────────────────────────
        logger.info("[Step 6] 동시복용 계산")
        overlap_df = calculate_overlaps_batch(merged)
        result.overlap_pairs = len(overlap_df)
        logger.info("  동시복용 쌍: %d건", result.overlap_pairs)

        # ── Step 7: 피처 집계 ────────────────────────────────────────────────
        logger.info("[Step 7] 피처 집계")
        all_features = aggregate_batch(
            df_prescriptions=merged,
            df_t40=t40,
            overlap_df=overlap_df,
            ddi_matrix=self.ddi_matrix,
            dup_groups=self.dup_groups,
        )
        result.features_written = len(all_features)

        # 위험도 분포 집계
        for f in all_features:
            if f.risk_level == "Red":
                result.red_count += 1
            elif f.risk_level == "Yellow":
                result.yellow_count += 1
            elif f.risk_level == "Green":
                result.green_count += 1
            else:
                result.normal_count += 1

        print_feature_summary(all_features)

        # ── Step 8: 저장 ─────────────────────────────────────────────────────
        logger.info("[Step 8] 피처 저장")
        try:
            write_features(all_features, partition, self.feature_base_dir, self.overwrite)
            write_overlap_pairs(overlap_df, partition, self.feature_base_dir, self.overwrite)
        except FileExistsError as e:
            logger.error(str(e))
            result.errors.append(str(e))

        result.elapsed_seconds = time.perf_counter() - t0
        write_pipeline_log(result, self.feature_base_dir)

        logger.info(
            "[완료] 파티션=%s, 환자=%d명, 소요=%.1f초",
            partition, result.total_patients, result.elapsed_seconds,
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _join_t20_t30(
        t20: pd.DataFrame,
        t30: pd.DataFrame,
        t40: Optional[pd.DataFrame],
        t50: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """T20+T30 inner join → T40/T50 left join."""
        merged = t30.merge(t20, on="MDCARE_BILL_NO", how="inner")

        if t40 is not None and "BNFCR_PSEUDO" in t40.columns:
            merged = merged.merge(
                t40[["BNFCR_PSEUDO", "SEX_TP_CD", "BTH_YYYY"]],
                on="BNFCR_PSEUDO",
                how="left",
            )

        if t50 is not None and "INST_PSEUDO" in t50.columns:
            merged = merged.merge(
                t50[["INST_PSEUDO", "CLNC_TP_CD"]],
                on="INST_PSEUDO",
                how="left",
            )

        return merged


def run_pipeline(
    t20: pd.DataFrame,
    t30: pd.DataFrame,
    t40: Optional[pd.DataFrame] = None,
    t50: Optional[pd.DataFrame] = None,
    partition: Optional[str] = None,
    **kwargs,
) -> PipelineResult:
    """편의 함수: ETLPipeline 인스턴스 생성 후 바로 실행."""
    pipeline = ETLPipeline(**kwargs)
    return pipeline.run(t20, t30, t40, t50, partition=partition)
