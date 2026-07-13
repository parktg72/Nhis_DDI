"""
ETL 파이프라인 오케스트레이터
T20/T30/T40/T60/요양기관 → 피처 스토어까지 전체 흐름 실행

실제 NHIS 레이아웃 기준 (lay_out/t20.txt ~ t60.txt, 요양기관.txt)

실행 흐름:
  1. 스키마 검증 (schema_validator)
  2. 가명처리 (pseudonymizer) ← 원본 데이터가 미가명 상태인 경우
  3. MCARE_DIV_CD→ATC 코드 표준화 (code_standardizer)
  4. 데이터 품질 검사 (quality_checker)
  5. T20+T30 조인 → 처방 레코드 (T40/T60/요양기관 선택 join)
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
from .drug_master import DrugMaster
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
        drug_master_path: str | Path = "data/processed/hira_drug_master.parquet",
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

        # DrugMaster: WK_COMPN_CD → DDI 매트릭스 ID 변환에 필요
        self.drug_master: Optional[DrugMaster] = None
        if Path(drug_master_path).exists():
            self.drug_master = DrugMaster.load_parquet(
                drug_master_path, ddi_matrix_path=ddi_matrix_path
            )
            logger.info("DrugMaster 로드: %s", drug_master_path)
        else:
            logger.warning("DrugMaster 없음 (DDI 매칭 비활성화): %s", drug_master_path)

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
        t60: Optional[pd.DataFrame] = None,
        yoyang: Optional[pd.DataFrame] = None,
        # 하위 호환 파라미터 (t50 → yoyang 로 대체)
        t50: Optional[pd.DataFrame] = None,
        partition: Optional[str] = None,
    ) -> PipelineResult:
        """
        전체 ETL 파이프라인 실행.

        Parameters
        ----------
        t20, t30  : 필수 — 진료명세서(T20), 진료내역(T30)
        t40       : 선택 — 상병내역(T40)
        t60       : 선택 — 처방전내역(T60, 원외처방)
        yoyang    : 선택 — 요양기관 현황 (t50 파라미터도 동일하게 허용)
        partition : 파티션 식별자 (YYYYMM). None이면 오늘 날짜 사용.
        """
        # 하위 호환: t50 → yoyang
        if yoyang is None and t50 is not None:
            yoyang = t50
        t0 = time.perf_counter()
        if partition is None:
            partition = date.today().strftime("%Y%m")

        result = PipelineResult(partition=partition)

        # ── Step 1: 스키마 검증 ─────────────────────────────────────────────
        logger.info("[Step 1] 스키마 검증")
        val_results = validate_all(t20, t30, t40=t40, t60=t60, yoyang=yoyang)
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
            from .pseudonymizer import PSEUDO_COLUMNS, pseudonymize_dataframe
            t20 = pseudonymize_dataframe(t20, PSEUDO_COLUMNS.get("T20", []))
            t30 = pseudonymize_dataframe(t30, PSEUDO_COLUMNS.get("T30", []))
            if t40 is not None:
                t40 = pseudonymize_dataframe(t40, PSEUDO_COLUMNS.get("T40", []))
            if t60 is not None:
                t60 = pseudonymize_dataframe(t60, PSEUDO_COLUMNS.get("T60", []))
            if yoyang is not None:
                yoyang = pseudonymize_dataframe(yoyang, PSEUDO_COLUMNS.get("YOYANG", []))
        else:
            logger.info("[Step 2] 가명처리 스킵 (pseudonymize_input=False)")

        # ── Step 3: MCARE_DIV_CD→ATC 코드 표준화 ────────────────────────────
        logger.info("[Step 3] 코드 표준화 (MCARE_DIV_CD→ATC)")
        t30_std = self.standardizer.standardize(t30, edi_col="MCARE_DIV_CD")
        wk_unknown = self.standardizer.unknown_rate(t30, wk_col="WK_COMPN_CD")
        logger.info("  MCARE_DIV_CD 미매핑율: %.1f%%", wk_unknown * 100)

        # T60도 동일한 표준화 적용 (atc_code, drug_name 없으면 ATC/위험 피처 누락)
        if t60 is not None and "MCARE_DIV_CD" in t60.columns:
            t60 = self.standardizer.standardize(t60, edi_col="MCARE_DIV_CD")
            logger.info("  T60 코드 표준화 완료")

        # ── Step 4: 품질 검사 ────────────────────────────────────────────────
        logger.info("[Step 4] 데이터 품질 검사")
        quality_reports = check_all(t20, t30_std, wk_compn_unknown_rate=wk_unknown)
        print_quality_summary(quality_reports)

        for tbl, qr in quality_reports.items():
            if not qr.passed:
                for w in qr.warnings:
                    logger.warning("[품질경고] %s: %s", tbl, w)

        # ── Step 5: T20+T30 조인 ────────────────────────────────────────────
        logger.info("[Step 5] T20+T30 조인")
        merged = self._join_t20_t30(t20, t30_std, t40, yoyang, t60)
        result.total_prescriptions = len(t20)
        result.total_drug_items = len(t30_std)
        result.total_patients = merged["INDI_DSCM_NO"].nunique()
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
            drug_master=self.drug_master,
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
        yoyang: Optional[pd.DataFrame],
        t60: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """T20+T30 inner join → T40/T60/요양기관 left join.

        조인 키:
          T20 ↔ T30    : CMN_KEY
          T20 ↔ T40    : CMN_KEY (상병 정보 보완)
          T20 ↔ T60    : CMN_KEY (원외처방 약품 보완)
          T20 ↔ 요양기관 : MDCARE_SYM (기관 종별코드 보완)
        """
        # T30에 T20 정보 병합 (명세서 단위 메타데이터)
        t20_cols = [
            c for c in [
                "CMN_KEY", "INDI_DSCM_NO", "MDCARE_SYM",
                "MDCARE_STRT_DT", "MDCARE_STRT_YYYYMM",
                "SICK_SYM1", "SICK_SYM2",
                "SEX_TYPE", "SUJIN_POTM_AGE_ID",
                "YOYANG_CLSFC_CD", "MCARE_TP", "WMED_OTMED_TYPE",
            ] if c in t20.columns
        ]
        merged = t30.merge(t20[t20_cols], on="CMN_KEY", how="inner", suffixes=("", "_t20"))

        # T40: 상병 정보 (MCEX_SICK_SYM) 보완 — 주상병 기준 1건만
        if t40 is not None and "CMN_KEY" in t40.columns:
            t40_main = (
                t40[t40["SICK_CLSF_TYPE"].astype(str) == "1"]
                if "SICK_CLSF_TYPE" in t40.columns
                else t40
            ).drop_duplicates("CMN_KEY")
            t40_cols = [c for c in ["CMN_KEY", "MCEX_SICK_SYM"] if c in t40_main.columns]
            if len(t40_cols) > 1:
                merged = merged.merge(t40_main[t40_cols], on="CMN_KEY", how="left")

        # 요양기관: 기관 상세 종별코드 보완
        if yoyang is not None and "MDCARE_SYM" in yoyang.columns and "MDCARE_SYM" in merged.columns:
            yoyang_cols = [
                c for c in ["MDCARE_SYM", "YOYANG_DETAIL_CLSFC_CD", "ADDR_SGG_CD"]
                if c in yoyang.columns
            ]
            if len(yoyang_cols) > 1:
                merged = merged.merge(
                    yoyang[yoyang_cols].drop_duplicates("MDCARE_SYM"),
                    on="MDCARE_SYM",
                    how="left",
                )

        # T60: 원외처방 약품 — T20 메타데이터 보완 후 처방 레코드에 합산
        if t60 is not None and "CMN_KEY" in t60.columns:
            t60_work = t60.copy()
            # RVSN_WK_COMPN_CD → WK_COMPN_CD (T30 컬럼명 호환)
            if "WK_COMPN_CD" not in t60_work.columns and "RVSN_WK_COMPN_CD" in t60_work.columns:
                t60_work = t60_work.rename(columns={"RVSN_WK_COMPN_CD": "WK_COMPN_CD"})
            t60_with_demo = t60_work.merge(
                t20[t20_cols], on="CMN_KEY", how="inner", suffixes=("", "_t20")
            )
            # T20에서 온 중복 컬럼(_t20 suffix) 제거
            t60_with_demo = t60_with_demo.drop(
                columns=[c for c in t60_with_demo.columns if c.endswith("_t20")]
            )
            # merged와 공통 컬럼만 선택하여 concat
            common_cols = [c for c in merged.columns if c in t60_with_demo.columns]
            merged = pd.concat([merged, t60_with_demo[common_cols]], ignore_index=True)

        return merged


def run_pipeline(
    t20: pd.DataFrame,
    t30: pd.DataFrame,
    t40: Optional[pd.DataFrame] = None,
    t60: Optional[pd.DataFrame] = None,
    yoyang: Optional[pd.DataFrame] = None,
    # 하위 호환
    t50: Optional[pd.DataFrame] = None,
    partition: Optional[str] = None,
    **kwargs,
) -> PipelineResult:
    """편의 함수: ETLPipeline 인스턴스 생성 후 바로 실행."""
    pipeline = ETLPipeline(**kwargs)
    return pipeline.run(t20, t30, t40=t40, t60=t60, yoyang=yoyang, t50=t50, partition=partition)
