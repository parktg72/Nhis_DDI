"""
피처 엔지니어링 통합 모듈

ETL PatientFeatures + CYP 피처 + 시계열 피처 → 최종 ML 입력 피처 행렬

파이프라인:
  1. ETL 피처 로드 (patient_features_{partition}.parquet)
  2. CYP 피처 추출 (overlap_pairs + cyp_matrix)
  3. 시계열 피처 추출 (처방 레코드)
  4. 병합 → 결측치 처리 → 정규화 → 피처 선택
  5. ML 입력용 최종 피처 저장

출력:
  data/features/ml_features_{partition}.parquet
  data/features/scaler.pkl
  data/features/selector.pkl
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from .cyp_features import CYPFeatureExtractor
from .normalizer import FeatureNormalizer
from .selector import FeatureSelector, ensure_sex_type_metadata
from .temporal_features import extract_temporal_batch

logger = logging.getLogger(__name__)

ML_FEATURE_OUTPUT = "data/features/ml_features_{partition}.parquet"

# ETL 피처 중 ML에 사용할 컬럼 (메타/레이블 제외)
ETL_NUMERIC_COLS = [
    "drug_count", "drug_count_7d", "institution_count",
    "ddi_contraindicated", "ddi_major", "ddi_moderate", "ddi_minor",
    "triple_whammy", "qt_risk_count",
    "dup_same_ingredient", "dup_atc5", "dup_atc4", "dup_atc3",
    "age",
]

# 레이블 컬럼
LABEL_COL = "risk_level"
BINARY_LABEL_COL = "is_high_risk"   # Red=1, Yellow/Green/Normal=0


class FeatureEngineer:
    """
    전체 피처 엔지니어링 파이프라인.

    Parameters
    ----------
    cyp_extractor : CYP 피처 추출기 (None이면 CYP 피처 생략)
    normalizer : 정규화기 (None이면 자동 생성)
    selector : 피처 선택기 (None이면 자동 생성)
    fit_mode : True이면 normalizer/selector를 데이터에 fit
    """

    def __init__(
        self,
        cyp_extractor: CYPFeatureExtractor | None = None,
        normalizer: FeatureNormalizer | None = None,
        selector: FeatureSelector | None = None,
        fit_mode: bool = True,
        feature_base: str | Path = "data/features",
    ):
        self.cyp = cyp_extractor
        self.normalizer = normalizer or FeatureNormalizer()
        self.selector = selector or FeatureSelector()
        self.fit_mode = fit_mode
        self.feature_base = Path(feature_base)

    # ──────────────────────────────────────────────────────────────────────────
    # 메인 실행
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        partition: str,
        prescription_records: dict | None = None,
    ) -> pd.DataFrame:
        """
        파티션 피처 엔지니어링 전체 실행.

        Parameters
        ----------
        partition : 'YYYYMM'
        prescription_records : {patient_id: [PrescriptionRecord]} (시계열 피처용)
        """
        t0 = time.perf_counter()
        logger.info("[FeatureEngineer] 파티션=%s 시작", partition)

        # ── Step 1: ETL 기본 피처 로드 ──────────────────────────────────────
        etl_path = self.feature_base / f"patient_features_{partition}.parquet"
        if not etl_path.exists():
            raise FileNotFoundError(f"ETL 피처 없음: {etl_path}")

        df = pd.read_parquet(etl_path)
        logger.info("  ETL 피처 로드: %d명", len(df))

        # ── Step 2: CYP 피처 ────────────────────────────────────────────────
        overlap_path = self.feature_base / f"overlap_pairs_{partition}.parquet"
        if self.cyp is not None and overlap_path.exists():
            cyp_df = self._extract_cyp_features(pd.read_parquet(overlap_path))
            df = df.merge(cyp_df, on="patient_id", how="left")
            logger.info("  CYP 피처 추가: %d컬럼", len(cyp_df.columns) - 1)
        else:
            logger.info("  CYP 피처 스킵")

        # ── Step 3: 시계열 피처 ──────────────────────────────────────────────
        if prescription_records:
            temp_df = extract_temporal_batch(prescription_records)
            df = df.merge(temp_df, on="patient_id", how="left")
            logger.info("  시계열 피처 추가: %d컬럼", len(temp_df.columns) - 1)

        # ── Step 4: 레이블 생성 ───────────────────────────────────────────────
        if LABEL_COL in df.columns:
            df[BINARY_LABEL_COL] = (df[LABEL_COL] == "Red").astype(int)

        # ── Step 5: sex 인코딩 ─────────────────────────────────────────────
        if "sex" in df.columns:
            df = ensure_sex_type_metadata(df)
            df["sex_male"] = (df["sex"] == "M").astype(float)
            df = df.drop(columns=["sex"])

        # ── Step 6: 정규화 ────────────────────────────────────────────────────
        if self.fit_mode:
            df_scaled = self.normalizer.fit_transform(df)
        else:
            df_scaled = self.normalizer.transform(df)

        # ── Step 7: 피처 선택 ────────────────────────────────────────────────
        if self.fit_mode:
            df_final = self.selector.fit_transform(df_scaled)
        else:
            df_final = self.selector.transform(df_scaled)

        # ── Step 8: 저장 ─────────────────────────────────────────────────────
        self.feature_base.mkdir(parents=True, exist_ok=True)
        out_path = self.feature_base / f"ml_features_{partition}.parquet"
        df_final.to_parquet(out_path, index=False)

        if self.fit_mode:
            self.normalizer.save(self.feature_base / "scaler.pkl")
            self.selector.save(self.feature_base / "selector.pkl")

        elapsed = time.perf_counter() - t0
        logger.info(
            "  [완료] %d명 × %d 피처 → %s (%.1f초)",
            len(df_final), len(df_final.columns), out_path, elapsed,
        )
        self.selector.report()
        return df_final

    # ──────────────────────────────────────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_cyp_features(self, overlap_df: pd.DataFrame) -> pd.DataFrame:
        """
        overlap_pairs DataFrame → 환자별 CYP 피처 DataFrame.
        ATC 코드가 없는 경우 건너뜀.
        """
        # 환자별 약물 ATC 코드 집합 수집
        patient_atcs: dict[str, set[str]] = {}
        for _, row in overlap_df.iterrows():
            pid = str(row.get("patient_id", ""))
            for col in ["drug_a_atc", "drug_b_atc"]:
                atc = row.get(col)
                if atc and pd.notna(atc):
                    patient_atcs.setdefault(pid, set()).add(str(atc))

        rows = []
        for pid, atcs in patient_atcs.items():
            feat = self.cyp.extract(list(atcs))
            feat["patient_id"] = pid
            rows.append(feat)

        if not rows:
            return pd.DataFrame(columns=["patient_id"])

        result = pd.DataFrame(rows)
        cols = ["patient_id"] + [c for c in result.columns if c != "patient_id"]
        return result[cols]


def build_ml_features(
    partition: str,
    feature_base: str | Path = "data/features",
    cyp_matrix_path: str | Path = "data/processed/cyp_matrix.parquet",
    drug_index_path: str | Path = "data/processed/drug_name_index.parquet",
    prescription_records: dict | None = None,
    fit_mode: bool = True,
) -> pd.DataFrame:
    """
    편의 함수: FeatureEngineer 인스턴스 생성 후 실행.

    Returns
    -------
    최종 ML 피처 DataFrame
    """
    cyp = None
    if Path(cyp_matrix_path).exists():
        cyp = CYPFeatureExtractor(
            cyp_matrix_path=cyp_matrix_path,
            drug_index_path=drug_index_path,
        )

    engineer = FeatureEngineer(
        cyp_extractor=cyp,
        fit_mode=fit_mode,
        feature_base=feature_base,
    )
    return engineer.run(partition, prescription_records)
