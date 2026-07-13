"""
Feature Store 저장
PatientFeatures 목록 → Parquet 파일 (파티션별)

파티션 구조:
  data/features/
    patient_features_{YYYYMM}.parquet  # 월별 파티션
    overlap_pairs_{YYYYMM}.parquet     # 동시복용 쌍
    pipeline_log_{YYYYMM}.json         # 파이프라인 실행 로그
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import PatientFeatures, PipelineResult

logger = logging.getLogger(__name__)

FEATURE_BASE = Path("data/features")


def features_to_df(features: list[PatientFeatures]) -> pd.DataFrame:
    """PatientFeatures 목록 → DataFrame."""
    if not features:
        return pd.DataFrame()
    rows = []
    for f in features:
        row = {
            "patient_id":          f.patient_id,
            "window_start":        f.window_start,
            "window_end":          f.window_end,
            "drug_count":          f.drug_count,
            "drug_count_7d":       f.drug_count_7d,
            "institution_count":   f.institution_count,
            "ddi_contraindicated": f.ddi_contraindicated,
            "ddi_major":           f.ddi_major,
            "ddi_moderate":        f.ddi_moderate,
            "ddi_minor":           f.ddi_minor,
            "triple_whammy":       int(f.triple_whammy),
            "qt_risk_count":       f.qt_risk_count,
            "dup_same_ingredient": f.dup_same_ingredient,
            "dup_atc5":            f.dup_atc5,
            "dup_atc4":            f.dup_atc4,
            "dup_atc3":            f.dup_atc3,
            "age":                 f.age,
            "sex":                 f.sex,
            "risk_level":          f.risk_level,
            "risk_reasons":        "|".join(f.risk_reasons),
            "yellow_subtype":      f.yellow_subtype,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def write_features(
    features: list[PatientFeatures],
    partition: str,
    base_dir: Path = FEATURE_BASE,
    overwrite: bool = False,
) -> Path:
    """
    PatientFeatures → parquet 저장.

    Parameters
    ----------
    partition : 'YYYYMM' 형식
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / f"patient_features_{partition}.parquet"

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"이미 존재: {out_path}. overwrite=True 로 재실행.")

    df = features_to_df(features)
    df.to_parquet(out_path, index=False)
    logger.info("피처 저장: %s (%d행)", out_path, len(df))
    return out_path


def write_overlap_pairs(
    overlap_df: pd.DataFrame,
    partition: str,
    base_dir: Path = FEATURE_BASE,
    overwrite: bool = False,
) -> Optional[Path]:
    """동시복용 쌍 DataFrame → parquet 저장."""
    if overlap_df.empty:
        return None

    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / f"overlap_pairs_{partition}.parquet"

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"이미 존재: {out_path}")

    overlap_df.to_parquet(out_path, index=False)
    logger.info("동시복용 저장: %s (%d쌍)", out_path, len(overlap_df))
    return out_path


def write_pipeline_log(
    result: PipelineResult,
    base_dir: Path = FEATURE_BASE,
) -> Path:
    """파이프라인 실행 결과 JSON 저장."""
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / f"pipeline_log_{result.partition}.json"

    log_dict = {
        "partition":           result.partition,
        "total_patients":      result.total_patients,
        "total_prescriptions": result.total_prescriptions,
        "total_drug_items":    result.total_drug_items,
        "overlap_pairs":       result.overlap_pairs,
        "features_written":    result.features_written,
        "risk_distribution": {
            "Red":    result.red_count,
            "Yellow": result.yellow_count,
            "Green":  result.green_count,
            "Normal": result.normal_count,
        },
        "elapsed_seconds": result.elapsed_seconds,
        "success":         result.success,
        "errors":          result.errors,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log_dict, f, ensure_ascii=False, indent=2, default=str)
    logger.info("파이프라인 로그: %s", out_path)
    return out_path


def load_features(
    partition: str,
    base_dir: Path = FEATURE_BASE,
) -> pd.DataFrame:
    """저장된 피처 파일 로드."""
    path = base_dir / f"patient_features_{partition}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"피처 파일 없음: {path}")
    return pd.read_parquet(path)


def print_feature_summary(features: list[PatientFeatures]) -> None:
    """피처 요약 출력."""
    if not features:
        print("[피처] 데이터 없음")
        return

    risk_counts = {"Red": 0, "Yellow": 0, "Green": 0, "Normal": 0}
    for f in features:
        risk_counts[f.risk_level] = risk_counts.get(f.risk_level, 0) + 1

    total = len(features)
    print(f"\n{'='*60}")
    print(f"[피처 집계 결과] 총 {total:,}명")
    print(f"{'='*60}")
    for level, cnt in risk_counts.items():
        pct = cnt / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {level:8s}: {cnt:8,}명 ({pct:5.1f}%) {bar}")
    print(f"{'='*60}")

    avg_drugs = sum(f.drug_count for f in features) / total
    avg_ddi = sum(f.ddi_contraindicated + f.ddi_major for f in features) / total
    print(f"  평균 약물 수:     {avg_drugs:.1f}종")
    print(f"  평균 고위험 DDI:  {avg_ddi:.2f}건")
