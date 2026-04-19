#!/usr/bin/env python3
"""
민감도 분석 예시: 약물 집계 기간별 코호트 크기 비교 (60일, 90일, 180일)

실행: python3 sensitivity_analysis_example.py
"""

from data_manager import DataManager
from cohort_builder import CohortBuilder
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_sensitivity_analysis(db_path: str):
    """
    민감도 분석 실행.

    약물 집계 기간을 60일, 90일, 180일로 변경하며 약물 분류 결과를 비교.
    각 기간별로 T2DM_INSULIN, T2DM_OHA, T2DM_NOMED 코호트 크기 출력.
    """
    dm = DataManager(db_path)
    cb = CohortBuilder(dm)

    logger.info("Step 1-3: 기저 코호트, DM 청구, DM 약물 추출")
    cb.step1_base_population()
    cb.step2_dm_claims()
    cb.step3_dm_medications()

    logger.info("\nStep 4: 기본값(90일) 약물 분류")
    result_90 = cb.step4_classify_groups(lookback_days=90)
    logger.info(f"90일 윈도우 결과:\n{result_90}")

    logger.info("\n민감도 분석 실행: 60일, 90일, 180일 비교")
    # 기본 분석이 끝난 후 sensitivity_analysis 호출
    sensitivity_results = cb.sensitivity_analysis(lookback_days_list=[60, 90, 180])

    logger.info("\n민감도 분석 결과 요약:")
    logger.info("=" * 80)

    # 결과 포맷팅
    print("\n약물 분류 결과 비교 (민감도 분석)")
    print("-" * 80)
    print(f"{'그룹':<20} {'60일':<15} {'90일':<15} {'180일':<15}")
    print("-" * 80)

    groups = {'T2DM_INSULIN', 'T2DM_OHA', 'T2DM_NOMED', 'T1DM', 'NON_DM'}
    for group in sorted(groups):
        counts = []
        for days in [60, 90, 180]:
            count = sensitivity_results[f'{days}days'].get(group, 0) if sensitivity_results[f'{days}days'] else 0
            counts.append(f"{count:>12,}")
        print(f"{group:<20} {counts[0]:<15} {counts[1]:<15} {counts[2]:<15}")

    print("-" * 80)

    # 변화율 계산 (90일 기준)
    logger.info("\n기간별 변화율 (90일 기준 = 100%):")
    print("\n기간별 변화율 (90일 기준 = 100%)")
    print("-" * 80)

    for group in sorted(groups):
        count_90 = sensitivity_results['90days'].get(group, 0) if sensitivity_results['90days'] else 1
        if count_90 > 0:
            count_60 = sensitivity_results['60days'].get(group, 0) if sensitivity_results['60days'] else 0
            count_180 = sensitivity_results['180days'].get(group, 0) if sensitivity_results['180days'] else 0

            pct_60 = (count_60 / count_90) * 100
            pct_180 = (count_180 / count_90) * 100

            print(f"{group:<20} 60일: {pct_60:>6.1f}%  180일: {pct_180:>6.1f}%")

    print("-" * 80)

    logger.info("\n해석:")
    logger.info("- T2DM_INSULIN/OHA가 기간에 따라 크게 변한다면: 초기 3개월 약물 치료 포착이 중요")
    logger.info("- 변화가 적다면: 약물 분류가 기간에 무관하게 안정적")
    logger.info("- 권고: 180일(6개월)에 비해 90일(3개월) 권장 (한국 가이드라인 기준)")


if __name__ == '__main__':
    import sys

    # 사용: python3 sensitivity_analysis_example.py [db_path]
    db_path = sys.argv[1] if len(sys.argv) > 1 else ':memory:'

    logger.info(f"데이터베이스: {db_path}")
    run_sensitivity_analysis(db_path)
