# -*- coding: utf-8 -*-
"""M7 — PSI 산출 작업. DAG 두 곳이 함께 쓴다.

종전에는 이 로직이 배치 예측 DAG 안에만 있었고, 계산 대상 컬럼을 손으로
나열했다 — `drug_count`·`ddi_count`·`rule_triggered`. 그중 `rule_triggered` 는
예측 parquet 에 **쓰인 적이 없고**, `ddi_count` 는 기준 분포의 피처명이 아니다.
실제로 PSI 가 계산되던 열은 **`drug_count` 하나뿐**이었다.

여기서는 나열하지 않고 **기준 분포와의 교집합**에 맡긴다. 기준 분포는 학습
파이프라인이 학습 피처 전량으로 fit 하므로, 예측 parquet 에 실린 피처가
늘어나는 만큼 커버리지가 자동으로 따라온다.

**남은 천장 둘.**

① 예측 parquet 은 현재 10개 필드만 싣고 그중 기준 분포와 겹치는 것은
`drug_count` 하나다. 실효 커버리지를 넓히려면 배치 DAG 가 그날의 피처를 결과에
붙여야 하는데, 그것은 운영 DAG 의 산출 스키마를 바꾸는 결정이라 여기서 하지
않는다(M7 잔여).

② **이 코드는 DAG 워커에서 돈다.** 여기서 세운 게이지는 이 프로세스의
레지스트리에 있고 태스크와 함께 사라진다. 서빙의 노출 엔드포인트가 내보내는
것은 서빙 프로세스의 레지스트리다. 따라서 **pushgateway 구성이 없으면 PSI 는
JSON 리포트에만 남는다.** `DDI_PUSHGATEWAY_URL` 이 그 전제이며, 없으면 경고를
남긴다 — 조용히 없어지는 것보다 낫다.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from monitoring.metrics import push_metrics, record_psi

logger = logging.getLogger(__name__)


def select_psi_columns(df, detector) -> list[str]:
    """PSI 를 계산할 컬럼 — 예측 결과와 기준 분포의 교집합.

    기준 분포에 없는 열은 조용히 빠진다. 예측 parquet 에 무엇이 더 실리든
    깨지지 않아야 한다.
    """
    reference = getattr(detector, "_reference", {}) or {}
    return [c for c in df.columns if c in reference]


def run_drift_job(
    predictions_path,
    reference_path,
    output_dir,
    *,
    partition: str,
):
    """예측 parquet 에서 PSI 를 산출해 리포트를 쓰고 Prometheus 에 올린다.

    입력이 없으면 조용히 건너뛰고 `None` 을 돌려준다 — 배포 초기에는 기준
    분포도 예측 결과도 없을 수 있고, 그때 DAG 를 죽일 이유가 없다.
    """
    import pandas as pd

    from monitoring.drift_detector import DriftDetector

    reference_path = Path(reference_path)
    predictions_path = Path(predictions_path)

    if not reference_path.exists():
        logger.warning(
            "기준 분포 없음 (%s) — PSI 건너뜀. 학습 파이프라인을 먼저 실행할 것",
            reference_path,
        )
        return None
    if not predictions_path.exists():
        logger.warning("예측 결과 없음 (%s) — PSI 건너뜀", predictions_path)
        return None

    df = pd.read_parquet(predictions_path)
    detector = DriftDetector.load(str(reference_path))
    cols = select_psi_columns(df, detector)
    if not cols:
        logger.warning(
            "기준 분포와 겹치는 컬럼 없음 (partition=%s) — PSI 건너뜀. 예측 결과 컬럼=%s",
            partition, list(df.columns),
        )
        return None

    report = detector.detect(df[cols], partition=partition)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    detector.save_report(report, str(out))

    for r in report.feature_results:
        try:
            record_psi(feature_name=r.feature_name, psi_value=r.psi)
        except Exception:
            logger.warning("PSI 메트릭 기록 실패 (%s)", r.feature_name, exc_info=True)

    # 여기는 **DAG 워커 프로세스**다. 위에서 세운 값은 이 프로세스의 레지스트리에
    # 있고 태스크가 끝나면 함께 사라진다. 서빙의 `/metrics/prometheus` 가 내보내는
    # 것은 서빙 프로세스의 레지스트리이므로, push 하지 않으면 PSI 는 JSON 리포트에만
    # 남고 대시보드에는 끝까지 나타나지 않는다.
    gateway = os.environ.get("DDI_PUSHGATEWAY_URL", "").strip()
    if gateway:
        push_metrics(gateway, job="ddi_drift")
    else:
        logger.warning(
            "DDI_PUSHGATEWAY_URL 미설정 — PSI 는 JSON 리포트에만 남는다. "
            "Grafana PSI 패널은 데이터 없음으로 표시된다."
        )

    logger.info(
        "PSI 산출 완료 (partition=%s): %d 피처, 드리프트 %d",
        partition, len(report.feature_results), report.n_drifted,
    )
    return report
