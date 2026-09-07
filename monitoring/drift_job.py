# -*- coding: utf-8 -*-
"""M7 — PSI 산출·노출 작업. DAG 두 곳이 함께 쓴다.

종전에는 이 로직이 배치 예측 DAG 안에만 있었고, 계산 대상 컬럼을 손으로
나열했다 — `drug_count`·`ddi_count`·`rule_triggered`. 그중 `rule_triggered` 는
예측 parquet 에 **쓰인 적이 없고**, `ddi_count` 는 기준 분포의 피처명이 아니다.
실제로 PSI 가 계산되던 열은 **`drug_count` 하나뿐**이었다.

여기서는 나열하지 않고 **기준 분포와의 교집합**에 맡긴다. 기준 분포는 학습
파이프라인이 학습 피처 전량으로 fit 하므로, 예측 parquet 에 실린 피처가
늘어나는 만큼 커버리지가 자동으로 따라온다.

**함수 둘의 역할이 다르다.**

  run_drift_job      새 예측 파티션을 채점하고 리포트를 쓴다. 배치 DAG 전용.
                     이 리포트가 **불변 기록**이며 덮어쓰지 않는다.
  push_latest_psi    이미 있는 최신 리포트를 읽어 노출만 한다. 일간 DAG 전용.
                     **다시 채점하지 않는다** — 같은 데이터를 다시 재면 새 정보
                     없이 리포트만 덮어쓰고 배치 DAG 와 파일 경합을 만든다.
                     대신 관측 데이터가 **며칠 된 것인지**를 숫자로 낸다.

**프로세스 경계.** 이 코드는 DAG 워커에서 돈다. 여기서 세운 게이지는 이
프로세스의 레지스트리에 있고 태스크와 함께 사라진다. 서빙의 노출 엔드포인트가
내보내는 것은 서빙 프로세스의 레지스트리다. 따라서 **pushgateway 구성이 없으면
PSI 는 JSON 리포트에만 남는다**(`DDI_PUSHGATEWAY_URL`).

push 에는 **보낼 것만 담은 새 레지스트리**를 쓴다. 전역 레지스트리를 밀면 이
프로세스에 정의만 되고 값이 없는 서빙 메트릭과 기본 수집기가 0 인 채로 함께
게시된다.

**남은 천장** — 예측 parquet 은 현재 10개 필드만 싣고 그중 기준 분포와 겹치는
것은 `drug_count` 하나다. 넓히려면 배치 DAG 가 그날의 피처를 결과에 붙여야
하는데, 그것은 운영 DAG 의 산출 스키마를 바꾸는 결정이라 여기서 하지 않는다.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

from monitoring.metrics import push_metrics, record_psi

logger = logging.getLogger(__name__)

PSI_JOB = "ddi_drift"


class PushFailed(RuntimeError):
    """게이트웨이가 구성돼 있는데 전송이 실패했다 — 조용히 넘기지 않는다."""


def select_psi_columns(df, detector) -> list[str]:
    """PSI 를 계산할 컬럼 — 예측 결과와 기준 분포의 교집합.

    기준 분포에 없는 열은 조용히 빠진다. 예측 parquet 에 무엇이 더 실리든
    깨지지 않아야 한다.
    """
    reference = getattr(detector, "_reference", {}) or {}
    return [c for c in df.columns if c in reference]


def _psi_registry(pairs, *, age_days: float | None = None):
    """PSI 만 담은 새 레지스트리. 전역 레지스트리를 밀지 않기 위한 것이다."""
    from prometheus_client import CollectorRegistry, Gauge

    reg = CollectorRegistry()
    g = Gauge("ddi_psi_score", "피처별 PSI (Population Stability Index)",
              ["feature_name"], registry=reg)
    for name, value in pairs:
        g.labels(feature_name=name).set(value)
    if age_days is not None:
        Gauge("ddi_psi_source_partition_age_days",
              "PSI 가 근거한 예측 파티션이 며칠 지난 것인지",
              registry=reg).set(age_days)
    return reg


def _push(pairs, *, age_days=None) -> None:
    gateway = os.environ.get("DDI_PUSHGATEWAY_URL", "").strip()
    if not gateway:
        logger.warning(
            "DDI_PUSHGATEWAY_URL 미설정 — PSI 는 JSON 리포트에만 남는다. "
            "Grafana PSI 패널은 데이터 없음으로 표시된다."
        )
        return
    if not push_metrics(gateway, PSI_JOB, _psi_registry(pairs, age_days=age_days)):
        # 관측하라고 구성해 놓고 나가지 않았다. 태스크가 초록불이면 안 된다.
        raise PushFailed(f"PSI push 실패: {gateway}")


def run_drift_job(predictions_path, reference_path, output_dir, *, partition: str):
    """예측 parquet 을 채점해 리포트를 쓰고 PSI 를 노출한다. **배치 DAG 전용.**

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
        record_psi(feature_name=r.feature_name, psi_value=r.psi)
    _push([(r.feature_name, r.psi) for r in report.feature_results])

    logger.info(
        "PSI 산출 완료 (partition=%s): %d 피처, 드리프트 %d",
        partition, len(report.feature_results), report.n_drifted,
    )
    return report


def _partition_age_days(partition: str, today: date | None = None) -> float | None:
    for fmt in ("%Y%m%d", "%Y%m"):
        try:
            d = datetime.strptime(partition, fmt).date()
        except ValueError:
            continue
        return float(((today or date.today()) - d).days)
    return None


def push_latest_psi(monitoring_dir, *, today: date | None = None) -> Path | None:
    """가장 최근 드리프트 리포트를 노출한다. **다시 채점하지 않는다.**

    일간 DAG 전용. 배치가 돌지 않는 날에도 PSI 계열이 살아 있게 하되, 그 값이
    **며칠 된 데이터인지**를 `ddi_psi_source_partition_age_days` 로 함께 낸다.
    재산출은 새 정보를 만들지 않으면서 리포트를 덮어쓰고 배치 DAG 와 파일
    경합을 만든다 — 하지 않는다.
    """
    d = Path(monitoring_dir)
    reports = sorted(d.glob("drift_*.json")) if d.exists() else []
    if not reports:
        logger.warning("드리프트 리포트 없음 (%s) — 노출할 PSI 가 없다", d)
        return None

    latest = reports[-1]
    data = json.loads(latest.read_text(encoding="utf-8"))
    partition = str(data.get("partition", ""))
    # 리포트 스키마는 `features: [{feature, psi, status}]` 다.
    pairs = [
        (f["feature"], float(f["psi"]))
        for f in data.get("features", [])
        if f.get("feature") is not None and f.get("psi") is not None
    ]
    if not pairs:
        logger.warning("리포트에 PSI 항목이 없다 (%s)", latest)
        return None

    age = _partition_age_days(partition, today)
    if age is None:
        logger.warning("파티션 형식을 읽을 수 없다 (%r) — 경과일 없이 노출", partition)
    for name, value in pairs:
        record_psi(feature_name=name, psi_value=value)
    _push(pairs, age_days=age)

    logger.info("최신 PSI 노출 (%s, 파티션 경과 %s일)", latest.name, age)
    return latest
