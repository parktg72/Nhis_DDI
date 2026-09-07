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

  run_drift_job      새 예측 파티션을 채점하고 리포트를 쓴다. **push 하지 않는다.**
                     같은 파티션을 다시 돌리면 그 파티션 리포트를 갱신한다 —
                     쓰기는 원자적이지만 **불변 기록은 아니다.**
  push_psi_report    지정한 리포트를 노출한다. 배치 DAG 는 **방금 쓴 그 리포트**를
                     넘긴다 — "최신" 을 고르면 채점이 무산출일 때 이전 파티션을
                     재게시하고 성공해 버린다.
  push_latest_psi    최신 리포트를 골라 노출한다. 일간 DAG 전용.
                     다시 채점하지 않는다 — 같은 데이터를 다시 재면 새 정보 없이
                     리포트만 덮어쓰고 파일 경합을 만든다.

**push 를 한 함수로 모은 이유가 둘이다.** ① 채점과 노출을 한 태스크에 두면 노출
실패가 채점 태스크를 죽이고, 직렬 DAG 에서 그 뒤의 **알림 생성까지 막는다.**
관측 실패가 알림을 막는 것은 우선순위가 뒤집힌 것이다. ② pushgateway 는 같은
job 그룹을 **교체**하므로, 서로 다른 metric family 를 싣는 push 가 둘이면 나중
것이 앞의 계열을 지운다. 한 곳에서 항상 같은 family 를 싣는다.

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
import re
from datetime import datetime, timezone
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


def _psi_registry(pairs, source_ts: float):
    """PSI 만 담은 새 레지스트리. 전역 레지스트리를 밀지 않기 위한 것이다.

    **metric family 는 항상 같다.** pushgateway 는 같은 job 그룹을 교체하므로,
    어떤 push 는 A 만 싣고 어떤 push 는 A·B 를 실으면 앞의 B 계열이 지워진다.
    """
    from prometheus_client import CollectorRegistry, Gauge

    reg = CollectorRegistry()
    g = Gauge("ddi_psi_score", "피처별 PSI (Population Stability Index)",
              ["feature_name"], registry=reg)
    for name, value in pairs:
        g.labels(feature_name=name).set(value)
    # 경과일이 아니라 **파티션 시각**을 보낸다. 경과일은 이 DAG 가 멈추면 함께
    # 얼어붙어 오래됐다는 사실 자체가 오래된 값이 된다. 시각을 보내면 PromQL 의
    # time() 으로 항상 현재 기준 경과를 계산할 수 있다.
    Gauge("ddi_psi_source_partition_timestamp_seconds",
          "PSI 가 근거한 예측 파티션의 시작 시각 (Unix). 월 파티션이면 그 달 1일",
          registry=reg).set(source_ts)
    return reg


def _push(pairs, source_ts: float) -> None:
    gateway = os.environ.get("DDI_PUSHGATEWAY_URL", "").strip()
    if not gateway:
        logger.warning(
            "DDI_PUSHGATEWAY_URL 미설정 — PSI 는 JSON 리포트에만 남는다. "
            "Grafana PSI 패널은 데이터 없음으로 표시된다."
        )
        return
    if not push_metrics(gateway, PSI_JOB, _psi_registry(pairs, source_ts)):
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
    written = detector.save_report(report, str(out))

    for r in report.feature_results:
        record_psi(feature_name=r.feature_name, psi_value=r.psi)

    logger.info(
        "PSI 산출 완료 (partition=%s): %d 피처, 드리프트 %d",
        partition, len(report.feature_results), report.n_drifted,
    )
    logger.debug("리포트 기록: %s", written)
    return report


_REPORT_RE = re.compile(r"^drift_(\d{8}|\d{6})\.json$")


def _partition_start(partition: str) -> datetime | None:
    """파티션 문자열 → 시작 시각. 형식이 정확히 맞을 때만 돌려준다.

    `strptime` 은 폭에 느슨해서 `202613` 같은 값도 다른 형식으로 통과한다.
    왕복 검증(`strftime` 결과가 원문과 같은지)으로 그것을 막는다.
    """
    for fmt in ("%Y%m%d", "%Y%m"):
        try:
            d = datetime.strptime(partition, fmt)
        except ValueError:
            continue
        if d.strftime(fmt) == partition:
            return d.replace(tzinfo=timezone.utc)
    return None


def _select_latest(monitoring_dir: Path):
    """가장 최근 리포트 — **파일명 순이 아니라 파싱한 날짜 순**으로 고른다.

    `sorted(glob)[-1]` 은 `drift_summary.json` 같은 파일에 지고, 형식이 아닌
    이름도 이긴다. 이름을 엄격히 검증하고 날짜로 비교한다.
    """
    best = None
    for p in monitoring_dir.glob("drift_*.json"):
        m = _REPORT_RE.match(p.name)
        if not m:
            logger.debug("리포트 이름 형식 아님 — 건너뜀: %s", p.name)
            continue
        started = _partition_start(m.group(1))
        if started is None:
            logger.warning("파티션 날짜로 읽을 수 없음 — 건너뜀: %s", p.name)
            continue
        if best is None or started > best[0]:
            best = (started, p, m.group(1))
    return best


def push_psi_report(report_path) -> Path | None:
    """**지정한** 리포트를 노출한다. push 하는 실체는 이 함수 하나다."""
    latest = Path(report_path) if report_path else None
    if latest is None or not latest.exists():
        logger.warning("노출할 리포트가 없다 (%s)", report_path)
        return None

    m = _REPORT_RE.match(latest.name)
    started = _partition_start(m.group(1)) if m else None

    data = json.loads(latest.read_text(encoding="utf-8"))
    inner = str(data.get("partition", ""))
    if inner:
        inner_started = _partition_start(inner)
        if started is not None and inner_started is not None and inner_started != started:
            logger.warning("파일명과 리포트 파티션 불일치: %s vs %s", latest.name, inner)
        if inner_started is not None:
            started = inner_started
    if started is None:
        logger.warning("파티션 날짜를 읽을 수 없다 (%s) — 노출하지 않는다", latest.name)
        return None

    # 리포트 스키마는 `features: [{feature, psi, status}]` 다.
    pairs = [
        (f["feature"], float(f["psi"]))
        for f in data.get("features", [])
        if f.get("feature") is not None and f.get("psi") is not None
    ]
    if not pairs:
        logger.warning("리포트에 PSI 항목이 없다 (%s)", latest)
        return None

    for name, value in pairs:
        record_psi(feature_name=name, psi_value=value)
    _push(pairs, started.timestamp())

    logger.info("PSI 노출 (%s, 파티션 시작 %s)", latest.name, started.date())
    return latest


def push_latest_psi(monitoring_dir) -> Path | None:
    """가장 최근 리포트를 골라 노출한다. **다시 채점하지 않는다.** 일간 DAG 전용."""
    d = Path(monitoring_dir)
    if not d.exists():
        logger.warning("모니터링 디렉터리 없음 (%s) — 노출할 PSI 가 없다", d)
        return None

    picked = _select_latest(d)
    if picked is None:
        logger.warning("드리프트 리포트 없음 (%s) — 노출할 PSI 가 없다", d)
        return None
    return push_psi_report(picked[1])
