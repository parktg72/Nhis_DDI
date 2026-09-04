"""
DDI Drift 일간 재산출 DAG (M7)

PSI 는 배치 예측 DAG 안에서만 계산되고 그 DAG 는 화~토(`0 5 * * 2-6`)에만 돈다.
일·월요일에는 PSI 계열이 끊기고, Grafana 의 드리프트 패널이 이틀씩 비어 있다.

배치 DAG 의 일정은 청구 데이터 도착에 묶인 **운영 결정**이므로 건드리지 않는다.
빈 날에 배치 예측을 돌리게 되기 때문이다. 대신 이 DAG 가 매일 아침 **가장 최근
예측 파티션**을 다시 채점한다.

**이 DAG 가 실제로 하는 일을 정확히 적는다.** 일·월요일에는 토요일 파티션을 다시
채점하고 같은 리포트를 덮어쓴다. **새로운 정보가 생기지는 않는다.** pushgateway 가
구성돼 있으면 push 시각이 갱신되어 "메트릭이 오래됐다" 류의 경보가 오탐하지 않고,
구성돼 있지 않으면 관측 가능한 변화는 없다(`DDI_PUSHGATEWAY_URL` — 배포 런북 부록).

**알림은 울리지 않는다.** 같은 데이터로 알림을 다시 내면 중복이다. 알림은 배치
예측 DAG 에 그대로 둔다.

스케줄: 매일 06:00 (배치 예측 05:00 이후)

환경변수:
  DDI_PREDICTIONS_DIR : 배치 예측 결과 디렉토리
  DDI_MONITORING_DIR  : 드리프트 리포트 저장 디렉토리
  DDI_DRIFT_REFERENCE : 기준 분포 pkl 경로
"""
from __future__ import annotations

import logging
from datetime import timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "ddi-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _latest_partition(**context) -> None:
    """가장 최근 `predictions_*.parquet` 의 파티션을 고른다.

    없으면 예외를 내지 않고 빈 값을 넘긴다 — 배포 초기에는 예측 결과가 없다.
    """
    from config import settings as _s

    files = sorted((_s.PREDICTIONS_DIR).glob("predictions_*.parquet"))
    partition = files[-1].stem.replace("predictions_", "") if files else ""
    if not partition:
        logger.warning("예측 결과 파일 없음 (%s) — PSI 재산출 건너뜀", _s.PREDICTIONS_DIR)
    context["ti"].xcom_push(key="partition", value=partition)


def _recompute_psi(partition: str) -> None:
    """가장 최근 파티션으로 PSI 를 다시 산출한다. 알림은 내지 않는다."""
    if not partition:
        return

    from config import settings as _s
    from monitoring.drift_job import run_drift_job

    run_drift_job(
        _s.PREDICTIONS_DIR / f"predictions_{partition}.parquet",
        _s.DRIFT_REFERENCE_PATH,
        _s.MONITORING_DIR,
        partition=partition,
    )


with DAG(
    dag_id="ddi_drift_daily",
    description="PSI 일간 재산출 — 배치가 돌지 않는 날에도 계열을 유지한다 (알림 없음)",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 6 * * *",   # 매일 06:00 (배치 예측 05:00 이후)
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ddi", "monitoring", "drift"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    t_partition = PythonOperator(
        task_id="latest_partition",
        python_callable=_latest_partition,
    )
    t_psi = PythonOperator(
        task_id="recompute_psi",
        python_callable=_recompute_psi,
        op_kwargs={
            "partition": "{{ ti.xcom_pull(key='partition', task_ids='latest_partition') }}"
        },
    )

    start >> t_partition >> t_psi >> end
