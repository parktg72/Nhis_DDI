"""
DDI Drift 일간 재산출 DAG (M7)

PSI 는 배치 예측 DAG 안에서만 계산되고 그 DAG 는 화~토(`0 5 * * 2-6`)에만 돈다.
일·월요일에는 PSI 계열이 끊기고, Grafana 의 드리프트 패널이 이틀씩 비어 있다.

배치 DAG 의 일정은 청구 데이터 도착에 묶인 **운영 결정**이므로 건드리지 않는다.
빈 날에 배치 예측을 돌리게 되기 때문이다. 대신 이 DAG 가 매일 아침 **가장 최근
예측 파티션**을 다시 채점한다.

**다시 채점하지 않는다.** 같은 파티션을 다시 재면 새 정보 없이 배치가 쓴 리포트를
덮어써 최초 산출 시각을 잃고, 05시 배치 DAG 와 파일 경합도 생긴다. 이 DAG 는
**이미 있는 최신 리포트를 읽어 노출**하고, 그 데이터가 **며칠 된 것인지**를
`ddi_psi_source_partition_age_days` 로 함께 낸다.

오래된 데이터를 최신인 양 보여주는 것이 위험의 본체이므로, 그 나이를 숨기지 않고
숫자로 낸다. 노출에는 pushgateway 가 필요하다(`DDI_PUSHGATEWAY_URL` — 배포 런북 부록).

**알림은 울리지 않는다.** 같은 데이터로 알림을 다시 내면 중복이다. 알림은 배치
예측 DAG 에 그대로 둔다.

스케줄: 매일 06:00 (배치 예측 05:00 이후)

환경변수:
  DDI_MONITORING_DIR   : 드리프트 리포트 디렉토리
  DDI_PUSHGATEWAY_URL  : 미설정 시 노출되지 않는다 (경고만 남는다)
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


def _push_psi_freshness(**context) -> None:
    """최신 드리프트 리포트를 노출하고 그것이 며칠 된 데이터인지 함께 낸다.

    **다시 채점하지 않는다.** 같은 파티션을 다시 재면 새 정보 없이 배치가 쓴
    리포트를 덮어써 최초 산출 시각을 잃고, 05시 배치 DAG 와 파일 경합도 생긴다.
    """
    from config import settings as _s
    from monitoring.drift_job import push_latest_psi

    push_latest_psi(_s.MONITORING_DIR)


with DAG(
    dag_id="ddi_drift_daily",
    description="PSI 일간 노출 — 최신 리포트와 그 경과일을 낸다 (재채점·알림 없음)",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 6 * * *",   # 매일 06:00 (배치 예측 05:00 이후)
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ddi", "monitoring", "drift"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    t_psi = PythonOperator(
        task_id="push_psi_freshness",
        python_callable=_push_psi_freshness,
    )

    start >> t_psi >> end
