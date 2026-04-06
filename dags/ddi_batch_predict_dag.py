"""
DDI Batch Prediction DAG

신규 청구 데이터에 대해 배치 위험도 분류를 수행하고 결과를 저장.
Serving API의 /predict/batch 엔드포인트를 활용하여 대규모 배치 처리.

스케줄: 매주 화~토 새벽 5시 (ETL + 피처 완료 후)
실행 시간: 약 10~30분 (환자 수에 따라 상이)

환경변수:
  DDI_FEATURES_DIR    : ML 피처 디렉토리 (기본: /app/data/features)
  DDI_PREDICTIONS_DIR : 예측 결과 저장 디렉토리 (기본: /app/data/predictions)
  DDI_SERVING_URL     : Serving API URL (기본: http://localhost:8000)
  DDI_BATCH_SIZE      : 배치 크기 (기본: 500)
  DDI_MATRIX_PATH     : DDI 매트릭스 경로 (기본: /app/data/processed/ddi_matrix_final.parquet)
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago

# ─────────────────────────────────────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner": "ddi-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

from config.settings import (
    FEATURES_DIR,
    PREDICTIONS_DIR,
    SERVING_URL,
    BATCH_SIZE,
    DDI_MATRIX_PATH,
    PROCESSED_DIR as PROC_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# 태스크 함수
# ─────────────────────────────────────────────────────────────────────────────

def _get_partition(**context) -> str:
    partition = context["execution_date"].strftime("%Y%m%d")
    context["ti"].xcom_push(key="partition", value=partition)
    return partition


def _check_serving_health(**context) -> None:
    """Serving API 헬스체크."""
    import requests
    try:
        resp = requests.get(f"{SERVING_URL}/health", timeout=10)
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") not in ("ok", "degraded"):
            raise ValueError(f"Serving 상태 비정상: {body}")
    except requests.RequestException as exc:
        raise RuntimeError(f"Serving API 연결 실패: {exc}") from exc


def _load_patients(**context) -> None:
    """당일 처방 환자 목록 로드."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import pandas as pd

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")

    # partition 형식 검증 (경로 탐색 공격 방지: YYYYMMDD 8자리 숫자만 허용)
    if not re.fullmatch(r"\d{8}", partition or ""):
        raise ValueError(f"partition 형식이 올바르지 않습니다 (YYYYMMDD 필요): {partition!r}")

    # ETL 결과에서 환자 목록 로드
    t30_path = f"{PROC_DIR}/t30_{partition}_std.parquet"
    t20_path = f"{PROC_DIR}/t20_{partition}_pseudo.parquet"

    if not os.path.exists(t30_path):
        raise FileNotFoundError(f"T30 데이터 없음: {t30_path}")

    t30 = pd.read_parquet(t30_path)
    t20 = pd.read_parquet(t20_path)
    joined = t30.merge(t20[["CMN_KEY", "INDI_DSCM_NO", "MDCARE_STRT_DT"]], on="CMN_KEY", how="left")

    # 환자별 약물 목록 구성
    patient_drugs = {}
    for patient_id, grp in joined.groupby("INDI_DSCM_NO"):
        drugs = []
        for _, row in grp.iterrows():
            drug = {
                "edi_code": str(row.get("MCARE_DIV_CD", "") or ""),
                "total_days": int(row.get("TOT_MCNT", 30) or 30),
            }
            if "atc_code" in row and row["atc_code"]:
                drug["atc_code"] = str(row["atc_code"])
            if "MCARE_DIV_CD_NM" in row and row["MCARE_DIV_CD_NM"]:
                drug["drug_name"] = str(row["MCARE_DIV_CD_NM"])
            elif "drug_name" in row and row["drug_name"]:
                drug["drug_name"] = str(row["drug_name"])
            if "MDCARE_STRT_DT" in row and row["MDCARE_STRT_DT"]:
                drug["start_date"] = str(row["MDCARE_STRT_DT"])[:10]
            drugs.append(drug)
        patient_drugs[str(patient_id)] = drugs

    # 스테이징 저장
    import json
    staging_path = f"{PROC_DIR}/batch_patients_{partition}.json"
    with open(staging_path, "w", encoding="utf-8") as f:
        json.dump(patient_drugs, f, ensure_ascii=False)

    context["ti"].xcom_push(key="staging_path", value=staging_path)
    context["ti"].xcom_push(key="n_patients", value=len(patient_drugs))

    import logging
    logging.info("배치 대상 환자: %d명", len(patient_drugs))


def _run_batch_predict(**context) -> None:
    """Serving API /predict/batch로 위험도 분류 수행."""
    import json
    import os
    import requests
    import pandas as pd

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")
    staging_path = context["ti"].xcom_pull(key="staging_path", task_ids="load_patients")

    with open(staging_path, "r", encoding="utf-8") as f:
        patient_drugs = json.load(f)

    patient_ids = list(patient_drugs.keys())
    all_results = []

    # BATCH_SIZE 단위로 API 호출
    for i in range(0, len(patient_ids), BATCH_SIZE):
        chunk = patient_ids[i : i + BATCH_SIZE]
        payload = {
            "requests": [
                {
                    "patient_id": pid,
                    "drugs": patient_drugs[pid],
                }
                for pid in chunk
                if patient_drugs[pid]  # 빈 약물 목록 제외
            ]
        }
        if not payload["requests"]:
            continue

        resp = requests.post(
            f"{SERVING_URL}/predict/batch",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()

        for pred in body.get("results", []):
            all_results.append({
                "patient_id": pred["patient_id"],
                "risk_level": pred["risk_level"],
                "rule_level": pred.get("rule_level"),
                "ml_level": pred.get("ml_level"),
                "ml_probability": pred.get("ml_probability"),
                "drug_count": pred.get("drug_count", 0),
                "ddi_count": len(pred.get("ddi_alerts", [])),
                "intervention": pred.get("intervention"),
                "reference_date": pred.get("reference_date"),
                "partition": partition,
            })

    if not all_results:
        import logging
        logging.warning("예측 결과 없음")
        return

    # 결과 저장
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    out_path = f"{PREDICTIONS_DIR}/predictions_{partition}.parquet"
    pd.DataFrame(all_results).to_parquet(out_path, index=False)

    context["ti"].xcom_push(key="predictions_path", value=out_path)
    context["ti"].xcom_push(key="n_predicted", value=len(all_results))

    import logging
    logging.info("예측 완료: %d명 → %s", len(all_results), out_path)


def _generate_summary(**context) -> None:
    """위험도 분포 통계 요약 리포트 생성."""
    import os
    import pandas as pd

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")
    predictions_path = context["ti"].xcom_pull(
        key="predictions_path", task_ids="run_batch_predict"
    )

    if not predictions_path or not os.path.exists(predictions_path):
        import logging
        logging.warning("예측 결과 파일 없음, 요약 생략")
        return

    df = pd.read_parquet(predictions_path)
    dist = df["risk_level"].value_counts().to_dict()

    summary = {
        "partition": partition,
        "total": len(df),
        "red_count":    int(dist.get("Red",    0)),
        "yellow_count": int(dist.get("Yellow", 0)),
        "green_count":  int(dist.get("Green",  0)),
        "normal_count": int(dist.get("Normal", 0)),
        "red_rate":    float(dist.get("Red",    0)) / max(len(df), 1),
        "yellow_rate": float(dist.get("Yellow", 0)) / max(len(df), 1),
    }

    import json
    summary_path = f"{PREDICTIONS_DIR}/summary_{partition}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    import logging
    logging.info(
        "요약 리포트: 총 %d명 — Red=%d(%.1f%%), Yellow=%d(%.1f%%), Green=%d, Normal=%d",
        summary["total"],
        summary["red_count"],    summary["red_rate"]    * 100,
        summary["yellow_count"], summary["yellow_rate"] * 100,
        summary["green_count"],
        summary["normal_count"],
    )


def _cleanup_staging(**context) -> None:
    """스테이징 파일 삭제."""
    import os
    staging_path = context["ti"].xcom_pull(
        key="staging_path", task_ids="load_patients"
    )
    if staging_path and os.path.exists(staging_path):
        os.remove(staging_path)


def _detect_drift(partition: str) -> None:
    """배치 예측 parquet에서 PSI 드리프트를 감지하고 JSON 리포트를 저장한다.

    settings 접근을 `from config import settings as _s` 패턴으로 처리해
    테스트에서 monkeypatch.setattr이 정상 작동한다.
    """
    import pandas as pd
    from monitoring.drift_detector import DriftDetector
    from config import settings as _s

    drift_ref = _s.DRIFT_REFERENCE_PATH
    predictions_dir = _s.PREDICTIONS_DIR
    monitoring_dir = _s.MONITORING_DIR

    if not drift_ref.exists():
        logger.warning(
            "drift_reference.pkl 없음 (%s) — 드리프트 감지 건너뜀 (학습 파이프라인을 먼저 실행하세요)",
            drift_ref,
        )
        return

    pred_path = predictions_dir / f"predictions_{partition}.parquet"
    if not pred_path.exists():
        logger.warning("예측 파일 없음 (%s) — 드리프트 감지 건너뜀", pred_path)
        return

    df = pd.read_parquet(pred_path)
    available_cols = [c for c in ("drug_count", "ddi_count", "rule_triggered") if c in df.columns]
    if not available_cols:
        logger.warning(
            "PSI 계산 가능한 컬럼 없음 (partition=%s, 컬럼=%s) — 드리프트 감지 건너뜀",
            partition, list(df.columns),
        )
        return

    detector = DriftDetector.load(str(drift_ref))
    report = detector.detect(df[available_cols], partition=partition)

    monitoring_dir.mkdir(parents=True, exist_ok=True)
    detector.save_report(report, str(monitoring_dir))
    logger.info(
        "드리프트 감지 완료 (partition=%s): %d 피처 분석, %d 드리프트",
        partition, len(report.feature_results), report.n_drifted,
    )


def _generate_alerts(partition: str) -> None:
    """드리프트 리포트와 Rule/ML 불일치율을 기반으로 알림을 생성하고 JSON으로 저장한다."""
    import json
    from types import SimpleNamespace
    from monitoring.alert_rules import AlertManager, Alert
    from monitoring.metrics_writer import MetricsWriter
    from config import settings as _s

    monitoring_dir = _s.MONITORING_DIR
    metrics_jsonl_path = _s.METRICS_JSONL_PATH

    mgr = AlertManager()
    alerts: list[Alert] = []

    # ── 드리프트 알림 ───────────────────────────────────────────────────────
    report_path = monitoring_dir / f"drift_{partition}.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            drift_data = json.load(f)
        feat_results = [
            SimpleNamespace(
                feature_name=feat["feature"],
                psi=feat["psi"],
                status=feat["status"],
                is_drifted=feat["status"] == "drift",
            )
            for feat in drift_data.get("features", [])
        ]
        drift_obj = SimpleNamespace(
            partition=drift_data.get("partition", partition),
            feature_results=feat_results,
            summary=drift_data.get("summary", {}),
        )
        alerts += mgr.evaluate_drift(drift_obj)
    else:
        logger.warning("드리프트 리포트 없음 (%s) — 드리프트 알림 건너뜀", report_path)

    # ── Rule/ML 불일치율 알림 ───────────────────────────────────────────────
    writer = MetricsWriter(path=metrics_jsonl_path)
    records = [r for r in writer.read_recent(hours=24) if r.get("partition") == partition]
    disagree_rate = 0.0
    if records:
        disagree_rate = sum(1 for r in records if r.get("disagree")) / len(records)
        logger.info(
            "Rule/ML 불일치율 (partition=%s): %.1f%% (%d / %d 건)",
            partition, disagree_rate * 100, sum(1 for r in records if r.get("disagree")), len(records),
        )
    alerts += mgr.evaluate_rule_ml_disagree(disagree_rate, partition)

    # ── 알림 저장 ───────────────────────────────────────────────────────────
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    alert_path = monitoring_dir / f"alerts_{partition}.json"
    with open(alert_path, "w", encoding="utf-8") as f:
        json.dump([a.to_dict() for a in alerts], f, ensure_ascii=False, indent=2)
    logger.info("알림 생성 완료 (partition=%s): %d건", partition, len(alerts))


# ─────────────────────────────────────────────────────────────────────────────
# DAG 정의
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="ddi_batch_predict",
    description="DDI 배치 위험도 분류 (Serving API /predict/batch)",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 5 * * 2-6",  # 화~토 05:00 (ETL+피처 완료 후)
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ddi", "predict", "batch"],
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end")

    # Feature Engineering DAG 완료 대기
    wait_features = ExternalTaskSensor(
        task_id="wait_for_features",
        external_dag_id="ddi_feature_engineering",
        external_task_id="end",
        timeout=3600,
        poke_interval=60,
        mode="reschedule",
    )

    t_partition = PythonOperator(
        task_id="get_partition",
        python_callable=_get_partition,
    )
    t_health = PythonOperator(
        task_id="check_serving_health",
        python_callable=_check_serving_health,
    )
    t_load = PythonOperator(
        task_id="load_patients",
        python_callable=_load_patients,
    )
    t_predict = PythonOperator(
        task_id="run_batch_predict",
        python_callable=_run_batch_predict,
        execution_timeout=timedelta(hours=2),
    )
    t_summary = PythonOperator(
        task_id="generate_summary",
        python_callable=_generate_summary,
    )
    t_detect_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=_detect_drift,
        op_kwargs={"partition": "{{ ti.xcom_pull(key='partition', task_ids='get_partition') }}"},
    )
    t_generate_alerts = PythonOperator(
        task_id="generate_alerts",
        python_callable=_generate_alerts,
        op_kwargs={"partition": "{{ ti.xcom_pull(key='partition', task_ids='get_partition') }}"},
    )
    t_cleanup = PythonOperator(
        task_id="cleanup_staging",
        python_callable=_cleanup_staging,
        trigger_rule="all_done",  # 성공/실패 무관하게 정리
    )

    (
        start
        >> wait_features
        >> t_partition
        >> t_health
        >> t_load
        >> t_predict
        >> t_summary
        >> t_detect_drift
        >> t_generate_alerts
        >> t_cleanup
        >> end
    )
