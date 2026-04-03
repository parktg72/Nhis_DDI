"""
DDI Model Training DAG

주간 모델 재훈련: ML 피처 → XGBoost/LightGBM 훈련 → 검증 → 모델 배포.

스케줄: 매주 월요일 새벽 4시
실행 시간: 약 30~90분 (Optuna 튜닝 포함)

환경변수:
  DDI_FEATURES_DIR   : ML 피처 디렉토리 (기본: /app/data/features)
  MODEL_DIR          : 모델 저장 디렉토리 (기본: /app/models)
  DDI_TRAIN_WEEKS    : 훈련에 사용할 주 수 (기본: 4)
  DDI_MODEL_TYPE     : xgboost | lightgbm | ensemble (기본: ensemble)
  DDI_OPTUNA_TRIALS  : Optuna 시도 횟수 (기본: 50)
  DDI_RECALL_THRESHOLD: 최소 Recall 기준 (기본: 0.90)
  DDI_AUC_THRESHOLD  : 최소 AUC 기준 (기본: 0.85)
  DDI_SERVING_URL    : 서빙 API URL (기본: http://localhost:8000)
  ADMIN_API_KEY      : /admin/reload 인증 키 (serving ADMIN_API_KEY와 동일)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
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
    "retries": 0,  # 훈련 실패 시 자동 재시도 안 함 (원인 파악 우선)
}

FEATURES_DIR     = os.environ.get("DDI_FEATURES_DIR", "/app/data/features")
MODEL_DIR        = os.environ.get("MODEL_DIR", "/app/models")
TRAIN_WEEKS      = int(os.environ.get("DDI_TRAIN_WEEKS", "4"))
MODEL_TYPE       = os.environ.get("DDI_MODEL_TYPE", "ensemble")
OPTUNA_TRIALS    = int(os.environ.get("DDI_OPTUNA_TRIALS", "50"))
RECALL_THRESHOLD = float(os.environ.get("DDI_RECALL_THRESHOLD", "0.90"))
AUC_THRESHOLD    = float(os.environ.get("DDI_AUC_THRESHOLD", "0.85"))


# ─────────────────────────────────────────────────────────────────────────────
# 태스크 함수
# ─────────────────────────────────────────────────────────────────────────────

def _load_features(**context) -> None:
    """최근 N주 ML 피처 파일 합산 로드."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import pandas as pd
    from datetime import timedelta

    execution_date = context["execution_date"]
    partitions = [
        (execution_date - timedelta(weeks=i)).strftime("%Y%m%d")
        for i in range(TRAIN_WEEKS)
    ]

    dfs = []
    for p in partitions:
        path = f"{FEATURES_DIR}/ml_features_{p}.parquet"
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path))

    if not dfs:
        raise FileNotFoundError(f"훈련 데이터 없음: {FEATURES_DIR}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["patient_id"])

    staging_path = f"{FEATURES_DIR}/ml_features_staging.parquet"
    combined.to_parquet(staging_path, index=False)
    context["ti"].xcom_push(key="n_samples", value=len(combined))
    context["ti"].xcom_push(key="staging_path", value=staging_path)

    import logging
    logging.info("훈련 데이터 로드: %d행 (%d 파티션)", len(combined), len(dfs))


def _run_training(**context) -> None:
    """Optuna 하이퍼파라미터 튜닝 + 모델 훈련."""
    import sys
    sys.path.insert(0, "/app")
    from scripts.train.pipeline import run_training

    result = run_training(
        partition="staging",
        model_type=MODEL_TYPE,
        feature_base=FEATURES_DIR,
        model_dir=MODEL_DIR,
        use_optuna=OPTUNA_TRIALS > 0,
        recall_threshold=RECALL_THRESHOLD,
        auc_threshold=AUC_THRESHOLD,
        optuna_trials=OPTUNA_TRIALS,
    )

    context["ti"].xcom_push(key="val_recall", value=result.val_recall)
    context["ti"].xcom_push(key="val_auc", value=result.val_auc)
    context["ti"].xcom_push(key="model_path", value=result.model_path)

    import logging
    logging.info(
        "훈련 완료 — val_recall=%.4f, val_auc=%.4f, model=%s",
        result.val_recall, result.val_auc, result.model_path,
    )


def _validate_model(**context) -> str:
    """성능 기준 충족 여부 판단 (브랜치 태스크)."""
    val_recall = context["ti"].xcom_pull(key="val_recall", task_ids="run_training")
    val_auc    = context["ti"].xcom_pull(key="val_auc",    task_ids="run_training")

    import logging
    logging.info("검증 — recall=%.4f (기준≥%.2f), auc=%.4f (기준≥%.2f)",
                 val_recall, RECALL_THRESHOLD, val_auc, AUC_THRESHOLD)

    if val_recall >= RECALL_THRESHOLD and val_auc >= AUC_THRESHOLD:
        return "deploy_model"
    else:
        return "validation_failed"


def _deploy_model(**context) -> None:
    """검증 통과 모델을 production 경로로 복사 + serving 핫스왑."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import shutil
    import requests

    model_path = context["ti"].xcom_pull(key="model_path", task_ids="run_training")
    prod_path  = os.path.join(MODEL_DIR, "model_prod.pkl")

    # 이전 모델 백업
    if os.path.exists(prod_path):
        backup = prod_path.replace("model_prod", "model_backup")
        shutil.copy2(prod_path, backup)

    shutil.copy2(model_path, prod_path)

    # .sha256 사이드카도 함께 복사
    sha_src = model_path + ".sha256"
    sha_dst = prod_path + ".sha256"
    if os.path.exists(sha_src):
        shutil.copy2(sha_src, sha_dst)
    else:
        import logging as _log
        _log.warning(".sha256 사이드카 없음, 서빙 로드 실패 가능: %s", sha_src)

    # 앙상블 서브모델(.xgb.pkl, .lgb.pkl) 및 해시 복사
    prod_base = prod_path[:-len(".pkl")]
    for ext in (".xgb.pkl", ".lgb.pkl"):
        sub_src = model_path[:-len(".pkl")] + ext
        sub_dst = prod_base + ext
        if os.path.exists(sub_src):
            shutil.copy2(sub_src, sub_dst)
            sub_sha_src = sub_src + ".sha256"
            sub_sha_dst = sub_dst + ".sha256"
            if os.path.exists(sub_sha_src):
                shutil.copy2(sub_sha_src, sub_sha_dst)

    # serving API 핫스왑 (가능한 경우)
    serving_url = os.environ.get("DDI_SERVING_URL", "http://localhost:8000")
    admin_key = os.environ.get("ADMIN_API_KEY", "")
    try:
        resp = requests.post(
            f"{serving_url}/admin/reload",
            json={"model_path": prod_path},
            headers={"X-Admin-Key": admin_key},
            timeout=30,
        )
        resp.raise_for_status()
        import logging
        logging.info("Serving 핫스왑 완료: %s", resp.json())
    except Exception as exc:
        import logging
        logging.warning("Serving 핫스왑 실패 (무시): %s", exc)


def _validation_failed(**context) -> None:
    """검증 실패 알림 (로그 기록)."""
    val_recall = context["ti"].xcom_pull(key="val_recall", task_ids="run_training")
    val_auc    = context["ti"].xcom_pull(key="val_auc",    task_ids="run_training")

    import logging
    logging.error(
        "모델 검증 실패 — recall=%.4f (기준≥%.2f), auc=%.4f (기준≥%.2f). "
        "모델이 배포되지 않았습니다.",
        val_recall, RECALL_THRESHOLD, val_auc, AUC_THRESHOLD,
    )
    # 실제 운영 시 Slack/이메일 알림 연동 가능


# ─────────────────────────────────────────────────────────────────────────────
# DAG 정의
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="ddi_model_training",
    description="DDI ML 모델 주간 재훈련 및 배포 (XGBoost/LightGBM Ensemble)",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 4 * * 1",  # 매주 월요일 04:00
    start_date=days_ago(7),
    catchup=False,
    max_active_runs=1,
    tags=["ddi", "train"],
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    t_load = PythonOperator(
        task_id="load_features",
        python_callable=_load_features,
    )
    t_train = PythonOperator(
        task_id="run_training",
        python_callable=_run_training,
        execution_timeout=timedelta(hours=3),
    )
    t_validate = BranchPythonOperator(
        task_id="validate_model",
        python_callable=_validate_model,
    )
    t_deploy = PythonOperator(
        task_id="deploy_model",
        python_callable=_deploy_model,
    )
    t_fail = PythonOperator(
        task_id="validation_failed",
        python_callable=_validation_failed,
    )

    (
        start
        >> t_load
        >> t_train
        >> t_validate
        >> [t_deploy, t_fail]
    )
    t_deploy >> end
    t_fail   >> end
