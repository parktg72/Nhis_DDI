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

from config.settings import (
    FEATURES_DIR,
    MODEL_DIR,
    TRAIN_WEEKS,
    MODEL_TYPE,
    OPTUNA_TRIALS,
    RECALL_THRESHOLD,
    AUC_THRESHOLD,
)

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


def _atomic_symlink_update(link_path, target_name: str) -> None:
    """POSIX 원자적 심링크 교체: tmp 심링크 생성 후 os.replace로 swap."""
    import os
    from pathlib import Path
    link_path = Path(link_path)
    tmp_link = link_path.parent / (link_path.name + ".tmp")
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(target_name)
    os.replace(tmp_link, link_path)


def _deploy_model(**context) -> None:
    """검증 통과 모델을 production 경로로 원자적 배포 + serving 핫스왑.

    배포 순서:
      Phase 1: 전체 아티팩트 존재 검증 (복사 없음 — 실패해도 prod_dir 불변)
      Phase 2: 임시 디렉터리에 전체 복사
      Phase 3: 기존 current→versioned_dir 파일 backup/ 에 보존
      Phase 4: tmp_dir → versioned_dir rename + current 심링크 원자적 교체
    """
    import sys
    sys.path.insert(0, "/app")
    import os
    import time
    import logging
    import shutil
    import requests
    from pathlib import Path
    from config import settings as _s

    model_path = Path(context["ti"].xcom_pull(key="model_path", task_ids="run_training"))
    prod_dir   = _s.MODEL_DIR
    prod_path  = _s.MODEL_PROD_PATH
    base_src   = model_path.with_suffix("")  # e.g. /app/models/model_v1

    # ── Phase 1: 전체 아티팩트 선검증 ────────────────────────────────────────
    artifacts: list = [
        (model_path,                              "model_prod.pkl"),
        (Path(str(model_path) + ".sha256"),        "model_prod.pkl.sha256"),
    ]
    for ext in (".xgb.pkl", ".lgb.pkl"):
        sub_src = Path(str(base_src) + ext)
        if sub_src.exists():
            sub_sha = Path(str(sub_src) + ".sha256")
            if not sub_sha.exists():
                raise RuntimeError(
                    f"배포 중단 — 서브모델 해시 없음: {sub_sha}"
                )
            artifacts.append((sub_src, "model_prod" + ext))
            artifacts.append((sub_sha, "model_prod" + ext + ".sha256"))

    for src, _ in artifacts:
        if not src.exists():
            raise RuntimeError(f"배포 중단 — 필수 아티팩트 없음: {src}")

    # ── Phase 2: 임시 디렉터리에 전체 복사 ───────────────────────────────────
    tmp_dir = prod_dir / ".deploy_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for src, dst_name in artifacts:
        shutil.copy2(src, tmp_dir / dst_name)

    # ── Phase 3: 기존 current → backup/ 에 보존 ──────────────────────────────
    current_link = prod_dir / "current"
    backup_dir = prod_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    if current_link.is_symlink():
        old_version_dir = current_link.resolve()
        for f in old_version_dir.glob("model_prod*"):
            if f.is_file():
                shutil.copy2(f, backup_dir / f.name)

    # ── Phase 4: versioned_dir rename + current 심링크 원자적 교체 ───────────
    versioned_name = f".v_{int(time.time())}"
    versioned_dir = prod_dir / versioned_name
    os.rename(tmp_dir, versioned_dir)
    _atomic_symlink_update(current_link, versioned_name)

    # ── Serving 핫스왑 ────────────────────────────────────────────────────────
    try:
        resp = requests.post(
            f"{_s.SERVING_URL}/admin/reload",
            json={"model_path": str(prod_path)},
            headers={"X-Admin-Key": _s.ADMIN_API_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        logging.info("Serving 핫스왑 완료: %s", resp.json())
    except Exception as exc:
        logging.warning("핫스왑 실패: %s", exc)
        raise RuntimeError(f"핫스왑 실패 — 구버전 모델로 서빙 중: {exc}") from exc


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
