"""
DDI Feature Engineering DAG

ETL 완료 후 ML 피처를 생성하고 정규화/선택까지 수행.

스케줄: ETL DAG 완료 후 자동 트리거 (ExternalTaskSensor)
         또는 매일 새벽 3시 30분 (ETL 완료 여유 시간 고려)
실행 시간: 약 10~20분

환경변수:
  DDI_PROCESSED_DIR    : ETL 결과 디렉토리 (기본: /app/data/processed)
  DDI_FEATURES_DIR     : ML 피처 저장 디렉토리 (기본: /app/data/features)
  DDI_CYP_MATRIX_PATH  : CYP 매트릭스 경로 (기본: /app/data/processed/cyp_matrix.parquet)
  DDI_NORMALIZER_PATH  : 정규화기 저장 경로 (기본: /app/models/normalizer.pkl)
  DDI_SELECTOR_PATH    : 피처 선택기 저장 경로 (기본: /app/models/selector.pkl)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

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
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

PROC_DIR       = os.environ.get("DDI_PROCESSED_DIR", "/app/data/processed")
FEATURES_DIR   = os.environ.get("DDI_FEATURES_DIR", "/app/data/features")
CYP_PATH       = os.environ.get("DDI_CYP_MATRIX_PATH", "/app/data/processed/cyp_matrix.parquet")
NORMALIZER_PATH = os.environ.get("DDI_NORMALIZER_PATH", "/app/models/normalizer.pkl")
SELECTOR_PATH  = os.environ.get("DDI_SELECTOR_PATH", "/app/models/selector.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 태스크 함수
# ─────────────────────────────────────────────────────────────────────────────

def _get_partition(**context) -> str:
    partition = context["execution_date"].strftime("%Y%m%d")
    context["ti"].xcom_push(key="partition", value=partition)
    return partition


def _extract_cyp_features(**context) -> None:
    """CYP 효소 상호작용 피처 추출."""
    import sys
    sys.path.insert(0, "/app")
    import pandas as pd
    from scripts.features.cyp_features import CYPFeatureExtractor

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")
    features_df = pd.read_parquet(f"{PROC_DIR}/patient_features_{partition}.parquet")

    extractor = CYPFeatureExtractor(cyp_matrix_path=CYP_PATH)

    cyp_rows = []
    for _, row in features_df.iterrows():
        atc_list = row.get("atc_codes", []) or []
        cyp_feat = extractor.extract(atc_list)
        cyp_feat["patient_id"] = row["patient_id"]
        cyp_rows.append(cyp_feat)

    cyp_df = pd.DataFrame(cyp_rows)
    features_df = features_df.merge(cyp_df, on="patient_id", how="left")
    features_df.to_parquet(f"{PROC_DIR}/patient_features_cyp_{partition}.parquet", index=False)


def _extract_temporal_features(**context) -> None:
    """시계열 처방 패턴 피처 추출."""
    import sys
    sys.path.insert(0, "/app")
    import pandas as pd
    from scripts.features.temporal_features import extract_temporal

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")
    features_df = pd.read_parquet(f"{PROC_DIR}/patient_features_cyp_{partition}.parquet")
    prescriptions_df = pd.read_parquet(f"{PROC_DIR}/t30_{partition}_std.parquet")
    t20_df = pd.read_parquet(f"{PROC_DIR}/t20_{partition}_pseudo.parquet")

    joined = prescriptions_df.merge(
        t20_df[["claim_id", "patient_id", "prescription_date"]],
        on="claim_id", how="left",
    )

    temporal_rows = []
    for patient_id, grp in joined.groupby("patient_id"):
        if grp.empty:
            continue
        dates = pd.to_datetime(grp["prescription_date"])
        window_end = dates.max()
        window_start = window_end - pd.Timedelta(days=90)
        feat = extract_temporal(grp, window_start, window_end)
        feat["patient_id"] = patient_id
        temporal_rows.append(feat)

    temporal_df = pd.DataFrame(temporal_rows)
    features_df = features_df.merge(temporal_df, on="patient_id", how="left")
    features_df.to_parquet(f"{PROC_DIR}/patient_features_temporal_{partition}.parquet", index=False)


def _create_labels(**context) -> None:
    """이진 레이블 생성 (Red=1, else=0)."""
    import sys
    sys.path.insert(0, "/app")
    import pandas as pd

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")
    df = pd.read_parquet(f"{PROC_DIR}/patient_features_temporal_{partition}.parquet")

    df["is_high_risk"] = (df["risk_level"] == "Red").astype(int)
    df.to_parquet(f"{PROC_DIR}/patient_features_labeled_{partition}.parquet", index=False)

    n_pos = int(df["is_high_risk"].sum())
    n_total = len(df)
    import logging
    logging.info("레이블 생성 완료: %d/%d 고위험 (%.1f%%)", n_pos, n_total, n_pos / max(n_total, 1) * 100)


def _normalize_features(**context) -> None:
    """RobustScaler 정규화 (fit 또는 transform)."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import pandas as pd
    from scripts.features.normalizer import FeatureNormalizer

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")
    df = pd.read_parquet(f"{PROC_DIR}/patient_features_labeled_{partition}.parquet")

    normalizer = FeatureNormalizer()
    if os.path.exists(NORMALIZER_PATH):
        normalizer = FeatureNormalizer.load(NORMALIZER_PATH)
        df_norm = normalizer.transform(df)
    else:
        df_norm = normalizer.fit_transform(df)
        os.makedirs(os.path.dirname(NORMALIZER_PATH), exist_ok=True)
        normalizer.save(NORMALIZER_PATH)

    df_norm.to_parquet(f"{PROC_DIR}/patient_features_norm_{partition}.parquet", index=False)


def _select_features(**context) -> None:
    """분산/상관 기반 피처 선택 후 ML 피처 파일 저장."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import pandas as pd
    from scripts.features.selector import FeatureSelector

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")
    df = pd.read_parquet(f"{PROC_DIR}/patient_features_norm_{partition}.parquet")

    selector = FeatureSelector()
    if os.path.exists(SELECTOR_PATH):
        selector = FeatureSelector.load(SELECTOR_PATH)
        df_sel = selector.transform(df)
    else:
        df_sel = selector.fit_transform(df)
        os.makedirs(os.path.dirname(SELECTOR_PATH), exist_ok=True)
        selector.save(SELECTOR_PATH)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    out_path = f"{FEATURES_DIR}/ml_features_{partition}.parquet"
    df_sel.to_parquet(out_path, index=False)

    import logging
    logging.info("ML 피처 저장 완료: %s (%d행 × %d열)", out_path, len(df_sel), len(df_sel.columns))


# ─────────────────────────────────────────────────────────────────────────────
# DAG 정의
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="ddi_feature_engineering",
    description="DDI ML 피처 엔지니어링 (CYP + 시계열 + 정규화 + 피처선택)",
    default_args=DEFAULT_ARGS,
    schedule_interval="30 3 * * *",  # 매일 03:30 (ETL 완료 후)
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ddi", "features"],
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end")

    # ETL DAG 완료 대기
    wait_etl = ExternalTaskSensor(
        task_id="wait_for_etl",
        external_dag_id="ddi_etl",
        external_task_id="end",
        timeout=3600,            # 최대 1시간 대기
        poke_interval=60,
        mode="reschedule",
    )

    t_partition = PythonOperator(
        task_id="get_partition",
        python_callable=_get_partition,
    )
    t_cyp = PythonOperator(
        task_id="extract_cyp_features",
        python_callable=_extract_cyp_features,
    )
    t_temporal = PythonOperator(
        task_id="extract_temporal_features",
        python_callable=_extract_temporal_features,
    )
    t_labels = PythonOperator(
        task_id="create_labels",
        python_callable=_create_labels,
    )
    t_norm = PythonOperator(
        task_id="normalize_features",
        python_callable=_normalize_features,
    )
    t_select = PythonOperator(
        task_id="select_features",
        python_callable=_select_features,
    )

    (
        start
        >> wait_etl
        >> t_partition
        >> t_cyp
        >> t_temporal
        >> t_labels
        >> t_norm
        >> t_select
        >> end
    )
