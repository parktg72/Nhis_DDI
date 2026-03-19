"""
DDI ETL DAG

청구 데이터(T20/T30/T40/T50)를 매일 처리하여 PatientFeatures를 생성.

스케줄: 매일 새벽 2시 (배치 데이터 적재 완료 후)
실행 시간: 약 30~60분 (800만 환자 기준)

환경변수:
  DDI_RAW_DATA_DIR   : 원천 청구 데이터 디렉토리 (기본: /app/data/raw)
  DDI_PROCESSED_DIR  : 처리 결과 저장 디렉토리 (기본: /app/data/processed)
  DDI_DRUG_INDEX_PATH: 약물 인덱스 경로 (기본: /app/data/processed/drug_name_index.parquet)
  DDI_PARTITION      : 데이터 파티션 식별자 (기본: YYYYMMDD)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
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
    "retry_delay": timedelta(minutes=10),
}

RAW_DIR = os.environ.get("DDI_RAW_DATA_DIR", "/app/data/raw")
PROC_DIR = os.environ.get("DDI_PROCESSED_DIR", "/app/data/processed")
DRUG_INDEX = os.environ.get("DDI_DRUG_INDEX_PATH", "/app/data/processed/drug_name_index.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# 태스크 함수
# ─────────────────────────────────────────────────────────────────────────────

def _get_partition(**context) -> str:
    """실행 날짜 기반 파티션 키 반환 (YYYYMMDD)."""
    execution_date = context["execution_date"]
    partition = execution_date.strftime("%Y%m%d")
    context["ti"].xcom_push(key="partition", value=partition)
    return partition


def _validate_schemas(**context) -> None:
    """T20/T30/T40/T50 스키마 검증."""
    import sys
    sys.path.insert(0, "/app")
    import pandas as pd
    from scripts.etl.schema_validator import validate_all

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")
    t20 = pd.read_parquet(f"{RAW_DIR}/t20_{partition}.parquet")
    t30 = pd.read_parquet(f"{RAW_DIR}/t30_{partition}.parquet")
    t40 = pd.read_parquet(f"{RAW_DIR}/t40_{partition}.parquet")
    t50 = pd.read_parquet(f"{RAW_DIR}/t50_{partition}.parquet")

    results = validate_all(t20, t30, t40, t50)
    failed = [r for r in results if not r.passed]
    if failed:
        msg = "; ".join(
            f"{r.table}: missing={r.missing_cols}, errors={r.type_errors}"
            for r in failed
        )
        raise ValueError(f"스키마 검증 실패: {msg}")


def _pseudonymize(**context) -> None:
    """환자 식별자 SHA-256 가명처리."""
    import sys
    sys.path.insert(0, "/app")
    import pandas as pd
    from scripts.etl.pseudonymizer import pseudonymize_dataframe

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")
    for table in ("t20", "t30", "t40"):
        df = pd.read_parquet(f"{RAW_DIR}/{table}_{partition}.parquet")
        df = pseudonymize_dataframe(df, id_cols=["patient_id"])
        df.to_parquet(f"{PROC_DIR}/{table}_{partition}_pseudo.parquet", index=False)


def _standardize_codes(**context) -> None:
    """EDI 코드 → ATC 코드 표준화."""
    import sys
    sys.path.insert(0, "/app")
    import pandas as pd
    from scripts.etl.code_standardizer import CodeStandardizer

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")
    std = CodeStandardizer(drug_index_path=DRUG_INDEX)

    t30 = pd.read_parquet(f"{PROC_DIR}/t30_{partition}_pseudo.parquet")
    t30 = std.standardize(t30)
    t30.to_parquet(f"{PROC_DIR}/t30_{partition}_std.parquet", index=False)

    unknown = std.unknown_rate(t30)
    if unknown > 0.10:
        import logging
        logging.warning("EDI 미매핑율 %.1f%% (임계: 10%%)", unknown * 100)


def _quality_check(**context) -> None:
    """데이터 품질 검사."""
    import sys
    sys.path.insert(0, "/app")
    import pandas as pd
    from scripts.etl.quality_checker import check_all

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")
    t20 = pd.read_parquet(f"{PROC_DIR}/t20_{partition}_pseudo.parquet")
    t30 = pd.read_parquet(f"{PROC_DIR}/t30_{partition}_std.parquet")

    report = check_all(t20, t30)
    if report.critical_issues:
        raise ValueError(f"품질 검사 치명적 오류: {report.critical_issues}")


def _calculate_overlaps(**context) -> None:
    """90일 슬라이딩 윈도우 동시복용 쌍 계산."""
    import sys
    sys.path.insert(0, "/app")
    import pandas as pd
    from scripts.etl.overlap_calculator import calculate_overlaps_batch

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")
    t30 = pd.read_parquet(f"{PROC_DIR}/t30_{partition}_std.parquet")
    t20 = pd.read_parquet(f"{PROC_DIR}/t20_{partition}_pseudo.parquet")

    # T20-T30 조인
    joined = t30.merge(
        t20[["claim_id", "patient_id", "prescription_date"]],
        on="claim_id", how="left",
    )
    pairs_df = calculate_overlaps_batch(joined)
    pairs_df.to_parquet(f"{PROC_DIR}/overlap_pairs_{partition}.parquet", index=False)


def _aggregate_features(**context) -> None:
    """PatientFeatures 집계."""
    import sys
    sys.path.insert(0, "/app")
    import pandas as pd
    from scripts.etl.prescription_aggregator import aggregate_batch

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")
    t30 = pd.read_parquet(f"{PROC_DIR}/t30_{partition}_std.parquet")
    t20 = pd.read_parquet(f"{PROC_DIR}/t20_{partition}_pseudo.parquet")
    pairs = pd.read_parquet(f"{PROC_DIR}/overlap_pairs_{partition}.parquet")

    joined = t30.merge(
        t20[["claim_id", "patient_id", "prescription_date"]],
        on="claim_id", how="left",
    )
    features = aggregate_batch(joined, pairs)
    context["ti"].xcom_push(key="n_patients", value=len(features))
    return features


def _write_features(**context) -> None:
    """피처 데이터 저장 및 파이프라인 로그 기록."""
    import sys
    sys.path.insert(0, "/app")
    import pandas as pd
    from scripts.etl.feature_writer import write_features, write_pipeline_log

    partition = context["ti"].xcom_pull(key="partition", task_ids="get_partition")
    t30 = pd.read_parquet(f"{PROC_DIR}/t30_{partition}_std.parquet")
    t20 = pd.read_parquet(f"{PROC_DIR}/t20_{partition}_pseudo.parquet")
    pairs = pd.read_parquet(f"{PROC_DIR}/overlap_pairs_{partition}.parquet")

    from scripts.etl.prescription_aggregator import aggregate_batch
    joined = t30.merge(
        t20[["claim_id", "patient_id", "prescription_date"]],
        on="claim_id", how="left",
    )
    features = aggregate_batch(joined, pairs)

    out_path = f"{PROC_DIR}/patient_features_{partition}.parquet"
    write_features(features, out_path)
    write_pipeline_log(
        partition=partition,
        n_patients=len(features),
        out_path=out_path,
        log_dir=f"{PROC_DIR}/logs",
    )


# ─────────────────────────────────────────────────────────────────────────────
# DAG 정의
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="ddi_etl",
    description="DDI 청구 데이터 ETL 파이프라인 (T20/T30/T40/T50 → PatientFeatures)",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 * * *",   # 매일 새벽 2시
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ddi", "etl"],
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end")

    t_partition = PythonOperator(
        task_id="get_partition",
        python_callable=_get_partition,
    )
    t_validate = PythonOperator(
        task_id="validate_schemas",
        python_callable=_validate_schemas,
    )
    t_pseudo = PythonOperator(
        task_id="pseudonymize",
        python_callable=_pseudonymize,
    )
    t_std = PythonOperator(
        task_id="standardize_codes",
        python_callable=_standardize_codes,
    )
    t_qc = PythonOperator(
        task_id="quality_check",
        python_callable=_quality_check,
    )
    t_overlap = PythonOperator(
        task_id="calculate_overlaps",
        python_callable=_calculate_overlaps,
    )
    t_agg = PythonOperator(
        task_id="aggregate_features",
        python_callable=_aggregate_features,
    )
    t_write = PythonOperator(
        task_id="write_features",
        python_callable=_write_features,
    )

    # 의존성 체인
    (
        start
        >> t_partition
        >> t_validate
        >> t_pseudo
        >> t_std
        >> t_qc
        >> t_overlap
        >> t_agg
        >> t_write
        >> end
    )
