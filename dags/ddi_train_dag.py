"""
DDI Model Training DAG

주간 모델 재훈련: ML 피처 → XGBoost/LightGBM 훈련 → 검증 → 모델 배포.

스케줄: 매주 월요일 새벽 4시
실행 시간: 약 30~90분 (Optuna 튜닝 포함)

환경변수:
  DDI_FEATURES_DIR   : ML 피처 디렉토리 (기본: /app/data/features)
  MODEL_DIR          : 모델 저장 디렉토리 (기본: /app/models)
  DDI_TRAIN_WEEKS    : 훈련에 사용할 주 수 (기본: 4)
  DDI_MODEL_TYPE     : xgboost | lightgbm | ensemble | ensemble_gat (기본: ensemble)
  DDI_OPTUNA_TRIALS  : Optuna 시도 횟수 (기본: 50)
  DDI_RECALL_THRESHOLD: 최소 Recall 기준 (기본: 0.90)
  DDI_AUC_THRESHOLD  : 최소 AUC 기준 (기본: 0.85)
  DDI_SERVING_URL    : 서빙 API URL (기본: http://localhost:8000)
  ADMIN_API_KEY      : /admin/reload 인증 키 (serving ADMIN_API_KEY와 동일)
  PRESCRIPTION_DATA_PATH : 처방 Parquet 경로 (ensemble_gat 전용, train split)
  DDI_DATA_PATH      : DDI 지식베이스 Parquet/CSV 경로 (ensemble_gat 전용)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
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
    """Optuna 하이퍼파라미터 튜닝 + 모델 훈련.

    ensemble_gat 모델 타입을 사용할 경우 다음 환경변수가 필요합니다:
      PRESCRIPTION_DATA_PATH : 처방 Parquet 경로 (train split)
      DDI_DATA_PATH          : DDI 지식베이스 Parquet/CSV 경로
    """
    import sys
    sys.path.insert(0, "/app")
    import os
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
        prescription_data_path=os.environ.get("PRESCRIPTION_DATA_PATH", ""),
        ddi_data_path=os.environ.get("DDI_DATA_PATH", ""),
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


def _prune_old_versioned_dirs(prod_dir, keep_n: int) -> None:
    """오래된 .v_* 버전 디렉터리를 삭제해 keep_n 개만 유지.

    타임스탬프 숫자 기준 오름차순 정렬 — 오래된 순으로 삭제.
    """
    import shutil
    from pathlib import Path
    prod_dir = Path(prod_dir)

    def _ts(p):
        suffix = p.name[3:]  # ".v_" 제거
        return int(suffix) if suffix.isdigit() else 0

    versioned = sorted(prod_dir.glob(".v_*"), key=_ts)
    for old_dir in versioned[:-keep_n]:
        shutil.rmtree(old_dir, ignore_errors=True)


def _atomic_symlink_update(link_path, target_name: str) -> None:
    """POSIX 원자적 심링크 교체: tmp 심링크 생성 후 os.replace로 swap.

    os.replace는 reader 동시성을 보장하나 전원 손실 내구성(crash durability)은
    부모 디렉터리 fsync 없이는 보장되지 않는다.
    아래에서 부모 디렉터리를 fsync해 저널 커밋을 강제한다.
    """
    import os as _os
    from pathlib import Path
    link_path = Path(link_path)
    tmp_link = link_path.parent / f"{link_path.name}.tmp.{_os.getpid()}"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(target_name)
    _os.replace(tmp_link, link_path)
    # 부모 디렉터리 fsync: 심링크 교체가 저널에 커밋됨을 보장
    dirfd = _os.open(str(link_path.parent), _os.O_RDONLY)
    try:
        _os.fsync(dirfd)
    finally:
        _os.close(dirfd)


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
    import tempfile
    import logging
    import shutil
    import hashlib
    import pickle
    import requests
    from pathlib import Path
    from config import settings as _s

    # H2: 사전 검증 (파일시스템 변경 전) — 실패 시 prod_dir 불변
    if not _s.ADMIN_API_KEY:
        raise RuntimeError("ADMIN_API_KEY 미설정 — serving 핫스왑 불가, 배포 중단")

    model_path = Path(context["ti"].xcom_pull(key="model_path", task_ids="run_training"))
    prod_dir   = _s.MODEL_DIR
    prod_path  = _s.MODEL_PROD_PATH
    base_src   = model_path.with_suffix("")  # e.g. /app/models/model_v1

    # M1: 이전 배포 크래시로 남겨진 .deploy_tmp_* 잔재 자동 정리
    import glob as _glob
    for _stale in _glob.glob(str(prod_dir / ".deploy_tmp_*")):
        shutil.rmtree(_stale, ignore_errors=True)
        logging.warning("크래시 잔재 정리: %s", _stale)

    # ── Phase 1: 전체 아티팩트 선검증 ────────────────────────────────────────
    artifacts: list = [
        (model_path,                              "model_prod.pkl"),
        (Path(str(model_path) + ".sha256"),        "model_prod.pkl.sha256"),
    ]
    model_content = model_path.read_bytes()
    model_sha_path = Path(str(model_path) + ".sha256")
    expected_model_sha = model_sha_path.read_text().strip().split()[0]
    actual_model_sha = hashlib.sha256(model_content).hexdigest()
    if actual_model_sha != expected_model_sha:
        raise RuntimeError(f"배포 중단 — 메인 모델 해시 불일치: {model_path}")
    model_state = pickle.loads(model_content)
    trainer_class = model_state.get("trainer_class")
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

    # GAT 아티팩트 (EnsembleTrainer3Way인 경우 필수)
    if trainer_class == "EnsembleTrainer3Way":
        required_gat = [
            ("gat_model.pt", "gat_model.pt.sha256"),
            ("gat_graph.pt", "gat_graph.pt.sha256"),
        ]
        for gat_file, gat_sha_suffix in required_gat:
            gat_src = base_src.parent / gat_file
            gat_sha = base_src.parent / gat_sha_suffix
            if not gat_src.exists():
                raise RuntimeError(f"배포 중단 — EnsembleTrainer3Way 필수 GAT 아티팩트 없음: {gat_src}")
            if not gat_sha.exists():
                raise RuntimeError(f"배포 중단 — GAT 아티팩트 해시 없음: {gat_sha}")
            artifacts.append((gat_src, gat_file))
            artifacts.append((gat_sha, gat_sha_suffix))

    # 선택적 GAT 아티팩트 (비-3way 저장물에 우연히 공존하는 경우만 복사)
    for gat_file, gat_sha_suffix in [
        ("gat_model.pt",   "gat_model.pt.sha256"),
        ("gat_graph.pt",   "gat_graph.pt.sha256"),
    ]:
        gat_src = base_src.parent / gat_file
        if trainer_class == "EnsembleTrainer3Way":
            continue
        if gat_src.exists():
            gat_sha = base_src.parent / gat_sha_suffix
            if not gat_sha.exists():
                raise RuntimeError(
                    f"배포 중단 — GAT 아티팩트 해시 없음: {gat_sha}"
                )
            artifacts.append((gat_src, gat_file))
            artifacts.append((gat_sha, gat_sha_suffix))
    # gat_graph_meta.json (sha256 없음 — 메타데이터)
    gat_meta_src = base_src.parent / "gat_graph_meta.json"
    if gat_meta_src.exists():
        artifacts.append((gat_meta_src, "gat_graph_meta.json"))

    for src, _ in artifacts:
        if not src.exists():
            raise RuntimeError(f"배포 중단 — 필수 아티팩트 없음: {src}")

    # ── Phase 2: 임시 디렉터리에 전체 복사 (per-run 격리 — 동시 배포 안전) ────
    tmp_dir = Path(tempfile.mkdtemp(dir=prod_dir, prefix=".deploy_tmp_"))
    try:
        for src, dst_name in artifacts:
            shutil.copy2(src, tmp_dir / dst_name)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # ── Phase 3: 기존 current → backup/ 에 보존 ──────────────────────────────
    current_link = prod_dir / "current"
    backup_dir = prod_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    prev_versioned_name: str | None = None
    if current_link.is_symlink():
        prev_versioned_name = os.readlink(current_link)
        old_version_dir = current_link.resolve()
        for f in old_version_dir.glob("model_prod*"):
            if f.is_file():
                shutil.copy2(f, backup_dir / f.name)

    # ── Phase 4: versioned_dir rename + current 심링크 원자적 교체 ───────────
    versioned_name = f".v_{time.time_ns()}"
    versioned_dir = prod_dir / versioned_name
    os.rename(tmp_dir, versioned_dir)
    # H1: os.rename 후 prod_dir fsync — versioned_dir가 저널에 커밋됨을 보장
    _dirfd = os.open(str(prod_dir), os.O_RDONLY)
    try:
        os.fsync(_dirfd)
    finally:
        os.close(_dirfd)
    _atomic_symlink_update(current_link, versioned_name)

    # ── Serving 핫스왑 (SERVING_URLS 다중 인스턴스 브로드캐스트) ─────────────
    # C1: symlink 경로(prod_path) 대신 versioned 절대 경로 전송
    #     → 동시 배포 경쟁 시 serving이 정확히 의도한 버전을 로드
    new_versioned_prod_path = versioned_dir / prod_path.name
    serving_urls = getattr(_s, "SERVING_URLS", None) or [_s.SERVING_URL]
    failures: list = []
    succeeded_urls: list = []
    for url in serving_urls:
        try:
            resp = requests.post(
                f"{url}/admin/reload",
                json={"model_path": str(new_versioned_prod_path)},
                headers={"X-Admin-Key": _s.ADMIN_API_KEY},
                timeout=30,
            )
            resp.raise_for_status()
            logging.info("Serving 핫스왑 완료 [%s]: %s", url, resp.json())
            succeeded_urls.append(url)
        except Exception as exc:
            logging.warning("핫스왑 실패 [%s]: %s", url, exc)
            failures.append((url, exc))

    if failures:
        # 롤백: current를 이전 버전으로 복구
        if prev_versioned_name:
            _atomic_symlink_update(current_link, prev_versioned_name)
            logging.warning("핫스왑 실패 — current를 %s 로 롤백", prev_versioned_name)
        # 보상 롤백: 이미 성공한 서버들에 구버전 경로 재전송
        comp_failures: list = []
        if succeeded_urls and prev_versioned_name:
            old_prod_path = prod_dir / prev_versioned_name / prod_path.name
            for url in succeeded_urls:
                try:
                    requests.post(
                        f"{url}/admin/reload",
                        json={"model_path": str(old_prod_path)},
                        headers={"X-Admin-Key": _s.ADMIN_API_KEY},
                        timeout=30,
                    ).raise_for_status()
                    logging.info("보상 롤백 완료 [%s]", url)
                except Exception as comp_exc:
                    logging.error("보상 롤백 실패 [%s]: %s", url, comp_exc)
                    comp_failures.append((url, comp_exc))
        detail = "; ".join(f"{u}: {e}" for u, e in failures)
        if comp_failures:
            comp_detail = "; ".join(f"{u}: {e}" for u, e in comp_failures)
            raise RuntimeError(
                f"핫스왑 실패 — {detail}; 보상롤백 실패(수동 확인 필요) — {comp_detail}"
            )
        raise RuntimeError(f"핫스왑 실패 — {detail}")

    # ── hotswap 성공 후 오래된 버전 디렉터리 정리 ─────────────────────────────
    _prune_old_versioned_dirs(prod_dir, keep_n=getattr(_s, "BACKUP_KEEP_N", 5))


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
