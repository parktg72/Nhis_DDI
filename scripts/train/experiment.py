"""
MLflow 실험 추적

폐쇄망 환경: mlflow tracking URI = 로컬 파일시스템 (mlruns/)
인터넷 환경: mlflow tracking URI = MLflow 서버 URL

기록 항목:
  - 하이퍼파라미터
  - 평가 지표 (AUC, Recall, Precision, F1)
  - 혼동행렬 (artifact)
  - 피처 중요도 상위 20개 (artifact)
  - 모델 파일 (artifact)
  - 데이터 정보 (파티션, 샘플 수, 클래스 분포)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .evaluator import EvalResult

logger = logging.getLogger(__name__)

_MLFLOW_AVAILABLE = False
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    pass


class ExperimentTracker:
    """
    MLflow 실험 추적기.
    mlflow 미설치 시 로컬 JSON 로그로 폴백.
    """

    def __init__(
        self,
        experiment_name: str = "ddi_risk_model",
        tracking_uri: str = "mlruns",
    ):
        self.experiment_name = experiment_name
        self._run_id: Optional[str] = None
        self._log_buffer: dict[str, Any] = {}

        if _MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info("MLflow 추적 활성화: %s", tracking_uri)
        else:
            logger.warning("MLflow 미설치. 로컬 JSON 폴백 사용.")

    # ──────────────────────────────────────────────────────────────────────────
    # 컨텍스트 매니저
    # ──────────────────────────────────────────────────────────────────────────

    def start_run(self, run_name: Optional[str] = None) -> "ExperimentTracker":
        if _MLFLOW_AVAILABLE:
            mlflow.start_run(run_name=run_name)
            self._run_id = mlflow.active_run().info.run_id
            logger.info("MLflow run 시작: %s", self._run_id)
        else:
            self._run_id = f"local_{run_name or 'run'}"
        self._log_buffer = {"run_id": self._run_id, "metrics": {}, "params": {}}
        return self

    def end_run(self, log_dir: str | Path = "mlruns/local") -> None:
        if _MLFLOW_AVAILABLE:
            mlflow.end_run()
        else:
            # 로컬 JSON 저장
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            out = Path(log_dir) / f"{self._run_id}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(self._log_buffer, f, ensure_ascii=False, indent=2, default=str)
            logger.info("로컬 로그 저장: %s", out)

    def __enter__(self) -> "ExperimentTracker":
        return self.start_run()

    def __exit__(self, *args) -> None:
        self.end_run()

    # ──────────────────────────────────────────────────────────────────────────
    # 로깅 메서드
    # ──────────────────────────────────────────────────────────────────────────

    def log_params(self, params: dict[str, Any]) -> None:
        if _MLFLOW_AVAILABLE:
            # MLflow 파라미터 값은 문자열로 변환
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        self._log_buffer.setdefault("params", {}).update(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if _MLFLOW_AVAILABLE:
            mlflow.log_metric(key, value, step=step)
        self._log_buffer.setdefault("metrics", {})[key] = value

    def log_eval_result(self, result: EvalResult) -> None:
        """EvalResult 전체 지표 기록."""
        prefix = result.split
        self.log_metric(f"{prefix}_auc_roc",   result.auc_roc)
        self.log_metric(f"{prefix}_auc_pr",    result.auc_pr)
        self.log_metric(f"{prefix}_recall",    result.recall)
        self.log_metric(f"{prefix}_precision", result.precision)
        self.log_metric(f"{prefix}_f1",        result.f1)
        self.log_metric(f"{prefix}_threshold", result.threshold)
        self.log_metric(f"{prefix}_tp",        result.tp)
        self.log_metric(f"{prefix}_fp",        result.fp)
        self.log_metric(f"{prefix}_fn",        result.fn)
        self.log_metric(f"{prefix}_tn",        result.tn)

    def log_artifact(self, local_path: str | Path, artifact_path: Optional[str] = None) -> None:
        if _MLFLOW_AVAILABLE:
            mlflow.log_artifact(str(local_path), artifact_path)
        self._log_buffer.setdefault("artifacts", []).append(str(local_path))

    def log_feature_importance(
        self,
        importance_df: Any,
        top_n: int = 20,
        log_dir: str | Path = "mlruns/local",
    ) -> None:
        """피처 중요도 상위 N개 기록."""
        if importance_df is None or len(importance_df) == 0:
            return
        top = importance_df.head(top_n)

        # 콘솔 출력
        print(f"\n[피처 중요도 Top {top_n}]")
        for _, row in top.iterrows():
            bar = "█" * int(row["importance"] / top["importance"].max() * 20)
            print(f"  {row['feature']:35s} {row['importance']:8.4f} {bar}")

        # 파일 저장 + MLflow artifact
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        out = Path(log_dir) / "feature_importance.csv"
        top.to_csv(out, index=False)
        self.log_artifact(out, "feature_importance")

    def log_dataset_info(self, dataset: Any) -> None:
        """데이터셋 정보 기록."""
        dist = dataset.class_distribution()
        self.log_params({
            "n_train":       dataset.n_train,
            "n_val":         dataset.n_val,
            "n_test":        dataset.n_test,
            "n_features":    dataset.n_features,
            "train_pos":     dist["train"]["pos"],
            "train_neg":     dist["train"]["neg"],
            "pos_weight":    round(dataset.pos_weight, 2),
        })
