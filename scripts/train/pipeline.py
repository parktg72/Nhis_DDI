"""
훈련 파이프라인 오케스트레이터

실행 흐름:
  1. 데이터셋 로드 (ml_features_{partition}.parquet)
  2. Optuna 하이퍼파라미터 탐색 (옵션)
  3. 모델 훈련 (XGBoost / LightGBM / Ensemble)
  4. 임계값 최적화 (Recall ≥ 90%)
  5. 평가 출력 (AUC, Recall, 혼동행렬)
  6. 모델 저장 (models/ddi_model_{partition}.pkl)
  7. MLflow 실험 기록
  8. 합격 기준 검증 (Recall ≥ 0.90, AUC ≥ 0.85)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .dataset import TrainDataset, load_dataset
from .evaluator import EvalResult, evaluate_all_splits
from .experiment import ExperimentTracker
from .hyperparams import TrainConfig
from .trainer import BaseTrainer, build_trainer

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """훈련 파이프라인 실행 결과."""
    partition: str
    model_type: str
    model_path: str = ""
    eval_results: dict[str, EvalResult] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    passed: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def val_recall(self) -> float:
        return self.eval_results.get("val", EvalResult("val")).recall

    @property
    def val_auc(self) -> float:
        return self.eval_results.get("val", EvalResult("val")).auc_roc

    def print_summary(self) -> None:
        status = "PASS ✓" if self.passed else "FAIL ✗"
        print(f"\n{'='*60}")
        print(f"[훈련 결과 요약] {status}")
        print(f"{'='*60}")
        print(f"  모델: {self.model_type}  파티션: {self.partition}")
        print(f"  소요: {self.elapsed_seconds:.1f}초")
        for split, res in self.eval_results.items():
            res.print()
        if self.errors:
            for e in self.errors:
                print(f"  [오류] {e}")
        print(f"{'='*60}")


class TrainPipeline:
    """
    전체 훈련 파이프라인.

    Parameters
    ----------
    config : TrainConfig
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.model_dir = Path(config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def run(self, partition: Optional[str] = None) -> TrainResult:
        partition = partition or self.config.partition
        t0 = time.perf_counter()
        result = TrainResult(partition=partition, model_type=self.config.model_type)

        tracker = ExperimentTracker(
            experiment_name=self.config.experiment_name,
        )
        tracker.start_run(run_name=f"{self.config.model_type}_{partition}")

        try:
            # ── Step 1: 데이터셋 로드 ─────────────────────────────────────
            logger.info("[Step 1] 데이터셋 로드: partition=%s", partition)
            dataset = load_dataset(
                partition=partition,
                feature_base=self.config.feature_base,
                random_state=self.config.random_state,
            )
            tracker.log_dataset_info(dataset)

            # ── Step 2: 하이퍼파라미터 탐색 (Optuna) ──────────────────────
            if self.config.use_optuna:
                logger.info("[Step 2] Optuna 하이퍼파라미터 탐색")
                self._run_optuna(dataset)
            else:
                logger.info("[Step 2] Optuna 스킵 (use_optuna=False)")

            # ── Step 3: 모델 훈련 ─────────────────────────────────────────
            logger.info("[Step 3] 모델 훈련: %s", self.config.model_type)
            trainer = build_trainer(self.config)
            trainer.fit(dataset)
            tracker.log_params(self.config.get_model_params())

            # ensemble_gat 추가 훈련: GAT 서브모델 + 가중치 최적화
            if self.config.model_type == "ensemble_gat":
                self._run_gat_training(trainer, dataset)

            # ── Step 4: 평가 + 임계값 최적화 ──────────────────────────────
            logger.info("[Step 4] 평가")
            eval_results = trainer.evaluate(
                dataset,
                min_recall=self.config.recall_threshold,
                min_auc=self.config.auc_threshold,
            )
            result.eval_results = eval_results
            for split, res in eval_results.items():
                tracker.log_eval_result(res)

            # ── Step 5: 합격 기준 검증 ────────────────────────────────────
            val_res = eval_results.get("val", EvalResult("val"))
            result.passed = val_res.passed_recall and val_res.passed_auc
            if not val_res.passed_recall:
                result.errors.append(
                    f"Recall {val_res.recall:.3f} < 목표 {self.config.recall_threshold}"
                )
            if not val_res.passed_auc:
                result.errors.append(
                    f"AUC {val_res.auc_roc:.3f} < 목표 {self.config.auc_threshold}"
                )

            # ── Step 6: 모델 저장 ─────────────────────────────────────────
            import os
            model_path = self.model_dir / f"ddi_model_{partition}.pkl"
            feature_base_resolved = str(Path(self.config.feature_base).resolve())
            model_dir_resolved = str(self.model_dir.resolve())
            scaler_abs = Path(self.config.feature_base).resolve() / "scaler.pkl"
            selector_abs = Path(self.config.feature_base).resolve() / "selector.pkl"

            # feature contract metadata
            trainer._extra_meta = {
                "artifact_version": 2,
                "feature_names": list(dataset.feature_names),
                "scaler_path": os.path.relpath(str(scaler_abs), start=model_dir_resolved),
                "selector_path": os.path.relpath(str(selector_abs), start=model_dir_resolved),
            }
            trainer.save(model_path)
            result.model_path = str(model_path)
            tracker.log_artifact(model_path, "model")

            # ── DriftDetector 기준 분포 저장 ──────────────────────────────
            try:
                import pandas as pd
                train_df_for_drift = pd.DataFrame(dataset.X_train, columns=dataset.feature_names)
                self._save_drift_reference(train_df_for_drift)
            except Exception:
                logger.warning("DriftDetector 기준 분포 저장 실패 — 학습은 계속", exc_info=True)

            # ── Step 7: 피처 중요도 ───────────────────────────────────────
            imp_df = trainer.feature_importance_df(dataset.feature_names)
            if imp_df is not None:
                tracker.log_feature_importance(imp_df)

        except Exception as e:
            logger.exception("훈련 파이프라인 오류")
            result.errors.append(str(e))
            result.passed = False
        finally:
            tracker.end_run(log_dir="mlruns/local")

        result.elapsed_seconds = time.perf_counter() - t0
        result.print_summary()
        return result

    def _save_drift_reference(self, train_df, drift_reference_path=None) -> None:
        """DriftDetector 기준 분포를 학습 데이터(train split)로 fit 후 저장.

        label, patient_id, split 컬럼은 피처에서 제외한다.
        배포 이전에 호출되어야 한다.

        Args:
            train_df: 학습 분할 DataFrame
            drift_reference_path: 저장 경로 오버라이드 (테스트용; None이면 settings.DRIFT_REFERENCE_PATH 사용)
        """
        from monitoring.drift_detector import DriftDetector
        from config import settings as _s

        path = Path(drift_reference_path) if drift_reference_path is not None else _s.DRIFT_REFERENCE_PATH
        exclude_cols = {"label", "patient_id", "split"}
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        if not feature_cols:
            logger.warning("DriftDetector fit 대상 피처가 없음 — 기준 분포 저장 건너뜀")
            return

        detector = DriftDetector()
        detector.fit(train_df[feature_cols])
        path.parent.mkdir(parents=True, exist_ok=True)
        detector.save(str(path))
        logger.info("DriftDetector 기준 분포 저장 완료: %s (%d 피처)", path, len(feature_cols))

    def _run_gat_training(self, trainer, dataset) -> None:
        """ensemble_gat: GAT 서브모델 훈련 + 가중치 최적화."""
        import pandas as pd
        from scripts.features.graph_builder import GraphBuilder
        from scripts.train.gat_dataset import GATDataset

        if not self.config.prescription_data_path:
            raise RuntimeError(
                "ensemble_gat 훈련에는 TrainConfig.prescription_data_path가 필요합니다."
            )
        if not self.config.ddi_data_path:
            raise RuntimeError(
                "ensemble_gat 훈련에는 TrainConfig.ddi_data_path가 필요합니다."
            )

        logger.info("[Step 3b] GAT 서브모델 훈련")

        # 처방 데이터 로드 (train split 전용)
        presc_path = Path(self.config.prescription_data_path)
        if presc_path.suffix == ".parquet":
            prescription_df = pd.read_parquet(presc_path)
        else:
            prescription_df = pd.read_csv(presc_path)

        # train split만 필터링 (split 컬럼 있는 경우)
        if "split" in prescription_df.columns:
            prescription_df = prescription_df[prescription_df["split"] == "train"].copy()
        # split 컬럼 없으면 전체를 train으로 간주 + attrs 설정
        prescription_df.attrs["split"] = "train"

        # DDI 지식베이스 로드
        ddi_path = Path(self.config.ddi_data_path)
        if ddi_path.suffix == ".parquet":
            ddi_df = pd.read_parquet(ddi_path)
        else:
            ddi_df = pd.read_csv(ddi_path)

        # GATDataset 생성 및 GAT 훈련
        gat_dataset = GATDataset(prescription_df=prescription_df, ddi_df=ddi_df)
        trainer.fit_gat(gat_dataset)  # fit_graph + auto-calibrate (Fix 2)

        # 가중치 최적화 (calibration split pairs 사용)
        if gat_dataset.pairs_calibration is not None and len(gat_dataset.pairs_calibration) > 0:
            logger.info("[Step 3c] 앙상블 가중치 최적화 (Recall ≥ %.2f)", self.config.recall_threshold)
            # calibration split의 tabular 데이터
            calib_pairs = gat_dataset.pairs_calibration
            # 약물 인덱스 → 약물 코드 역매핑
            idx_to_drug = {v: k for k, v in trainer._gat._graph_builder.drug_to_idx.items()}
            drug_pairs_calib = [
                (idx_to_drug.get(int(pair[0]), ""), idx_to_drug.get(int(pair[1]), ""))
                for pair in calib_pairs
            ]
            y_calib = calib_pairs[:, 2].astype(int)
            # tabular 피처: calibration 쌍에 해당하는 X, y (dataset에서 추출)
            # calibration 쌍의 약물쌍 인덱스 → tabular X는 dataset.X_val 사용 (간소화)
            # Note: 이상적으로는 calibration split의 tabular X가 필요하지만,
            # 현재 TrainDataset은 drug pair → row 매핑을 제공하지 않으므로
            # val split 데이터로 근사합니다.
            X_calib = dataset.X_val
            y_calib_tab = dataset.y_val
            try:
                trainer.optimize_weights(
                    X_calib=X_calib,
                    y_calib=y_calib_tab,
                    drug_pairs_calib=drug_pairs_calib[:len(X_calib)],
                    recall_threshold=self.config.recall_threshold,
                )
            except RuntimeError as e:
                logger.error("가중치 최적화 실패: %s — 균등 가중치 유지", e)
        else:
            logger.warning("calibration 쌍 없음 — 균등 가중치 사용")

    def _run_optuna(self, dataset: TrainDataset) -> None:
        """Optuna 하이퍼파라미터 최적화."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna 미설치. 기본 하이퍼파라미터 사용.")
            return

        from .hyperparams import xgb_search_space, lgb_search_space

        def objective(trial):
            if self.config.model_type == "lightgbm":
                search = lgb_search_space(trial)
                params = {**self.config.lgb_params, **search}
                from .trainer import LGBMTrainer
                trainer = LGBMTrainer(params, self.config)
            else:
                search = xgb_search_space(trial)
                params = {**self.config.xgb_params, **search}
                from .trainer import XGBoostTrainer
                trainer = XGBoostTrainer(params, self.config)

            trainer.fit(dataset)
            proba = trainer.predict_proba(dataset.X_val)
            from .evaluator import find_optimal_threshold
            _, res = find_optimal_threshold(dataset.y_val, proba, self.config.recall_threshold)
            # Recall 우선: Recall < 목표면 페널티
            if res.recall < self.config.recall_threshold:
                return res.recall - 1.0  # 페널티
            return res.auc_roc

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=self.config.optuna_trials,
            timeout=self.config.optuna_timeout,
        )

        best = study.best_params
        logger.info("Optuna 최적 파라미터: %s (score=%.4f)", best, study.best_value)
        if self.config.model_type == "lightgbm":
            self.config.lgb_params.update(best)
        else:
            self.config.xgb_params.update(best)


def run_training(
    partition: str,
    model_type: str = "xgboost",
    feature_base: str = "data/features",
    model_dir: str = "models",
    use_optuna: bool = False,
    **kwargs,
) -> TrainResult:
    """편의 함수: TrainConfig 생성 후 파이프라인 실행."""
    config = TrainConfig(
        model_type=model_type,
        partition=partition,
        feature_base=feature_base,
        model_dir=model_dir,
        use_optuna=use_optuna,
        **{k: v for k, v in kwargs.items() if hasattr(TrainConfig, k)},
    )
    pipeline = TrainPipeline(config)
    return pipeline.run(partition)
