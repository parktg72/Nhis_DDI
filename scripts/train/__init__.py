"""
scripts/train - DDI 모델 ML 훈련 패키지 (Phase 2)

주요 공개 API:
  - run_training      : 편의 함수 (config 없이 바로 실행)
  - TrainPipeline     : 전체 훈련 파이프라인
  - TrainConfig       : 훈련 설정
  - XGBoostTrainer    : XGBoost 훈련기
  - LGBMTrainer       : LightGBM 훈련기
  - EnsembleTrainer   : XGBoost+LightGBM 앙상블
  - TrainDataset      : 데이터셋 (train/val/test 분할)
  - EvalResult        : 평가 결과 데이터클래스
"""
from .dataset import TrainDataset, load_dataset, load_dataset_from_df
from .evaluator import EvalResult, compute_metrics, find_optimal_threshold
from .experiment import ExperimentTracker
from .hyperparams import TrainConfig, XGB_DEFAULT, LGB_DEFAULT
from .pipeline import TrainPipeline, TrainResult, run_training
from .trainer import BaseTrainer, XGBoostTrainer, LGBMTrainer, EnsembleTrainer, build_trainer

__all__ = [
    # 파이프라인
    "run_training",
    "TrainPipeline",
    "TrainResult",
    # 설정
    "TrainConfig",
    "XGB_DEFAULT",
    "LGB_DEFAULT",
    # 훈련기
    "BaseTrainer",
    "XGBoostTrainer",
    "LGBMTrainer",
    "EnsembleTrainer",
    "build_trainer",
    # 데이터셋
    "TrainDataset",
    "load_dataset",
    "load_dataset_from_df",
    # 평가
    "EvalResult",
    "compute_metrics",
    "find_optimal_threshold",
    # 실험
    "ExperimentTracker",
]
