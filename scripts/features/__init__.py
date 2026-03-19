"""
scripts/features - DDI 모델 피처 엔지니어링 패키지

주요 공개 API:
  - build_ml_features   : 전체 피처 엔지니어링 실행 (편의 함수)
  - FeatureEngineer     : 통합 파이프라인 클래스
  - CYPFeatureExtractor : CYP 상호작용 피처 추출
  - FeatureNormalizer   : RobustScaler 기반 정규화
  - FeatureSelector     : 분산·상관관계 기반 피처 선택
"""
from .cyp_features import CYPFeatureExtractor, CYP_FEATURE_COLS
from .feature_engineer import FeatureEngineer, build_ml_features
from .normalizer import FeatureNormalizer
from .selector import FeatureSelector, PROTECTED_FEATURES
from .temporal_features import extract_temporal, TEMPORAL_FEATURE_COLS

__all__ = [
    "build_ml_features",
    "FeatureEngineer",
    "CYPFeatureExtractor",
    "CYP_FEATURE_COLS",
    "FeatureNormalizer",
    "FeatureSelector",
    "PROTECTED_FEATURES",
    "extract_temporal",
    "TEMPORAL_FEATURE_COLS",
]
