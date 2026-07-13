"""
scripts/etl - DDI 모델 ETL 패키지

주요 공개 API:
  - ETLPipeline : 전체 파이프라인 오케스트레이터
  - run_pipeline : 편의 함수
  - SampleFactory : 합성 테스트 데이터 생성
  - PatientFeatures, PipelineResult : 결과 데이터클래스
"""
from .models import (
    DrugOverlapPair,
    PatientFeatures,
    PipelineResult,
    PrescriptionRecord,
    QualityReport,
    ValidationResult,
)
from .pipeline import ETLPipeline, run_pipeline
from .sample_factory import make_edi_atc_map, make_t20_t30, make_t40, make_t50

__all__ = [
    # 파이프라인
    "ETLPipeline",
    "run_pipeline",
    # 샘플 데이터
    "make_t20_t30",
    "make_t40",
    "make_t50",
    "make_edi_atc_map",
    # 데이터 모델
    "PatientFeatures",
    "PipelineResult",
    "PrescriptionRecord",
    "DrugOverlapPair",
    "ValidationResult",
    "QualityReport",
]
