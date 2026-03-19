"""
Prometheus 메트릭 정의 및 수집기

prometheus_client 미설치 환경에서도 import 가능 (lazy import).
설치 없이 실행 시 InMemoryMetrics(dict 기반)로 폴백.

메트릭 목록:
  - ddi_prediction_total        : 예측 횟수 (risk_level, source 레이블)
  - ddi_prediction_latency_ms   : 예측 지연시간 히스토그램
  - ddi_rule_ml_disagree_total  : Rule vs ML 불일치 횟수
  - ddi_psi_score               : 피처별 PSI 값
  - ddi_model_recall            : 서브그룹별 Recall
  - ddi_model_precision         : 서브그룹별 Precision
  - ddi_batch_size              : 배치 예측 크기
  - ddi_high_risk_rate          : 고위험(Red) 비율
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus 클라이언트 lazy import
# ─────────────────────────────────────────────────────────────────────────────

try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, push_to_gateway
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.info("prometheus_client 미설치 — InMemoryMetrics 폴백 모드로 동작")


# ─────────────────────────────────────────────────────────────────────────────
# 인메모리 폴백 메트릭 (prometheus_client 없을 때)
# ─────────────────────────────────────────────────────────────────────────────

class _InMemoryCounter:
    def __init__(self, name: str, description: str, labelnames: list[str] = ()):
        self.name = name
        self._data: Dict[tuple, float] = defaultdict(float)
        self._labelnames = list(labelnames)

    def labels(self, **kwargs) -> "_InMemoryCounter":
        self._current_labels = tuple(kwargs.get(k, "") for k in self._labelnames)
        return self

    def inc(self, amount: float = 1.0) -> None:
        key = getattr(self, "_current_labels", ())
        self._data[key] += amount

    def get(self, **kwargs) -> float:
        key = tuple(kwargs.get(k, "") for k in self._labelnames)
        return self._data.get(key, 0.0)

    def collect(self) -> dict:
        return dict(self._data)


class _InMemoryGauge(_InMemoryCounter):
    def set(self, value: float) -> None:
        key = getattr(self, "_current_labels", ())
        self._data[key] = value


class _InMemoryHistogram:
    def __init__(self, name: str, description: str, labelnames: list[str] = (), buckets=()):
        self.name = name
        self._observations: list[float] = []

    def labels(self, **kwargs) -> "_InMemoryHistogram":
        return self

    def observe(self, value: float) -> None:
        self._observations.append(value)

    def get_observations(self) -> list[float]:
        return list(self._observations)


# ─────────────────────────────────────────────────────────────────────────────
# 메트릭 레지스트리
# ─────────────────────────────────────────────────────────────────────────────

def _make_counter(name, description, labelnames=()):
    if _PROMETHEUS_AVAILABLE:
        return Counter(name, description, labelnames)
    return _InMemoryCounter(name, description, labelnames)


def _make_gauge(name, description, labelnames=()):
    if _PROMETHEUS_AVAILABLE:
        return Gauge(name, description, labelnames)
    return _InMemoryGauge(name, description, labelnames)


def _make_histogram(name, description, labelnames=(), buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000)):
    if _PROMETHEUS_AVAILABLE:
        return Histogram(name, description, labelnames, buckets=buckets)
    return _InMemoryHistogram(name, description, labelnames, buckets)


# ─────────────────────────────────────────────────────────────────────────────
# 전역 메트릭 인스턴스 (싱글턴)
# ─────────────────────────────────────────────────────────────────────────────

PREDICTION_TOTAL = _make_counter(
    "ddi_prediction_total",
    "DDI 위험도 예측 총 횟수",
    ["risk_level", "source"],   # source: api | batch
)

PREDICTION_LATENCY = _make_histogram(
    "ddi_prediction_latency_ms",
    "예측 지연시간 (ms)",
    ["source"],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000),
)

RULE_ML_DISAGREE = _make_counter(
    "ddi_rule_ml_disagree_total",
    "Rule 등급 vs ML 등급 불일치 횟수",
    ["rule_level", "ml_level"],
)

PSI_SCORE = _make_gauge(
    "ddi_psi_score",
    "피처별 PSI (Population Stability Index)",
    ["feature_name"],
)

MODEL_RECALL = _make_gauge(
    "ddi_model_recall",
    "모델 고위험 Recall",
    ["subgroup"],   # all | age_75plus | female | male
)

MODEL_PRECISION = _make_gauge(
    "ddi_model_precision",
    "모델 고위험 Precision",
    ["subgroup"],
)

BATCH_SIZE = _make_histogram(
    "ddi_batch_size",
    "배치 예측 요청 크기",
    [],
    buckets=(1, 10, 50, 100, 250, 500, 1000),
)

HIGH_RISK_RATE = _make_gauge(
    "ddi_high_risk_rate",
    "고위험(Red) 환자 비율",
    ["partition"],  # YYYYMMDD
)


# ─────────────────────────────────────────────────────────────────────────────
# 편의 함수
# ─────────────────────────────────────────────────────────────────────────────

def record_prediction(
    risk_level: str,
    source: str,
    latency_ms: float,
    rule_level: Optional[str] = None,
    ml_level: Optional[str] = None,
) -> None:
    """단건 예측 결과 메트릭 기록."""
    PREDICTION_TOTAL.labels(risk_level=risk_level, source=source).inc()
    PREDICTION_LATENCY.labels(source=source).observe(latency_ms)

    if rule_level and ml_level and rule_level != ml_level:
        RULE_ML_DISAGREE.labels(rule_level=rule_level, ml_level=ml_level).inc()


def record_psi(feature_name: str, psi_value: float) -> None:
    """PSI 값 업데이트."""
    PSI_SCORE.labels(feature_name=feature_name).set(psi_value)


def record_model_performance(recall: float, precision: float, subgroup: str = "all") -> None:
    """모델 성능 메트릭 업데이트."""
    MODEL_RECALL.labels(subgroup=subgroup).set(recall)
    MODEL_PRECISION.labels(subgroup=subgroup).set(precision)


def record_batch(
    n_patients: int,
    risk_distribution: Dict[str, int],
    partition: str,
) -> None:
    """배치 예측 결과 요약 기록."""
    BATCH_SIZE.observe(n_patients)
    total = max(n_patients, 1)
    red = risk_distribution.get("Red", 0)
    HIGH_RISK_RATE.labels(partition=partition).set(red / total)


# ─────────────────────────────────────────────────────────────────────────────
# Pushgateway 전송 (선택적)
# ─────────────────────────────────────────────────────────────────────────────

def push_metrics(gateway_url: str, job: str = "ddi_serving") -> None:
    """Prometheus Pushgateway로 메트릭 전송."""
    if not _PROMETHEUS_AVAILABLE:
        logger.warning("prometheus_client 미설치 — push 생략")
        return
    try:
        push_to_gateway(gateway_url, job=job, registry=CollectorRegistry())
        logger.info("메트릭 push 완료: %s", gateway_url)
    except Exception as exc:
        logger.warning("메트릭 push 실패 (무시): %s", exc)
