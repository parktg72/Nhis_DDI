"""hana_app.core.ml_runner.stratified_sample_from_parquet 회귀 테스트.

DuckDB parameterized query 에 numpy.int64 을 그대로 전달하면
NotImplementedException 이 발생. .item() 으로 Python native 로 변환 후
전달해야 함.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hana_app.core.ml_runner import stratified_sample_from_parquet


@pytest.fixture
def imbalanced_parquet(tmp_path):
    """극단 불균형 파일럿 데이터 (Yellow 500, Green 5) — numpy.int64 라벨."""
    import numpy as np
    n_yellow, n_green = 500, 5
    df = pd.DataFrame({
        "patient_id": [f"P{i:06d}" for i in range(n_yellow + n_green)],
        # risk_label 은 RISK_LABEL_MAP 값 (int) — pandas/numpy.int64 로 저장됨
        "risk_label": np.array([2] * n_yellow + [1] * n_green, dtype=np.int64),
        "feat1": np.random.default_rng(42).random(n_yellow + n_green),
    })
    # sanity check — 실제 프로덕션 데이터와 동일하게 numpy.int64 로 저장됨
    assert df["risk_label"].dtype == np.int64

    parquet_path = tmp_path / "features.parquet"
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def test_stratified_sample_handles_numpy_int64_labels(imbalanced_parquet):
    """회귀: Parquet 파일의 numpy.int64 라벨로 DuckDB 바인딩 시 에러 나지 않아야 함.

    이전 버그: `cls_val = row[target_col]` 가 numpy.int64 를 반환하고
    DuckDB `execute(..., [cls_val])` 가 NotImplementedException 발생.
    수정: `cls_val = raw_val.item() if hasattr(raw_val, 'item') else raw_val`.
    """
    # sample_size = 400 (총 505 중 ~80%) 이면 Green 5 * 0.79 = 3.96 → 3건 샘플링
    result = stratified_sample_from_parquet(
        parquet_paths=imbalanced_parquet,
        target_col="risk_label",
        sample_size=400,
        seed=42,
        memory_limit_mb=512,
    )

    assert not result.empty
    assert "risk_label" in result.columns
    # 두 클래스 모두 표본에 포함 (numpy.int64 바인딩 성공 증거)
    assert set(result["risk_label"].unique()) == {1, 2}
    # sample_size 대략 준수 (반올림 오차)
    assert abs(len(result) - 400) <= 2


def test_stratified_sample_small_population_returns_all(imbalanced_parquet):
    """전체 크기가 sample_size 이하면 전부 반환 (기존 동작 보존)."""
    result = stratified_sample_from_parquet(
        parquet_paths=imbalanced_parquet,
        target_col="risk_label",
        sample_size=10_000,  # 총 505 < 10000
        seed=42,
        memory_limit_mb=512,
    )
    assert len(result) == 505
