"""Stage 2 전용 층화 샘플링 (risk_level != Red prefilter + yellow_subtype 6-class)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hana_app.core.hierarchical_runner import stratified_sample_stage2


def _make_parquet(tmp_path: Path) -> Path:
    """Red 10 + 각 Yellow 서브라벨 20 + Normal 200 의 혼합 parquet."""
    rows = []
    for i in range(10):
        rows.append({"patient_id": f"R{i}", "risk_level": "Red",
                     "yellow_subtype": None, "feat1": 0.5})
    for sub in ("Y_MIX", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG"):
        for i in range(20):
            rows.append({"patient_id": f"{sub}_{i}", "risk_level": "Yellow",
                         "yellow_subtype": sub, "feat1": 0.5})
    for i in range(200):
        rows.append({"patient_id": f"N{i}", "risk_level": "Normal",
                     "yellow_subtype": None, "feat1": 0.5})
    df = pd.DataFrame(rows)
    out = tmp_path / "features.parquet"
    df.to_parquet(out, index=False)
    return out


def test_stage2_sampling_excludes_red(tmp_path):
    parquet = _make_parquet(tmp_path)
    sample = stratified_sample_stage2(parquet, sample_size=100, seed=42)
    assert len(sample) > 0
    assert (sample["risk_level"] != "Red").all()


def test_stage2_sampling_includes_no_alert_class(tmp_path):
    """No_Alert (Green/Normal) 도 stage2 클래스로 포함."""
    parquet = _make_parquet(tmp_path)
    sample = stratified_sample_stage2(parquet, sample_size=100, seed=42)
    assert "stage2_label" in sample.columns
    assert "No_Alert" in set(sample["stage2_label"].unique())


def test_stage2_sampling_covers_all_yellow_subtypes(tmp_path):
    parquet = _make_parquet(tmp_path)
    sample = stratified_sample_stage2(parquet, sample_size=100, seed=42)
    subtypes_in_sample = set(sample["stage2_label"].unique()) - {"No_Alert"}
    # 5 Yellow 서브라벨 모두 최소 1건 이상 (각 20건 모집단에서 층화 추출)
    assert subtypes_in_sample == {"Y_MIX", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG"}


def test_stage2_sampling_excludes_y_other(tmp_path):
    """Y_OTHER 는 학습셋에서 제외."""
    df = pd.DataFrame([
        {"patient_id": f"Y{i}", "risk_level": "Yellow",
         "yellow_subtype": "Y_OTHER" if i < 5 else "Y_MIX",
         "feat1": 0.5}
        for i in range(20)
    ])
    p = tmp_path / "features.parquet"
    df.to_parquet(p, index=False)
    sample = stratified_sample_stage2(p, sample_size=100, seed=42)
    assert "Y_OTHER" not in set(sample["stage2_label"].unique())
    assert ~sample["yellow_subtype"].fillna("").eq("Y_OTHER").any()


def test_stage2_sampling_reproducible_with_seed(tmp_path):
    parquet = _make_parquet(tmp_path)
    s1 = stratified_sample_stage2(parquet, sample_size=80, seed=42)
    s2 = stratified_sample_stage2(parquet, sample_size=80, seed=42)
    pd.testing.assert_frame_equal(
        s1.sort_values("patient_id").reset_index(drop=True),
        s2.sort_values("patient_id").reset_index(drop=True),
    )


def test_stage2_sampling_raises_on_yellow_with_null_subtype(tmp_path):
    """데이터 품질 결함: Yellow 인데 yellow_subtype 이 None → build_stage2_label 로 ValueError."""
    import pytest
    df = pd.DataFrame([
        {"patient_id": "P001", "risk_level": "Yellow",
         "yellow_subtype": None, "feat1": 0.5},   # 데이터 결함
        {"patient_id": "P002", "risk_level": "Normal",
         "yellow_subtype": None, "feat1": 0.5},
    ])
    p = tmp_path / "features.parquet"
    df.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="yellow_subtype"):
        stratified_sample_stage2(p, sample_size=100, seed=42)
