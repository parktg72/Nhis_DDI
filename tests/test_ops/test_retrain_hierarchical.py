"""scripts.ops.retrain_hierarchical — 헤드리스 계층 재학습 오케스트레이션 테스트.

build_patient_features_from_parquet(무거운 실 피처계산)는 monkeypatch 로 합성
PatientFeatures 를 반환시키고, 스크립트가 df 조립 → train_hierarchical → 7-class 번들
산출을 올바르게 수행하는지(앱 경로 미러링) 검증한다.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ops import retrain_hierarchical as rh
from hana_app.core.hierarchical_runner import STAGE2_LABELS
from scripts.etl.models import PatientFeatures
from scripts.etl.prescription_aggregator import _assign_risk_level, _assign_yellow_subtype


def _feat(idx: int, **kw) -> PatientFeatures:
    base = dict(
        patient_id=f"P{idx:05d}",
        window_start=date(2024, 7, 1),
        window_end=date(2024, 9, 30),
    )
    base.update(kw)
    f = PatientFeatures(**base)
    _assign_risk_level(f)
    _assign_yellow_subtype(f)
    return f


def _synthetic_cohort() -> list[PatientFeatures]:
    """7-class 전부 + Red/Normal 을 커버하는 합성 코호트."""
    feats: list[PatientFeatures] = []
    n = 0

    def add(count, **kw):
        nonlocal n
        kw.setdefault("drug_count", 6)
        for _ in range(count):
            feats.append(_feat(n, **kw))
            n += 1

    add(30, ddi_contraindicated=1)                                        # Red
    add(30, ddi_major=1, dup_same_ingredient=1, institution_count=3)     # Y_TRIPLE (3 dims)
    add(30, ddi_major=1, dup_same_ingredient=1)                          # Y_DOUBLE (2 dims)
    add(30, ddi_major=1)                                                  # Y_DDI_MAJOR
    add(30, ddi_moderate=2)                                               # Y_DDI_MOD
    add(30, dup_same_ingredient=1)                                        # Y_DUP
    add(30, institution_count=3)                                          # Y_FRAG
    add(60, drug_count=2)                                                 # Normal → No_Alert
    return feats


def test_retrain_produces_7class_bundle_from_parquet_paths(tmp_path, monkeypatch):
    """실제 계약: build_patient_features_from_parquet 는 피처 배치 Parquet 경로를 반환.

    mock 이 _patient_features_to_row 스키마 parquet 를 써서 list[Path] 를 반환 →
    스크립트의 pd.read_parquet 분기(실 풀런 경로)를 행사한다.
    """
    import pandas as pd
    from hana_app.core.ml_runner import _patient_features_to_row

    cohort = _synthetic_cohort()
    feat_path = tmp_path / "features_batch_0000.parquet"
    pd.DataFrame([_patient_features_to_row(f) for f in cohort]).to_parquet(feat_path, index=False)
    monkeypatch.setattr(rh, "build_patient_features_from_parquet", lambda **kw: [feat_path])

    out = tmp_path / "bundle"
    result = rh.retrain_hierarchical(
        raw_paths=[tmp_path / "records_dummy.parquet"],
        output_dir=out,
        seed=7,
        log_cb=lambda *_a, **_k: None,
    )

    assert (out / "stage1_red.joblib").exists()
    assert (out / "stage2_yellow.joblib").exists()
    meta = json.loads((out / "stage_meta.json").read_text())
    # 라벨 공간 = 현재 7-class (서빙 가드 통과 조건)
    assert meta["stage2_labels"] == list(STAGE2_LABELS)
    assert len(STAGE2_LABELS) == 7
    assert result["n_patients"] == len(cohort)
    counts = meta["stage2_label_counts"]
    assert counts.get("Y_TRIPLE", 0) > 0
    assert counts.get("Y_DOUBLE", 0) > 0


def test_retrain_handles_patient_features_list(tmp_path, monkeypatch):
    """방어적 분기: 빌더가 list[PatientFeatures] 를 반환해도 동작."""
    cohort = _synthetic_cohort()
    monkeypatch.setattr(rh, "build_patient_features_from_parquet", lambda **kw: cohort)
    out = tmp_path / "bundle2"
    result = rh.retrain_hierarchical(
        raw_paths=[tmp_path / "records_dummy.parquet"],
        output_dir=out,
        seed=7,
        log_cb=lambda *_a, **_k: None,
    )
    meta = json.loads((out / "stage_meta.json").read_text())
    assert meta["stage2_labels"] == list(STAGE2_LABELS)
    assert result["n_patients"] == len(cohort)


def test_collect_raw_paths_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        rh.collect_raw_paths(tmp_path / "nope")


def test_collect_raw_paths_no_match(tmp_path):
    (tmp_path / "other.parquet").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        rh.collect_raw_paths(tmp_path, "records_*.parquet")


def test_collect_raw_paths_sorted(tmp_path):
    for name in ("records_b.parquet", "records_a.parquet"):
        (tmp_path / name).write_bytes(b"")
    paths = rh.collect_raw_paths(tmp_path)
    assert [p.name for p in paths] == ["records_a.parquet", "records_b.parquet"]


def test_collect_raw_paths_multiple_globs(tmp_path):
    """다중 glob 패턴 합집합·dedupe (윈도우 학습: 07~11 = Dec 제외)."""
    for name in ("records_202407.parquet", "records_202411.parquet",
                 "records_202412.parquet"):
        (tmp_path / name).write_bytes(b"")
    paths = rh.collect_raw_paths(
        tmp_path, ["records_20240[7-9]*.parquet", "records_20241[01]*.parquet"])
    names = [p.name for p in paths]
    assert names == ["records_202407.parquet", "records_202411.parquet"]  # Dec 제외
    # 패턴 겹쳐도 dedupe
    dup = rh.collect_raw_paths(tmp_path, ["records_*.parquet", "records_202407.parquet"])
    assert len(dup) == 3
