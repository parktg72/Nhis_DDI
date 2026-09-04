"""A0 ③ 이 번들의 재현 정보를 읽는지 (M-2 의 읽는 쪽).

M-2 는 쓰는 쪽만으로 닫히지 않는다. A0 가 "메타에 입력 파일 목록·기간·코드
커밋이 없다" 고 계속 경고하면 스탬프가 있어도 운영자는 없다고 읽는다.
"""
from __future__ import annotations

import json

import pytest

from scripts.ops import a0_baseline_check as a0
from scripts.ops.bundle_provenance import collect_provenance, stamp_bundle

_ABSENT_WARNING = "메타에 입력 파일 목록·기간·코드 커밋이 없다"


def _bundle(tmp_path, *, with_provenance: bool):
    b = tmp_path / "bundle"
    b.mkdir()
    for name in ("stage1_red.joblib", "stage2_yellow.joblib"):
        (b / name).write_bytes(name.encode())
    meta = {
        "clinical_standards_version": "v1",
        "thresholds": {"tau_red": 0.7, "tau_review": 0.3},
        "stage1_sha256": a0.digest(b / "stage1_red.joblib", normalize=False),
        "stage2_sha256": a0.digest(b / "stage2_yellow.joblib", normalize=False),
    }
    (b / "stage_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if with_provenance:
        raw = tmp_path / "records_20240701.parquet"
        raw.write_bytes(b"raw")
        stamp_bundle(b, collect_provenance([raw], code_root=tmp_path))
    return b


@pytest.fixture(autouse=True)
def _reset_out(monkeypatch):
    monkeypatch.setattr(a0, "OUT", [])


def _run(tmp_path, monkeypatch, bundle):
    monkeypatch.setenv("HIERARCHICAL_MODEL_DIR", str(bundle))
    verdict = a0.check_bundle(tmp_path)
    return verdict, "\n".join(a0.OUT)


def test_provenance_is_printed_and_the_absent_warning_is_dropped(tmp_path, monkeypatch):
    verdict, out = _run(tmp_path, monkeypatch, _bundle(tmp_path, with_provenance=True))

    assert "재현 정보 있음" in verdict
    assert _ABSENT_WARNING not in out
    assert "records_20240701" in out or "입력 1개 파일" in out
    assert "20240701" in out


def test_a_bundle_without_the_stamp_still_warns(tmp_path, monkeypatch):
    verdict, out = _run(tmp_path, monkeypatch, _bundle(tmp_path, with_provenance=False))

    assert "재현 정보 없음" in verdict
    assert _ABSENT_WARNING in out


def test_the_stamp_does_not_mask_a_bundle_mismatch(tmp_path, monkeypatch):
    """재현 정보가 있다고 해서 SHA 불일치 판정이 흐려지면 안 된다."""
    b = _bundle(tmp_path, with_provenance=True)
    (b / "stage1_red.joblib").write_bytes(b"tampered")

    verdict, _ = _run(tmp_path, monkeypatch, b)

    assert verdict.startswith("불일치")
    assert "재현 정보 있음" in verdict
