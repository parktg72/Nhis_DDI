"""M-2 — 번들 재현 정보 스탬프.

배포 번들에 입력 파일 목록·SHA-256·기간·코드 참조가 없어서 학습 코호트를
재현할 수 없었다(개선계획 B-1). 여기서 고정하는 것은 그 스탬프의 계약이다.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.ops.bundle_provenance import (
    PROVENANCE_SCHEMA_VERSION,
    collect_provenance,
    normalized_digest,
    stamp_bundle,
)


def _raw(tmp_path: Path, name: str, body: bytes = b"x") -> Path:
    p = tmp_path / name
    p.write_bytes(body)
    return p


# ── 입력 파일 기록 ────────────────────────────────────────────────────────
def test_input_files_record_name_size_and_sha256(tmp_path):
    a = _raw(tmp_path, "records_20240701.parquet", b"aaa")
    b = _raw(tmp_path, "records_20240702.parquet", b"bbbb")

    prov = collect_provenance([a, b], code_root=tmp_path)

    assert prov["input_file_count"] == 2
    got = {f["name"]: f for f in prov["input_files"]}
    assert got["records_20240701.parquet"]["size"] == 3
    assert got["records_20240702.parquet"]["size"] == 4
    assert got["records_20240701.parquet"]["sha256"] == hashlib.sha256(b"aaa").hexdigest()


def test_input_files_are_sorted_so_the_stamp_is_order_independent(tmp_path):
    a = _raw(tmp_path, "records_20240702.parquet", b"a")
    b = _raw(tmp_path, "records_20240701.parquet", b"b")

    first = collect_provenance([a, b], code_root=tmp_path)
    second = collect_provenance([b, a], code_root=tmp_path)

    assert [f["name"] for f in first["input_files"]] == [
        "records_20240701.parquet", "records_20240702.parquet",
    ]
    assert first["input_files"] == second["input_files"]


def test_sha256_is_streamed_not_whole_file(tmp_path, monkeypatch):
    """대용량 raw 를 통째 메모리에 올리지 않는다 — read_bytes 금지."""
    p = _raw(tmp_path, "records_20240701.parquet", b"z" * 200_000)

    def _boom(self, *a, **k):  # pragma: no cover - 호출되면 실패
        raise AssertionError("read_bytes 로 전체를 읽었다")

    monkeypatch.setattr(Path, "read_bytes", _boom)
    prov = collect_provenance([p], code_root=tmp_path)

    assert prov["input_files"][0]["sha256"] == hashlib.sha256(b"z" * 200_000).hexdigest()


def test_missing_input_file_raises_before_training(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_provenance([tmp_path / "없는파일.parquet"], code_root=tmp_path)


# ── 기간 ──────────────────────────────────────────────────────────────────
def test_period_comes_from_filenames_and_says_so(tmp_path):
    paths = [_raw(tmp_path, n) for n in
             ("records_20240701.parquet", "records_20241130.parquet")]

    prov = collect_provenance(paths, code_root=tmp_path)

    assert prov["period"] == {"from": "20240701", "to": "20241130", "source": "filename"}


def test_period_accepts_month_granularity(tmp_path):
    paths = [_raw(tmp_path, n) for n in
             ("records_202407.parquet", "records_202411.parquet")]

    prov = collect_provenance(paths, code_root=tmp_path)

    assert prov["period"]["from"] == "202407"
    assert prov["period"]["to"] == "202411"


def test_mixed_granularity_leaves_period_null(tmp_path):
    """월 파일과 일 파일을 섞으면 문자열 비교가 틀린 기간을 만든다."""
    paths = [_raw(tmp_path, "records_202407.parquet"),
             _raw(tmp_path, "records_20241130.parquet")]

    prov = collect_provenance(paths, code_root=tmp_path)

    assert prov["period"] is None
    assert prov["period_reason"] == "mixed_granularity"


def test_unparseable_filename_leaves_period_null_with_a_reason(tmp_path):
    paths = [_raw(tmp_path, "records_20240701.parquet"),
             _raw(tmp_path, "cohort_v2.parquet")]

    prov = collect_provenance(paths, code_root=tmp_path)

    assert prov["period"] is None
    assert prov["period_reason"] == "unparseable_filename"


# ── 코호트 파라미터 ───────────────────────────────────────────────────────
def test_cohort_params_are_recorded(tmp_path):
    p = _raw(tmp_path, "records_20240701.parquet")

    prov = collect_provenance(
        [p], code_root=tmp_path,
        glob_patterns=["records_20240[7-9]*.parquet"],
        seed=42, window_days=90, poly_threshold=5,
    )

    assert prov["cohort_params"] == {
        "seed": 42, "window_days": 90, "poly_threshold": 5,
    }
    assert prov["glob_patterns"] == ["records_20240[7-9]*.parquet"]


# ── 코드 참조 ─────────────────────────────────────────────────────────────
def test_code_ref_falls_back_to_fingerprints_when_git_is_absent(tmp_path, monkeypatch):
    """운영 PC 에는 .git 이 없다. 그때도 무엇으로 만들었는지 남아야 한다."""
    src = tmp_path / "serving"
    src.mkdir()
    (src / "predictor.py").write_text("print('x')\n", encoding="utf-8")

    def _no_git(*a, **k):
        raise FileNotFoundError("git 없음")

    monkeypatch.setattr(subprocess, "run", _no_git)
    prov = collect_provenance(
        [_raw(tmp_path, "records_20240701.parquet")], code_root=tmp_path,
    )

    assert prov["code"]["source"] == "none"
    assert prov["code"]["commit"] is None
    assert prov["code"]["fingerprints"]["serving/predictor.py"] is not None


def test_absent_source_file_is_recorded_as_null_not_omitted(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))

    prov = collect_provenance(
        [_raw(tmp_path, "records_20240701.parquet")], code_root=tmp_path,
    )

    assert "serving/predictor.py" in prov["code"]["fingerprints"]
    assert prov["code"]["fingerprints"]["serving/predictor.py"] is None


def test_fingerprint_matches_the_a0_baseline_vocabulary(tmp_path):
    """a0 의 지문표와 같은 숫자여야 대조가 성립한다. 두 구현이 갈라지면 실패."""
    from scripts.ops.a0_baseline_check import digest as a0_digest

    f = tmp_path / "x.py"
    f.write_bytes(b"line1\r\nline2\r\n")

    assert normalized_digest(f) == a0_digest(f, normalize=True)[:16]


# ── JSON 계약 ─────────────────────────────────────────────────────────────
def test_stamp_adds_exactly_one_key_and_keeps_the_rest(tmp_path):
    meta = {"clinical_standards_version": "v1", "stage1_sha256": "abc",
            "feature_cols": ["a", "b"], "thresholds": {"tau_red": 0.7}}
    (tmp_path / "stage_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    stamp_bundle(tmp_path, collect_provenance(
        [_raw(tmp_path, "records_20240701.parquet")], code_root=tmp_path))

    after = json.loads((tmp_path / "stage_meta.json").read_text(encoding="utf-8"))
    assert set(after) - set(meta) == {"provenance"}
    for k, v in meta.items():
        assert after[k] == v


def test_stamp_preserves_non_ascii_meta_values(tmp_path):
    """스탬프는 기존 값을 그대로 둔다 — 한글 값이 있어도 깨지지 않는다."""
    meta = {"clinical_standards_version": "v1", "note": "한글 값"}
    (tmp_path / "stage_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    stamp_bundle(tmp_path, collect_provenance(
        [_raw(tmp_path, "records_20240701.parquet")], code_root=tmp_path))

    after = json.loads((tmp_path / "stage_meta.json").read_text(encoding="utf-8"))
    assert after["note"] == "한글 값"


def test_stamp_json_is_ascii_only_so_a_cp949_console_can_read_it(tmp_path):
    (tmp_path / "stage_meta.json").write_text("{}", encoding="utf-8")

    stamp_bundle(tmp_path, collect_provenance(
        [_raw(tmp_path, "cohort.parquet")], code_root=tmp_path))

    prov = json.loads((tmp_path / "stage_meta.json").read_text(encoding="utf-8"))["provenance"]
    json.dumps(prov).encode("ascii")  # 비ASCII 가 섞이면 여기서 실패


def test_stamp_without_meta_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        stamp_bundle(tmp_path, {"schema_version": PROVENANCE_SCHEMA_VERSION})


def test_schema_version_is_recorded(tmp_path):
    prov = collect_provenance(
        [_raw(tmp_path, "records_20240701.parquet")], code_root=tmp_path)
    assert prov["schema_version"] == PROVENANCE_SCHEMA_VERSION


# ── 기록하지 않는 것 ──────────────────────────────────────────────────────
def test_no_row_level_content_is_recorded(tmp_path):
    """파일 이름·크기·해시만 남긴다. 행 수·환자 식별자는 남기지 않는다."""
    prov = collect_provenance(
        [_raw(tmp_path, "records_20240701.parquet")], code_root=tmp_path)

    assert set(prov["input_files"][0]) == {"name", "size", "sha256"}
    assert "n_rows" not in json.dumps(prov)
    assert "patient" not in json.dumps(prov)
