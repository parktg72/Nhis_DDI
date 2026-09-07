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


# ── A0 ⑤ 관측 노출 상태 (3차 검토 반영) ─────────────────────────────────
#
# 종전 구현은 `process_start_time_seconds` 로 worker 수를 셌다. 그 계열은 /proc
# 기반이라 **Linux 전용**이고, 운영 대상인 Windows 파이썬에서는 아예 노출되지
# 않는다. 배포 대상에서 못 도는 검사였다. 실행 명령을 읽는 방식으로 바꿨다.


def _launcher(tmp_path, name, line):
    (tmp_path / name).write_text(line, encoding="utf-8")


def test_worker_setting_is_read_from_the_launcher(tmp_path):
    _launcher(tmp_path, "run.bat", 'python -m uvicorn serving.main:app --workers 4\n')

    assert a0.find_worker_setting(tmp_path) == [("run.bat", "4")]


def test_a_launcher_without_the_flag_is_reported_as_default_one(tmp_path):
    _launcher(tmp_path, "run.bat", "python -m uvicorn serving.main:app --port 8000\n")

    assert a0.find_worker_setting(tmp_path) == [("run.bat", "미지정(기본 1)")]


def test_multiple_workers_are_called_out_as_not_an_aggregate(tmp_path):
    _launcher(tmp_path, "run.bat", "uvicorn serving.main:app --workers 4\n")

    v = a0.check_exposition("http://x", None, tmp_path)

    assert "worker 4개 설정 발견" in v
    assert "집계가 아니다" in v


def test_a_single_worker_is_reported_as_safe_to_read(tmp_path):
    _launcher(tmp_path, "run.bat", "uvicorn serving.main:app\n")

    assert "worker 1개 설정" in a0.check_exposition("http://x", None, tmp_path)


def test_no_launcher_is_reported_as_unknown_not_as_fine(tmp_path):
    """찾지 못한 것을 조용히 통과시키면 확인했다고 오해된다."""
    v = a0.check_exposition("http://x", None, tmp_path)

    assert "worker 미확인" in v


def test_without_a_key_the_exposition_itself_is_not_claimed_checked(tmp_path):
    _launcher(tmp_path, "run.bat", "uvicorn serving.main:app\n")

    assert "노출 경로 미확인" in a0.check_exposition("http://x", None, tmp_path)


def test_exposition_without_ddi_series_is_flagged(tmp_path, monkeypatch):
    """노출은 열렸는데 ddi_* 가 없으면 배선이 없는 것이다 — 통과로 읽히면 안 된다."""
    import urllib.request

    class _R:
        def read(self):
            return b"# TYPE python_info gauge\npython_info 1.0\n"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _R())
    _launcher(tmp_path, "run.bat", "uvicorn serving.main:app\n")

    v = a0.check_exposition("http://x", "k", tmp_path)

    assert "ddi_* 없음" in v
