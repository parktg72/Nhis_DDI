"""M9 — 폐쇄망 반입 무결성 매니페스트.

반입 산출물(코드·휠하우스·참조 데이터)에 대한 무결성 검증 절차가 없다
(기술검토 P1-13). 이 도구는 그 절차의 기계 부분이다.

**이 도구가 증명하지 못하는 것을 먼저 고정한다** — 대상과 같은 매체에 실려 온
매니페스트로는 공급망 위조를 잡을 수 없다. 그래서 매니페스트 자신의 해시를
따로 내고, 대조는 다른 경로로 받은 값과 하도록 한다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.ops.intake_manifest import (
    EXIT_MISMATCH,
    EXIT_NO_MANIFEST,
    EXIT_OK,
    EXIT_UNLISTED,
    generate,
    manifest_digest,
    verify,
)


def _pkg(tmp_path, files=(("a.txt", b"aaa"), ("sub/b.txt", b"bb"))):
    root = tmp_path / "delivery"
    for name, body in files:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    return root


# ── 생성 ─────────────────────────────────────────────────────────────────
def test_manifest_lists_every_file_with_size_and_sha256(tmp_path):
    root = _pkg(tmp_path)
    out = tmp_path / "MANIFEST.json"

    generate(root, out)

    m = json.loads(out.read_text(encoding="utf-8"))
    entries = {e["path"]: e for e in m["files"]}
    assert set(entries) == {"a.txt", "sub/b.txt"}
    assert entries["a.txt"]["sha256"] == hashlib.sha256(b"aaa").hexdigest()
    assert entries["a.txt"]["size"] == 3


def test_paths_are_relative_and_slash_separated(tmp_path):
    """Windows 에서 만들고 Windows 에서 검증하지만, 경로 표기는 한 가지여야 한다."""
    generate(_pkg(tmp_path), tmp_path / "M.json")

    m = json.loads((tmp_path / "M.json").read_text(encoding="utf-8"))
    for e in m["files"]:
        assert "\\" not in e["path"]
        assert not Path(e["path"]).is_absolute()


def test_entries_are_sorted_so_two_runs_match(tmp_path):
    root = _pkg(tmp_path, [("z.txt", b"z"), ("a.txt", b"a"), ("m/n.txt", b"n")])
    generate(root, tmp_path / "1.json")
    generate(root, tmp_path / "2.json")

    a = json.loads((tmp_path / "1.json").read_text(encoding="utf-8"))["files"]
    b = json.loads((tmp_path / "2.json").read_text(encoding="utf-8"))["files"]
    assert a == b
    assert [e["path"] for e in a] == sorted(e["path"] for e in a)


def test_hashing_is_streamed(tmp_path, monkeypatch):
    """휠하우스는 크다. 통째로 읽으면 반입 PC 에서 죽는다."""
    root = tmp_path / "d"
    root.mkdir()
    (root / "big.bin").write_bytes(b"x" * 300_000)

    def _boom(self, *a, **k):  # pragma: no cover
        raise AssertionError("read_bytes 로 전체를 읽었다")

    monkeypatch.setattr(Path, "read_bytes", _boom)
    generate(root, tmp_path / "M.json")

    m = json.loads((tmp_path / "M.json").read_text(encoding="utf-8"))
    assert m["files"][0]["sha256"] == hashlib.sha256(b"x" * 300_000).hexdigest()


# ── 매니페스트 자신의 해시 ───────────────────────────────────────────────
def test_manifest_reports_its_own_digest(tmp_path, capsys):
    """대상과 같은 매체로 온 매니페스트는 그 자체로 신뢰할 수 없다.
    자기 해시를 내고 **다른 경로로 받은 값**과 대조하게 한다."""
    out = tmp_path / "M.json"
    generate(_pkg(tmp_path), out)

    d = manifest_digest(out)

    assert len(d) == 64
    assert d == hashlib.sha256(out.read_bytes()).hexdigest()


# ── 검증 ─────────────────────────────────────────────────────────────────
def test_unchanged_package_verifies(tmp_path):
    root = _pkg(tmp_path)
    generate(root, tmp_path / "M.json")

    assert verify(root, tmp_path / "M.json")["exit_code"] == EXIT_OK


def test_modified_file_is_reported(tmp_path):
    root = _pkg(tmp_path)
    generate(root, tmp_path / "M.json")
    (root / "a.txt").write_bytes(b"tampered")

    r = verify(root, tmp_path / "M.json")

    assert r["exit_code"] == EXIT_MISMATCH
    assert r["changed"] == ["a.txt"]


def test_missing_file_is_reported(tmp_path):
    root = _pkg(tmp_path)
    generate(root, tmp_path / "M.json")
    (root / "sub" / "b.txt").unlink()

    r = verify(root, tmp_path / "M.json")

    assert r["exit_code"] == EXIT_MISMATCH
    assert r["missing"] == ["sub/b.txt"]


def test_unlisted_file_is_reported_separately(tmp_path):
    """매니페스트에 없는 파일은 변조와 다른 사건이다 — 반입 절차의 누락일 수도 있다."""
    root = _pkg(tmp_path)
    generate(root, tmp_path / "M.json")
    (root / "extra.dll").write_bytes(b"?")

    r = verify(root, tmp_path / "M.json")

    assert r["exit_code"] == EXIT_UNLISTED
    assert r["unlisted"] == ["extra.dll"]


def test_tampering_outranks_an_unlisted_file(tmp_path):
    """둘 다면 더 나쁜 쪽으로 판정한다."""
    root = _pkg(tmp_path)
    generate(root, tmp_path / "M.json")
    (root / "a.txt").write_bytes(b"tampered")
    (root / "extra.dll").write_bytes(b"?")

    assert verify(root, tmp_path / "M.json")["exit_code"] == EXIT_MISMATCH


def test_absent_manifest_is_not_a_pass(tmp_path):
    """매니페스트가 없으면 '검증했는데 이상 없음' 이 아니라 '검증하지 못함' 이다."""
    r = verify(_pkg(tmp_path), tmp_path / "없음.json")

    assert r["exit_code"] == EXIT_NO_MANIFEST


def test_corrupt_manifest_is_not_a_pass(tmp_path):
    bad = tmp_path / "M.json"
    bad.write_text("{ this is not json", encoding="utf-8")

    assert verify(_pkg(tmp_path), bad)["exit_code"] == EXIT_NO_MANIFEST


def test_empty_package_is_not_silently_ok(tmp_path):
    """빈 디렉터리를 '전부 일치' 로 통과시키면 반입 실패가 초록불이 된다."""
    root = tmp_path / "empty"
    root.mkdir()
    generate(_pkg(tmp_path), tmp_path / "M.json")

    r = verify(root, tmp_path / "M.json")

    assert r["exit_code"] == EXIT_MISMATCH
    assert len(r["missing"]) == 2


# ── 기록하지 않는 것 ─────────────────────────────────────────────────────
def test_manifest_records_no_file_contents(tmp_path):
    root = _pkg(tmp_path, [("secret.txt", b"salt-value-should-not-appear")])
    generate(root, tmp_path / "M.json")

    text = (tmp_path / "M.json").read_text(encoding="utf-8")

    assert "salt-value" not in text
    assert set(json.loads(text)["files"][0]) == {"path", "size", "sha256"}


# ── 교차검토 1차 반영 ────────────────────────────────────────────────────

def test_the_documented_procedure_passes(tmp_path):
    """런북은 매니페스트를 **반입 폴더 안에** 넣으라고 한다. 그대로 하면 정상
    반입본이 통과해야 한다 — 종전에는 매니페스트 자신이 '목록 외' 로 잡혔다."""
    root = _pkg(tmp_path)
    inside = root / "MANIFEST.json"

    generate(root, inside)

    r = verify(root, inside)
    assert r["exit_code"] == EXIT_OK
    assert r["unlisted"] == []


def test_regenerating_in_place_is_stable(tmp_path):
    """폴더 안에서 다시 만들어도 이전 매니페스트를 해싱해 넣지 않는다."""
    root = _pkg(tmp_path)
    inside = root / "MANIFEST.json"
    generate(root, inside)
    generate(root, inside)

    assert verify(root, inside)["exit_code"] == EXIT_OK


def test_an_empty_manifest_never_passes(tmp_path):
    """빈 목록으로 빈 폴더를 통과시키면 반입 실패가 초록불이 된다."""
    root = tmp_path / "empty"
    root.mkdir()
    m = tmp_path / "M.json"
    m.write_text(json.dumps({"schema_version": 1, "file_count": 0, "files": []}),
                 encoding="utf-8")

    assert verify(root, m)["exit_code"] == EXIT_NO_MANIFEST


def test_file_count_must_match_the_list(tmp_path):
    """머리말과 목록이 어긋나면 잘린 매니페스트다."""
    root = _pkg(tmp_path)
    generate(root, tmp_path / "M.json")
    d = json.loads((tmp_path / "M.json").read_text(encoding="utf-8"))
    d["file_count"] = 99
    (tmp_path / "M.json").write_text(json.dumps(d), encoding="utf-8")

    assert verify(root, tmp_path / "M.json")["exit_code"] == EXIT_NO_MANIFEST


def test_unknown_schema_version_is_refused(tmp_path):
    root = _pkg(tmp_path)
    generate(root, tmp_path / "M.json")
    d = json.loads((tmp_path / "M.json").read_text(encoding="utf-8"))
    d["schema_version"] = 99
    (tmp_path / "M.json").write_text(json.dumps(d), encoding="utf-8")

    assert verify(root, tmp_path / "M.json")["exit_code"] == EXIT_NO_MANIFEST


def test_duplicate_paths_are_refused_not_silently_deduped(tmp_path):
    """dict 로 접으면 마지막 항목이 이긴다 — 앞의 해시가 조용히 사라진다."""
    root = _pkg(tmp_path)
    m = tmp_path / "M.json"
    m.write_text(json.dumps({
        "schema_version": 1, "file_count": 2,
        "files": [
            {"path": "a.txt", "size": 3, "sha256": "0" * 64},
            {"path": "a.txt", "size": 3, "sha256": "1" * 64},
        ],
    }), encoding="utf-8")

    assert verify(root, m)["exit_code"] == EXIT_NO_MANIFEST


def test_an_unreadable_file_is_reported_not_raised(tmp_path, monkeypatch):
    """접근 오류가 새어 나가면 계약 밖 코드로 끝난다 — 운영자는 그것을 못 읽는다."""
    root = _pkg(tmp_path)
    generate(root, tmp_path / "M.json")

    real_open = open

    def _deny(path, *a, **k):
        if str(path).endswith("a.txt"):
            raise PermissionError("locked")
        return real_open(path, *a, **k)

    import scripts.ops.intake_manifest as im
    monkeypatch.setattr(im, "open", _deny, raising=False)

    r = verify(root, tmp_path / "M.json")

    assert r["exit_code"] == EXIT_MISMATCH
    assert "a.txt" in r["unreadable"]


def test_a_file_that_cannot_be_stat_is_reported_not_raised(tmp_path, monkeypatch):
    """Windows 에서 백신·인덱서가 잡고 있으면 stat 도 실패한다."""
    root = _pkg(tmp_path)
    generate(root, tmp_path / "M.json")

    real_stat = Path.stat

    def _deny(self, *a, **k):
        if self.name == "a.txt":
            raise PermissionError("locked by scanner")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", _deny)
    r = verify(root, tmp_path / "M.json")

    assert r["exit_code"] == EXIT_MISMATCH
    assert "a.txt" in r["unreadable"]
