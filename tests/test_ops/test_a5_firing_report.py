"""A5 발화 집계 도구 — 산출이 맞는지, 그리고 조용히 틀리지 않는지.

이 도구의 출력은 P0-1 종결 판단과 A4 활성 해제에 쓰인다. 가장 위험한 실패는
예외가 아니라 **잘못된 0** 이다. 기록에 규칙 ID 가 없을 뿐인데 "규칙이 안
터진다" 로 읽히면 없는 결론이 만들어진다. 그 구분을 여기서 고정한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ops.a5_firing_report import main


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    return path


def _rec(pid: str, day: str, level: str, ids: list[str] | None, *, new_format=True) -> dict:
    r = {
        "timestamp": f"{day}T01:00:00+00:00", "partition": day,
        "patient_id": pid, "risk_level": level, "rule_level": level,
        "ml_level": None, "disagree": False, "latency_ms": 10.0, "source": "api",
    }
    if new_format:
        r["rule_ids"] = ids or []
        r["n_reasons"] = len(ids or [])
    return r


def test_missing_file_is_not_a_zero(tmp_path, capsys):
    """파일이 없으면 발화 0 이 아니라 **집계 불가**로 끝나야 한다."""
    rc = main(["--path", str(tmp_path / "nope.jsonl")])
    out = capsys.readouterr().out

    assert rc == 2, "파일 부재가 정상 종료로 처리됐다 — 0건으로 오독될 수 있다"
    assert "입력 파일이 없다" in out


def test_old_format_only_refuses_to_report_zero(tmp_path, capsys):
    """구형식(rule_ids 없음)만 있으면 발화 0 으로 보고하지 않아야 한다.

    이것이 이 도구의 가장 위험한 오독 경로다 — 규칙이 안 터진 것이 아니라
    기록에 규칙 ID 가 없을 뿐이다.
    """
    p = _write(tmp_path / "m.jsonl", [
        _rec("A", "2026-09-03", "Green", None, new_format=False),
        _rec("B", "2026-09-03", "Red", None, new_format=False),
    ])
    rc = main(["--path", str(p)])
    out = capsys.readouterr().out

    assert rc == 2, "구형식만 있는데 정상 종료했다"
    assert "전부 구형식" in out
    assert "발화 0 으로 읽으면 안 된다" in out


def test_counts_rules_per_patient(tmp_path, capsys):
    """규칙별 발화 환자 수와 환자 단위 발화율."""
    p = _write(tmp_path / "m.jsonl", [
        _rec("P1", "2026-09-05", "Yellow", ["TOP01"]),
        _rec("P2", "2026-09-05", "Red", ["TOP01", "TOP09"]),
        _rec("P3", "2026-09-05", "Green", []),
        _rec("P4", "2026-09-05", "Yellow", ["GRADE_POLYPHARMACY_HIGH_RISK"]),
    ])
    rc = main(["--path", str(p)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "TOP01" in out and "2" in out
    # Top-10 해당은 P1·P2 두 명 = 50%
    assert "2 / 4" in out and "50.00%" in out
    # 규칙 외 사유는 별도 절에 나온다
    assert "GRADE_POLYPHARMACY_HIGH_RISK" in out


def test_reasonless_red_blocks(tmp_path, capsys):
    """사유 없는 Red 가 있으면 차단 유지로 보고해야 한다."""
    p = _write(tmp_path / "m.jsonl", [
        _rec("P1", "2026-09-05", "Red", ["TOP01"]),
        _rec("P2", "2026-09-06", "Red", []),          # ← 사유 없는 Red
    ])
    main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "그중 사유 0건  1건" in out
    assert "차단 유지" in out
    assert "2026-09-06" in out, "발생 일자를 알려주지 않으면 추적할 수 없다"


def test_no_reasonless_red_clears(tmp_path, capsys):
    """사유 없는 Red 가 0 이면 해제 조건 충족으로 보고한다."""
    p = _write(tmp_path / "m.jsonl", [
        _rec("P1", "2026-09-05", "Red", ["TOP05"]),
        _rec("P2", "2026-09-05", "Green", []),
    ])
    main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "그중 사유 0건  0건" in out
    assert "해제 조건을 충족" in out


def test_silent_rules_are_named_not_assumed(tmp_path, capsys):
    """무발화 규칙은 이름을 대고, 원인 단정 대신 확인 절차를 지시해야 한다."""
    p = _write(tmp_path / "m.jsonl", [_rec("P1", "2026-09-05", "Yellow", ["TOP01"])])
    main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "무발화 규칙 9종" in out
    assert "TOP05" in out and "TOP10" in out
    assert "해소 결함인지 실제로 그 병용이 없는 것인지" in out, (
        "무발화의 원인을 단정하면 안 된다 — 두 가능성을 모두 제시해야 한다"
    )


def test_short_observation_is_flagged(tmp_path, capsys):
    """관측 기간이 짧으면 경고해야 한다 — 요일 편향."""
    p = _write(tmp_path / "m.jsonl", [_rec("P1", "2026-09-05", "Green", [])])
    main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "관측 일자 1일" in out and "요일 편향" in out


def test_date_filter(tmp_path, capsys):
    """기간 필터가 실제로 잘라야 한다."""
    p = _write(tmp_path / "m.jsonl", [
        _rec("P1", "2026-09-05", "Yellow", ["TOP01"]),
        _rec("P2", "2026-09-20", "Yellow", ["TOP09"]),
    ])
    main(["--path", str(p), "--since", "2026-09-10"])
    out = capsys.readouterr().out

    assert "레코드            1건" in out
    assert "2026-09-20 ~ 2026-09-20" in out


def test_broken_lines_are_counted_not_fatal(tmp_path, capsys):
    """깨진 줄은 세고 넘어가되, 조용히 사라지면 안 된다."""
    p = tmp_path / "m.jsonl"
    p.write_text(
        json.dumps(_rec("P1", "2026-09-05", "Yellow", ["TOP01"]), ensure_ascii=False)
        + "\n{ 깨진 줄\n", encoding="utf-8")
    rc = main(["--path", str(p)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "파싱 실패         1줄" in out


def test_empty_after_filter_is_not_a_zero(tmp_path, capsys):
    """필터 결과가 0건이면 발화 0 이 아니라 집계 불가다."""
    p = _write(tmp_path / "m.jsonl", [_rec("P1", "2026-09-05", "Yellow", ["TOP01"])])
    rc = main(["--path", str(p), "--since", "2026-10-01"])
    out = capsys.readouterr().out

    assert rc == 2
    assert "0건" in out


def test_report_can_be_saved(tmp_path):
    """--out 으로 저장되어야 한다 — 폐쇄망에서 화면 캡처 대신 파일로 반출한다."""
    p = _write(tmp_path / "m.jsonl", [_rec("P1", "2026-09-05", "Yellow", ["TOP01"])])
    dest = tmp_path / "a5.txt"
    main(["--path", str(p), "--out", str(dest)])

    assert dest.exists()
    assert "A5 — 운영 발화 집계" in dest.read_text(encoding="utf-8")
