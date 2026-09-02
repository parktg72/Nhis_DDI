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

from scripts.ops.a5_firing_report import (
    EXIT_NO_DATA,
    EXIT_OK,
    EXIT_REASONLESS_RED,
    EXIT_SHORT_WINDOW,
    main,
)


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    return path


def _rec(pid: str, day: str, level: str, ids: list[str] | None, *,
         rule_level: str | None = None, other: int = 0, version: int | None = 2) -> dict:
    r = {
        "timestamp": f"{day}T01:00:00+00:00", "partition": day,
        "patient_id": pid, "risk_level": level,
        "rule_level": rule_level if rule_level is not None else level,
        "ml_level": None, "disagree": False, "latency_ms": 10.0, "source": "api",
    }
    if version is not None:
        r["schema_version"] = version
        r["rule_ids"] = ids or []
        r["n_reasons"] = len(ids or []) + other
        r["n_other_reasons"] = other
    return r


def test_missing_file_is_not_a_zero(tmp_path, capsys):
    """파일이 없으면 발화 0 이 아니라 **집계 불가**로 끝나야 한다."""
    rc = main(["--path", str(tmp_path / "nope.jsonl")])
    out = capsys.readouterr().out

    assert rc == EXIT_NO_DATA, "파일 부재가 정상 종료로 처리됐다 — 0건으로 오독될 수 있다"
    assert "입력 파일이 없다" in out


def test_old_format_only_refuses_to_report_zero(tmp_path, capsys):
    """구형식(rule_ids 없음)만 있으면 발화 0 으로 보고하지 않아야 한다.

    이것이 이 도구의 가장 위험한 오독 경로다 — 규칙이 안 터진 것이 아니라
    기록에 규칙 ID 가 없을 뿐이다.
    """
    p = _write(tmp_path / "m.jsonl", [
        _rec("A", "2026-09-03", "Green", None, version=None),
        _rec("B", "2026-09-03", "Red", None, version=None),
    ])
    rc = main(["--path", str(p)])
    out = capsys.readouterr().out

    assert rc == EXIT_NO_DATA, "구형식만 있는데 정상 종료했다"
    assert "전부 구형식" in out
    assert "발화 0 으로 읽으면 안 된다" in out


def test_counts_unique_patients_not_rows(tmp_path, capsys):
    """규칙별 계수는 **고유 환자** 기준이어야 한다.

    리포트가 "환자" 라고 쓰므로 실제로 환자로 세야 한다. 요청 행으로 세면
    재요청이 많은 환자가 비율을 끌어올려 A5 본체 수치가 라벨과 달라진다.
    """
    p = _write(tmp_path / "m.jsonl", [
        _rec("P1", "2026-09-05", "Yellow", ["TOP01"]),
        _rec("P1", "2026-09-06", "Yellow", ["TOP01"]),   # 같은 환자 재요청
        _rec("P1", "2026-09-07", "Yellow", ["TOP01"]),
        _rec("P2", "2026-09-05", "Green", []),
    ])
    main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "고유 환자         2명  (요청 4행)" in out
    # TOP01 은 환자 1명 = 50% (행으로 세면 3명·75% 가 된다)
    assert "50.00%" in out
    assert "1 / 2" in out


def test_non_id_reasons_are_excluded_from_rules(tmp_path, capsys):
    """ID 형식이 아닌 사유는 규칙으로 세지 않되, 사라지지도 않아야 한다.

    `동일성분중복 3건`·`ML 모델 Red 확률: 45.2%` 처럼 값이 매번 달라지는 문구를
    ID 로 세면 집계표가 오염되고 카디널리티가 무한히 늘어난다. 생산자가 개수만
    넘기고, 집계기는 그 개수를 따로 보고한다.
    """
    p = _write(tmp_path / "m.jsonl", [
        _rec("P1", "2026-09-05", "Yellow", ["TOP01"], other=2),
    ])
    main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "ID 형식이 아닌 사유 2건은 집계에서 제외" in out


def test_string_rule_ids_are_not_iterated(tmp_path, capsys):
    """`rule_ids` 가 리스트가 아니면 집계에서 제외해야 한다.

    문자열을 순회하면 문자 하나가 규칙 ID 가 된다.
    """
    bad = _rec("P1", "2026-09-05", "Yellow", [])
    bad["rule_ids"] = "TOP01"
    p = _write(tmp_path / "m.jsonl", [bad])
    rc = main(["--path", str(p)])
    out = capsys.readouterr().out

    assert rc == EXIT_NO_DATA, "리스트가 아닌 rule_ids 를 집계에 넣었다"
    assert "구형식            1행" in out


def test_detect_only_reasonless_red_is_caught(tmp_path, capsys):
    """**탐지 전용 배포에서 사유 없는 Red 를 놓치면 안 된다.**

    탐지 전용에서는 최종 `risk_level` 이 Red 로 올라가지 않는다 — 그것이 그
    플래그의 목적이다. 최종 등급만 세면 항상 0 이 나오고, 도구가 A4 차단을
    거짓으로 해제한다. 규칙층 등급이 Red 인데 사유가 없는 경우를 함께 세야 한다.

    런북이 지시하는 배포 모드가 바로 탐지 전용이므로, 이 경로를 놓치면 도구가
    실제 운영에서 아무것도 측정하지 못한다.
    """
    p = _write(tmp_path / "m.jsonl", [
        # 최종은 Normal, 규칙층만 Red, 사유 없음 — 탐지 전용의 전형
        _rec("P1", "2026-09-05", "Normal", [], rule_level="Red"),
        _rec("P2", "2026-09-05", "Normal", ["TOP09"], rule_level="Red"),
    ])
    rc = main(["--path", str(p)])
    out = capsys.readouterr().out

    assert rc == EXIT_REASONLESS_RED, (
        "탐지 전용에서 사유 없는 Red 를 놓쳤다 — A4 차단이 거짓으로 풀린다"
    )
    assert "사유 0건 요청   1행" in out
    assert "차단 유지" in out
    assert "2026-09-05" in out, "발생 일자를 알려주지 않으면 추적할 수 없다"


def test_reasonless_red_returns_nonzero(tmp_path, capsys):
    """사유 없는 Red 는 화면 문구가 아니라 **종료 코드**로도 드러나야 한다.

    운영자가 리포트를 읽지 않고 반환값만 보는 경우에도 판정이 새면 안 된다.
    """
    p = _write(tmp_path / "m.jsonl", [
        _rec("P1", "2026-09-05", "Red", ["TOP01"]),
        _rec("P2", "2026-09-06", "Red", []),
    ])
    assert main(["--path", str(p)]) == EXIT_REASONLESS_RED
    capsys.readouterr()


def test_other_reasons_are_not_reasonless(tmp_path, capsys):
    """ID 가 아닌 사유만 있는 Red 는 "사유 없음" 이 아니다.

    중복탐지 사유만 붙은 Red 는 근거가 있는 것이다 — 이것까지 차단으로 세면
    거짓 경보가 된다.
    """
    p = _write(tmp_path / "m.jsonl", [
        _rec("P1", "2026-09-05", "Red", [], other=1),
    ])
    rc = main(["--path", str(p)])
    out = capsys.readouterr().out

    assert rc != EXIT_REASONLESS_RED, "ID 가 아닌 사유가 있는데 사유 없음으로 셌다"
    assert "사유 0건 요청   0행" in out


def test_zero_reasonless_does_not_declare_the_gate_open(tmp_path, capsys):
    """사유 없는 Red 가 0 이어도 **A4 해제를 선언하지 않아야 한다.**

    ③ 은 해제 조건 하나일 뿐이고, 관측 창·트래픽 대표성이 남는다. 판정란에
    "해제 조건을 충족" 이라고 쓰면 A4 가 열린 것처럼 읽힌다.
    """
    p = _write(tmp_path / "m.jsonl",
               [_rec(f"P{i}", f"2026-09-{5+i:02d}", "Green", ["TOP01"]) for i in range(8)])
    main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "사유 0건 Red 는 관측되지 않았다" in out
    assert "이것만으로 A4 차단이 해제되지는 않는다" in out
    assert "해제 조건을 충족" not in out


def test_short_window_returns_nonzero(tmp_path, capsys):
    """관측 창이 1주 미만이면 종료 코드로도 드러나야 한다."""
    p = _write(tmp_path / "m.jsonl", [_rec("P1", "2026-09-05", "Green", ["TOP01"])])
    assert main(["--path", str(p)]) == EXIT_SHORT_WINDOW
    capsys.readouterr()


def test_observation_window_ignores_old_format_dates(tmp_path, capsys):
    """관측 창은 **집계 가능한 레코드** 기준이어야 한다.

    구형식 날짜를 섞으면 신형식이 하루뿐인데도 창이 충분한 것처럼 보인다.
    """
    rows = [_rec(f"O{i}", f"2026-08-{1+i:02d}", "Green", None, version=None)
            for i in range(10)]
    rows.append(_rec("P1", "2026-09-05", "Green", ["TOP01"]))
    p = _write(tmp_path / "m.jsonl", rows)
    rc = main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "관측 일자         1일" in out
    assert rc == EXIT_SHORT_WINDOW, "구형식 날짜가 관측 창을 부풀렸다"


def test_report_states_it_is_not_p0_closure_evidence(tmp_path, capsys):
    """리포트 첫머리가 P0-1 종결 증거가 아님을 밝혀야 한다."""
    p = _write(tmp_path / "m.jsonl", [_rec("P1", "2026-09-05", "Green", [])])
    main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "P0-1 종결의 증거가 아니다" in out


def test_bad_date_argument_is_rejected(tmp_path, capsys):
    """--since/--until 은 문자열 비교이므로 형식을 검증해야 한다."""
    p = _write(tmp_path / "m.jsonl", [_rec("P1", "2026-09-05", "Green", [])])
    assert main(["--path", str(p), "--since", "2026-9-5"]) == EXIT_NO_DATA
    capsys.readouterr()


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

    assert "관측 일자         1일" in out
    assert "요일 편향" in out


def test_date_filter(tmp_path, capsys):
    """기간 필터가 실제로 잘라야 한다."""
    p = _write(tmp_path / "m.jsonl", [
        _rec("P1", "2026-09-05", "Yellow", ["TOP01"]),
        _rec("P2", "2026-09-20", "Yellow", ["TOP09"]),
    ])
    rc = main(["--path", str(p), "--since", "2026-09-10"])
    out = capsys.readouterr().out

    assert rc == EXIT_SHORT_WINDOW, "필터 후 하루만 남았는데 정상 종료했다"
    assert "레코드            1행" in out
    assert "2026-09-20 ~ 2026-09-20" in out


def test_broken_lines_are_counted_not_fatal(tmp_path, capsys):
    """깨진 줄은 세고 넘어가되, 조용히 사라지면 안 된다."""
    p = tmp_path / "m.jsonl"
    p.write_text(
        json.dumps(_rec("P1", "2026-09-05", "Yellow", ["TOP01"]), ensure_ascii=False)
        + "\n{ 깨진 줄\n", encoding="utf-8")
    rc = main(["--path", str(p)])
    out = capsys.readouterr().out

    assert rc != EXIT_NO_DATA, "깨진 줄 때문에 집계가 중단됐다"
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
