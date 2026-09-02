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
         rule_level: str | None = None, other: int = 0, version: int | None = 3) -> dict:
    r = {
        "timestamp": f"{day}T01:00:00+00:00", "partition": day,
        "patient_id": pid, "risk_level": level,
        "rule_level": rule_level if rule_level is not None else level,
        "ml_level": None, "disagree": False, "latency_ms": 10.0, "source": "api",
    }
    if version is not None:
        r["schema_version"] = version
        r["serving_flags"] = {"SERVING_ENABLE_EDI_NAME_RESOLUTION": True,
                              "SERVING_RULE_DETECT_ONLY": True}
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


def test_corrupt_line_stops_the_report(tmp_path, capsys):
    """마지막 줄이 아닌 위치의 손상은 **집계 불가**여야 한다.

    세고 넘어가면 조용한 과소집계가 된다. append-only 파일에서 동시 쓰기로
    잘릴 수 있는 것은 마지막 줄뿐이므로, 그 밖의 파싱 실패는 실제 손상이다.
    """
    p = tmp_path / "m.jsonl"
    p.write_text(
        "{ 손상된 줄\n"
        + json.dumps(_rec("P1", "2026-09-05", "Yellow", ["TOP01"]), ensure_ascii=False)
        + "\n", encoding="utf-8")
    rc = main(["--path", str(p)])
    out = capsys.readouterr().out

    assert rc == EXIT_NO_DATA, "손상된 줄이 있는데 집계를 계속했다"
    assert "집계를 신뢰할 수 없다" in out


def test_incomplete_tail_is_not_corruption(tmp_path, capsys):
    """기록 중인 마지막 줄은 손상이 아니라 제외 대상이다.

    집계기는 기록기의 락을 잡지 않으므로(폐쇄망 제약), 서빙이 append 하는 중에
    읽으면 마지막 줄이 잘려 있을 수 있다. 이것까지 손상으로 처리하면 트래픽이
    있는 시간대에는 집계가 아예 되지 않는다.
    """
    p = tmp_path / "m.jsonl"
    good = [_rec(f"P{i}", f"2026-09-{5+i:02d}", "Yellow", ["TOP01"]) for i in range(8)]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in good)
                 + "\n" + '{"timestamp": "2026-09-13T01:00:00+0',  # ← 잘린 채 끝
                 encoding="utf-8")
    rc = main(["--path", str(p)])
    out = capsys.readouterr().out

    assert rc == EXIT_OK, "기록 중인 마지막 줄 때문에 집계가 중단됐다"
    assert "마지막 줄 미완성" in out


def test_empty_patient_ids_are_not_one_patient(tmp_path, capsys):
    """빈 patient_id 를 한 환자로 접으면 안 된다 — 발화율이 왜곡된다."""
    p = _write(tmp_path / "m.jsonl", [
        _rec("", "2026-09-05", "Yellow", ["TOP01"]),
        _rec("", "2026-09-05", "Yellow", ["TOP09"]),
        _rec("P1", "2026-09-05", "Green", []),
    ])
    main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "고유 환자         3명" in out, "빈 ID 두 행이 한 환자로 합쳐졌다"
    assert "patient_id 없는 행 2개" in out


def test_flag_change_during_window_is_flagged(tmp_path, capsys):
    """관측 중 플래그가 바뀌면 경고해야 한다 — 꺼진 날과 켜진 날이 섞인다."""
    off = _rec("P1", "2026-09-05", "Green", [])
    off["serving_flags"] = {"SERVING_ENABLE_EDI_NAME_RESOLUTION": False,
                            "SERVING_RULE_DETECT_ONLY": False}
    p = _write(tmp_path / "m.jsonl", [off, _rec("P2", "2026-09-06", "Yellow", ["TOP01"])])
    main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "서빙 플래그가 2가지로 관측됐다" in out
    assert "발화율을 그대로 인용하지 말 것" in out


def test_both_rates_are_printed(tmp_path, capsys):
    """기간 유병(환자)과 요청 기준을 함께 인쇄해야 한다.

    합집합만 인쇄하면 그 값이 "운영 발화율" 로 나간다. 환자가 여러 번 요청하면
    환자 기준이 요청 기준보다 크게 나온다.
    """
    rows = [_rec("P1", f"2026-09-{5+i:02d}", "Yellow", ["TOP01"]) for i in range(4)]
    rows += [_rec(f"Q{i}", "2026-09-05", "Green", []) for i in range(6)]
    p = _write(tmp_path / "m.jsonl", rows)
    main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "ⓐ 기간 유병" in out and "ⓑ 요청 기준" in out
    # 환자 기준 1/7 = 14.29%, 요청 기준 4/10 = 40.00%
    assert "1 / 7" in out and "4 / 10" in out
    assert '"운영 발화율" 로 인용하지 말 것' in out


def test_non_rule_uppercase_tokens_are_not_ids(tmp_path, capsys):
    """네임스페이스 밖의 대문자 토큰은 규칙 ID 가 아니다.

    생산자 쪽 문법이지만, 집계기가 그런 값을 받아도 Top-10 으로 세지 않아야 한다.
    """
    p = _write(tmp_path / "m.jsonl", [
        _rec("P1", "2026-09-05", "Yellow", ["TIMEOUT", "UNKNOWN", "TOP01"]),
    ])
    main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "TIMEOUT" in out, "알 수 없는 ID 를 조용히 버리면 안 된다"
    # Top-10 표에는 TOP01 만
    assert "TOP01   항응고제 + NSAID" in out


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


def test_schema2_records_do_not_trigger_a_false_flag_change(tmp_path, capsys):
    """`serving_flags` 가 없는 구 스키마(2)를 플래그 전환으로 오판하면 안 된다."""
    old = _rec("P1", "2026-09-05", "Yellow", ["TOP01"], version=2)
    del old["serving_flags"]
    p = _write(tmp_path / "m.jsonl", [old, _rec("P2", "2026-09-06", "Green", [])])
    main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "서빙 플래그가" not in out, "없는 플래그 전환을 보고했다"


def test_rate_direction_is_not_asserted(tmp_path, capsys):
    """두 발화율의 대소를 단정하지 않아야 한다 — 방향은 고정이 아니다."""
    p = _write(tmp_path / "m.jsonl", [_rec("P1", "2026-09-05", "Yellow", ["TOP01"])])
    main(["--path", str(p)])
    out = capsys.readouterr().out

    assert "어느 쪽이 큰지는 고정돼 있지 않다" in out
