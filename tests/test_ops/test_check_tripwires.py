"""`check_tripwires` 게이트 자체를 검사한다.

12차 리뷰에서 fable-advisor 와 codex-terra 가 같은 것을 짚었다 — "릴리스 게이트가
STRICT=1 로 돈다"는 것이 **진술로만** 남아 있었다. 그 진술을 호출 가능한 검사로
바꿨고, 이 파일은 그 검사가 실제로 무엇을 잡는지 확인한다.

특히 skip 을 성공으로 읽지 않는지가 핵심이다. 트립와이어는 무발화가 곧 안전 판정의
근거이므로, 돌지 않은 것을 통과로 읽으면 게이트가 있으나 마나다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ops.check_tripwires import (  # noqa: E402
    STRICT_ENV,
    TRIPWIRE_TESTS,
    build_env,
    check_tripwires,
    emit,
    judge,
)


def _report(tmp_path, *, tests: int, failures: int = 0, errors: int = 0, skipped: int = 0):
    """pytest 가 쓰는 junit XML 을 흉내낸 파일을 만든다."""
    p = tmp_path / "tripwires.xml"
    p.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<testsuites><testsuite name="pytest" '
        f'tests="{tests}" failures="{failures}" errors="{errors}" skipped="{skipped}">'
        "</testsuite></testsuites>\n",
        encoding="utf-8",
    )
    return p


def _runner(returncode: int, stdout: str):
    """주입 가능한 가짜 러너 — 실제 pytest 를 돌리지 않고 판정 로직만 검사한다."""
    captured = {}

    def run(argv, env):
        captured["argv"] = argv
        captured["env"] = env
        return returncode, stdout

    run.captured = captured
    return run


def test_build_env_sets_strict_and_carries_it_across_the_wsl_boundary():
    """STRICT 를 켜는 것만으로는 부족하다 — WSLENV 에 실려야 Windows 로 건너간다.

    이 프로젝트의 인터프리터는 Windows 쪽에 있고 호출은 WSL 에서 나간다. 12차
    측정에서 이 누락 때문에 env 주입이 조용히 무효가 됐고, 게이트가 STRICT 로 돈다고
    믿는 상태에서 실제로는 아니었다. 그 사고를 코드에 박아 둔다.
    """
    env = build_env({"PATH": "/usr/bin"})

    assert env[STRICT_ENV] == "1"
    assert STRICT_ENV in env.get("WSLENV", "").split(":"), (
        f"STRICT 가 WSLENV 에 실리지 않았다 — Windows 프로세스에 전달되지 않는다: "
        f"{env.get('WSLENV')!r}"
    )
    assert env["PATH"] == "/usr/bin", "기존 환경을 보존해야 한다"


def test_build_env_forces_utf8_in_the_child_process():
    """자식 pytest 의 출력 인코딩까지 지정하고, 그것도 경계를 넘겨야 한다.

    지정하지 않으면 Windows 쪽에서 cp949 로 나오고, 그 바이트를 utf-8 로 읽으면
    대체문자가 섞인다. 그 문자가 다시 부모 stdout 으로 나가다 죽는다 — 게이트가
    판정에 성공하고도 보고에서 터진다.
    """
    env = build_env({})
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUTF8"] == "1"
    entries = env["WSLENV"].split(":")
    for k in ("PYTHONIOENCODING", "PYTHONUTF8", STRICT_ENV):
        assert k in entries, f"{k} 가 WSLENV 에 없다: {entries}"


def test_emit_survives_a_stdout_that_cannot_encode_the_message():
    """보고가 인코딩 때문에 죽으면 안 된다 — 실패 사유를 못 읽는 게이트는 무용하다."""
    import io

    buf = io.TextIOWrapper(io.BytesIO(), encoding="cp949", errors="strict")
    emit("트립와이어 실패: � 를 포함한 원문", stream=buf)
    buf.flush()
    written = buf.buffer.getvalue().decode("cp949", errors="replace")
    assert "트립와이어 실패" in written, written


def test_existing_wslenv_entries_are_preserved():
    env = build_env({"WSLENV": "FOO/p:BAR"})
    entries = env["WSLENV"].split(":")
    assert "FOO/p" in entries and "BAR" in entries, entries
    assert STRICT_ENV in entries, entries


def test_passing_report_is_ok(tmp_path):
    report = _report(tmp_path, tests=len(TRIPWIRE_TESTS))
    result = judge(0, f"{len(TRIPWIRE_TESTS)} passed in 3.00s\n", report, TRIPWIRE_TESTS)
    assert result.ok, result.message


def test_stdout_cannot_fake_a_pass(tmp_path):
    """**이 게이트가 존재하는 이유.** 요약 텍스트는 판정 근거가 아니다.

    13차에서 codex-terra 와 agy 가 함께 짚었다 — 앵커 없는 정규식이 전체 stdout 을
    훑으므로, 아무 데나 "2 passed" 가 있으면 통과로 읽혔다. 종료코드 0 을 믿지
    않겠다고 만든 물건이 텍스트에는 속고 있었다. 이제 구조화된 결과만 본다.
    """
    report = _report(tmp_path, tests=0)
    stdout = "test output mentioning 2 passed somewhere\nERROR: nothing ran\n"
    result = judge(0, stdout, report, TRIPWIRE_TESTS)
    assert not result.ok, "가짜 요약 텍스트에 속았다"


def test_skipped_report_is_not_ok(tmp_path):
    """skip 은 통과가 아니다 — 이것이 STRICT 승격의 존재 이유다."""
    report = _report(tmp_path, tests=len(TRIPWIRE_TESTS), skipped=1)
    result = judge(0, "1 passed, 1 skipped\n", report, TRIPWIRE_TESTS)
    assert not result.ok
    assert "skip" in result.message.lower() or "생략" in result.message


def test_failing_report_is_not_ok(tmp_path):
    report = _report(tmp_path, tests=len(TRIPWIRE_TESTS), failures=1)
    result = judge(1, "1 failed, 1 passed\n", report, TRIPWIRE_TESTS)
    assert not result.ok


def test_error_report_is_not_ok(tmp_path):
    report = _report(tmp_path, tests=len(TRIPWIRE_TESTS), errors=1)
    result = judge(0, "", report, TRIPWIRE_TESTS)
    assert not result.ok


def test_missing_report_fails_closed(tmp_path):
    """리포트가 없으면 통과로 치지 않는다 — 모름은 근거가 아니다."""
    result = judge(0, "2 passed in 3.00s\n", tmp_path / "absent.xml", TRIPWIRE_TESTS)
    assert not result.ok
    assert "리포트" in result.message or "확인" in result.message


def test_unparsable_report_fails_closed(tmp_path):
    p = tmp_path / "broken.xml"
    p.write_text("not xml at all", encoding="utf-8")
    result = judge(0, "2 passed\n", p, TRIPWIRE_TESTS)
    assert not result.ok


def test_partial_run_is_not_ok(tmp_path):
    """등록된 트립와이어보다 적게 돌면 통과가 아니다."""
    report = _report(tmp_path, tests=1)
    result = judge(0, "1 passed\n", report, TRIPWIRE_TESTS)
    assert not result.ok
    assert str(len(TRIPWIRE_TESTS)) in result.message


def test_runner_is_asked_for_a_structured_report(tmp_path):
    """실행 인자에 junit XML 출력이 반드시 들어가야 한다."""
    run = _runner(0, "")
    check_tripwires(runner=run)
    assert any(a.startswith("--junit-xml=") for a in run.captured["argv"]), \
        run.captured["argv"]
    assert run.captured["env"][STRICT_ENV] == "1"
    for t in TRIPWIRE_TESTS:
        assert t in run.captured["argv"], run.captured["argv"]


def test_registered_tripwires_exist_in_the_suite():
    """게이트가 가리키는 테스트가 실제로 존재해야 한다 — 이름이 바뀌면 게이트가 빈다."""
    for nodeid in TRIPWIRE_TESTS:
        path, _, name = nodeid.partition("::")
        src = (ROOT / path).read_text(encoding="utf-8")
        assert (ROOT / path).exists(), f"{path} 없음"
        assert f"def {name}(" in src, f"{nodeid} 가 더 이상 존재하지 않는다"
