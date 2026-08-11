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
)


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


def test_passing_run_is_ok():
    run = _runner(0, f"{len(TRIPWIRE_TESTS)} passed in 3.00s\n")
    result = check_tripwires(runner=run)
    assert result.ok, result.message
    assert run.captured["env"][STRICT_ENV] == "1"
    for t in TRIPWIRE_TESTS:
        assert t in run.captured["argv"], run.captured["argv"]


def test_skipped_run_is_not_ok():
    """skip 은 통과가 아니다 — 이것이 이 게이트의 존재 이유다."""
    run = _runner(0, "1 passed, 1 skipped in 3.00s\n")
    result = check_tripwires(runner=run)
    assert not result.ok
    assert "skip" in result.message.lower() or "생략" in result.message


def test_failing_run_is_not_ok():
    run = _runner(1, "1 failed, 1 passed in 3.00s\n")
    result = check_tripwires(runner=run)
    assert not result.ok


def test_unreadable_summary_fails_closed():
    """통과 건수를 읽지 못하면 통과로 치지 않는다.

    종료코드 0 은 '요청이 성공했다'는 뜻일 뿐 '트립와이어가 돌았다'는 뜻이 아니다.
    건수를 확인하지 못한 상태는 모름이며, 모름은 근거가 아니다.
    """
    run = _runner(0, "no summary line here\n")
    result = check_tripwires(runner=run)
    assert not result.ok
    assert "확인" in result.message or "요약" in result.message


def test_partial_run_is_not_ok():
    """등록된 트립와이어보다 적게 돌면 통과가 아니다."""
    run = _runner(0, "1 passed in 3.00s\n")
    result = check_tripwires(runner=run)
    assert not result.ok
    assert str(len(TRIPWIRE_TESTS)) in result.message


def test_registered_tripwires_exist_in_the_suite():
    """게이트가 가리키는 테스트가 실제로 존재해야 한다 — 이름이 바뀌면 게이트가 빈다."""
    for nodeid in TRIPWIRE_TESTS:
        path, _, name = nodeid.partition("::")
        src = (ROOT / path).read_text(encoding="utf-8")
        assert (ROOT / path).exists(), f"{path} 없음"
        assert f"def {name}(" in src, f"{nodeid} 가 더 이상 존재하지 않는다"
