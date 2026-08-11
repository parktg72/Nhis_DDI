"""Fail when the duplicate-detection tripwires did not actually run.

The serving branch behind SERVING_ENABLE_EDI_NAME_RESOLUTION rests on a data fact:
`lookup_edi` resolves none of the real claim EDI codes, so `resolve_codes()` never
fills `atc_code`, so the duplicate detector never produces an entry. Two tripwire
tests watch that premise. Their silence is what the "unreachable" judgement -- and
the decision not to add a `_run_duplicate_detector` fail-safe -- stands on.

Those tests skip when the real reference data is absent, and a skip reads exactly
like a pass in a summary line. SERVING_TRIPWIRE_STRICT=1 turns that skip into a
failure; this script is what sets it, so "the gate runs with STRICT" stops being a
statement and becomes something callable.

Exit code 0 = tripwires ran and passed, 1 = anything else, including a run whose
outcome could not be read. Exit zero from pytest means the request succeeded, not
that the tripwires executed.

  $ python -m scripts.ops.check_tripwires
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]

STRICT_ENV = "SERVING_TRIPWIRE_STRICT"

TRIPWIRE_TESTS: tuple[str, ...] = (
    "tests/test_serving/test_safety_net_edi_resolution.py"
    "::test_tripwire_lookup_edi_still_resolves_no_real_edi",
    "tests/test_serving/test_safety_net_edi_resolution.py"
    "::test_tripwire_resolve_codes_leaves_atc_empty_for_real_edi",
)

_PASSED = re.compile(r"(\d+) passed")
_NOT_PASSED = re.compile(r"\d+ (failed|skipped|error|errors|deselected|xfailed)")


@dataclass(frozen=True)
class Result:
    ok: bool
    message: str


def build_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """STRICT 를 켠 환경. WSLENV 에 실어 Windows 프로세스까지 건너가게 한다.

    이 프로젝트의 인터프리터는 Windows 쪽(.venv/Scripts)에 있고 호출은 WSL 에서
    나가는 경우가 있다. 그 경계에서 환경변수는 WSLENV 에 이름이 올라야만 전달되며,
    올리지 않으면 **조용히** 사라진다 -- 게이트가 STRICT 로 돈다고 믿는 상태에서
    실제로는 아닌 상황이 만들어진다. 실제로 12차 측정에서 그 일이 있었다.
    """
    env = dict(os.environ if base is None else base)
    # 자식의 출력 인코딩도 지정한다. 지정하지 않으면 Windows 쪽에서 cp949 로 나오고,
    # 그 바이트를 utf-8 로 읽으면 대체문자가 섞여 보고 단계에서 터진다.
    forced = {STRICT_ENV: "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    env.update(forced)
    entries = [e for e in env.get("WSLENV", "").split(":") if e]
    for key in forced:
        if key not in entries:
            entries.append(key)
    env["WSLENV"] = ":".join(entries)
    return env


def emit(message: str, stream=None) -> None:
    """어떤 stdout 인코딩에서도 죽지 않고 보고한다.

    실패 사유를 읽을 수 없는 게이트는 게이트가 아니다. 자식 출력에는 대체문자가
    섞일 수 있고 부모 stdout 은 cp949 일 수 있으므로, 인코딩 실패는 문자 손실로
    떨어뜨리되 보고 자체는 반드시 나가게 한다.
    """
    out = sys.stdout if stream is None else stream
    enc = getattr(out, "encoding", None) or "utf-8"
    safe = message.encode(enc, errors="replace").decode(enc, errors="replace")
    print(safe, file=out)


def _default_runner(argv: Sequence[str], env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True)
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    return proc.returncode, out + err


def check_tripwires(
    runner: Callable[[Sequence[str], dict[str, str]], tuple[int, str]] = _default_runner,
    tests: Sequence[str] = TRIPWIRE_TESTS,
) -> Result:
    """트립와이어를 STRICT 로 돌리고 **실제로 돌았는지**까지 확인한다."""
    argv = [sys.executable, "-m", "pytest", *tests, "-q", "--no-header", "-p",
            "no:cacheprovider"]
    env = build_env()
    code, out = runner(argv, env)

    if code != 0:
        return Result(False, f"트립와이어 실행이 실패했다 (exit={code}).\n{out.strip()}")

    if _NOT_PASSED.search(out):
        return Result(False, (
            "통과하지 않은 트립와이어가 있다 — skip 은 통과가 아니다. 실데이터가 없는 "
            f"환경이라면 그 환경은 '중복탐지 도달 불가'를 검증하지 못한 것이다.\n{out.strip()}"
        ))

    m = _PASSED.search(out)
    if not m:
        return Result(False, (
            "통과 건수를 요약에서 확인하지 못했다. 종료코드 0 은 요청이 성공했다는 "
            f"뜻일 뿐 트립와이어가 돌았다는 뜻이 아니므로 통과로 치지 않는다.\n{out.strip()}"
        ))

    passed = int(m.group(1))
    if passed != len(tests):
        return Result(False, (
            f"트립와이어 {len(tests)}건 중 {passed}건만 돌았다. 나머지가 왜 수집되지 "
            f"않았는지 확인하라.\n{out.strip()}"
        ))

    return Result(True, f"트립와이어 {passed}건이 STRICT 로 실행되어 통과했다.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = check_tripwires()
    emit(result.message)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
