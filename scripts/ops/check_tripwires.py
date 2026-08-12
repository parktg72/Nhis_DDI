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
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]

STRICT_ENV = "SERVING_TRIPWIRE_STRICT"

TRIPWIRE_TESTS: tuple[str, ...] = (
    "tests/test_serving/test_safety_net_edi_resolution.py"
    "::test_tripwire_lookup_edi_still_resolves_no_real_edi",
    "tests/test_serving/test_safety_net_edi_resolution.py"
    "::test_tripwire_resolve_codes_leaves_atc_empty_for_real_edi",
)

_TAIL = 4000   # 리포트가 판정하고, stdout 은 사람이 읽을 맥락으로만 붙인다


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


TIMEOUT_S = 900   # 트립와이어는 참조DB 적재가 있어 느리지만, 무한 대기는 실패다


def _default_runner(argv: Sequence[str], env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True,
                          timeout=TIMEOUT_S)
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    return proc.returncode, out + err


def _identity(nodeid: str) -> tuple[str, str]:
    """nodeid → junit 의 (classname, name).

    pytest 는 `tests/a/b.py::test_x` 를 classname `tests.a.b`, name `test_x` 로 쓴다.
    """
    path, _, name = nodeid.partition("::")
    classname = path[:-3] if path.endswith(".py") else path
    return classname.replace("/", ".").replace("\\", "."), name


def judge(code: int, out: str, report: Path, tests: Sequence[str]) -> Result:
    """**구조화된 결과만** 보고 판정한다. 요약 텍스트는 근거가 아니다.

    13차에서 codex-terra 와 agy 가 함께 짚었다 — 앵커 없는 정규식이 전체 stdout 을
    훑았으므로, 아무 데나 "2 passed" 가 섞여 있으면 통과로 읽혔다. 종료코드 0 을
    믿지 않겠다고 만든 물건이 텍스트에는 속고 있었다. 이제 pytest 가 쓴 junit XML 의
    수치만 본다. stdout 은 사람이 사유를 읽을 맥락으로만 붙인다.
    """
    ctx = out.strip()[-_TAIL:]

    if not report.exists():
        return Result(False, (
            f"결과 리포트가 없다 (exit={code}). 종료코드는 요청이 끝났다는 뜻일 뿐 "
            f"트립와이어가 돌았다는 뜻이 아니므로 통과로 치지 않는다.\n{ctx}"
        ))
    try:
        root = ElementTree.parse(report).getroot()
        suites = list(root.iter("testsuite"))
        if not suites:
            raise ValueError("testsuite 요소 없음")
        # **모든** suite 를 합산한다 — 첫 suite 만 보면 나머지의 실패를 놓친다
        counts = {k: sum(int(s.get(k, 0)) for s in suites)
                  for k in ("tests", "failures", "errors", "skipped")}
        seen = {(c.get("classname", ""), c.get("name", ""))
                for c in root.iter("testcase")}
    except (ElementTree.ParseError, ValueError, TypeError) as exc:
        return Result(False, f"결과 리포트를 읽지 못했다 ({exc}). 통과로 치지 않는다.\n{ctx}")

    if counts["failures"] or counts["errors"]:
        return Result(False, (
            f"트립와이어가 실패했다 (failures={counts['failures']}, "
            f"errors={counts['errors']}).\n{ctx}"
        ))
    if counts["skipped"]:
        return Result(False, (
            f"트립와이어 {counts['skipped']}건이 생략됐다 — skip 은 통과가 아니다. "
            "실데이터가 없는 환경이라면 그 환경은 '중복탐지 도달 불가'를 검증하지 "
            f"못한 것이다.\n{ctx}"
        ))
    ran = counts["tests"] - counts["skipped"]
    if ran != len(tests):
        return Result(False, (
            f"등록된 트립와이어 {len(tests)}건 중 {ran}건만 돌았다. 나머지가 왜 "
            f"수집되지 않았는지 확인하라.\n{ctx}"
        ))

    # 수치가 맞아도 **다른 테스트**가 돌았을 수 있다. 신원까지 대조하지 않으면
    # 게이트는 "어떤 2건이든 통과했다"만 말하고 트립와이어에 대해서는 아무것도
    # 말하지 않는다(codex-terra·fable-advisor 14차 지적).
    expected = {_identity(t) for t in tests}
    if seen != expected:
        return Result(False, (
            "돌아간 테스트의 신원이 등록된 트립와이어와 일치하지 않는다.\n"
            f"  기대: {sorted(expected)}\n  실제: {sorted(seen)}\n{ctx}"
        ))
    if code != 0:
        return Result(False, f"수치는 정상이나 종료코드가 {code} 다. 통과로 치지 않는다.\n{ctx}")

    return Result(True, f"트립와이어 {ran}건이 STRICT 로 실행되어 통과했다.")


def check_tripwires(
    runner: Callable[[Sequence[str], dict[str, str]], tuple[int, str]] = _default_runner,
    tests: Sequence[str] = TRIPWIRE_TESTS,
) -> Result:
    """트립와이어를 STRICT 로 돌리고 **실제로 돌았는지**까지 확인한다."""
    with tempfile.TemporaryDirectory() as td:
        report = Path(td) / "tripwires.xml"
        argv = [sys.executable, "-m", "pytest", *tests, "-q", "--no-header",
                "-p", "no:cacheprovider", f"--junit-xml={report}"]
        try:
            code, out = runner(argv, build_env())
        except subprocess.TimeoutExpired as exc:
            return Result(False, (
                f"트립와이어 실행이 제한 시간({exc.timeout}초)을 넘겼다. 무한 대기는 "
                "게이트에서 통과보다 나쁘다 — 아무도 결과를 못 받는다."
            ))
        return judge(code, out, report, tests)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = check_tripwires()
    emit(result.message)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
