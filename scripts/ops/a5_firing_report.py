"""A5 — 운영 트래픽에서 규칙이 실제로 얼마나 발화하는지 집계한다.

입력은 서빙이 남기는 메트릭 JSONL(`DDI_METRICS_JSONL_PATH`). 읽기 전용이며
표준 라이브러리만 쓴다 — 폐쇄망 운영 PC 에서 venv 없이 돌아야 한다.

산출은 배포 런북 5절의 세 항목이다.
  ① 규칙별 발화 **환자 수**   — 고유 patient_id 기준. A5 의 본체
  ② 환자 단위 발화율          — 고유 환자 중 규칙이 하나라도 붙은 비율
  ③ 사유 없는 Red             — A4 활성의 차단 항목

**③ 은 최종 등급만 보지 않는다.** 탐지 전용 배포에서는 최종 `risk_level` 이
Red 로 올라가지 않으므로(그것이 그 플래그의 목적이다), 최종 등급만 세면 항상
0 이 나오고 차단이 거짓으로 풀린다. 규칙층 등급(`rule_level`)이 Red 인데 사유가
없는 경우를 함께 센다.

종료 코드 — **이 도구 전용 계약이다.** 저장소의 다른 ops 스크립트는 비정상을
1 로 반환한다(`check_tripwires.py` 등). 여기서 나누는 이유는 운영자가 화면
문구를 읽지 않고 반환값만 보는 경우에도 판정이 새지 않게 하기 위해서다.
  0  집계 성공 · 사유 없는 Red 0 · 관측 창 충분
  2  집계 불가 (파일 없음 · 레코드 0건 · 구형식만)
  3  사유 없는 Red 발견 — A4 차단 유지
  4  관측 창 부족 (7일 미만)
우선순위는 2 > 3 > 4.

사용:
    python3 scripts/ops/a5_firing_report.py --path /app/data/monitoring/metrics_live.jsonl
    python3 scripts/ops/a5_firing_report.py --path <경로> --since 2026-09-05 --out a5.txt

주의 — `schema_version >= 2` 인 레코드만 발화 집계에 넣는다. 그 이전 레코드에는
규칙 ID 가 없으므로, 발화 0 으로 읽으면 "규칙이 안 터진다" 는 없는 결론이 된다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# 규칙 ID 분류 — 리포트에서 묶어 보여준다
TOP_RULES = [f"TOP{i:02d}" for i in range(1, 11)]
TOP_LABEL = {
    "TOP01": "항응고제 + NSAID",
    "TOP02": "clopidogrel + PPI",
    "TOP03": "Triple Whammy (신손상)",
    "TOP04": "digoxin + amiodarone·verapamil",
    "TOP05": "methotrexate + trimethoprim (금기)",
    "TOP06": "SSRI + MAOI",
    "TOP07": "SSRI + triptan",
    "TOP08": "lithium + NSAID·이뇨제",
    "TOP09": "QT 연장 3종 이상",
    "TOP10": "statin + macrolide (횡문근융해)",
}
RED_LEVELS = {"Red", "RED", "red"}
SCHEMA_MIN = 2                 # 규칙 ID 를 담기 시작한 기록 스키마
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

EXIT_OK, EXIT_NO_DATA, EXIT_REASONLESS_RED, EXIT_SHORT_WINDOW = 0, 2, 3, 4

OUT: list[str] = []


def say(line: str = "") -> None:
    OUT.append(line)
    print(line)


def head(title: str) -> None:
    say()
    say("─" * 72)
    say(f" {title}")
    say("─" * 72)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="A5 발화 집계 (읽기 전용)")
    ap.add_argument("--path", required=True, help="메트릭 JSONL 경로")
    ap.add_argument("--since", help="이 날짜(YYYY-MM-DD) 이후만 집계")
    ap.add_argument("--until", help="이 날짜(YYYY-MM-DD)까지만 집계")
    ap.add_argument("--out", help="리포트를 이 파일에도 저장")
    return ap.parse_args(argv)


def load(path: Path, since: str | None, until: str | None):
    """JSONL 을 읽어 (레코드, 손상 줄 수, 미완성 마지막 줄 여부) 반환.

    **집계기는 기록기의 락을 잡지 않는다** — 기록기는 filelock 을 쓰고 이 도구는
    표준 라이브러리만 쓰기 때문이다(폐쇄망 제약). 그래서 서빙이 append 하는 중에
    읽으면 마지막 줄이 잘려 있을 수 있다.

    append-only 파일에서 동시 쓰기로 잘릴 수 있는 것은 **마지막 줄뿐**이다.
    파일이 개행으로 끝나지 않으면 그 줄은 기록 중인 것으로 보고 제외한다(정상).
    그 밖의 위치에서 파싱이 실패하면 그것은 실제 손상이며, 조용히 넘기면
    과소집계가 된다 — 호출자가 집계 불가로 끝낸다.
    """
    raw = path.read_text(encoding="utf-8")
    incomplete_tail = bool(raw) and not raw.endswith("\n")
    lines = raw.splitlines()
    if incomplete_tail and lines:
        lines = lines[:-1]

    rows, corrupt = [], 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            corrupt += 1
            continue
        part = r.get("partition") or (r.get("timestamp") or "")[:10]
        if since and part < since:
            continue
        if until and part > until:
            continue
        rows.append(r)
    return rows, corrupt, incomplete_tail


def _is_new(r: dict) -> bool:
    """발화 집계에 넣을 수 있는 레코드인가.

    필드 존재로 추측하지 않고 스키마 버전을 본다. `rule_ids` 가 리스트가 아닌
    레코드(문자열 등)는 제외한다 — 문자열을 순회하면 문자 하나가 규칙 ID 가 된다.
    """
    try:
        if int(r.get("schema_version", 0)) < SCHEMA_MIN:
            return False
    except (TypeError, ValueError):
        return False
    return isinstance(r.get("rule_ids"), list)


def _part(r: dict) -> str:
    return r.get("partition") or (r.get("timestamp") or "")[:10]


def main(argv=None) -> int:
    args = parse_args(argv)
    path = Path(args.path)

    for label, v in (("--since", args.since), ("--until", args.until)):
        if v and not DATE_RE.match(v):
            print(f"{label} 는 YYYY-MM-DD 형식이어야 한다: {v!r}")
            return EXIT_NO_DATA

    say("A5 — 운영 발화 집계")
    say("**이 출력은 P0-1 종결의 증거가 아니다.** 관측 기간·대상 범위와 함께만")
    say("인용할 것. 종결 판단은 별도 체크리스트와 서명으로 한다.")
    say()
    say(f"실행 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC "
        f"(일자 구분도 UTC 기준이다)")
    say(f"입력 {path}")
    if args.since or args.until:
        say(f"기간 {args.since or '처음'} ~ {args.until or '끝'}")

    if not path.exists():
        say(f"\n입력 파일이 없다: {path}")
        say("  → 서빙이 아직 기록하지 않았거나 경로가 다르다. DDI_METRICS_JSONL_PATH 확인.")
        return EXIT_NO_DATA

    rows, corrupt, incomplete_tail = load(path, args.since, args.until)
    if corrupt:
        say(f"\n손상된 줄 {corrupt:,}개가 있다 — 집계를 신뢰할 수 없다.")
        say("  → 마지막 줄이 아닌 위치의 파싱 실패는 기록 중 잘림이 아니라 손상이다.")
        say("    파일을 확인하기 전까지 이 결과를 쓰지 말 것. 발화 0 이 아니라 집계 불가다.")
        return EXIT_NO_DATA
    if not rows:
        say("\n집계 대상 레코드가 0건이다. 기간 조건 또는 기록 여부를 확인할 것.")
        say("  → 이것은 발화 0 이 아니라 집계 불가다.")
        return EXIT_NO_DATA

    new_fmt = [r for r in rows if _is_new(r)]
    old_fmt = [r for r in rows if not _is_new(r)]

    head("0. 입력 상태")
    say(f"  레코드            {len(rows):,}행")
    say(f"  발화 집계 가능    {len(new_fmt):,}행  (schema_version ≥ {SCHEMA_MIN})")
    say(f"  구형식            {len(old_fmt):,}행  (규칙 ID 없음 — 발화 집계 제외)")
    if incomplete_tail:
        say("  마지막 줄 미완성  1줄  (기록 중 — 제외. 집계기는 기록기의 락을 잡지 않는다)")

    if not new_fmt:
        parts = sorted({_part(r) for r in rows})
        say(f"  관측 일자         {len(parts)}일  {parts[0]} ~ {parts[-1]}")
        say()
        say("  ⚠ 전부 구형식이다. 발화 0 으로 읽으면 안 된다 — 기록에 규칙 ID 가 없을 뿐이다.")
        say("    서빙을 규칙 ID 기록 판 이후로 올린 뒤 다시 관측할 것.")
        return EXIT_NO_DATA

    # 관측 창은 **집계 가능한 레코드 기준**이다. 구형식 날짜를 섞으면 신형식이
    # 하루뿐인데도 창이 충분한 것처럼 보인다.
    parts = sorted({_part(r) for r in new_fmt})
    say(f"  관측 일자         {len(parts)}일  {parts[0]} ~ {parts[-1]}  (집계 가능분 기준)")

    # ── 플래그 상태를 먼저 확정한다 ─────────────────────────────────────
    # 혼합 분모로는 어떤 비율도 의미가 없다. 꺼진 날과 켜진 날이 섞이면 발화율이
    # 아무것도 재지 않는다. 경고만 하고 계속 세면 그 숫자가 그대로 인용된다.
    n_rows = len(new_fmt)
    flag_sets = {json.dumps(r["serving_flags"], sort_keys=True)
                 for r in new_fmt if isinstance(r.get("serving_flags"), dict)}
    n_unknown = sum(1 for r in new_fmt if not isinstance(r.get("serving_flags"), dict))

    if n_unknown:
        say()
        say(f"  ✗ 플래그 상태를 알 수 없는 행 {n_unknown:,}개 (구 스키마).")
        say("    어떤 플래그로 얻은 수치인지 모르면 발화율은 아무것도 재지 않는다.")
        say("    --since 로 플래그가 기록되기 시작한 이후 구간만 집계할 것.")
        return EXIT_NO_DATA

    if len(flag_sets) > 1:
        say()
        say(f"  ✗ 관측 구간에서 서빙 플래그가 {len(flag_sets)}가지로 관측됐다.")
        for fs in sorted(flag_sets):
            d = json.loads(fs)
            on = sorted(k for k, v in d.items() if v) or ["(전부 꺼짐)"]
            say(f"      켜짐: {', '.join(on)}")
        say("    분모가 섞여 어떤 비율도 의미가 없다. --since/--until 로 플래그가")
        say("    일정한 구간을 잘라 각각 집계할 것.")
        return EXIT_NO_DATA

    if flag_sets:
        d = json.loads(next(iter(flag_sets)))
        on = sorted(k for k, v in d.items() if v) or ["(전부 꺼짐)"]
        say(f"  서빙 플래그       {', '.join(on)}")
        if not d.get("SERVING_ENABLE_EDI_NAME_RESOLUTION"):
            say()
            say("  ✗ 이름 해소가 꺼진 구간이다. EDI-only 요청에서 규칙은 원래 발화하지 않는다.")
            say("    이 구간의 발화 0 은 규칙의 문제가 아니라 플래그의 문제다.")
            return EXIT_NO_DATA

    # ── 환자 단위로 접는다 ────────────────────────────────────────────────
    # 리포트가 "환자" 라고 쓰므로 실제로 환자로 세야 한다. 같은 환자의 여러 요청을
    # 그대로 세면 재요청이 많은 환자가 비율을 끌어올린다.
    #
    # patient_id 가 없는 행은 **환자 지표에서 제외한다.** 한 명으로 접으면 서로 다른
    # 환자가 한 명이 되고, 행마다 별개로 세면 고유 환자 수가 부풀려진다. 둘 다
    # 틀린 숫자다. 요청 기준 지표(ⓑ)와 사유 없는 Red(행 기준)에는 그대로 쓴다.
    per_patient: dict[str, set] = {}
    red_flag: dict[str, bool] = {}
    anonymous = 0
    for r in new_fmt:
        pid = str(r.get("patient_id") or "").strip()
        if not pid:
            anonymous += 1
            continue
        ids = per_patient.setdefault(pid, set())
        ids.update(x for x in r["rule_ids"] if isinstance(x, str))
        is_red = (str(r.get("risk_level")) in RED_LEVELS
                  or str(r.get("rule_level")) in RED_LEVELS)
        red_flag[pid] = red_flag.get(pid, False) or is_red
    n = len(per_patient)
    say(f"  고유 환자         {n:,}명  (요청 {n_rows:,}행)")
    if anonymous:
        say(f"  ⚠ patient_id 없는 행 {anonymous:,}개 — **환자 지표(ⓐ·①)에서 제외**했다.")
        say("     요청 기준 지표(ⓑ)와 사유 없는 Red 판정에는 그대로 포함된다.")
    if n == 0:
        say()
        say("  ✗ patient_id 가 있는 행이 없다. 환자 단위 지표를 낼 수 없다.")
        return EXIT_NO_DATA

    hits = Counter()
    for ids in per_patient.values():
        for rid in ids:
            hits[rid] += 1

    # ── ① 규칙별 발화 환자 수 ───────────────────────────────────────────
    head("① 규칙별 발화 환자 수 (Top-10)")
    say(f"  {'규칙':<8}{'임상 내용':<34}{'환자':>8}{'비율':>9}")
    silent = []
    for rid in TOP_RULES:
        c = hits.get(rid, 0)
        if not c:
            silent.append(rid)
        say(f"  {rid:<8}{TOP_LABEL[rid]:<34}{c:>8,}{c / n:>8.2%}"
            f"{'' if c else '  ← 무발화'}")

    other = sorted(k for k in hits if k not in TOP_RULES)
    if other:
        head("① -2 그 밖의 규칙 ID")
        for k in other:
            say(f"  {k:<44}{hits[k]:>8,}{hits[k] / n:>8.2%}")

    n_other = sum(int(r.get("n_other_reasons") or 0) for r in new_fmt)
    if n_other:
        say()
        say(f"  ID 형식이 아닌 사유 {n_other:,}건은 집계에서 제외했다"
            " (중복탐지·ML 확률 등 값이 매번 달라지는 문구).")

    # ── ② 환자 단위 발화율 ──────────────────────────────────────────────
    any_top = sum(1 for ids in per_patient.values() if ids & set(TOP_RULES))
    any_rule = sum(1 for ids in per_patient.values() if ids)
    any_top_rows = sum(1 for r in new_fmt if set(r["rule_ids"]) & set(TOP_RULES))
    head("② 발화율 — 두 기준을 함께 본다")
    say("  ⓐ 기간 유병 (고유 환자 기준) — 관측 창에서 **한 번이라도** 그 규칙이 닿은 환자")
    say(f"     Top-10 중 1개 이상   {any_top:,} / {n:,}명  = {any_top / n:.2%}")
    say(f"     규칙 ID 가 하나라도  {any_rule:,} / {n:,}명  = {any_rule / n:.2%}")
    say()
    say("  ⓑ 요청 기준 — 예측 1회당 얼마나 터지는가")
    say(f"     Top-10 중 1개 이상   {any_top_rows:,} / {n_rows:,}행  = {any_top_rows / n_rows:.2%}")
    say()
    say("  두 값은 다르고, **어느 쪽이 큰지는 고정돼 있지 않다.** 발화하는 환자가 자주")
    say("  재요청하면 ⓑ 가 커지고, 발화하지 않는 환자가 자주 재요청하면 ⓐ 가 커진다.")
    say("  **ⓐ 를 \"운영 발화율\" 로 인용하지 말 것** — 그것은 기간 유병이다.")

    # ── ③ 사유 없는 Red ─────────────────────────────────────────────────
    # 최종 등급만 보면 안 된다 — 탐지 전용에서는 최종 등급이 Red 로 올라가지
    # 않으므로 항상 0 이 나오고 차단이 거짓으로 풀린다.
    reasonless = []
    for r in new_fmt:
        is_red = (str(r.get("risk_level")) in RED_LEVELS
                  or str(r.get("rule_level")) in RED_LEVELS)
        if is_red and not r["rule_ids"] and not int(r.get("n_other_reasons") or 0):
            reasonless.append(r)
    n_red_pat = sum(1 for v in red_flag.values() if v)

    head("③ 사유 없는 Red — A4 활성 차단 항목")
    say(f"  Red 환자        {n_red_pat:,}명  ({n_red_pat / n:.2%})"
        "   ※ 최종 등급 또는 규칙층 등급 기준")
    say(f"  사유 0건 요청   {len(reasonless):,}행  ※ 환자가 아니라 **요청 행** 기준 — 한 요청이라도 있으면 차단이다")
    if reasonless:
        say()
        say("  ✗ 차단 유지. 약사가 근거 없이 즉각 개입 지시를 받는 경우가 있다.")
        say("    발생 일자: " + ", ".join(sorted({_part(r) for r in reasonless})[:6]))
    else:
        say()
        say("  사유 0건 Red 는 관측되지 않았다.")
        say("  → 이것만으로 A4 차단이 해제되지는 않는다. 관측 창·트래픽 대표성을")
        say("    함께 확인한 뒤 계획서 S2.2a 의 해제 판단에 넣을 것.")

    # ── 판정 ────────────────────────────────────────────────────────────
    head("판정")
    if silent:
        say(f"  무발화 규칙 {len(silent)}종: {' '.join(silent)}")
        say("    → 해소 결함인지 실제로 그 병용이 없는 것인지 구분이 필요하다.")
        say("      scripts/ops/a3_remeasure.py 로 코퍼스 상한을 먼저 확인할 것.")
    else:
        say("  Top-10 전량이 운영 트래픽에서 관측됐다.")

    rc = EXIT_OK
    if reasonless:
        rc = EXIT_REASONLESS_RED
        say("  종료 코드 3 — 사유 없는 Red 발견. A4 차단 유지.")
    elif len(parts) < 7:
        rc = EXIT_SHORT_WINDOW
        say(f"  종료 코드 4 — 관측 {len(parts)}일. 런북은 최소 1주를 요구한다(요일 편향).")

    say()
    say("  이 수치는 관측 기간·트래픽에 묶인다. 비율만 떼어 인용하면 근거가 사라진다.")

    if args.out:
        try:
            Path(args.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
            print(f"\n저장: {args.out}")
        except OSError as e:
            # 저장 실패를 문서에 없는 exit 1 로 흘리면 반환값 계약이 깨진다.
            print(f"\n리포트 저장 실패: {args.out} — {e}")
            return EXIT_NO_DATA
    return rc


if __name__ == "__main__":
    sys.exit(main())
