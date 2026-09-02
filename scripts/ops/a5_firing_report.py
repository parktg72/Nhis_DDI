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
    """JSONL 을 읽어 (레코드, 파싱실패수) 반환. 깨진 줄은 세고 넘어간다."""
    rows, broken = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                broken += 1
                continue
            part = r.get("partition") or (r.get("timestamp") or "")[:10]
            if since and part < since:
                continue
            if until and part > until:
                continue
            rows.append(r)
    return rows, broken


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

    rows, broken = load(path, args.since, args.until)
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
    if broken:
        say(f"  파싱 실패         {broken:,}줄")

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

    # ── 환자 단위로 접는다 ────────────────────────────────────────────────
    # 리포트가 "환자" 라고 쓰므로 실제로 환자로 세야 한다. 같은 환자의 여러 요청을
    # 그대로 세면 재요청이 많은 환자가 비율을 끌어올린다.
    per_patient: dict[str, set] = {}
    red_flag: dict[str, bool] = {}
    for r in new_fmt:
        pid = str(r.get("patient_id") or "")
        ids = per_patient.setdefault(pid, set())
        ids.update(x for x in r["rule_ids"] if isinstance(x, str))
        # 최종 등급과 규칙층 등급 중 하나라도 Red 면 Red 환자다
        is_red = (str(r.get("risk_level")) in RED_LEVELS
                  or str(r.get("rule_level")) in RED_LEVELS)
        red_flag[pid] = red_flag.get(pid, False) or is_red
    n = len(per_patient)
    say(f"  고유 환자         {n:,}명  (요청 {len(new_fmt):,}행)")

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
    head("② 환자 단위 발화율")
    say(f"  Top-10 중 1개 이상   {any_top:,} / {n:,}  = {any_top / n:.2%}")
    say(f"  규칙 ID 가 하나라도  {any_rule:,} / {n:,}  = {any_rule / n:.2%}")

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
    say(f"  사유 0건 요청   {len(reasonless):,}행")
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
        Path(args.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
        print(f"\n저장: {args.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
