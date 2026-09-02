"""A5 — 운영 트래픽에서 규칙이 실제로 얼마나 발화하는지 집계한다.

입력은 서빙이 남기는 메트릭 JSONL(`DDI_METRICS_JSONL_PATH`). 읽기 전용이며
표준 라이브러리만 쓴다 — 폐쇄망 운영 PC 에서 venv 없이 돌아야 한다.

산출은 배포 런북 5절의 세 항목이다.
  ① 규칙별 발화 환자 수      — A5 의 본체
  ② 환자 단위 발화율          — 요청 중 규칙이 하나라도 붙은 비율
  ③ 사유 없는 Red 건수        — A4 활성의 차단 항목. 0 이어야 해제된다

사용:
    python3 scripts/ops/a5_firing_report.py --path /app/data/monitoring/metrics_live.jsonl
    python3 scripts/ops/a5_firing_report.py --path <경로> --since 2026-09-05 --out a5.txt

주의 — 이 도구는 `rule_ids` 필드가 있는 레코드만 발화 집계에 넣는다. 그 필드는
2026-09-02 에 추가됐으므로, 그 전 레코드는 "구형식" 으로 따로 세어 보고한다.
구형식만 있는데 발화 0 으로 읽으면 "규칙이 안 터진다" 는 없는 결론이 나온다.
"""
from __future__ import annotations

import argparse
import json
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


def main(argv=None) -> int:
    args = parse_args(argv)
    path = Path(args.path)

    say("A5 — 운영 발화 집계")
    say(f"실행 {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"입력 {path}")
    if args.since or args.until:
        say(f"기간 {args.since or '처음'} ~ {args.until or '끝'}")

    if not path.exists():
        say(f"\n입력 파일이 없다: {path}")
        say("  → 서빙이 아직 기록하지 않았거나 경로가 다르다. DDI_METRICS_JSONL_PATH 확인.")
        return 2

    rows, broken = load(path, args.since, args.until)
    if not rows:
        say("\n집계 대상 레코드가 0건이다. 기간 조건 또는 기록 여부를 확인할 것.")
        return 2

    # ── 형식 구분 ────────────────────────────────────────────────────────
    new_fmt = [r for r in rows if "rule_ids" in r]
    old_fmt = [r for r in rows if "rule_ids" not in r]

    head("0. 입력 상태")
    say(f"  레코드            {len(rows):,}건")
    say(f"  발화 집계 가능    {len(new_fmt):,}건  (rule_ids 있음)")
    say(f"  구형식            {len(old_fmt):,}건  (rule_ids 없음 — 발화 집계 제외)")
    if broken:
        say(f"  파싱 실패         {broken:,}줄")
    parts = sorted({r.get("partition") or (r.get("timestamp") or "")[:10] for r in rows})
    say(f"  관측 일자         {len(parts)}일  {parts[0]} ~ {parts[-1]}" if parts else "")
    if old_fmt and not new_fmt:
        say()
        say("  ⚠ 전부 구형식이다. 발화 0 으로 읽으면 안 된다 — 기록에 규칙 ID 가 없을 뿐이다.")
        say("    서빙을 2026-09-02 이후 판으로 올린 뒤 다시 관측할 것.")
        return 2
    if len(parts) < 7:
        say()
        say(f"  ⚠ 관측 일자 {len(parts)}일 — 런북은 최소 1주를 권한다(요일 편향).")

    # ── ① 규칙별 발화 환자 수 ───────────────────────────────────────────
    hits = Counter()
    for r in new_fmt:
        for rid in r.get("rule_ids") or []:
            hits[rid] += 1

    n = len(new_fmt)
    head("① 규칙별 발화 (Top-10)")
    say(f"  {'규칙':<8}{'임상 내용':<34}{'환자':>8}{'비율':>9}")
    silent = []
    for rid in TOP_RULES:
        c = hits.get(rid, 0)
        mark = "" if c else "  ← 무발화"
        if not c:
            silent.append(rid)
        say(f"  {rid:<8}{TOP_LABEL[rid]:<34}{c:>8,}{c / n:>8.2%}{mark}")

    other = sorted(k for k in hits if k not in TOP_RULES)
    if other:
        head("① -2 그 밖의 사유")
        for k in other:
            say(f"  {k:<44}{hits[k]:>8,}{hits[k] / n:>8.2%}")

    # ── ② 환자 단위 발화율 ──────────────────────────────────────────────
    any_top = sum(1 for r in new_fmt if any(x in TOP_RULES for x in (r.get("rule_ids") or [])))
    any_rule = sum(1 for r in new_fmt if r.get("rule_ids"))
    head("② 환자 단위 발화율")
    say(f"  Top-10 중 1개 이상   {any_top:,} / {n:,}  = {any_top / n:.2%}")
    say(f"  사유가 하나라도 있음  {any_rule:,} / {n:,}  = {any_rule / n:.2%}")

    # ── ③ 사유 없는 Red ─────────────────────────────────────────────────
    reds = [r for r in new_fmt if str(r.get("risk_level")) in RED_LEVELS]
    reasonless = [r for r in reds if not (r.get("rule_ids") or [])]
    head("③ 사유 없는 Red — A4 활성 차단 항목")
    say(f"  Red            {len(reds):,}건  ({len(reds) / n:.2%})")
    say(f"  그중 사유 0건  {len(reasonless):,}건")
    if reasonless:
        say()
        say("  ✗ 차단 유지. 약사가 근거 없이 즉각 개입 지시를 받는 경우가 있다.")
        say("    발생 일자: " + ", ".join(sorted({
            (r.get("partition") or (r.get("timestamp") or "")[:10]) for r in reasonless
        })[:6]))
    else:
        say()
        say("  ✓ 0건. 이 항목의 해제 조건을 충족한다 (계획서 S2.2a).")

    # ── 판정 ────────────────────────────────────────────────────────────
    head("판정")
    if silent:
        say(f"  무발화 규칙 {len(silent)}종: {' '.join(silent)}")
        say("    → 해소 결함인지 실제로 그 병용이 없는 것인지 구분이 필요하다.")
        say("      scripts/ops/a3_remeasure.py 로 코퍼스 상한을 먼저 확인할 것.")
    else:
        say("  Top-10 전량이 운영 트래픽에서 관측됐다.")
    say()
    say("  이 수치는 관측 기간·트래픽에 묶인다. P0-1 종결 판단에 쓰려면 관측 기간과")
    say("  대상 범위를 함께 인용할 것 — 비율만 떼어 인용하면 근거가 사라진다.")

    if args.out:
        Path(args.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
