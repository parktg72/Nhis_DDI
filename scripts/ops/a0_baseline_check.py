#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A0 — 운영 PC 기준선 확인 (개선계획 0단계).

운영 PC 가 **무엇을 돌리고 있는지** 를 네 축으로 확인한다. 이 값이 없으면
P0-1(안전망) 의 대상이 특정되지 않아 어떤 조치의 효과도 측정할 수 없다.

  ① 서빙 소스 상태   — 어느 계열인지 (Step 0 이름해소 유무)
  ② 서빙 플래그 값   — /health 의 serving_flags. 키 부재 = 구코드
  ③ 배포 번들 동일성 — 모델 파일 SHA-256 ↔ 번들 메타 기록값, 재현 정보 유무
  ④ 개입 전달 경로   — 발송 구현이 실재하는지

읽기만 한다. 파일·설정·프로세스를 바꾸지 않으며 네트워크는 localhost 만 쓴다.

사용:
    python scripts/ops/a0_baseline_check.py
    python scripts/ops/a0_baseline_check.py --api http://localhost:8000 --out a0_report.txt

폐쇄망 Windows 기준. 표준 라이브러리만 쓰므로 venv 없이도 돈다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# ── 기준값 ────────────────────────────────────────────────────────────────
# 개행 정규화(CR 제거) 후 SHA-256 앞 16자. Windows/Unix 체크아웃 차이를 흡수한다.
# 세대 구분이 2026-09-02 에 바뀌었다. A1 병합 전에는 "main = Step 0 없음 /
# 안전망 = 별도 브랜치" 였으나, 병합으로 안전망 코드가 main 이 됐다. 지금 구분은
# "구판(병합 전)" 과 "현판(병합 후)" 이다.
SOURCE_FINGERPRINTS = {
    "serving/predictor.py": {
        "83a2821467b45c44": "구판 — Step 0 이름해소 자체가 없음 (A1 병합 전 main)",
        "368c65a1edfec681": "구판 — 안전망 브랜치 (A1 병합 전)",
        "eed5448d36477ab8": "구판 — 안전망 + A2 사유 병합 (A1 병합 전)",
        "f6ce1a0c379c1c44": "현판 — A1·RS1~RS3·A1a 병합 후. 세 플래그 전부 존재",
    },
    "serving/routers/health.py": {
        "935f4c8f22770966": "구판 — /health 에 serving_flags 없음",
        "e7555b7ae3803e9c": "현판 — serving_flags 노출",
    },
    "serving/schemas.py": {
        "774337c64fdb93fd": "구판",
        "e3ffc338f8cde3d1": "현판",
    },
}
BUNDLE_DEFAULT = "hana_app/models/hierarchical/retrain_prod_0711_hierarchy_cur"
DELIVERY_MARKERS = ("smtplib", "twilio", "solapi", "kakao", "send_sms", "sendmail")

OUT: list[str] = []


def say(line: str = "") -> None:
    print(line)
    OUT.append(line)


def head(title: str) -> None:
    say()
    say(f"{'─' * 72}")
    say(f" {title}")
    say(f"{'─' * 72}")


def digest(path: Path, *, normalize: bool) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if normalize:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


# ── ① 서빙 소스 상태 ──────────────────────────────────────────────────────
def check_source(root: Path) -> str:
    head("① 서빙 소스 상태")
    verdicts: list[str] = []
    for rel, table in SOURCE_FINGERPRINTS.items():
        p = root / rel
        d = digest(p, normalize=True)
        if d is None:
            say(f"  {rel:<28} 파일 없음")
            verdicts.append("missing")
            continue
        short = d[:16]
        label = table.get(short, "알 수 없는 판 — 로컬 수정 또는 제3의 계열")
        say(f"  {rel:<28} {short}  {label}")
        verdicts.append(
            "current" if "현판" in label else
            "old" if "구판" in label else "unknown"
        )
    uniq = set(verdicts)
    if uniq == {"current"}:
        v = ("현판 — 세 플래그가 모두 코드에 있다. 남은 것은 활성 결정이며, "
             "탐지만 켜려면 이름 해소 + 탐지 전용을 함께 켠다(배포 런북 참조).")
    elif uniq == {"old"}:
        v = "구판 — 배포본이 A1 병합 전이다. 재배포가 선행이다."
    else:
        v = f"혼재/불명 {sorted(uniq)} — 배포본이 단일 판이 아니다. 개별 대조 필요."
    say(f"\n  판정: {v}")
    return v


# ── ② 서빙 플래그 ────────────────────────────────────────────────────────
def check_flags(api: str) -> str:
    head("② 서빙 플래그 (/health)")
    url = api.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            body = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        say(f"  {url} 조회 실패: {type(e).__name__}: {e}")
        say("  → API 미기동이거나 포트가 다르다. --api 로 지정하거나 서버 기동 후 재실행.")
        return "UNKNOWN — /health 조회 실패"

    say(f"  status={body.get('status')}  model_mode={body.get('model_mode')}")
    say(f"  hierarchical_loaded={body.get('hierarchical_loaded')}  dl_loaded={body.get('dl_loaded')}")
    drift = body.get("schema_drift")
    if drift:
        say(f"  ⚠ schema_drift={drift}")

    if "serving_flags" not in body:
        say("\n  serving_flags 키 없음")
        say("  → 구코드다. 플래그 경로 자체가 존재하지 않으므로 Top-10 은 EDI-only 요청에서 무발화다.")
        return "구코드 — 플래그 경로 없음"

    flags = body["serving_flags"] or {}
    for k, v in sorted(flags.items()):
        say(f"  {k} = {v}")
    on = [k for k, v in flags.items() if v]
    if on:
        say(f"\n  ⚠ 활성 플래그 {on}")
        say("  → 이미 켜져 있다. 즉각 개입 부하가 발생 중일 수 있으므로 건수를 즉시 확인할 것.")
        return f"활성 — {on}"
    say("\n  전부 비활성 — 종전 동작. 활성화는 운영 용량 결정(A4).")
    return "비활성"


# ── ③ 배포 번들 ──────────────────────────────────────────────────────────
def check_bundle(root: Path) -> str:
    head("③ 배포 번들 동일성")
    env_dir = os.environ.get("HIERARCHICAL_MODEL_DIR")
    say(f"  HIERARCHICAL_MODEL_DIR = {env_dir or '(미설정 — 기본 경로 사용)'}")
    bundle = Path(env_dir) if env_dir else (root / BUNDLE_DEFAULT)
    say(f"  대상: {bundle}")
    meta_p = bundle / "stage_meta.json"
    if not meta_p.exists():
        say("  stage_meta.json 없음 — 번들 경로를 확인할 것.")
        return "번들 없음"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    say(f"  clinical_standards_version = {meta.get('clinical_standards_version')}")
    say(f"  feature_semantics_version  = {meta.get('feature_semantics_version')}")
    say(f"  stage1_trained={meta.get('stage1_trained')}  stage1_red_count={meta.get('stage1_red_count')}")
    th = meta.get("thresholds") or {}
    if th:
        gap = None
        try:
            gap = float(th.get("tau_red", 0)) - float(th.get("tau_review", 0))
        except (TypeError, ValueError):
            pass
        say(f"  thresholds tau_red={th.get('tau_red')} tau_review={th.get('tau_review')}"
            + (f"  (간격 {gap:.1e})" if gap is not None else ""))
        if gap is not None and gap <= 1e-5:
            say("  ⚠ 임계 밴드 붕괴 — 검수 큐가 사실상 비어 있다.")

    ok = True
    for stage, fname in (("stage1", "stage1_red.joblib"), ("stage2", "stage2_yellow.joblib")):
        recorded = meta.get(f"{stage}_sha256")
        actual = digest(bundle / fname, normalize=False)
        if actual is None:
            say(f"  {fname:<24} 파일 없음"); ok = False; continue
        mark = "일치" if recorded == actual else "불일치"
        if recorded != actual:
            ok = False
        say(f"  {fname:<24} {actual[:16]}  기록 {str(recorded)[:16]}  {mark}")
    v = "일치 — 번들이 메타와 같다" if ok else "불일치 — 배포본 재식별 필요"

    # M-2 이후 번들에는 재현 정보가 실린다. 없으면 그 이전에 만든 번들이다.
    prov = meta.get("provenance")
    if prov:
        period = prov.get("period") or {}
        code = prov.get("code") or {}
        say()
        say(f"  재현 정보 — 입력 {prov.get('input_file_count')}개 파일")
        if period:
            say(f"    기간   {period.get('from')} ~ {period.get('to')} ({period.get('source')} 파생)")
        else:
            say(f"    기간   확인 불가 ({prov.get('period_reason')})")
        if code.get("source") == "git":
            dirty = code.get("dirty")
            say(f"    코드   git {str(code.get('commit'))[:12]}"
                + ("  ⚠ 커밋되지 않은 변경 있음" if dirty else ""))
        else:
            say("    코드   git 기록 없음 — 소스 지문으로 대조할 것")
        for rel, fp in (code.get("fingerprints") or {}).items():
            label = SOURCE_FINGERPRINTS.get(rel, {}).get(fp or "", "")
            say(f"      {rel:<34} {fp or '파일 없음'}  {label}")
        say(f"    생성   {prov.get('build_time_utc')}")
        v += " · 재현 정보 있음"
    else:
        say("  주의: 메타에 입력 파일 목록·기간·코드 커밋이 없다. 학습 코호트는 이 값으로 재현되지 않는다.")
        v += " · 재현 정보 없음(M-2 이전 번들)"

    say(f"\n  판정: {v}")
    return v


# ── ④ 개입 전달 경로 ──────────────────────────────────────────────────────
def check_delivery(root: Path) -> str:
    head("④ 개입 전달 경로")
    hits: list[str] = []
    for base in ("serving", "hana_app", "dags", "scripts"):
        d = root / base
        if not d.is_dir():
            continue
        for p in d.rglob("*.py"):
            if p.resolve() == Path(__file__).resolve():
                continue            # 이 스크립트 자신이 마커 문자열을 담고 있다
            if "test" in p.name or "_internal" in str(p):
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for m in DELIVERY_MARKERS:
                if m in txt:
                    hits.append(f"{p.relative_to(root)} :: {m}")
    if hits:
        say("  발송 관련 참조 발견:")
        for h in hits[:20]:
            say(f"    {h}")
        v = "발송 경로 존재 가능 — 개별 확인 필요"
    else:
        say("  SMS·메일·알림톡 발송 구현 없음.")
        say("  개입은 API 응답 필드와 오프라인 산출물(대상자 CSV·DOCX 보고서)로만 전달된다.")
        v = "미구현 — 자동 발송 없음"
    say(f"\n  판정: {v}")
    say("  이 값은 규제 유권해석(B2) 의뢰서에 그대로 기재한다. 설계와 현행을 섞지 말 것.")
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="A0 운영 기준선 확인 (읽기 전용)")
    ap.add_argument("--api", default="http://localhost:8000", help="서빙 API 베이스 URL")
    ap.add_argument("--root", default=".", help="저장소 루트")
    ap.add_argument("--out", default="a0_baseline_report.txt", help="리포트 저장 경로")
    a = ap.parse_args()
    root = Path(a.root).resolve()

    say("A0 — 운영 PC 기준선 확인")
    say(f"실행 {datetime.now():%Y-%m-%d %H:%M:%S}  ·  호스트 {os.environ.get('COMPUTERNAME') or os.uname().nodename}")
    say(f"루트 {root}")
    say(f"python {sys.version.split()[0]}")

    v1 = check_source(root)
    v2 = check_flags(a.api)
    v3 = check_bundle(root)
    v4 = check_delivery(root)

    head("요약 — 개선계획 A0 산출물")
    say(f"  ① 소스 상태     {v1}")
    say(f"  ② 플래그        {v2}")
    say(f"  ③ 번들          {v3}")
    say(f"  ④ 전달 경로     {v4}")
    say()
    say("  다음: ①②가 A4(활성 방식 결정)의 입력이고, ④는 B2 의뢰서에 그대로 들어간다.")

    try:
        Path(a.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
        print(f"\n리포트 저장: {Path(a.out).resolve()}")
    except OSError as e:
        print(f"\n리포트 저장 실패: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
