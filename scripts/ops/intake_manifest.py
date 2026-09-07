#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M9 — 폐쇄망 반입 무결성 매니페스트 (개선계획 1단계 배포 위생).

폐쇄망으로 들여오는 산출물(코드·휠하우스·참조 데이터)에 대해 **무결성 검증
절차가 정의돼 있지 않았다**(기술검토 P1-13). 이 도구는 그 절차의 기계 부분이다.

읽기만 한다. 표준 라이브러리만 쓴다 — 반입 시점에는 venv 가 아직 없다.

## 이 도구가 증명하지 못하는 것 — 먼저 적는다

**대상과 같은 매체에 실려 온 매니페스트로는 공급망 위조를 잡을 수 없다.**
매체를 바꿀 수 있는 사람은 매니페스트도 함께 바꾼다. 그래서 이 도구는
매니페스트 **자신의 SHA-256** 을 따로 내고, 그 값을 **다른 경로**(전화·사내
메신저·별도 인쇄물)로 받은 값과 대조하도록 한다. 그 대조를 하지 않으면
이 검증은 "매체 안에서 자기 자신과 일치한다" 는 것만 말한다.

**서명 체계가 아니다.** 서명·인증서·타임스탬프는 이 도구의 범위 밖이며,
필요하다면 별도 결정이다.

## 종료 코드 — 이 도구 전용 계약

  0  전량 일치
  2  **검증 불가** — 매니페스트가 없거나 읽히지 않는다 (통과 아님)
  3  변조·누락 발견
  4  매니페스트에 없는 파일만 발견 (반입 절차 누락일 수 있다)

우선순위는 2 > 3 > 4. 저장소의 다른 ops 도구가 비정상을 1 로 내는 것과 다르게
나눈 이유는, 운영자가 화면 문구를 읽지 않고 반환값만 보는 경우에도 **"검증
못 함" 과 "검증했고 이상 없음" 이 섞이지 않게** 하기 위해서다.

사용:
    python scripts/ops/intake_manifest.py generate <반입폴더> --out MANIFEST.json
    python scripts/ops/intake_manifest.py verify   <반입폴더> --manifest MANIFEST.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

EXIT_OK = 0
EXIT_NO_MANIFEST = 2
EXIT_MISMATCH = 3
EXIT_UNLISTED = 4

MANIFEST_SCHEMA_VERSION = 1
_CHUNK = 64 * 1024


def file_digest(path: Path) -> str:
    """SHA-256. 휠하우스는 크므로 통째로 읽지 않고 스트리밍한다."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk(root: Path, exclude: Path | None = None, problems: list | None = None):
    """대상 파일 목록. 경로는 루트 기준 상대·슬래시 구분으로 통일한다.

    `exclude` 는 **매니페스트 자신**이다. 절차상 매니페스트를 반입 폴더 안에 넣으므로,
    제외하지 않으면 정상 반입본이 "목록 외 파일" 로 잡히고 재생성 시에는 이전
    매니페스트를 해싱해 넣은 뒤 덮어써 다음 검증이 불일치로 떨어진다.

    `problems` 를 주면 **접근할 수 없는 항목을 거기에 모으고 훑기를 계속한다.**
    `is_file()` 은 권한 오류를 삼키지 않고 다시 던지므로, 그대로 두면 **잠긴 파일
    하나가 전체를 "검증 불가" 로 떨어뜨린다.** 잠긴 것은 그 파일 하나다.
    """
    ex = None
    if exclude is not None:
        try:
            ex = Path(exclude).resolve()
        except OSError:
            ex = None
    for p in sorted(root.rglob("*")):
        try:
            if not p.is_file():
                continue
        except OSError:
            if problems is not None:
                problems.append(p.relative_to(root).as_posix())
            continue
        if ex is not None:
            try:
                if p.resolve() == ex:
                    continue
            except OSError:
                pass
        yield p, p.relative_to(root).as_posix()


def _self_digest() -> str:
    """검증기 자신의 코드 해시. **진본성 보장이 아니라 출처 추적용이다** —
    같은 매체로 온 검증기라면 이 값도 함께 바뀔 수 있다."""
    try:
        return file_digest(Path(__file__))
    except OSError:
        return ""


def generate(root, out_path, *, intake_id: str = "", target_platform: str = "") -> Path:
    """반입 폴더를 훑어 매니페스트를 만든다. **원본 쪽에서 만드는 것이 전제다.**

    매니페스트 자신은 목록에서 제외한다 — 절차상 반입 폴더 안에 넣기 때문이다.
    """
    root = Path(root)
    out = Path(out_path)
    files = [
        {"path": rel, "size": p.stat().st_size, "sha256": file_digest(p)}
        for p, rel in _walk(root, exclude=out)
    ]
    doc = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file_count": len(files),
        "intake_id": intake_id or "",
        "target_platform": target_platform or "",
        "generator_sha256": _self_digest(),
        # 파일 이름·크기·해시만 담는다. 내용은 담지 않는다 — 매니페스트는
        # 반입 매체에 함께 실려 다니고, 사람이 눈으로 읽는다.
        "files": files,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def manifest_digest(manifest_path) -> str:
    """매니페스트 자신의 해시. **다른 경로로 받은 값과 대조할 것.**"""
    return file_digest(Path(manifest_path))


def _load(mp: Path):
    """매니페스트를 읽고 **구조까지** 검사한다. 하나라도 어긋나면 검증 불가다."""
    doc = json.loads(mp.read_text(encoding="utf-8"))
    if doc.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"schema_version={doc.get('schema_version')}")
    files = doc["files"]
    if not isinstance(files, list) or not files:
        # 빈 목록으로 빈 폴더를 통과시키면 반입 실패가 초록불이 된다.
        raise ValueError("files 가 비었거나 목록이 아니다")
    paths = [e["path"] for e in files]
    if len(set(paths)) != len(paths):
        # dict 로 접으면 마지막 항목이 이기고 앞의 해시가 조용히 사라진다.
        raise ValueError("중복 경로")
    if doc.get("file_count") != len(files):
        raise ValueError(f"file_count={doc.get('file_count')} ≠ 목록 {len(files)}")
    return {e["path"]: e for e in files}


def verify(root, manifest_path) -> dict:
    """반입 폴더를 매니페스트와 대조한다. **계약 밖 예외를 내지 않는다.**"""
    root, mp = Path(root), Path(manifest_path)
    empty = {"changed": [], "missing": [], "unlisted": [], "unreadable": [], "ok": 0}
    try:
        listed = _load(mp)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(empty, exit_code=EXIT_NO_MANIFEST,
                    reason=f"매니페스트를 읽을 수 없다: {type(exc).__name__}: {exc}")

    walk_problems: list[str] = []
    try:
        present = {rel: p for p, rel in _walk(root, exclude=mp, problems=walk_problems)}
    except OSError as exc:
        return dict(empty, exit_code=EXIT_NO_MANIFEST,
                    reason=f"반입 폴더를 훑을 수 없다: {type(exc).__name__}: {exc}")

    changed, missing, ok = [], [], 0
    unreadable = list(walk_problems)
    for rel, entry in sorted(listed.items()):
        p = present.get(rel)
        if p is None:
            # 목록에 없는 이유가 둘이다 — 정말 없는 것과, 있는데 못 읽는 것.
            # `is_file()` 은 접근 오류를 False 로 삼키므로 여기서 갈라 준다.
            # 운영자가 지운 파일을 찾을지 잠금을 풀지 알아야 한다.
            if rel in unreadable:
                continue          # 훑는 중에 이미 접근 불가로 잡혔다
            try:
                os.stat(root / rel)
                unreadable.append(rel)   # 있는데 목록에 안 들어왔다 = 접근 문제
            except FileNotFoundError:
                missing.append(rel)
            except OSError:
                unreadable.append(rel)
            continue
        try:
            same = p.stat().st_size == entry.get("size") and file_digest(p) == entry.get("sha256")
        except OSError:
            # 잠금·권한으로 못 읽는 것은 "일치" 도 "불일치" 도 아니다. 통과시키지 않는다.
            unreadable.append(rel)
            continue
        if same:
            ok += 1
        else:
            changed.append(rel)
    unlisted = sorted(set(present) - set(listed) - set(unreadable))
    unreadable = sorted(set(unreadable))

    if changed or missing or unreadable:
        code = EXIT_MISMATCH
    elif unlisted:
        code = EXIT_UNLISTED
    else:
        code = EXIT_OK
    return {"exit_code": code, "reason": "", "changed": changed, "missing": missing,
            "unlisted": unlisted, "unreadable": unreadable, "ok": ok}


def _report(r: dict, manifest_path: Path) -> None:
    print(f"매니페스트: {manifest_path}")
    if r["exit_code"] == EXIT_NO_MANIFEST:
        print(f"  {r['reason']}")
        print("  → **검증하지 못했다.** 이상 없음이 아니다. 반입 절차를 확인할 것.")
        return
    print(f"  자기 해시: {manifest_digest(manifest_path)}")
    print("  → 이 값을 **다른 경로로 받은 값**과 대조할 것. 대조하지 않으면")
    print("     이 검증은 매체 안에서 자기 자신과 일치한다는 것만 말한다.")
    print(f"\n  일치 {r['ok']}건 · 변조 {len(r['changed'])}건 · 누락 {len(r['missing'])}건 · "
          f"읽기 실패 {len(r['unreadable'])}건 · 목록 외 {len(r['unlisted'])}건")
    for label, items in (("변조", r["changed"]), ("누락", r["missing"]),
                         ("읽기 실패", r["unreadable"]), ("목록 외", r["unlisted"])):
        for name in items[:50]:
            print(f"    [{label}] {name}")
        if len(items) > 50:
            print(f"    [{label}] … 외 {len(items) - 50}건")
    verdict = {
        EXIT_OK: "전량 일치",
        EXIT_MISMATCH: "변조·누락·읽기 실패 — 반입본을 쓰지 말 것",
        EXIT_UNLISTED: "목록 외 파일 있음 — 반입 절차 누락 여부 확인",
    }[r["exit_code"]]
    print(f"\n  판정: {verdict}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="폐쇄망 반입 무결성 매니페스트 (읽기 전용)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="원본 쪽에서 매니페스트 생성")
    g.add_argument("root")
    g.add_argument("--out", default="MANIFEST.json")
    g.add_argument("--intake-id", default="", help="반입 건/릴리스 식별자")
    g.add_argument("--target-platform", default="", help="대상 플랫폼·Python/ABI 조건")
    v = sub.add_parser("verify", help="반입 쪽에서 대조")
    v.add_argument("root")
    v.add_argument("--manifest", default="MANIFEST.json")
    a = ap.parse_args(argv)

    if a.cmd == "generate":
        out = generate(a.root, a.out, intake_id=a.intake_id,
                       target_platform=a.target_platform)
        doc = json.loads(out.read_text(encoding="utf-8"))
        print(f"매니페스트 생성: {out}  ({doc['file_count']}개 파일)")
        print(f"  자기 해시: {manifest_digest(out)}")
        print("  → 이 값을 **매체와 다른 경로**로 반입 담당자에게 전달할 것.")
        return EXIT_OK

    r = verify(a.root, a.manifest)
    print(f"검증기 {Path(__file__).name} · 코드 해시 {_self_digest()[:16]} · "
          f"python {sys.version.split()[0]}")
    _report(r, Path(a.manifest))
    return r["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
