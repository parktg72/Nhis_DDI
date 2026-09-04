#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M-2 — 학습 번들에 재현 정보를 스탬프한다 (개선계획 1단계 선행 구현).

배포 번들의 메타에는 피처명·라벨 수·모델 SHA-256 만 있었다. 입력 파일 목록도,
기간도, 코드 참조도 없어서 **배포 코호트를 재현할 수 없었다**(차단 요인 B-1).
28배 수치가 커밋 메시지에만 남아 있던 것과 같은 종류의 부채다.

여기서 만드는 것은 그 부채를 앞으로 쌓지 않게 하는 스탬프다. 이미 배포된
번들을 소급 복원하지는 못한다 — 그것은 M-1(기준 코호트 v2)의 몫이다.

**기록하는 것** — 입력 파일의 이름·크기·SHA-256, 파일 수, glob 패턴, 기간,
코호트 파라미터(seed·window_days·poly_threshold), 코드 참조(git 커밋 또는
소스 지문), 생성 시각.

**기록하지 않는 것** — 행 수, 환자 수, 환자 식별자, 그 밖에 데이터 내용에서
파생된 어떤 값도. 번들 메타는 배포물과 함께 이동하므로 민감정보가 실려서는
안 된다. 파일 이름(`records_YYYYMMDD.parquet`)은 날짜이며 식별자가 아니다.

값은 모두 ASCII 로 둔다. 폐쇄망 Windows 콘솔(cp949)에서 메타를 그대로 열어
읽는 경우가 있어서다. 판정 문구는 이 파일이 아니라 A0 리포트가 낸다.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROVENANCE_SCHEMA_VERSION = 1

_CHUNK = 64 * 1024

# 번들이 어느 코드로 만들어졌는지. A0 지문표와 같은 어휘(개행 정규화 SHA-256
# 앞 16자)를 쓴다 — 두 곳의 숫자가 같아야 대조가 성립한다.
# A0 는 폐쇄망에 단독 복사해 돌리는 도구라 import 하지 않는다(표준 라이브러리
# 전용·의존 없음이 그 파일의 계약). 두 구현이 갈라지는 것은 테스트로 막는다.
FINGERPRINT_SOURCES = (
    "serving/predictor.py",
    "serving/schemas.py",
    "hana_app/core/hierarchical_runner.py",
    "hana_app/core/ml_runner.py",
)

_DATE_RE = re.compile(r"(\d{8}|\d{6})")


def file_digest(path: Path) -> str:
    """파일의 SHA-256. 대용량 raw 를 통째 메모리에 올리지 않고 스트리밍한다."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_digest(path: Path) -> str | None:
    """개행 정규화 후 SHA-256 앞 16자. 없으면 None.

    A0 의 `digest(normalize=True)[:16]` 과 같은 값이어야 한다.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()[:16]


def _period(names: list[str]) -> tuple[dict | None, str | None]:
    stamps: list[str] = []
    for name in names:
        m = _DATE_RE.search(name)
        if not m:
            return None, "unparseable_filename"
        stamps.append(m.group(1))
    if len({len(s) for s in stamps}) > 1:
        # 6자리(월)와 8자리(일)를 섞으면 문자열 비교가 틀린 기간을 만든다.
        return None, "mixed_granularity"
    return {"from": min(stamps), "to": max(stamps), "source": "filename"}, None


def _code_ref(code_root: Path) -> dict:
    commit: str | None = None
    dirty: bool | None = None
    source = "none"
    try:
        rev = subprocess.run(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if rev.returncode == 0 and rev.stdout.strip():
            commit = rev.stdout.strip()
            source = "git"
            st = subprocess.run(
                ["git", "-C", str(code_root), "status", "--porcelain"],
                capture_output=True, text=True, timeout=10,
            )
            if st.returncode == 0:
                dirty = bool(st.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        # 운영 PC 에는 git 이 없다. 지문으로 대신한다.
        pass
    return {
        "source": source,
        "commit": commit,
        "dirty": dirty,
        # 없는 파일도 키를 남긴다 — 키 부재와 파일 부재를 구분하기 위해서다.
        "fingerprints": {
            rel: normalized_digest(code_root / rel) for rel in FINGERPRINT_SOURCES
        },
    }


def collect_provenance(
    raw_paths,
    *,
    code_root,
    glob_patterns=None,
    seed=None,
    window_days=None,
    poly_threshold=None,
) -> dict:
    """학습 **전에** 호출한다 — 읽으려는 그 파일의 해시여야 하고, 파일이
    없으면 몇 분짜리 학습을 시작하기 전에 멈추는 편이 낫다.
    """
    paths = sorted(Path(p) for p in raw_paths)
    files = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"입력 파일 없음: {p}")
        files.append({
            "name": p.name,
            "size": p.stat().st_size,
            "sha256": file_digest(p),
        })

    period, reason = _period([f["name"] for f in files])
    prov = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "build_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_file_count": len(files),
        "input_files": files,
        "glob_patterns": list(glob_patterns) if glob_patterns else None,
        "period": period,
        "cohort_params": {
            "seed": seed,
            "window_days": window_days,
            "poly_threshold": poly_threshold,
        },
        "code": _code_ref(Path(code_root)),
    }
    if reason:
        prov["period_reason"] = reason
    return prov


def stamp_bundle(output_dir, provenance: dict) -> Path:
    """번들의 `stage_meta.json` 에 `provenance` 키를 더한다. 다른 키는 건드리지 않는다."""
    meta_p = Path(output_dir) / "stage_meta.json"
    if not meta_p.exists():
        raise FileNotFoundError(f"stage_meta.json 없음: {meta_p}")
    raw = meta_p.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # 인코딩을 지정하지 않고 쓰인 구 번들(Windows 기본 cp949) 대비.
        text = raw.decode("cp949")
    meta = json.loads(text)
    meta["provenance"] = provenance
    meta_p.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta_p
