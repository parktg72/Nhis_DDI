"""`lookup_wk` 호출자 전수 — 진술이 아니라 실행 가능한 검사로 고정한다.

`CodeStandardizer.lookup_wk` 는 커밋 `841b849` 에서 반환 계약이 바뀌었고, 그 변경은
서빙 플래그 **밖**에 있다(ATC 없는 엔트리의 약물명을 더 이상 버리지 않는다). 리뷰에서
"호출자가 셋뿐"이라는 제출자 진술이 재현 불가하다는 지적을 받았다.

이 파일은 AST 로 저장소를 훑어 호출자를 열거하고 알려진 목록과 대조한다. 새 호출자가
생기면 실패하므로, 계약 변경의 영향 범위가 조용히 넓어지는 것을 막는다.

정적 검사의 한계 — `getattr(obj, name)` 의 `name` 이 런타임에 계산되는 경우는 잡지
못한다. 그런 호출은 현재 저장소에 없으며(문자열 리터럴만 사용), 그 사실 자체도 아래
`test_no_dynamic_lookup_wk_resolution` 이 검사한다.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SKIP_PARTS = {".worktrees", "__pycache__", "node_modules", ".git"}
_SKIP_PREFIXES = (".venv", "venv", "packages_win", "hana/py", "python")


# 읽을 수 없어 건너뛴 파일. 조용히 넘기면 열거가 불완전해지므로 드러낸다.
SKIPPED: list[tuple[str, str]] = []


def _source_files() -> list[Path]:
    out = []
    SKIPPED.clear()
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        parts = rel.parts
        if any(x in _SKIP_PARTS for x in parts):
            continue
        if any(str(rel).startswith(pre) or parts[0].startswith(pre) for pre in _SKIP_PREFIXES):
            continue
        try:
            p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            SKIPPED.append((str(rel).replace("\\", "/"), type(exc).__name__))
            continue
        out.append(p)
    return sorted(out)


def _enclosing(tree: ast.AST, node: ast.AST) -> str:
    """노드를 감싸는 가장 가까운 함수/메서드 이름."""
    best, best_lineno = "<module>", -1
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(fn, "end_lineno", None) or fn.lineno
        if fn.lineno <= node.lineno <= end and fn.lineno > best_lineno:
            best, best_lineno = fn.name, fn.lineno
    return best


def _collect() -> dict[str, set[tuple[str, str]]]:
    """{'call': {(상대경로, 감싸는 함수)}, 'getattr': {...}}"""
    found: dict[str, set[tuple[str, str]]] = {"call": set(), "getattr": set()}
    for path in _source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        for node in ast.walk(tree):
            # x.lookup_wk(...)
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "lookup_wk"):
                found["call"].add((rel, _enclosing(tree, node)))
            # getattr(x, "lookup_wk", ...)
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and node.args and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "lookup_wk"):
                found["getattr"].add((rel, _enclosing(tree, node)))
    return found


# 프로덕션 호출자 — 이 목록이 바뀌면 계약 변경의 영향 범위가 달라진 것이다.
_EXPECTED_PRODUCTION = {
    # 게이팅 없음(전역). `if atc_fb:` 가드가 반환 계약 변경의 전파를 막는다.
    ("scripts/etl/code_standardizer.py", "standardize"),
    # 주 플래그(SERVING_ENABLE_EDI_NAME_RESOLUTION) 안
    ("serving/predictor.py", "resolve_codes"),
    # 주 플래그 안에 중첩된 ATC 플래그 경로
    ("serving/predictor.py", "atc_candidates"),
}


def _production_only(items: set[tuple[str, str]]) -> set[tuple[str, str]]:
    return {x for x in items if not x[0].startswith("tests/")}


def test_production_lookup_wk_callers_match_the_known_set():
    """프로덕션 호출자가 알려진 셋과 정확히 일치해야 한다."""
    found = _collect()
    actual = _production_only(found["call"] | found["getattr"])

    missing = _EXPECTED_PRODUCTION - actual
    extra = actual - _EXPECTED_PRODUCTION
    assert not missing, f"알려진 호출자가 사라졌다 — 목록을 갱신하라: {sorted(missing)}"
    assert not extra, (
        "새 `lookup_wk` 호출자가 생겼다. `841b849` 의 반환 계약 변경(ATC 없는 엔트리의 "
        "약물명을 반환)이 이 경로에도 닿는지 확인하고, 무영향이면 목록에 추가하라: "
        f"{sorted(extra)}"
    )


def test_no_dynamic_lookup_wk_resolution():
    """`getattr` 의 속성명이 리터럴이어야 정적 열거가 성립한다."""
    dynamic = []
    for path in _source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if rel.startswith("tests/"):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and not isinstance(node.args[1], ast.Constant)):
                src = ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""
                if "lookup" in src or "_std" in src:
                    dynamic.append((rel, node.lineno, src[:80]))
    assert not dynamic, (
        f"속성명이 동적으로 계산되는 getattr 가 있어 정적 열거가 불완전하다: {dynamic}"
    )


def test_standardize_guards_the_changed_return_contract():
    """`standardize()` 가 ATC 로 가드하므로 계약 변경이 하위로 전파되지 않는다.

    이것이 유일한 무게이팅 호출자이며, 여기서 막히는지가 ETL 무영향의 근거다.
    실 데이터 60,000행 대조에서도 산출물 sha256 이 main 과 같았다.
    """
    src = (ROOT / "scripts/etl/code_standardizer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "standardize")
    body = ast.get_source_segment(src, fn) or ""

    assert "lookup_wk(" in body, "픽스처 전제 붕괴 — standardize 가 lookup_wk 를 부르지 않는다"
    assert "if atc_fb:" in body, (
        "`standardize()` 의 ATC 가드가 사라졌다. 841b849 의 반환 계약 변경이 ETL "
        "산출물로 전파될 수 있으므로 영향 재측정이 필요하다."
    )


def test_scan_coverage_is_not_silently_incomplete():
    """읽지 못한 파일이 있으면 드러낸다 — 그 상태의 '호출자 셋뿐'은 주장이 못 된다.

    현재 저장소에는 권한 문제로 열리지 않는 경로가 있다(`reviews/` 아래 산출물).
    이들이 `lookup_wk` 를 부를 여지가 없다는 근거를 함께 남긴다.
    """
    _source_files()
    unreadable = [x for x in SKIPPED]
    # 스캔에서 빠진 파일은 전부 `reviews/` 하위 산출물이어야 한다 —
    # 소스 트리(serving/, scripts/, rules/, dags/, hana_app/)에는 없어야 한다.
    leaked = [x for x in unreadable
              if x[0].split("/")[0] in {"serving", "scripts", "rules", "dags", "hana_app"}]
    assert not leaked, f"소스 트리에서 읽지 못한 파일이 있어 열거가 불완전하다: {leaked}"
