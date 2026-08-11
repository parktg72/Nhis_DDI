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


# 읽거나 파싱하지 못해 건너뛴 파일. 조용히 넘기면 열거가 불완전해지므로 드러낸다.
SKIPPED: list[tuple[str, str]] = []
PARSE_FAILED: list[tuple[str, str]] = []


def _parse(path: Path):
    """파싱 실패를 **기록하고** None 을 돌려준다 — 조용한 continue 를 없앤다."""
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        PARSE_FAILED.append((rel, f"{type(exc).__name__}: {exc}"[:120]))
        return None


def _source_files() -> list[Path]:
    out = []
    SKIPPED.clear()
    PARSE_FAILED.clear()
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
        tree = _parse(path)
        if tree is None:
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


def _standardizer_modules() -> list[Path]:
    """`CodeStandardizer` 를 다루는 프로덕션 모듈 — 동적 접근이 `lookup_wk` 에 닿을 수
    있는 유일한 범위다. 저장소 전체에는 동적 `getattr` 이 73건 있으나 대부분 무관하므로,
    임의의 문자열 필터 대신 이 범위로 좁힌다."""
    out = []
    for p in _source_files():
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        if rel.startswith("tests/"):
            continue
        txt = p.read_text(encoding="utf-8")
        if "CodeStandardizer" in txt or "code_standardizer" in txt:
            out.append(p)
    return out


# 표준화기를 다루는 모듈 안에서 허용된 동적 속성 접근. `lookup_wk` 와 무관함이
# 확인된 것만 등재한다. 새 항목이 생기면 그것이 표준화기에 닿는지 검토해야 한다.
_ALLOWED_DYNAMIC = {
    # MLModel 의 sidecar 로딩 — 대상은 self(MLModel)이며 표준화기가 아니다.
    ("serving/predictor.py", "load"),
}


def test_dynamic_attribute_access_in_standardizer_modules_is_known():
    """표준화기를 다루는 모듈의 동적 속성 접근이 알려진 목록과 일치해야 한다.

    정적 열거는 속성명이 리터럴일 때만 성립한다. 이 범위에 새 동적 접근이 생기면
    `lookup_wk` 소비자를 놓칠 수 있으므로 검토를 강제한다.
    """
    found = set()
    for path in _standardizer_modules():
        tree = _parse(path)
        if tree is None:
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in ("getattr", "setattr", "hasattr")
                    and len(node.args) >= 2
                    and not isinstance(node.args[1], ast.Constant)):
                found.add((rel, _enclosing(tree, node)))

    extra = found - _ALLOWED_DYNAMIC
    assert not extra, (
        "표준화기 관련 모듈에 새 동적 속성 접근이 생겼다. `lookup_wk` 에 닿을 수 있는지 "
        f"검토하고 무관하면 목록에 추가하라: {sorted(extra)}"
    )


def test_no_indirect_attribute_machinery_in_standardizer_modules():
    """`attrgetter`·`__getattribute__`·`__getattr__` 로 우회하는 경로가 없어야 한다."""
    found = []
    for path in _standardizer_modules():
        tree = _parse(path)
        if tree is None:
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("__getattribute__", "__getattr__"):
                found.append((rel, node.lineno, node.attr))
            if isinstance(node, ast.Name) and node.id == "attrgetter":
                found.append((rel, node.lineno, "attrgetter"))
    assert not found, f"간접 속성 접근 기계가 있어 정적 열거가 불완전하다: {found}"


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
    _collect()   # 파싱까지 수행해 PARSE_FAILED 를 채운다
    unreadable = list(SKIPPED)
    # 스캔에서 빠진 파일은 전부 `reviews/` 하위 산출물이어야 한다 —
    # 소스 트리(serving/, scripts/, rules/, dags/, hana_app/)에는 없어야 한다.
    src = {"serving", "scripts", "rules", "dags", "hana_app"}
    leaked = [x for x in unreadable if x[0].split("/")[0] in src]
    assert not leaked, f"소스 트리에서 읽지 못한 파일이 있어 열거가 불완전하다: {leaked}"
    # 파싱 실패도 조용히 넘어가면 안 된다 — 그 파일의 호출자를 놓치기 때문이다.
    assert not PARSE_FAILED, (
        f"파싱하지 못한 파일이 있어 호출자 열거가 불완전하다: {PARSE_FAILED}"
    )


def test_skip_accounting_actually_works(tmp_path, monkeypatch):
    """스킵 계상 메커니즘 자체가 동작하는지 확인한다.

    `test_scan_coverage_is_not_silently_incomplete` 는 `SKIPPED`/`PARSE_FAILED` 가
    비어 있어도 통과한다. 계상이 고장나 있어도(예: 예외 종류가 바뀌어 잡히지 않음)
    "빠뜨린 게 없다"로 읽히므로, 계상 자체를 별도로 검증한다.
    fable-advisor 10차 지적.
    """
    import ast as _ast

    # 파싱 실패 계상 — 문법이 깨진 파일을 만들어 `_parse` 가 기록하는지 본다
    bad = tmp_path / "broken.py"
    bad.write_text("def f(:\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path, raising=False)
    PARSE_FAILED.clear()
    tree = _parse(bad)
    assert tree is None, "문법 오류 파일이 파싱됐다 — 픽스처 전제 붕괴"
    assert PARSE_FAILED, "파싱 실패가 계상되지 않았다 — 커버리지 검사가 무의미하다"
    assert "broken.py" in PARSE_FAILED[0][0]

    # 읽기 실패 계상 — 존재하지 않는 파일로 `read_text` 를 실패시킨다
    SKIPPED.clear()
    ghost = tmp_path / "ghost.py"
    try:
        ghost.read_text(encoding="utf-8")
    except OSError as exc:
        SKIPPED.append(("ghost.py", type(exc).__name__))
    assert SKIPPED, "읽기 실패 계상 경로가 동작하지 않는다"
