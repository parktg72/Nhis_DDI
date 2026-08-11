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
from collections import Counter
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


def _collect() -> Counter:
    """{(상대경로, 감싸는 함수): 접점 수}

    **집합이 아니라 개수**다. 이미 알려진 함수 안에 호출을 하나 더 넣어도 집합은
    변하지 않으므로, 집합 기준 검사는 그 변경을 놓친다(codex-terra 11차 지적).
    """
    found: Counter = Counter()
    for path in _source_files():
        tree = _parse(path)
        if tree is None:
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        for node in ast.walk(tree):
            hit = False
            # x.lookup_wk(...)
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "lookup_wk"):
                hit = True
            # getattr(x, "lookup_wk", ...)
            elif (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and node.args and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "lookup_wk"):
                hit = True
            if hit:
                found[(rel, _enclosing(tree, node))] += 1
    return found


# 프로덕션 접점 — 함수마다 **몇 번** 닿는지까지 고정한다. 이 표가 바뀌면 계약 변경의
# 영향 범위가 달라진 것이다.
_EXPECTED_PRODUCTION = {
    # 게이팅 없음(전역). `if atc_fb:` 가드가 반환 계약 변경의 전파를 막는다.
    ("scripts/etl/code_standardizer.py", "standardize"): 1,
    # 주 플래그(SERVING_ENABLE_EDI_NAME_RESOLUTION) 안 — getattr 로 한 번 획득한다.
    # 획득한 지역 이름으로 부르는 `lookup_wk(wk)` 는 정적으로 보이지 않으므로,
    # 이 getattr 1건이 그 경로 전체의 관문이라는 사실 자체가 계상 대상이다.
    ("serving/predictor.py", "resolve_codes"): 1,
    # 주 플래그 안에 중첩된 ATC 플래그 경로
    ("serving/predictor.py", "atc_candidates"): 1,
}


def _production_only(counts: Counter) -> dict[tuple[str, str], int]:
    return {k: v for k, v in counts.items() if not k[0].startswith("tests/")}


def test_production_lookup_wk_callers_match_the_known_set():
    """프로덕션 접점이 알려진 표와 **개수까지** 일치해야 한다."""
    actual = _production_only(_collect())

    missing = {k: v for k, v in _EXPECTED_PRODUCTION.items() if k not in actual}
    extra = {k: v for k, v in actual.items() if k not in _EXPECTED_PRODUCTION}
    changed = {k: (_EXPECTED_PRODUCTION[k], v) for k, v in actual.items()
               if k in _EXPECTED_PRODUCTION and _EXPECTED_PRODUCTION[k] != v}

    assert not missing, f"알려진 접점이 사라졌다 — 표를 갱신하라: {sorted(missing)}"
    assert not extra, (
        "새 `lookup_wk` 접점이 생겼다. `841b849` 의 반환 계약 변경(ATC 없는 엔트리의 "
        "약물명을 반환)이 이 경로에도 닿는지 확인하고, 무영향이면 표에 추가하라: "
        f"{sorted(extra.items())}"
    )
    assert not changed, (
        "기존 함수 안의 `lookup_wk` 접점 수가 바뀌었다(기대, 실제). 같은 함수라도 "
        "새 접점은 새 소비 경로이므로 반환 계약 변경의 영향을 다시 봐야 한다: "
        f"{sorted(changed.items())}"
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


# 속성 접근을 우회시키는 `operator` 도구들. 이름만 보면 안 되고 별칭을 풀어야 한다.
_INDIRECT_OPERATOR_FUNCS = ("attrgetter", "methodcaller")
_INDIRECT_DUNDERS = ("__getattribute__", "__getattr__")


def _indirect_aliases(tree: ast.AST) -> set[str]:
    """이 모듈에서 간접 접근 도구를 가리키게 된 **지역 이름** 전부.

    `from operator import attrgetter as ag` 는 `ag` 를 등록한다. 별칭을 풀지 않으면
    `ag("lookup_wk")` 가 그냥 통과한다(codex-terra 11차 지적).

    `import operator as op` 형태는 여기서 등록하지 않는다 — 사용 시점이 반드시
    `op.attrgetter` 라는 **속성 접근**이라 아래 `ast.Attribute` 분기가 이미 잡고,
    모듈 이름 자체를 금지하면 `operator.add` 같은 무관한 사용까지 오탐이 된다.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "operator":
            for a in node.names:
                if a.name in _INDIRECT_OPERATOR_FUNCS:
                    names.add(a.asname or a.name)
    return names


def test_no_indirect_attribute_machinery_in_standardizer_modules():
    """`attrgetter`·`methodcaller`·`__getattribute__`·`__getattr__` 우회가 없어야 한다.

    별칭·모듈 경유(`operator.attrgetter`, `import operator as op`,
    `from operator import attrgetter as ag`)까지 모두 잡는다.
    """
    found = []
    for path in _standardizer_modules():
        tree = _parse(path)
        if tree is None:
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        aliases = _indirect_aliases(tree)
        for node in ast.walk(tree):
            # x.__getattr__ / x.attrgetter / operator.attrgetter / op.attrgetter
            if isinstance(node, ast.Attribute) and node.attr in (
                    _INDIRECT_DUNDERS + _INDIRECT_OPERATOR_FUNCS):
                found.append((rel, node.lineno, node.attr))
            # attrgetter(...) / ag(...) — 임포트로 이 모듈에 묶인 이름
            elif isinstance(node, ast.Name) and (
                    node.id in _INDIRECT_OPERATOR_FUNCS or node.id in aliases):
                found.append((rel, node.lineno, f"{node.id} (operator 별칭)"))
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

    **양쪽 다 `_source_files()`/`_collect()` 를 실제로 통과시킨다.** 이전 판은 읽기
    실패 절반이 테스트가 직접 `SKIPPED.append` 를 하는 자기충족 형태여서, 계상이
    고장나도 통과했다(codex-terra·fable-advisor 11차 공통 지적).
    """
    mod = sys.modules[__name__]
    monkeypatch.setattr(mod, "ROOT", tmp_path, raising=False)

    (tmp_path / "fine.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "broken.py").write_text("def f(:\n    pass\n", encoding="utf-8")
    (tmp_path / "denied.py").write_text("y = 2\n", encoding="utf-8")

    # 읽기 실패를 실제로 일으킨다. chmod 는 이 저장소가 놓인 파일시스템(NTFS/DrvFs)
    # 에서 신뢰할 수 없으므로, `read_text` 가 그 파일에 대해서만 PermissionError 를
    # 내도록 한다 — `_source_files()` 의 except 절이 이를 잡아 계상해야 한다.
    real_read_text = Path.read_text

    def fake_read_text(self, *args, **kwargs):
        if self.name == "denied.py":
            raise PermissionError(13, "permission denied (테스트가 주입한 읽기 실패)")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    scanned = {p.name for p in _source_files()}
    assert scanned == {"fine.py", "broken.py"}, (
        f"스캐너가 읽은 파일 집합이 기대와 다르다 — 픽스처 전제 붕괴: {sorted(scanned)}"
    )
    assert [n for n, _ in SKIPPED] == ["denied.py"], (
        f"읽기 실패가 계상되지 않았다 — 커버리지 검사가 무의미하다: {SKIPPED}"
    )
    assert SKIPPED[0][1] == "PermissionError", f"예외 종류가 잘못 계상됐다: {SKIPPED}"

    # 파싱 실패 계상 — `_collect()` 가 `_source_files()` → `_parse()` 를 다 태운다.
    # (`_source_files()` 가 두 목록을 비우므로 순서가 아니라 이 호출이 채운 값을 본다)
    _collect()
    assert [n for n, _ in PARSE_FAILED] == ["broken.py"], (
        f"파싱 실패가 계상되지 않았다 — 커버리지 검사가 무의미하다: {PARSE_FAILED}"
    )
    assert [n for n, _ in SKIPPED] == ["denied.py"], (
        f"`_collect()` 경로에서 읽기 실패 계상이 사라졌다: {SKIPPED}"
    )
