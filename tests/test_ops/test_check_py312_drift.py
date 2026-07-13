from __future__ import annotations

import importlib.metadata
import sys
from unittest.mock import patch


def _write_constraints(path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _base_constraints() -> list[str]:
    return [
        "# constraints-py312.txt",
        "",
        "numpy==2.4.3",
        "pandas==2.3.3",
        "scikit-learn==1.8.0",
        "",
        "# comment line",
        "pydantic==2.12.5",
    ]


def test_check_drift_no_drift_when_all_match(tmp_path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, _base_constraints())

    fake_versions = {
        "numpy": "2.4.3",
        "pandas": "2.3.3",
        "scikit-learn": "1.8.0",
        "pydantic": "2.12.5",
    }

    with patch.object(importlib.metadata, "version",
                      side_effect=lambda pkg: fake_versions[pkg]):
        report = check_drift(constraints)

    assert report.ok is True
    assert report.drifts == []


def test_check_drift_detects_version_mismatch(tmp_path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, _base_constraints())

    fake_versions = {
        "numpy": "2.5.0",
        "pandas": "2.3.3",
        "scikit-learn": "1.8.0",
        "pydantic": "2.12.5",
    }

    with patch.object(importlib.metadata, "version",
                      side_effect=lambda pkg: fake_versions[pkg]):
        report = check_drift(constraints)

    assert report.ok is False
    assert any(d.package == "numpy" and d.pinned == "2.4.3" and d.installed == "2.5.0"
               for d in report.drifts)


def test_check_drift_detects_missing_package(tmp_path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, _base_constraints())

    def fake_version(pkg):
        if pkg == "pydantic":
            raise importlib.metadata.PackageNotFoundError(pkg)
        return {"numpy": "2.4.3", "pandas": "2.3.3", "scikit-learn": "1.8.0"}[pkg]

    with patch.object(importlib.metadata, "version", side_effect=fake_version):
        report = check_drift(constraints)

    assert report.ok is False
    assert any(d.package == "pydantic" and "<not installed>" in d.installed
               for d in report.drifts)


def test_check_drift_skips_comment_and_blank_lines(tmp_path) -> None:
    from scripts.ops.check_py312_drift import _parse_constraints

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, [
        "# header comment",
        "",
        "numpy==2.4.3",
        "  # indented comment",
        "pandas==2.3.3",
    ])

    pins = _parse_constraints(constraints)

    assert pins == {"numpy": "2.4.3", "pandas": "2.3.3"}


def test_check_drift_fails_closed_on_malformed_active_constraint(tmp_path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, ["numpy=2.4.3"])

    report = check_drift(constraints)

    assert report.ok is False
    assert report.constraint_errors == [
        "line 1: malformed active constraint: numpy=2.4.3",
    ]


def test_check_drift_python_version_check(tmp_path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, _base_constraints())

    fake_versions = {
        "numpy": "2.4.3",
        "pandas": "2.3.3",
        "scikit-learn": "1.8.0",
        "pydantic": "2.12.5",
    }

    with patch.object(importlib.metadata, "version",
                      side_effect=lambda pkg: fake_versions[pkg]), \
         patch.object(sys, "version_info", (3, 11, 0, "final", 0)):
        report = check_drift(constraints)

    assert report.ok is False
    assert report.python_ok is False
    assert "3.11" in report.python_version


def test_check_drift_cli_returns_zero_when_ok(tmp_path) -> None:
    from scripts.ops.check_py312_drift import main

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, _base_constraints())

    fake_versions = {
        "numpy": "2.4.3",
        "pandas": "2.3.3",
        "scikit-learn": "1.8.0",
        "pydantic": "2.12.5",
    }

    with patch.object(importlib.metadata, "version",
                      side_effect=lambda pkg: fake_versions[pkg]):
        assert main([str(constraints)]) == 0


def test_check_drift_cli_returns_nonzero_on_drift(tmp_path) -> None:
    from scripts.ops.check_py312_drift import main

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, _base_constraints())

    fake_versions = {
        "numpy": "99.0.0",
        "pandas": "2.3.3",
        "scikit-learn": "1.8.0",
        "pydantic": "2.12.5",
    }

    with patch.object(importlib.metadata, "version",
                      side_effect=lambda pkg: fake_versions[pkg]):
        assert main([str(constraints)]) == 1
