from __future__ import annotations

import importlib.metadata
import inspect
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from unittest.mock import patch

import pytest


def _write(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(f"{text}\n" if text else "", encoding="utf-8")
    return path


def _profiles(
    tmp_path: Path,
    constraints: list[str],
    requirements: list[str],
) -> tuple[Path, Path]:
    return (
        _write(tmp_path / "constraints.txt", constraints),
        _write(tmp_path / "requirements.txt", requirements),
    )


def test_public_report_contract_uses_tuple_collections() -> None:
    import scripts.ops.check_py312_drift as drift_module

    assert [item.name for item in fields(drift_module.DriftItem)] == [
        "package",
        "installed",
        "declarations",
    ]
    assert [item.name for item in fields(drift_module.LockGap)] == [
        "package",
        "declarations",
    ]
    assert [item.name for item in fields(drift_module.DriftReport)] == [
        "python_version",
        "python_ok",
        "configuration_errors",
        "drifts",
        "lock_gaps",
    ]
    report = drift_module.DriftReport(python_version="3.12.0", python_ok=True)
    assert report.configuration_errors == ()
    assert report.drifts == ()
    assert report.lock_gaps == ()
    report_signature = inspect.signature(drift_module.DriftReport)
    for name in ("configuration_errors", "drifts", "lock_gaps"):
        assert report_signature.parameters[name].default == ()
    assert list(inspect.signature(drift_module.check_drift).parameters) == [
        "constraints_path",
        "requirement_paths",
    ]
    assert not hasattr(drift_module, "_parse_constraints")


def test_report_dataclasses_are_frozen() -> None:
    from scripts.ops.check_py312_drift import DriftItem, DriftReport, LockGap

    declarations = ("requirements.txt:1: widget==1.0",)
    instances_and_attributes = (
        (
            DriftItem(
                package="widget",
                installed="2.0",
                declarations=declarations,
            ),
            "package",
        ),
        (LockGap(package="widget", declarations=declarations), "package"),
        (DriftReport(python_version="3.12.0", python_ok=True), "python_version"),
    )

    for instance, attribute in instances_and_attributes:
        with pytest.raises(FrozenInstanceError):
            setattr(instance, attribute, "changed")


def test_check_drift_no_drift_when_all_declarations_match(tmp_path: Path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ["widget==1.0"],
        ["widget>=0.9,<2"],
    )

    with patch.object(importlib.metadata, "version", return_value="1.0"):
        report = check_drift(constraints, (requirements,))

    assert report.ok is True
    assert report.configuration_errors == ()
    assert report.drifts == ()
    assert report.lock_gaps == ()


@pytest.mark.parametrize(
    ("installed", "expected_ok"),
    [
        ("2.11.0+cu126", True),
        ("2.11.0+cu128", False),
    ],
)
def test_selected_local_pin_accepts_only_the_selected_cuda_build(
    tmp_path: Path,
    installed: str,
    expected_ok: bool,
) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ["torch>=2.0"],
        ["torch==2.11.0+cu126"],
    )

    with patch.object(importlib.metadata, "version", return_value=installed):
        report = check_drift(constraints, (requirements,))

    assert report.ok is expected_ok
    assert report.configuration_errors == ()
    assert report.lock_gaps == ()
    if expected_ok:
        assert report.drifts == ()
    else:
        assert tuple(item.package for item in report.drifts) == ("torch",)


def test_public_exact_pin_accepts_installed_local_version(tmp_path: Path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ["torch==2.11.0"],
        ["torch>=2.2"],
    )

    with patch.object(importlib.metadata, "version", return_value="2.11.0+cu126"):
        report = check_drift(constraints, (requirements,))

    assert report.ok is True
    assert report.configuration_errors == ()
    assert report.drifts == ()
    assert report.lock_gaps == ()


def test_incompatible_range_is_drift(tmp_path: Path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ["widget==2.0"],
        ["widget>=3.0"],
    )

    with patch.object(importlib.metadata, "version", return_value="2.0"):
        report = check_drift(constraints, (requirements,))

    assert report.ok is False
    assert report.configuration_errors == ()
    assert report.lock_gaps == ()
    assert tuple(item.package for item in report.drifts) == ("widget",)
    assert len(report.drifts[0].declarations) == 2


def test_missing_selected_package_is_drift(tmp_path: Path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ["anchor==1.0"],
        ["widget==2.0"],
    )

    def fake_version(package: str) -> str:
        if package == "widget":
            raise importlib.metadata.PackageNotFoundError(package)
        return "1.0"

    with patch.object(importlib.metadata, "version", side_effect=fake_version):
        report = check_drift(constraints, (requirements,))

    assert report.ok is False
    assert report.configuration_errors == ()
    assert report.lock_gaps == ()
    assert tuple(item.package for item in report.drifts) == ("widget",)
    assert report.drifts[0].installed == "<not installed>"


def test_constraints_only_package_is_still_checked_without_lock_gap(
    tmp_path: Path,
) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ["orphan==1.0"],
        ["widget==2.0"],
    )

    def fake_version(package: str) -> str:
        if package == "orphan":
            raise importlib.metadata.PackageNotFoundError(package)
        return "2.0"

    with patch.object(importlib.metadata, "version", side_effect=fake_version):
        report = check_drift(constraints, (requirements,))

    assert tuple(item.package for item in report.drifts) == ("orphan",)
    assert report.lock_gaps == ()


def test_unpinned_direct_requirement_has_lock_gap(tmp_path: Path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ["widget>=1.0"],
        ["widget<3.0"],
    )

    with patch.object(importlib.metadata, "version", return_value="2.0"):
        report = check_drift(constraints, (requirements,))

    assert report.ok is False
    assert report.configuration_errors == ()
    assert report.drifts == ()
    assert tuple(item.package for item in report.lock_gaps) == ("widget",)
    assert len(report.lock_gaps[0].declarations) == 2


@pytest.mark.parametrize(
    ("constraint", "requirement"),
    [
        ("widget==2.0", "widget>=1.0"),
        ("widget>=1.0", "widget==2.0"),
    ],
)
def test_exact_pin_in_constraint_or_selected_requirement_closes_lock_gap(
    tmp_path: Path,
    constraint: str,
    requirement: str,
) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        [constraint],
        [requirement],
    )

    with patch.object(importlib.metadata, "version", return_value="2.0"):
        report = check_drift(constraints, (requirements,))

    assert report.ok is True
    assert report.configuration_errors == ()
    assert report.drifts == ()
    assert report.lock_gaps == ()


def test_wildcard_equality_does_not_count_as_exact_lock(tmp_path: Path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ["widget==2.*"],
        ["widget>=2.0"],
    )

    with patch.object(importlib.metadata, "version", return_value="2.3"):
        report = check_drift(constraints, (requirements,))

    assert report.configuration_errors == ()
    assert report.drifts == ()
    assert tuple(item.package for item in report.lock_gaps) == ("widget",)


def test_names_are_canonicalized_and_inline_comments_are_ignored(
    tmp_path: Path,
) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ["# heading", "", "My_Pkg==1.0  # exact lock"],
        ["  # profile", "my.pkg>=0.9 # supported range"],
    )

    with patch.object(importlib.metadata, "version", return_value="1.0") as version:
        report = check_drift(constraints, (requirements,))

    version.assert_called_once_with("my-pkg")
    assert report.ok is True
    assert report.configuration_errors == ()
    assert report.drifts == ()
    assert report.lock_gaps == ()


def test_false_markers_are_not_checked_or_counted_as_lock_gaps(
    tmp_path: Path,
) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        [
            "anchor==1.0",
            'constraint-skip==9.0; python_version < "3.0"',
        ],
        [
            "anchor>=1.0",
            'direct-skip==2.0; python_version < "3.0"',
        ],
    )

    with patch.object(importlib.metadata, "version", return_value="1.0") as version:
        report = check_drift(constraints, (requirements,))

    version.assert_called_once_with("anchor")
    assert report.ok is True
    assert report.configuration_errors == ()
    assert report.drifts == ()
    assert report.lock_gaps == ()


def test_marker_evaluation_error_is_configuration_error(tmp_path: Path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ['widget==1.0; python_version >= "3.0"'],
        ["widget>=1.0"],
    )

    with (
        patch("packaging.markers.Marker.evaluate", side_effect=RuntimeError("boom")),
        patch.object(importlib.metadata, "version", return_value="1.0"),
    ):
        report = check_drift(constraints, (requirements,))

    assert report.ok is False
    assert any("marker evaluation failed" in error for error in report.configuration_errors)


@pytest.mark.parametrize(
    ("target", "contents", "expected_text"),
    [
        ("constraints", None, "cannot read"),
        ("constraints", [], "no applicable requirements"),
        ("constraints", ["# comments only", ""], "no applicable requirements"),
        ("constraints", ["widget=1.0"], "malformed requirement"),
        ("requirements", None, "cannot read"),
        ("requirements", [], "no applicable requirements"),
        ("requirements", ["widget=1.0"], "malformed requirement"),
    ],
)
def test_invalid_input_files_are_configuration_errors(
    tmp_path: Path,
    target: str,
    contents: list[str] | None,
    expected_text: str,
) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints = tmp_path / "constraints.txt"
    requirements = tmp_path / "requirements.txt"
    if target != "constraints" or contents is not None:
        _write(constraints, contents if target == "constraints" else ["widget==1.0"])
    if target != "requirements" or contents is not None:
        _write(requirements, contents if target == "requirements" else ["widget==1.0"])

    with patch.object(importlib.metadata, "version", return_value="1.0"):
        report = check_drift(constraints, (requirements,))

    assert report.ok is False
    assert any(expected_text in error for error in report.configuration_errors)
    assert report.drifts == ()
    assert report.lock_gaps == ()


def test_unreadable_input_is_configuration_error(tmp_path: Path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints = _write(tmp_path / "constraints.txt", ["anchor==1.0"])
    unreadable_profile = tmp_path / "profile-directory"
    unreadable_profile.mkdir()

    with patch.object(importlib.metadata, "version", return_value="1.0"):
        report = check_drift(constraints, (unreadable_profile,))

    assert report.ok is False
    assert any("cannot read" in error for error in report.configuration_errors)
    assert report.drifts == ()
    assert report.lock_gaps == ()


def test_false_marker_only_file_is_empty_active_configuration_error(
    tmp_path: Path,
) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ["anchor==1.0"],
        ['ignored==2.0; python_version < "3.0"'],
    )

    with patch.object(importlib.metadata, "version", return_value="1.0"):
        report = check_drift(constraints, (requirements,))

    assert report.ok is False
    assert any("no applicable requirements" in error for error in report.configuration_errors)
    assert report.drifts == ()
    assert report.lock_gaps == ()


def test_no_selected_profiles_is_configuration_error(tmp_path: Path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints = _write(tmp_path / "constraints.txt", ["widget==1.0"])

    with patch.object(importlib.metadata, "version", return_value="1.0"):
        report = check_drift(constraints, ())

    assert report.ok is False
    assert report.configuration_errors == (
        "no requirement profiles selected; pass at least one path",
    )
    assert report.drifts == ()
    assert report.lock_gaps == ()


def test_invalid_installed_version_is_drift(tmp_path: Path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ["widget==1.0"],
        ["widget>=1.0"],
    )

    with patch.object(importlib.metadata, "version", return_value="not-a-version"):
        report = check_drift(constraints, (requirements,))

    assert tuple(item.package for item in report.drifts) == ("widget",)
    assert report.drifts[0].installed == (
        "not-a-version (invalid PEP 440 version)"
    )


def test_non_string_metadata_version_is_deterministic_drift_and_cli_safe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.ops.check_py312_drift import check_drift, main

    constraints, requirements = _profiles(
        tmp_path,
        ["widget==1.0"],
        ["widget>=1.0"],
    )
    expected_installed = "<metadata version is not a string: NoneType>"

    with patch.object(importlib.metadata, "version", return_value=None):
        report = check_drift(constraints, (requirements,))

    assert report.ok is False
    assert report.configuration_errors == ()
    assert report.lock_gaps == ()
    assert tuple(item.package for item in report.drifts) == ("widget",)
    assert report.drifts[0].installed == expected_installed

    with patch.object(importlib.metadata, "version", return_value=None):
        return_code = main([str(constraints), "-r", str(requirements)])

    output = capsys.readouterr().out
    assert return_code == 1
    assert f"DRIFT: widget installed={expected_installed}" in output
    assert "Traceback" not in output


def test_metadata_read_error_is_deterministic_drift_and_cli_safe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.ops.check_py312_drift import check_drift, main

    constraints, requirements = _profiles(
        tmp_path,
        ["widget==1.0"],
        ["widget>=1.0"],
    )
    expected_installed = "<metadata lookup failed: OSError>"

    with patch.object(
        importlib.metadata,
        "version",
        side_effect=OSError("private metadata path\nread failed"),
    ):
        report = check_drift(constraints, (requirements,))

    assert report.ok is False
    assert report.configuration_errors == ()
    assert report.lock_gaps == ()
    assert tuple(item.package for item in report.drifts) == ("widget",)
    assert report.drifts[0].installed == expected_installed

    with patch.object(
        importlib.metadata,
        "version",
        side_effect=OSError("private metadata path\nread failed"),
    ):
        return_code = main([str(constraints), "-r", str(requirements)])

    output = capsys.readouterr().out
    assert return_code == 1
    assert f"DRIFT: widget installed={expected_installed}" in output
    assert "private metadata path" not in output
    assert "Traceback" not in output


def test_drift_items_are_sorted_by_canonical_package_name(tmp_path: Path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ["Zeta==1.0", "alpha_pkg==1.0"],
        ["zeta>=1.0", "alpha-pkg>=1.0"],
    )

    with patch.object(importlib.metadata, "version", return_value="2.0"):
        report = check_drift(constraints, (requirements,))

    assert tuple(item.package for item in report.drifts) == ("alpha-pkg", "zeta")
    assert report.lock_gaps == ()


def test_lock_gaps_are_sorted_by_canonical_package_name(tmp_path: Path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ["anchor==1.0"],
        ["zeta>=1.0", "alpha_pkg>=1.0"],
    )
    versions = {"alpha-pkg": "2.0", "anchor": "1.0", "zeta": "2.0"}

    with patch.object(
        importlib.metadata,
        "version",
        side_effect=lambda package: versions[package],
    ):
        report = check_drift(constraints, (requirements,))

    assert report.configuration_errors == ()
    assert report.drifts == ()
    assert tuple(item.package for item in report.lock_gaps) == ("alpha-pkg", "zeta")


def test_python_311_runtime_fails_even_when_dependencies_match(
    tmp_path: Path,
) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints, requirements = _profiles(
        tmp_path,
        ["widget==1.0"],
        ["widget>=1.0"],
    )

    with (
        patch.object(importlib.metadata, "version", return_value="1.0"),
        patch.object(sys, "version_info", (3, 11, 0, "final", 0)),
    ):
        report = check_drift(constraints, (requirements,))

    assert report.ok is False
    assert report.python_ok is False
    assert report.python_version == "3.11.0"
    assert report.configuration_errors == ()
    assert report.drifts == ()
    assert report.lock_gaps == ()


def test_cli_requires_at_least_one_requirement_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.ops.check_py312_drift import main

    constraints = _write(tmp_path / "constraints.txt", ["widget==1.0"])

    with patch.object(importlib.metadata, "version", return_value="1.0"):
        return_code = main([str(constraints)])

    output = capsys.readouterr().out
    assert return_code == 1
    assert "ERROR:" in output
    assert "no requirement profiles selected" in output
    assert "status=failed" in output


def test_cli_repeatable_profiles_return_zero_and_print_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.ops.check_py312_drift import main

    constraints = _write(tmp_path / "constraints.txt", ["widget==1.0"])
    base = _write(tmp_path / "base.txt", ["widget>=1.0"])
    extra = _write(tmp_path / "extra.txt", ["other==2.0"])
    versions = {"other": "2.0", "widget": "1.0"}

    with patch.object(
        importlib.metadata,
        "version",
        side_effect=lambda package: versions[package],
    ):
        return_code = main(
            [str(constraints), "-r", str(base), "--requirement", str(extra)],
        )

    output = capsys.readouterr().out
    assert return_code == 0
    assert "configuration_errors=0" in output
    assert "drifts=0" in output
    assert "lock_gaps=0" in output
    assert "status=ok" in output


def test_cli_failure_prints_error_drift_lock_gap_and_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.ops.check_py312_drift import main

    constraints = _write(tmp_path / "constraints.txt", ["widget>=1.0"])
    requirements = _write(tmp_path / "requirements.txt", ["widget<3.0"])
    malformed = _write(tmp_path / "malformed.txt", ["broken=1.0"])

    with patch.object(importlib.metadata, "version", return_value="4.0"):
        return_code = main(
            [
                str(constraints),
                "-r",
                str(requirements),
                "-r",
                str(malformed),
            ],
        )

    output = capsys.readouterr().out
    assert return_code == 1
    assert "ERROR:" in output
    assert "DRIFT: widget" in output
    assert "LOCK_GAP: widget" in output
    assert "configuration_errors=1" in output
    assert "drifts=1" in output
    assert "lock_gaps=1" in output
    assert "status=failed" in output
