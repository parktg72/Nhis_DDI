"""Check selected dependency profiles against installed Python 3.12 packages.

Check-only: reports configuration errors, version drift, and exact-lock gaps.
Exit code 0 means the selected profiles are valid and fully satisfied.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


@dataclass(frozen=True)
class DriftItem:
    package: str
    installed: str
    declarations: tuple[str, ...]


@dataclass(frozen=True)
class LockGap:
    package: str
    declarations: tuple[str, ...]


@dataclass(frozen=True)
class DriftReport:
    python_version: str
    python_ok: bool
    configuration_errors: tuple[str, ...] = ()
    drifts: tuple[DriftItem, ...] = ()
    lock_gaps: tuple[LockGap, ...] = ()

    @property
    def ok(self) -> bool:
        return (
            self.python_ok
            and not self.configuration_errors
            and not self.drifts
            and not self.lock_gaps
        )


@dataclass(frozen=True)
class _Declaration:
    requirement: Requirement
    display: str


def _parse_file(path: Path) -> tuple[tuple[_Declaration, ...], tuple[str, ...]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return (), (f"{path}: cannot read requirement file: {exc}",)

    declarations: list[_Declaration] = []
    errors: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        active = re.split(r"\s+#", stripped, maxsplit=1)[0].rstrip()
        try:
            requirement = Requirement(active)
        except InvalidRequirement as exc:
            errors.append(
                f"{path}:{line_number}: malformed requirement {active!r}: {exc}",
            )
            continue

        if requirement.marker is not None:
            try:
                applies = requirement.marker.evaluate()
            except Exception as exc:  # packaging can surface environment-specific errors
                errors.append(
                    f"{path}:{line_number}: marker evaluation failed "
                    f"for {active!r}: {exc}",
                )
                continue
            if not applies:
                continue

        declarations.append(
            _Declaration(
                requirement=requirement,
                display=f"{path}:{line_number}: {active}",
            ),
        )

    if not declarations and not errors:
        errors.append(f"{path}: no applicable requirements")
    return tuple(declarations), tuple(errors)


def _has_exact_pin(declarations: Sequence[_Declaration]) -> bool:
    return any(
        specifier.operator == "==" and "*" not in specifier.version
        for declaration in declarations
        for specifier in declaration.requirement.specifier
    )


def check_drift(
    constraints_path: Path | str,
    requirement_paths: Sequence[Path | str],
) -> DriftReport:
    """Check applicable constraints and explicitly selected requirement profiles."""
    version_info = sys.version_info
    python_version = f"{version_info[0]}.{version_info[1]}.{version_info[2]}"
    python_ok = tuple(version_info[:2]) == (3, 12)

    configuration_errors: list[str] = []
    selected_paths = tuple(Path(path) for path in requirement_paths)
    if not selected_paths:
        configuration_errors.append(
            "no requirement profiles selected; pass at least one path",
        )

    constraints, errors = _parse_file(Path(constraints_path))
    configuration_errors.extend(errors)

    selected_declarations: list[_Declaration] = []
    direct_packages: set[str] = set()
    for path in selected_paths:
        declarations, errors = _parse_file(path)
        configuration_errors.extend(errors)
        selected_declarations.extend(declarations)
        direct_packages.update(
            canonicalize_name(declaration.requirement.name)
            for declaration in declarations
        )

    declarations_by_package: dict[str, list[_Declaration]] = {}
    for declaration in (*constraints, *selected_declarations):
        package = canonicalize_name(declaration.requirement.name)
        declarations_by_package.setdefault(package, []).append(declaration)

    drifts: list[DriftItem] = []
    for package in sorted(declarations_by_package):
        declarations = declarations_by_package[package]
        displays = tuple(declaration.display for declaration in declarations)
        try:
            installed_text = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            drifts.append(
                DriftItem(
                    package=package,
                    installed="<not installed>",
                    declarations=displays,
                ),
            )
            continue
        except (OSError, UnicodeError, TypeError) as exc:
            drifts.append(
                DriftItem(
                    package=package,
                    installed=f"<metadata lookup failed: {type(exc).__name__}>",
                    declarations=displays,
                ),
            )
            continue

        if not isinstance(installed_text, str):
            drifts.append(
                DriftItem(
                    package=package,
                    installed=(
                        "<metadata version is not a string: "
                        f"{type(installed_text).__name__}>"
                    ),
                    declarations=displays,
                ),
            )
            continue

        try:
            installed_version = Version(installed_text)
        except InvalidVersion:
            drifts.append(
                DriftItem(
                    package=package,
                    installed=f"{installed_text} (invalid PEP 440 version)",
                    declarations=displays,
                ),
            )
            continue

        if any(
            not declaration.requirement.specifier.contains(
                installed_version,
                installed=True,
            )
            for declaration in declarations
        ):
            drifts.append(
                DriftItem(
                    package=package,
                    installed=installed_text,
                    declarations=displays,
                ),
            )

    lock_gaps = tuple(
        LockGap(
            package=package,
            declarations=tuple(
                declaration.display
                for declaration in declarations_by_package[package]
            ),
        )
        for package in sorted(direct_packages)
        if not _has_exact_pin(declarations_by_package[package])
    )
    return DriftReport(
        python_version=python_version,
        python_ok=python_ok,
        configuration_errors=tuple(configuration_errors),
        drifts=tuple(drifts),
        lock_gaps=lock_gaps,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check installed packages against constraints and selected "
            "requirement profiles."
        ),
    )
    parser.add_argument(
        "constraints_path",
        nargs="?",
        default="constraints-py312.txt",
        help="Path to constraints file (default: constraints-py312.txt)",
    )
    parser.add_argument(
        "-r",
        "--requirement",
        dest="requirement_paths",
        action="append",
        default=[],
        metavar="PATH",
        help="Selected requirement profile; repeat for additional profiles",
    )
    return parser


def _format_declarations(declarations: tuple[str, ...]) -> str:
    return " | ".join(declarations)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = check_drift(args.constraints_path, args.requirement_paths)
    if not report.python_ok:
        print(
            f"ERROR: Python {report.python_version} != 3.12 "
            "(dev/prod parity violation)",
        )
    for error in report.configuration_errors:
        print(f"ERROR: {error}")
    for drift in report.drifts:
        print(
            f"DRIFT: {drift.package} installed={drift.installed} "
            f"declarations={_format_declarations(drift.declarations)}",
        )
    for gap in report.lock_gaps:
        print(
            f"LOCK_GAP: {gap.package} "
            f"declarations={_format_declarations(gap.declarations)}",
        )
    print(
        f"SUMMARY: python={report.python_version} python_ok={report.python_ok} "
        f"configuration_errors={len(report.configuration_errors)} "
        f"drifts={len(report.drifts)} lock_gaps={len(report.lock_gaps)} "
        f"status={'ok' if report.ok else 'failed'}",
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
