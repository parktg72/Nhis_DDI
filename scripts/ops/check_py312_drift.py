"""Check installed package versions against constraints-py312.txt pins.

Check-only: reports drift, does not fix. Verifies Python 3.12 runtime parity.
Exit code 0 = no drift, 1 = drift or Python version mismatch.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DriftItem:
    package: str
    pinned: str
    installed: str


@dataclass(frozen=True)
class DriftReport:
    python_version: str
    python_ok: bool
    drifts: list[DriftItem] = field(default_factory=list)
    constraint_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.python_ok and not self.drifts and not self.constraint_errors


def _parse_constraints(path: Path) -> dict[str, str]:
    """Parse == pins from constraints file. Skips comments and blank lines."""
    pins: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        active = re.split(r"\s+#", stripped, maxsplit=1)[0].rstrip()
        match = re.fullmatch(r"([a-zA-Z0-9_.-]+)==(\S+)", active)
        if match is None:
            raise ValueError(
                f"line {line_number}: malformed active constraint: {stripped}",
            )
        pins[match.group(1).lower()] = match.group(2)
    return pins


def check_drift(constraints_path: Path | str) -> DriftReport:
    """Compare constraints pins against installed package versions."""
    constraints_path = Path(constraints_path)
    try:
        pins = _parse_constraints(constraints_path)
        constraint_errors: list[str] = []
    except ValueError as exc:
        pins = {}
        constraint_errors = [str(exc)]
    vi = sys.version_info
    python_version = f"{vi[0]}.{vi[1]}.{vi[2]}"
    python_ok = vi[:2] == (3, 12)
    drifts: list[DriftItem] = []
    for pkg, pinned in sorted(pins.items()):
        try:
            installed = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            drifts.append(DriftItem(package=pkg, pinned=pinned, installed="<not installed>"))
            continue
        if installed != pinned:
            drifts.append(DriftItem(package=pkg, pinned=pinned, installed=installed))
    return DriftReport(
        python_version=python_version,
        python_ok=python_ok,
        drifts=drifts,
        constraint_errors=constraint_errors,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check installed package versions against constraints-py312.txt pins.",
    )
    parser.add_argument(
        "constraints_path",
        nargs="?",
        default="constraints-py312.txt",
        help="Path to constraints file (default: constraints-py312.txt)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = check_drift(args.constraints_path)
    if not report.python_ok:
        print(f"ERROR: Python {report.python_version} != 3.12 (dev/prod parity violation)")
    for error in report.constraint_errors:
        print(f"ERROR: {error}")
    for d in report.drifts:
        print(f"DRIFT: {d.package} pinned={d.pinned} installed={d.installed}")
    print(f"python={report.python_version} python_ok={report.python_ok} drifts={len(report.drifts)}")
    print(f"status={'ok' if report.ok else 'failed'}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
