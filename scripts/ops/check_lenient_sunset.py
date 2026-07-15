"""Check if FEATURE_SCHEMA_LENIENT escape hatch is active past its sunset date.

Check-only: warns if FEATURE_SCHEMA_LENIENT=1 is set after the sunset date.
Reads the sunset default from serving/predictor.py via AST parsing (does NOT
import serving, to avoid pydantic dependency).
Exit code 0 = no warning, 1 = sunset warning.
"""
from __future__ import annotations

import argparse
import ast
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Sequence


def _read_sunset_default_from_source() -> date | None:
    """Read _FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT from serving/predictor.py via AST.

    Avoids importing serving.predictor (which requires pydantic) by parsing
    the source file directly. Returns None if the authoritative value is unavailable.
    """
    predictor_path = Path(__file__).resolve().parents[2] / "serving" / "predictor.py"
    try:
        tree = ast.parse(predictor_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if (isinstance(target, ast.Name)
                    and target.id == "_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT"):
                if (isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "date"
                        and len(node.value.args) == 3):
                    try:
                        return date(
                            ast.literal_eval(node.value.args[0]),
                            ast.literal_eval(node.value.args[1]),
                            ast.literal_eval(node.value.args[2]),
                        )
                    except Exception:
                        return None
                return None
    return None


@dataclass(frozen=True)
class SunsetReport:
    lenient_env: bool
    lenient_active: bool
    sunset_date: date | None
    today: date
    warning: bool
    warning_reason: Literal[
        "invalid_override", "authoritative_date_unavailable", "sunset_reached"
    ] | None

    @property
    def ok(self) -> bool:
        return not self.warning


def check_sunset(today: date | None = None) -> SunsetReport:
    """Check if FEATURE_SCHEMA_LENIENT is active past its sunset date."""
    today = today or date.today()
    lenient_env = os.environ.get("FEATURE_SCHEMA_LENIENT", "").strip().lower() in (
        "1", "true", "yes",
    )
    raw_sunset = os.environ.get("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "").strip()
    if raw_sunset:
        from datetime import datetime
        try:
            sunset = datetime.strptime(raw_sunset, "%Y-%m-%d").date()
            if sunset.isoformat() != raw_sunset:
                raise ValueError
        except ValueError:
            # Invalid env date: safe side, block lenient (per Codex 2026-05-07 #6)
            sunset = _read_sunset_default_from_source()
            lenient_active = False
            warning = lenient_env
            return SunsetReport(
                lenient_env=lenient_env,
                lenient_active=lenient_active,
                sunset_date=sunset,
                today=today,
                warning=warning,
                warning_reason="invalid_override" if lenient_env else None,
            )
    else:
        sunset = _read_sunset_default_from_source()
    if sunset is None:
        return SunsetReport(
            lenient_env=lenient_env,
            lenient_active=False,
            sunset_date=None,
            today=today,
            warning=lenient_env,
            warning_reason=(
                "authoritative_date_unavailable" if lenient_env else None
            ),
        )
    lenient_active = lenient_env and today < sunset
    warning = lenient_env and today >= sunset
    return SunsetReport(
        lenient_env=lenient_env,
        lenient_active=lenient_active,
        sunset_date=sunset,
        today=today,
        warning=warning,
        warning_reason="sunset_reached" if warning else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check if FEATURE_SCHEMA_LENIENT is active past its sunset date.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    report = check_sunset()
    if report.warning:
        if report.warning_reason == "invalid_override":
            raw_sunset = os.environ.get(
                "FEATURE_SCHEMA_LENIENT_SUNSET_DATE", ""
            ).strip()
            print(
                f"WARNING: FEATURE_SCHEMA_LENIENT_SUNSET_DATE='{raw_sunset}' "
                "is invalid; expected YYYY-MM-DD. Lenient is blocked."
            )
        elif report.warning_reason == "authoritative_date_unavailable":
            print(
                "WARNING: FEATURE_SCHEMA_LENIENT=1 is set but the authoritative "
                "sunset date is unavailable. Lenient is blocked."
            )
        else:
            sunset_phrase = (
                "has been reached"
                if report.today == report.sunset_date
                else "has passed"
            )
            print(
                f"WARNING: FEATURE_SCHEMA_LENIENT=1 is set but sunset date "
                f"{report.sunset_date} {sunset_phrase} (today={report.today}). "
                f"Lenient is blocked. Align train/serve schema and unset the env var."
            )
    else:
        print(
            f"lenient_env={report.lenient_env} lenient_active={report.lenient_active} "
            f"sunset={report.sunset_date} today={report.today} status=ok"
        )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
