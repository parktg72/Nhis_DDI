from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch


def test_read_sunset_default_from_source_returns_2026_08_01() -> None:
    """AST parser reads the actual constant from serving/predictor.py."""
    from scripts.ops.check_lenient_sunset import _read_sunset_default_from_source

    result = _read_sunset_default_from_source()

    assert result == date(2026, 8, 1)


def test_check_sunset_blocks_lenient_when_source_is_unreadable(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    with patch.object(Path, "read_text", side_effect=OSError("unreadable")):
        report = check_sunset(today=date(2026, 7, 1))

    assert report.ok is False
    assert report.warning is True
    assert report.lenient_active is False
    assert report.sunset_date is None
    assert report.warning_reason == "authoritative_date_unavailable"


def test_check_sunset_blocks_lenient_when_source_is_unparseable(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    with patch.object(Path, "read_text", return_value="not valid python ("):
        report = check_sunset(today=date(2026, 7, 1))

    assert report.ok is False
    assert report.warning is True
    assert report.lenient_active is False
    assert report.sunset_date is None
    assert report.warning_reason == "authoritative_date_unavailable"


def test_check_sunset_blocks_lenient_when_source_is_missing(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    with patch.object(Path, "read_text", side_effect=FileNotFoundError("missing")):
        report = check_sunset(today=date(2026, 7, 1))

    assert report.ok is False
    assert report.warning is True
    assert report.lenient_active is False
    assert report.sunset_date is None
    assert report.warning_reason == "authoritative_date_unavailable"


def test_check_sunset_invalid_env_date_no_warning_when_lenient_unset(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "garbage-date")

    report = check_sunset(today=date(2026, 12, 31))

    assert report.ok is True
    assert report.warning is False
    assert report.lenient_active is False
    assert report.warning_reason is None


def test_check_sunset_no_warning_when_before_sunset(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    report = check_sunset(today=date(2026, 7, 1))

    assert report.ok is True
    assert report.warning is False
    assert report.lenient_active is True
    assert report.warning_reason is None


def test_check_sunset_warns_when_lenient_set_after_sunset(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    report = check_sunset(today=date(2026, 8, 2))

    assert report.ok is False
    assert report.warning is True
    assert report.lenient_active is False
    assert report.warning_reason == "sunset_reached"


def test_check_sunset_warns_when_lenient_set_on_sunset_date(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    report = check_sunset(today=date(2026, 8, 1))

    assert report.ok is False
    assert report.warning is True
    assert report.lenient_active is False
    assert report.warning_reason == "sunset_reached"


def test_check_sunset_env_override_extends_sunset(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "2027-12-31")

    report = check_sunset(today=date(2027, 6, 1))

    assert report.ok is True
    assert report.warning is False
    assert report.lenient_active is True
    assert report.warning_reason is None


def test_check_sunset_invalid_env_date_blocks_lenient(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "garbage-date")

    report = check_sunset(today=date(2026, 1, 1))

    assert report.ok is False
    assert report.warning is True
    assert report.lenient_active is False
    assert report.sunset_date == date(2026, 8, 1)
    assert report.warning_reason == "invalid_override"


def test_check_sunset_noncanonical_env_date_blocks_lenient(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "2027-1-1")

    report = check_sunset(today=date(2026, 1, 1))

    assert report.warning_reason == "invalid_override"
    assert report.warning is True
    assert report.lenient_active is False
    assert report.ok is False
    assert report.sunset_date == date(2026, 8, 1)


def test_check_sunset_empty_env_uses_default(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "")

    report = check_sunset(today=date(2026, 7, 1))

    assert report.ok is True
    assert report.warning is False
    assert report.lenient_active is True
    assert report.warning_reason is None


def test_check_sunset_cli_returns_zero_when_ok(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import main

    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    with patch("scripts.ops.check_lenient_sunset.date") as mock_date:
        mock_date.today.return_value = date(2026, 7, 1)
        mock_date.side_effect = lambda *a, **k: date(*a, **k)
        assert main([]) == 0


def test_check_sunset_cli_reports_invalid_override(monkeypatch, capsys) -> None:
    from scripts.ops.check_lenient_sunset import main

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "garbage-date")

    assert main([]) == 1

    output = capsys.readouterr().out
    assert output == (
        "WARNING: FEATURE_SCHEMA_LENIENT_SUNSET_DATE='garbage-date' is invalid; "
        "expected YYYY-MM-DD. Lenient is blocked.\n"
    )
    assert "has passed" not in output


def test_check_sunset_cli_reports_sunset_reached_on_exact_date(
    monkeypatch, capsys
) -> None:
    from scripts.ops.check_lenient_sunset import main

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    with patch("scripts.ops.check_lenient_sunset.date") as mock_date:
        mock_date.today.return_value = date(2026, 8, 1)
        mock_date.side_effect = lambda *a, **k: date(*a, **k)
        assert main([]) == 1

    output = capsys.readouterr().out
    assert "has been reached" in output
    assert "has passed" not in output


def test_check_sunset_cli_reports_sunset_passed_after_date(
    monkeypatch, capsys
) -> None:
    from scripts.ops.check_lenient_sunset import main

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    with patch("scripts.ops.check_lenient_sunset.date") as mock_date:
        mock_date.today.return_value = date(2026, 8, 2)
        mock_date.side_effect = lambda *a, **k: date(*a, **k)
        assert main([]) == 1

    assert "has passed" in capsys.readouterr().out
