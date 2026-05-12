from pathlib import Path


def test_setup_bat_escapes_parentheses_inside_hana_skip_echo():
    """cmd.exe parses unescaped ')' inside IF blocks as block terminators."""
    setup_text = Path("setup.bat").read_text(encoding="utf-8")

    assert "로컬 파일^(.parquet/.sas7bdat^)만" in setup_text
