"""hana_app/core/table_validator.py 단위 테스트."""
import pytest


class TestCheckColumnMapping:
    """check_column_mapping(actual_cols, expected_map) -> {"ok": [...], "missing": [...]}"""

    def test_all_match(self):
        from hana_app.core.table_validator import check_column_mapping
        actual = ["INDI_DSCM_NO", "CMN_KEY", "MDCARE_STRT_DT"]
        expected = {"patient_id": "INDI_DSCM_NO", "bill_no": "CMN_KEY"}
        result = check_column_mapping(actual, expected)
        assert result["ok"] == ["patient_id", "bill_no"]
        assert result["missing"] == []

    def test_some_missing(self):
        from hana_app.core.table_validator import check_column_mapping
        actual = ["INDI_DSCM_NO"]
        expected = {"patient_id": "INDI_DSCM_NO", "bill_no": "CMN_KEY"}
        result = check_column_mapping(actual, expected)
        assert "patient_id" in result["ok"]
        assert "bill_no" in result["missing"]

    def test_all_missing(self):
        from hana_app.core.table_validator import check_column_mapping
        actual = ["OTHER_COL"]
        expected = {"patient_id": "INDI_DSCM_NO"}
        result = check_column_mapping(actual, expected)
        assert result["ok"] == []
        assert "patient_id" in result["missing"]

    def test_empty_expected(self):
        from hana_app.core.table_validator import check_column_mapping
        result = check_column_mapping(["COL_A"], {})
        assert result == {"ok": [], "missing": []}

    def test_empty_actual(self):
        from hana_app.core.table_validator import check_column_mapping
        result = check_column_mapping([], {"patient_id": "INDI_DSCM_NO"})
        assert result["missing"] == ["patient_id"]
        assert result["ok"] == []

    def test_case_sensitive(self):
        """컬럼명 비교는 대소문자를 구분한다."""
        from hana_app.core.table_validator import check_column_mapping
        actual = ["indi_dscm_no"]   # 소문자
        expected = {"patient_id": "INDI_DSCM_NO"}  # 대문자
        result = check_column_mapping(actual, expected)
        assert "patient_id" in result["missing"]


class TestValidateAllIdentifiers:
    """validate_all_identifiers(column_map) — 안전하지 않은 식별자 있으면 ValueError."""

    def test_all_safe(self):
        from hana_app.core.table_validator import validate_all_identifiers
        # 예외 없어야 함
        validate_all_identifiers({"patient_id": "INDI_DSCM_NO", "bill_no": "CMN_KEY"})

    def test_unsafe_value_raises(self):
        from hana_app.core.table_validator import validate_all_identifiers
        with pytest.raises(ValueError, match="안전하지 않은"):
            validate_all_identifiers({"patient_id": "col'; DROP TABLE--"})

    def test_unsafe_key_raises(self):
        from hana_app.core.table_validator import validate_all_identifiers
        with pytest.raises(ValueError):
            validate_all_identifiers({"bad key!": "INDI_DSCM_NO"})

    def test_empty_map_passes(self):
        from hana_app.core.table_validator import validate_all_identifiers
        validate_all_identifiers({})  # 예외 없어야 함

    def test_dollar_and_hash_allowed(self):
        """HANA는 $·# 허용."""
        from hana_app.core.table_validator import validate_all_identifiers
        validate_all_identifiers({"col_a": "COL$NAME", "col_b": "COL#2"})
