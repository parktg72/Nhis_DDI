"""hana_app/core/page_guards.py 단위 테스트."""
from hana_app.core.page_guards import check_hana_validated, get_validation_error

BASE_CFG = {
    "data_source": "hana",
    "connection": {"host": "192.168.1.1", "port": 30015, "user": "u", "password": ""},
    "validated": False,
    "validated_at": "",
    "validated_host": "",
}


class TestCheckHanaValidated:
    def test_returns_true_when_validated_and_host_matches(self):
        cfg = {**BASE_CFG, "validated": True, "validated_host": "192.168.1.1"}
        assert check_hana_validated(cfg) is True

    def test_returns_false_when_not_validated(self):
        cfg = {**BASE_CFG, "validated": False}
        assert check_hana_validated(cfg) is False

    def test_returns_false_when_host_mismatch(self):
        cfg = {**BASE_CFG, "validated": True, "validated_host": "10.0.0.1"}
        assert check_hana_validated(cfg) is False

    def test_returns_true_for_non_hana_source(self):
        cfg = {**BASE_CFG, "data_source": "sas", "validated": False}
        assert check_hana_validated(cfg) is True

    def test_returns_true_when_validated_host_empty(self):
        """validated_host가 빈 문자열이면 호스트 불일치 검사 생략."""
        cfg = {**BASE_CFG, "validated": True, "validated_host": ""}
        assert check_hana_validated(cfg) is True


class TestGetValidationError:
    def test_returns_none_when_ok(self):
        cfg = {**BASE_CFG, "validated": True, "validated_host": "192.168.1.1"}
        assert get_validation_error(cfg) is None

    def test_returns_message_when_not_validated(self):
        cfg = {**BASE_CFG, "validated": False}
        msg = get_validation_error(cfg)
        assert msg is not None
        assert "검증" in msg

    def test_returns_message_when_host_mismatch(self):
        cfg = {**BASE_CFG, "validated": True, "validated_host": "10.0.0.1"}
        msg = get_validation_error(cfg)
        assert msg is not None
        assert "10.0.0.1" in msg
        assert "192.168.1.1" in msg

    def test_returns_none_for_non_hana(self):
        cfg = {**BASE_CFG, "data_source": "sas"}
        assert get_validation_error(cfg) is None
