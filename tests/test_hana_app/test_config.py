"""hana_app/core/config.py validated 플래그 테스트."""
import importlib
import json
from pathlib import Path

from hana_app.core.config import DEFAULT_CONFIG


class TestValidatedFlag:
    def test_project_hana_config_has_operational_host_without_credentials(self):
        """프로젝트 config에는 운영 host/port만 저장하고 ID/PW는 사용자 입력으로 둔다."""
        cfg_path = Path(__file__).resolve().parents[2] / "hana_app" / "hana_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        assert cfg["connection"]["host"] == "10.1.67.115"
        assert cfg["connection"]["port"] == 30015
        assert cfg["connection"].get("user", "") == ""
        assert "password" not in cfg["connection"]
        assert cfg["connection"].get("password_enc", "") == ""

    def test_default_config_has_validated_false(self):
        """DEFAULT_CONFIG에 validated=False가 있다."""
        assert DEFAULT_CONFIG.get("validated") is False

    def test_default_config_has_validated_at(self):
        """DEFAULT_CONFIG에 validated_at 키가 있다 (빈 문자열)."""
        assert "validated_at" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["validated_at"] == ""

    def test_default_config_has_validated_host(self):
        """DEFAULT_CONFIG에 validated_host 키가 있다 (빈 문자열)."""
        assert "validated_host" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["validated_host"] == ""

    def test_load_config_merges_validated_keys_if_missing(self, tmp_path, monkeypatch):
        """기존 config 파일에 validated 키 없으면 기본값으로 병합."""
        cfg_file = tmp_path / "hana_config.json"
        # validated 없는 구버전 config
        cfg_file.write_text(json.dumps({"connection": {"host": "h"}}), encoding="utf-8")

        import hana_app.core.config as _cfg_mod
        importlib.reload(_cfg_mod)
        # reload 이후에 패치 (reload가 CONFIG_FILE을 원래 값으로 복원하기 때문)
        monkeypatch.setattr(_cfg_mod, "CONFIG_FILE", cfg_file)
        loaded = _cfg_mod.load_config()
        assert loaded.get("validated") is False
        assert "validated_at" in loaded
        assert "validated_host" in loaded

    def test_load_config_deep_merges_columns(self, tmp_path, monkeypatch):
        """기존 config에 columns 키가 있어도 누락된 tbl_key는 병합된다."""
        import hana_app.core.config as _cfg_mod
        cfg_file = tmp_path / "hana_config.json"
        # only t20 exists, t30 is missing
        cfg_file.write_text(
            json.dumps({"columns": {"t20": {"patient_id": "INDI_DSCM_NO"}}}),
            encoding="utf-8",
        )
        monkeypatch.setattr("hana_app.core.config.CONFIG_FILE", cfg_file)
        importlib.reload(_cfg_mod)
        loaded = _cfg_mod.load_config()
        # t20 should be preserved, t30 should be merged from DEFAULT_CONFIG (if it exists)
        assert "t20" in loaded.get("columns", {})
        # If DEFAULT_CONFIG has t30, it should be present
        if "t30" in _cfg_mod.DEFAULT_CONFIG.get("columns", {}):
            assert "t30" in loaded["columns"]

    def test_save_and_load_validated_true(self, tmp_path, monkeypatch):
        """validated=True로 저장 후 다시 로드하면 True."""
        cfg_file = tmp_path / "hana_config.json"
        import hana_app.core.config as _cfg_mod
        importlib.reload(_cfg_mod)
        # reload 이후에 패치 (reload가 CONFIG_FILE을 원래 값으로 복원하기 때문)
        monkeypatch.setattr(_cfg_mod, "CONFIG_FILE", cfg_file)

        cfg = _cfg_mod.load_config()
        cfg["validated"] = True
        cfg["validated_at"] = "2026-04-07T09:00:00"
        cfg["validated_host"] = "192.168.1.1"
        _cfg_mod.save_config(cfg)

        loaded = _cfg_mod.load_config()
        assert loaded["validated"] is True
        assert loaded["validated_at"] == "2026-04-07T09:00:00"
        assert loaded["validated_host"] == "192.168.1.1"
