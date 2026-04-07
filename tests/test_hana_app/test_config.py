"""hana_app/core/config.py validated 플래그 테스트."""
import importlib
import json
from pathlib import Path

import pytest

from hana_app.core.config import DEFAULT_CONFIG, load_config, save_config


class TestValidatedFlag:
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
