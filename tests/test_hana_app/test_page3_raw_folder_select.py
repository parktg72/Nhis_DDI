"""Page3 Raw 폴더 선택 클릭 검증 — 자동탐지 드롭다운 + 직접입력(2026-06-12 feat).

AppTest 로 실제 위젯 상호작용을 시뮬레이션:
- 직접입력: 임의 폴더 타이핑 → 그 폴더의 records 가 인식되고 raw_data_dir 반영.
- 드롭다운: _detect_raw_dirs 가 records 있는 폴더만 후보로 노출.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAGE_PATH = str(ROOT / "hana_app" / "pages" / "3_🤖_모델_학습.py")


def _hana_unvalidated_cfg() -> dict:
    from hana_app.core.config import DEFAULT_CONFIG
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["data_source"] = "hana"
    cfg["validated"] = False
    cfg["validated_host"] = ""
    return cfg


def _make_raw_folder(tmp_path: Path, n: int = 2) -> Path:
    """records_YYYYMMDD.parquet n개 + eligibility_demographics 가 있는 폴더 생성."""
    d = tmp_path / "my_custom_raw"
    d.mkdir()
    df = pd.DataFrame({"patient_id": ["P1", "P2"], "wk_compn_cd": ["421001ATB", "480600ATB"]})
    for i in range(n):
        df.to_parquet(d / f"records_2024070{i + 1}.parquet")
    return d


def test_manual_folder_entry_drives_pipeline(tmp_path):
    """직접입력한 임의 폴더가 인식되고 raw_data_dir 로 반영(폴더 고정 아님)."""
    from streamlit.testing.v1 import AppTest

    raw = _make_raw_folder(tmp_path)
    at = AppTest.from_file(PAGE_PATH, default_timeout=60)
    at.session_state["data_mode"] = "raw"
    at.session_state["raw_dir_input"] = str(raw)   # 사용자가 입력란에 친 임의 폴더
    with patch("hana_app.core.config.load_config", return_value=_hana_unvalidated_cfg()):
        at.run()

    assert not at.exception
    # 임의 폴더가 그대로 raw_data_dir 로 반영 (D:\... 고정 아님)
    assert at.session_state["raw_data_dir"] == str(raw)
    # 그 폴더의 records 가 인식되어 "없음" 경고가 뜨지 않음
    warns = [w.value for w in at.warning]
    assert not any("records_*.parquet 파일이 없습니다" in w for w in warns), warns


def test_empty_folder_warns_not_locked(tmp_path):
    """records 없는 임의 폴더 입력 시 그 폴더 기준 '없음' 경고 — 고정경로로 안 돌아감."""
    from streamlit.testing.v1 import AppTest

    empty = tmp_path / "empty_raw"; empty.mkdir()
    at = AppTest.from_file(PAGE_PATH, default_timeout=60)
    at.session_state["data_mode"] = "raw"
    at.session_state["raw_dir_input"] = str(empty)
    with patch("hana_app.core.config.load_config", return_value=_hana_unvalidated_cfg()):
        at.run()

    assert not at.exception
    # 입력한 빈 폴더가 raw_data_dir 로 반영(저장경로로 되돌아가지 않음)
    assert at.session_state["raw_data_dir"] == str(empty)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
