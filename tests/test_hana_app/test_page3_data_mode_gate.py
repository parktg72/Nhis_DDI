"""Page 3 (모델 학습) — 데이터 소스 모드 게이트 회귀 테스트.

회귀 배경
---------
HANA 테이블 검증 게이트(`st.stop()`)가 데이터 모드 라디오보다 **먼저** 무조건
실행되던 버그가 있었다. 그 결과 운영 PC(HANA 설정)에서 테이블 검증이 안 된 상태면
**RAW / SAVED 모드 자체에 도달할 수 없었다** — 둘 다 로컬 parquet 만 읽어 라이브 DB가
불필요한데도 차단됨. (헤드리스 검증이 `data_source=sas` 로 이 게이트를 우회해서 발견이
늦어졌다.)

수정: 데이터 모드 라디오를 게이트 위로 올리고, 검증/재연결 게이트를
`data_mode == EXTRACT` 일 때만 적용한다.

이 테스트는 **HANA-이면서-미검증** config 에서:
- RAW / SAVED 모드는 게이트 경고 없이 해당 섹션이 렌더되어야 하고,
- EXTRACT 모드는 게이트 경고 + st.stop 으로 여전히 차단되어야 한다.
"""
import glob
import json
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PAGE_PATH = glob.glob(str(ROOT / "hana_app" / "pages" / "3_*.py"))[0]

GATE_WARNING_SUBSTR = "HANA 테이블 검증"
RAW_SECTION_SUBSTR = "다운로드 받은 Raw 데이터"


def _hana_unvalidated_cfg() -> dict:
    """data_source=hana, validated=False 인 최소 config (DEFAULT_CONFIG 기반)."""
    from hana_app.core.config import DEFAULT_CONFIG

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    cfg["data_source"] = "hana"
    cfg["validated"] = False
    cfg["validated_host"] = ""
    return cfg


def _run_page(data_mode: str, raw_dir: str | None = None):
    """주어진 data_mode 로 페이지를 헤드리스 실행한 AppTest 반환."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(PAGE_PATH, default_timeout=60)
    at.session_state["data_mode"] = data_mode
    if raw_dir is not None:
        at.session_state["raw_dir_input"] = raw_dir
    # 페이지의 `from hana_app.core.config import load_config` 가 실행 시점에
    # 패치본을 바인딩하도록 source 모듈 속성을 패치한다.
    with patch("hana_app.core.config.load_config", return_value=_hana_unvalidated_cfg()):
        at.run()
    return at


def _warning_texts(at) -> list[str]:
    return [w.value for w in at.warning]


def test_raw_mode_not_blocked_by_hana_validation_gate(tmp_path):
    """HANA 미검증이어도 RAW 모드는 게이트에 막히지 않고 섹션이 렌더된다."""
    at = _run_page("raw", raw_dir=str(tmp_path))  # 빈 폴더 → records 없음 경고는 무방

    warnings = _warning_texts(at)
    assert not any(GATE_WARNING_SUBSTR in w for w in warnings), (
        f"RAW 모드인데 HANA 검증 게이트가 트립됨: {warnings}"
    )
    # 게이트 st.stop 에 막혔다면 RAW 서브헤더 자체가 렌더되지 않는다.
    subheaders = [s.value for s in at.subheader]
    assert any(RAW_SECTION_SUBSTR in s for s in subheaders), (
        f"RAW 섹션이 렌더되지 않음 (게이트 차단 의심). subheaders={subheaders}"
    )
    assert not at.exception


def test_saved_mode_not_blocked_by_hana_validation_gate():
    """HANA 미검증이어도 SAVED 모드는 게이트에 막히지 않는다."""
    at = _run_page("saved")
    warnings = _warning_texts(at)
    assert not any(GATE_WARNING_SUBSTR in w for w in warnings), (
        f"SAVED 모드인데 HANA 검증 게이트가 트립됨: {warnings}"
    )
    assert not at.exception


def test_extract_mode_still_blocked_when_unvalidated():
    """회귀 방지: EXTRACT 모드는 HANA 미검증 시 여전히 게이트로 차단되어야 한다."""
    at = _run_page("extract")
    warnings = _warning_texts(at)
    assert any(GATE_WARNING_SUBSTR in w for w in warnings), (
        f"EXTRACT 모드인데 HANA 검증 게이트가 트립되지 않음: {warnings}"
    )


def test_raw_mode_fails_closed_when_disease_filter_enabled():
    """Raw 다운로드 모드는 T40 조회가 없으므로 ICD-10 질환 필터를 silent-ignore 하면 안 된다."""
    source = Path(PAGE_PATH).read_text(encoding="utf-8")

    assert "data_mode == DATA_MODE_RAW and use_disease_filter" in source
    assert "질환 필터는 HANA/SAS 추출 모드에서만 적용됩니다" in source
