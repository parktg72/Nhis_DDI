"""Page 4 (결과 분석) — 결과 렌더 회귀 테스트.

회귀 배경
---------
- 계층(hierarchical) 결과는 metrics 에 train_size/test_size 가 없다. line 80 의
  `f"{metrics.get('train_size','?'):,}"` 가 '?'(str) 에 `,` 포맷을 적용해
  ValueError → 페이지가 제목만 뜨고 빈 화면처럼 보임.
- 현재 세션 결과(last_result)가 없고 저장된 결과만 있을 때 라디오 기본값이
  '현재 세션'(빈 값)이라 '분석할 결과가 없습니다'로 막힘(잘못된 빈 화면).
"""
import glob
import json
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGE_PATH = glob.glob(str(ROOT / "hana_app" / "pages" / "4_*.py"))[0]


def _cfg() -> dict:
    from hana_app.core.config import DEFAULT_CONFIG

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["data_source"] = "sas"  # is_hana False → 검증 게이트 우회
    return cfg


def _hier_result() -> dict:
    return {
        "model_name": "hierarchical",
        "target": "hierarchical",
        "model_path": "models/hierarchical/x",
        "metrics": {"tau_red": 0.7, "tau_review": 0.3, "f1_macro": 0.0},
        "meta": {"thresholds": {"tau_red": 0.7}},
    }


def _run(at):
    with patch("hana_app.core.config.load_config", return_value=_cfg()):
        at.run()
    return at


def test_hierarchical_result_does_not_crash_page4():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(PAGE_PATH, default_timeout=60)
    at.session_state["last_result"] = _hier_result()
    _run(at)
    assert not at.exception, (
        "계층 결과로 Page 4 가 크래시함(train_size 포맷 ValueError 추정): "
        + "; ".join(str(e) for e in at.exception)
    )
    # 계층 임계값 섹션이 떠야 한다(빈 화면 아님).
    subs = [s.value for s in at.subheader]
    assert any("핵심 지표" in s for s in subs), f"핵심 지표 섹션 없음. subs={subs}"


def test_saved_result_shown_when_no_session_result():
    """현재 세션 결과가 없고 저장된 결과만 있을 때 빈 화면으로 막히지 않는다."""
    from hana_app.core.ml_runner import RESULTS_DIR
    from streamlit.testing.v1 import AppTest

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rp = RESULTS_DIR / "result_zz_page4test.json"
    rp.write_text(json.dumps({
        "timestamp": "20260101_000000",
        "model_name": "xgboost",
        "target": "risk_binary",
        "metrics": {"accuracy": 0.9, "f1_macro": 0.8, "train_size": 100, "test_size": 25},
    }), encoding="utf-8")
    try:
        at = AppTest.from_file(PAGE_PATH, default_timeout=60)
        # last_result(현재 세션) 없음 → 저장된 결과로 기본 선택돼야 함.
        _run(at)
        assert not at.exception, [str(e) for e in at.exception]
        warns = [w.value for w in at.warning]
        assert not any("분석할 결과가 없습니다" in w for w in warns), (
            f"저장된 결과가 있는데 빈 화면으로 막힘: warnings={warns}"
        )
    finally:
        rp.unlink(missing_ok=True)
