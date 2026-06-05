"""Page 3 (모델 학습) — 예측 타겟 선택 보존 회귀 테스트.

회귀 배경
---------
사용자가 '계층 분류(hierarchical)' 를 선택했는데도 학습 시작 시 일반 4분류
(risk_label) XGBoost 분기로 새는 버그가 PyWebView 실사용에서 보고됨
(헤드리스 통과 / 실데스크톱 실패).

원인: 예측 타겟 selectbox 가 keyless 였다. keyless selectbox 는 앞쪽 조건부 위젯이
rerun 마다 렌더/언렌더되면 structural id 가 흔들려 값이 config 기본값으로 리셋된다.

수정: selectbox 에 stable `key="train_target_select"` 부여 → 값이 session_state 에
고정되어 위치 변동·페이지 이동·환경 차이에도 보존된다.

이 테스트는
1) selectbox 가 stable key 를 통해 session_state 에 값을 고정하는지,
2) SAVED 모드 전체 시나리오(불러오기 → 계층 선택 → 학습 시작)에서 계층 분기로
   올바르게 dispatch 되는지
를 검증한다.
"""
import glob
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAGE_PATH = glob.glob(str(ROOT / "hana_app" / "pages" / "3_*.py"))[0]
DATASET_DIR = ROOT / "hana_app" / "data" / "datasets"

TARGET_KEY = "train_target_select"


def _make_saved_dataset(tmp_name: str) -> Path:
    from hana_app.core.ml_runner import FEATURE_COLS, RISK_LABEL_MAP

    rng = np.random.default_rng(0)
    n = 400
    levels = rng.choice(["Red", "Yellow", "Green", "Normal"], size=n, p=[0.15, 0.35, 0.25, 0.25])
    ysub = np.where(
        levels == "Yellow",
        rng.choice(["Y_MIX", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG", "Y_OTHER"], size=n),
        None,
    )
    data = {c: rng.integers(0, 10, size=n).astype(float) for c in FEATURE_COLS}
    data["age"] = rng.integers(20, 90, size=n)
    data["sex_m"] = rng.integers(0, 2, size=n)
    df = pd.DataFrame(data)
    df["risk_level"] = levels
    df["risk_label"] = [RISK_LABEL_MAP[x] for x in levels]
    df["risk_binary"] = [1 if x in ("Red", "Yellow") else 0 for x in levels]
    df["yellow_subtype"] = ysub
    df["drug_count"] = rng.integers(5, 20, size=n)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    p = DATASET_DIR / f"features_{tmp_name}.parquet"
    df.to_parquet(p, index=False)
    (DATASET_DIR / f"features_{tmp_name}.json").write_text(
        json.dumps({"source": "test", "total_patients": n}), encoding="utf-8"
    )
    return p


def _cfg() -> dict:
    from hana_app.core.config import DEFAULT_CONFIG

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["data_source"] = "hana"
    cfg["validated"] = True
    return cfg


def _sas_cfg() -> dict:
    """EXTRACT 모드에서 HANA 재연결 게이트(st.stop)를 우회 — 전체 페이지 렌더."""
    from hana_app.core.config import DEFAULT_CONFIG

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["data_source"] = "sas"
    return cfg


def _find_target_sb(at):
    for sb in at.selectbox:
        if sb.label and "예측 타겟" in sb.label:
            return sb
    return None


def _click(at, label_sub):
    for b in at.button:
        if b.label and label_sub in b.label:
            b.click().run()
            return True
    return False


def test_target_selectbox_has_stable_key():
    """타겟 selectbox 는 반드시 stable key 로 값을 session_state 에 고정한다."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(PAGE_PATH, default_timeout=120)
    at.session_state["data_mode"] = "extract"
    with patch("hana_app.core.config.load_config", return_value=_sas_cfg()):
        at.run()
        assert not at.exception, [str(e) for e in at.exception]
        sb = _find_target_sb(at)
        assert sb is not None, "예측 타겟 selectbox 없음"

        sb.set_value("hierarchical").run()
        # key 가 있어야 session_state 에 값이 보존된다.
        assert TARGET_KEY in at.session_state, (
            f"타겟 selectbox 가 keyless 다 (key='{TARGET_KEY}' 누락) — rerun 시 리셋 위험"
        )
        assert at.session_state[TARGET_KEY] == "hierarchical"
        assert _find_target_sb(at).value == "hierarchical"


def test_saved_mode_hierarchical_dispatch():
    """SAVED 모드 전체 플로우: 계층 선택 → 학습 시작 → 계층 분기로 dispatch."""
    from streamlit.testing.v1 import AppTest

    name = "ZZ_target_persist_test"
    p = _make_saved_dataset(name)
    try:
        at = AppTest.from_file(PAGE_PATH, default_timeout=180)
        at.session_state["data_mode"] = "saved"
        with patch("hana_app.core.config.load_config", return_value=_cfg()):
            at.run()
            assert not at.exception, [str(e) for e in at.exception]

            # test 데이터셋(features_ZZ_*)은 reverse 정렬상 맨 앞 → 기본 선택됨.
            assert _click(at, "데이터 불러오기"), "불러오기 버튼 없음"
            _fdf = at.session_state["features_df"] if "features_df" in at.session_state else None
            assert _fdf is not None, "features_df 로드 안됨"

            sb = _find_target_sb(at)
            assert sb is not None, f"타겟 selectbox 없음. labels={[s.label for s in at.selectbox]}"
            sb.set_value("hierarchical").run()
            assert _find_target_sb(at).value == "hierarchical"

            assert _click(at, "학습 시작"), "학습시작 버튼 없음"

            subs = [s.value for s in at.subheader]
            hier = any("계층 분류 학습 중" in s for s in subs)
            normal = any("모델 학습 중" in s for s in subs)
            assert hier and not normal, (
                f"계층분류 선택했는데 일반 분기 실행됨 "
                f"(hier={hier}, normal={normal}, last_target={_find_target_sb(at).value!r}, "
                f"exc={[str(e) for e in at.exception]})"
            )
    finally:
        p.unlink(missing_ok=True)
        p.with_suffix(".json").unlink(missing_ok=True)
