"""Y_OTHER 증가율 모니터링 — Task 7.

Y_OTHER 는 clinical_rules 가 분류 못 한 Yellow 케이스. 새로운 약물/조합이 들어와
규칙이 못 따라가면 Y_OTHER 비율이 상승 → 규칙 드리프트 시그널.

본 모듈은 features_df 의 yellow_subtype 분포 스냅샷을 계산하고, baseline 스냅샷과
비교해 드리프트 알람을 발생시킨다. 시계열 로그 적재는 후속 과제.
"""
from __future__ import annotations

import pandas as pd

# 기본 알람 임계 — 운영 보정 가능
DEFAULT_DELTA_PP = 5.0       # 절대 증가 (퍼센트 포인트)
DEFAULT_DELTA_RELATIVE = 0.20  # 상대 증가 (20% 이상)


def compute_y_other_snapshot(df: pd.DataFrame) -> dict:
    """현재 features_df 의 Y_OTHER 분포 스냅샷.

    Returns
    -------
    dict
        - y_other_count : Y_OTHER 환자 수
        - total_yellow_count : Yellow 전체 (Y_OTHER 포함)
        - y_other_pct_in_yellow : Y_OTHER / Yellow 비율 (%, 0~100)
        - total_count : df 행 수
        - y_other_pct_total : Y_OTHER / 전체 비율 (%, 0~100)
        - has_yellow_subtype_col : bool (컬럼 존재 여부)

    yellow_subtype 컬럼이 없으면 모든 카운트 0, has=False.
    """
    n_total = int(len(df))
    if "yellow_subtype" not in df.columns or n_total == 0:
        return {
            "y_other_count": 0,
            "total_yellow_count": 0,
            "y_other_pct_in_yellow": 0.0,
            "total_count": n_total,
            "y_other_pct_total": 0.0,
            "has_yellow_subtype_col": "yellow_subtype" in df.columns,
        }

    s = df["yellow_subtype"].dropna()
    y_other = int((s == "Y_OTHER").sum())
    total_yellow = int(len(s))
    pct_in_yellow = (100.0 * y_other / total_yellow) if total_yellow else 0.0
    pct_total = 100.0 * y_other / n_total
    return {
        "y_other_count": y_other,
        "total_yellow_count": total_yellow,
        "y_other_pct_in_yellow": pct_in_yellow,
        "total_count": n_total,
        "y_other_pct_total": pct_total,
        "has_yellow_subtype_col": True,
    }


def compare_snapshots(
    current: dict,
    baseline: dict,
    delta_pp_threshold: float = DEFAULT_DELTA_PP,
    delta_relative_threshold: float = DEFAULT_DELTA_RELATIVE,
) -> dict:
    """현재 스냅샷 vs baseline 비교 — 드리프트 알람 정보 생성.

    "y_other_pct_in_yellow" 기준 (Yellow 모집단 내 Y_OTHER 비율 변화).

    Returns
    -------
    dict
        - current_pct, baseline_pct, delta_pp, delta_relative
        - alert : bool — 절대(pp) 또는 상대(%) 임계 초과
        - reasons : list[str] — 알람 발생 원인
        - message : str — 한국어 요약

    delta_relative 은 baseline 이 0 일 때 정의 불가 → +inf 로 보고, 그 경우
    current_pct 만 양수면 alert=True.
    """
    cur = current["y_other_pct_in_yellow"]
    base = baseline["y_other_pct_in_yellow"]
    delta_pp = cur - base
    if base > 0:
        delta_relative = (cur - base) / base
    else:
        delta_relative = float("inf") if cur > 0 else 0.0

    reasons: list[str] = []
    if delta_pp >= delta_pp_threshold:
        reasons.append(
            f"절대 증가 {delta_pp:.2f}pp ≥ 임계 {delta_pp_threshold:.2f}pp"
        )
    if delta_relative >= delta_relative_threshold:
        if delta_relative == float("inf"):
            reasons.append("baseline 이 0 인데 현재 양수 — 신규 출현")
        else:
            reasons.append(
                f"상대 증가 {delta_relative * 100:.1f}% ≥ 임계 "
                f"{delta_relative_threshold * 100:.1f}%"
            )
    alert = bool(reasons)

    if alert:
        message = (
            f"⚠️ Y_OTHER 비율 상승 감지 — baseline {base:.2f}% → 현재 {cur:.2f}%. "
            f"clinical_rules 드리프트 가능성, 규칙 갱신 검토 필요."
        )
    else:
        message = f"✅ 드리프트 없음 (baseline {base:.2f}% → 현재 {cur:.2f}%)."

    return {
        "current_pct": cur,
        "baseline_pct": base,
        "delta_pp": delta_pp,
        "delta_relative": delta_relative,
        "alert": alert,
        "reasons": reasons,
        "message": message,
    }


def render_y_other_drift_section(  # pragma: no cover
    df: pd.DataFrame,
    baseline_df: pd.DataFrame | None = None,
    delta_pp_threshold: float = DEFAULT_DELTA_PP,
    delta_relative_threshold: float = DEFAULT_DELTA_RELATIVE,
) -> None:
    """Streamlit 렌더링 — page 6 신규 탭에서 호출."""
    import streamlit as st

    snap = compute_y_other_snapshot(df)
    if not snap["has_yellow_subtype_col"]:
        st.info(
            "현재 features_df 에 `yellow_subtype` 컬럼이 없습니다. "
            "계층 분류 ETL (CLINICAL_STANDARDS_VERSION ≥ v1.0) 산출물에서만 표시됩니다."
        )
        return

    st.subheader("🟡 Y_OTHER 드리프트")
    st.caption(
        "Y_OTHER 은 clinical_rules 로 분류 못 한 Yellow 케이스. "
        "비율 상승은 새로운 약물/조합 출현으로 규칙 갱신이 필요한 신호."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Y_OTHER 환자수", f"{snap['y_other_count']:,}")
    c2.metric("Yellow 내 Y_OTHER 비율", f"{snap['y_other_pct_in_yellow']:.2f}%")
    c3.metric("전체 환자 중 Y_OTHER 비율", f"{snap['y_other_pct_total']:.2f}%")

    if baseline_df is not None:
        baseline_snap = compute_y_other_snapshot(baseline_df)
        if baseline_snap["has_yellow_subtype_col"]:
            cmp = compare_snapshots(
                snap, baseline_snap,
                delta_pp_threshold=delta_pp_threshold,
                delta_relative_threshold=delta_relative_threshold,
            )
            (st.error if cmp["alert"] else st.success)(cmp["message"])
            d1, d2 = st.columns(2)
            d1.metric("baseline → 현재 (pp)", f"{cmp['delta_pp']:+.2f}")
            if cmp["delta_relative"] == float("inf"):
                d2.metric("상대 증가율", "∞ (신규)")
            else:
                d2.metric("상대 증가율", f"{cmp['delta_relative'] * 100:+.1f}%")
            if cmp["reasons"]:
                with st.expander("알람 원인"):
                    for r in cmp["reasons"]:
                        st.write(f"- {r}")
    else:
        st.info(
            "baseline 스냅샷이 없습니다 — 드리프트 비교를 위해 기준 시점의 "
            "features_df 를 함께 로드하세요."
        )
