"""Yellow 세분화 UI 헬퍼.

페이지 4 (결과 분석) 의 위험도 분포 탭에서 yellow_subtype / red_suspect / action
3 컬럼의 분포를 렌더링한다. 순수 집계 함수와 Streamlit 렌더링 함수를 분리해
집계 로직은 단위 테스트 가능하도록 설계.
"""
from __future__ import annotations

import pandas as pd

from hana_app.core.hierarchical_runner import ACTION_BY_LABEL, YELLOW_SUBTYPE_LABELS

# Red 권장 개입 라벨 — risk_level=="Red" 건은 stage2 라벨이 아니므로
# ACTION_BY_LABEL 대신 여기서 개입 분포에 "즉각 개입"으로 합산한다.
RED_ACTION = "즉각 개입"

# Yellow 세부 라벨 컬러 팔레트 — 동일 "Yellow" 계열 내에서 세분화 시각화
YELLOW_SUBTYPE_COLORS: dict[str, str] = {
    "Y_TRIPLE":    "#c0392b",  # 진한 적오렌지 — 의료인 전화 (yellow 요소 3개+)
    "Y_DOUBLE":    "#d35400",  # 진한 오렌지 — 문자 알림 (yellow 요소 2개)
    "Y_DDI_MAJOR": "#e67e22",  # 오렌지 — 약사 전화
    "Y_DDI_MOD":   "#f39c12",  # 황색 — 문자 알림
    "Y_DUP":       "#f1c40f",  # 옅은 황색 — 문서 + 문자 알림
    "Y_FRAG":      "#f7dc6f",  # 연노랑 — 문자 알림
    "Y_OTHER":     "#95a5a6",  # 회색 — 비분류 (모니터링 대상)
}


def summarize_yellow_subtypes(df: pd.DataFrame) -> pd.DataFrame:
    """df["yellow_subtype"] 분포 집계. None/NaN 제외 후 정렬.

    Returns
    -------
    DataFrame with columns ["yellow_subtype", "count", "action"].
    yellow_subtype 컬럼이 없으면 빈 DataFrame.
    """
    if "yellow_subtype" not in df.columns:
        return pd.DataFrame(columns=["yellow_subtype", "count", "action"])
    s = df["yellow_subtype"].dropna()
    if len(s) == 0:
        return pd.DataFrame(columns=["yellow_subtype", "count", "action"])
    counts = s.value_counts().reset_index()
    counts.columns = ["yellow_subtype", "count"]
    counts["action"] = counts["yellow_subtype"].map(
        lambda lbl: ACTION_BY_LABEL.get(lbl, "알림 없음")
    )
    return counts


def count_red_suspect(df: pd.DataFrame) -> int | None:
    """red_suspect=True 건수. 컬럼 없으면 None."""
    if "red_suspect" not in df.columns:
        return None
    return int(df["red_suspect"].eq(True).sum())


def summarize_actions(df: pd.DataFrame) -> pd.DataFrame:
    """권장 개입(action) 분포 — Red 포함. count 내림차순 정렬.

    Red(risk_level=="Red")는 stage2 라벨이 아니므로 RED_ACTION("즉각 개입")으로,
    Yellow 세부 라벨은 ACTION_BY_LABEL 로 매핑해 같은 action 끼리 합산한다.

    Returns
    -------
    DataFrame with columns ["action", "count"]. 둘 다 없으면 빈 DataFrame.
    """
    frames: list[pd.DataFrame] = []
    sub = summarize_yellow_subtypes(df)
    if not sub.empty:
        frames.append(sub[["action", "count"]])
    if "risk_level" in df.columns:
        red_count = int((df["risk_level"] == "Red").sum())
        if red_count > 0:
            frames.append(pd.DataFrame({"action": [RED_ACTION], "count": [red_count]}))
    if not frames:
        return pd.DataFrame(columns=["action", "count"])
    out = (
        pd.concat(frames, ignore_index=True)
        .groupby("action", as_index=False)["count"].sum()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return out


def render_yellow_subtype_section(df: pd.DataFrame) -> None:  # pragma: no cover
    """Streamlit 렌더링 래퍼 — st 컨텍스트 필요. 로직은 위 순수 함수에 위임."""
    import plotly.express as px
    import streamlit as st

    summary = summarize_yellow_subtypes(df)
    if summary.empty:
        return  # 계층 라벨 없음 — 섹션 전체 생략 (backward compat)

    st.subheader("🟡 Yellow 세분화")
    st.caption(
        "yellow_subtype 컬럼은 clinical_rules 기반 세분화 (Y_TRIPLE=요소3+ / Y_DOUBLE=요소2 / "
        "Y_DDI_MAJOR / Y_DDI_MOD / Y_DUP / Y_FRAG / Y_OTHER). action 은 계층 설계 상 매핑된 개입."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.pie(
            summary, names="yellow_subtype", values="count",
            color="yellow_subtype", color_discrete_map=YELLOW_SUBTYPE_COLORS,
            title="Yellow 세부 라벨 분포",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.bar(
            summary, x="yellow_subtype", y="count",
            color="yellow_subtype", color_discrete_map=YELLOW_SUBTYPE_COLORS,
            title="Yellow 세부 라벨별 환자 수", text="count",
            category_orders={
                "yellow_subtype": list(YELLOW_SUBTYPE_LABELS) + ["Y_OTHER"],
            },
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # 개입 (action) 분포 — Red("즉각 개입") + Yellow 세부 라벨 합산
    action_summary = summarize_actions(df)
    fig = px.bar(
        action_summary, x="action", y="count",
        title="권장 개입 (action) 분포 — Red 포함", text="count",
        color="action",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Red 의심 (red_suspect) 카운트 — 컬럼 존재 시만
    rs_count = count_red_suspect(df)
    if rs_count is not None:
        total = len(df)
        pct = (rs_count / total * 100) if total else 0.0
        st.metric(
            "Red 의심 (red_suspect=True)",
            f"{rs_count:,}건",
            delta=f"{pct:.1f}% (τ_review ≤ p_red < τ_red 구간)",
        )
