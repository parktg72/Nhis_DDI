"""
페이지 4: 학습 결과 분석 – 피처 중요도 / ROC / 혼동행렬 / 위험도 분포
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hana_app.core.ml_runner import list_saved_results, load_model, RISK_LABEL_MAP
from hana_app.core.config import load_config, is_hana
from hana_app.core.page_guards import check_hana_validated, get_validation_error
from hana_app.core.yellow_subtype_view import render_yellow_subtype_section

st.set_page_config(page_title="결과 분석", page_icon="📊", layout="wide")
st.title("📊 학습 결과 분석")

_cfg = load_config()
if is_hana(_cfg) and not check_hana_validated(_cfg):
    st.warning(get_validation_error(_cfg))
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 결과 선택
# ─────────────────────────────────────────────────────────────────────────────
# 현재 세션 결과 or 저장된 결과
current = st.session_state.get("last_result")
saved = list_saved_results()

col_src, col_sel = st.columns([1, 3])
with col_src:
    source = st.radio(
        "결과 소스",
        ["현재 세션", "저장된 결과"],
        disabled=(current is None and not saved),
    )

result = None
if source == "현재 세션" and current:
    result = current
    st.info(f"현재 세션 결과: **{current.get('model_name', '?')}**")
elif source == "저장된 결과" and saved:
    with col_sel:
        options = {
            r.get("timestamp", "?") + " – " + r.get("model_name", "?"): r
            for r in saved
        }
        sel_label = st.selectbox("결과 선택", list(options.keys()))
        result = options[sel_label]

if not result:
    st.warning("분석할 결과가 없습니다. 3단계 모델학습을 먼저 실행하세요.")
    st.stop()

metrics = result.get("metrics", {})
model_name = result.get("model_name", "?")
target = result.get("target", "risk_binary")

# ─────────────────────────────────────────────────────────────────────────────
# 핵심 지표 요약
# ─────────────────────────────────────────────────────────────────────────────
st.subheader(f"📌 {model_name} – 핵심 지표")

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
mc2.metric("F1 (macro)", f"{metrics.get('f1_macro', 0):.4f}")
mc3.metric("AUC", f"{metrics.get('roc_auc', metrics.get('roc_auc_ovr', 0)):.4f}")
mc4.metric("CV 평균", f"{metrics.get('cv_mean', 0):.4f}")
mc5.metric("CV 표준편차", f"±{metrics.get('cv_std', 0):.4f}")

st.markdown(f"학습 {metrics.get('train_size', '?'):,}건 | 테스트 {metrics.get('test_size', '?'):,}건")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 탭 구성
# ─────────────────────────────────────────────────────────────────────────────
tab_fi, tab_cm, tab_cv, tab_roc, tab_dist, tab_report, tab_compare = st.tabs([
    "📈 피처 중요도",
    "🔲 혼동 행렬",
    "📉 교차검증",
    "📉 ROC Curve",
    "🧮 위험도 분포",
    "📋 분류 보고서",
    "⚖️ 모델 비교",
])

# ── 탭1: 피처 중요도 ─────────────────────────────────────────────────────────
with tab_fi:
    fi_data = result.get("feature_importance")

    if fi_data:
        if isinstance(fi_data, list):
            fi_df = pd.DataFrame(fi_data)
        else:
            fi_df = fi_data

        fi_df = fi_df.sort_values("importance", ascending=True)

        FEAT_LABELS = {
            "drug_count": "총 약물 수",
            "drug_count_7d": "최근 7일 동시 복용",
            "institution_count": "처방 기관 수",
            "ddi_contraindicated": "금기 DDI",
            "ddi_major": "Major DDI",
            "ddi_moderate": "Moderate DDI",
            "ddi_minor": "Minor DDI",
            "triple_whammy": "Triple Whammy",
            "qt_risk_count": "QT 위험 약물",
            "dup_same_ingredient": "동일성분 중복",
            "dup_atc5": "ATC5 중복",
            "dup_atc4": "ATC4 중복",
            "dup_atc3": "ATC3 중복",
            "dup_efmdc": "약효분류 중복",
            "age": "연령",
            "sex_m": "성별(남)",
        }
        fi_df["label"] = fi_df["feature"].map(FEAT_LABELS).fillna(fi_df["feature"])

        fig = px.bar(
            fi_df,
            x="importance",
            y="label",
            orientation="h",
            title=f"피처 중요도 ({model_name})",
            color="importance",
            color_continuous_scale="Blues",
            height=500,
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("원본 데이터"):
            st.dataframe(fi_df[["feature", "label", "importance"]].sort_values("importance", ascending=False))
    else:
        st.info("피처 중요도 데이터가 없습니다.")

# ── 탭2: 혼동 행렬 ───────────────────────────────────────────────────────────
with tab_cm:
    cm = metrics.get("confusion_matrix")
    classes = metrics.get("classes", [])
    if cm:
        cm_arr = np.array(cm)
        class_names_map = {0: "Normal", 1: "위험"} if target == "risk_binary" else {
            v: k for k, v in RISK_LABEL_MAP.items()
        }
        labels = [class_names_map.get(c, str(c)) for c in classes]

        fig = ff.create_annotated_heatmap(
            z=cm_arr,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
            annotation_text=cm_arr.astype(str),
        )
        fig.update_layout(
            title="혼동 행렬",
            xaxis_title="예측",
            yaxis_title="실제",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # 정밀도 / 재현율 계산
        if cm_arr.shape[0] == 2:
            tn, fp, fn, tp = cm_arr.ravel()
            col_p, col_r, col_f = st.columns(3)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            col_p.metric("정밀도 (Precision)", f"{precision:.4f}")
            col_r.metric("재현율 (Recall)", f"{recall:.4f}")
            col_f.metric("F1 Score", f"{f1:.4f}")
    else:
        st.info("혼동 행렬 데이터가 없습니다.")

# ── 탭3: 교차검증 결과 ───────────────────────────────────────────────────────
with tab_cv:
    cv_scores = metrics.get("cv_scores", [])
    if cv_scores:
        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
            "Score": cv_scores,
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cv_df["Fold"],
            y=cv_df["Score"],
            marker_color="steelblue",
            name="CV Score",
        ))
        fig.add_hline(
            y=np.mean(cv_scores),
            line_dash="dash",
            line_color="red",
            annotation_text=f"평균: {np.mean(cv_scores):.4f}",
        )
        fig.update_layout(
            title=f"교차검증 결과 ({len(cv_scores)}-fold)",
            yaxis_title="Score",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        cv_stats = pd.DataFrame({
            "지표": ["평균", "표준편차", "최솟값", "최댓값"],
            "값": [
                f"{np.mean(cv_scores):.4f}",
                f"{np.std(cv_scores):.4f}",
                f"{np.min(cv_scores):.4f}",
                f"{np.max(cv_scores):.4f}",
            ],
        })
        st.dataframe(cv_stats, use_container_width=False)
    else:
        st.info("교차검증 결과가 없습니다.")

# ── ROC Curve 탭 ──────────────────────────────────────────────────────────────
with tab_roc:
    roc_data = metrics.get("roc_curve")
    if roc_data and "fpr" in roc_data and "tpr" in roc_data:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roc_data["fpr"],
            y=roc_data["tpr"],
            mode="lines",
            name=f"ROC (AUC={metrics.get('roc_auc', 0):.4f})",
            line={"color": "steelblue", "width": 2},
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random",
            line={"color": "gray", "dash": "dash"},
        ))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
    elif metrics.get("roc_auc_ovr"):
        st.info("ROC Curve는 이진 분류(risk_binary)에서만 표시됩니다. 다중 분류의 AUC(OvR)는 핵심 지표 탭을 확인하세요.")
    else:
        st.info("ROC Curve 데이터가 없습니다. 이번 개선 이전에 저장된 결과에는 roc_curve가 포함되지 않습니다.")

# ── 탭4: 위험도 분포 ─────────────────────────────────────────────────────────
with tab_dist:
    df = st.session_state.get("features_df")
    risk_summary = result.get("risk_summary")
    drug_stats = result.get("drug_count_stats")
    ddi_means = result.get("ddi_means")

    color_map = {"Red": "#e74c3c", "Yellow": "#f39c12", "Green": "#27ae60", "Normal": "#95a5a6"}

    if df is not None:
        # 현재 세션 데이터 — 원본 DataFrame 사용
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            risk_dist = df["risk_level"].value_counts().reset_index()
            risk_dist.columns = ["위험도", "환자수"]
            fig = px.pie(risk_dist, names="위험도", values="환자수",
                         color="위험도", color_discrete_map=color_map, title="위험도 분포")
            st.plotly_chart(fig, use_container_width=True)
        with col_d2:
            fig = px.bar(risk_dist.sort_values("위험도"), x="위험도", y="환자수",
                         color="위험도", color_discrete_map=color_map,
                         title="위험도별 환자 수", text="환자수")
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("약물 수 분포")
        fig = px.histogram(df, x="drug_count", color="risk_level",
                           color_discrete_map=color_map, nbins=30, barmode="overlay",
                           title="다재약물 환자의 약물 수 분포",
                           labels={"drug_count": "약물 수", "risk_level": "위험도"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("DDI 심각도 분포")
        ddi_cols = ["ddi_contraindicated", "ddi_major", "ddi_moderate", "ddi_minor"]
        ddi_labels = ["금기", "Major", "Moderate", "Minor"]
        ddi_means_live = [df[c].mean() for c in ddi_cols]
        fig = go.Figure(go.Bar(
            x=ddi_labels, y=ddi_means_live,
            marker_color=["#e74c3c", "#e67e22", "#f1c40f", "#3498db"],
            text=[f"{v:.2f}" for v in ddi_means_live], textposition="outside",
        ))
        fig.update_layout(title="DDI 심각도별 평균 쌍 수", yaxis_title="평균 DDI 쌍 수")
        st.plotly_chart(fig, use_container_width=True)

        # ── Yellow 세분화 (계층 분류) — yellow_subtype 컬럼 있을 때만 표시 ──────
        render_yellow_subtype_section(df)

    elif risk_summary:
        # 저장된 결과 — 요약 통계로 차트 재구성
        st.caption("저장된 결과에서 요약 통계를 불러와 표시합니다.")
        risk_dist = pd.DataFrame(
            [{"위험도": k, "환자수": v} for k, v in risk_summary.items()]
        )
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig = px.pie(risk_dist, names="위험도", values="환자수",
                         color="위험도", color_discrete_map=color_map, title="위험도 분포 (저장 요약)")
            st.plotly_chart(fig, use_container_width=True)
        with col_d2:
            fig = px.bar(risk_dist, x="위험도", y="환자수",
                         color="위험도", color_discrete_map=color_map,
                         title="위험도별 환자 수", text="환자수")
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        if ddi_means:
            st.subheader("DDI 심각도 평균 (저장 요약)")
            labels_map = {
                "ddi_contraindicated": "금기", "ddi_major": "Major",
                "ddi_moderate": "Moderate", "ddi_minor": "Minor",
            }
            ddi_labels = [labels_map[k] for k in ddi_means]
            ddi_vals = list(ddi_means.values())
            fig = go.Figure(go.Bar(
                x=ddi_labels, y=ddi_vals,
                marker_color=["#e74c3c", "#e67e22", "#f1c40f", "#3498db"],
                text=[f"{v:.2f}" for v in ddi_vals], textposition="outside",
            ))
            fig.update_layout(title="DDI 심각도별 평균 쌍 수", yaxis_title="평균 DDI 쌍 수")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("피처 데이터가 없습니다. 3단계 모델학습을 먼저 실행하세요.")

# ── 탭5: 분류 보고서 ─────────────────────────────────────────────────────────
with tab_report:
    report = metrics.get("classification_report", "")
    if report:
        st.text(report)
    else:
        st.info("분류 보고서가 없습니다.")

# ── 탭6: 모델 비교 ───────────────────────────────────────────────────────────
with tab_compare:
    if not saved:
        st.info("저장된 결과가 없습니다. 여러 모델을 학습한 후 비교하세요.")
    else:
        compare_rows = []
        for r in saved:
            m_r = r.get("metrics", {})
            compare_rows.append({
                "시간": r.get("timestamp", "?"),
                "모델": r.get("model_name", "?"),
                "타겟": r.get("target", "?"),
                "Accuracy": round(m_r.get("accuracy", 0), 4),
                "F1 (macro)": round(m_r.get("f1_macro", 0), 4),
                "AUC": round(m_r.get("roc_auc", m_r.get("roc_auc_ovr", 0)), 4),
                "CV 평균": round(m_r.get("cv_mean", 0), 4),
                "학습 수": m_r.get("train_size", 0),
            })

        cmp_df = pd.DataFrame(compare_rows)
        st.dataframe(
            cmp_df.style.highlight_max(
                subset=["Accuracy", "F1 (macro)", "AUC", "CV 평균"],
                color="lightgreen",
            ),
            use_container_width=True,
        )

        # 비교 차트
        fig = go.Figure()
        for metric_name in ["Accuracy", "F1 (macro)", "AUC"]:
            fig.add_trace(go.Bar(
                name=metric_name,
                x=cmp_df["모델"] + "\n" + cmp_df["시간"].str[:8],
                y=cmp_df[metric_name],
            ))
        fig.update_layout(
            barmode="group",
            title="모델별 성능 비교",
            yaxis_title="Score",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
