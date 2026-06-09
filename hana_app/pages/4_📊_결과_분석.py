"""
페이지 4: 학습 결과 분석 – 피처 중요도 / ROC / 혼동행렬 / 위험도 분포
"""
import datetime as dt
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
from hana_app.core.report_exporter import build_csv_bytes, build_docx_bytes, DOCX_AVAILABLE

st.set_page_config(page_title="결과 분석", page_icon="📊", layout="wide")
st.title("📊 학습 결과 분석")

_cfg = load_config()
# NOTE: 결과 분석은 session_state/디스크(RESULTS_DIR)의 결과만 읽고 라이브 HANA 를
# 쿼리하지 않는다. 따라서 HANA 테이블 검증 게이트로 페이지를 막으면 안 된다
# (RAW/저장 데이터로 학습한 결과를 미검증 HANA 설정 PC 에서 못 보던 버그 —
#  Page 3 data_mode 게이트 회귀와 동일 계열). 미검증 시 안내만 노출하고 진행한다.
if is_hana(_cfg) and not check_hana_validated(_cfg):
    st.caption(f"ℹ️ {get_validation_error(_cfg)} (결과 분석은 검증 없이도 가능)")

# ─────────────────────────────────────────────────────────────────────────────
# 결과 선택
# ─────────────────────────────────────────────────────────────────────────────
# 현재 세션 결과 or 저장된 결과
current = st.session_state.get("last_result")
saved = list_saved_results()

col_src, col_sel = st.columns([1, 3])
with col_src:
    # 현재 세션 결과가 없고 저장된 결과만 있으면 '저장된 결과'를 기본 선택한다.
    # (그렇지 않으면 라디오가 '현재 세션'(빈 값)에 머물러 결과가 있는데도
    #  '분석할 결과가 없습니다'로 잘못 막힌다 — 페이지 이동으로 세션 결과 유실 시.)
    _src_index = 0 if current else (1 if saved else 0)
    source = st.radio(
        "결과 소스",
        ["현재 세션", "저장된 결과"],
        index=_src_index,
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


def _fmt_n(v) -> str:
    """정수/실수만 천단위 콤마, 그 외('?')는 그대로 — `f'{\"?\":,}'` ValueError 방지."""
    return f"{v:,}" if isinstance(v, (int, float)) else str(v)


if target == "hierarchical":
    # 계층 결과 metrics 에는 accuracy/auc/train_size 가 없다(τ 임계값·f1_macro 만).
    # flat 지표/포맷을 그대로 쓰면 train_size='?' 에 콤마 포맷 → ValueError 로
    # 페이지가 제목만 뜨고 빈 화면처럼 보였다. 임계값 위주로 표시한다.
    _th = {**metrics, **result.get("meta", {}).get("thresholds", {})}
    _tau_red = _th.get("tau_red")
    _tau_review = _th.get("tau_review")
    hc1, hc2, hc3 = st.columns(3)
    hc1.metric("τ_red", f"{_tau_red:.3f}" if isinstance(_tau_red, (int, float)) else "?")
    hc2.metric("τ_review", f"{_tau_review:.3f}" if isinstance(_tau_review, (int, float)) else "?")
    hc3.metric("F1 (macro)", f"{metrics.get('f1_macro', 0):.4f}")
    st.info(
        "계층 분류 결과입니다. 아래 **🧮 위험도 분포** 탭에서 Yellow 세분화"
        "(yellow_subtype)·Red 의심(red_suspect) 분포를 확인하세요. "
        "Accuracy/AUC·혼동행렬 등 단일 분류기 지표는 계층 모델에 적용되지 않습니다."
    )
else:
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    mc2.metric("F1 (macro)", f"{metrics.get('f1_macro', 0):.4f}")
    mc3.metric("AUC", f"{metrics.get('roc_auc', metrics.get('roc_auc_ovr', 0)):.4f}")
    mc4.metric("CV 평균", f"{metrics.get('cv_mean', 0):.4f}")
    mc5.metric("CV 표준편차", f"±{metrics.get('cv_std', 0):.4f}")

    st.markdown(
        f"학습 {_fmt_n(metrics.get('train_size', '?'))}건 | "
        f"테스트 {_fmt_n(metrics.get('test_size', '?'))}건"
    )

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
    _fi_ok = fi_data is not None and (fi_data.empty is False if hasattr(fi_data, "empty") else bool(fi_data))

    if _fi_ok:
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


# ── 다운로드 섹션 ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("📥 결과 다운로드")

_dl_df = st.session_state.get("features_df")
_has_df = _dl_df is not None and not _dl_df.empty

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    if not DOCX_AVAILABLE:
        st.error("python-docx 미설치 — DOCX 내보내기 불가 (운영 PC: packages_win에 wheel 추가 필요)")
    else:
        try:
            docx_bytes = build_docx_bytes(result, _dl_df)
            _ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📄 DOCX 보고서 다운로드",
                data=docx_bytes,
                file_name=f"위험예측_보고서_{_ts}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except Exception as _e:
            st.error(f"DOCX 생성 오류: {_e}")

with col_dl2:
    if not _has_df:
        st.button("📋 대상자 CSV 다운로드", disabled=True)
        st.caption("현재 세션 학습 결과가 필요합니다 (저장된 결과에서는 비활성).")
    else:
        try:
            csv_bytes = build_csv_bytes(_dl_df)
            _ys = _dl_df.get("yellow_subtype", None)
            _red_n = int((_dl_df["risk_level"] == "Red").sum())
            _major_n = int((_ys == "Y_DDI_MAJOR").sum()) if _ys is not None else 0
            _triple_n = int((_ys == "Y_TRIPLE").sum()) if _ys is not None else 0
            _target_n = _red_n + _major_n + _triple_n
            if _target_n == 0:
                st.warning("추출 대상(Red / Y_DDI_MAJOR / Y_TRIPLE) 없음")
            else:
                _ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label=f"📋 대상자 CSV 다운로드 ({_target_n:,}명)",
                    data=csv_bytes,
                    file_name=f"대상자_위험분류_{_ts}.csv",
                    mime="text/csv; charset=utf-8",
                )
        except Exception as _e:
            st.error(f"CSV 생성 오류: {_e}")
