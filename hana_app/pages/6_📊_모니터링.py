"""
모니터링 대시보드 — Streamlit 6번 페이지

4-tab 구성:
  Tab 1: 실시간 예측 현황 (metrics_live.jsonl)
  Tab 2: 드리프트 감지 (drift_{partition}.json)
  Tab 3: 알림 이력 (alerts_{partition}.json)
  Tab 4: 시스템 상태
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from hana_app.pages._monitoring_helpers import (
    compute_disagree_rate,
    get_recent_partitions,
    load_alerts,
    load_drift_report,
    load_recent_metrics,
    psi_status_label,
)
from config.settings import (
    METRICS_JSONL_PATH,
    MONITORING_DIR,
    DRIFT_REFERENCE_PATH,
    SERVING_URL,
)

# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="모니터링 대시보드", layout="wide")
st.title("📊 모니터링 대시보드")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 실시간 예측 현황",
    "🌊 드리프트 감지",
    "⚠️ 알림 이력",
    "🔧 시스템 상태",
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: 실시간 예측 현황
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.subheader("실시간 예측 현황 (최근 24시간)")

    records = load_recent_metrics(METRICS_JSONL_PATH, hours=24)

    if not records:
        st.info("아직 예측 데이터가 없습니다. API를 통해 예측을 실행하면 여기에 표시됩니다.")
    else:
        df = pd.DataFrame(records)

        col1, col2, col3, col4 = st.columns(4)
        total = len(df)
        disagree_rate = compute_disagree_rate(records)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            n_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
            period_label = f"지난 {n_days}일" if n_days < 30 else "최근 30일"
        else:
            period_label = "최근 24시간"

        col1.metric("총 예측 건수", f"{total:,}", help=period_label)
        col2.metric("Rule/ML 불일치율", f"{disagree_rate:.1%}", help="rule_level ≠ ml_level 비율")
        if "risk_level" in df.columns:
            red_count = (df["risk_level"] == "RED").sum()
            col3.metric("고위험(RED) 비율", f"{red_count/total:.1%}" if total else "0.0%")
        if "latency_ms" in df.columns:
            col4.metric("평균 응답시간", f"{df['latency_ms'].mean():.1f}ms")

        if "risk_level" in df.columns:
            st.subheader("위험도 분포")
            dist = df["risk_level"].value_counts().reset_index()
            dist.columns = ["risk_level", "count"]
            st.bar_chart(dist.set_index("risk_level"))

        if "timestamp" in df.columns:
            st.subheader("시간대별 예측 추이 (1시간 집계)")
            df_hourly = df.set_index("timestamp").resample("1h").size().reset_index()
            df_hourly.columns = ["timestamp", "count"]
            st.line_chart(df_hourly.set_index("timestamp"))

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: 드리프트 감지
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("드리프트 감지 (PSI)")

    partitions = get_recent_partitions(MONITORING_DIR, prefix="drift_", n=7)
    if not partitions:
        st.info("아직 드리프트 데이터가 없습니다. 배치 DAG를 실행하면 여기에 표시됩니다.")
    else:
        selected_partition = st.selectbox("파티션 선택", partitions, index=0)
        report = load_drift_report(MONITORING_DIR, selected_partition)
        if report is None:
            st.info(f"파티션 {selected_partition}의 드리프트 데이터가 없습니다.")
        else:
            summary = report.get("summary", {})
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("분석 피처 수", summary.get("total_features", "N/A"))
            col2.metric("🟢 Stable", summary.get("stable", 0))
            col3.metric("🟡 Warning", summary.get("warning", 0))
            col4.metric("🔴 Drift", summary.get("drift", 0))

            if report.get("trigger_retrain"):
                st.error("⚡ 긴급 재학습 트리거 — 드리프트 피처 2개 이상 감지")

            features = report.get("features", [])
            if features:
                feat_df = pd.DataFrame(features)
                feat_df["상태"] = feat_df["psi"].apply(psi_status_label)
                feat_df = feat_df.rename(columns={"feature": "피처", "psi": "PSI", "status": "status"})
                st.dataframe(
                    feat_df[["피처", "PSI", "상태"]].sort_values("PSI", ascending=False),
                    use_container_width=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: 알림 이력
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("알림 이력 (최근 7일)")

    alert_partitions = get_recent_partitions(MONITORING_DIR, prefix="alerts_", n=7)
    alerts = load_alerts(MONITORING_DIR, alert_partitions)

    if not alerts:
        st.success("✅ 정상 — 최근 7일 내 발생한 알림이 없습니다.")
    else:
        alert_df = pd.DataFrame(alerts)
        severity_icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}
        if "severity" in alert_df.columns:
            alert_df["severity"] = alert_df["severity"].apply(
                lambda s: f"{severity_icon.get(s, '')} {s}"
            )
        display_cols = [c for c in ("generated_at", "alert_type", "severity", "message") if c in alert_df.columns]
        st.dataframe(
            alert_df[display_cols].sort_values("generated_at", ascending=False)
            if "generated_at" in alert_df.columns
            else alert_df[display_cols],
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: 시스템 상태
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.subheader("시스템 상태")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Serving API**")
        try:
            import requests
            resp = requests.get(f"{SERVING_URL}/health", timeout=3)
            if resp.status_code == 200:
                st.success(f"✅ 정상 ({SERVING_URL})")
                health_data = resp.json()
                st.json(health_data)
            else:
                st.error(f"❌ 응답 오류 (HTTP {resp.status_code})")
        except Exception as e:
            st.warning(f"⚠️ 연결 실패: {e}")

    with col2:
        st.markdown("**모니터링 파일 상태**")
        if METRICS_JSONL_PATH.exists():
            stat = METRICS_JSONL_PATH.stat()
            st.write(f"📄 metrics_live.jsonl: {stat.st_size / 1024:.1f} KB")
            mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
            st.write(f"   마지막 수정: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning("metrics_live.jsonl 없음")

        if DRIFT_REFERENCE_PATH.exists():
            stat = DRIFT_REFERENCE_PATH.stat()
            mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
            age_days = (datetime.datetime.now() - mtime).days
            label = f"✅ drift_reference.pkl ({age_days}일 전)"
            if age_days > 180:
                st.warning(f"⚠️ {label} — 180일 이상 경과, 재학습 권고")
            else:
                st.success(label)
        else:
            st.error("❌ drift_reference.pkl 없음 — 학습 파이프라인 실행 필요")
