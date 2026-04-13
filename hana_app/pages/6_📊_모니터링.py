"""
모니터링 대시보드 — Streamlit 6번 페이지

구성:
  상태 요약 바: HANA 연결 / ETL 이력 / 모델 상태 / 저장소 — 항상 표시
  Tab 1: 🔌 HANA 연결 상태
  Tab 2: 📋 ETL 실행 이력
  Tab 3: 🤖 모델 학습 이력
  Tab 4: 💾 시스템 상태
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from hana_app.core.config import load_config, is_hana
from hana_app.core.db import get_connection
from hana_app.core.etl_logger import load_etl_log
from hana_app.core.ml_runner import list_saved_results, RESULTS_DIR, MODELS_DIR

st.set_page_config(page_title="모니터링 대시보드", layout="wide")
st.title("📊 모니터링 대시보드")

# ─────────────────────────────────────────────────────────────────────────────
# 상태 계산 (탭과 독립적으로 항상 수행)
# ─────────────────────────────────────────────────────────────────────────────
cfg = load_config()

# HANA 연결 상태
_hana_mode = is_hana(cfg)
if _hana_mode:
    _conn = get_connection(st.session_state)
    _hana_connected = _conn.is_connected()
    _hana_validated = cfg.get("validated", False)
else:
    _hana_connected = None   # SAS 모드 — 해당 없음
    _hana_validated = True

# ETL 이력
_etl_records = load_etl_log(n=1)
_etl_ok = len(_etl_records) > 0

# 모델 이력
_saved_results = list_saved_results()
_model_ok = len(_saved_results) > 0

# 저장소 상태
_storage_ok = RESULTS_DIR.exists() and MODELS_DIR.exists()

# ─────────────────────────────────────────────────────────────────────────────
# 상태 요약 바
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 시스템 상태 요약")
sb1, sb2, sb3, sb4 = st.columns(4)

if _hana_mode:
    if _hana_connected and _hana_validated:
        sb1.success("🟢 HANA 연결됨")
    elif _hana_connected and not _hana_validated:
        sb1.warning("🟡 연결됨 (미검증)")
    else:
        sb1.error("🔴 HANA 연결 끊김")
else:
    sb1.info("⚪ SAS 모드")

if _etl_ok:
    _last_etl = _etl_records[0]
    sb2.success(f"🟢 ETL 완료 ({_last_etl['ts'][:10]})")
else:
    sb2.warning("🟡 ETL 이력 없음")

if _model_ok:
    _latest = _saved_results[0]
    sb3.success(f"🟢 모델 {len(_saved_results)}개 ({_latest.get('timestamp','?')[:8]})")
else:
    sb3.error("🔴 모델 없음")

if _storage_ok:
    sb4.success("🟢 저장소 정상")
else:
    sb4.error("🔴 저장소 경로 없음")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔌 HANA 연결 상태",
    "📋 ETL 실행 이력",
    "🤖 모델 학습 이력",
    "💾 시스템 상태",
])

# ─── Tab 1: HANA 연결 상태 ───────────────────────────────────────────────────
with tab1:
    if not _hana_mode:
        st.info("SAS 파일 모드에서는 HANA 연결 상태가 필요하지 않습니다.")
    else:
        conn_cfg = cfg.get("connection", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("호스트", conn_cfg.get("host", "—") or "—")
        c2.metric("포트", str(conn_cfg.get("port", "—")))
        c3.metric("사용자", conn_cfg.get("user", "—") or "—")

        st.markdown("#### 검증 상태")
        v1, v2, v3 = st.columns(3)
        v1.metric("검증 완료", "✅ 예" if cfg.get("validated") else "❌ 아니오")
        v2.metric("검증 시각", cfg.get("validated_at", "—") or "—")
        v3.metric("검증 호스트", cfg.get("validated_host", "—") or "—")

        st.markdown("#### 실시간 연결 확인")
        if st.button("🔄 연결 상태 확인", key="btn_check_conn"):
            with st.spinner("연결 확인 중..."):
                alive = _conn.is_connected()
            if alive:
                st.success("✅ HANA DB 연결 정상")
            else:
                st.error("❌ 연결 끊김")

        hana_creds = st.session_state.get("hana_creds")
        if hana_creds and st.button("🔌 재연결", key="btn_reconnect"):
            with st.spinner("재연결 시도 중..."):
                try:
                    _conn.ensure_connected(hana_creds, session_state=st.session_state)
                    st.success("✅ 재연결 성공")
                except Exception as e:
                    st.error(f"❌ 재연결 실패: {e}")
        elif not hana_creds:
            st.caption("재연결하려면 1번 페이지에서 먼저 연결하세요.")

# ─── Tab 2: ETL 실행 이력 ────────────────────────────────────────────────────
with tab2:
    etl_records = load_etl_log(n=50)
    if not etl_records:
        st.info(
            "ETL 실행 이력이 없습니다.\n\n"
            "3단계 모델 학습 탭에서 ETL을 실행하면 이력이 자동으로 기록됩니다.\n"
            "이력은 앱을 재시작해도 유지됩니다."
        )
    else:
        st.caption(f"총 {len(etl_records)}건 (최근 50건 표시, 최신순)")
        etl_df = pd.DataFrame(etl_records)
        etl_df = etl_df.rename(columns={
            "ts": "실행 시각", "period_from": "시작 기간", "period_to": "종료 기간",
            "row_count": "추출 건수", "elapsed_sec": "소요(초)", "status": "상태", "error": "오류",
        })
        etl_df["추출 건수"] = etl_df["추출 건수"].apply(lambda x: f"{x:,}")
        st.dataframe(etl_df, use_container_width=True, hide_index=True)

# ─── Tab 3: 모델 학습 이력 ────────────────────────────────────────────────────
with tab3:
    if not _saved_results:
        st.info("저장된 모델 결과가 없습니다. 3단계 모델 학습을 먼저 실행하세요.")
    else:
        rows = []
        for r in _saved_results:
            m = r.get("metrics", {})
            rows.append({
                "시각": r.get("timestamp", "?"),
                "모델": r.get("model_name", "?"),
                "타겟": r.get("target", "?"),
                "Accuracy": round(m.get("accuracy", 0), 4),
                "F1": round(m.get("f1_macro", 0), 4),
                "AUC": round(m.get("roc_auc", m.get("roc_auc_ovr", 0)), 4),
                "학습 수": m.get("train_size", 0),
                "_file": r.get("_file", ""),
            })
        hist_df = pd.DataFrame(rows)

        # 성능 추이 차트
        if len(hist_df) > 1:
            fig = go.Figure()
            for metric in ["Accuracy", "F1", "AUC"]:
                fig.add_trace(go.Scatter(
                    x=hist_df["시각"], y=hist_df[metric],
                    mode="lines+markers", name=metric,
                ))
            fig.update_layout(
                title="모델 성능 추이",
                xaxis_title="학습 시각", yaxis_title="Score",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        # 결과 테이블 (최신 강조)
        display_df = hist_df.drop(columns=["_file"])
        st.dataframe(
            display_df.style.apply(
                lambda row: ["background-color: #e8f5e9" if row.name == 0 else "" for _ in row],
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

        # 삭제
        st.markdown("#### 결과 삭제")
        del_options = {
            f"{r['시각']} — {r['모델']}": r["_file"]
            for r in rows if r["_file"]
        }
        if del_options:
            del_label = st.selectbox("삭제할 결과 선택", list(del_options.keys()), key="del_result_sel")
            if st.button("🗑️ 선택 결과 삭제", key="btn_del_result"):
                del_path = Path(del_options[del_label])
                if del_path.exists():
                    del_path.unlink()
                    st.success(f"삭제 완료: {del_path.name}")
                    st.rerun()
                else:
                    st.error("파일을 찾을 수 없습니다.")

# ─── Tab 4: 시스템 상태 ──────────────────────────────────────────────────────
with tab4:
    st.markdown("#### 저장소 현황")
    s1, s2 = st.columns(2)

    def _dir_info(d: Path) -> tuple[int, float]:
        """(파일 수, 총 MB)"""
        if not d.exists():
            return 0, 0.0
        files = list(d.iterdir())
        total = sum(f.stat().st_size for f in files if f.is_file())
        return len(files), total / (1024 * 1024)

    with s1:
        st.markdown(f"**📁 results/** `{RESULTS_DIR}`")
        n_r, mb_r = _dir_info(RESULTS_DIR)
        st.write(f"파일 {n_r}개 / {mb_r:.1f} MB")
        if n_r:
            r_files = sorted(RESULTS_DIR.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
            r_df = pd.DataFrame([
                {"파일명": f.name, "크기(KB)": round(f.stat().st_size / 1024, 1)}
                for f in r_files if f.is_file()
            ])
            st.dataframe(r_df, use_container_width=True, hide_index=True)

    with s2:
        st.markdown(f"**📁 models/** `{MODELS_DIR}`")
        n_m, mb_m = _dir_info(MODELS_DIR)
        st.write(f"파일 {n_m}개 / {mb_m:.1f} MB")
        if n_m:
            m_files = sorted(MODELS_DIR.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
            m_df = pd.DataFrame([
                {"파일명": f.name, "크기(MB)": round(f.stat().st_size / (1024 * 1024), 1)}
                for f in m_files if f.is_file()
            ])
            st.dataframe(m_df, use_container_width=True, hide_index=True)

    total_mb = mb_r + mb_m
    st.metric("총 디스크 사용량", f"{total_mb:.1f} MB")

    st.markdown("#### 설정 파일")
    from hana_app.core.config import CONFIG_FILE
    if CONFIG_FILE.exists():
        st.success(f"✅ {CONFIG_FILE.name} 존재 ({CONFIG_FILE.stat().st_size / 1024:.1f} KB)")
    else:
        st.warning(f"⚠️ {CONFIG_FILE.name} 없음 — 1번 페이지에서 설정 후 저장하세요.")
