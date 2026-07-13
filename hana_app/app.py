"""
NHIS 다재약물 DDI 위험도 분류 시스템
SAP HANA 연동 웹 애플리케이션
"""
import sys
from pathlib import Path

import streamlit as st

# 프로젝트 루트 경로 설정
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="NHIS 다재약물 위험도 분류",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 전역 session_state 초기화 ────────────────────────────────────────────────
if "connected" not in st.session_state:
    st.session_state.connected = False
if "records" not in st.session_state:
    st.session_state.records = None
if "features_df" not in st.session_state:
    st.session_state.features_df = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
# 페이지 간 공유 키 (page1 설정 → page3/6 사용)
if "hana_creds" not in st.session_state:
    st.session_state.hana_creds = None
if "conn_host" not in st.session_state:
    st.session_state.conn_host = ""
if "sas_ready" not in st.session_state:
    st.session_state.sas_ready = False

# ── 홈 화면 ──────────────────────────────────────────────────────────────────
st.title("💊 NHIS 다재약물 DDI 위험도 분류 시스템")
st.caption("SAP HANA 연동 머신러닝 학습 플랫폼")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 🔌 1단계")
    st.markdown("**HANA DB 연결**")
    st.caption("연결 정보 입력 및 T20/T30/T40/T60 테이블 위치 지정")
    conn_ok = st.session_state.get("connected", False)
    if conn_ok:
        st.success("연결됨")
    else:
        st.warning("미연결")

with col2:
    st.markdown("### 🔍 2단계")
    st.markdown("**데이터 미리보기**")
    st.caption("테이블 통계 및 샘플 데이터 확인")

with col3:
    st.markdown("### 🤖 3단계")
    st.markdown("**모델 학습**")
    st.caption("알고리즘 선택 → 하이퍼파라미터 → 학습 실행")
    if st.session_state.get("last_result"):
        m = st.session_state.last_result.get("metrics", {})
        st.success(f"F1={m.get('f1_macro', 0):.3f}")

with col4:
    st.markdown("### 📊 4단계")
    st.markdown("**결과 분석**")
    st.caption("피처 중요도 · ROC · 혼동행렬 · 위험도 분포")

st.markdown("---")

st.markdown("""
### 시스템 개요

| 구성 | 내용 |
|------|------|
| **데이터 소스** | 건강보험공단 NHIS 청구 데이터 (SAP HANA) |
| **대상** | 90일 이내 5종 이상 약물 처방 환자 (다재약물) |
| **위험도** | 🔴 Red (즉시개입) · 🟡 Yellow (주의) · 🟢 Green (관찰) · ⚪ Normal |
| **DDI 탐지** | DrugBank ATC 기반 + NHIS 주성분코드 기반 |
| **ML 모델** | XGBoost · LightGBM · Random Forest · Logistic Regression |

### 좌측 메뉴에서 단계를 선택하세요
""")

# ── 사이드바 상태 표시 ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📌 현재 상태")

    if st.session_state.connected:
        host = st.session_state.get("conn_host", "")
        st.success(f"✅ HANA 연결: {host}")
    else:
        st.error("❌ HANA 미연결")

    records = st.session_state.get("records")
    if records is not None:
        st.info(f"📦 추출 데이터: {len(records):,}건")

    df = st.session_state.get("features_df")
    if df is not None:
        st.info(f"📊 피처 벡터: {len(df):,}명")

    result = st.session_state.get("last_result")
    if result:
        m = result.get("metrics", {})
        st.success(
            f"🤖 {result.get('model_name','?')} 학습 완료\n"
            f"F1={m.get('f1_macro', 0):.3f} | "
            f"AUC={m.get('roc_auc', m.get('roc_auc_ovr', 0)):.3f}"
        )
