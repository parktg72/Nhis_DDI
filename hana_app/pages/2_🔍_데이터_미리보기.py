"""
페이지 2: 데이터 미리보기 (HANA DB 또는 SAS 파일)
"""
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hana_app.core.config import load_config, is_hana, is_sas
from hana_app.core.db import get_connection
from hana_app.core.sas_reader import (
    read_sas_chunks,
    get_sas_columns,
    get_sas_row_count,
    scan_sas_files,
)

st.set_page_config(page_title="데이터 미리보기", page_icon="🔍", layout="wide")
st.title("🔍 데이터 미리보기")

cfg  = load_config()
conn = get_connection()

# ── 데이터 소스 상태 확인 ────────────────────────────────────────────────────
using_hana = is_hana(cfg)
using_sas  = is_sas(cfg)

if using_hana and not (st.session_state.get("connected") and conn.is_connected()):
    st.warning("⚠️ HANA DB 미연결. **1단계 설정** 페이지에서 연결하거나 SAS 파일로 전환하세요.")
    st.stop()

if using_sas:
    sas_folder = Path(cfg["sas"].get("folder", ""))
    if not sas_folder.exists():
        st.warning("⚠️ SAS 폴더가 설정되지 않았습니다. **1단계 설정** 페이지에서 폴더를 지정하세요.")
        st.stop()
    src_label = "📂 SAS 파일"
else:
    src_label = "🗄️ HANA DB"

st.info(f"데이터 소스: **{src_label}**")

# ── 테이블 선택 ─────────────────────────────────────────────────────────────
TABLE_LABELS = {
    "t20":    "T20 – 요양급여비용명세서",
    "t30":    "T30 – 진료내역 (원내 약품)",
    "t40":    "T40 – 상병내역",
    "t60":    "T60 – 원외처방전 내역",
    "yoyang": "요양기관 현황",
}

selected_key = st.selectbox(
    "테이블 선택",
    options=list(TABLE_LABELS.keys()),
    format_func=lambda k: TABLE_LABELS[k],
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SAS 모드
# ═════════════════════════════════════════════════════════════════════════════
if using_sas:
    fname = cfg["sas"]["files"].get(selected_key, "")
    if not fname:
        st.warning(f"{TABLE_LABELS[selected_key]} 파일이 지정되지 않았습니다. 1단계에서 파일을 선택하세요.")
        st.stop()

    fpath    = sas_folder / fname
    encoding = cfg["sas"].get("encoding", "cp949")
    chunksize = int(cfg["sas"].get("chunksize", 100_000))

    if not fpath.exists():
        st.error(f"파일을 찾을 수 없습니다: `{fpath}`")
        st.stop()

    st.info(f"**{fpath.name}**  ({fpath.stat().st_size / 1024**2:.1f} MB)")

    # ── 기본 정보 ─────────────────────────────────────────────────────────
    with st.spinner("파일 정보 로딩 중..."):
        sas_cols  = get_sas_columns(fpath, encoding)
        try:
            row_count = get_sas_row_count(fpath, encoding)
        except Exception:
            row_count = -1

    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("컬럼 수", len(sas_cols))
    ic2.metric("전체 행 수 (추정)", f"{row_count:,}" if row_count >= 0 else "집계 필요")
    ic3.metric("인코딩", encoding)

    tab_prev, tab_cols, tab_dist = st.tabs(["📄 샘플 데이터", "📋 컬럼 정보", "📊 분포 분석"])

    with tab_prev:
        limit = st.slider("미리볼 행 수", 10, 500, 50, 10)
        filter_yyyymm = st.text_input(
            "YYYYMM 필터 (선택사항)",
            placeholder="예: 202301  또는 비워두면 전체",
        )

        if st.button("🔄 데이터 조회", type="primary"):
            with st.spinner("SAS 파일 읽는 중..."):
                rows_collected: list[pd.DataFrame] = []
                total_needed = limit
                for chunk in read_sas_chunks(fpath, encoding, chunksize=min(chunksize, 5000)):
                    if filter_yyyymm.strip():
                        yyyymm_col = cfg["columns"].get(selected_key, {}).get("yyyymm", "MDCARE_STRT_YYYYMM")
                        if yyyymm_col in chunk.columns:
                            chunk = chunk[
                                chunk[yyyymm_col].astype(str).str.strip().str[:6] == filter_yyyymm.strip()
                            ]
                    rows_collected.append(chunk)
                    if sum(len(r) for r in rows_collected) >= total_needed:
                        break

                if rows_collected:
                    preview_df = pd.concat(rows_collected, ignore_index=True).head(limit)
                    st.dataframe(preview_df, use_container_width=True, height=400)
                    st.caption(f"{len(preview_df):,}행 표시")
                else:
                    st.info("조건에 맞는 데이터가 없습니다.")

    with tab_cols:
        # 매핑 현황
        mapped_cols = cfg["columns"].get(selected_key, {})
        col_df = pd.DataFrame([
            {
                "컬럼명": c,
                "역할": next((k for k, v in mapped_cols.items() if v == c), ""),
                "매핑됨": "✅" if c in mapped_cols.values() else "",
            }
            for c in sas_cols
        ])
        st.dataframe(col_df, use_container_width=True, height=500)
        st.caption(f"총 {len(sas_cols)}개 컬럼 | 매핑됨: {col_df['매핑됨'].eq('✅').sum()}개")

    with tab_dist:
        sel_cols = st.multiselect(
            "분석할 컬럼 선택 (최대 5개)",
            options=sas_cols,
            max_selections=5,
            default=sas_cols[:3] if len(sas_cols) >= 3 else sas_cols,
        )
        sample_n = st.number_input("샘플 행 수", 500, 50_000, 5_000, 500)

        if st.button("📊 분포 분석 실행") and sel_cols:
            with st.spinner("샘플 읽는 중..."):
                parts: list[pd.DataFrame] = []
                for chunk in read_sas_chunks(fpath, encoding, usecols=sel_cols, chunksize=chunksize):
                    parts.append(chunk)
                    if sum(len(p) for p in parts) >= sample_n:
                        break
                df = pd.concat(parts, ignore_index=True).head(sample_n) if parts else pd.DataFrame()

            if df.empty:
                st.warning("데이터를 읽지 못했습니다.")
            else:
                import plotly.express as px
                for col_name in sel_cols:
                    if col_name not in df.columns:
                        continue
                    st.markdown(f"**{col_name}**")
                    col_data = df[col_name]
                    c1, c2 = st.columns(2)
                    c1.metric("NULL 비율", f"{col_data.isna().mean()*100:.1f}%")
                    c2.metric("고유값 수", col_data.nunique())

                    if pd.api.types.is_numeric_dtype(col_data):
                        fig = px.histogram(df, x=col_name, nbins=30, height=200)
                        st.plotly_chart(fig, use_container_width=True)
                    elif col_data.nunique() <= 30:
                        vc = col_data.value_counts().reset_index()
                        vc.columns = [col_name, "건수"]
                        fig = px.bar(vc, x=col_name, y="건수", height=250)
                        st.plotly_chart(fig, use_container_width=True)
                    st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# HANA 모드 (기존 로직)
# ═════════════════════════════════════════════════════════════════════════════
else:
    tbl_cfg = cfg["tables"].get(selected_key, {})
    schema  = tbl_cfg.get("schema", "")
    table   = tbl_cfg.get("table", "")

    if not schema or not table:
        st.error("테이블이 설정되지 않았습니다. 1단계에서 테이블 위치를 설정하세요.")
        st.stop()

    st.info(f"**{schema}.{table}**")

    with st.spinner("테이블 정보 로딩 중..."):
        try:
            row_count = conn.get_row_count(schema, table)
            columns   = conn.get_columns(schema, table)
        except Exception as e:
            st.error(f"테이블 조회 실패: {e}")
            st.stop()

    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("전체 행 수", f"{row_count:,}")
    ic2.metric("컬럼 수",    len(columns))
    ic3.metric("테이블",     f"{schema}.{table}")

    DATE_COLS = {
        "t20": "MDCARE_STRT_YYYYMM", "t30": "MDCARE_STRT_YYYYMM",
        "t40": "MDCARE_STRT_DT",     "t60": "MDCARE_STRT_YYYYMM",
        "yoyang": "STD_YYYY",
    }
    date_col = DATE_COLS.get(selected_key)
    if date_col and any(c["name"] == date_col for c in columns):
        try:
            dr = conn.get_date_range(schema, table, date_col)
            st.caption(f"📅 {date_col}: {dr['min']} ~ {dr['max']}")
        except Exception:
            pass

    st.markdown("---")
    tab_prev, tab_cols, tab_dist = st.tabs(["📄 샘플 데이터", "📋 컬럼 정보", "📊 분포 분석"])

    with tab_prev:
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            limit = st.slider("조회 행 수", 10, 500, 50, 10)
        with col_opt2:
            where_clause = st.text_input("WHERE 조건", placeholder="예: MDCARE_STRT_YYYYMM = '202301'")

        if st.button("🔄 데이터 조회", type="primary"):
            with st.spinner("조회 중..."):
                try:
                    _where = where_clause.strip()
                    if _where:
                        # SQL Injection 방지: DDL/DML 키워드 + 다중 구문/주석 차단
                        _FORBIDDEN = re.compile(
                            r"(;|--)"
                            r"|\b(DROP|DELETE|INSERT|UPDATE|ALTER|TRUNCATE|CREATE|EXEC|MERGE|"
                            r"GRANT|REVOKE|UNION|CALL|UPSERT)\b",
                            re.IGNORECASE,
                        )
                        if _FORBIDDEN.search(_where):
                            st.error("WHERE 조건에 허용되지 않는 키워드가 포함되어 있습니다.")
                        else:
                            sql = f'SELECT * FROM "{schema}"."{table}" WHERE {_where} LIMIT {limit}'
                            df  = conn.query_df(sql)
                            st.dataframe(df, use_container_width=True, height=400)
                            st.caption(f"{len(df):,}행 조회됨")
                    else:
                        df  = conn.preview(schema, table, limit)
                        st.dataframe(df, use_container_width=True, height=400)
                        st.caption(f"{len(df):,}행 조회됨")
                except Exception as e:
                    st.error(f"조회 실패: {e}")

    with tab_cols:
        col_df = pd.DataFrame(columns)
        col_df.columns = ["컬럼명", "데이터타입", "Nullable"]
        st.dataframe(col_df, use_container_width=True, height=500)

        mapped = cfg["columns"].get(selected_key, {})
        if mapped:
            st.subheader("현재 컬럼 매핑")
            st.dataframe(
                pd.DataFrame([(f, c) for f, c in mapped.items()], columns=["역할", "실제 컬럼명"]),
                use_container_width=True,
            )

    with tab_dist:
        col_names = [c["name"] for c in columns]
        sel_cols  = st.multiselect("분석할 컬럼 선택 (최대 5개)", col_names,
                                   max_selections=5, default=col_names[:3])
        sample_n  = st.number_input("샘플 수", 100, 10_000, 1_000, 100)

        if st.button("📊 분포 분석 실행") and sel_cols:
            with st.spinner("분석 중..."):
                import plotly.express as px
                try:
                    cols_q = ", ".join(f'"{c}"' for c in sel_cols)
                    df     = conn.query_df(f'SELECT {cols_q} FROM "{schema}"."{table}" LIMIT {sample_n}')
                    for col_name in sel_cols:
                        st.markdown(f"**{col_name}**")
                        col_data = df[col_name]
                        c1, c2 = st.columns(2)
                        c1.metric("NULL 비율", f"{col_data.isna().mean()*100:.1f}%")
                        c2.metric("고유값 수", col_data.nunique())
                        if pd.api.types.is_numeric_dtype(col_data):
                            fig = px.histogram(df, x=col_name, nbins=30, height=200)
                            st.plotly_chart(fig, use_container_width=True)
                        elif col_data.nunique() <= 30:
                            vc  = col_data.value_counts().reset_index()
                            vc.columns = [col_name, "건수"]
                            fig = px.bar(vc, x=col_name, y="건수", height=250)
                            st.plotly_chart(fig, use_container_width=True)
                        st.markdown("---")
                except Exception as e:
                    st.error(f"분석 실패: {e}")
