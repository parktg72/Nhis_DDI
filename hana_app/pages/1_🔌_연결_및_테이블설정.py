"""
페이지 1: 데이터 소스 선택 + HANA DB 연결 또는 SAS 파일 설정
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hana_app.core.config import (
    DATA_SOURCE_HANA, DATA_SOURCE_SAS,
    DEFAULT_TABLE_COLS,
    get_password, is_hana, is_sas,
    load_config, save_config, set_password,
)
from hana_app.core.db import get_connection
from hana_app.core.sas_reader import scan_sas_files, guess_table_type, get_sas_columns

st.set_page_config(page_title="데이터 소스 설정", page_icon="🔌", layout="wide")
st.title("🔌 데이터 소스 설정")

cfg  = load_config()
conn = get_connection()


# ═════════════════════════════════════════════════════════════════════════════
# 최상단: 데이터 소스 선택
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("데이터 소스 선택")

source_choice = st.radio(
    "학습 데이터를 어디서 가져올까요?",
    options=[DATA_SOURCE_HANA, DATA_SOURCE_SAS],
    format_func=lambda x: (
        "🗄️  SAP HANA DB  (건보 내부망 직접 접속)"
        if x == DATA_SOURCE_HANA
        else "📂  SAS 파일  (로컬 폴더의 .sas7bdat / .xpt 파일)"
    ),
    index=0 if is_hana(cfg) else 1,
    horizontal=True,
)
cfg["data_source"] = source_choice

if source_choice == DATA_SOURCE_HANA:
    st.info("⚙️  HANA DB 탭에서 접속 정보를 입력하고 연결 테스트를 실행하세요.")
else:
    st.info("📂  SAS 파일 탭에서 데이터 폴더와 각 테이블 파일을 지정하세요.")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# 탭 구성: HANA 연결 / SAS 파일 / 테이블 위치 / 컬럼 매핑
# ═════════════════════════════════════════════════════════════════════════════
tab_hana, tab_sas, tab_tbl, tab_col = st.tabs([
    "🗄️ HANA DB 연결",
    "📂 SAS 파일 설정",
    "📋 테이블 위치 (HANA)",
    "🗂️ 컬럼 매핑",
])


# ─────────────────────────────────────────────────────────────────────────────
# 탭 1: HANA DB 연결
# ─────────────────────────────────────────────────────────────────────────────
with tab_hana:
    st.subheader("HANA DB 접속 정보")
    st.caption("입력한 정보는 hana_config.json에 저장됩니다 (패스워드 난독화).")

    col1, col2 = st.columns(2)
    with col1:
        host = st.text_input(
            "호스트 (IP 또는 도메인)",
            value=cfg["connection"]["host"],
            placeholder="예: 192.168.1.100",
        )
        port = st.number_input(
            "포트 번호",
            value=int(cfg["connection"]["port"]),
            min_value=1, max_value=65535, step=1,
            help="HANA 기본 포트: 30015 (instance 00)",
        )
    with col2:
        user = st.text_input(
            "사용자 ID",
            value=cfg["connection"]["user"],
            placeholder="예: NHIS_USER",
        )
        password = st.text_input(
            "패스워드",
            type="password",
            value=get_password(cfg),
        )

    c_save, c_test, c_disc = st.columns(3)

    with c_save:
        if st.button("💾 설정 저장", use_container_width=True):
            cfg["connection"].update({"host": host, "port": int(port), "user": user})
            set_password(cfg, password)
            save_config(cfg)
            st.success("저장 완료.")

    with c_test:
        if st.button("🔗 연결 테스트", type="primary", use_container_width=True):
            if not (host and user and password):
                st.error("호스트 / 사용자 ID / 패스워드를 모두 입력하세요.")
            else:
                with st.spinner("연결 시도 중..."):
                    try:
                        conn.connect(host, int(port), user, password)
                        if conn.is_connected():
                            info = conn.server_info()
                            st.success("✅ 연결 성공!")
                            st.session_state.connected = True
                            st.session_state.conn_host = host
                            if info:
                                st.json({k: v for k, v in info.items()
                                         if k in ("SYSTEM_ID", "DATABASE_NAME", "VERSION")})
                            cfg["connection"].update({"host": host, "port": int(port), "user": user})
                            set_password(cfg, password)
                            save_config(cfg)
                        else:
                            st.error("연결 실패: 서버 응답 없음")
                            st.session_state.connected = False
                    except Exception as e:
                        st.error(f"연결 오류: {e}")
                        st.session_state.connected = False

    with c_disc:
        if st.button("🔌 연결 해제", use_container_width=True):
            conn.close()
            st.session_state.connected = False
            st.info("연결 해제.")

    st.markdown("---")
    if st.session_state.get("connected") and conn.is_connected():
        st.success(f"✅ 현재 연결됨: {cfg['connection']['host']}:{cfg['connection']['port']}")
    else:
        st.warning("⚠️ 미연결 상태입니다.")


# ─────────────────────────────────────────────────────────────────────────────
# 탭 2: SAS 파일 설정
# ─────────────────────────────────────────────────────────────────────────────
with tab_sas:
    st.subheader("SAS 데이터 파일 설정")
    st.caption(
        "HANA DB 접속이 불가한 경우, 사전에 반출된 SAS 파일(.sas7bdat / .xpt)을 "
        "직접 지정하여 학습합니다."
    )

    sas_cfg = cfg.setdefault("sas", {
        "folder": "", "encoding": "cp949", "chunksize": 100000,
        "files": {"t20": "", "t30": "", "t40": "", "t60": "", "yoyang": ""},
    })

    # ── 폴더 경로 ─────────────────────────────────────────────────────────
    col_folder, col_enc = st.columns([3, 1])
    with col_folder:
        folder_input = st.text_input(
            "📁 SAS 데이터 폴더 경로",
            value=sas_cfg.get("folder", ""),
            placeholder=r"예: C:\NHIS_DATA\2023 또는 D:\SAS_FILES",
        )
    with col_enc:
        enc_opts = ["cp949", "euc-kr", "utf-8", "latin-1"]
        enc_idx  = enc_opts.index(sas_cfg.get("encoding", "cp949")) \
                   if sas_cfg.get("encoding", "cp949") in enc_opts else 0
        encoding = st.selectbox(
            "파일 인코딩",
            options=enc_opts,
            index=enc_idx,
            help="NHIS 데이터는 보통 cp949 (EUC-KR 확장) 인코딩입니다.",
        )

    chunksize = st.number_input(
        "청크 크기 (행 수, 메모리 관리)",
        value=int(sas_cfg.get("chunksize", 100_000)),
        min_value=10_000, max_value=1_000_000, step=10_000,
        help="파일을 이 행 수 단위로 나눠 읽습니다. RAM이 부족하면 줄이세요.",
    )

    # ── 폴더 스캔 ─────────────────────────────────────────────────────────
    sas_files_in_folder: list[Path] = []
    if folder_input and Path(folder_input).exists():
        sas_files_in_folder = scan_sas_files(folder_input)

    if folder_input:
        folder_path = Path(folder_input)
        if not folder_path.exists():
            st.error(f"폴더가 존재하지 않습니다: `{folder_input}`")
        elif not sas_files_in_folder:
            st.warning("폴더에 SAS 파일(.sas7bdat / .xpt)이 없습니다.")
        else:
            st.success(f"✅ SAS 파일 {len(sas_files_in_folder)}개 발견")
            with st.expander("발견된 파일 목록"):
                for f in sas_files_in_folder:
                    guessed = guess_table_type(f.name)
                    label   = f"  → {guessed.upper()}" if guessed else ""
                    st.text(f"  {f.name}  ({f.stat().st_size / 1024**2:.1f} MB){label}")

    # ── 테이블별 파일 지정 ────────────────────────────────────────────────
    st.markdown("#### 테이블별 SAS 파일 지정")
    st.caption("각 테이블에 해당하는 파일을 선택하세요. 선택하지 않은 테이블은 건너뜁니다.")

    TABLE_META = {
        "t20":    ("T20",    "요양급여비용명세서 (진료명세서)"),
        "t30":    ("T30",    "진료내역 – 원내 약품"),
        "t40":    ("T40",    "상병내역"),
        "t60":    ("T60",    "원외처방전 내역"),
        "yoyang": ("요양기관", "요양기관 현황"),
    }

    file_opts = ["(선택 안함)"] + [f.name for f in sas_files_in_folder]
    sas_files_map: dict[str, str] = sas_cfg.get("files", {})

    for key, (label, desc) in TABLE_META.items():
        col_label, col_sel, col_btn = st.columns([1, 3, 1])
        with col_label:
            st.markdown(f"**{label}**")
            st.caption(desc)

        cur_fname = sas_files_map.get(key, "")

        with col_sel:
            if sas_files_in_folder:
                # 드롭다운 선택
                default_idx = 0
                if cur_fname in file_opts:
                    default_idx = file_opts.index(cur_fname)
                elif cur_fname:
                    # 이전에 설정된 파일이 현재 폴더에 없는 경우
                    file_opts_ext = [cur_fname + " (파일 없음)"] + file_opts
                    st.selectbox(f"파일 ({label})", file_opts_ext, key=f"sas_{key}_ext",
                                 disabled=True)
                    default_idx = 0

                sel = st.selectbox(
                    f"파일 ({label})",
                    options=file_opts,
                    index=default_idx,
                    key=f"sas_{key}",
                    label_visibility="collapsed",
                )
                chosen = "" if sel == "(선택 안함)" else sel
            else:
                # 직접 입력
                chosen = st.text_input(
                    f"파일명 ({label})",
                    value=cur_fname,
                    placeholder=f"예: T{label}_2023.sas7bdat",
                    key=f"sas_{key}",
                    label_visibility="collapsed",
                )
            sas_files_map[key] = chosen

        with col_btn:
            st.write("")
            if chosen and folder_input:
                fpath = Path(folder_input) / chosen
                if fpath.exists():
                    if st.button("👁️", key=f"sas_prev_{key}", help="컬럼 미리보기"):
                        with st.spinner("컬럼 조회 중..."):
                            cols = get_sas_columns(fpath, encoding)
                        if cols:
                            st.success(f"{len(cols)}개 컬럼")
                            with st.expander(f"{label} 컬럼 목록"):
                                st.write(cols)
                        else:
                            st.error("컬럼 조회 실패")
                else:
                    st.warning("파일 없음")

    # ── 저장 ──────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("💾 SAS 설정 저장", type="primary"):
        sas_cfg.update({
            "folder":    folder_input,
            "encoding":  encoding,
            "chunksize": int(chunksize),
            "files":     sas_files_map,
        })
        cfg["sas"] = sas_cfg
        save_config(cfg)
        # 세션에 SAS 준비 상태 기록
        ready = (
            bool(folder_input)
            and Path(folder_input).exists()
            and any(v for v in sas_files_map.values())
        )
        st.session_state.sas_ready = ready
        if ready:
            st.success("✅ SAS 설정 저장 완료. 3단계에서 학습을 진행하세요.")
        else:
            st.warning("설정은 저장됐지만 폴더 또는 파일이 지정되지 않았습니다.")

    # 현재 상태 요약
    st.markdown("---")
    st.markdown("##### 현재 SAS 설정 요약")
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    sum_col1.metric("데이터 폴더", sas_cfg.get("folder", "미설정") or "미설정")
    sum_col2.metric("인코딩",      sas_cfg.get("encoding", "cp949"))
    mapped = sum(1 for v in sas_files_map.values() if v)
    sum_col3.metric("파일 지정", f"{mapped} / {len(TABLE_META)}개 테이블")


# ─────────────────────────────────────────────────────────────────────────────
# 탭 3: 테이블 위치 (HANA 전용)
# ─────────────────────────────────────────────────────────────────────────────
with tab_tbl:
    st.subheader("테이블 위치 설정 (HANA DB)")
    if not is_hana(cfg):
        st.info("현재 데이터 소스가 **SAS 파일**로 설정되어 있습니다. "
                "HANA DB를 선택한 경우에만 이 탭을 사용합니다.")

    table_keys = {
        "t20":    ("T20",    "요양급여비용명세서", "NHISBDA",  "HHDT_TEMSBJ20"),
        "t30":    ("T30",    "진료내역 (원내)",    "NHISBDA",  "HHDT_TEMSBJ30"),
        "t40":    ("T40",    "상병내역",           "NHISBDA",  "HHDT_TEMSBJ40"),
        "t60":    ("T60",    "원외처방전 내역",    "NHISBDA",  "HHDT_TEMSBJ60"),
        "yoyang": ("요양기관","요양기관 현황",     "NHISBASE", "HHDT_MDCIN_GNRL_INFO"),
    }

    schema_options: list[str] = []
    if st.session_state.get("connected") and conn.is_connected():
        try:
            schema_options = conn.get_schemas("NHIS")
        except Exception:
            pass

    for key, (label, desc, def_schema, def_table) in table_keys.items():
        st.markdown(f"#### {label} – {desc}")
        col_s, col_t, col_btn = st.columns([2, 3, 1])

        cur_schema = cfg["tables"].get(key, {}).get("schema", def_schema)
        cur_table  = cfg["tables"].get(key, {}).get("table",  def_table)

        with col_s:
            if schema_options:
                idx = schema_options.index(cur_schema) if cur_schema in schema_options else 0
                schema = st.selectbox(f"스키마 ({label})", schema_options, index=idx, key=f"schema_{key}")
            else:
                schema = st.text_input(f"스키마 ({label})", value=cur_schema, key=f"schema_{key}")

        with col_t:
            table_options: list[str] = []
            if schema_options and conn.is_connected():
                try:
                    table_options = conn.get_tables(schema)
                except Exception:
                    pass
            if table_options:
                t_idx = table_options.index(cur_table) if cur_table in table_options else 0
                table = st.selectbox(f"테이블 ({label})", table_options, index=t_idx, key=f"table_{key}")
            else:
                table = st.text_input(f"테이블 ({label})", value=cur_table, key=f"table_{key}")

        with col_btn:
            st.write(""); st.write("")
            if conn.is_connected() and st.button("👁️", key=f"check_{key}"):
                try:
                    cnt = conn.get_row_count(schema, table)
                    st.success(f"{cnt:,}행")
                except Exception as e:
                    st.error(str(e))

        cfg["tables"][key] = {"schema": schema, "table": table}
        st.markdown("---")

    if st.button("💾 테이블 설정 저장", type="primary"):
        save_config(cfg)
        st.success("테이블 위치가 저장되었습니다.")


# ─────────────────────────────────────────────────────────────────────────────
# 탭 4: 컬럼 매핑 (HANA / SAS 공통)
# ─────────────────────────────────────────────────────────────────────────────
with tab_col:
    st.subheader("컬럼 매핑 설정")
    st.caption(
        "기본값은 표준 NHIS 컬럼명입니다. "
        "HANA DB와 SAS 파일 모두 동일한 매핑을 사용합니다. "
        "컬럼명이 다른 경우에만 수정하세요."
    )

    col_labels = {
        "t20": {
            "patient_id":      "환자 ID (INDI_DSCM_NO)",
            "bill_no":         "명세서 키 (CMN_KEY)",
            "institution_id":  "요양기관기호 (MDCARE_SYM)",
            "start_date":      "요양개시일자 (MDCARE_STRT_DT)",
            "yyyymm":          "요양개시년월 (MDCARE_STRT_YYYYMM) ★ 월 필터",
            "sex":             "성별구분 (SEX_TYPE)",
            "age_id":          "수진연령ID (SUJIN_POTM_AGE_ID)",
            "institution_type":"기관종별코드 (YOYANG_CLSFC_CD)",
            "prsc_drug_count": "원외처방약품수 (INSOUT_PRSC_MEDI_ITM_SU)",
            "total_prsc_days": "총처방일수 (TOT_PRSC_DD_CNT)",
        },
        "t30": {
            "patient_id":    "환자 ID (INDI_DSCM_NO)",
            "bill_no":       "명세서 키 (CMN_KEY)",
            "start_date":    "요양개시일자 (MDCARE_STRT_DT)",
            "yyyymm":        "요양개시년월 (MDCARE_STRT_YYYYMM) ★ 월 필터",
            "drug_code":     "주성분코드 (WK_COMPN_CD) ★",
            "drug_code_alt": "보정주성분코드 (RVSN_WK_COMPN_CD)",
            "edi_code":      "EDI코드 (MCARE_DIV_CD)",
            "efmdc":         "약효분류번호 (EFMDC_CLSF_NO)",
            "dose_once":     "1회투여량 (TIME1_MDCT_CPCT)",
            "dose_freq":     "1일투여횟수 (DD1_MQTY_FREQ)",
            "total_days":    "총투여일수 (TOT_MCNT) ★",
        },
        "t40": {
            "patient_id": "환자 ID (INDI_DSCM_NO)",
            "bill_no":    "명세서 키 (CMN_KEY)",
            "start_date": "요양개시일자 (MDCARE_STRT_DT)",
            "yyyymm":     "요양개시년월 (MDCARE_STRT_YYYYMM) ★ 월 필터",
            "sick_code":  "상병기호 (MCEX_SICK_SYM) ★",
            "sick_type":  "상병분류구분 (SICK_CLSF_TYPE)",
        },
        "t60": {
            "patient_id":    "환자 ID (INDI_DSCM_NO)",
            "bill_no":       "명세서 키 (CMN_KEY)",
            "start_date":    "요양개시일자 (MDCARE_STRT_DT)",
            "yyyymm":        "요양개시년월 (MDCARE_STRT_YYYYMM) ★ 월 필터",
            "drug_code":     "일반명코드 (GNL_NM_CD) ★",
            "drug_code_alt": "보정주성분코드 (RVSN_WK_COMPN_CD)",
            "edi_code":      "EDI코드 (MCARE_DIV_CD)",
            "dose_once":     "1회투약량 (MPRSC_TIME1_TUYAK_CPCT)",
            "dose_freq":     "1일투약량 (MPRSC_DD1_TUYAK_CPCT)",
            "total_days":    "총투여일수 (TOT_MCNT) ★",
            "sick_code":     "주상병기호 (SICK_SYM1)",
            "institution_id":"요양기관기호 (MDCARE_SYM)",
        },
        "yoyang": {
            "institution_id":   "요양기관기호 (MDCARE_SYM)",
            "institution_type": "기관종별구분 (YOYANG_CLSFC_CD)",
            "std_year":         "기준년도 (STD_YYYY)",
            "inst_name":        "기관명 (INST_NM)",
            "addr_sgg":         "시군구코드 (ADDR_SGG_CD)",
        },
    }

    if "columns" not in cfg:
        cfg["columns"] = DEFAULT_TABLE_COLS.copy()

    inner_tabs = st.tabs(["T20", "T30", "T40", "T60", "요양기관"])
    for tab, key in zip(inner_tabs, ["t20", "t30", "t40", "t60", "yoyang"]):
        with tab:
            if key not in cfg["columns"]:
                cfg["columns"][key] = DEFAULT_TABLE_COLS.get(key, {}).copy()

            labels   = col_labels.get(key, {})
            defaults = DEFAULT_TABLE_COLS.get(key, {})

            # SAS 파일의 실제 컬럼 목록 (연결 or SAS 파일 기준)
            actual_cols: list[str] = []
            if is_sas(cfg):
                sas_fpath = None
                fname = cfg["sas"]["files"].get(key, "")
                folder = cfg["sas"].get("folder", "")
                if fname and folder:
                    candidate = Path(folder) / fname
                    if candidate.exists():
                        sas_fpath = candidate
                if sas_fpath:
                    actual_cols = get_sas_columns(sas_fpath, cfg["sas"].get("encoding", "cp949"))
            elif conn.is_connected():
                try:
                    tbl_info = cfg["tables"].get(key, {})
                    raw      = conn.get_columns(tbl_info.get("schema", ""), tbl_info.get("table", ""))
                    actual_cols = [c["name"] for c in raw]
                except Exception:
                    pass

            for field, label in labels.items():
                cur_val = cfg["columns"][key].get(field, defaults.get(field, ""))
                if actual_cols:
                    idx = actual_cols.index(cur_val) if cur_val in actual_cols else 0
                    val = st.selectbox(label, actual_cols, index=idx, key=f"col_{key}_{field}")
                else:
                    val = st.text_input(label, value=cur_val, key=f"col_{key}_{field}")
                cfg["columns"][key][field] = val

    if st.button("💾 컬럼 매핑 저장", type="primary"):
        save_config(cfg)
        st.success("컬럼 매핑이 저장되었습니다.")

    st.markdown("---")
    st.caption("★ 표시 항목이 DDI 분석 핵심 컬럼입니다. yyyymm 컬럼은 월 파티션 필터에 사용됩니다.")
