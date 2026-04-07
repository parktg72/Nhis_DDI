"""
페이지 1: 데이터 소스 선택 + HANA DB 연결 또는 SAS 파일 설정
"""
import datetime
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
from hana_app.core.table_validator import check_column_mapping, validate_all_identifiers
from hana_app.core.sas_reader import scan_sas_files, guess_table_type, get_sas_columns

st.set_page_config(page_title="데이터 소스 설정", page_icon="🔌", layout="wide")
st.title("🔌 데이터 소스 설정")

cfg  = load_config()
conn = get_connection(st.session_state)


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
tab_hana, tab_sas, tab_tbl, tab_col, tab_validate = st.tabs([
    "🗄️ HANA DB 연결",
    "📂 SAS 파일 설정",
    "📋 테이블 위치 (HANA)",
    "🗂️ 컬럼 매핑",
    "🔍 테이블 검증",
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
                            st.session_state["hana_creds"] = {
                                "host": host,
                                "port": int(port),
                                "user": user,
                                "password": password,
                            }
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


# ─────────────────────────────────────────────────────────────────────────────
# 탭 5: 테이블 검증 Wizard (HANA 전용)
# ─────────────────────────────────────────────────────────────────────────────
with tab_validate:
    st.subheader("🔍 HANA 테이블 검증")
    st.caption(
        "실제 HANA DB의 테이블·컬럼이 학습 코드와 일치하는지 확인합니다. "
        "3번 페이지(모델 학습)에서 데이터를 추출하기 전에 반드시 완료해야 합니다."
    )

    # ── 호스트 변경 감지 → wizard 캐시 전체 무효화 ─────────────────────────
    _current_host = cfg.get("connection", {}).get("host", "")
    if st.session_state.get("_wizard_last_host") != _current_host:
        for _k in [k for k in st.session_state if k.startswith("_wizard_")]:
            del st.session_state[_k]
        st.session_state["_wizard_last_host"] = _current_host

    if not (st.session_state.get("connected") and conn.is_connected()):
        st.warning("⚠️ HANA DB에 먼저 연결하세요. (🗄️ HANA DB 연결 탭)")
        st.stop()  # 주의: 페이지 전체 렌더링 중단. tab_validate는 반드시 마지막 탭이어야 함.

    # ── 현재 검증 상태 표시 ───────────────────────────────────────────────
    if cfg.get("validated"):
        _vat_raw = cfg.get("validated_at", "")
        try:
            import datetime as _dt
            _vat_display = _dt.datetime.fromisoformat(_vat_raw).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            _vat_display = _vat_raw
        st.success(
            f"✅ 검증 완료  |  "
            f"{_vat_display}  |  "
            f"호스트: {cfg.get('validated_host', '')}"
        )
        if cfg.get("validated_host") and cfg["validated_host"] != cfg["connection"]["host"]:
            st.warning(
                "⚠️ 검증된 호스트와 현재 연결 호스트가 다릅니다. 재검증을 권장합니다."
            )
    else:
        st.info("ℹ️ 아직 검증되지 않았습니다. 아래 단계를 순서대로 진행하세요.")

    st.markdown("---")

    TABLE_LOGICAL = {
        "t20":    "T20 (요양명세서)",
        "t30":    "T30 (원내 약품)",
        "t40":    "T40 (상병내역)",
        "t60":    "T60 (원외처방)",
        "yoyang": "요양기관",
    }

    # ── Step 1: 스키마 선택 ───────────────────────────────────────────────
    st.markdown("#### Step 1: 스키마 선택")

    col_refresh1, _ = st.columns([1, 5])
    with col_refresh1:
        if st.button("🔄 스키마 새로고침", key="refresh_schemas"):
            st.session_state.pop("_wizard_schemas", None)

    if "_wizard_schemas" not in st.session_state:
        with st.spinner("스키마 목록 조회 중..."):
            try:
                st.session_state["_wizard_schemas"] = conn.get_schemas()
            except Exception as e:
                st.error(f"❌ 스키마 조회 실패: {e}")
                st.stop()

    schema_list = st.session_state["_wizard_schemas"]
    if not schema_list:
        st.error("❌ 접근 가능한 스키마가 없습니다. 계정 권한을 확인하세요.")
        st.stop()

    schema_selections: dict[str, str] = {}
    cols_s = st.columns(len(TABLE_LOGICAL))
    for (tbl_key, tbl_label), col in zip(TABLE_LOGICAL.items(), cols_s):
        current_schema = cfg["tables"].get(tbl_key, {}).get("schema", "")
        default_idx = schema_list.index(current_schema) if current_schema in schema_list else 0
        with col:
            schema_selections[tbl_key] = st.selectbox(
                tbl_label,
                options=schema_list,
                index=default_idx,
                key=f"wiz_schema_{tbl_key}",
            )

    # ── Step 2: 테이블 선택 ───────────────────────────────────────────────
    st.markdown("#### Step 2: 테이블 선택")

    table_selections: dict[str, str] = {}
    for tbl_key, tbl_label in TABLE_LOGICAL.items():
        schema = schema_selections[tbl_key]
        cache_key = f"_wizard_tables_{schema}"
        col_lbl, col_sel, col_ref = st.columns([1, 3, 1])
        with col_ref:
            if st.button("🔄", key=f"refresh_tbl_{tbl_key}", help="테이블 목록 새로고침"):
                st.session_state.pop(cache_key, None)
        if cache_key not in st.session_state:
            with st.spinner(f"{schema} 테이블 목록 조회 중..."):
                try:
                    st.session_state[cache_key] = conn.get_tables(schema)
                except Exception as e:
                    st.session_state[cache_key] = []
                    st.error(f"❌ {schema} 테이블 조회 실패: {e}")
        tbl_list = st.session_state[cache_key]
        current_tbl = cfg["tables"].get(tbl_key, {}).get("table", "")
        default_idx = tbl_list.index(current_tbl) if current_tbl in tbl_list else 0
        with col_lbl:
            st.markdown(f"**{tbl_label}**")
        with col_sel:
            if tbl_list:
                table_selections[tbl_key] = st.selectbox(
                    "테이블",
                    options=tbl_list,
                    index=default_idx,
                    key=f"wiz_table_{tbl_key}",
                    label_visibility="collapsed",
                )
            else:
                st.error(f"❌ {schema} 에 테이블이 없습니다")
                table_selections[tbl_key] = ""

    # ── Step 3: 컬럼 매핑 검증 ───────────────────────────────────────────
    st.markdown("#### Step 3: 컬럼 매핑 검증")
    st.caption("ETL에 필요한 컬럼만 검증합니다. 🔴 항목은 드롭다운으로 실제 컬럼을 선택하세요.")

    # wizard에서 선택한 컬럼 매핑 (논리명 → 실제 DB 컬럼명)
    updated_col_map: dict[str, dict[str, str]] = {}

    for tbl_key, tbl_label in TABLE_LOGICAL.items():
        schema = schema_selections[tbl_key]
        table = table_selections.get(tbl_key, "")
        if not table:
            continue

        cache_key = f"_wizard_cols_{schema}_{table}"
        if cache_key not in st.session_state:
            with st.spinner(f"{table} 컬럼 조회 중..."):
                try:
                    col_info = conn.get_columns(schema, table)
                    st.session_state[cache_key] = [c["name"] for c in col_info]
                except Exception as e:
                    st.session_state[cache_key] = []
                    st.error(f"❌ {table} 컬럼 조회 실패: {e}")

        actual_cols = st.session_state[cache_key]
        expected_map: dict[str, str] = cfg.get("columns", {}).get(tbl_key, {})

        check_result = check_column_mapping(actual_cols, expected_map)

        with st.expander(
            f"**{tbl_label}** — "
            f"✅ {len(check_result['ok'])}개 일치 / "
            f"{'🔴 ' + str(len(check_result['missing'])) + '개 불일치' if check_result['missing'] else '전체 일치'}",
            expanded=bool(check_result["missing"]),
        ):
            tbl_col_map: dict[str, str] = {}
            for logical_name, db_col in expected_map.items():
                status = "✅" if logical_name in check_result["ok"] else "🔴"
                c1, c2, c3 = st.columns([1, 2, 2])
                with c1:
                    st.markdown(status)
                with c2:
                    st.markdown(f"`{logical_name}`")
                with c3:
                    if logical_name in check_result["ok"]:
                        st.markdown(f"`{db_col}`")
                        tbl_col_map[logical_name] = db_col
                    else:
                        # 불일치: 실제 컬럼 중에서 선택
                        opts = actual_cols if actual_cols else ["(컬럼 없음)"]
                        sel = st.selectbox(
                            f"{logical_name} 대체 컬럼",
                            options=opts,
                            key=f"wiz_col_{tbl_key}_{logical_name}",
                            label_visibility="collapsed",
                        )
                        tbl_col_map[logical_name] = sel

            updated_col_map[tbl_key] = tbl_col_map

    # ── Step 4: 저장 ─────────────────────────────────────────────────────
    st.markdown("#### Step 4: 저장")

    col_save, col_revalidate = st.columns(2)
    with col_save:
        if st.button("✅ 검증 완료 & 저장", type="primary", use_container_width=True):
            # 테이블 미선택 확인 (조회 실패 등으로 비어있는 경우 저장 차단)
            empty_tables = [
                TABLE_LOGICAL[k] for k in TABLE_LOGICAL
                if not table_selections.get(k)
            ]
            if empty_tables:
                st.error(
                    f"❌ 다음 테이블이 선택되지 않았습니다. 스키마/테이블 조회를 확인하세요: "
                    f"{', '.join(empty_tables)}"
                )
                st.stop()

            # 저장 전 일괄 식별자 재검증
            try:
                for tbl_key, col_map in updated_col_map.items():
                    validate_all_identifiers(col_map)
            except ValueError as e:
                st.error(f"❌ 식별자 검증 실패: {e}")
                st.stop()

            # config 업데이트
            for tbl_key, tbl_label in TABLE_LOGICAL.items():
                cfg["tables"][tbl_key] = {
                    "schema": schema_selections.get(tbl_key, cfg["tables"].get(tbl_key, {}).get("schema", "")),
                    "table":  table_selections.get(tbl_key, cfg["tables"].get(tbl_key, {}).get("table", "")),
                }
                if tbl_key in updated_col_map:
                    if "columns" not in cfg:
                        cfg["columns"] = {}
                    cfg["columns"][tbl_key] = updated_col_map[tbl_key]

            cfg["validated"] = True
            cfg["validated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            cfg["validated_host"] = cfg["connection"]["host"]
            save_config(cfg)
            st.success("✅ 검증 완료 — 3번 페이지에서 학습을 시작할 수 있습니다.")
            st.rerun()

    with col_revalidate:
        if cfg.get("validated") and st.button(
            "🔄 재검증", use_container_width=True,
            help="DB 스키마 변경 후 재검증"
        ):
            cfg["validated"] = False
            save_config(cfg)
            for key in list(st.session_state.keys()):
                if key.startswith("_wizard_"):
                    del st.session_state[key]
            st.rerun()
