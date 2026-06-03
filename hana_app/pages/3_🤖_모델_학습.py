"""
페이지 3: 데이터 추출 + 모델 선택 + 학습 실행
"""
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hana_app.core.config import load_config, save_config, is_hana, is_sas
from hana_app.core.db import get_connection
from hana_app.core.hana_etl import HANAExtractor
from hana_app.core.etl_logger import append_etl_log
from hana_app.core.sas_reader import SASExtractor
from hana_app.core.ml_runner import (
    build_patient_features, build_patient_features_from_parquet,
    features_to_dataframe, train_model,
)
from hana_app.core.sparse_research import (
    DATASETS_ROOT,
    build_smoke_command,
    dataset_display_rows,
    default_smoke_output_dir,
    find_report_paths,
    list_sparse_datasets,
    log_path_for,
    read_log_tail,
)

st.set_page_config(page_title="모델 학습", page_icon="🤖", layout="wide")
st.title("🤖 모델 선택 및 학습")

cfg  = load_config()
conn = get_connection(st.session_state)


def _quote_command(command: list[str]) -> str:
    import subprocess

    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


@st.cache_data(ttl=300)
def _cached_sparse_dataset_summaries():
    return list_sparse_datasets(DATASETS_ROOT)


def _render_sparse_research_section() -> None:
    """Show project-level sparse research artifacts without touching features_df."""
    st.markdown("---")
    with st.expander("🧪 추출 산출물 학습 (Research/Smoke)", expanded=False):
        st.caption(
            "프로젝트 루트 `data/datasets`의 `X_csr.npz` + `y.npy` + `metadata.json` 산출물을 "
            "기존 DB/SAS 학습 경로와 분리해서 확인합니다."
        )
        if st.button("데이터셋 목록 새로고침", key="refresh_sparse_research_datasets"):
            _cached_sparse_dataset_summaries.clear()
            st.rerun()
        summaries = _cached_sparse_dataset_summaries()
        if not summaries:
            st.info(f"사용 가능한 sparse 데이터셋이 없습니다. `{DATASETS_ROOT}` 경로를 확인하세요.")
            return

        st.dataframe(pd.DataFrame(dataset_display_rows(summaries)), use_container_width=True, hide_index=True)

        selected_name = st.selectbox(
            "Sparse 데이터셋 선택",
            options=[summary.name for summary in summaries],
            key="sparse_research_dataset",
        )
        selected = next(summary for summary in summaries if summary.name == selected_name)
        output_dir = default_smoke_output_dir(selected.dataset_dir, model="linear")
        command = build_smoke_command(selected.dataset_dir, output_dir)
        reports = find_report_paths(selected.dataset_dir)

        metric_cols = st.columns(5)
        metric_cols[0].metric("환자 수", f"{selected.n_patients:,}")
        metric_cols[1].metric("양성률", f"{selected.label_positive_rate_pct:.4f}%")
        metric_cols[2].metric("입력 차원", f"{selected.input_dim:,}")
        metric_cols[3].metric("평가 맥락", selected.evaluation_context)
        metric_cols[4].metric("상태", selected.status)

        with st.expander("metadata.json", expanded=False):
            st.json(selected.metadata)

        st.markdown("**Smoke 학습 CLI**")
        st.code(_quote_command(command), language="bat")
        st.caption(
            "Raw→dataset 빌드는 메모리와 시간이 큰 작업이라 앱 버튼으로 실행하지 않습니다. "
            "위 명령은 이미 생성된 sparse dataset에 대한 smoke 학습만 실행합니다."
        )

        if reports.markdown:
            st.markdown("**기존 리포트**")
            st.markdown(reports.markdown.read_text(encoding="utf-8", errors="replace"))
        elif reports.json:
            st.markdown("**기존 리포트(JSON)**")
            st.json(json.loads(reports.json.read_text(encoding="utf-8")))
        else:
            st.info(f"아직 smoke 리포트가 없습니다. 실행 후 `{output_dir}`에 리포트가 생성됩니다.")

        log_tail = read_log_tail(log_path_for(output_dir), max_lines=30)
        if log_tail:
            with st.expander("최근 실행 로그", expanded=False):
                st.code(log_tail, language=None)


_render_sparse_research_section()

# ── validated 가드 ────────────────────────────────────────────────────────
if is_hana(cfg) and not cfg.get("validated"):
    st.warning("⚠️ HANA 테이블 검증이 완료되지 않았습니다.")
    st.page_link(
        "pages/1_🔌_연결_및_테이블설정.py",
        label="👉 1번 페이지 → 🔍 테이블 검증 탭에서 완료 후 돌아오세요",
    )
    st.stop()

if is_hana(cfg):
    if cfg.get("validated_host") and \
       cfg["validated_host"] != cfg["connection"]["host"]:
        st.warning("⚠️ 검증된 DB 호스트와 현재 연결 호스트가 다릅니다. 1번 페이지에서 재검증을 권장합니다.")

# ── 자동 재연결 ───────────────────────────────────────────────────────────
_hana_creds = st.session_state.get("hana_creds")
if is_hana(cfg):
    if _hana_creds:
        try:
            conn.ensure_connected(_hana_creds, session_state=st.session_state)
        except Exception as _conn_err:
            st.error(f"❌ DB 재연결 실패: {_conn_err}")
            st.stop()
    elif not conn.is_connected():
        st.error("❌ DB 연결이 없습니다. 1번 페이지에서 먼저 연결하세요.")
        st.stop()

using_hana = is_hana(cfg)
using_sas  = is_sas(cfg)

# ─────────────────────────────────────────────────────────────────────────────
# 저장 데이터 디렉토리
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent.parent / "data" / "datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)


def _list_saved_datasets() -> list[Path]:
    return sorted(DATASET_DIR.glob("features_*.parquet"), reverse=True)


def _save_dataset(df: pd.DataFrame, meta: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parquet_path = DATASET_DIR / f"features_{ts}.parquet"
    meta_path    = DATASET_DIR / f"features_{ts}.json"
    df.to_parquet(parquet_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return parquet_path


def _load_dataset(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(path)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return df, meta


# ─────────────────────────────────────────────────────────────────────────────
# 다운로드 받은 Raw 데이터 (records_YYYYMMDD.parquet) 헬퍼
#   HANA에서 내려받은 일별 처방 원본 파일을 직접 선택해 피처 계산 → 학습.
# ─────────────────────────────────────────────────────────────────────────────
# 기본 후보 폴더: H: 전송 드라이브 → 프로젝트 data/raw 순.
_RAW_DIR_CANDIDATES = [
    r"H:\mode_11_hana\data\raw",
    str(ROOT / "data" / "raw"),
]


def _resolve_default_raw_dir(configured: str = "") -> str:
    """설정값 → 후보 폴더 순으로 존재하는 첫 raw 폴더를 반환."""
    if configured and Path(configured).is_dir():
        return configured
    for cand in _RAW_DIR_CANDIDATES:
        if Path(cand).is_dir():
            return cand
    return configured or _RAW_DIR_CANDIDATES[0]


def _parse_record_date(path: Path):
    """`records_YYYYMMDD.parquet` 파일명에서 날짜(date)를 파싱. 실패 시 None."""
    stem = path.stem
    prefix = "records_"
    if not stem.startswith(prefix):
        return None
    token = stem[len(prefix):]
    try:
        return datetime.strptime(token[:8], "%Y%m%d").date()
    except ValueError:
        return None


def _list_raw_records(raw_dir: Path) -> list[tuple[Path, object]]:
    """raw 폴더의 records_*.parquet 파일을 (경로, date|None) 목록으로 반환 (날짜 오름차순)."""
    if not raw_dir.is_dir():
        return []
    rows = [(p, _parse_record_date(p)) for p in raw_dir.glob("records_*.parquet")]
    # 날짜 있는 것 먼저(날짜순), 날짜 없는 것은 이름순으로 뒤에
    rows.sort(key=lambda r: (r[1] is None, r[1] or r[0].name))
    return rows


def _ensure_demographics_from_raw(raw_dir: Path, log=None) -> str:
    """raw 폴더의 eligibility_demographics.parquet 를 정규 DEMOGRAPHICS_PATH 로 복사.

    피처 빌더(`build_patient_features_from_parquet`)는 age/sex 를 record 컬럼이 아니라
    저장된 인구통계 파일(_load_demographics/_load_age_map)에서 읽는다. 따라서 raw 학습
    전에 이 파일을 정규 경로로 옮겨두지 않으면 age/sex_m 이 null 로 학습되어
    **조용히 성능이 저하된 모델**이 만들어진다(에러 없이). 이를 방지한다.

    Returns: "missing" | "in_place" | "copied" | "error:<msg>"
    """
    import shutil
    from hana_app.core.ml_runner import DEMOGRAPHICS_PATH, AGE_MAP_PATH

    src = raw_dir / "eligibility_demographics.parquet"
    if not src.exists():
        return "missing"
    try:
        if src.resolve() == DEMOGRAPHICS_PATH.resolve():
            return "in_place"
        DEMOGRAPHICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, DEMOGRAPHICS_PATH)
        src_age = raw_dir / "eligibility_ages.parquet"
        if src_age.exists() and src_age.resolve() != AGE_MAP_PATH.resolve():
            shutil.copy2(src_age, AGE_MAP_PATH)
        if log:
            log(f"인구통계 파일 정규 경로로 복사: {DEMOGRAPHICS_PATH}")
        return "copied"
    except Exception as e:  # noqa: BLE001 — 복사 실패는 치명적이지 않음(경고만)
        if log:
            log(f"인구통계 복사 실패: {e}")
        return f"error:{e}"


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 소스 모드 선택
# ─────────────────────────────────────────────────────────────────────────────
DATA_MODE_EXTRACT = "extract"
DATA_MODE_SAVED   = "saved"
DATA_MODE_RAW     = "raw"

saved_files = _list_saved_datasets()

_DATA_MODE_LABELS = {
    DATA_MODE_EXTRACT: "🔗  DB / SAS 파일에서 추출",
    DATA_MODE_SAVED:   f"📂  저장된 데이터 불러오기  ({len(saved_files)}개 보유)",
    DATA_MODE_RAW:     "📥  다운로드 받은 Raw 데이터 (records_*.parquet)",
}

data_mode = st.radio(
    "데이터 준비 방식",
    options=[DATA_MODE_EXTRACT, DATA_MODE_SAVED, DATA_MODE_RAW],
    format_func=lambda x: _DATA_MODE_LABELS[x],
    horizontal=True,
    key="data_mode",
)

# ── 저장된 데이터 불러오기 모드 ───────────────────────────────────────────────
if data_mode == DATA_MODE_SAVED:
    st.markdown("---")
    if not saved_files:
        st.warning(
            "저장된 데이터셋이 없습니다.  \n"
            "먼저 **DB / SAS 파일에서 추출** 모드로 데이터를 추출하고 💾 저장하세요."
        )
        st.stop()

    file_labels = {
        str(p): f"{p.stem}  ({p.stat().st_size / 1024 / 1024:.1f} MB)"
        for p in saved_files
    }
    selected_file = st.selectbox(
        "불러올 데이터셋 선택",
        options=list(file_labels.keys()),
        format_func=lambda x: file_labels[x],
    )

    if st.button("📂 데이터 불러오기", type="primary"):
        with st.spinner("데이터 로딩 중..."):
            loaded_df, loaded_meta = _load_dataset(Path(selected_file))
        # 이전 features_df 명시적 해제 (메모리 절약)
        if st.session_state.get("features_df") is not None:
            del st.session_state.features_df
            import gc as _gc; _gc.collect()
        st.session_state.features_df = loaded_df

        st.success(
            f"✅ 로드 완료: **{len(loaded_df):,}명**  |  "
            f"컬럼: {list(loaded_df.columns)[:6]}..."
        )
        if loaded_meta:
            with st.expander("📋 데이터셋 메타 정보"):
                st.json(loaded_meta)

        risk_dist = loaded_df["risk_level"].value_counts() if "risk_level" in loaded_df.columns else {}
        if risk_dist is not None and len(risk_dist):
            cols = st.columns(4)
            for col, (level, cnt) in zip(cols, risk_dist.items()):
                emoji = {"Red": "🔴", "Yellow": "🟡", "Green": "🟢", "Normal": "⚪"}.get(level, "")
                col.metric(f"{emoji} {level}", f"{cnt:,}명")

    # 이미 로드된 경우 표시
    if st.session_state.get("features_df") is not None:
        df_loaded = st.session_state.features_df
        st.info(f"📊 현재 로드된 데이터: **{len(df_loaded):,}명** — 아래에서 모델 학습을 진행하세요.")

    st.markdown("---")

# ── 다운로드 받은 Raw 데이터 모드 ──────────────────────────────────────────────
elif data_mode == DATA_MODE_RAW:
    st.markdown("---")
    st.subheader("📥 다운로드 받은 Raw 데이터")
    st.caption(
        "HANA에서 내려받은 일별 처방 원본(`records_YYYYMMDD.parquet`)을 직접 선택해 학습합니다. "
        "선택한 파일은 아래 **피처 계산**(동시복용 기간·다재약물 기준) 후 학습에 사용됩니다."
    )

    _trn_raw = cfg.get("training", {})
    _default_raw = _resolve_default_raw_dir(_trn_raw.get("raw_data_dir", ""))
    raw_dir_str = st.text_input(
        "Raw 데이터 폴더",
        value=_default_raw,
        key="raw_dir_input",
        help=r"records_YYYYMMDD.parquet 파일이 있는 폴더 (예: H:\mode_11_hana\data\raw)",
    )
    raw_dir = Path(raw_dir_str.strip()) if raw_dir_str.strip() else Path(_default_raw)
    st.session_state["raw_data_dir"] = str(raw_dir)

    if not raw_dir.is_dir():
        st.warning(f"⚠️ 폴더를 찾을 수 없습니다: `{raw_dir}`")
        st.stop()

    raw_records = _list_raw_records(raw_dir)
    if not raw_records:
        st.warning(f"⚠️ `{raw_dir}` 에 records_*.parquet 파일이 없습니다.")
        st.stop()

    # ── 인구통계(나이/성별) 파일 상태 ──────────────────────────────────────
    _demo_src = raw_dir / "eligibility_demographics.parquet"
    if _demo_src.exists():
        st.success(
            "✅ `eligibility_demographics.parquet` 감지 — 나이·성별 피처가 학습에 포함됩니다."
        )
    else:
        st.warning(
            "⚠️ `eligibility_demographics.parquet` 이(가) 없습니다. "
            "나이·성별(age·sex_m) 없이 학습되어 성능이 저하될 수 있습니다. "
            "가능하면 같은 폴더에 인구통계 파일을 두세요."
        )

    _dated = [(p, d) for p, d in raw_records if d is not None]
    _undated = [p for p, d in raw_records if d is None]

    # ── 선택 방식: 기간 설정 vs 파일 직접 선택 ─────────────────────────────
    sel_method = st.radio(
        "선택 방식",
        options=["date_range", "files"],
        format_func=lambda x: {
            "date_range": "📅 기간으로 선택",
            "files": "🗂️ 파일 직접 선택",
        }[x],
        horizontal=True,
        key="raw_sel_method",
        disabled=not _dated,
    )

    chosen_paths: list[Path] = []
    if sel_method == "date_range" and _dated:
        _min_d = _dated[0][1]
        _max_d = _dated[-1][1]
        st.caption(
            f"사용 가능 기간: **{_min_d.isoformat()} ~ {_max_d.isoformat()}** "
            f"(일별 파일 {len(_dated)}개)"
        )
        # 키를 폴더에 종속시켜 폴더 전환 시 이전 날짜가 새 범위 밖이라 크래시 나는 것을 방지
        _dk = abs(hash(str(raw_dir)))
        rc1, rc2 = st.columns(2)
        with rc1:
            d_from = st.date_input(
                "시작일", value=_min_d, min_value=_min_d, max_value=_max_d, key=f"raw_date_from_{_dk}",
            )
        with rc2:
            d_to = st.date_input(
                "종료일", value=_max_d, min_value=_min_d, max_value=_max_d, key=f"raw_date_to_{_dk}",
            )
        if d_from > d_to:
            st.error("⚠️ 시작일이 종료일보다 늦습니다.")
            st.stop()
        chosen_paths = [p for p, d in _dated if d_from <= d <= d_to]
    else:
        # 파일 직접 선택 (날짜 없는 파일 포함)
        _all_paths = [p for p, _ in raw_records]
        _labels = {
            str(p): f"{p.name}  ({p.stat().st_size / 1024 / 1024:.1f} MB)"
            for p in _all_paths
        }
        _picked = st.multiselect(
            "학습에 사용할 records 파일 선택",
            options=list(_labels.keys()),
            default=list(_labels.keys()),
            format_func=lambda x: _labels[x],
            key=f"raw_file_pick_{abs(hash(str(raw_dir)))}",
        )
        chosen_paths = [Path(x) for x in _picked]

    if _undated and sel_method == "date_range":
        st.caption(
            f"ℹ️ 날짜를 해석할 수 없는 파일 {len(_undated)}개는 기간 선택에서 제외됩니다. "
            "포함하려면 '파일 직접 선택'을 사용하세요."
        )

    if not chosen_paths:
        st.warning("선택된 파일이 없습니다. 기간 또는 파일을 선택하세요.")
        st.session_state["raw_selected_paths"] = []
        st.stop()

    st.session_state["raw_selected_paths"] = [str(p) for p in chosen_paths]

    _total_mb = sum(p.stat().st_size for p in chosen_paths) / 1024 / 1024
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("선택 파일 수", f"{len(chosen_paths):,}개")
    mcol2.metric("총 용량", f"{_total_mb:,.1f} MB")
    _chosen_dates = [d for p, d in _dated if p in set(chosen_paths)]
    if _chosen_dates:
        mcol3.metric("기간", f"{min(_chosen_dates).isoformat()} ~ {max(_chosen_dates).isoformat()}")
    else:
        mcol3.metric("기간", "—")

    st.info(
        "📌 아래 **2️⃣ 모델 선택** 이후 **🚀 학습 시작** 을 누르면 선택한 Raw 파일에서 "
        "피처를 계산한 뒤 학습합니다. (동시복용 기간·다재약물 기준은 아래 설정 사용)"
    )
    st.markdown("---")

else:
    # 데이터 소스 표시
    if using_sas:
        st.info("📂 데이터 소스: **SAS 파일**")
    else:
        st.info("🗄️ 데이터 소스: **SAP HANA DB**")

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 1: 데이터 추출 범위 (추출 모드에서만 표시)
# ─────────────────────────────────────────────────────────────────────────────
_show_extract_section = (data_mode == DATA_MODE_EXTRACT)
st.header("1️⃣ 데이터 추출 범위")
if data_mode == DATA_MODE_RAW:
    st.info(
        "📥 다운로드 Raw 모드 — 아래 시작/종료 년·월은 무시되고, 위에서 선택한 파일/기간을 사용합니다. "
        "단, **동시복용 판단 기간**·**다재약물 기준**은 피처 계산에 그대로 적용됩니다."
    )
elif not _show_extract_section:
    st.info("📂 저장된 데이터 모드 — 추출 범위 설정을 건너뜁니다.")
col1, col2, col3, col4 = st.columns(4)

trn = cfg.get("training", {})
with col1:
    year_from = st.selectbox(
        "시작 년도",
        options=[str(y) for y in range(2015, 2026)],
        index=[str(y) for y in range(2015, 2026)].index(trn.get("year_from", "2023")),
    )
with col2:
    month_from = st.selectbox(
        "시작 월",
        options=[f"{m:02d}" for m in range(1, 13)],
        index=int(trn.get("month_from", "01")) - 1,
    )
with col3:
    year_to = st.selectbox(
        "종료 년도",
        options=[str(y) for y in range(2015, 2026)],
        index=[str(y) for y in range(2015, 2026)].index(trn.get("year_to", "2023")),
    )
with col4:
    month_to = st.selectbox(
        "종료 월",
        options=[f"{m:02d}" for m in range(1, 13)],
        index=int(trn.get("month_to", "12")) - 1,
    )

col_w1, col_w2 = st.columns(2)
with col_w1:
    window_days = st.number_input(
        "동시복용 판단 기간 (일)",
        value=int(trn.get("window_days", 90)),
        min_value=30,
        max_value=365,
        step=30,
        help="이 기간 내 처방된 약물을 동시복용으로 판단합니다 (기본: 90일)",
    )
with col_w2:
    poly_threshold = st.number_input(
        "다재약물 기준 (종 이상)",
        value=int(trn.get("poly_threshold", 5)),
        min_value=2,
        max_value=20,
        step=1,
        help="이 수 이상의 약물을 복용하는 환자만 분석합니다 (기본: 5종)",
    )

# ── 메모리 절약 청크 모드 ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("💾 메모리 절약 설정")

# ── RAM 한도 설정 (핵심) ─────────────────────────────────────────────────
_sys_ram_mb = 8192  # 기본값
try:
    import psutil
    _sys_ram_mb = int(psutil.virtual_memory().total / 1024 / 1024)
except ImportError:
    pass

ram_col1, ram_col2 = st.columns([3, 1])
with ram_col1:
    memory_limit_mb = st.slider(
        "🧠 RAM 사용 한도 (MB)",
        min_value=512,
        max_value=max(32768, _sys_ram_mb),
        value=int(trn.get("memory_limit_mb", min(4096, _sys_ram_mb * 3 // 4))),
        step=256,
        help=(
            "전체 파이프라인(추출·피처·학습)에서 사용할 최대 RAM.\n\n"
            "**DuckDB 디스크 스필**: 이 한도를 초과하면 DuckDB가 자동으로 "
            "중간 결과를 디스크에 저장하여 메모리 오류를 방지합니다.\n\n"
            "- **2 GB 이하**: 소규모 테스트용 (병렬 작업 제한)\n"
            "- **4 GB**: 일반적인 사용 권장\n"
            "- **8 GB 이상**: 대규모 데이터 고속 처리\n\n"
            f"현재 시스템 RAM: {_sys_ram_mb:,} MB"
        ),
    )
with ram_col2:
    st.metric("시스템 RAM", f"{_sys_ram_mb:,} MB")
    _usage_pct = memory_limit_mb / _sys_ram_mb * 100
    if _usage_pct > 85:
        st.warning(f"⚠️ {_usage_pct:.0f}% 사용")
    elif _usage_pct > 60:
        st.info(f"📊 {_usage_pct:.0f}% 사용")
    else:
        st.success(f"✅ {_usage_pct:.0f}% 사용")

chunk_col1, chunk_col2, chunk_col2b, chunk_col3, chunk_col4 = st.columns(5)
with chunk_col1:
    use_chunked = st.toggle(
        "청크 모드 (대용량 데이터)",
        value=True,
        help="청크 단위로 나눠 추출 후 Parquet에 저장 → 메모리 부족 방지. 대용량 데이터에 권장.",
    )
with chunk_col2:
    chunk_unit = st.selectbox(
        "청크 단위",
        options=["month", "day"],
        format_func=lambda x: {"month": "월 단위", "day": "일 단위 (대용량)"}[x],
        index=0,
        disabled=not use_chunked,
        help="대용량(10TB+)은 일 단위를 권장합니다. 월 1개도 메모리에 안 들어가면 일 단위를 선택하세요.",
    )
with chunk_col2b:
    if chunk_unit == "month":
        chunk_months = st.number_input(
            "청크 크기 (개월)",
            value=1, min_value=1, max_value=6, step=1,
            disabled=not use_chunked,
            help="한 번에 처리할 개월 수. 메모리가 부족하면 1로 설정.",
        )
        chunk_days = 1
    else:
        chunk_days = st.number_input(
            "청크 크기 (일)",
            value=1, min_value=1, max_value=30, step=1,
            disabled=not use_chunked,
            help="한 번에 처리할 일 수. 4천만 명 규모는 1일 권장.",
        )
        chunk_months = 1
with chunk_col3:
    # 메모리 한도에 따라 환자 배치 크기 기본값 자동 조정
    _default_batch = max(500, min(10000, memory_limit_mb * 2))
    patient_batch = st.number_input(
        "환자 배치 크기",
        value=_default_batch, min_value=500, max_value=50000, step=500,
        disabled=not use_chunked,
        help=(
            "피처 계산 시 한 번에 처리할 환자 수. "
            f"현재 RAM 한도({memory_limit_mb:,} MB) 기준 권장: {_default_batch:,}명"
        ),
    )
with chunk_col4:
    gpu_memory_pct = st.slider(
        "GPU 메모리 한도 (%)",
        min_value=30, max_value=100, value=70, step=5,
        help=(
            "GPU가 있을 때 최대 사용할 VRAM 비율.\n"
            "나머지는 OS·디스플레이용으로 유지합니다.\n"
            "torch 또는 cupy 설치 시 가장 정확하게 적용됩니다."
        ),
    )
    gpu_memory_fraction = gpu_memory_pct / 100.0

st.subheader("📅 버퍼 설정")
st.caption("동시복용 판단을 위해 분석 기간 전후로 추가 데이터를 추출합니다.")
buf_col1, buf_col2 = st.columns(2)
with buf_col1:
    buffer_before_days = st.number_input(
        "시작 전 버퍼 (일)",
        value=int(trn.get("buffer_before_days", window_days)),
        min_value=0,
        max_value=365,
        step=30,
        help="분석 시작일 이전에 추가로 추출할 기간 (기본: 동시복용 판단 기간과 동일)",
    )
with buf_col2:
    buffer_after_days = st.number_input(
        "종료 후 버퍼 (일)",
        value=int(trn.get("buffer_after_days", 0)),
        min_value=0,
        max_value=365,
        step=30,
        help="분석 종료일 이후에 추가로 추출할 기간 (종료 시점 처방의 동시복용 판단용)",
    )

_before_months = max(1, (buffer_before_days + 29) // 30) if buffer_before_days > 0 else 0
_after_months = max(1, (buffer_after_days + 29) // 30) if buffer_after_days > 0 else 0
_parts = []
if buffer_before_days > 0:
    _parts.append(f"시작일({year_from}/{month_from}) 이전 {_before_months}개월")
if buffer_after_days > 0:
    _parts.append(f"종료일({year_to}/{month_to}) 이후 {_after_months}개월")
if _parts:
    st.info(f"📅 버퍼 적용: {' + '.join(_parts)}치 처방도 추출합니다.")
else:
    st.info("📅 버퍼 없음: 설정된 기간만 추출합니다.")

# ── 질환 필터 (ICD-10) ──────────────────────────────────────────────
st.subheader("🩺 질환 필터 (ICD-10)")
st.caption(
    "T40 상병코드로 특정 질환 환자만 먼저 추출합니다. "
    "3자리 입력 시 하위 코드 전체 포함 (예: E11 → E11.0, E11.1 …). "
    "여러 코드는 쉼표(,)로 구분하여 입력하세요."
)
_dis_col1, _dis_col2 = st.columns([1, 3])
with _dis_col1:
    use_disease_filter = st.toggle(
        "질환 필터 활성화",
        value=False,
        help="T40에서 ICD-10 코드에 해당하는 환자만 추출합니다.",
    )
with _dis_col2:
    icd10_input = st.text_input(
        "ICD-10 코드 (쉼표 구분)",
        value="",
        placeholder="예: E11, I10, J45.0",
        disabled=not use_disease_filter,
        help=(
            "3자리 입력 → 하위 코드 포함 (E11 = E11.* 전체).\n"
            "전체 코드 입력도 가능합니다 (E11.9 → E11.9 정확히 또는 하위)."
        ),
    )


def _parse_icd10(text: str) -> list[str]:
    """쉼표 구분 ICD-10 입력 → 정규화된 prefix 리스트."""
    if not text:
        return []
    return [c.strip().upper() for c in text.split(",") if c.strip()]


st.subheader("👤 자격DB 인구통계 설정")
st.caption("자격DB의 BYEAR·SEX_TYPE·RVSN_ADDR_CD를 처방 추출 후 대상자만 조회합니다.")
_elig_cfg = cfg.get("analysis_dbs", {}).get("eligibility", {})
_age_col1, _age_col2, _age_col3 = st.columns(3)
with _age_col1:
    use_eligibility = st.toggle(
        "자격DB 인구통계 활성화",
        value=_elig_cfg.get("enabled", False),
        help="처방 추출 완료 후 고유 환자 ID로만 자격DB를 조회합니다. 비활성화 시 나이/성별/지역 없이 분석합니다.",
    )
with _age_col2:
    reference_year = st.number_input(
        "기준년도 (STD_YYYY 필터)",
        value=int(year_to),
        min_value=2015,
        max_value=2030,
        step=1,
        disabled=not use_eligibility,
        help="자격DB의 STD_YYYY가 이 년도와 일치하는 레코드만 조회합니다.",
    )
with _age_col3:
    addr_digits = st.selectbox(
        "지역코드 자릿수",
        options=[5, 8],
        index=0,
        disabled=not use_eligibility,
        help="RVSN_ADDR_CD 앞 몇 자리를 사용할지 선택합니다 (5=시군구, 8=읍면동).",
    )

# ── 사전 층화 샘플링 ────────────────────────────────────────────────
st.markdown("**👥 사전 층화 샘플링** — 자격DB에서 먼저 환자를 샘플링한 뒤 처방 추출")
st.caption(
    "자격DB를 먼저 조회하여 성별·연령·지역 비율을 유지하며 N명을 샘플링합니다. "
    "그 환자들의 T20/T30 등 처방 데이터만 추출하므로 전체 데이터를 내려받지 않아도 됩니다."
)
_ps_col1, _ps_col2, _ps_col3 = st.columns(3)
with _ps_col1:
    use_pre_sampling = st.toggle(
        "사전 층화 샘플링 활성화",
        value=False,
        disabled=not use_eligibility,
        help="자격DB 인구통계 활성화 시에만 사용 가능합니다.",
    )
with _ps_col2:
    pre_sample_size = st.number_input(
        "샘플 환자 수 (명)",
        value=500_000,
        min_value=1_000,
        step=10_000,
        disabled=not (use_eligibility and use_pre_sampling),
        help="자격DB에서 층화 추출할 총 환자 수.",
    )
with _ps_col3:
    pre_sample_seed = st.number_input(
        "샘플링 시드",
        value=42,
        min_value=0,
        step=1,
        disabled=not (use_eligibility and use_pre_sampling),
        help="재현성을 위한 랜덤 시드.",
    )

if use_eligibility and use_pre_sampling:
    st.caption(
        f"층화 기준: 성별(SEX_TYPE) × 연령구간(0-19·20-39·40-59·60-74·75+) × "
        f"지역코드 앞 {addr_digits}자리 → {pre_sample_size:,}명 추출 후 처방 데이터 조회"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 2: 모델 선택
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("2️⃣ 모델 선택 및 하이퍼파라미터")

# 예측 타겟
target = st.selectbox(
    "예측 타겟",
    options=["risk_binary", "hierarchical", "risk_label"],
    format_func=lambda x: {
        "risk_binary":   "이진 분류 (위험 / 정상)",
        "hierarchical":  "계층 분류 — Stage 1 Red 이진 + Stage 2 Yellow 6-class (권장)",
        "risk_label":    "4분류 (Red / Yellow / Green / Normal) — 레거시",
    }[x],
    index=["risk_binary", "hierarchical", "risk_label"].index(
        trn.get("target", "risk_binary")
        if trn.get("target") in ("risk_binary", "hierarchical", "risk_label")
        else "risk_binary"
    ),
)

# ── Phase 탭 ─────────────────────────────────────────────────────────────
_saved_p2 = trn.get("models_phase2", ["xgboost"])
_saved_p3 = trn.get("models_phase3", [])

tab_p2, tab_p3 = st.tabs(["📊 Phase 2 — 앙상블/전통 ML", "🧪 Phase 3 — 딥러닝 (실험적)"])

params_map: dict = {}

with tab_p2:
    st.caption("체크박스로 학습할 모델을 복수 선택할 수 있습니다. 선택된 모든 모델이 순차 학습됩니다.")
    p2_col1, p2_col2 = st.columns(2)
    with p2_col1:
        sel_xgb = st.checkbox("XGBoost (권장)", value="xgboost" in _saved_p2, key="sel_xgb")
        sel_lgbm = st.checkbox("LightGBM (빠름)", value="lightgbm" in _saved_p2, key="sel_lgbm")
        sel_cat = st.checkbox("CatBoost", value="catboost" in _saved_p2, key="sel_cat")
    with p2_col2:
        sel_rf = st.checkbox("Random Forest", value="random_forest" in _saved_p2, key="sel_rf")
        sel_lr = st.checkbox("Logistic Regression (기준선)", value="logistic" in _saved_p2, key="sel_lr")
        sel_stack = st.checkbox("Stacking Ensemble", value="stacking" in _saved_p2, key="sel_stack")

    # 모델별 하이퍼파라미터
    if sel_xgb:
        with st.expander("⚙️ XGBoost 하이퍼파라미터", expanded=False):
            _xp = {}
            c1, c2, c3 = st.columns(3)
            with c1:
                _xp["n_estimators"] = st.slider("트리 수", 50, 500, 200, 50, key="xgb_n")
                _xp["max_depth"] = st.slider("최대 깊이", 2, 15, 6, key="xgb_d")
            with c2:
                _xp["learning_rate"] = st.slider("학습률", 0.01, 0.5, 0.1, 0.01, format="%.2f", key="xgb_lr")
                _xp["subsample"] = st.slider("Subsample", 0.5, 1.0, 0.8, 0.05, key="xgb_ss")
            with c3:
                _xp["colsample_bytree"] = st.slider("Colsample", 0.5, 1.0, 0.8, 0.05, key="xgb_cs")
            params_map["xgboost"] = _xp

    if sel_lgbm:
        with st.expander("⚙️ LightGBM 하이퍼파라미터", expanded=False):
            _lp = {}
            c1, c2, c3 = st.columns(3)
            with c1:
                _lp["n_estimators"] = st.slider("트리 수", 50, 500, 200, 50, key="lgb_n")
                _lp["max_depth"] = st.slider("최대 깊이", 2, 15, 6, key="lgb_d")
            with c2:
                _lp["learning_rate"] = st.slider("학습률", 0.01, 0.5, 0.1, 0.01, format="%.2f", key="lgb_lr")
                _lp["subsample"] = st.slider("Subsample", 0.5, 1.0, 0.8, 0.05, key="lgb_ss")
            with c3:
                _lp["colsample_bytree"] = st.slider("Colsample", 0.5, 1.0, 0.8, 0.05, key="lgb_cs")
            params_map["lightgbm"] = _lp

    if sel_cat:
        with st.expander("⚙️ CatBoost 하이퍼파라미터", expanded=False):
            _cp = {}
            c1, c2 = st.columns(2)
            with c1:
                _cp["iterations"] = st.slider("반복 횟수", 50, 500, 200, 50, key="cat_n")
                _cp["depth"] = st.slider("트리 깊이", 2, 10, 6, key="cat_d")
            with c2:
                _cp["learning_rate"] = st.slider("학습률", 0.01, 0.5, 0.1, 0.01, format="%.2f", key="cat_lr")
            params_map["catboost"] = _cp

    if sel_rf:
        with st.expander("⚙️ Random Forest 하이퍼파라미터", expanded=False):
            _rp = {}
            c1, c2 = st.columns(2)
            with c1:
                _rp["n_estimators"] = st.slider("트리 수", 50, 500, 200, 50, key="rf_n")
                _rp["max_depth"] = st.slider("최대 깊이", 3, 20, 10, key="rf_d")
            with c2:
                _rp["min_samples_leaf"] = st.slider("최소 리프 샘플", 1, 50, 5, key="rf_ml")
            params_map["random_forest"] = _rp

    if sel_lr:
        with st.expander("⚙️ Logistic Regression 하이퍼파라미터", expanded=False):
            _lrp = {}
            c1, c2 = st.columns(2)
            with c1:
                _lrp["C"] = st.slider("정규화 강도 C", 0.01, 10.0, 1.0, 0.01, key="lr_c")
            with c2:
                _lrp["solver"] = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"], key="lr_s")
            params_map["logistic"] = _lrp

    if sel_stack:
        with st.expander("⚙️ Stacking Ensemble 설정", expanded=False):
            stack_base = st.multiselect(
                "Base 모델 (2개 이상 선택)",
                options=["xgboost", "lightgbm", "random_forest"],
                default=["xgboost", "lightgbm", "random_forest"],
                key="stack_base",
            )
            params_map["stacking"] = {"base_models": stack_base}

if target == "hierarchical":
    st.info(
        "**계층 분류 모드**: Stage 1 / Stage 2 모두 XGBoost 내부 고정.\n"
        "아래 모델 선택 및 하이퍼파라미터는 적용되지 않으며, "
        "Stage 1·2 파라미터는 '학습 전략 옵션' 섹션에서 별도 설정합니다."
    )

with tab_p3:
    st.warning(
        "⚠️ 현재 Phase 3 탭은 **환자 단위 집계 피처 기반 실험 모델**입니다.\n\n"
        "- **TabNet**: `pytorch-tabnet` 패키지 필요 (즉시 사용 가능)\n"
        "- **GNN / Transformer**: `torch` 패키지 필요, 현재는 pseudo-DL 래퍼입니다.\n\n"
        "운영 DL 추론은 별도 DL 데이터셋이 필요합니다: 처방 이력 시퀀스, "
        "drug_vocab, edge_index, model_config, MANIFEST 해시 검증 번들."
    )
    sel_tabnet = st.checkbox("TabNet (tabular attention)", value="tabnet" in _saved_p3, key="sel_tabnet")
    sel_gnn = st.checkbox("GNN pseudo-DL (집계 피처 기반)", value="gnn" in _saved_p3, key="sel_gnn")
    sel_tt = st.checkbox("Temporal Transformer pseudo-DL (집계 피처 기반)", value="temporal_transformer" in _saved_p3, key="sel_tt")

    if sel_tabnet:
        with st.expander("⚙️ TabNet 하이퍼파라미터", expanded=False):
            _tp = {}
            c1, c2 = st.columns(2)
            with c1:
                _tp["n_steps"] = st.slider("N steps", 1, 10, 3, key="tn_ns")
                _tp["n_d"] = st.slider("N_d (결정 차원)", 4, 64, 8, key="tn_nd")
            with c2:
                _tp["max_epochs"] = st.slider("Epochs", 10, 200, 100, 10, key="tn_ep")
                _tp["patience"] = st.slider("Patience", 5, 30, 15, key="tn_pt")
            params_map["tabnet"] = _tp

    if sel_gnn:
        with st.expander("⚙️ GNN 하이퍼파라미터", expanded=False):
            _gp = {}
            c1, c2 = st.columns(2)
            with c1:
                _gp["hidden_dim"] = st.slider("Hidden 차원", 16, 128, 64, 16, key="gnn_hd")
                _gp["num_layers"] = st.slider("레이어 수", 1, 4, 2, key="gnn_nl")
            with c2:
                _gp["max_epochs"] = st.slider("Epochs", 10, 100, 50, 10, key="gnn_ep")
                _gp["lr"] = st.slider("학습률", 0.0001, 0.01, 0.001, 0.0001, format="%.4f", key="gnn_lr")
            params_map["gnn"] = _gp

    if sel_tt:
        with st.expander("⚙️ Temporal Transformer 하이퍼파라미터", expanded=False):
            _ttp = {}
            c1, c2 = st.columns(2)
            with c1:
                _ttp["d_model"] = st.slider("D model", 16, 64, 32, 8, key="tt_dm")
                _ttp["nhead"] = st.selectbox("Attention Heads", [2, 4, 8], index=1, key="tt_nh")
            with c2:
                _ttp["max_epochs"] = st.slider("Epochs", 10, 100, 50, 10, key="tt_ep")
                _ttp["lr"] = st.slider("학습률", 0.0001, 0.01, 0.001, 0.0001, format="%.4f", key="tt_lr")
            params_map["temporal_transformer"] = _ttp

# ── 학습 전략 옵션 ───────────────────────────────────────────────────────
st.subheader("🎯 학습 전략 옵션")

# ── 층화 샘플링 ──────────────────────────────────────────────────────
st.markdown("**📊 층화 샘플링** — 대용량 데이터에서 메모리 오류 없이 학습")
samp_col1, samp_col2, samp_col3 = st.columns(3)
with samp_col1:
    use_sampling = st.toggle(
        "층화 샘플링 사용",
        value=True,
        help=(
            "4천만 명 전체를 메모리에 올리지 않고, 위험도 비율을 유지하면서 "
            "지정한 크기만 추출하여 학습합니다. 통계적으로 100만~500만 명이면 충분합니다."
        ),
    )
with samp_col2:
    sampling_size = st.number_input(
        "샘플 크기 (만 명)",
        value=100, min_value=10, max_value=1000, step=10,
        disabled=not use_sampling,
        help="한 번에 학습할 환자 수. RAM 4GB → 100만 명, 8GB → 300만 명 권장.",
    )
    sampling_size_actual = sampling_size * 10000 if use_sampling else 0
with samp_col3:
    sampling_rounds = st.number_input(
        "샘플링 반복 횟수",
        value=1, min_value=1, step=1,
        disabled=not use_sampling,
        help=(
            "서로 다른 시드로 N회 샘플링하여 각각 학습 후 최고 성능 모델을 선택합니다.\n"
            "3~5회 추천. 각 라운드의 F1/AUC 평균±표준편차도 확인 가능합니다.\n"
            "횟수 제한 없이 직접 입력할 수 있습니다."
        ),
    )
    if not use_sampling:
        sampling_rounds = 1

if use_sampling:
    _est_mem_mb = sampling_size_actual * 16 * 8 / 1024 / 1024  # 대략적 피처 행렬 크기
    st.caption(
        f"📦 각 라운드: {sampling_size_actual:,}명 × {sampling_rounds}회 | "
        f"예상 피처 메모리: ~{_est_mem_mb:,.0f} MB"
    )

# ── 기타 학습 옵션 ───────────────────────────────────────────────────
opt_col1, opt_col2 = st.columns(2)
with opt_col1:
    use_threshold_opt = st.toggle(
        "Threshold Optimization",
        value=trn.get("threshold_optimization", False),
        help="이진 분류 시 FP/FN 비용 기반으로 최적 결정 임계값 탐색",
    )
with opt_col2:
    use_cost_sensitive = st.toggle(
        "Cost-Sensitive Learning",
        value=trn.get("cost_sensitive", False),
        help="FN(위험 환자를 정상으로 분류)에 더 높은 비용을 부여하여 고위험 Recall 향상",
    )

cost_fp, cost_fn = 1.0, 5.0
if use_threshold_opt or use_cost_sensitive:
    cs_col1, cs_col2 = st.columns(2)
    with cs_col1:
        cost_fp = st.number_input(
            "FP 비용 (정상→위험 오분류)", value=float(trn.get("cost_fp", 1.0)),
            min_value=0.1, step=0.5, key="cost_fp",
        )
    with cs_col2:
        cost_fn = st.number_input(
            "FN 비용 (위험→정상 오분류)", value=float(trn.get("cost_fn", 5.0)),
            min_value=0.1, step=0.5, key="cost_fn",
            help="의료 도메인에서 FN이 더 위험하므로 기본값 5.0",
        )

# ── 계층 분류 전용 임계값 파라미터 ──────────────────────────────────
recall_floor = 0.90
review_recall_target = 0.98
if target == "hierarchical":
    st.markdown("---")
    st.subheader("🎯 계층 분류 임계값 설정 (Stage 1)")
    st.caption(
        "τ_red: Red 확정 임계값 (Recall ≥ recall_floor 보장).  "
        "τ_review: Red 의심 태그 임계값 (FN 영구 유실 방지용)."
    )
    hr_col1, hr_col2 = st.columns(2)
    with hr_col1:
        recall_floor = st.slider(
            "Recall Floor (τ_red 결정 기준)",
            min_value=0.80, max_value=0.99, value=float(trn.get("recall_floor", 0.90)),
            step=0.01, format="%.2f",
            help="Stage 1 Red 탐지 최소 Recall. 낮추면 τ_red 완화 → FP 증가.",
        )
    with hr_col2:
        review_recall_target = st.slider(
            "Review Recall Target (τ_review 결정 기준)",
            min_value=0.90, max_value=1.00, value=float(trn.get("review_recall_target", 0.98)),
            step=0.01, format="%.2f",
            help="Red 의심 태그를 달 τ_review 기준 Recall. recall_floor 보다 높게 유지하세요.",
        )
    if review_recall_target <= recall_floor:
        st.warning("⚠️ Review Recall Target 은 Recall Floor 보다 높아야 합니다.")

# ── 선택된 모델 목록 조합 ─────────────────────────────────────────────────
selected_models_p2 = [k for k, v in {
    "xgboost": sel_xgb, "lightgbm": sel_lgbm, "catboost": sel_cat,
    "random_forest": sel_rf, "logistic": sel_lr, "stacking": sel_stack,
}.items() if v]
selected_models_p3 = [k for k, v in {
    "tabnet": sel_tabnet, "gnn": sel_gnn, "temporal_transformer": sel_tt,
}.items() if v]
all_selected_models = selected_models_p2 + selected_models_p3

# ── 평가 설정 ──────────────────────────────────────────────────────────
col_ev1, col_ev2 = st.columns(2)
with col_ev1:
    test_size = st.slider("테스트 비율", 0.1, 0.4, float(trn.get("test_size", 0.2)), 0.05)
with col_ev2:
    cv_folds = st.slider("교차검증 Fold 수", 3, 10, int(trn.get("cv_folds", 5)))

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 3: 피처 선택
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("3️⃣ 피처 선택")

from hana_app.core.ml_runner import FEATURE_COLS

FEATURE_LABELS = {
    "drug_count": "총 약물 수 (고유 주성분)",
    "drug_count_7d": "최근 7일 동시 복용 수",
    "institution_count": "처방 기관 수",
    "ddi_contraindicated": "금기 DDI 쌍 수",
    "ddi_major": "Major DDI 쌍 수",
    "ddi_moderate": "Moderate DDI 쌍 수",
    "ddi_minor": "Minor DDI 쌍 수",
    "triple_whammy": "Triple Whammy 여부",
    "qt_risk_count": "QT 연장 위험 약물 수",
    "dup_same_ingredient": "동일 성분 중복 수",
    "dup_atc5": "ATC 5단계 중복 수",
    "dup_atc4": "ATC 4단계 중복 수",
    "dup_atc3": "ATC 3단계 중복 수",
    "dup_efmdc": "약효 분류 중복 수",
    "age": "연령",
    "sex_m": "성별 (남=1)",
}

selected_features = st.multiselect(
    "학습에 사용할 피처",
    options=FEATURE_COLS,
    default=FEATURE_COLS,
    format_func=lambda x: f"{x} – {FEATURE_LABELS.get(x, '')}",
)

# ─────────────────────────────────────────────────────────────────────────────
# 설정 저장 + 학습 실행
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
col_save, col_run = st.columns(2)
with col_save:
    if st.button("💾 설정 저장", use_container_width=True):
        cfg["training"].update({
            "year_from": year_from,
            "year_to": year_to,
            "month_from": month_from,
            "month_to": month_to,
            "window_days": window_days,
            "poly_threshold": poly_threshold,
            "buffer_before_days": buffer_before_days,
            "buffer_after_days": buffer_after_days,
            "test_size": test_size,
            "cv_folds": cv_folds,
            "model": all_selected_models[0] if all_selected_models else "xgboost",
            "models_phase2": selected_models_p2,
            "models_phase3": selected_models_p3,
            "target": target,
            "memory_limit_mb": memory_limit_mb,
            "threshold_optimization": use_threshold_opt,
            "cost_sensitive": use_cost_sensitive,
            "cost_fp": cost_fp,
            "cost_fn": cost_fn,
            "recall_floor": recall_floor,
            "review_recall_target": review_recall_target,
            "raw_data_dir": st.session_state.get("raw_data_dir", trn.get("raw_data_dir", "")),
        })
        save_config(cfg)
        st.success("학습 설정이 저장되었습니다.")

with col_run:
    run_btn = st.button("🚀 학습 시작", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 학습 실행
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not selected_features:
        st.error("최소 1개 이상의 피처를 선택하세요.")
        st.stop()
    if target != "hierarchical" and not all_selected_models:
        st.error("최소 1개 이상의 모델을 선택하세요.")
        st.stop()

    import time as _time

    progress_bar = st.progress(0, text="준비 중...")
    status_text = st.empty()
    log_expander = st.expander("📋 상세 로그", expanded=True)
    log_container = log_expander.empty()
    log_lines: list[str] = []
    _phase = {"lo": 0.0, "hi": 1.0, "start": _time.time()}
    _etl_start: float = _time.time()   # ETL 전체 시작 시각 (로그용)

    def _set_phase(lo: float, hi: float) -> None:
        _phase["lo"] = lo
        _phase["hi"] = hi
        _phase["start"] = _time.time()

    def log(msg: str) -> None:
        elapsed = _time.time() - _phase["start"]
        log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        log_container.code("\n".join(log_lines[-15:]), language=None)
        progress_bar.progress(
            min(_phase["lo"], 0.99),
            text=f"{msg}  ({elapsed:.0f}s)",
        )
        status_text.info(f"**{msg}**")

    def update_pct(frac: float) -> None:
        lo, hi = _phase["lo"], _phase["hi"]
        overall = lo + max(0.0, min(1.0, frac)) * (hi - lo)
        elapsed = _time.time() - _phase["start"]
        last_msg = log_lines[-1].split("] ", 1)[-1] if log_lines else "처리 중..."
        progress_bar.progress(
            min(overall, 0.99),
            text=f"{last_msg}  ({elapsed:.0f}s)",
        )

    # ── 저장된 데이터 모드: 추출·피처계산 건너뜀 ─────────────────────────
    if data_mode == DATA_MODE_SAVED:
        features_df = st.session_state.get("features_df")
        if features_df is None:
            st.error("⚠️ 불러온 데이터가 없습니다. 위에서 데이터셋을 먼저 선택·불러오기 하세요.")
            st.stop()
        log(f"📂 저장된 데이터 사용: {len(features_df):,}명")
        _set_phase(0.60, 0.95)

    elif data_mode == DATA_MODE_RAW:
        # ── 다운로드 Raw 모드: 추출 건너뛰고 선택 파일에서 피처 계산 ─────────
        from hana_app.core.memory_guard import MemoryGuard, MemoryLimitExceeded
        from hana_app.core.ml_runner import (
            InsufficientDiskSpaceError,
            load_features_from_parquet, _duckdb_available, _duck_con,
        )

        _raw_paths = [Path(p) for p in (st.session_state.get("raw_selected_paths") or [])]
        _raw_paths = [p for p in _raw_paths if p.exists()]
        if not _raw_paths:
            st.error(
                "⚠️ 선택된 Raw 파일이 없습니다(또는 경로가 사라짐). "
                "위 **📥 다운로드 받은 Raw 데이터** 에서 파일/기간을 다시 선택하세요."
            )
            st.stop()

        _mem_guard = MemoryGuard(limit_mb=memory_limit_mb, on_warning=log, on_critical=log)
        log(f"MemoryGuard 활성화: {_mem_guard.info()}")

        # 인구통계(나이/성별) 파일을 정규 경로로 확보 — age/sex_m silent 저하 방지
        _raw_dir_for_run = Path(st.session_state.get("raw_data_dir", str(_raw_paths[0].parent)))
        _demo_status = _ensure_demographics_from_raw(_raw_dir_for_run, log=log)
        if _demo_status == "missing":
            st.warning("⚠️ 인구통계 파일이 없어 나이·성별 없이 피처를 계산합니다.")
        elif _demo_status.startswith("error:"):
            st.warning(f"⚠️ 인구통계 파일 확보 실패 — 나이·성별 없이 진행할 수 있습니다 ({_demo_status[6:]}).")

        st.subheader("⚙️ 피처 계산 중 (다운로드 Raw)...")
        _set_phase(0.10, 0.60)
        log(
            f"Raw 파일 {len(_raw_paths)}개 → 피처 계산 시작 "
            f"(window={window_days}일, 다재약물≥{poly_threshold}종, 배치 {int(patient_batch):,}명)"
        )

        try:
            features_list = build_patient_features_from_parquet(
                parquet_paths=_raw_paths,
                window_days=window_days,
                poly_threshold=poly_threshold,
                patient_batch_size=int(patient_batch),
                memory_limit_mb=memory_limit_mb,
                progress_cb=log,
                progress_pct_cb=update_pct,
                guard=_mem_guard,
            )
        except InsufficientDiskSpaceError as _de:
            st.error(
                "⚠️ **임시 디스크 공간 부족**\n\n"
                f"{_de}\n\n"
                "여유 있는 드라이브를 `HANA_FEAT_TMP` 환경변수로 지정 후 다시 시도하세요 "
                r"(예: `set HANA_FEAT_TMP=D:\hana_tmp`)."
            )
            st.stop()
        except (MemoryError, MemoryLimitExceeded) as _me:
            _rss = getattr(_me, 'rss_mb', '?')
            st.error(
                "⚠️ **메모리 한도 도달**\n\n"
                "피처 계산 중 RAM 한도에 도달하여 안전하게 중단되었습니다.\n\n"
                "**해결 방법:**\n"
                "- 👥 환자 배치 크기를 줄이세요\n"
                "- 🧠 RAM 사용 한도를 높이세요\n"
                "- 📅 선택한 Raw 파일(기간)을 줄이세요\n"
                f"- 현재 RAM: {_rss} MB / 한도: {memory_limit_mb:,} MB / 배치: {int(patient_batch):,}명"
            )
            st.stop()
        except Exception as e:
            st.error("❌ Raw 피처 계산 실패")
            st.exception(e)
            st.stop()

        _features_parquet_paths = features_list  # list[Path] (피처 Parquet)
        features_df = None
        if not _features_parquet_paths:
            st.error(
                f"⚠️ 다재약물 기준({poly_threshold}종 이상) 충족 환자가 없습니다. "
                "선택 파일(기간)이나 약물 기준을 조정하세요."
            )
            st.stop()

        # 위험도 분포 (전체 로드 없이 DuckDB 집계)
        _fp_list = ", ".join(f"'{Path(p).as_posix()}'" for p in _features_parquet_paths)
        if _duckdb_available():
            with _duck_con(memory_limit_mb=max(256, memory_limit_mb // 4)) as _con:
                _total_feat = _con.execute(
                    f"SELECT COUNT(*) FROM read_parquet([{_fp_list}])"
                ).fetchone()[0]
                _risk_dist_df = _con.execute(
                    f"SELECT risk_level, COUNT(*) AS cnt FROM read_parquet([{_fp_list}]) GROUP BY risk_level"
                ).df()
        else:
            from collections import Counter as _Counter
            _risk_cnt = _Counter()
            for _fp in _features_parquet_paths:
                _chunk = pd.read_parquet(_fp, columns=["risk_level"])
                _risk_cnt.update(_chunk["risk_level"].value_counts().to_dict())
                del _chunk
            _total_feat = sum(_risk_cnt.values())
            _risk_dist_df = pd.DataFrame([{"risk_level": k, "cnt": v} for k, v in _risk_cnt.items()])

        progress_bar.progress(0.60, text="피처 계산 완료")
        st.success(
            f"✅ 피처 완료 (디스크 기반): {_total_feat:,}명 / "
            f"Parquet {len(_features_parquet_paths)}개 파일"
        )
        _rlevels = list(_risk_dist_df["risk_level"])
        risk_cols = st.columns(4)
        for _, row in _risk_dist_df.iterrows():
            level = row["risk_level"]
            cnt = int(row["cnt"])
            emoji = {"Red": "🔴", "Yellow": "🟡", "Green": "🟢", "Normal": "⚪"}.get(level, "")
            risk_cols[_rlevels.index(level) % 4].metric(f"{emoji} {level}", f"{cnt:,}명")

        # ── 피처 데이터 저장 (HANA 없이 재사용) ──────────────────────────────
        st.markdown("---")
        st.markdown("**💾 추출된 피처 데이터 저장** — 나중에 '저장된 데이터'로 재사용할 수 있습니다.")
        _rsave_col, _ = st.columns([2, 3])
        with _rsave_col:
            if st.button("💾 피처 데이터를 파일로 저장", use_container_width=True, key="raw_save_feat"):
                _save_df = load_features_from_parquet(
                    _features_parquet_paths, memory_limit_mb=memory_limit_mb,
                ) if _duckdb_available() else pd.concat(
                    [pd.read_parquet(p) for p in _features_parquet_paths], ignore_index=True,
                )
                meta = {
                    "source": "raw_download",
                    "raw_dir": str(_raw_dir_for_run),
                    "raw_files": len(_raw_paths),
                    "window_days": window_days,
                    "poly_threshold": poly_threshold,
                    "total_patients": len(_save_df),
                    "saved_at": datetime.now().isoformat(),
                }
                saved_path = _save_dataset(_save_df, meta)
                del _save_df
                st.success(
                    f"✅ 저장 완료: `{saved_path.name}`  "
                    f"({saved_path.stat().st_size/1024/1024:.1f} MB)"
                )

    else:
        # ── 데이터 소스 사전 검증 ──────────────────────────────────────────
        if using_hana:
            if not (st.session_state.get("connected") and conn.is_connected()):
                st.error("⚠️ HANA DB에 연결되지 않았습니다. 1단계 연결설정 페이지를 먼저 완료하세요.")
                st.stop()
        else:
            sas_folder = cfg["sas"].get("folder", "")
            sas_files  = cfg["sas"].get("files", {})
            if not sas_folder or not Path(sas_folder).exists():
                st.error("⚠️ SAS 폴더가 설정되지 않았습니다.")
                st.stop()
            missing = [k for k in ("t20", "t30", "t60") if not sas_files.get(k)]
            if missing:
                st.error(f"⚠️ 학습에 필요한 SAS 파일 미설정: {', '.join(missing).upper()}")
                st.stop()

        # ── MemoryGuard 생성 ──────────────────────────────────────────────
        from hana_app.core.memory_guard import MemoryGuard, MemoryLimitExceeded
        _mem_guard = MemoryGuard(
            limit_mb=memory_limit_mb,
            on_warning=log,
            on_critical=log,
        )
        log(f"MemoryGuard 활성화: {_mem_guard.info()}")

        # ── 1단계: 데이터 추출 ─────────────────────────────────────────────
        st.subheader("📥 데이터 추출 중...")
        src_label = "HANA" if using_hana else "SAS"
        _set_phase(0.0, 0.35)
        log(f"[{year_from}/{month_from}~{year_to}/{month_to}] {src_label}에서 처방 데이터 추출 시작")

        # eligibility 테이블 설정 병합 (나이 계산용)
        _table_cfg = dict(cfg["tables"])
        if use_eligibility:
            _elig = cfg.get("analysis_dbs", {}).get("eligibility", {})
            if _elig:
                _table_cfg["eligibility"] = {
                    "schema": _elig.get("schema", "NHISBDA"),
                    "table": _elig.get("table", "HHDV_DSES_YY"),
                }

        extractor = (
            HANAExtractor(conn=conn, table_cfg=_table_cfg, col_cfg=cfg["columns"])
            if using_hana
            else SASExtractor(sas_cfg=cfg["sas"], col_cfg=cfg["columns"])
        )

        # ── Step 0: ICD-10 질환 필터 → 질환 환자 ID 추출 ────────────────
        _disease_pids: list[str] | None = None
        _icd10_prefixes = _parse_icd10(icd10_input) if use_disease_filter else []
        if use_disease_filter:
            if not _icd10_prefixes:
                st.error("⚠️ 질환 필터가 활성화되었지만 ICD-10 코드가 입력되지 않았습니다.")
                st.stop()
            st.subheader("🩺 ICD-10 질환 필터 적용 중...")
            try:
                from hana_app.core.hana_etl import _yyyymm_range as _ym_range, _shift_yyyymm as _shift_ym
                _buf = max(1, (int(buffer_before_days) + 29) // 30)
                _qstart = _shift_ym(f"{year_from}{month_from}", -_buf) if _buf else f"{year_from}{month_from}"
                _bufA = max(1, (int(buffer_after_days) + 29) // 30) if int(buffer_after_days) > 0 else 0
                _qend = _shift_ym(f"{year_to}{month_to}", _bufA) if _bufA else f"{year_to}{month_to}"
                _full_yyyymm = _ym_range(_qstart[:4], _qstart[4:], _qend[:4], _qend[4:])

                log(f"T40 ICD-10 조회: {', '.join(_icd10_prefixes)} (하위 코드 포함, {len(_full_yyyymm)}개월)")
                _disease_pids = extractor.fetch_patients_by_icd10(
                    icd10_prefixes=_icd10_prefixes,
                    yyyymm_list=_full_yyyymm,
                    progress_cb=log,
                )
                if not _disease_pids:
                    st.error(
                        f"⚠️ ICD-10 코드 {', '.join(_icd10_prefixes)}에 해당하는 환자가 없습니다. "
                        "코드 또는 추출 기간을 확인하세요."
                    )
                    st.stop()
                st.success(f"✅ 질환 필터: {', '.join(_icd10_prefixes)} → {len(_disease_pids):,}명")
                log(f"질환 환자 {len(_disease_pids):,}명 추출 완료")
            except Exception as e:
                st.error(f"❌ ICD-10 질환 필터 실패: {e}")
                st.exception(e)
                st.stop()

        # ── Step 1: 사전 층화 샘플링 (자격DB + 질환 필터 교집합) ─────────
        _pre_sampled_pids: list[str] | None = None
        if use_eligibility and use_pre_sampling:
            st.subheader("🔍 사전 층화 샘플링 중...")
            try:
                from hana_app.core.ml_runner import stratify_and_sample_patients, save_demographics
                c_elig = cfg.get("columns", {}).get("eligibility", {})

                if _disease_pids is not None:
                    # 질환 환자만 자격DB 조회 → 메모리 효율적
                    log(
                        f"자격DB 조회 (질환 환자 {len(_disease_pids):,}명 대상, "
                        f"STD_YYYY={reference_year})..."
                    )
                    elig_df = extractor.fetch_eligibility_demographics(
                        patient_ids=_disease_pids,
                        std_year=str(int(reference_year)),
                        addr_digits=int(addr_digits),
                        progress_cb=log,
                    )
                    # fetch_eligibility_demographics → dict 반환, DataFrame으로 변환
                    import pandas as _epd
                    pid_col_  = c_elig.get("patient_id",   "INDI_DSCM_NO")
                    byear_col_= c_elig.get("byear",        "BYEAR")
                    sex_col_  = c_elig.get("sex_type",     "SEX_TYPE")
                    addr_col_ = c_elig.get("rvsn_addr_cd", "RVSN_ADDR_CD")
                    elig_df = _epd.DataFrame([
                        {pid_col_: pid, byear_col_: d["byear"],
                         sex_col_: d["sex_type"], addr_col_: d["addr_cd"]}
                        for pid, d in elig_df.items()
                    ])
                else:
                    # 전체 자격DB 조회 — DB/SAS에서 seed 기반 quota 추출
                    log(
                        f"자격DB 인구통계 조회 중 (STD_YYYY={reference_year}, "
                        f"목표 {int(pre_sample_size):,}명, seed={int(pre_sample_seed)})..."
                    )
                    elig_df = extractor.fetch_eligibility_for_sampling(
                        std_year=str(int(reference_year)),
                        addr_digits=int(addr_digits),
                        sample_size=int(pre_sample_size),
                        seed=int(pre_sample_seed),
                        progress_cb=log,
                    )

                if elig_df.empty:
                    st.warning("⚠️ 자격DB에서 데이터를 가져오지 못했습니다. 전체 환자 대상으로 계속합니다.")
                    _pre_sampled_pids = _disease_pids  # 질환 필터만 적용
                else:
                    log(f"층화 샘플링: {len(elig_df):,}명 → {int(pre_sample_size):,}명 추출 중...")
                    _pre_sampled_pids, _pre_demo, _strata_sum = stratify_and_sample_patients(
                        elig_df=elig_df,
                        sample_size=int(pre_sample_size),
                        reference_year=int(reference_year),
                        seed=int(pre_sample_seed),
                        pid_col=c_elig.get("patient_id",   "INDI_DSCM_NO"),
                        byear_col=c_elig.get("byear",      "BYEAR"),
                        sex_col=c_elig.get("sex_type",      "SEX_TYPE"),
                        addr_col=c_elig.get("rvsn_addr_cd","RVSN_ADDR_CD"),
                        addr_digits=int(addr_digits),
                    )
                    del elig_df
                    save_demographics(_pre_demo, reference_year=int(reference_year))
                    del _pre_demo
                    log(f"✅ 사전 샘플링 완료: {len(_pre_sampled_pids):,}명 ({len(_strata_sum)}개 층)")
                    with st.expander("층별 샘플 수"):
                        import pandas as _stpd
                        st.dataframe(_stpd.DataFrame(
                            {"층": list(_strata_sum.keys()), "환자 수": list(_strata_sum.values())}
                        ).sort_values("환자 수", ascending=False), use_container_width=True)
            except Exception as e:
                log(f"사전 샘플링 오류: {e}")
                if _disease_pids is None:
                    st.error(
                        f"❌ 사전 층화 샘플링 실패: {e}\n\n"
                        "⚠️ 질환 필터가 비활성화된 상태에서 샘플링이 실패하여, "
                        "전체 환자 대상의 대규모 데이터 추출이 시도될 위험이 있습니다. "
                        "메모리 초과 및 세션 타임아웃 방지를 위해 프로세스를 안전하게 중단합니다. "
                        "샘플 크기(pre_sample_size) 또는 자격DB 상태를 재검토해 주십시오."
                    )
                    st.stop()
                else:
                    st.warning(f"⚠️ 사전 층화 샘플링 실패: {e}. 질환 필터 적용 환자 대상으로 계속합니다.")
                    _pre_sampled_pids = _disease_pids  # 질환 필터는 유지


        elif _disease_pids is not None:
            # 샘플링 없이 질환 필터만 적용
            _pre_sampled_pids = _disease_pids

        if use_chunked:
            # ── 청크 모드: 월별 추출 -> Parquet 저장 -> 배치 피처 계산 ──
            try:
                parquet_paths, stats = extractor.extract_prescriptions_chunked(
                    year_from=year_from, month_from=month_from,
                    year_to=year_to,     month_to=month_to,
                    chunk_months=int(chunk_months),
                    chunk_unit=chunk_unit,
                    chunk_days=int(chunk_days),
                    window_days=window_days,
                    poly_threshold=poly_threshold,
                    buffer_days=int(buffer_before_days),
                    buffer_after_days=int(buffer_after_days),
                    memory_limit_mb=memory_limit_mb,
                    progress_cb=log,
                    patient_ids=_pre_sampled_pids,
                )
            except (MemoryError, MemoryLimitExceeded) as _me:
                _rss = getattr(_me, 'rss_mb', '?')
                st.error(
                    "⚠️ **메모리 한도 도달**\n\n"
                    "데이터 추출 중 RAM 한도에 도달하여 안전하게 중단되었습니다.\n\n"
                    "**해결 방법:**\n"
                    "- 📅 청크 크기를 줄이세요 (1개월 → 일 단위)\n"
                    "- 📊 추출 기간을 줄이세요\n"
                    "- 🧠 RAM 사용 한도를 높이세요\n"
                    f"- 현재 RAM: {_rss} MB / 한도: {memory_limit_mb:,} MB"
                )
                st.stop()
            except Exception as e:
                st.error("❌ 데이터 추출 실패")
                st.exception(e)
                st.info("💡 오류가 지속되면 1번 페이지 → 🔍 테이블 검증 탭에서 재검증하세요.")
                st.stop()

            progress_bar.progress(0.35, text="추출 완료")
            st.success(f"✅ 추출 완료: {stats['total_records']:,}건 / {stats['unique_patients']:,}명")
            append_etl_log(
                period_from=f"{year_from}/{month_from}",
                period_to=f"{year_to}/{month_to}",
                row_count=stats["total_records"],
                elapsed_sec=_time.time() - _etl_start,
            )
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("T20 행 수", f"{stats['t20_rows']:,}")
            sc2.metric("T30 행 수", f"{stats['t30_rows']:,}")
            sc3.metric("T40 행 수", f"{stats.get('t40_rows', 0):,}")
            sc4.metric("T60 행 수", f"{stats['t60_rows']:,}")
            sc5.metric("처방 레코드", f"{stats['total_records']:,}")

            # ── 자격DB 인구통계 조회 (추출 완료 후, 사전 샘플링 미수행 시만) ──
            if use_eligibility and not _pre_sampled_pids:
                try:
                    from hana_app.core.hana_etl import collect_unique_patient_ids
                    from hana_app.core.ml_runner import save_demographics
                    log("처방 Parquet에서 고유 환자 ID 수집 중...")
                    _pids = collect_unique_patient_ids(parquet_paths)
                    log(f"고유 환자 {len(_pids):,}명 → 자격DB 조회 시작 (STD_YYYY={reference_year})")
                    demographics = extractor.fetch_eligibility_demographics(
                        patient_ids=_pids,
                        std_year=str(int(reference_year)),
                        addr_digits=int(addr_digits),
                        progress_cb=log,
                    )
                    if demographics:
                        save_demographics(demographics, reference_year=int(reference_year))
                        log(f"✅ 인구통계 매핑 완료: {len(demographics):,}명 (나이·성별·지역코드 {addr_digits}자리)")
                    else:
                        st.warning("⚠️ 자격DB에서 인구통계를 가져오지 못했습니다. 나이/성별/지역 없이 계속합니다.")
                except Exception as e:
                    st.warning(f"⚠️ 자격DB 조회 실패: {e}. 나이/성별/지역 없이 계속합니다.")
                    log(f"자격DB 오류: {e}")

            st.subheader("⚙️ 피처 계산 중 (배치 모드)...")
            _set_phase(0.35, 0.60)
            log(f"환자 배치 크기: {int(patient_batch):,}명씩 처리")

            try:
                features_list = build_patient_features_from_parquet(
                    parquet_paths=parquet_paths,
                    window_days=window_days,
                    poly_threshold=poly_threshold,
                    patient_batch_size=int(patient_batch),
                    memory_limit_mb=memory_limit_mb,
                    progress_cb=log,
                    progress_pct_cb=update_pct,
                    guard=_mem_guard,
                )
            except (MemoryError, MemoryLimitExceeded) as _me:
                _rss = getattr(_me, 'rss_mb', '?')
                st.error(
                    "⚠️ **메모리 한도 도달**\n\n"
                    "피처 계산 중 RAM 한도에 도달하여 안전하게 중단되었습니다.\n\n"
                    "**해결 방법:**\n"
                    "- 👥 환자 배치 크기를 줄이세요\n"
                    "- 🧠 RAM 사용 한도를 높이세요\n"
                    "- 📦 `pip install duckdb`로 디스크 스필 활성화\n"
                    f"- 현재 RAM: {_rss} MB / 한도: {memory_limit_mb:,} MB / 배치: {int(patient_batch):,}명"
                )
                st.stop()
            except Exception as e:
                st.error("❌ 데이터 추출 실패")
                st.exception(e)
                st.info("💡 오류가 지속되면 1번 페이지 → 🔍 테이블 검증 탭에서 재검증하세요.")
                st.stop()

        else:
            # ── 일반 모드: 전체 일괄 추출 ─────────────────────────────────
            try:
                records, stats = extractor.extract_prescriptions(
                    year_from=year_from, month_from=month_from,
                    year_to=year_to,     month_to=month_to,
                    window_days=window_days,
                    poly_threshold=poly_threshold,
                    buffer_days=int(buffer_before_days),
                    buffer_after_days=int(buffer_after_days),
                    progress_cb=log,
                    patient_ids=_pre_sampled_pids,
                )
            except (MemoryError, MemoryLimitExceeded) as _me:
                _rss = getattr(_me, 'rss_mb', '?')
                st.error(
                    "⚠️ **메모리 한도 도달**\n\n"
                    "일괄 추출 중 RAM 한도에 도달했습니다. **청크 모드를 활성화**하세요.\n"
                    "- 💾 위의 '청크 모드 (대용량 데이터)' 토글을 켜세요\n"
                    "- 🧠 RAM 사용 한도를 확인하세요\n"
                    f"- 현재 RAM: {_rss} MB / 한도: {memory_limit_mb:,} MB"
                )
                st.stop()
            except Exception as e:
                st.error("❌ 데이터 추출 실패")
                st.exception(e)
                st.info("💡 오류가 지속되면 1번 페이지 → 🔍 테이블 검증 탭에서 재검증하세요.")
                st.stop()

            # records는 session_state에 저장하지 않음 (메모리 절약)
            progress_bar.progress(0.35, text="추출 완료")
            st.success(f"✅ 추출 완료: {stats['total_records']:,}건 / {stats['unique_patients']:,}명")
            append_etl_log(
                period_from=f"{year_from}/{month_from}",
                period_to=f"{year_to}/{month_to}",
                row_count=stats["total_records"],
                elapsed_sec=_time.time() - _etl_start,
            )
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("T20 행 수", f"{stats['t20_rows']:,}")
            sc2.metric("T30 행 수", f"{stats['t30_rows']:,}")
            sc3.metric("T40 행 수", f"{stats.get('t40_rows', 0):,}")
            sc4.metric("T60 행 수", f"{stats['t60_rows']:,}")
            sc5.metric("처방 레코드", f"{stats['total_records']:,}")

            # ── 자격DB 인구통계 조회 (추출 완료 후, 사전 샘플링 미수행 시만) ──
            if use_eligibility and not _pre_sampled_pids:
                try:
                    from hana_app.core.ml_runner import save_demographics
                    _pids = list({r.patient_id for r in records})
                    log(f"고유 환자 {len(_pids):,}명 → 자격DB 조회 시작 (STD_YYYY={reference_year})")
                    demographics = extractor.fetch_eligibility_demographics(
                        patient_ids=_pids,
                        std_year=str(int(reference_year)),
                        addr_digits=int(addr_digits),
                        progress_cb=log,
                    )
                    if demographics:
                        save_demographics(demographics, reference_year=int(reference_year))
                        log(f"✅ 인구통계 매핑 완료: {len(demographics):,}명 (나이·성별·지역코드 {addr_digits}자리)")
                    else:
                        st.warning("⚠️ 자격DB에서 인구통계를 가져오지 못했습니다. 나이/성별/지역 없이 계속합니다.")
                except Exception as e:
                    st.warning(f"⚠️ 자격DB 조회 실패: {e}. 나이/성별/지역 없이 계속합니다.")
                    log(f"자격DB 오류: {e}")

            st.subheader("⚙️ 피처 계산 중...")
            _set_phase(0.35, 0.60)
            log("다재약물 환자 피처 계산 시작...")

            try:
                features_list = build_patient_features(
                    records=records,
                    window_days=window_days,
                    poly_threshold=poly_threshold,
                    progress_cb=log,
                    progress_pct_cb=update_pct,
                    guard=_mem_guard,
                )
            except (MemoryError, MemoryLimitExceeded) as _me:
                _rss = getattr(_me, 'rss_mb', '?')
                st.error(
                    "⚠️ **메모리 한도 도달**\n\n"
                    "피처 계산 중 RAM 한도에 도달했습니다. **청크 모드를 활성화**하세요.\n"
                    "- 💾 위의 '청크 모드 (대용량 데이터)' 토글을 켜세요\n"
                    "- 📦 `pip install duckdb`로 디스크 스필 활성화\n"
                    f"- 현재 RAM: {_rss} MB / 한도: {memory_limit_mb:,} MB"
                )
                st.stop()
            except Exception as e:
                st.error("❌ 데이터 추출 실패")
                st.exception(e)
                st.info("💡 오류가 지속되면 1번 페이지 → 🔍 테이블 검증 탭에서 재검증하세요.")
                st.stop()

        # ── 결과 처리: 청크 모드 → Parquet 경로, 일반 모드 → PatientFeatures 리스트
        _features_parquet_paths = None  # Parquet 기반 (대용량)
        features_df = None              # DataFrame 기반 (소량/호환)

        if use_chunked:
            # features_list는 실제로 list[Path] (Parquet 경로들)
            _features_parquet_paths = features_list
            if not _features_parquet_paths:
                st.error(
                    f"⚠️ 다재약물 기준({poly_threshold}종 이상) 충족 환자가 없습니다. "
                    "추출 기간이나 약물 기준을 조정하세요."
                )
                st.stop()

            # DuckDB로 위험도 분포만 조회 (전체 로드 안 함)
            from hana_app.core.ml_runner import load_features_from_parquet, _duckdb_available, _duck_con
            _fp_list = ", ".join(f"'{Path(p).as_posix()}'" for p in _features_parquet_paths)
            if _duckdb_available():
                with _duck_con(memory_limit_mb=max(256, memory_limit_mb // 4)) as _con:
                    _total_feat = _con.execute(f"SELECT COUNT(*) FROM read_parquet([{_fp_list}])").fetchone()[0]
                    _risk_dist_df = _con.execute(
                        f"SELECT risk_level, COUNT(*) AS cnt FROM read_parquet([{_fp_list}]) GROUP BY risk_level"
                    ).df()
            else:
                from collections import Counter as _Counter
                _risk_cnt = _Counter()
                for _fp in _features_parquet_paths:
                    _chunk = pd.read_parquet(_fp, columns=["risk_level"])
                    _risk_cnt.update(_chunk["risk_level"].value_counts().to_dict())
                    del _chunk
                _total_feat = sum(_risk_cnt.values())
                _risk_dist_df = pd.DataFrame(
                    [{"risk_level": k, "cnt": v} for k, v in _risk_cnt.items()]
                )

            progress_bar.progress(0.60, text="피처 계산 완료")
            st.success(f"✅ 피처 완료 (디스크 기반): {_total_feat:,}명 / Parquet {len(_features_parquet_paths)}개 파일")
            risk_cols = st.columns(4)
            for _, row in _risk_dist_df.iterrows():
                level = row["risk_level"]
                cnt = int(row["cnt"])
                emoji = {"Red": "🔴", "Yellow": "🟡", "Green": "🟢", "Normal": "⚪"}.get(level, "")
                risk_cols[list(_risk_dist_df["risk_level"]).index(level) % 4].metric(f"{emoji} {level}", f"{cnt:,}명")

        else:
            # 일반 모드: features_list는 list[PatientFeatures]
            if not features_list:
                st.error(
                    f"⚠️ 다재약물 기준({poly_threshold}종 이상) 충족 환자가 없습니다. "
                    "추출 기간이나 약물 기준을 조정하세요."
                )
                st.stop()

            features_df = features_to_dataframe(features_list)
            del features_list  # 리스트 해제
            # 이전 features_df 명시적 해제 (메모리 절약)
            if st.session_state.get("features_df") is not None:
                del st.session_state.features_df
                import gc as _gc; _gc.collect()
            st.session_state.features_df = features_df
            progress_bar.progress(0.60, text="피처 계산 완료")

            risk_dist = features_df["risk_level"].value_counts()
            st.success(f"✅ 피처 완료: {len(features_df):,}명")
            risk_cols = st.columns(4)
            for col_r, (level, cnt) in zip(risk_cols, risk_dist.items()):
                emoji = {"Red": "🔴", "Yellow": "🟡", "Green": "🟢", "Normal": "⚪"}.get(level, "")
                col_r.metric(f"{emoji} {level}", f"{cnt:,}명")

        # ── 데이터 저장 버튼 ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**💾 추출된 피처 데이터 저장** — 나중에 HANA 없이 재사용할 수 있습니다.")
        save_col, _ = st.columns([2, 3])
        with save_col:
            if st.button("💾 피처 데이터를 파일로 저장", use_container_width=True):
                if _features_parquet_paths:
                    # 디스크 기반: DuckDB로 합쳐서 저장
                    _save_df = load_features_from_parquet(
                        _features_parquet_paths, memory_limit_mb=memory_limit_mb,
                    ) if _duckdb_available() else pd.concat(
                        [pd.read_parquet(p) for p in _features_parquet_paths], ignore_index=True,
                    )
                else:
                    _save_df = features_df
                meta = {
                    "year_from": year_from, "month_from": month_from,
                    "year_to":   year_to,   "month_to":   month_to,
                    "window_days":    window_days,
                    "poly_threshold": poly_threshold,
                    "total_patients": len(_save_df),
                    "source":         "hana" if using_hana else "sas",
                    "saved_at":       datetime.now().isoformat(),
                }
                saved_path = _save_dataset(_save_df, meta)
                del _save_df
                st.success(f"✅ 저장 완료: `{saved_path.name}`  ({saved_path.stat().st_size/1024/1024:.1f} MB)")

    # ── 현재 사용할 데이터 소스 확정 ─────────────────────────────────
    # 우선순위: Parquet 경로 (디스크 기반) > DataFrame (메모리)
    _train_parquet = None
    _train_df = None

    if data_mode == DATA_MODE_SAVED:
        _train_df = st.session_state.get("features_df")
        if _train_df is None:
            st.stop()
    elif _features_parquet_paths:
        _train_parquet = _features_parquet_paths  # 디스크 기반
    elif features_df is not None:
        _train_df = features_df
    else:
        st.stop()

    # ── 3단계: 모델 학습 ─────────────────────────────────────────────────────
    if target == "hierarchical":
        # ── 계층 분류 전용 경로 ────────────────────────────────────────────
        from hana_app.core.hierarchical_runner import train_hierarchical, predict_risk
        from hana_app.core.ml_runner import load_features_from_parquet, _duckdb_available

        st.subheader("🤖 계층 분류 학습 중 (Stage 1 + Stage 2)...")
        _set_phase(0.60, 0.98)
        log("계층 분류 학습 시작: Stage 1 Red 이진 + Stage 2 Yellow 6-class")

        # Parquet 기반이면 메모리에 로드 (hierarchical_runner 는 DataFrame 입력)
        if _train_parquet:
            st.warning(
                "⚠️ **계층 분류 + 대용량 Parquet**: `train_hierarchical` 은 DataFrame 전체를 메모리에 올립니다. "
                f"RAM 한도({memory_limit_mb:,} MB)가 충분한지 확인하세요. "
                "부족하면 층화 샘플링 후 저장 데이터를 사용하세요."
            )
            log(f"Parquet {len(_train_parquet)}개 파일 → DataFrame 로드 중...")
            _train_df = load_features_from_parquet(
                _train_parquet, memory_limit_mb=memory_limit_mb
            ) if _duckdb_available() else pd.concat(
                [pd.read_parquet(p) for p in _train_parquet], ignore_index=True
            )

        if "yellow_subtype" not in _train_df.columns:
            st.error(
                "⚠️ 계층 분류에 필요한 'yellow_subtype' 컬럼이 없습니다.\n\n"
                "ETL 파이프라인이 yellow_subtype 을 생성해야 합니다. "
                "데이터를 재추출하거나 다른 타겟을 선택하세요."
            )
            st.stop()

        _hier_out = (
            Path(__file__).parent.parent / "models" / "hierarchical"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        try:
            _hier_result = train_hierarchical(
                df=_train_df,
                feature_cols=selected_features,
                output_dir=_hier_out,
                recall_floor=recall_floor,
                review_recall_target=review_recall_target,
                cost_sensitive=use_cost_sensitive,
            )
        except Exception as e:
            st.error(f"❌ 계층 분류 학습 실패: {e}")
            st.exception(e)
            st.stop()

        progress_bar.progress(1.0, text="계층 분류 학습 완료!")
        status_text.success("**계층 분류 학습 완료!**")

        import json as _json
        _meta = _json.loads((_hier_out / "stage_meta.json").read_text(encoding="utf-8"))
        thresholds = _meta.get("thresholds", {})
        label_counts = _meta.get("stage2_label_counts", {})

        st.markdown("---")
        st.subheader("📊 계층 분류 학습 결과")

        _t_col1, _t_col2, _t_col3 = st.columns(3)
        _t_col1.metric("τ_red", f"{thresholds.get('tau_red', '?'):.3f}" if isinstance(thresholds.get('tau_red'), float) else "?")
        _t_col2.metric("τ_review", f"{thresholds.get('tau_review', '?'):.3f}" if isinstance(thresholds.get('tau_review'), float) else "?")
        _t_col3.metric("Y_OTHER 제외", f"{_meta.get('y_other_excluded_count', 0):,}명")

        st.markdown("**Stage 2 라벨 분포**")
        _lc_cols = st.columns(len(label_counts) or 1)
        _label_emoji = {
            "Y_MIX": "🟡", "Y_DDI_MAJOR": "🔴", "Y_DDI_MOD": "🟠",
            "Y_DUP": "🟣", "Y_FRAG": "🔵", "No_Alert": "⚪",
        }
        for _lc_col, (_lbl, _cnt) in zip(_lc_cols, label_counts.items()):
            _lc_col.metric(f"{_label_emoji.get(_lbl, '')} {_lbl}", f"{_cnt:,}명")

        with st.expander("📋 stage_meta.json 전체"):
            st.json(_meta)

        st.success(
            f"🏆 계층 분류 모델 저장: `{_hier_out}`\n\n"
            "**4단계 결과분석** 페이지에서 Yellow 서브타입 분포를 확인하세요."
        )
        _hier_last = {
            "model_name": "hierarchical",
            "target": "hierarchical",
            "model_path": str(_hier_out),
            "metrics": {
                "tau_red": thresholds.get("tau_red"),
                "tau_review": thresholds.get("tau_review"),
                "f1_macro": 0.0,
            },
            "meta": _meta,
        }
        st.session_state.last_result = _hier_last
        st.session_state.train_results = {"hierarchical": _hier_last}

        # predict_risk → features_df 에 red_suspect / action 컬럼 채우기
        # (page 4 의 yellow_subtype_view 가 이 컬럼을 사용)
        _feat_df = st.session_state.get("features_df") if _train_parquet else _train_df
        if _feat_df is not None and selected_features and \
                all(c in _feat_df.columns for c in selected_features):
            try:
                import joblib as _jl
                _bundle = _jl.load(_hier_out / "stage2_yellow.joblib")
                log("predict_risk: red_suspect / action 컬럼 계산 중...")
                _preds = predict_risk(
                    _feat_df[selected_features].to_numpy(),
                    stage1_model=_hier_result["stage1_model"],
                    stage2_model=_bundle["model"],
                    stage2_encoder=_hier_result["stage2_encoder"],
                    thresholds=_hier_result["thresholds"],
                    classes_present=_bundle["classes_present"],
                )
                _feat_df = _feat_df.copy()
                _feat_df["red_suspect"] = [p["red_suspect"] for p in _preds]
                _feat_df["action"] = [p["action"] for p in _preds]
                st.session_state.features_df = _feat_df
                _rs_cnt = int(_feat_df["red_suspect"].sum())
                log(f"✅ predict_risk 완료 — red_suspect: {_rs_cnt:,}건")
                st.info(f"📌 **Red 의심 (red_suspect=True)**: {_rs_cnt:,}건 — 4단계 결과분석에서 확인 가능")
            except Exception as _pe:
                st.warning(f"⚠️ predict_risk 실패 (학습 결과는 저장됨): {_pe}")

        st.balloons()

    else:
        # ── 기존 단일/앙상블 모델 학습 (복수 모델 순차 실행) ─────────────
        _model_label = ", ".join(all_selected_models)
        st.subheader(f"🤖 모델 학습 중... ({_model_label})")

        all_results: dict = {}
        _n_models = len(all_selected_models)

        for mi, mname in enumerate(all_selected_models):
            _lo = 0.60 + mi / _n_models * 0.35
            _hi = 0.60 + (mi + 1) / _n_models * 0.35
            _set_phase(_lo, _hi)
            log(f"[{mi+1}/{_n_models}] {mname} 학습 시작 | 타겟: {target} | 피처: {len(selected_features)}개")

            try:
                result = train_model(
                    df=_train_df,
                    features_parquet=_train_parquet,
                    model_name=mname,
                    target=target,
                    params=params_map.get(mname),
                    test_size=test_size,
                    cv_folds=cv_folds,
                    sampling_size=sampling_size_actual,
                    sampling_rounds=sampling_rounds,
                    threshold_optimization=use_threshold_opt,
                    cost_sensitive=use_cost_sensitive,
                    cost_fp=cost_fp,
                    cost_fn=cost_fn,
                    progress_cb=log,
                    progress_pct_cb=update_pct,
                    gpu_memory_fraction=gpu_memory_fraction,
                    memory_limit_mb=memory_limit_mb,
                    feature_cols=selected_features,
                    guard=_mem_guard,
                    features_df=st.session_state.get("features_df"),
                )
                all_results[mname] = result
                log(f"[{mi+1}/{_n_models}] {mname} 완료 — F1={result['metrics']['f1_macro']:.4f}")
            except (MemoryError, MemoryLimitExceeded) as _me:
                _rss = getattr(_me, 'rss_mb', '?')
                st.error(
                    f"⚠️ **메모리 한도 도달** — {mname} 학습 중\n\n"
                    "처리가 안전하게 중단되었습니다.\n\n"
                    "**해결 방법:**\n"
                    "- 📊 층화 샘플링 크기를 줄이세요\n"
                    "- 🧠 RAM 사용 한도를 높이세요\n"
                    "- 🌲 트리 수(n_estimators)를 줄이세요\n"
                    f"- 현재 RAM: {_rss} MB / 한도: {memory_limit_mb:,} MB / "
                    f"샘플: {sampling_size_actual:,}명"
                )
                st.stop()
            except Exception as e:
                log(f"[{mi+1}/{_n_models}] {mname} 실패: {e}")
                st.warning(f"⚠️ {mname} 학습 실패: {e}")
                st.exception(e)

        if not all_results:
            st.error("모든 모델 학습이 실패했습니다.")
            st.stop()

        # 모델 객체를 all_results에서 제거 (디스크에 이미 저장됨, 메모리 절약)
        for _res in all_results.values():
            _res.pop("model", None)
        import gc as _gc; _gc.collect()
        st.session_state.last_result = list(all_results.values())[-1]
        st.session_state.train_results = all_results
        progress_bar.progress(1.0, text="학습 완료!")
        status_text.success(f"**{len(all_results)}개 모델 학습 완료!**")

        # ── 결과 비교 테이블 ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 모델 비교 결과")

        compare_rows = []
        for mname, res in all_results.items():
            m = res["metrics"]
            row = {
                "모델": mname,
                "Accuracy": f"{m.get('accuracy', 0):.4f}",
                "F1 (macro)": f"{m.get('f1_macro', 0):.4f}",
                "AUC": f"{m.get('roc_auc', m.get('roc_auc_ovr', 0)):.4f}",
                f"CV {cv_folds}-fold": f"{m.get('cv_mean', 0):.4f} \u00b1 {m.get('cv_std', 0):.4f}",
            }
            if use_threshold_opt and "optimal_threshold" in m:
                row["최적 임계값"] = f"{m['optimal_threshold']:.2f}"
                row["F1@최적"] = f"{m.get('f1_at_optimal', 0):.4f}"
            compare_rows.append(row)

        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

        # ── 각 모델별 상세 결과 ──────────────────────────────────────
        for mname, res in all_results.items():
            m = res["metrics"]
            with st.expander(f"📋 {mname} 상세 결과"):
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
                rc2.metric("F1 (macro)", f"{m.get('f1_macro', 0):.4f}")
                rc3.metric("AUC", f"{m.get('roc_auc', m.get('roc_auc_ovr', 0)):.4f}")
                rc4.metric("CV", f"{m.get('cv_mean', 0):.4f}")
                st.text(m.get("classification_report", "N/A"))
                st.caption(f"모델 저장: `{res.get('model_path', 'N/A')}`")

        # ── 샘플링 라운드 요약 (반복 학습 시) ────────────────────────
        for mname, res in all_results.items():
            rs = res.get("round_summary")
            if rs and rs["total_rounds"] > 1:
                with st.expander(f"📊 {mname} — {rs['total_rounds']}회 샘플링 라운드 요약"):
                    round_rows = []
                    for rr in res.get("all_rounds", []):
                        rm = rr["metrics"]
                        round_rows.append({
                            "Round": rr["sampling_round"],
                            "Seed": rr["sampling_seed"],
                            "Accuracy": f"{rm['accuracy']:.4f}",
                            "F1": f"{rm['f1_macro']:.4f}",
                            "AUC": f"{rm.get('roc_auc', rm.get('roc_auc_ovr', 0)):.4f}",
                            "CV": f"{rm['cv_mean']:.4f}",
                        })
                    st.dataframe(pd.DataFrame(round_rows), use_container_width=True, hide_index=True)
                    st.info(
                        f"F1 평균: **{rs['f1_mean']:.4f}** ± {rs['f1_std']:.4f} | "
                        f"AUC 평균: **{rs['auc_mean']:.4f}** ± {rs['auc_std']:.4f} | "
                        f"최고 Round: **{rs['best_round']}**"
                    )

        # 최고 성능 모델 표시
        _best = max(all_results.items(), key=lambda x: x[1]["metrics"].get("f1_macro", 0))
        st.success(
            f"🏆 최고 성능: **{_best[0]}** "
            f"(F1={_best[1]['metrics']['f1_macro']:.4f})\n\n"
            "**4단계 결과분석** 페이지에서 상세 결과를 확인하세요."
        )
        st.balloons()
