"""
페이지 5: 분석 DB 관리
건강검진 / 문진 / 자격DB 등 추가 분석 데이터베이스를 활성화·설정·제거합니다.

데이터셋 구조
─────────────────────────────────────────────────────────────
• 건강검진+문진 통합 (2002–2017) : 검진 결과 + 문진이 한 테이블
• 건강검진 단독   (2018+)         : 검진 결과만 분리
• 건강문진 단독   (2018+)         : 문진만 분리, EXMDM_NO/SEQ 로 검진과 조인
• 자격 DB                         : RVSN_ADDR_CD 앞 5자리=시군구, 전체 8자리=읍면동
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hana_app.core.config import load_config, save_config, is_hana
from hana_app.core.db import get_connection

st.set_page_config(page_title="분석 DB 관리", page_icon="🗄️", layout="wide")
st.title("🗄️ 분석 DB 관리")
st.caption("건강검진 · 문진 · 자격DB 등 추가 데이터를 DDI 위험도 분석에 포함하거나 제외합니다.")

cfg  = load_config()
conn = get_connection(st.session_state)

# ── analysis_dbs 누락 시 초기화 ──────────────────────────────────────────────
_defaults: dict[str, dict[str, Any]] = {
    "checkup_integrated": {
        "enabled": False, "schema": "NHISBDA",  "table": "HHDT_HLTH_EXAM_INTG",
        "year_from": "2002", "year_to": "2017",
    },
    "checkup_2018": {
        "enabled": False, "schema": "NHISBDA",  "table": "HHDT_HLTH_EXAM",
        "year_from": "2018", "year_to": "2024",
    },
    "questionnaire_2018": {
        "enabled": False, "schema": "NHISBDA",  "table": "HHDT_HLTH_EXAM_QSTN",
        "year_from": "2018", "year_to": "2024",
    },
    "eligibility": {
        "enabled": False, "schema": "NHISBASE", "table": "HHDT_ELIG_INFO",
        "addr_level": "sgg",
    },
}
if "analysis_dbs" not in cfg:
    cfg["analysis_dbs"] = {}
for k, v in _defaults.items():
    if k not in cfg["analysis_dbs"]:
        cfg["analysis_dbs"][k] = v.copy()
    else:
        for fk, fv in v.items():
            cfg["analysis_dbs"][k].setdefault(fk, fv)

adb = cfg["analysis_dbs"]

# ── DB 메타데이터 ─────────────────────────────────────────────────────────────
DB_META = {
    "checkup_integrated": {
        "icon":   "🏥",
        "label":  "건강검진 (2002–2017) — 검진 + 문진 통합",
        "desc":   "2002–2017년 건강검진 결과와 문진이 하나의 테이블에 포함된 구버전 데이터셋.",
        "badge":  "통합본",
        "badge_color": "green",
        "key_cols":   ["EXMD_BZ_YYYY (검진사업년도)", "INDI_DSCM_NO (개인식별번호)"],
        "join_hint":  "INDI_DSCM_NO → T20/T40 INDI_DSCM_NO",
        "has_year":   True,
        "has_addr":   False,
        "warn":       None,
    },
    "checkup_2018": {
        "icon":  "🏥",
        "label": "건강검진 (2018+) — 검진 단독",
        "desc":  "2018년부터 검진 결과만 분리된 테이블. 문진과 EXMDM_NO / EXMDM_SEQ 로 조인.",
        "badge": "검진 단독",
        "badge_color": "blue",
        "key_cols":  ["EXMD_BZ_YYYY", "EXMDM_NO (검진청구번호)", "EXMDM_SEQ (검진청구순번)"],
        "join_hint": "INDI_DSCM_NO → T20/T40  |  EXMDM_NO+SEQ → 문진 2018+",
        "has_year":  True,
        "has_addr":  False,
        "warn": "⚠️ 2018+ 검진을 사용하려면 **건강문진 (2018+)** 도 함께 활성화하는 것을 권장합니다.",
    },
    "questionnaire_2018": {
        "icon":  "📋",
        "label": "건강문진 (2018+) — 문진 단독",
        "desc":  "2018년부터 분리된 문진 테이블. 과거력·가족력·흡연·음주·신체활동·노인문진 포함.",
        "badge": "문진 단독",
        "badge_color": "blue",
        "key_cols":  ["EXMD_BZ_YYYY", "EXMDM_NO (검진청구번호)", "EXMDM_SEQ (검진청구순번)"],
        "join_hint": "EXMDM_NO + EXMDM_SEQ → 건강검진 2018+",
        "has_year":  True,
        "has_addr":  False,
        "warn": "⚠️ 문진 단독은 검진 결과 없이 설문 항목만 포함됩니다. 보통 **건강검진 (2018+)** 과 함께 사용합니다.",
    },
    "eligibility": {
        "icon":  "👤",
        "label": "자격 DB",
        "desc":  "건강보험 가입자 자격 · 주소 · 소득분위 · 장애등급 정보. RVSN_ADDR_CD 로 지역 분석 가능.",
        "badge": "자격",
        "badge_color": "orange",
        "key_cols":  ["STD_YYYY (기준년도)", "INDI_DSCM_NO (개인식별번호)"],
        "join_hint": "INDI_DSCM_NO → T20/T40  |  RVSN_ADDR_CD → 지역 분석",
        "has_year":  False,
        "has_addr":  True,
        "warn": None,
    },
}

# ── 연결 상태 배너 ─────────────────────────────────────────────────────────────
if is_hana(cfg) and conn.is_connected():
    st.success(f"✅ HANA 연결됨: {cfg['connection']['host']}  —  테이블 행 수 미리보기 가능")
else:
    st.info("ℹ️ HANA 미연결 — 스키마/테이블 설정만 가능합니다. 행 수 미리보기는 연결 후 가능합니다.")

st.markdown("---")

# ── 현황 요약 ──────────────────────────────────────────────────────────────────
enabled_list = [k for k, v in adb.items() if v.get("enabled")]
if enabled_list:
    st.markdown(
        "**활성화된 분석 DB:** "
        + " · ".join(f"`{DB_META[k]['label']}`" for k in enabled_list if k in DB_META)
    )
else:
    st.warning("현재 활성화된 분석 DB가 없습니다. 아래에서 필요한 DB를 켜세요.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# DB 카드 렌더링
# ─────────────────────────────────────────────────────────────────────────────

def render_db_card(key: str) -> None:
    meta = DB_META[key]
    db   = adb[key]

    enabled = db.get("enabled", False)

    # 카드 헤더 행
    col_toggle, col_title = st.columns([1, 11])
    with col_toggle:
        new_enabled = st.toggle(
            "활성화",
            value=enabled,
            key=f"toggle_{key}",
            label_visibility="collapsed",
        )
    with col_title:
        badge_emoji = "🟢" if new_enabled else "⚪"
        st.markdown(
            f"### {meta['icon']} {meta['label']}  {badge_emoji}"
        )

    db["enabled"] = new_enabled

    # 설명 + 조인 힌트
    st.caption(meta["desc"])
    st.markdown(
        f"**키 컬럼:** {' · '.join(f'`{c}`' for c in meta['key_cols'])}  "
        f"　**조인:** `{meta['join_hint']}`"
    )

    if meta["warn"] and new_enabled:
        st.warning(meta["warn"])

    # ── 상세 설정 (활성화 시 펼쳐짐) ─────────────────────────────────────────
    with st.expander("⚙️ 테이블 설정" + (" (활성)" if new_enabled else ""), expanded=new_enabled):
        c1, c2 = st.columns(2)
        with c1:
            db["schema"] = st.text_input(
                "스키마", value=db.get("schema", ""),
                key=f"schema_{key}",
            )
        with c2:
            db["table"] = st.text_input(
                "테이블명", value=db.get("table", ""),
                key=f"table_{key}",
            )

        # 연도 필터
        if meta["has_year"]:
            st.markdown("**조회 연도 범위**")
            yc1, yc2 = st.columns(2)
            with yc1:
                db["year_from"] = st.text_input(
                    "시작 연도 (YYYY)", value=db.get("year_from", ""),
                    key=f"yfrom_{key}", max_chars=4,
                )
            with yc2:
                db["year_to"] = st.text_input(
                    "종료 연도 (YYYY)", value=db.get("year_to", ""),
                    key=f"yto_{key}", max_chars=4,
                )

        # 자격DB 전용: RVSN_ADDR_CD 지역 수준 설정
        if meta["has_addr"]:
            st.markdown("**RVSN_ADDR_CD 지역 분석 수준**")
            st.markdown(
                "> `RVSN_ADDR_CD`는 8자리 행정동 코드입니다.  \n"
                "> - **앞 5자리** → 시군구 수준  \n"
                "> - **전체 8자리** → 읍면동 수준"
            )
            addr_options = {
                "sgg":  "시군구 (앞 5자리) — RVSN_ADDR_CD[:5]",
                "dong": "읍면동 (전체 8자리) — RVSN_ADDR_CD",
            }
            current_addr = db.get("addr_level", "sgg")
            db["addr_level"] = st.radio(
                "지역 분석 수준",
                options=list(addr_options.keys()),
                format_func=lambda x: addr_options[x],
                index=list(addr_options.keys()).index(current_addr),
                key=f"addr_{key}",
                horizontal=True,
            )

            st.info(
                "💡 **시군구** 수준은 집계 단위가 넓어 분석이 안정적입니다.  \n"
                "   **읍면동** 수준은 세밀한 지역별 위험도 분포 파악에 유용하지만 소규모 지역에서 개인 식별 위험에 주의하세요."
            )

            # 자격DB 추가 분석 컬럼 안내
            with st.expander("📌 자격DB 주요 분석 컬럼 안내"):
                st.markdown("""
| 컬럼 | 설명 | 활용 |
|------|------|------|
| `RVSN_ADDR_CD` | 보정 행정동 코드 (8자리) | 지역별 위험도 지도 |
| `RVSN_ADDR_CD[:5]` | 시군구 (5자리 슬라이스) | 시군구 집계 |
| `SEX_TYPE` | 성별 | 성별 층화 분석 |
| `BYEAR` | 출생연도 | 연령 계산 |
| `SES05` | 소득분위 (5분위) | 사회경제적 요인 |
| `GAIBJA_TYPE` | 가입자구분 (직장/지역/의료급여) | 보험 유형별 분석 |
| `CMPR_DSB_GRADE` | 종합장애등급 | 장애인 취약군 |
| `MCBNF_CLSFC_CD` | 의료급여종별 | 급여 유형 |
                """)

        # HANA 연결 시 행 수 미리보기
        if is_hana(cfg) and conn.is_connected() and new_enabled:
            if st.button(f"📊 행 수 확인", key=f"rowcnt_{key}"):
                schema = db.get("schema", "")
                table  = db.get("table", "")
                if schema and table:
                    try:
                        cnt = conn.get_row_count(schema, table)
                        st.success(f"`{schema}.{table}` — 총 **{cnt:,}** 행")
                    except Exception as e:
                        st.error(f"조회 실패: {e}")
                else:
                    st.warning("스키마와 테이블명을 먼저 입력하세요.")


# ─────────────────────────────────────────────────────────────────────────────
# 건강검진 연도 구분 안내 박스
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📖 건강검진 / 문진 데이터셋 구조 변경 이력", expanded=False):
    st.markdown("""
### 연도별 데이터셋 구조

| 기간 | 검진 | 문진 | 키 컬럼 |
|------|------|------|---------|
| **2002 – 2017** | ✅ 포함 | ✅ 포함 | `EXMD_BZ_YYYY`, `INDI_DSCM_NO` |
| **2018 –** | 검진 단독 테이블 | 문진 단독 테이블 | `EXMD_BZ_YYYY`, `EXMDM_NO`, `EXMDM_SEQ` |

### 2018+ 조인 방법
```
건강검진_2018  ←→  건강문진_2018
  EXMD_BZ_YYYY = EXMD_BZ_YYYY
  EXMDM_NO     = EXMDM_NO
  EXMDM_SEQ    = EXMDM_SEQ
```

### 청구데이터(T20)와 연결
```
T20 / T40 INDI_DSCM_NO = 검진 INDI_DSCM_NO = 자격DB INDI_DSCM_NO
```

> **주의:** 건강검진은 수검자 기준이므로, 동일 연도 내 T20 청구 환자와 수검 환자가 일치하지 않을 수 있습니다.
    """)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# DB 카드 출력
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("검진 데이터")
render_db_card("checkup_integrated")
st.markdown("---")
render_db_card("checkup_2018")
st.markdown("---")
render_db_card("questionnaire_2018")

st.markdown("---")
st.subheader("자격 데이터")
render_db_card("eligibility")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 저장 버튼
# ─────────────────────────────────────────────────────────────────────────────
col_save, col_reset = st.columns([2, 1])

with col_save:
    if st.button("💾 설정 저장", type="primary", use_container_width=True):
        cfg["analysis_dbs"] = adb
        save_config(cfg)
        enabled_names = [
            DB_META[k]["label"]
            for k, v in adb.items()
            if v.get("enabled") and k in DB_META
        ]
        if enabled_names:
            st.success(
                "✅ 저장 완료!  활성 DB: " + " / ".join(enabled_names)
            )
        else:
            st.success("✅ 저장 완료! (활성화된 분석 DB 없음)")

with col_reset:
    if st.button("🔄 기본값으로 초기화", use_container_width=True):
        for k, v in _defaults.items():
            cfg["analysis_dbs"][k] = v.copy()
        save_config(cfg)
        st.warning("기본값으로 초기화되었습니다. 페이지를 새로고침하세요.")
        st.rerun()

# ── 사이드바 요약 ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗄️ 분석 DB 현황")
    for k, v in adb.items():
        meta = DB_META.get(k)
        if meta is None:
            continue
        if v.get("enabled"):
            st.success(f"✅ {meta['icon']} {meta['label']}")
        else:
            st.caption(f"⚪ {meta['icon']} {meta['label']}")
