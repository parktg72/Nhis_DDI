"""
설정 저장 / 불러오기 (JSON + OS Keychain 패스워드 저장)
비밀번호는 JSON에 저장하지 않고 OS Keychain(keyring)에만 보관.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KEYRING_SERVICE = "hana_ddi_app"

CONFIG_FILE = Path(__file__).parent.parent / "hana_config.json"

# 데이터 소스 종류
DATA_SOURCE_HANA = "hana"
DATA_SOURCE_SAS  = "sas"

DEFAULT_TABLE_COLS = {
    "t20": {
        "patient_id":      "INDI_DSCM_NO",
        "bill_no":         "CMN_KEY",
        "institution_id":  "MDCARE_SYM",
        "start_date":      "MDCARE_STRT_DT",
        "yyyymm":          "MDCARE_STRT_YYYYMM",
        "sex":             "SEX_TYPE",
        "age_id":          "SUJIN_POTM_AGE_ID",
        "institution_type":"YOYANG_CLSFC_CD",
        "prsc_drug_count": "INSOUT_PRSC_MEDI_ITM_SU",
        "total_prsc_days": "TOT_PRSC_DD_CNT",
    },
    "t30": {
        "patient_id":    "INDI_DSCM_NO",
        "bill_no":       "CMN_KEY",
        "start_date":    "MDCARE_STRT_DT",
        "yyyymm":        "MDCARE_STRT_YYYYMM",
        "drug_code":     "WK_COMPN_CD",
        "drug_code_alt": "RVSN_WK_COMPN_CD",
        "edi_code":      "MCARE_DIV_CD",
        "efmdc":         "EFMDC_CLSF_NO",
        "dose_once":     "TIME1_MDCT_CPCT",
        "dose_freq":     "DD1_MQTY_FREQ",
        "total_days":    "TOT_MCNT",
    },
    "t40": {
        "patient_id": "INDI_DSCM_NO",
        "bill_no":    "CMN_KEY",
        "start_date": "MDCARE_STRT_DT",
        "yyyymm":     "MDCARE_STRT_YYYYMM",
        "sick_code":  "MCEX_SICK_SYM",
        "sick_type":  "SICK_CLSF_TYPE",
    },
    "t60": {
        "patient_id":    "INDI_DSCM_NO",
        "bill_no":       "CMN_KEY",
        "start_date":    "MDCARE_STRT_DT",
        "yyyymm":        "MDCARE_STRT_YYYYMM",
        "drug_code":     "GNL_NM_CD",
        "drug_code_alt": "RVSN_WK_COMPN_CD",
        "edi_code":      "MCARE_DIV_CD",
        "dose_once":     "MPRSC_TIME1_TUYAK_CPCT",
        "dose_freq":     "MPRSC_DD1_TUYAK_CPCT",
        "total_days":    "TOT_MCNT",
        "sick_code":     "SICK_SYM1",
        "institution_id":"MDCARE_SYM",
    },
    "yoyang": {
        "institution_id":   "MDCARE_SYM",
        "institution_type": "YOYANG_CLSFC_CD",
        "std_year":         "STD_YYYY",
        "inst_name":        "INST_NM",
        "addr_sgg":         "ADDR_SGG_CD",
    },
    "eligibility": {
        "patient_id":   "INDI_DSCM_NO",   # 개인식별번호
        "byear":        "BYEAR",           # 출생년도 (나이 = 분석기준년 - BYEAR)
        "sex_type":     "SEX_TYPE",        # 성별 (1=남성, 2=여성)
        "std_year":     "STD_YYYY",        # 기준연도 (년도 필터용)
        "rvsn_addr_cd": "RVSN_ADDR_CD",   # 지역코드 (앞 5자리 또는 8자리)
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    # ── 데이터 소스 ─────────────────────────────────────────────
    "data_source": DATA_SOURCE_HANA,   # "hana" | "sas"

    # ── HANA DB 연결 ────────────────────────────────────────────
    "connection": {
        "host": "",
        "port": 30015,
        "user": "",
        "password_enc": "",
    },

    # ── HANA 테이블 위치 ─────────────────────────────────────────
    "tables": {
        "t20":    {"schema": "NHISBDA",  "table": "HHDT_TEMSBJ20"},
        "t30":    {"schema": "NHISBDA",  "table": "HHDT_TEMSBJ30"},
        "t40":    {"schema": "NHISBDA",  "table": "HHDT_TEMSBJ40"},
        "t60":    {"schema": "NHISBDA",  "table": "HHDT_TEMSBJ60"},
        "yoyang": {"schema": "NHISBASE", "table": "HHDT_MDCIN_GNRL_INFO"},
    },

    # ── SAS 파일 설정 ─────────────────────────────────────────────
    "sas": {
        "folder":   "",          # SAS 파일이 있는 폴더 경로
        "encoding": "cp949",     # 파일 인코딩 (cp949 / utf-8 / euc-kr)
        "chunksize": 100000,     # 청크 단위 행 수 (메모리 관리)
        "files": {               # 테이블별 파일명 (파일명만, 확장자 포함)
            "t20":    "",
            "t30":    "",
            "t40":    "",
            "t60":    "",
            "yoyang": "",
        },
    },

    # ── 컬럼 매핑 (HANA / SAS 공통) ──────────────────────────────
    "columns": DEFAULT_TABLE_COLS,

    # ── 분석 추가 DB ──────────────────────────────────────────────
    # 건강검진/문진/자격DB 등 청구데이터 외 추가 분석 DB 활성화 여부
    "analysis_dbs": {
        "checkup_integrated": {        # 건강검진+문진 통합 (2002–2017)
            "enabled": False,
            "schema": "NHISBDA",
            "table": "HHDT_HLTH_EXAM_INTG",
            "year_from": "2002",
            "year_to":   "2017",
        },
        "checkup_2018": {              # 건강검진 단독 (2018+)
            "enabled": False,
            "schema": "NHISBDA",
            "table": "HHDT_HLTH_EXAM",
            "year_from": "2018",
            "year_to":   "2024",
        },
        "questionnaire_2018": {        # 건강문진 단독 (2018+)
            "enabled": False,
            "schema": "NHISBDA",
            "table": "HHDT_HLTH_EXAM_QSTN",
            "year_from": "2018",
            "year_to":   "2024",
        },
        "eligibility": {               # 자격 DB
            "enabled": False,
            "schema": "NHISBASE",
            "table": "HHDT_ELIG_INFO",
            "addr_level": "sgg",       # "sgg" (5자리 시군구) | "dong" (8자리 읍면동)
        },
    },

    # ── 테이블 검증 상태 ──────────────────────────────────────────────────
    # Page 1 wizard 완료 시 True. 검증 DB 호스트가 변경되면 False로 초기화.
    "validated":      False,
    "validated_at":   "",    # ISO 8601 (예: "2026-04-07T09:00:00")
    "validated_host": "",    # 검증 시 사용된 HANA 호스트

    # ── 학습 설정 ─────────────────────────────────────────────────
    "training": {
        "year_from":      "2023",
        "year_to":        "2023",
        "month_from":     "01",
        "month_to":       "12",
        "window_days":    90,
        "poly_threshold": 5,
        "test_size":      0.2,
        "cv_folds":       5,
        "model":          "xgboost",
        "target":         "risk_binary",
        "memory_limit_mb": 0,  # 0 = 자동 (시스템 RAM의 75%)
        "buffer_before_days": 90,
        "buffer_after_days": 0,
        # Phase 2/3 모델 선택
        "models_phase2": ["xgboost"],
        "models_phase3": [],
        # 학습 전략
        "threshold_optimization": False,
        "cost_sensitive": False,
        "cost_fp": 1.0,
        "cost_fn": 5.0,
    },
}


def _keyring_user(cfg: dict[str, Any]) -> str:
    """Keychain 조회 키: 서비스명 + 사용자명."""
    return cfg["connection"].get("user", "") or "default"


def load_config() -> dict[str, Any]:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)
        # 누락 최상위 키 병합
        for key, val in DEFAULT_CONFIG.items():
            if key not in data:
                data[key] = val
        # sas 하위 키 병합
        if "sas" not in data:
            data["sas"] = DEFAULT_CONFIG["sas"].copy()
        else:
            for k, v in DEFAULT_CONFIG["sas"].items():
                if k not in data["sas"]:
                    data["sas"][k] = v
        # columns 하위 키 병합 (DEFAULT_TABLE_COLS 기준)
        default_cols = DEFAULT_CONFIG.get("columns", {})
        if default_cols and isinstance(data.get("columns"), dict):
            for tbl_key, col_map in default_cols.items():
                if tbl_key not in data["columns"]:
                    data["columns"][tbl_key] = col_map
        elif default_cols and "columns" not in data:
            data["columns"] = default_cols
        # 구버전 base64 비밀번호 마이그레이션 → Keychain으로 이전 후 JSON에서 제거
        legacy_enc = data.get("connection", {}).pop("password_enc", None)
        if legacy_enc:
            try:
                import base64
                plain = base64.b64decode(legacy_enc.encode()).decode()
                set_password(data, plain)
                save_config(data)  # password_enc 없는 상태로 재저장
                logger.info("구버전 base64 비밀번호를 Keychain으로 마이그레이션 완료")
            except Exception as e:
                logger.warning("비밀번호 마이그레이션 실패: %s", e)
        return data
    return json.loads(json.dumps(DEFAULT_CONFIG))   # deep copy


def save_config(cfg: dict[str, Any]) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def get_password(cfg: dict[str, Any]) -> str:
    """OS Keychain에서 비밀번호 조회. keyring 미설치 시 빈 문자열 반환."""
    try:
        import keyring
        pw = keyring.get_password(_KEYRING_SERVICE, _keyring_user(cfg))
        return pw or ""
    except Exception as e:
        logger.warning("Keychain 조회 실패 (keyring 미설치 또는 권한 오류): %s", e)
        return ""


def set_password(cfg: dict[str, Any], password: str) -> None:
    """비밀번호를 OS Keychain에 저장. JSON에는 기록하지 않음."""
    try:
        import keyring
        keyring.set_password(_KEYRING_SERVICE, _keyring_user(cfg), password)
    except Exception as e:
        logger.error("Keychain 저장 실패: %s", e)
        raise RuntimeError(
            "비밀번호를 안전하게 저장할 수 없습니다. "
            "`pip install keyring` 설치 후 재시도하세요."
        ) from e


def is_hana(cfg: dict[str, Any]) -> bool:
    return cfg.get("data_source", DATA_SOURCE_HANA) == DATA_SOURCE_HANA


def is_sas(cfg: dict[str, Any]) -> bool:
    return cfg.get("data_source", DATA_SOURCE_HANA) == DATA_SOURCE_SAS
