#!/usr/bin/env bash
# ============================================================
# 전체 시스템 통합 설치 스크립트 (Mac/Linux)
#
# 인터넷 환경이면 PyPI에서 자동 설치,
# 폐쇄망이면 packages_mac/pyXXX + hana/pyXXX 오프라인 패키지 사용
#
# 사용법:
#   bash install_all.sh                  (자동 감지, 시스템 Python)
#   bash install_all.sh --py 311         (Python 3.11 지정)
#   bash install_all.sh --venv           (가상환경 자동 생성)
#   bash install_all.sh --py 311 --venv  (버전 지정 + 가상환경)
#   bash install_all.sh --offline        (오프라인 강제)
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAC_DIR="$PROJECT_ROOT/packages_mac"
HANA_DIR="$PROJECT_ROOT/hana"
APP_REQ="$PROJECT_ROOT/hana_app/requirements.txt"

# 인자 파싱
SPECIFIC_PY=""
CREATE_VENV=false
FORCE_OFFLINE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --py) SPECIFIC_PY="$2"; shift 2 ;;
        --venv) CREATE_VENV=true; shift ;;
        --offline) FORCE_OFFLINE=true; shift ;;
        *) echo "알 수 없는 옵션: $1"; exit 1 ;;
    esac
done

# Python 바이너리 및 버전 감지
PY_BIN="${PYTHON_BIN:-python3}"
PY_VERSION=$("$PY_BIN" -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")

if [[ -n "$SPECIFIC_PY" ]]; then
    PY_VERSION="$SPECIFIC_PY"
fi

# 인터넷 연결 확인
ONLINE=false
if [[ "$FORCE_OFFLINE" == false ]]; then
    if "$PY_BIN" -c "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=3)" 2>/dev/null; then
        ONLINE=true
    fi
fi

echo "================================================"
echo " NHIS 다재약물 DDI 시스템 - 패키지 설치"
echo "================================================"
echo " Python 버전  : $PY_VERSION ($("$PY_BIN" --version))"
echo " 인터넷 연결  : $([ "$ONLINE" == true ] && echo '있음 (온라인 설치)' || echo '없음 (오프라인 설치)')"
echo ""

# 가상환경 생성
if [[ "$CREATE_VENV" == true ]]; then
    VENV_PATH="$PROJECT_ROOT/.venv_hana"
    if [[ ! -d "$VENV_PATH" ]]; then
        echo "가상환경 생성 중: $VENV_PATH"
        "$PY_BIN" -m venv "$VENV_PATH"
    fi
    PY_BIN="$VENV_PATH/bin/python"
    echo "가상환경: $VENV_PATH"
    echo ""
fi

# ── pip 업그레이드 ─────────────────────────────────────────
echo "[1/4] pip 업그레이드..."
"$PY_BIN" -m pip install --upgrade pip -q 2>/dev/null || true

if [[ "$ONLINE" == true ]]; then
    # ── 온라인 설치 ───────────────────────────────────────────
    echo ""
    echo "[2/4] PyPI에서 핵심 패키지 설치..."
    "$PY_BIN" -m pip install \
        numpy pandas pyarrow scipy scikit-learn xgboost lightgbm shap joblib \
        -q

    echo ""
    echo "[3/4] 웹앱 + HANA 패키지 설치..."
    "$PY_BIN" -m pip install -r "$APP_REQ" -q

    echo ""
    echo "[4/4] OpenMP 확인 (Mac 전용)..."
    if [[ "$(uname)" == "Darwin" ]]; then
        if ! brew list libomp &>/dev/null 2>&1; then
            echo "  libomp 설치 중 (XGBoost/LightGBM 필수)..."
            brew install libomp -q || echo "  [경고] libomp 설치 실패 — brew install libomp 를 수동 실행하세요."
        else
            echo "  libomp 이미 설치됨"
        fi
    fi
else
    # ── 오프라인 설치 ─────────────────────────────────────────
    MAC_PKG_DIR="$MAC_DIR/py${PY_VERSION}"
    HANA_PKG_DIR="$HANA_DIR/py${PY_VERSION}"

    PKG_MISSING=0
    [[ ! -d "$MAC_PKG_DIR" ]] && { echo "[경고] packages_mac/py${PY_VERSION} 없음"; PKG_MISSING=1; }
    [[ ! -d "$HANA_PKG_DIR" ]] && { echo "[경고] hana/py${PY_VERSION} 없음"; PKG_MISSING=1; }

    if [[ "$PKG_MISSING" -eq 1 ]]; then
        echo ""
        echo "인터넷 환경에서 먼저 실행하세요:"
        echo "  bash packages_mac/download.sh --py $PY_VERSION"
        exit 1
    fi

    FIND_LINKS="--find-links=$MAC_PKG_DIR --find-links=$HANA_PKG_DIR"

    echo ""
    echo "[2/4] 핵심 패키지 오프라인 설치..."
    "$PY_BIN" -m pip install --no-index $FIND_LINKS \
        numpy pandas pyarrow scipy scikit-learn xgboost lightgbm shap joblib

    echo ""
    echo "[3/4] 전체 패키지 오프라인 설치..."
    "$PY_BIN" -m pip install --no-index $FIND_LINKS --upgrade \
        -r "$MAC_DIR/requirements.txt"
    "$PY_BIN" -m pip install --no-index $FIND_LINKS --upgrade \
        -r "$HANA_DIR/requirements.txt"

    echo ""
    echo "[4/4] (오프라인 모드 — OpenMP 수동 설치 필요 시 brew install libomp)"
fi

# ── 설치 검증 ─────────────────────────────────────────────
echo ""
echo "================================================"
echo " 설치 검증"
echo "================================================"
FAIL=0
check() {
    "$PY_BIN" -c "import $1; print('  ✅ $1', $1.__version__)" 2>/dev/null || \
        { echo "  ❌ $1 [실패]"; FAIL=1; }
}

check streamlit
check pandas
check numpy
check sklearn
check xgboost
check lightgbm
check shap
check matplotlib
check plotly
"$PY_BIN" -c "import hdbcli; print('  ✅ hdbcli', hdbcli.__version__)" 2>/dev/null || \
    echo "  ⚠️  hdbcli 미설치 (HANA DB 연결 불가, SAS 파일 모드는 정상)"

echo ""
echo "================================================"
if [[ "$FAIL" -eq 0 ]]; then
    echo " ✅ 설치 완료!"
    echo ""
    echo " 웹앱 실행:"
    echo "   bash hana_app/run.sh"
    echo "   → http://localhost:8501"
else
    echo " ❌ 일부 패키지 실패 — 위 항목을 확인하세요."
fi
echo "================================================"
