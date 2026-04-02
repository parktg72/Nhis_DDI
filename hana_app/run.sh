#!/usr/bin/env bash
# ============================================================
# NHIS 다재약물 DDI 위험도 분류 - 웹앱 실행 스크립트 (Mac/Linux)
#
# 사용법:
#   bash hana_app/run.sh              (기본 8501 포트)
#   bash hana_app/run.sh 8080         (포트 지정)
#   bash hana_app/run.sh 8501 venv    (가상환경 사용)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT="${1:-8501}"
APP_FILE="$SCRIPT_DIR/app.py"

# Python 바이너리 결정 (가상환경 자동 감지)
VENV_PATH="$PROJECT_ROOT/.venv_hana"
if [[ -f "$VENV_PATH/bin/python" ]]; then
    PYTHON_BIN="$VENV_PATH/bin/python"
    echo "가상환경 사용: $VENV_PATH"
elif [[ "${2:-}" == "venv" ]]; then
    echo "[경고] 가상환경 없음(.venv_hana). install_all.sh --venv 를 먼저 실행하세요."
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python3"
fi

echo "=============================================="
echo " NHIS 다재약물 DDI 위험도 분류 시스템"
echo "=============================================="
echo " URL: http://localhost:$PORT"
echo " 종료: Ctrl+C"
echo ""

# Streamlit 설치 확인
if ! "$PYTHON_BIN" -c "import streamlit" 2>/dev/null; then
    echo "[오류] streamlit이 설치되지 않았습니다."
    echo "install_all.sh 를 먼저 실행하세요."
    exit 1
fi

# hdbcli 설치 확인 (경고만)
if ! "$PYTHON_BIN" -c "import hdbcli" 2>/dev/null; then
    echo "[경고] hdbcli가 설치되지 않았습니다. HANA DB 연결 기능이 제한됩니다."
    echo "hana/install.sh 를 실행하여 hdbcli를 설치하세요."
fi

"$PYTHON_BIN" -m streamlit run "$APP_FILE" \
    --server.port "$PORT" \
    --server.address localhost \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.base light \
    --theme.primaryColor "#1f77b4"
