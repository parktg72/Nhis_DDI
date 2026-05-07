# ============================================================
# DEPRECATED — 2026-05-07
# ============================================================
# 운영 배포 대상: Windows 폐쇄망 + Python 3.12 only (park 결정 2026-05-07,
#   CLAUDE.md 의 "Python 3.12 강제" 와 정합).
# 본 Dockerfile 은 stale artifact 로 보존 — 추후 Linux 컨테이너 부활 시
#   starting point. 현 시점에는 빌드/배포 사용 X.
#
# Stale 사실 (cross-family 라운드 2026-05-07 grep 확인):
#   - FROM python:3.11-slim — 프로젝트 기준 3.12 위반
#   - COPY packages_linux/py311/ — 디렉터리 부재
#   - monitoring/ 미복사 — serving/main.py:30 의 import 와 모순
#   - 마지막 본 파일 commit 2026-04-03, 라운드 작업과 한 달+ 격차
#
# 부활 시 필요 작업 (docs/reports/2026-05-07-park-decisions.md 참조):
#   - python:3.12-slim 으로 전환
#   - packages_linux/py312/ wheelhouse 구축
#   - monitoring/ 추가 + joblib/sklearn/torch/filelock 등 누락 deps
#   - constraints-py312.txt 적용
#   - 본 헤더 DEPRECATED 마크 제거
# ============================================================
# 본래 의도 (2026-04-03 시점, 참고용):
# DDI 위험도 분류 API 서버
# 빌드: docker build -t ddi-serving:1.0 .
# 실행: docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models ddi-serving:1.0

FROM python:3.11-slim

# ── 시스템 의존성 ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── 작업 디렉토리 ──────────────────────────────────────────────────────────
WORKDIR /app

# ── 오프라인 패키지 복사 ───────────────────────────────────────────────────
# 폐쇄망: packages_linux/py311/ 에서 설치
COPY packages_linux/py311/ /tmp/packages/

# ── 패키지 설치 ────────────────────────────────────────────────────────────
RUN pip install --no-index --find-links=/tmp/packages \
    fastapi \
    uvicorn \
    pydantic \
    pandas \
    numpy \
    pyarrow \
    pyyaml \
    requests \
    xgboost \
    lightgbm \
    && rm -rf /tmp/packages

# ── 소스 코드 복사 ─────────────────────────────────────────────────────────
COPY rules/        /app/rules/
COPY scripts/etl/  /app/scripts/etl/
COPY scripts/features/ /app/scripts/features/
COPY serving/      /app/serving/
COPY config/       /app/config/

# ── 데이터 마운트 포인트 ───────────────────────────────────────────────────
# 실행 시 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models 로 마운트
RUN mkdir -p /app/data /app/models

# ── 비권한 사용자 (보안) ─────────────────────────────────────────────────────
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app/data /app/models
USER appuser

# ── 환경변수 기본값 ────────────────────────────────────────────────────────
ENV DDI_MATRIX_PATH=/app/data/processed/ddi_matrix_final.parquet \
    DRUG_INDEX_PATH=/app/data/processed/drug_name_index.parquet \
    CYP_MATRIX_PATH=/app/data/processed/cyp_matrix.parquet \
    MODEL_DIR=/app/models \
    LOG_LEVEL=INFO \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# ── 포트 노출 ──────────────────────────────────────────────────────────────
EXPOSE 8000

# ── 헬스체크 ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── 실행 ───────────────────────────────────────────────────────────────────
CMD ["uvicorn", "serving.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--timeout-keep-alive", "30"]
