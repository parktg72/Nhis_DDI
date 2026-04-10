# Page 4/6 개선 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Page 4 무결성 버그 5종 수정 + Page 6을 HANA 데스크탑 앱 맥락으로 전면 재작성 (상태 요약 바 + 4탭)

**Architecture:** (1) etl_logger 모듈 신규 — ETL 이력을 jsonl 파일에 영속 저장. (2) ml_runner._save_result 확장 — risk_summary/roc_curve를 result JSON에 포함. (3) Page 3에 etl_logger 호출 삽입. (4) Page 4 버그 수정 + ROC 탭 추가. (5) Page 6 전면 재작성 — Docker 의존 제거, 상태 요약 바, HANA/ETL/모델/시스템 4탭.

**Tech Stack:** Python 3.12, Streamlit, pandas, plotly, sklearn(roc_curve), pytest

---

## 파일 변경 범위

| 파일 | 유형 |
|------|------|
| `hana_app/core/etl_logger.py` | 신규 |
| `tests/test_hana_app/test_etl_logger.py` | 신규 |
| `hana_app/core/ml_runner.py` | 수정 — `_save_result()` + `train_model()` |
| `hana_app/pages/3_🤖_모델_학습.py` | 수정 — ETL 완료 2곳에 log 삽입 |
| `hana_app/pages/4_📊_결과_분석.py` | 수정 — P4-A~E 버그 + ROC 탭 |
| `hana_app/pages/6_📊_모니터링.py` | 전면 재작성 |

---

## Task 1: etl_logger.py — ETL 이력 영속 저장 모듈

**Files:**
- Create: `hana_app/core/etl_logger.py`
- Test: `tests/test_hana_app/test_etl_logger.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_hana_app/test_etl_logger.py` 생성:

```python
"""hana_app/core/etl_logger.py 단위 테스트."""
import json
import pytest
from pathlib import Path


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "etl_log.jsonl"


def test_append_creates_file(log_path, monkeypatch):
    """append_etl_log() 호출 시 파일이 생성되고 JSON 라인이 추가됨."""
    import hana_app.core.etl_logger as m
    monkeypatch.setattr(m, "ETL_LOG_PATH", log_path)

    m.append_etl_log(
        period_from="2023/01", period_to="2023/12",
        row_count=100000, elapsed_sec=42.1,
    )
    assert log_path.exists()
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["period_from"] == "2023/01"
    assert record["row_count"] == 100000
    assert record["status"] == "ok"
    assert record["error"] == ""


def test_append_multiple_lines(log_path, monkeypatch):
    """append 두 번 호출 시 두 줄이 기록됨."""
    import hana_app.core.etl_logger as m
    monkeypatch.setattr(m, "ETL_LOG_PATH", log_path)

    m.append_etl_log("2023/01", "2023/06", 50000, 20.0)
    m.append_etl_log("2023/07", "2023/12", 55000, 22.3)

    lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2


def test_append_error_status(log_path, monkeypatch):
    """status='error' 로 기록 가능."""
    import hana_app.core.etl_logger as m
    monkeypatch.setattr(m, "ETL_LOG_PATH", log_path)

    m.append_etl_log("2023/01", "2023/12", 0, 5.0, status="error", error="connection timeout")
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["status"] == "error"
    assert "timeout" in record["error"]


def test_load_returns_empty_when_no_file(tmp_path, monkeypatch):
    """파일 없으면 빈 리스트 반환."""
    import hana_app.core.etl_logger as m
    monkeypatch.setattr(m, "ETL_LOG_PATH", tmp_path / "missing.jsonl")
    assert m.load_etl_log() == []


def test_load_returns_recent_n(log_path, monkeypatch):
    """load_etl_log(n=2) 는 마지막 2건만 반환, 최신이 index 0."""
    import hana_app.core.etl_logger as m
    monkeypatch.setattr(m, "ETL_LOG_PATH", log_path)

    for i in range(5):
        m.append_etl_log(f"2023/0{i+1}", f"2023/0{i+1}", i * 1000, float(i))

    records = m.load_etl_log(n=2)
    assert len(records) == 2
    assert records[0]["period_from"] == "2023/05"   # 최신이 먼저
    assert records[1]["period_from"] == "2023/04"


def test_load_skips_malformed_lines(log_path, monkeypatch):
    """깨진 줄은 건너뜀."""
    import hana_app.core.etl_logger as m
    monkeypatch.setattr(m, "ETL_LOG_PATH", log_path)

    log_path.write_text(
        '{"ts":"t","period_from":"a","period_to":"b","row_count":1,"elapsed_sec":1.0,"status":"ok","error":""}\n'
        'NOT_JSON\n',
        encoding="utf-8",
    )
    records = m.load_etl_log()
    assert len(records) == 1
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
python3 -m pytest tests/test_hana_app/test_etl_logger.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'hana_app.core.etl_logger'`

- [ ] **Step 3: etl_logger.py 구현**

`hana_app/core/etl_logger.py` 생성:

```python
"""ETL 실행 이력을 JSONL 파일에 영속 저장하는 헬퍼."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ETL_LOG_PATH = Path(__file__).parent.parent / "etl_log.jsonl"


def append_etl_log(
    period_from: str,
    period_to: str,
    row_count: int,
    elapsed_sec: float,
    status: str = "ok",
    error: str = "",
) -> None:
    """ETL 실행 결과 1건을 etl_log.jsonl 에 append한다."""
    record = {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "period_from": period_from,
        "period_to": period_to,
        "row_count": row_count,
        "elapsed_sec": round(elapsed_sec, 1),
        "status": status,
        "error": error,
    }
    ETL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(ETL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("ETL 로그 기록 실패: %s", e)


def load_etl_log(n: int = 50) -> list[dict]:
    """etl_log.jsonl 에서 최근 n건을 최신순으로 반환한다."""
    if not ETL_LOG_PATH.exists():
        return []
    records: list[dict] = []
    for line in ETL_LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return list(reversed(records[-n:]))
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_hana_app/test_etl_logger.py -v
```

Expected: 6개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add hana_app/core/etl_logger.py tests/test_hana_app/test_etl_logger.py
git commit -m "feat: etl_logger.py — ETL 이력 JSONL 영속 저장 (테스트 6건)"
```

---

## Task 2: ml_runner.py — risk_summary + roc_curve 저장

**Files:**
- Modify: `hana_app/core/ml_runner.py` — `train_model()` + `_save_result()`

- [ ] **Step 1: train_model()에 features_df 파라미터 추가 + roc_curve 캡처**

`train_model()` 시그니처에 `features_df=None` 추가 (기존 마지막 파라미터 `guard=None,` 뒤에):

```python
    guard=None,
    features_df=None,   # ← 추가: 위험도 분포 요약 저장용
) -> dict[str, Any]:
```

같은 함수 내 `y_proba` 계산 직후 (기존 `metrics["roc_auc"] = ...` 바로 아래) roc_curve 캡처 추가:

```python
            if target == "risk_binary":
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
                # ROC Curve 포인트 (최대 200점으로 다운샘플)
                try:
                    from sklearn.metrics import roc_curve as _roc_curve
                    _fpr, _tpr, _ = _roc_curve(y_test, y_proba)
                    _step = max(1, len(_fpr) // 200)
                    metrics["roc_curve"] = {
                        "fpr": _fpr[::_step].tolist(),
                        "tpr": _tpr[::_step].tolist(),
                    }
                except Exception:
                    pass
```

그리고 result dict 구성부 (기존 `"sampling_size": _sample,` 바로 다음)에 features_df 추가:

```python
        result = {
            ...
            "sampling_size": _sample,
            "features_df": features_df,   # ← 추가 (None 가능)
        }
```

- [ ] **Step 2: _save_result()에 risk_summary 추출 추가**

`_save_result()` 내 `meta = {k: v ...}` 구성 직후:

```python
    meta = {
        k: v for k, v in result.items()
        if k not in ("model", "feature_importance", "features_df")  # features_df 제외
    }
    meta["model_path"] = str(model_path)
    meta["timestamp"] = ts
    if isinstance(result.get("feature_importance"), pd.DataFrame):
        meta["feature_importance"] = result["feature_importance"].to_dict("records")

    # 위험도 분포 요약 (features_df → 요약 통계만 JSON에 저장)
    _fdf = result.get("features_df")
    if _fdf is not None and not _fdf.empty:
        try:
            meta["risk_summary"] = _fdf["risk_level"].value_counts().to_dict()
            meta["drug_count_stats"] = {
                "mean": round(float(_fdf["drug_count"].mean()), 2),
                "max": int(_fdf["drug_count"].max()),
            }
            meta["ddi_means"] = {
                c: round(float(_fdf[c].mean()), 4)
                for c in ["ddi_contraindicated", "ddi_major", "ddi_moderate", "ddi_minor"]
                if c in _fdf.columns
            }
        except Exception:
            pass
```

- [ ] **Step 3: 문법 오류 확인**

```bash
python3 -c "import ast; ast.parse(open('hana_app/core/ml_runner.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: 커밋**

```bash
git add hana_app/core/ml_runner.py
git commit -m "feat: ml_runner — roc_curve + risk_summary result JSON에 저장"
```

---

## Task 3: Page 3 — ETL 완료 시 etl_logger 호출

**Files:**
- Modify: `hana_app/pages/3_🤖_모델_학습.py`
- Modify: `hana_app/core/ml_runner.py` — `train_model()` 호출부에 `features_df` 전달

Page 3에는 ETL 완료 지점이 두 곳 있다: HANA 경로(~L1021)와 SAS 경로(~L1116).

- [ ] **Step 1: import 추가**

Page 3 상단 import 블록(기존 `from hana_app.core.hana_etl import HANAExtractor` 근처)에 추가:

```python
from hana_app.core.etl_logger import append_etl_log
```

- [ ] **Step 2: ETL 시작 시각 캡처 변수 추가**

`_phase = {"lo": 0.0, "hi": 1.0, "start": _time.time()}` 바로 아래:

```python
    _etl_start: float = _time.time()   # ETL 전체 시작 시각 (로그용)
```

- [ ] **Step 3: HANA 경로 ETL 완료 지점에 log 호출 삽입**

`st.success(f"✅ 추출 완료: {stats['total_records']:,}건 / {stats['unique_patients']:,}명")` (HANA 경로) 바로 아래:

```python
            st.success(f"✅ 추출 완료: {stats['total_records']:,}건 / {stats['unique_patients']:,}명")
            append_etl_log(
                period_from=f"{year_from}/{month_from}",
                period_to=f"{year_to}/{month_to}",
                row_count=stats["total_records"],
                elapsed_sec=_time.time() - _etl_start,
            )
```

- [ ] **Step 4: SAS 경로 ETL 완료 지점에 동일하게 삽입**

두 번째 `st.success(f"✅ 추출 완료: ...")` (SAS 경로) 바로 아래:

```python
            st.success(f"✅ 추출 완료: {stats['total_records']:,}건 / {stats['unique_patients']:,}명")
            append_etl_log(
                period_from=f"{year_from}/{month_from}",
                period_to=f"{year_to}/{month_to}",
                row_count=stats["total_records"],
                elapsed_sec=_time.time() - _etl_start,
            )
```

- [ ] **Step 5: train_model() 호출부에 features_df 전달**

Page 3에서 `train_model(...)` 을 호출하는 곳(grep으로 확인)에 `features_df=st.session_state.get("features_df")` 추가:

```bash
grep -n "train_model(" "hana_app/pages/3_🤖_모델_학습.py"
```

해당 호출부 마지막 인자 뒤에:

```python
            result = train_model(
                ...
                guard=_mem_guard,
                features_df=st.session_state.get("features_df"),  # ← 추가
            )
```

- [ ] **Step 6: 문법 확인**

```bash
python3 -c "import ast; ast.parse(open('hana_app/pages/3_🤖_모델_학습.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 7: 커밋**

```bash
git add "hana_app/pages/3_🤖_모델_학습.py"
git commit -m "feat: page3 — ETL 완료 시 etl_log.jsonl append + train_model features_df 전달"
```

---

## Task 4: Page 4 — 버그 수정 + ROC 탭 추가

**Files:**
- Modify: `hana_app/pages/4_📊_결과_분석.py`

- [ ] **Step 1: P4-D — page_guards 적용**

파일 상단 import 블록(기존 `from hana_app.core.ml_runner import ...` 아래)에 추가:

```python
from hana_app.core.config import load_config, is_hana
from hana_app.core.page_guards import check_hana_validated, get_validation_error
```

`st.title("📊 학습 결과 분석")` 바로 아래:

```python
_cfg = load_config()
if is_hana(_cfg) and not check_hana_validated(_cfg):
    st.warning(get_validation_error(_cfg))
    st.stop()
```

- [ ] **Step 2: P4-A — dead code 제거**

기존:
```python
fi_data = result.get("feature_importance")
if fi_data is None and source == "저장된 결과":
    fi_data = result.get("feature_importance")  # JSON에서 이미 로드
```

수정 후:
```python
fi_data = result.get("feature_importance")
```

- [ ] **Step 3: P4-C — ROC Curve 탭 추가 (탭 목록 변경)**

기존:
```python
tab_fi, tab_cm, tab_cv, tab_dist, tab_report, tab_compare = st.tabs([
    "📈 피처 중요도",
    "🔲 혼동 행렬",
    "📉 교차검증",
    "🧮 위험도 분포",
    "📋 분류 보고서",
    "⚖️ 모델 비교",
])
```

수정 후:
```python
tab_fi, tab_cm, tab_cv, tab_roc, tab_dist, tab_report, tab_compare = st.tabs([
    "📈 피처 중요도",
    "🔲 혼동 행렬",
    "📉 교차검증",
    "📉 ROC Curve",
    "🧮 위험도 분포",
    "📋 분류 보고서",
    "⚖️ 모델 비교",
])
```

- [ ] **Step 4: ROC Curve 탭 내용 추가**

`tab_cv` 블록 끝(`st.dataframe(cv_stats, ...)` 아래)과 `tab_dist` 블록 사이에 삽입:

```python
# ── ROC Curve 탭 ──────────────────────────────────────────────────────────────
with tab_roc:
    roc_data = metrics.get("roc_curve")
    if roc_data and "fpr" in roc_data and "tpr" in roc_data:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roc_data["fpr"],
            y=roc_data["tpr"],
            mode="lines",
            name=f"ROC (AUC={metrics.get('roc_auc', 0):.4f})",
            line={"color": "steelblue", "width": 2},
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random",
            line={"color": "gray", "dash": "dash"},
        ))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
    elif metrics.get("roc_auc_ovr"):
        st.info("ROC Curve는 이진 분류(risk_binary)에서만 표시됩니다. 다중 분류의 AUC(OvR)는 핵심 지표 탭을 확인하세요.")
    else:
        st.info("ROC Curve 데이터가 없습니다. 이번 개선 이전에 저장된 결과에는 roc_curve가 포함되지 않습니다.")
```

- [ ] **Step 5: P4-B+E — 위험도 분포 탭 저장 결과 복원**

`with tab_dist:` 블록을 아래와 같이 교체:

```python
# ── 탭: 위험도 분포 ──────────────────────────────────────────────────────────
with tab_dist:
    df = st.session_state.get("features_df")
    risk_summary = result.get("risk_summary")
    drug_stats = result.get("drug_count_stats")
    ddi_means = result.get("ddi_means")

    color_map = {"Red": "#e74c3c", "Yellow": "#f39c12", "Green": "#27ae60", "Normal": "#95a5a6"}

    if df is not None:
        # 현재 세션 데이터 — 원본 DataFrame 사용
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            risk_dist = df["risk_level"].value_counts().reset_index()
            risk_dist.columns = ["위험도", "환자수"]
            fig = px.pie(risk_dist, names="위험도", values="환자수",
                         color="위험도", color_discrete_map=color_map, title="위험도 분포")
            st.plotly_chart(fig, use_container_width=True)
        with col_d2:
            fig = px.bar(risk_dist.sort_values("위험도"), x="위험도", y="환자수",
                         color="위험도", color_discrete_map=color_map,
                         title="위험도별 환자 수", text="환자수")
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("약물 수 분포")
        fig = px.histogram(df, x="drug_count", color="risk_level",
                           color_discrete_map=color_map, nbins=30, barmode="overlay",
                           title="다재약물 환자의 약물 수 분포",
                           labels={"drug_count": "약물 수", "risk_level": "위험도"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("DDI 심각도 분포")
        ddi_cols = ["ddi_contraindicated", "ddi_major", "ddi_moderate", "ddi_minor"]
        ddi_labels = ["금기", "Major", "Moderate", "Minor"]
        ddi_means_live = [df[c].mean() for c in ddi_cols]
        fig = go.Figure(go.Bar(
            x=ddi_labels, y=ddi_means_live,
            marker_color=["#e74c3c", "#e67e22", "#f1c40f", "#3498db"],
            text=[f"{v:.2f}" for v in ddi_means_live], textposition="outside",
        ))
        fig.update_layout(title="DDI 심각도별 평균 쌍 수", yaxis_title="평균 DDI 쌍 수")
        st.plotly_chart(fig, use_container_width=True)

    elif risk_summary:
        # 저장된 결과 — 요약 통계로 차트 재구성
        st.caption("저장된 결과에서 요약 통계를 불러와 표시합니다.")
        risk_dist = pd.DataFrame(
            [{"위험도": k, "환자수": v} for k, v in risk_summary.items()]
        )
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig = px.pie(risk_dist, names="위험도", values="환자수",
                         color="위험도", color_discrete_map=color_map, title="위험도 분포 (저장 요약)")
            st.plotly_chart(fig, use_container_width=True)
        with col_d2:
            fig = px.bar(risk_dist, x="위험도", y="환자수",
                         color="위험도", color_discrete_map=color_map,
                         title="위험도별 환자 수", text="환자수")
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        if ddi_means:
            st.subheader("DDI 심각도 평균 (저장 요약)")
            labels_map = {
                "ddi_contraindicated": "금기", "ddi_major": "Major",
                "ddi_moderate": "Moderate", "ddi_minor": "Minor",
            }
            ddi_labels = [labels_map[k] for k in ddi_means]
            ddi_vals = list(ddi_means.values())
            fig = go.Figure(go.Bar(
                x=ddi_labels, y=ddi_vals,
                marker_color=["#e74c3c", "#e67e22", "#f1c40f", "#3498db"],
                text=[f"{v:.2f}" for v in ddi_vals], textposition="outside",
            ))
            fig.update_layout(title="DDI 심각도별 평균 쌍 수", yaxis_title="평균 DDI 쌍 수")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("피처 데이터가 없습니다. 3단계 모델학습을 먼저 실행하세요.")
```

- [ ] **Step 6: 문법 확인**

```bash
python3 -c "import ast; ast.parse(open('hana_app/pages/4_📊_결과_분석.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 7: 커밋**

```bash
git add "hana_app/pages/4_📊_결과_분석.py"
git commit -m "fix: page4 — P4-A dead code, P4-B/E risk_summary 복원, P4-C ROC탭, P4-D page_guards"
```

---

## Task 5: Page 6 — 전면 재작성

**Files:**
- Rewrite: `hana_app/pages/6_📊_모니터링.py`

- [ ] **Step 1: 파일 전체를 아래 코드로 교체**

```python
"""
모니터링 대시보드 — Streamlit 6번 페이지

구성:
  상태 요약 바: HANA 연결 / ETL 이력 / 모델 상태 / 저장소 — 항상 표시
  Tab 1: 🔌 HANA 연결 상태
  Tab 2: 📋 ETL 실행 이력
  Tab 3: 🤖 모델 학습 이력
  Tab 4: 💾 시스템 상태
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from hana_app.core.config import load_config, is_hana
from hana_app.core.db import get_connection
from hana_app.core.etl_logger import load_etl_log
from hana_app.core.ml_runner import list_saved_results, RESULTS_DIR, MODELS_DIR

st.set_page_config(page_title="모니터링 대시보드", layout="wide")
st.title("📊 모니터링 대시보드")

# ─────────────────────────────────────────────────────────────────────────────
# 상태 계산 (탭과 독립적으로 항상 수행)
# ─────────────────────────────────────────────────────────────────────────────
cfg = load_config()

# HANA 연결 상태
_hana_mode = is_hana(cfg)
if _hana_mode:
    _conn = get_connection(st.session_state)
    _hana_connected = _conn.is_connected()
    _hana_validated = cfg.get("validated", False)
else:
    _hana_connected = None   # SAS 모드 — 해당 없음
    _hana_validated = True

# ETL 이력
_etl_records = load_etl_log(n=1)
_etl_ok = len(_etl_records) > 0

# 모델 이력
_saved_results = list_saved_results()
_model_ok = len(_saved_results) > 0

# 저장소 상태
_storage_ok = RESULTS_DIR.exists() and MODELS_DIR.exists()

# ─────────────────────────────────────────────────────────────────────────────
# 상태 요약 바
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 시스템 상태 요약")
sb1, sb2, sb3, sb4 = st.columns(4)

if _hana_mode:
    if _hana_connected and _hana_validated:
        sb1.success("🟢 HANA 연결됨")
    elif _hana_connected and not _hana_validated:
        sb1.warning("🟡 연결됨 (미검증)")
    else:
        sb1.error("🔴 HANA 연결 끊김")
else:
    sb1.info("⚪ SAS 모드")

if _etl_ok:
    _last_etl = _etl_records[0]
    sb2.success(f"🟢 ETL 완료 ({_last_etl['ts'][:10]})")
else:
    sb2.warning("🟡 ETL 이력 없음")

if _model_ok:
    _latest = _saved_results[0]
    sb3.success(f"🟢 모델 {len(_saved_results)}개 ({_latest.get('timestamp','?')[:8]})")
else:
    sb3.error("🔴 모델 없음")

if _storage_ok:
    sb4.success("🟢 저장소 정상")
else:
    sb4.error("🔴 저장소 경로 없음")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔌 HANA 연결 상태",
    "📋 ETL 실행 이력",
    "🤖 모델 학습 이력",
    "💾 시스템 상태",
])

# ─── Tab 1: HANA 연결 상태 ───────────────────────────────────────────────────
with tab1:
    if not _hana_mode:
        st.info("SAS 파일 모드에서는 HANA 연결 상태가 필요하지 않습니다.")
    else:
        conn_cfg = cfg.get("connection", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("호스트", conn_cfg.get("host", "—") or "—")
        c2.metric("포트", str(conn_cfg.get("port", "—")))
        c3.metric("사용자", conn_cfg.get("user", "—") or "—")

        st.markdown("#### 검증 상태")
        v1, v2, v3 = st.columns(3)
        v1.metric("검증 완료", "✅ 예" if cfg.get("validated") else "❌ 아니오")
        v2.metric("검증 시각", cfg.get("validated_at", "—") or "—")
        v3.metric("검증 호스트", cfg.get("validated_host", "—") or "—")

        st.markdown("#### 실시간 연결 확인")
        if st.button("🔄 연결 상태 확인", key="btn_check_conn"):
            with st.spinner("연결 확인 중..."):
                alive = _conn.is_connected()
            if alive:
                st.success("✅ HANA DB 연결 정상")
            else:
                st.error("❌ 연결 끊김")

        hana_creds = st.session_state.get("hana_creds")
        if hana_creds and st.button("🔌 재연결", key="btn_reconnect"):
            with st.spinner("재연결 시도 중..."):
                try:
                    _conn.ensure_connected(hana_creds, session_state=st.session_state)
                    st.success("✅ 재연결 성공")
                except Exception as e:
                    st.error(f"❌ 재연결 실패: {e}")
        elif not hana_creds:
            st.caption("재연결하려면 1번 페이지에서 먼저 연결하세요.")

# ─── Tab 2: ETL 실행 이력 ────────────────────────────────────────────────────
with tab2:
    etl_records = load_etl_log(n=50)
    if not etl_records:
        st.info(
            "ETL 실행 이력이 없습니다.\n\n"
            "3단계 모델 학습 탭에서 ETL을 실행하면 이력이 자동으로 기록됩니다.\n"
            "이력은 앱을 재시작해도 유지됩니다."
        )
    else:
        st.caption(f"총 {len(etl_records)}건 (최근 50건 표시, 최신순)")
        etl_df = pd.DataFrame(etl_records)
        etl_df = etl_df.rename(columns={
            "ts": "실행 시각", "period_from": "시작 기간", "period_to": "종료 기간",
            "row_count": "추출 건수", "elapsed_sec": "소요(초)", "status": "상태", "error": "오류",
        })
        etl_df["추출 건수"] = etl_df["추출 건수"].apply(lambda x: f"{x:,}")
        st.dataframe(etl_df, use_container_width=True, hide_index=True)

# ─── Tab 3: 모델 학습 이력 ────────────────────────────────────────────────────
with tab3:
    if not _saved_results:
        st.info("저장된 모델 결과가 없습니다. 3단계 모델 학습을 먼저 실행하세요.")
    else:
        rows = []
        for r in _saved_results:
            m = r.get("metrics", {})
            rows.append({
                "시각": r.get("timestamp", "?"),
                "모델": r.get("model_name", "?"),
                "타겟": r.get("target", "?"),
                "Accuracy": round(m.get("accuracy", 0), 4),
                "F1": round(m.get("f1_macro", 0), 4),
                "AUC": round(m.get("roc_auc", m.get("roc_auc_ovr", 0)), 4),
                "학습 수": m.get("train_size", 0),
                "_file": r.get("_file", ""),
            })
        hist_df = pd.DataFrame(rows)

        # 성능 추이 차트
        if len(hist_df) > 1:
            fig = go.Figure()
            for metric in ["Accuracy", "F1", "AUC"]:
                fig.add_trace(go.Scatter(
                    x=hist_df["시각"], y=hist_df[metric],
                    mode="lines+markers", name=metric,
                ))
            fig.update_layout(
                title="모델 성능 추이",
                xaxis_title="학습 시각", yaxis_title="Score",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        # 결과 테이블 (최신 강조)
        display_df = hist_df.drop(columns=["_file"])
        st.dataframe(
            display_df.style.apply(
                lambda row: ["background-color: #e8f5e9" if row.name == 0 else "" for _ in row],
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

        # 삭제
        st.markdown("#### 결과 삭제")
        del_options = {
            f"{r['시각']} — {r['모델']}": r["_file"]
            for r in rows if r["_file"]
        }
        if del_options:
            del_label = st.selectbox("삭제할 결과 선택", list(del_options.keys()), key="del_result_sel")
            if st.button("🗑️ 선택 결과 삭제", key="btn_del_result"):
                del_path = Path(del_options[del_label])
                if del_path.exists():
                    del_path.unlink()
                    st.success(f"삭제 완료: {del_path.name}")
                    st.rerun()
                else:
                    st.error("파일을 찾을 수 없습니다.")

# ─── Tab 4: 시스템 상태 ──────────────────────────────────────────────────────
with tab4:
    st.markdown("#### 저장소 현황")
    s1, s2 = st.columns(2)

    def _dir_info(d: Path) -> tuple[int, float]:
        """(파일 수, 총 MB)"""
        if not d.exists():
            return 0, 0.0
        files = list(d.iterdir())
        total = sum(f.stat().st_size for f in files if f.is_file())
        return len(files), total / (1024 * 1024)

    with s1:
        st.markdown(f"**📁 results/** `{RESULTS_DIR}`")
        n_r, mb_r = _dir_info(RESULTS_DIR)
        st.write(f"파일 {n_r}개 / {mb_r:.1f} MB")
        if n_r:
            r_files = sorted(RESULTS_DIR.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
            r_df = pd.DataFrame([
                {"파일명": f.name, "크기(KB)": round(f.stat().st_size / 1024, 1)}
                for f in r_files if f.is_file()
            ])
            st.dataframe(r_df, use_container_width=True, hide_index=True)

    with s2:
        st.markdown(f"**📁 models/** `{MODELS_DIR}`")
        n_m, mb_m = _dir_info(MODELS_DIR)
        st.write(f"파일 {n_m}개 / {mb_m:.1f} MB")
        if n_m:
            m_files = sorted(MODELS_DIR.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
            m_df = pd.DataFrame([
                {"파일명": f.name, "크기(MB)": round(f.stat().st_size / (1024*1024), 1)}
                for f in m_files if f.is_file()
            ])
            st.dataframe(m_df, use_container_width=True, hide_index=True)

    total_mb = mb_r + mb_m
    st.metric("총 디스크 사용량", f"{total_mb:.1f} MB")

    st.markdown("#### 설정 파일")
    from hana_app.core.config import CONFIG_FILE
    if CONFIG_FILE.exists():
        st.success(f"✅ {CONFIG_FILE.name} 존재 ({CONFIG_FILE.stat().st_size / 1024:.1f} KB)")
    else:
        st.warning(f"⚠️ {CONFIG_FILE.name} 없음 — 1번 페이지에서 설정 후 저장하세요.")
```

- [ ] **Step 2: 문법 확인**

```bash
python3 -c "import ast; ast.parse(open('hana_app/pages/6_📊_모니터링.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: 커밋**

```bash
git add "hana_app/pages/6_📊_모니터링.py"
git commit -m "feat: page6 전면 재작성 — 상태 요약 바 + HANA/ETL/모델/시스템 4탭 (Docker 의존 제거)"
```

---

## Task 6: 최종 문법 일괄 검증 + 커밋

- [ ] **Step 1: 변경된 모든 파일 문법 확인**

```bash
python3 -c "
import ast
files = [
    'hana_app/core/etl_logger.py',
    'hana_app/core/ml_runner.py',
    'hana_app/pages/3_🤖_모델_학습.py',
    'hana_app/pages/4_📊_결과_분석.py',
    'hana_app/pages/6_📊_모니터링.py',
]
for f in files:
    try:
        ast.parse(open(f, encoding='utf-8').read())
        print('OK  ' + f)
    except SyntaxError as e:
        print('ERR ' + f + ': ' + str(e))
"
```

Expected: 5줄 모두 `OK`

- [ ] **Step 2: etl_logger 테스트 재실행**

```bash
python3 -m pytest tests/test_hana_app/test_etl_logger.py -v
```

Expected: 6개 PASSED

- [ ] **Step 3: 전체 test_hana_app 테스트 실행**

```bash
python3 -m pytest tests/test_hana_app/ -v --tb=short
```

Expected: 모든 기존 테스트 PASSED (최소 38건)
