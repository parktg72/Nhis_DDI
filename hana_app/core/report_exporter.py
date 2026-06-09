"""
보고서 내보내기 — 종합 서비스 DOCX 보고서 + 대상자 CSV.

공개 API:
    build_csv_bytes(features_df)              -> bytes  (UTF-8 BOM)
    build_docx_bytes(last_result, features_df) -> bytes  (차트 포함 종합 보고서)
    DOCX_AVAILABLE                            -> bool
    MPL_AVAILABLE                             -> bool
"""
from __future__ import annotations

import datetime as dt
import io
from typing import Optional

import pandas as pd

# ── matplotlib (차트 생성) ────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# ── python-docx ──────────────────────────────────────────────────────────────
try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

from scripts.ops.multi_institution_label import MULTI_INSTITUTION_THRESHOLD

# ── 색상 팔레트 ──────────────────────────────────────────────────────────────
_C = {
    "Red":       "#e74c3c",
    "MAJOR":     "#e67e22",
    "TRIPLE":    "#f1c40f",
    "monitor":   "#3498db",
    "noalert":   "#bdc3c7",
    "Green":     "#27ae60",
    "Yellow":    "#f39c12",
    "Normal":    "#95a5a6",
    "fi_bar":    "#2980b9",
    "ddi_contra":"#e74c3c",
    "ddi_major": "#e67e22",
    "ddi_mod":   "#f1c40f",
    "ddi_minor": "#3498db",
}

_INTERVENTION_KO = {
    "Red":       "즉각 개입",
    "Y_DDI_MAJOR": "약사 전화",
    "Y_TRIPLE":  "문자 안내",
}

_INTERVENTION_ROWS = [
    ("즉각 개입",  "Red",                        _C["Red"]),
    ("약사 전화",  "Y_DDI_MAJOR",                _C["MAJOR"]),
    ("문자 안내",  "Y_TRIPLE",                   _C["TRIPLE"]),
    ("모니터링",   "Y_DOUBLE·Y_DDI_MOD·Y_DUP·Y_FRAG", _C["monitor"]),
    ("관여 안함",  "No_Alert·Green·Normal",       _C["noalert"]),
]

_MONITORING_SUBTYPES = frozenset({"Y_DOUBLE", "Y_DDI_MOD", "Y_DUP", "Y_FRAG"})

_FEAT_LABELS = {
    "drug_count": "총 약물 수",
    "drug_count_7d": "최근 7일 복용",
    "institution_count": "처방 기관 수",
    "ddi_contraindicated": "금기 DDI",
    "ddi_major": "Major DDI",
    "ddi_moderate": "Moderate DDI",
    "ddi_minor": "Minor DDI",
    "triple_whammy": "Triple Whammy",
    "qt_risk_count": "QT 위험 약물",
    "dup_same_ingredient": "동일성분 중복",
    "dup_atc5": "ATC5 중복",
    "dup_atc4": "ATC4 중복",
    "dup_atc3": "ATC3 중복",
    "dup_efmdc": "약효분류 중복",
    "age": "연령",
    "sex_m": "성별(남)",
}


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _fig_to_png(fig) -> bytes:
    canvas = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    plt.close(fig)
    return buf.getvalue()


def _count_distribution(features_df: pd.DataFrame) -> dict[str, int]:
    ys = (features_df["yellow_subtype"]
          if "yellow_subtype" in features_df.columns
          else pd.Series("", index=features_df.index))
    red_n  = int((features_df["risk_level"] == "Red").sum())
    maj_n  = int((ys == "Y_DDI_MAJOR").sum())
    tri_n  = int((ys == "Y_TRIPLE").sum())
    mon_n  = int(ys.isin(_MONITORING_SUBTYPES).sum())
    none_n = max(0, len(features_df) - red_n - maj_n - tri_n - mon_n)
    return {"Red": red_n, "Y_DDI_MAJOR": maj_n, "Y_TRIPLE": tri_n,
            "모니터링": mon_n, "관여안함": none_n}


def _chart_intervention(counts: dict, total: int) -> bytes:
    """개입 위계 수평 바차트."""
    labels   = ["즉각 개입\n(Red)", "약사 전화\n(Y_DDI_MAJOR)", "문자 안내\n(Y_TRIPLE)",
                "모니터링", "관여 안함"]
    keys     = ["Red", "Y_DDI_MAJOR", "Y_TRIPLE", "모니터링", "관여안함"]
    colors   = [_C["Red"], _C["MAJOR"], _C["TRIPLE"], _C["monitor"], _C["noalert"]]
    values   = [counts.get(k, 0) for k in keys]
    pcts     = [v / total * 100 if total else 0 for v in values]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    for bar, pct, val in zip(bars, pcts[::-1], values[::-1]):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}명 ({pct:.1f}%)", va="center", fontsize=9)
    ax.set_xlabel("환자 수")
    ax.set_title("개입 위계별 환자 분포")
    ax.set_xlim(0, (max(values) * 1.25) if max(values, default=0) > 0 else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_risk_pie(features_df: pd.DataFrame) -> bytes:
    """위험도 분포 파이차트."""
    dist = features_df["risk_level"].value_counts()
    labels = list(dist.index)
    values = list(dist.values)
    colors = [_C.get(l, "#aaaaaa") for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("위험도 분포")
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_ddi(ddi_means: dict) -> bytes:
    """DDI 심각도별 평균 바차트."""
    label_map = [
        ("ddi_contraindicated", "금기",    _C["ddi_contra"]),
        ("ddi_major",           "Major",   _C["ddi_major"]),
        ("ddi_moderate",        "Moderate",_C["ddi_mod"]),
        ("ddi_minor",           "Minor",   _C["ddi_minor"]),
    ]
    items = [(lbl, col, ddi_means[k]) for k, lbl, col in label_map if k in ddi_means]
    if not items:
        return b""
    labels, colors, vals = zip(*items)

    fig, ax = plt.subplots(figsize=(5.5, 3))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("평균 쌍 수")
    ax.set_title("DDI 심각도별 평균 쌍 수")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_feature_importance(fi_df: pd.DataFrame, top_n: int = 15) -> bytes:
    """피처 중요도 수평 바차트."""
    fi_top = fi_df.sort_values("importance", ascending=False).head(top_n)
    fi_top = fi_top.sort_values("importance", ascending=True)
    labels = [_FEAT_LABELS.get(str(f), str(f)) for f in fi_top["feature"]]
    values = list(fi_top["importance"])

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.38)))
    ax.barh(labels, values, color=_C["fi_bar"], edgecolor="white")
    ax.set_xlabel("중요도")
    ax.set_title(f"피처 중요도 Top {top_n}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _fig_to_png(fig)


def _add_png(doc, png_bytes: bytes, width_inches: float = 5.5) -> None:
    if png_bytes:
        doc.add_picture(io.BytesIO(png_bytes), width=Inches(width_inches))


def _add_table(doc, headers: list[str], rows: list[list[str]]) -> None:
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = tbl.rows[0].cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
    for row_data in rows:
        row = tbl.add_row().cells
        for i, val in enumerate(row_data):
            row[i].text = str(val)


# ── CSV ──────────────────────────────────────────────────────────────────────

def _effective_label(row) -> str:
    if row.get("risk_level") == "Red":
        return "Red"
    return row.get("yellow_subtype") or ""


def _build_reason(row) -> str:
    label = _effective_label(row)
    if label == "Red":
        return f"금기 DDI {int(row.get('ddi_contraindicated', 0))}건"
    if label == "Y_DDI_MAJOR":
        return f"중증 DDI {int(row.get('ddi_major', 0))}건"
    dims = []
    if row.get("drug_count", 0) >= 10:
        dims.append(f"다약제({int(row.get('drug_count', 0))}종)")
    if row.get("has_high_risk_drug") or row.get("has_renal_risk_drug") or row.get("has_hepatic_risk_drug"):
        dims.append("고위험약물/장기부전")
    if row.get("institution_count", 0) >= MULTI_INSTITUTION_THRESHOLD:
        dims.append(f"다기관({int(row.get('institution_count', 0))}개)")
    if row.get("triple_whammy"):
        dims.append("Triple Whammy")
    prefix = "3중위험" if label == "Y_TRIPLE" else "위험"
    return (f"{prefix} — " + "+".join(dims)) if dims else prefix


def build_csv_bytes(features_df: pd.DataFrame) -> bytes:
    """Red / Y_DDI_MAJOR / Y_TRIPLE 환자 행 → UTF-8 BOM CSV bytes."""
    ys = (features_df["yellow_subtype"]
          if "yellow_subtype" in features_df.columns
          else pd.Series("", index=features_df.index))
    mask = (features_df["risk_level"] == "Red") | ys.isin({"Y_DDI_MAJOR", "Y_TRIPLE"})
    filtered = features_df[mask].copy()

    filtered["개입조치"] = filtered.apply(
        lambda r: _INTERVENTION_KO.get(_effective_label(r), ""), axis=1)
    filtered["위험라벨"] = filtered.apply(_effective_label, axis=1)
    filtered["사유"]    = filtered.apply(_build_reason, axis=1)

    out_cols = {
        "patient_id": "환자ID", "개입조치": "개입조치", "위험라벨": "위험라벨",
        "사유": "사유", "drug_count": "다약제수", "ddi_major": "중증DDI건수",
        "ddi_contraindicated": "금기DDI건수", "dup_same_ingredient": "중복처방수",
        "institution_count": "다기관수",
    }
    present = [c for c in out_cols if c in filtered.columns
               or c in ("개입조치", "위험라벨", "사유")]
    buf = io.BytesIO()
    filtered[present].rename(columns=out_cols).to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()


# ── DOCX 종합 보고서 ──────────────────────────────────────────────────────────

def build_docx_bytes(last_result: dict,
                     features_df: Optional[pd.DataFrame] = None) -> bytes:
    """종합 서비스 보고서 DOCX bytes (차트 포함). python-docx 미설치 시 ImportError."""
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx가 설치되지 않았습니다.")

    doc  = Document()
    now  = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    mname = last_result.get("model_name", "?")
    has_df = features_df is not None and not features_df.empty

    # ── 표지 ─────────────────────────────────────────────────────────────────
    title = doc.add_heading("처방 위험도 예측 종합 보고서", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"생성일시: {now}    모델: {mname}")
    doc.add_paragraph()

    # ── 1. 위험도 분포 ────────────────────────────────────────────────────────
    doc.add_heading("1. 위험도 분포", level=1)
    if has_df:
        counts = _count_distribution(features_df)
        total  = len(features_df)
    else:
        risk_summary = last_result.get("risk_summary", {})
        counts = dict(risk_summary)
        total  = sum(counts.values())

    total_d = total or 1

    if has_df and MPL_AVAILABLE:
        _add_png(doc, _chart_risk_pie(features_df), width_inches=4.5)
    doc.add_paragraph()

    # ── 2. 개입 위계 분포 ─────────────────────────────────────────────────────
    doc.add_heading("2. 개입 위계별 환자 분포", level=1)
    if MPL_AVAILABLE:
        _add_png(doc, _chart_intervention(counts, total_d), width_inches=6.0)
    doc.add_paragraph()

    _add_table(doc,
        ["개입조치", "라벨", "건수", "비율(%)"],
        [[action, lbl, f"{counts.get(lbl.split('·')[0], counts.get(action, 0)):,}",
          f"{counts.get(lbl.split('·')[0], counts.get(action, 0)) / total_d * 100:.2f}"]
         for action, lbl, _ in _INTERVENTION_ROWS]
    )
    doc.add_paragraph()

    # ── 3. DDI 심각도 통계 ────────────────────────────────────────────────────
    ddi_means  = last_result.get("ddi_means") or {}
    drug_stats = last_result.get("drug_count_stats") or {}

    if ddi_means or drug_stats:
        doc.add_heading("3. DDI 심각도 / 다약제 통계", level=1)
        if ddi_means and MPL_AVAILABLE:
            _add_png(doc, _chart_ddi(ddi_means), width_inches=5.0)
        ddi_label_map = {
            "ddi_contraindicated": "금기 DDI 평균 쌍 수",
            "ddi_major":           "Major DDI 평균 쌍 수",
            "ddi_moderate":        "Moderate DDI 평균 쌍 수",
            "ddi_minor":           "Minor DDI 평균 쌍 수",
        }
        rows = [[lbl, f"{ddi_means[k]:.4f}"]
                for k, lbl in ddi_label_map.items() if k in ddi_means]
        if drug_stats:
            if "mean" in drug_stats:
                rows.append(["약물 수 평균", str(drug_stats["mean"])])
            if "max" in drug_stats:
                rows.append(["약물 수 최대", str(drug_stats["max"])])
        if rows:
            _add_table(doc, ["항목", "값"], rows)
        doc.add_paragraph()

    # ── 4. 피처 중요도 ────────────────────────────────────────────────────────
    fi_data = last_result.get("feature_importance")
    _fi_ok  = fi_data is not None and (
        fi_data.empty is False if hasattr(fi_data, "empty") else bool(fi_data))
    if _fi_ok:
        doc.add_heading("4. 피처 중요도 Top 15", level=1)
        try:
            fi_df  = pd.DataFrame(fi_data) if isinstance(fi_data, list) else fi_data
            fi_top = fi_df.sort_values("importance", ascending=False).head(15)
            if MPL_AVAILABLE:
                _add_png(doc, _chart_feature_importance(fi_df), width_inches=6.0)
            _add_table(doc, ["피처", "한국어명", "중요도"],
                [[str(r["feature"]),
                  _FEAT_LABELS.get(str(r["feature"]), ""),
                  f"{r['importance']:.4f}"]
                 for _, r in fi_top.iterrows()])
        except Exception:
            doc.add_paragraph("피처 중요도 데이터를 표시할 수 없습니다.")
        doc.add_paragraph()

    # ── 5. 모델 성능 지표 ─────────────────────────────────────────────────────
    metrics = last_result.get("metrics", {})
    doc.add_heading("5. 모델 성능 지표", level=1)
    perf_rows = [
        [name, f"{val:.4f}" if isinstance(val, float) else str(val)]
        for name, val in [
            ("F1 (macro)",  metrics.get("f1_macro")),
            ("Accuracy",    metrics.get("accuracy")),
            ("AUC",         metrics.get("roc_auc") or metrics.get("roc_auc_ovr")),
            ("CV 평균",     metrics.get("cv_mean")),
            ("τ_red",       metrics.get("tau_red")),
            ("τ_review",    metrics.get("tau_review")),
        ] if val is not None
    ]
    if perf_rows:
        _add_table(doc, ["지표", "값"], perf_rows)
    doc.add_paragraph()

    # ── 6. 분석 메모 ─────────────────────────────────────────────────────────
    doc.add_heading("6. 분석 메모", level=1)
    notes = [
        "본 보고서는 MODE_11_hana 배치 학습 예측 결과 기준입니다.",
        "서빙 실시간 예측과 수치 차이가 있을 수 있습니다.",
    ]
    if not MPL_AVAILABLE:
        notes.append("※ matplotlib 미설치로 차트가 포함되지 않았습니다.")
    for note in notes:
        doc.add_paragraph(note)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
