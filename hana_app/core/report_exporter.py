"""
보고서 내보내기 — DOCX 분석 보고서 + 대상자 CSV.

공개 API:
    build_csv_bytes(features_df)   -> bytes  (UTF-8 BOM, Red/Y_DDI_MAJOR/Y_TRIPLE)
    build_docx_bytes(last_result, features_df=None) -> bytes
    DOCX_AVAILABLE                 -> bool
"""
from __future__ import annotations

import datetime as dt
import io
from typing import Optional

import pandas as pd

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

from scripts.ops.multi_institution_label import MULTI_INSTITUTION_THRESHOLD

# ── CSV ─────────────────────────────────────────────────────────────────────

_CSV_TARGET_LABELS = frozenset({"Red", "Y_DDI_MAJOR", "Y_TRIPLE"})

_INTERVENTION_KO = {
    "Red": "즉각 개입",
    "Y_DDI_MAJOR": "약사 전화",
    "Y_TRIPLE": "문자 안내",
}


def _effective_label(row) -> str:
    if row.get("risk_level") == "Red":
        return "Red"
    return row.get("yellow_subtype") or ""


def _build_reason(row) -> str:
    label = _effective_label(row)
    if label == "Red":
        n = int(row.get("ddi_contraindicated", 0))
        return f"금기 DDI {n}건"
    if label == "Y_DDI_MAJOR":
        n = int(row.get("ddi_major", 0))
        return f"중증 DDI {n}건"
    # Y_TRIPLE — 충족 차원 나열
    dims = []
    drug_count = row.get("drug_count", 0)
    if drug_count >= 10:
        dims.append(f"다약제({int(drug_count)}종)")
    if row.get("has_high_risk_drug") or row.get("has_renal_risk_drug") or row.get("has_hepatic_risk_drug"):
        dims.append("고위험약물/장기부전")
    inst = row.get("institution_count", 0)
    if inst >= MULTI_INSTITUTION_THRESHOLD:
        dims.append(f"다기관({int(inst)}개)")
    if row.get("triple_whammy"):
        dims.append("Triple Whammy")
    prefix = "3중위험" if label == "Y_TRIPLE" else "위험"
    return (f"{prefix} — " + "+".join(dims)) if dims else prefix


def build_csv_bytes(features_df: pd.DataFrame) -> bytes:
    """Red / Y_DDI_MAJOR / Y_TRIPLE 환자 행만 추출 → UTF-8 BOM CSV bytes."""
    ys = features_df["yellow_subtype"] if "yellow_subtype" in features_df.columns else pd.Series("", index=features_df.index)
    mask = (features_df["risk_level"] == "Red") | ys.isin({"Y_DDI_MAJOR", "Y_TRIPLE"})
    filtered = features_df[mask].copy()

    filtered["개입조치"] = filtered.apply(lambda r: _INTERVENTION_KO.get(_effective_label(r), ""), axis=1)
    filtered["위험라벨"] = filtered.apply(_effective_label, axis=1)
    filtered["사유"] = filtered.apply(_build_reason, axis=1)

    out_cols = {
        "patient_id": "환자ID",
        "개입조치": "개입조치",
        "위험라벨": "위험라벨",
        "사유": "사유",
        "drug_count": "다약제수",
        "ddi_major": "중증DDI건수",
        "ddi_contraindicated": "금기DDI건수",
        "dup_same_ingredient": "중복처방수",
        "institution_count": "다기관수",
    }
    present = [c for c in out_cols if c in filtered.columns or c in ("개입조치", "위험라벨", "사유")]
    out = filtered[present].rename(columns=out_cols)

    buf = io.BytesIO()
    out.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()


# ── DOCX ────────────────────────────────────────────────────────────────────

_DOCX_LABEL_MAP = {
    "ddi_contraindicated": "금기 DDI 평균",
    "ddi_major": "Major DDI 평균",
    "ddi_moderate": "Moderate DDI 평균",
    "ddi_minor": "Minor DDI 평균",
}

_INTERVENTION_ROWS = [
    ("즉각 개입", "Red"),
    ("약사 전화", "Y_DDI_MAJOR"),
    ("문자 안내", "Y_TRIPLE"),
    ("모니터링", "Y_DOUBLE·Y_DDI_MOD·Y_DUP·Y_FRAG"),
    ("관여 안함", "No_Alert·Green·Normal"),
]

_MONITORING_SUBTYPES = frozenset({"Y_DOUBLE", "Y_DDI_MOD", "Y_DUP", "Y_FRAG"})


def _count_distribution(features_df: pd.DataFrame) -> dict[str, int]:
    ys = features_df["yellow_subtype"] if "yellow_subtype" in features_df.columns else pd.Series("", index=features_df.index)
    return {
        "Red": int((features_df["risk_level"] == "Red").sum()),
        "Y_DDI_MAJOR": int((ys == "Y_DDI_MAJOR").sum()),
        "Y_TRIPLE": int((ys == "Y_TRIPLE").sum()),
        "모니터링": int(ys.isin(_MONITORING_SUBTYPES).sum()),
        "관여안함": max(0, len(features_df) - int((features_df["risk_level"] == "Red").sum()) - int(ys.isin(_MONITORING_SUBTYPES | {"Y_DDI_MAJOR", "Y_TRIPLE"}).sum())),
    }


def build_docx_bytes(last_result: dict, features_df: Optional[pd.DataFrame] = None) -> bytes:
    """분석 보고서 DOCX bytes. python-docx 미설치 시 ImportError."""
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx가 설치되지 않았습니다.")

    doc = Document()
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    model_name = last_result.get("model_name", "?")

    # ── 표지
    doc.add_heading("처방 위험도 예측 분석 보고서", 0)
    doc.add_paragraph(f"생성일시: {now_str}")
    doc.add_paragraph(f"모델명: {model_name}")

    # ── 섹션 2: 개입 위계 분포
    doc.add_heading("개입 위계 분포", level=1)
    if features_df is not None and not features_df.empty:
        counts = _count_distribution(features_df)
        total = len(features_df)
    else:
        risk_summary = last_result.get("risk_summary", {})
        counts = dict(risk_summary)
        total = sum(counts.values())

    total_denom = total if total > 0 else 1
    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = "Table Grid"
    for i, h in enumerate(("개입조치", "라벨", "건수", "비율")):
        tbl.rows[0].cells[i].text = h
    for action, label_key in _INTERVENTION_ROWS:
        n = counts.get(label_key, counts.get(action, 0))
        row = tbl.add_row().cells
        row[0].text = action
        row[1].text = label_key
        row[2].text = f"{n:,}"
        row[3].text = f"{n / total_denom * 100:.2f}%"

    doc.add_paragraph()

    # ── 섹션 3: 모델 성능
    doc.add_heading("모델 성능 지표", level=1)
    metrics = last_result.get("metrics", {})
    metric_rows = [
        ("F1 (macro)", metrics.get("f1_macro")),
        ("Accuracy", metrics.get("accuracy")),
        ("AUC", metrics.get("roc_auc") or metrics.get("roc_auc_ovr")),
        ("CV 평균", metrics.get("cv_mean")),
        ("τ_red", metrics.get("tau_red")),
        ("τ_review", metrics.get("tau_review")),
    ]
    tbl_m = doc.add_table(rows=1, cols=2)
    tbl_m.style = "Table Grid"
    tbl_m.rows[0].cells[0].text = "지표"
    tbl_m.rows[0].cells[1].text = "값"
    for name, val in metric_rows:
        if val is not None:
            r = tbl_m.add_row().cells
            r[0].text = name
            r[1].text = f"{val:.4f}" if isinstance(val, float) else str(val)

    doc.add_paragraph()

    # ── 섹션 4: 피처 중요도 Top 15
    fi_data = last_result.get("feature_importance")
    _fi_ok = fi_data is not None and (fi_data.empty is False if hasattr(fi_data, "empty") else bool(fi_data))
    if _fi_ok:
        doc.add_heading("피처 중요도 Top 15", level=1)
        try:
            fi_df = pd.DataFrame(fi_data) if isinstance(fi_data, list) else fi_data
            fi_top = fi_df.sort_values("importance", ascending=False).head(15)
            tbl_fi = doc.add_table(rows=1, cols=2)
            tbl_fi.style = "Table Grid"
            tbl_fi.rows[0].cells[0].text = "피처"
            tbl_fi.rows[0].cells[1].text = "중요도"
            for _, r in fi_top.iterrows():
                row = tbl_fi.add_row().cells
                row[0].text = str(r["feature"])
                row[1].text = f"{r['importance']:.4f}"
        except Exception:
            doc.add_paragraph("피처 중요도 데이터를 표시할 수 없습니다.")
        doc.add_paragraph()

    # ── 섹션 5: DDI / 다약제 통계
    ddi_means = last_result.get("ddi_means")
    drug_stats = last_result.get("drug_count_stats")
    if ddi_means or drug_stats:
        doc.add_heading("DDI / 다약제 통계", level=1)
        tbl_ddi = doc.add_table(rows=1, cols=2)
        tbl_ddi.style = "Table Grid"
        tbl_ddi.rows[0].cells[0].text = "항목"
        tbl_ddi.rows[0].cells[1].text = "값"
        if ddi_means:
            for k, label in _DOCX_LABEL_MAP.items():
                if k in ddi_means:
                    r = tbl_ddi.add_row().cells
                    r[0].text = label
                    r[1].text = f"{ddi_means[k]:.4f}"
        if drug_stats:
            for stat_k, stat_label in (("mean", "약물 수 평균"), ("max", "약물 수 최대")):
                if stat_k in drug_stats:
                    r = tbl_ddi.add_row().cells
                    r[0].text = stat_label
                    r[1].text = str(drug_stats[stat_k])
        doc.add_paragraph()

    # ── 섹션 6: 분석 메모
    doc.add_heading("분석 메모", level=1)
    doc.add_paragraph(
        "본 보고서는 MODE_11_hana 배치 학습 예측 결과 기준입니다.\n"
        "서빙 실시간 예측과 수치 차이가 있을 수 있습니다."
    )

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
