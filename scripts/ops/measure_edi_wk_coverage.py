"""EDI→WK 맵 커버리지 & 미매핑 약물의 DDI 임상중요도 측정 (Task B P1).

서빙은 edi→wk(HIRA 급여목록 기반 맵)으로만 약물을 식별한다. 맵에 없는 edi 는 DDI
평가에서 제외(degraded, "미매핑≠음성")된다. 본 스크립트는 raw records(edi+wk 동시
보유, ground truth)로 그 누락의 **임상중요도**를 정량화한다:

  1) 맵 커버리지: records edi 중 HIRA edi→wk 맵 적중률.
  2) 미매핑 약물의 DDI-capability: 미매핑 edi 의 records-wk 가 DrugMaster.get_ddi_ids
     비어있지 않은 비율(= DDI 평가 대상 약물인데 서빙이 못 보는 것).
  3) 처방행/환자 커버리지: 미매핑·미매핑-DDI약물이 차지하는 비중.
  4) 쌍 단위 영향(환자 샘플): records-wk(full) vs HIRA-map(serving) 의 DDI 카운트 차이
     = 서빙이 놓치는 실제 DDI 이벤트(major/contraindicated 중심).

산출 dict 를 출력 — ops 문서/배포 가이드의 근거 수치로 사용.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.etl.code_standardizer import CodeStandardizer
from scripts.etl.models import PrescriptionRecord
from scripts.etl.overlap_calculator import calculate_overlaps_for_patient
from scripts.etl.prescription_aggregator import count_ddi_severities
import hana_app.core.ml_runner as M

EDIWK_MAP = "data/processed/edi_to_wk.parquet"


def _rec(row, wk: str) -> PrescriptionRecord:
    return PrescriptionRecord(
        patient_id=str(row["patient_id"]), institution_id=str(row.get("institution_id") or ""),
        bill_no=str(row.get("bill_no") or ""), wk_compn_cd=wk, edi_code=str(row["edi_code"]),
        start_date=date.fromisoformat(str(row["start_date"])),
        end_date=date.fromisoformat(str(row["end_date"])),
        total_days=int(row["total_days"]), source="T30",
    )


def measure(raw_paths: list[str], sample_patients: int = 3000) -> dict:
    std = CodeStandardizer()                 # 기본 경로(MASTER/edi_to_wk) 로드
    dm = M._load_drug_master()
    ddi_matrix = M._load_ddi_matrix()
    edi_wk_map = pd.read_parquet(EDIWK_MAP).set_index("edi_code")["wk_compn_cd"].to_dict()

    df = pd.concat([pd.read_parquet(p) for p in raw_paths], ignore_index=True)
    df = df.dropna(subset=["edi_code", "wk_compn_cd", "start_date", "end_date"])

    # ── 1) 맵 커버리지 (unique edi) ──────────────────────────────────────────
    rec_edi_wk: dict[str, str] = {}          # records ground-truth edi→wk
    for edi, wk in zip(df["edi_code"].astype(str), df["wk_compn_cd"].astype(str)):
        rec_edi_wk.setdefault(edi, wk)
    norm = {e: std._normalize_edi(e) for e in rec_edi_wk}
    mapped = {e for e, n in norm.items() if n in edi_wk_map}
    unmapped = [e for e in rec_edi_wk if e not in mapped]

    # ── 2) 미매핑 약물의 DDI-capability (records-wk 기준) ────────────────────
    unmapped_ddi_capable = [e for e in unmapped if dm.get_ddi_ids(rec_edi_wk[e])]

    # ── 3) 처방행/환자 커버리지 ──────────────────────────────────────────────
    df["_edi"] = df["edi_code"].astype(str)
    unmapped_set = set(unmapped)
    unmapped_ddi_set = set(unmapped_ddi_capable)
    total_rows = len(df)
    rows_unmapped = int(df["_edi"].isin(unmapped_set).sum())
    rows_unmapped_ddi = int(df["_edi"].isin(unmapped_ddi_set).sum())
    pts_total = df["patient_id"].nunique()
    pts_unmapped_ddi = df.loc[df["_edi"].isin(unmapped_ddi_set), "patient_id"].nunique()

    # ── 4) 쌍 단위 영향 (환자 샘플) — full(records-wk) vs serving(HIRA-map) ──
    full = {"Contraindicated": 0, "Major": 0, "Moderate": 0, "Minor": 0}
    serve = {"Contraindicated": 0, "Major": 0, "Moderate": 0, "Minor": 0}
    pids = list(dict.fromkeys(df["patient_id"].tolist()))[:sample_patients]
    sample_df = df[df["patient_id"].isin(set(pids))]
    n_sampled = 0
    for pid, g in sample_df.groupby("patient_id"):
        rows = list(g.to_dict("records"))
        # full view: records wk 직접
        recs_full = [_rec(r, str(r["wk_compn_cd"])) for r in rows]
        # serving view: edi→HIRA map (미매핑 제외)
        recs_serve = []
        for r in rows:
            wk = edi_wk_map.get(std._normalize_edi(str(r["edi_code"])))
            if wk:
                recs_serve.append(_rec(r, wk))
        if len(recs_full) >= 2:
            for k, v in count_ddi_severities(
                calculate_overlaps_for_patient(recs_full, window_days=90), ddi_matrix, dm).items():
                full[k] += v
        if len(recs_serve) >= 2:
            for k, v in count_ddi_severities(
                calculate_overlaps_for_patient(recs_serve, window_days=90), ddi_matrix, dm).items():
                serve[k] += v
        n_sampled += 1

    pct = lambda a, b: round(a / b * 100, 2) if b else 0.0
    return {
        "raw_files": len(raw_paths),
        "unique_edi": len(rec_edi_wk),
        "mapped_edi": len(mapped),
        "map_coverage_pct": pct(len(mapped), len(rec_edi_wk)),
        "unmapped_edi": len(unmapped),
        "unmapped_ddi_capable": len(unmapped_ddi_capable),
        "unmapped_ddi_capable_pct_of_unmapped": pct(len(unmapped_ddi_capable), len(unmapped)),
        "total_rx_rows": total_rows,
        "rows_unmapped": rows_unmapped,
        "rows_unmapped_pct": pct(rows_unmapped, total_rows),
        "rows_unmapped_ddi_capable": rows_unmapped_ddi,
        "rows_unmapped_ddi_capable_pct": pct(rows_unmapped_ddi, total_rows),
        "patients_total": int(pts_total),
        "patients_touched_by_unmapped_ddi": int(pts_unmapped_ddi),
        "patients_touched_pct": pct(pts_unmapped_ddi, pts_total),
        "pair_sample_patients": n_sampled,
        "pair_full_ddi": full,
        "pair_serving_ddi": serve,
        "pair_missed_major": full["Major"] - serve["Major"],
        "pair_missed_contraindicated": full["Contraindicated"] - serve["Contraindicated"],
        "pair_missed_major_pct": pct(full["Major"] - serve["Major"], full["Major"]),
        "pair_missed_contra_pct": pct(full["Contraindicated"] - serve["Contraindicated"], full["Contraindicated"]),
    }


def main(argv: list[str] | None = None) -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    p = argparse.ArgumentParser(description="EDI→WK 커버리지 & 미매핑 DDI 임상중요도 측정")
    p.add_argument("--raw-dir", default="data/Raw")
    p.add_argument("--glob", default="records_20240701.parquet")
    p.add_argument("--sample-patients", type=int, default=3000)
    args = p.parse_args(argv)
    raw_paths = sorted(str(x) for x in Path(args.raw_dir).glob(args.glob))
    if not raw_paths:
        print(f"[ERR] raw 파일 없음: {args.raw_dir}/{args.glob}")
        return 1
    result = measure(raw_paths, sample_patients=args.sample_patients)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
