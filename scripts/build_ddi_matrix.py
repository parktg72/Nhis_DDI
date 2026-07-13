#!/usr/bin/env python3
"""
DDI 매트릭스 통합 빌더

DrugBank DDI + 식약처 DUR 병용금기를 병합하여 최종 DDI 매트릭스 생성.
우선순위: HIRA DUR > DrugBank (프로젝트 합의)

Input:
  data/drugbank/drugbank_drugs.parquet
  data/drugbank/drugbank_ddi.parquet
  data/drugbank/drugbank_cyp.parquet
  data/dur/dur_ddi_contraindicated_std.parquet  (optional)
  data/dur/dur_therapeutic_duplicate_std.parquet (optional)

Output:
  data/processed/ddi_matrix_final.parquet     - 통합 DDI 매트릭스
  data/processed/drug_name_index.parquet      - 약물명 → ID 인덱스
  data/processed/cyp_matrix.parquet          - CYP450 약물별 피처
  data/processed/efcy_duplicate_groups.parquet - 효능군 중복 그룹

사용법:
  python scripts/build_ddi_matrix.py
  python scripts/build_ddi_matrix.py --skip-dur    (DUR 데이터 없을 때)
"""
import argparse
from pathlib import Path

import pandas as pd

SEVERITY_RANK = {"Contraindicated": 0, "Major": 1, "Moderate": 2, "Minor": 3, "Unknown": 4}


def load_parquet_safe(path: Path, description: str) -> pd.DataFrame:
    """파일이 없으면 빈 DataFrame 반환."""
    if path.exists():
        df = pd.read_parquet(path)
        print(f"  [로드] {description}: {len(df):,} 건  ({path})")
        return df
    else:
        print(f"  [없음] {description}: {path} (건너뜀)")
        return pd.DataFrame()


def normalize_name(name: str) -> str:
    """약물명 정규화 (소문자, 공백 정리)."""
    if not isinstance(name, str):
        return ""
    return name.lower().strip()


def merge_ddi_sources(df_drugbank: pd.DataFrame, df_dur: pd.DataFrame) -> pd.DataFrame:
    """
    DrugBank DDI + DUR 병용금기 병합.
    동일 쌍이 두 소스에 모두 있을 경우: DUR 우선 (심각도 상향 가능).
    """
    all_rows = []

    # DrugBank DDI 로드
    if not df_drugbank.empty:
        for _, row in df_drugbank.iterrows():
            all_rows.append({
                "drug_a_name": row.get("drug_a_name", ""),
                "drug_b_name": row.get("drug_b_name", ""),
                "drug_a_id": row.get("drug_a_id", ""),
                "drug_b_id": row.get("drug_b_id", ""),
                "severity": row.get("severity", "Unknown"),
                "description": row.get("description", ""),
                "source": "DrugBank",
                "source_priority": 2,
            })

    # DUR 병용금기 로드
    if not df_dur.empty:
        for _, row in df_dur.iterrows():
            all_rows.append({
                "drug_a_name": row.get("drug_a_name", ""),
                "drug_b_name": row.get("drug_b_name", ""),
                "drug_a_id": row.get("drug_a_code", ""),
                "drug_b_id": row.get("drug_b_code", ""),
                "severity": "Contraindicated",   # DUR 병용금기는 무조건 Contraindicated
                "description": row.get("prohibition_detail", "DUR 병용금기"),
                "source": "HIRA_DUR",
                "source_priority": 1,
            })

    if not all_rows:
        print("  [경고] DDI 데이터 없음!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # 이름 정규화 (정렬용)
    df["_name_a_norm"] = df["drug_a_name"].apply(normalize_name)
    df["_name_b_norm"] = df["drug_b_name"].apply(normalize_name)

    # 쌍 정규화 (순서 무관)
    def sort_pair_names(row):
        if row["_name_a_norm"] <= row["_name_b_norm"]:
            return row["_name_a_norm"], row["_name_b_norm"]
        return row["_name_b_norm"], row["_name_a_norm"]

    df[["_pair_a", "_pair_b"]] = df.apply(sort_pair_names, axis=1, result_type="expand")

    # 우선순위 + 심각도 기준 정렬 → 최고 우선순위 유지
    df["_sev_rank"] = df["severity"].map(SEVERITY_RANK).fillna(4)
    df = df.sort_values(["source_priority", "_sev_rank"])
    df = df.drop_duplicates(subset=["_pair_a", "_pair_b"], keep="first")

    # 정리
    df = df.drop(columns=["_name_a_norm", "_name_b_norm", "_pair_a", "_pair_b", "_sev_rank", "source_priority"])
    return df.reset_index(drop=True)


def build_drug_name_index(df_drugs: pd.DataFrame, df_ddi: pd.DataFrame) -> pd.DataFrame:
    """약물명 → DrugBank ID 인덱스 (대소문자 무관 검색용)."""
    rows = []

    if not df_drugs.empty:
        for _, row in df_drugs.iterrows():
            rows.append({
                "drug_name": row.get("name", ""),
                "drug_name_lower": normalize_name(row.get("name", "")),
                "drugbank_id": row.get("drugbank_id", ""),
                "atc_codes": row.get("atc_codes", ""),
                "groups": row.get("groups", ""),
            })

    # DDI 에서 파트너 약물 보충 (drugs 에 없는 경우)
    if not df_ddi.empty:
        existing_ids = {r["drugbank_id"] for r in rows}
        for _, row in df_ddi.iterrows():
            for id_col, name_col in [("drug_a_id", "drug_a_name"), ("drug_b_id", "drug_b_name")]:
                did = row.get(id_col, "")
                name = row.get(name_col, "")
                if did and did not in existing_ids:
                    existing_ids.add(did)
                    rows.append({
                        "drug_name": name,
                        "drug_name_lower": normalize_name(name),
                        "drugbank_id": did,
                        "atc_codes": "",
                        "groups": "",
                    })

    return pd.DataFrame(rows).drop_duplicates(subset=["drug_name_lower"]).reset_index(drop=True)


def build_cyp_feature_matrix(df_cyp: pd.DataFrame) -> pd.DataFrame:
    """
    약물별 CYP450 피처 25개 생성.
    CLINICAL_STANDARDS_v1.0.md Group C 피처.
    """
    if df_cyp.empty:
        return pd.DataFrame()

    enzymes = ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2"]
    action_types = ["substrate", "inhibitor_strong", "inhibitor_moderate", "inhibitor", "inducer"]

    # 각 약물 × 효소 × 액션 피벗
    records = {}
    for _, row in df_cyp.iterrows():
        did = row.get("drugbank_id", "")
        name = row.get("drug_name", "")
        enzyme = row.get("enzyme", "").upper()
        action = row.get("action", "").lower()
        known = row.get("known_action", "").lower()

        if not did or known == "no":
            continue

        key = did
        if key not in records:
            records[key] = {"drugbank_id": did, "drug_name": name}
            for e in enzymes:
                for at in action_types:
                    records[key][f"{e.lower()}_{at}_count"] = 0
                records[key][f"{e.lower()}_interaction_risk"] = 0

        # 해당 효소 매칭
        for e in enzymes:
            if e in enzyme:
                col_prefix = e.lower()
                if "substrate" in action:
                    records[key][f"{col_prefix}_substrate_count"] += 1
                if "inhibitor" in action:
                    # 강도 추정 (DrugBank에서 강도 정보가 있으면 활용)
                    if "strong" in action:
                        records[key][f"{col_prefix}_inhibitor_strong_count"] += 1
                    elif "moderate" in action:
                        records[key][f"{col_prefix}_inhibitor_moderate_count"] += 1
                    else:
                        records[key][f"{col_prefix}_inhibitor_count"] += 1
                if "inducer" in action:
                    records[key][f"{col_prefix}_inducer_count"] += 1

    df = pd.DataFrame(list(records.values()))

    # interaction_risk: substrate 가 있으면서 inhibitor/inducer 도 있으면 위험
    for e in enzymes:
        p = e.lower()
        df[f"{p}_interaction_risk"] = (
            (df[f"{p}_substrate_count"] > 0) &
            (df[f"{p}_inhibitor_count"] + df[f"{p}_inhibitor_strong_count"] + df[f"{p}_inhibitor_moderate_count"] + df[f"{p}_inducer_count"] > 0)
        ).astype(int)

    return df.reset_index(drop=True)


def print_ddi_stats(df: pd.DataFrame):
    """DDI 매트릭스 통계 출력."""
    print(f"\n[DDI 매트릭스 통계]")
    print(f"  총 DDI 쌍   : {len(df):,}")
    if df.empty:
        return
    sev_counts = df["severity"].value_counts()
    for sev in ["Contraindicated", "Major", "Moderate", "Minor", "Unknown"]:
        cnt = sev_counts.get(sev, 0)
        bar = "█" * min(cnt // max(len(df) // 50, 1), 40)
        print(f"  {sev:>15}: {cnt:>8,}  {bar}")
    src_counts = df["source"].value_counts()
    print(f"\n[소스별 통계]")
    for src, cnt in src_counts.items():
        print(f"  {src:>20}: {cnt:>8,}")


def main():
    parser = argparse.ArgumentParser(description="DDI 매트릭스 통합 빌더")
    parser.add_argument("--drugbank-dir", default="data/drugbank", help="DrugBank 파싱 결과 디렉토리")
    parser.add_argument("--dur-dir", default="data/dur", help="DUR 수집 결과 디렉토리")
    parser.add_argument("--out-dir", default="data/processed", help="출력 디렉토리")
    parser.add_argument("--skip-dur", action="store_true", help="DUR 데이터 없이 DrugBank만 사용")
    args = parser.parse_args()

    db_dir = Path(args.drugbank_dir)
    dur_dir = Path(args.dur_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[DrugBank 데이터 로드]")
    df_drugs = load_parquet_safe(db_dir / "drugbank_drugs.parquet", "DrugBank 약물")
    df_db_ddi = load_parquet_safe(db_dir / "drugbank_ddi.parquet", "DrugBank DDI")
    df_cyp = load_parquet_safe(db_dir / "drugbank_cyp.parquet", "DrugBank CYP450")

    df_dur_ddi = pd.DataFrame()
    df_efcy_dup = pd.DataFrame()
    if not args.skip_dur:
        print("\n[DUR 데이터 로드]")
        df_dur_ddi = load_parquet_safe(dur_dir / "dur_ddi_contraindicated_std.parquet", "DUR 병용금기")
        df_efcy_dup = load_parquet_safe(dur_dir / "dur_therapeutic_duplicate_std.parquet", "DUR 효능군중복")

    # 1. DDI 매트릭스 통합
    print("\n[DDI 매트릭스 통합 중...]")
    df_ddi_final = merge_ddi_sources(df_db_ddi, df_dur_ddi)
    print_ddi_stats(df_ddi_final)

    # 2. 약물명 인덱스
    print("\n[약물명 인덱스 생성 중...]")
    df_drug_index = build_drug_name_index(df_drugs, df_ddi_final)
    print(f"  약물 인덱스: {len(df_drug_index):,} 건")

    # 3. CYP 피처 매트릭스
    print("\n[CYP450 피처 생성 중...]")
    df_cyp_matrix = build_cyp_feature_matrix(df_cyp)
    print(f"  CYP 피처: {len(df_cyp_matrix):,} 약물 × {df_cyp_matrix.shape[1] if not df_cyp_matrix.empty else 0} 피처")

    # 저장
    ddi_path = out_dir / "ddi_matrix_final.parquet"
    index_path = out_dir / "drug_name_index.parquet"
    cyp_path = out_dir / "cyp_matrix.parquet"

    df_ddi_final.to_parquet(ddi_path, index=False)
    df_drug_index.to_parquet(index_path, index=False)
    df_cyp_matrix.to_parquet(cyp_path, index=False) if not df_cyp_matrix.empty else None

    if not df_efcy_dup.empty:
        dup_path = out_dir / "efcy_duplicate_groups.parquet"
        df_efcy_dup.to_parquet(dup_path, index=False)
        print(f"\n[저장] {dup_path}")

    print(f"\n[저장 완료]")
    print(f"  {ddi_path}")
    print(f"  {index_path}")
    print(f"  {cyp_path}")
    print(f"\n다음 단계: python -c \"from rules.safety_net import SafetyNet; sn = SafetyNet(); print(sn.info())\"")


if __name__ == "__main__":
    main()
