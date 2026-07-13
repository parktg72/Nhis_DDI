#!/usr/bin/env python3
"""
DrugBank XML 스트리밍 파서

Input : drugbank/full database.xml  (1.8 GB)
Output:
  data/drugbank/drugbank_drugs.parquet  - 약물 기본 정보 (ID, 이름, ATC, 그룹)
  data/drugbank/drugbank_ddi.parquet    - 약물 상호작용 쌍 (심각도 추론 포함)
  data/drugbank/drugbank_cyp.parquet    - CYP450 효소 기질/억제/유도 정보

사용법:
  python scripts/parse_drugbank.py
  python scripts/parse_drugbank.py --xml drugbank/full\ database.xml --out data/drugbank
"""
import argparse
import re
import sys
from pathlib import Path

try:
    from lxml import etree
    HAS_LXML = True
except ImportError:
    import xml.etree.ElementTree as etree
    HAS_LXML = False

import pandas as pd

NS = "http://www.drugbank.ca"
TAG = lambda name: f"{{{NS}}}{name}"


# ─── 심각도 추론 ──────────────────────────────────────────────────────────────
# DrugBank XML 에는 severity 필드가 없으므로 description 키워드로 분류
SEVERITY_PATTERNS = {
    "Contraindicated": [
        r"\bcontraindicated\b",
        r"must not be (used|taken|combined)",
        r"should not be (used|taken|combined) (together|concomitantly)",
        r"absolutely contraindicated",
    ],
    "Major": [
        r"\blife[ -]threatening\b",
        r"\bfatal\b",
        r"\bdeath\b",
        r"risk or severity of (bleeding|hemorrhage|cardiac|toxicity|seizure|arrhythmia|rhabdomyolysis|neutropenia|thrombocytopenia|bone marrow)",
        r"\bsevere (toxicity|adverse|bleeding|hemorrhage)\b",
        r"\bserious (adverse|toxicity|bleeding)\b",
        r"risk of (torsades|ventricular fibrillation|cardiac arrest)",
        r"serotonin syndrome",
        r"bone marrow (suppression|toxicity)",
        r"acute (kidney|renal) (failure|injury)",
    ],
    "Moderate": [
        r"risk or severity of",
        r"may (increase|decrease|enhance|reduce) the .*(activit|effect|level|concentration)",
        r"(can|could|may) be (increased|decreased|enhanced|reduced)",
        r"(adverse|side) effects can be (increased|reduced)",
        r"(increase|decrease)s? the (risk|likelihood) of",
    ],
    "Minor": [
        r"therapeutic efficacy .* (can be increased|may be increased)",
        r"\bminor\b",
        r"\bmild\b",
        r"\bslightly\b",
    ],
}

_COMPILED = {
    severity: [re.compile(pat, re.IGNORECASE) for pat in patterns]
    for severity, patterns in SEVERITY_PATTERNS.items()
}

_SEVERITY_ORDER = ["Contraindicated", "Major", "Moderate", "Minor"]


def infer_severity(description: str) -> str:
    """DDI description 에서 심각도를 추론."""
    if not description:
        return "Unknown"
    for severity in _SEVERITY_ORDER:
        for pattern in _COMPILED[severity]:
            if pattern.search(description):
                return severity
    return "Minor"   # 기본값: 경미한 상호작용으로 처리


# ─── XML 파서 ─────────────────────────────────────────────────────────────────

def _text(elem, tag: str, default: str = "") -> str:
    child = elem.find(TAG(tag))
    return (child.text or default).strip() if child is not None else default


def _is_primary_id(id_elem) -> bool:
    return id_elem.get("primary", "false").lower() == "true"


def parse_drug_element(drug_elem) -> dict:
    """<drug> 요소 하나를 파싱하여 딕셔너리 반환."""
    result = {
        "drugbank_id": None,
        "name": "",
        "groups": [],
        "atc_codes": [],
        "description": "",
        "indication": "",
        "ddi_pairs": [],       # (partner_id, partner_name, severity, description)
        "cyp_enzymes": [],     # (enzyme_name, actions, known_action)
    }

    # DrugBank ID (primary)
    for id_elem in drug_elem.findall(TAG("drugbank-id")):
        if _is_primary_id(id_elem) and id_elem.text:
            result["drugbank_id"] = id_elem.text.strip()
            break

    result["name"] = _text(drug_elem, "name")
    result["description"] = _text(drug_elem, "description")
    result["indication"] = _text(drug_elem, "indication")

    # Groups
    groups_elem = drug_elem.find(TAG("groups"))
    if groups_elem is not None:
        result["groups"] = [g.text.strip() for g in groups_elem.findall(TAG("group")) if g.text]

    # ATC codes
    atc_codes_elem = drug_elem.find(TAG("atc-codes"))
    if atc_codes_elem is not None:
        for atc_elem in atc_codes_elem.findall(TAG("atc-code")):
            code = atc_elem.get("code", "").strip()
            if code:
                result["atc_codes"].append(code)

    # Drug interactions
    interactions_elem = drug_elem.find(TAG("drug-interactions"))
    if interactions_elem is not None:
        for inter in interactions_elem.findall(TAG("drug-interaction")):
            partner_id_elem = inter.find(TAG("drugbank-id"))
            partner_name_elem = inter.find(TAG("name"))
            desc_elem = inter.find(TAG("description"))

            partner_id = partner_id_elem.text.strip() if partner_id_elem is not None and partner_id_elem.text else ""
            partner_name = partner_name_elem.text.strip() if partner_name_elem is not None and partner_name_elem.text else ""
            desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""

            if partner_id:
                severity = infer_severity(desc)
                result["ddi_pairs"].append((partner_id, partner_name, severity, desc))

    # CYP450 enzymes
    for section_tag in ["enzymes", "transporters", "carriers"]:
        section = drug_elem.find(TAG(section_tag))
        if section is None:
            continue
        item_tag = section_tag[:-1]   # enzymes → enzyme
        for item in section.findall(TAG(item_tag)):
            name_elem = item.find(TAG("name"))
            known_action_elem = item.find(TAG("known-action"))
            actions_elem = item.find(TAG("actions"))

            enzyme_name = name_elem.text.strip() if name_elem is not None and name_elem.text else ""
            known = known_action_elem.text.strip() if known_action_elem is not None and known_action_elem.text else "unknown"
            actions = []
            if actions_elem is not None:
                actions = [a.text.strip() for a in actions_elem.findall(TAG("action")) if a.text]

            # CYP450 만 추출 (나머지는 선택)
            if enzyme_name and ("CYP" in enzyme_name.upper() or section_tag == "transporters"):
                result["cyp_enzymes"].append((enzyme_name, actions, known, section_tag))

    return result


def stream_parse(xml_path: Path, verbose: bool = True):
    """DrugBank XML 을 스트리밍으로 파싱. <drug> 단위로 yield."""
    drug_tag = TAG("drug")
    context = etree.iterparse(str(xml_path), events=("end",))

    count = 0
    for event, elem in context:
        if elem.tag == drug_tag:
            count += 1
            if verbose and count % 1000 == 0:
                print(f"  파싱 중: {count:,} 약물 처리...", end="\r", flush=True)
            yield parse_drug_element(elem)
            # 메모리 해제 (1.8GB 파일 처리 위해 필수)
            elem.clear()
            # 부모 요소도 정리 (lxml 전용)
            if HAS_LXML:
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

    if verbose:
        print(f"\n  총 {count:,} 약물 파싱 완료.")


def build_dataframes(xml_path: Path, verbose: bool = True) -> tuple:
    """파싱 결과를 3개 DataFrame 으로 변환."""
    drugs_rows = []
    ddi_rows = []
    cyp_rows = []

    for drug in stream_parse(xml_path, verbose):
        if not drug["drugbank_id"]:
            continue

        did = drug["drugbank_id"]
        name = drug["name"]
        atc_str = "|".join(drug["atc_codes"])      # pipe-separated
        groups_str = "|".join(drug["groups"])

        drugs_rows.append({
            "drugbank_id": did,
            "name": name,
            "groups": groups_str,
            "atc_codes": atc_str,
            "description": drug["description"][:500],     # 길이 제한
            "indication": drug["indication"][:500],
        })

        # DDI pairs (중복 방지: pair 순서 정규화는 build_ddi_matrix.py 에서)
        for (pid, pname, sev, desc) in drug["ddi_pairs"]:
            ddi_rows.append({
                "drug_a_id": did,
                "drug_a_name": name,
                "drug_b_id": pid,
                "drug_b_name": pname,
                "severity": sev,
                "description": desc[:500],
                "source": "DrugBank",
            })

        # CYP 정보
        for (ename, actions, known, section) in drug["cyp_enzymes"]:
            for action in actions:
                cyp_rows.append({
                    "drugbank_id": did,
                    "drug_name": name,
                    "enzyme": ename,
                    "action": action,       # substrate / inhibitor / inducer
                    "known_action": known,
                    "section": section,     # enzymes / transporters
                })

    df_drugs = pd.DataFrame(drugs_rows)
    df_ddi = pd.DataFrame(ddi_rows) if ddi_rows else pd.DataFrame(
        columns=["drug_a_id", "drug_a_name", "drug_b_id", "drug_b_name", "severity", "description", "source"]
    )
    df_cyp = pd.DataFrame(cyp_rows) if cyp_rows else pd.DataFrame(
        columns=["drugbank_id", "drug_name", "enzyme", "action", "known_action", "section"]
    )

    return df_drugs, df_ddi, df_cyp


def deduplicate_ddi(df_ddi: pd.DataFrame) -> pd.DataFrame:
    """
    DrugBank 는 A→B, B→A 양방향으로 DDI 저장.
    pair 를 (min_id, max_id) 기준 정규화하여 중복 제거.
    심각도가 다른 경우: 더 높은 심각도 채택.
    """
    severity_order = {"Contraindicated": 0, "Major": 1, "Moderate": 2, "Minor": 3, "Unknown": 4}

    def sort_pair(row):
        if row["drug_a_id"] <= row["drug_b_id"]:
            return row["drug_a_id"], row["drug_a_name"], row["drug_b_id"], row["drug_b_name"]
        else:
            return row["drug_b_id"], row["drug_b_name"], row["drug_a_id"], row["drug_a_name"]

    records = []
    for _, row in df_ddi.iterrows():
        a_id, a_name, b_id, b_name = sort_pair(row)
        records.append({
            "drug_a_id": a_id,
            "drug_a_name": a_name,
            "drug_b_id": b_id,
            "drug_b_name": b_name,
            "severity": row["severity"],
            "description": row["description"],
            "source": row["source"],
            "_sev_order": severity_order.get(row["severity"], 4),
        })

    df = pd.DataFrame(records)
    # 동일 pair 중 가장 높은 심각도 우선 유지
    df = df.sort_values("_sev_order")
    df = df.drop_duplicates(subset=["drug_a_id", "drug_b_id"], keep="first")
    df = df.drop(columns=["_sev_order"])
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="DrugBank XML 파서")
    parser.add_argument(
        "--xml",
        default="drugbank/full database.xml",
        help="DrugBank XML 파일 경로 (기본: drugbank/full database.xml)",
    )
    parser.add_argument(
        "--out",
        default="data/drugbank",
        help="출력 디렉토리 (기본: data/drugbank)",
    )
    parser.add_argument("--no-dedup", action="store_true", help="DDI 중복 제거 건너뛰기")
    args = parser.parse_args()

    xml_path = Path(args.xml)
    out_dir = Path(args.out)

    if not xml_path.exists():
        print(f"[오류] XML 파일을 찾을 수 없습니다: {xml_path}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[파싱 시작] {xml_path} ({xml_path.stat().st_size / 1e9:.2f} GB)")
    print(f"[출력 경로] {out_dir}/")
    if not HAS_LXML:
        print("[경고] lxml 미설치 - 표준 xml.etree 사용 (메모리 사용량 증가)")

    df_drugs, df_ddi, df_cyp = build_dataframes(xml_path, verbose=True)

    # DDI 중복 제거
    if not args.no_dedup:
        before = len(df_ddi)
        df_ddi = deduplicate_ddi(df_ddi)
        after = len(df_ddi)
        print(f"[중복 제거] DDI: {before:,} → {after:,} ({before - after:,} 제거)")

    # 저장
    drugs_path = out_dir / "drugbank_drugs.parquet"
    ddi_path = out_dir / "drugbank_ddi.parquet"
    cyp_path = out_dir / "drugbank_cyp.parquet"

    df_drugs.to_parquet(drugs_path, index=False)
    df_ddi.to_parquet(ddi_path, index=False)
    df_cyp.to_parquet(cyp_path, index=False)

    print(f"\n[결과 요약]")
    print(f"  약물 수        : {len(df_drugs):>8,}")
    print(f"  DDI 쌍 수     : {len(df_ddi):>8,}")
    print(f"    Contraindicated: {(df_ddi['severity'] == 'Contraindicated').sum():>6,}")
    print(f"    Major          : {(df_ddi['severity'] == 'Major').sum():>6,}")
    print(f"    Moderate       : {(df_ddi['severity'] == 'Moderate').sum():>6,}")
    print(f"    Minor          : {(df_ddi['severity'] == 'Minor').sum():>6,}")
    print(f"  CYP 레코드 수  : {len(df_cyp):>8,}")
    print(f"\n[저장 완료]")
    print(f"  {drugs_path}")
    print(f"  {ddi_path}")
    print(f"  {cyp_path}")


if __name__ == "__main__":
    main()
