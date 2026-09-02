"""A3 재측정 — 엔진의 DrugMatcher 를 직접 호출한다.

1차 측정은 name_keywords 부분일치만 재구현해 atc_prefixes 경로를 빠뜨렸고,
TOP09(requires_count:3)·TOP03(requires_triple)의 발화 조건도 다른 규칙과
같은 "각 그룹 1건"으로 잘못 적용했다. 여기서는 매칭을 재구현하지 않고
rules.safety_net 의 matcher 와 각 규칙의 실제 분기 조건을 그대로 쓴다.

측정 대상은 코퍼스 상한(하루치 EDI 전량을 한 환자가 받은 극단 가정)이지
운영 발화율이 아니다. e2e 도 아니다.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from rules.safety_net import SafetyNet
from scripts.etl.code_standardizer import CodeStandardizer

RECORDS = ROOT / "data" / "Raw" / "records_20240701.parquet"

def resolve_names(edis, cs):
    """serving 플래그 ON 경로와 같은 사슬: lookup_edi → get_wk → lookup_wk."""
    names, resolved = set(), 0
    for edi in edis:
        _, name = cs.lookup_edi(edi)
        if not name:
            wk = cs.get_wk(edi)
            if wk:
                _, name = cs.lookup_wk(wk)
        if name:
            names.add(name)
            resolved += 1
    return names, resolved

def main():
    df = pd.read_parquet(RECORDS, columns=["edi_code"])
    edis = sorted({str(e).strip() for e in df["edi_code"].dropna().unique() if str(e).strip()})

    cs = CodeStandardizer()
    names, resolved = resolve_names(edis, cs)
    drugs = sorted(names)

    sn = SafetyNet()
    m = sn._matcher

    def in_group(g):
        if isinstance(g, list):
            return [d for sub in g for d in m.drugs_in_group(drugs, sub)]
        return m.drugs_in_group(drugs, g)

    out = {
        "measured_edi_unique": len(edis),
        "resolved_edi": resolved,
        "resolve_rate": round(resolved / len(edis), 4),
        "distinct_names": len(drugs),
        "rules": {},
    }

    for rule in sn._top10_rules:
        rid, entry = rule["id"], {"name": rule["name"]}

        if rule.get("requires_count"):
            hits = in_group(rule["drug_group_a"])
            need = rule["requires_count"]
            entry.update(condition=f"group_a >= {need}", found=len(hits),
                         fires=len(hits) >= need, sample=sorted(hits)[:5])

        elif rule.get("requires_triple"):
            # 엔진은 yaml 의 group_a/b/c 가 아니라 acei/arb + k_sparing + nsaids 를 본다
            rasi = in_group("acei") + in_group("arb")
            k = in_group("k_sparing_diuretics")
            ns = in_group("nsaids")
            entry.update(condition="(acei|arb) & k_sparing & nsaids",
                         found={"rasi": len(rasi), "k_sparing": len(k), "nsaids": len(ns)},
                         fires=bool(rasi and k and ns),
                         sample={"rasi": sorted(rasi)[:3], "k_sparing": sorted(k)[:3],
                                 "nsaids": sorted(ns)[:3]})
        else:
            a, b = in_group(rule.get("drug_group_a", "")), in_group(rule.get("drug_group_b", ""))
            entry.update(condition="group_a & group_b",
                         found={"a": len(a), "b": len(b)}, fires=bool(a and b),
                         sample={"a": sorted(a)[:3], "b": sorted(b)[:3]})

        out["rules"][rid] = entry

    out["fires"] = sorted(r for r, v in out["rules"].items() if v["fires"])
    out["not_fires"] = sorted(r for r, v in out["rules"].items() if not v["fires"])
    (ROOT / "a3_coverage.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"고유 EDI {out['measured_edi_unique']:,} · 해소 {resolved:,} "
          f"({out['resolve_rate']:.1%}) · 고유 약물명 {len(drugs):,}")
    for rid, v in out["rules"].items():
        print(f"  {rid} {'발화가능' if v['fires'] else '미발화  '} "
              f"[{v['condition']}] {v['found']}")
    print(f"\n발화 가능 {len(out['fires'])}종: {' '.join(out['fires'])}")
    print(f"미발화 {len(out['not_fires'])}종: {' '.join(out['not_fires'])}")

if __name__ == "__main__":
    main()
