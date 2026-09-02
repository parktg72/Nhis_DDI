"""미발화 4종의 원인 분류 - 참조DB 부재인가, 해소 결함인가.

사슬: wk_compn_cd → DrugMaster 성분명 → _name_to_ddi_id → ddi_id → _edi_map → drug_name
각 미발화 그룹의 대표 성분에 대해 이 사슬이 어느 단계에서 끊기는지 센다.
  · 하루치에 처방 0건        → 데이터 부재 (#14 잔여)
  · 처방은 있는데 이름 0건   → 해소 결함 (고칠 수 있음)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from scripts.etl.code_standardizer import CodeStandardizer
from scripts.etl.drug_master import DrugMaster

RECORDS = ROOT / "data" / "Raw" / "records_20240701.parquet"

TARGETS = {
    "TOP03 / k_sparing_diuretics": ["spironolactone", "eplerenone", "amiloride", "triamterene",
                                    "스피로노락톤", "에플레레논", "아밀로라이드", "트리암테렌"],
    "TOP04 / amiodarone·verapamil": ["amiodarone", "verapamil", "아미오다론", "베라파밀"],
    "TOP05 / methotrexate":        ["methotrexate", "메토트렉세이트", "메토트렉산"],
    "TOP10 / statin":              ["atorvastatin", "rosuvastatin", "simvastatin", "pravastatin",
                                    "pitavastatin", "아토르바스타틴", "로수바스타틴",
                                    "심바스타틴", "프라바스타틴", "피타바스타틴"],
    "TOP10 / macrolide_strong":    ["clarithromycin", "erythromycin", "클래리스로마이신",
                                    "클라리스로마이신", "에리스로마이신"],
}

def main():
    df = pd.read_parquet(RECORDS, columns=["edi_code", "wk_compn_cd"])
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    day_wk = df.groupby("wk_compn_cd")["edi_code"].first().to_dict()
    day_wk.pop("", None); day_wk.pop("nan", None)

    cs = CodeStandardizer()
    dm = cs._master  # serving 과 동일한 로딩 경로를 그대로 쓴다
    print(f"DrugMaster: 주성분코드 {len(dm._code_to_components):,} · "
          f"DDI 성분명 인덱스 {len(dm._name_to_ddi_id):,}")
    print(f"하루치 고유 주성분코드 {len(day_wk):,}\n")

    for label, kws in TARGETS.items():
        low = [k.lower() for k in kws]
        master_hits = {c: raw for c, raw in dm._code_to_raw.items()
                       if any(k in raw.lower() for k in low)}
        in_day = {c: day_wk[c] for c in master_hits if c in day_wk}

        stage = {"이름해소": 0, "DDI ID 미연결": 0, "성분 파싱 없음": 0}
        ex = []
        for wk, edi in sorted(in_day.items()):
            _, nm = cs.lookup_edi(edi)
            if nm:
                stage["이름해소"] += 1
                if len(ex) < 2: ex.append(f"{wk}/{edi} → {nm} (lookup_edi)")
                continue
            _, nm = cs.lookup_wk(wk)
            if nm:
                stage["이름해소"] += 1
                if len(ex) < 2: ex.append(f"{wk}/{edi} → {nm} (lookup_wk)")
                continue
            comps = dm.get_components(wk)
            if not comps:
                stage["성분 파싱 없음"] += 1
            else:
                stage["DDI ID 미연결"] += 1
                if len(ex) < 2:
                    ex.append(f"{wk}/{edi} 성분={comps[:2]} → DDI ID 미연결 "
                              f"(원문: {master_hits[wk][:40]})")

        print(f"■ {label}")
        print(f"   DrugMaster 주성분코드 {len(master_hits):,}건 · 하루치 처방 등장 {len(in_day):,}건")
        print(f"   → {stage}")
        for e in ex: print(f"     예: {e}")
        if not in_day:
            print("   판정: 데이터 부재 - 하루치에 처방 자체가 없음")
        elif stage["이름해소"] == 0:
            print("   판정: 해소 결함 - 처방은 있으나 이름이 안 나옴")
        else:
            print(f"   판정: {stage['이름해소']}건 해소됨 - 그룹 키워드/성분 선택 확인 필요")
        print()

if __name__ == "__main__":
    main()
