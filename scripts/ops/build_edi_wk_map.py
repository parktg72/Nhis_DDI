"""EDI(제품코드) → WK_COMPN_CD(주성분코드) 매핑 빌더 (Task B serving DDI parity).

서빙은 요청에서 edi_code 만 받지만, 학습 DDI 경로는 wk_compn_cd(주성분코드) 기반
(DrugMaster.get_ddi_ids). HIRA 약제급여목록 xlsx 는 제품코드(EDI)와 주성분코드(WK)를
한 행에 함께 갖는다. 이 스크립트는 그 둘만 추출해 **검증된 edi→wk parquet** 를 만든다.

설계 합의(Codex Q6③): 서빙에서 xlsx 즉석 파싱 금지 → 사전 빌드한 parquet 를 배포·로드.
edi→wk 는 함수(다수 제품 → 한 주성분코드)여야 한다. 한 edi 가 서로 다른 wk 로 충돌하면
빌드 실패(데이터 이상 가시화).

산출: data/processed/edi_to_wk.parquet  (cols: edi_code[str9], wk_compn_cd[str])
      + .meta.json (source_sha256, row counts, build time)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_XLSX = Path("hira/약제급여목록및급여상한금액표.xlsx")
DEFAULT_OUT = Path("data/processed/edi_to_wk.parquet")

EDI_COL = "제품코드"
WK_COL = "주성분코드"
EFMDC_COL = "분류"   # 약효분류번호 (records efmdc_clsf_no 와 99.9% 동일 — dup_efmdc 정합용)


def normalize_edi(value) -> str | None:
    """제품코드를 9자리 문자열로 정규화. 비숫자/빈값은 None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s in ("", "nan", "None"):
        return None
    # 엑셀이 int/float 로 읽은 경우 소수점 제거
    if s.endswith(".0"):
        s = s[:-2]
    if not s.isdigit():
        return None
    return s.zfill(9)


def normalize_wk(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    return s or None


def build_edi_wk_map(xlsx_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """HIRA xlsx → edi→wk DataFrame + 메타. edi 충돌 시 ValueError."""
    xlsx = Path(xlsx_path)
    if not xlsx.exists():
        raise FileNotFoundError(f"HIRA xlsx 없음: {xlsx}")
    raw = pd.read_excel(xlsx)
    if EDI_COL not in raw.columns or WK_COL not in raw.columns:
        raise ValueError(
            f"xlsx 에 '{EDI_COL}'/'{WK_COL}' 컬럼 없음. 실제 컬럼: {list(raw.columns)}"
        )

    has_efmdc = EFMDC_COL in raw.columns
    efmdc_series = raw[EFMDC_COL] if has_efmdc else [None] * len(raw)

    rows: dict[str, str] = {}            # edi → wk
    efmdc: dict[str, str] = {}           # edi → efmdc(분류, nullable)
    conflicts: list[tuple[str, str, str]] = []
    n_raw = 0
    for edi_raw, wk_raw, ef_raw in zip(raw[EDI_COL], raw[WK_COL], efmdc_series):
        edi = normalize_edi(edi_raw)
        wk = normalize_wk(wk_raw)
        if edi is None or wk is None:
            continue
        n_raw += 1
        prev = rows.get(edi)
        if prev is None:
            rows[edi] = wk
        elif prev != wk:
            conflicts.append((edi, prev, wk))
        ef = normalize_wk(ef_raw)        # 숫자/문자 그대로, .0 만 정리
        if ef is not None and ef.endswith(".0"):
            ef = ef[:-2]
        if ef is not None:
            efmdc.setdefault(edi, ef)

    if conflicts:
        sample = conflicts[:10]
        raise ValueError(
            f"edi→wk 함수 위반: {len(conflicts)}개 edi 가 복수 wk 로 충돌. 예: {sample}"
        )

    if rows:
        df = pd.DataFrame(
            [(edi, wk, efmdc.get(edi)) for edi, wk in sorted(rows.items())],
            columns=["edi_code", "wk_compn_cd", "efmdc_clsf_no"],
        )
    else:
        df = pd.DataFrame(columns=["edi_code", "wk_compn_cd", "efmdc_clsf_no"])
    meta = {
        "source_xlsx": str(xlsx),
        "source_sha256": hashlib.sha256(xlsx.read_bytes()).hexdigest(),
        "valid_rows": n_raw,
        "unique_edi": len(df),
        "unique_wk": int(df["wk_compn_cd"].nunique()) if len(df) else 0,
        "edi_with_efmdc": int(df["efmdc_clsf_no"].notna().sum()) if len(df) else 0,
    }
    return df, meta


def write_edi_wk_map(df: pd.DataFrame, meta: dict, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    (out.with_suffix(".meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    p = argparse.ArgumentParser(description="EDI→WK 매핑 빌더 (HIRA xlsx → parquet)")
    p.add_argument("--xlsx", default=str(DEFAULT_XLSX))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args(argv)

    df, meta = build_edi_wk_map(args.xlsx)
    out = write_edi_wk_map(df, meta, args.out)
    print(f"[OK] {out}")
    print(f"unique_edi={meta['unique_edi']} unique_wk={meta['unique_wk']} "
          f"valid_rows={meta['valid_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
