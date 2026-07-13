"""scripts.ops.measure_edi_wk_coverage — 레코드 구성 로직 스모크."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.etl.models import PrescriptionRecord
from scripts.ops import measure_edi_wk_coverage as MC


def test_rec_parses_iso_dates_and_wk():
    row = {
        "patient_id": "P1", "institution_id": "I1", "bill_no": "B1",
        "edi_code": "660700010", "wk_compn_cd": "IGNORED_IN_REC_ARG",
        "start_date": "2024-07-01", "end_date": "2024-07-30", "total_days": 30,
    }
    rec = MC._rec(row, wk="421001ATB")
    assert isinstance(rec, PrescriptionRecord)
    assert rec.wk_compn_cd == "421001ATB"          # wk 는 인자로 주입(full vs serving 분기용)
    assert rec.edi_code == "660700010"
    assert rec.start_date == date(2024, 7, 1)
    assert rec.end_date == date(2024, 7, 30)
    assert rec.total_days == 30
    assert rec.source == "T30"


def test_rec_handles_missing_optional_fields():
    row = {
        "patient_id": "P2", "edi_code": "1", "wk_compn_cd": "X",
        "start_date": "2024-07-01", "end_date": "2024-07-01", "total_days": 1,
    }
    rec = MC._rec(row, wk="W")
    assert rec.institution_id == ""   # 누락 시 빈 문자열
    assert rec.bill_no == ""
    assert rec.start_date == rec.end_date
