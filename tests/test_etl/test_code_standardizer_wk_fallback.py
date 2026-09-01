"""`CodeStandardizer.lookup_wk` 반환 계약 — ATC 유무가 약물명을 좌우하면 안 된다.

`lookup_wk` 는 EDI→ATC 매핑 실패 시의 폴백이며, 서빙에서는 이 함수가 돌려주는
**약물명**이 이름 기반 Rule Safety Net(Top-10·QT·고위험약)의 유일한 입력이다.

종전 구현은 `if atc_list:` 로 반환 전체를 게이팅해, 성분은 해소됐는데 그 인덱스
엔트리에 ATC 가 없으면 확보한 약물명까지 버렸다. 실 데이터에서 이 형태는 드물지 않고
(하루치 고유 EDI 15,017개 중 429건) 버려지는 약물에 `aspirin` 이 포함된다 —
와파린 병용 시 TOP01 대상이다.

이 파일은 실 참조DB 없이도 그 계약을 고정한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.etl.code_standardizer import CodeStandardizer

# ATC 가 없는 엔트리 (실 데이터의 aspirin/D000452 형태)
_WK_NO_ATC = "111001ATE"
# ATC 가 있는 엔트리 (대조군)
_WK_WITH_ATC = "249103ATB"


@pytest.fixture
def standardizer(tmp_path):
    """실 참조DB 없이 wk→성분→DDI ID→이름 사슬만 갖춘 CodeStandardizer."""
    ddi = tmp_path / "ddi.parquet"
    pd.DataFrame(
        [
            {"drug_a_name": "Aspirin", "drug_a_id": "D000452",
             "drug_b_name": "Warfarin", "drug_b_id": "DB00682",
             "severity": "Major", "description": "bleeding", "source": "TEST"},
        ]
    ).to_parquet(ddi, index=False)

    master = tmp_path / "master.parquet"
    pd.DataFrame(
        [
            {"ingr_code": _WK_NO_ATC, "is_combo": False,
             "ingr_name_raw": "aspirin 100mg", "components": "aspirin", "ingr_count": 1},
            {"ingr_code": _WK_WITH_ATC, "is_combo": False,
             "ingr_name_raw": "warfarin 5mg", "components": "warfarin", "ingr_count": 1},
        ]
    ).to_parquet(master, index=False)

    index = tmp_path / "drug_name_index.parquet"
    pd.DataFrame(
        [
            # ATC 없음 — 빈 문자열로 둔다(실 인덱스의 결측 형태)
            {"drugbank_id": "D000452", "drug_name": "Aspirin", "atc_codes": ""},
            {"drugbank_id": "DB00682", "drug_name": "Warfarin", "atc_codes": "B01AA03"},
        ]
    ).to_parquet(index, index=False)

    return CodeStandardizer(
        index_path=index,
        extra_csv=None,
        master_parquet=master,
        ddi_matrix_path=ddi,
        edi_wk_path=tmp_path / "absent_edi_wk.parquet",
    )


def test_lookup_wk_returns_name_when_entry_has_no_atc(standardizer):
    """ATC 없는 엔트리라도 약물명은 반환해야 한다 — 규칙 발화의 유일한 입력이다."""
    atc, name = standardizer.lookup_wk(_WK_NO_ATC)

    assert name == "Aspirin", f"ATC 가 없다는 이유로 약물명이 폐기됐다 — name={name!r}"
    assert atc is None, f"없는 ATC 를 지어내면 안 된다 — atc={atc!r}"


def test_lookup_wk_still_returns_atc_when_present(standardizer):
    """대조군 — ATC 가 있으면 종전과 동일하게 (atc, name) 을 반환한다."""
    atc, name = standardizer.lookup_wk(_WK_WITH_ATC)

    assert atc == "B01AA03"
    assert name == "Warfarin"


def test_lookup_wk_returns_nothing_for_unknown_code(standardizer):
    """미등록 주성분코드는 종전과 동일하게 (None, None)."""
    assert standardizer.lookup_wk("NOPE999") == (None, None)
    assert standardizer.lookup_wk("") == (None, None)
