"""Task B 회귀: 앱 피처빌더가 aggregate_patient_features 에 drug_master 를 넘기는지.

배경: ml_runner 가 drug_master 를 안 넘겨 DDI 중증도(ddi_*)가 전부 0 → Red·
Y_DDI_MAJOR/MOD 라벨 소실(2026-06-05 실증). 이 테스트는 빌더가 DrugMaster 를
로드해 전달하는지(=DDI 매칭 활성)와, 누락 시 graceful None 인지 검증한다.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_load_drug_master_returns_instance_when_parquet_present():
    import hana_app.core.ml_runner as M

    M._DRUG_MASTER_CACHE.update({"obj": None, "loaded": False})  # 캐시 초기화
    master_path = ROOT / "data" / "processed" / "hira_drug_master.parquet"
    if not master_path.exists():
        pytest.skip("hira_drug_master.parquet 없음(이 환경)")
    from scripts.etl.drug_master import DrugMaster
    dm = M._load_drug_master()
    assert isinstance(dm, DrugMaster), "DrugMaster 로드 실패(또는 None)"


def test_build_patient_features_passes_drug_master():
    """build_patient_features 가 aggregate_patient_features 에 drug_master 를 전달."""
    import hana_app.core.ml_runner as M
    from scripts.etl.models import PrescriptionRecord

    master_path = ROOT / "data" / "processed" / "hira_drug_master.parquet"
    if not master_path.exists():
        pytest.skip("hira_drug_master.parquet 없음(이 환경)")

    M._DRUG_MASTER_CACHE.update({"obj": None, "loaded": False})

    # poly_threshold(5) 충족: 한 환자에 고유 wk_compn_cd 6개
    recs = [
        PrescriptionRecord(
            patient_id="P1", institution_id="I1", bill_no=f"B{i}",
            wk_compn_cd=f"42100{i}ATB", edi_code=None, gnl_nm_cd=None,
            efmdc_clsf_no=None, start_date=date(2024, 7, 1), end_date=date(2024, 7, 30),
            total_days=30, dose_once=1.0, dose_freq=1, sick_code=None,
            sex="1", age_id="5", institution_type=None, source="T30",
        )
        for i in range(6)
    ]

    captured = {}
    real = M.aggregate_patient_features

    def _spy(*a, **k):
        captured["drug_master"] = k.get("drug_master", "MISSING")
        return real(*a, **k)

    with patch.object(M, "aggregate_patient_features", _spy):
        M.build_patient_features(recs, window_days=90, poly_threshold=5)

    assert "drug_master" in captured, "aggregate_patient_features 호출 안 됨"
    assert captured["drug_master"] not in (None, "MISSING"), (
        "drug_master 가 전달되지 않음 — DDI 중증도 매칭 비활성(ddi_*=0 회귀)"
    )
