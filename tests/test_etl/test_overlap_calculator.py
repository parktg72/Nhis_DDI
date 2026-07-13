"""
overlap_calculator 단위 테스트
핵심 알고리즘: 동시복용 기간 계산
"""
from __future__ import annotations

from datetime import date, timedelta

from scripts.etl.models import PrescriptionRecord
from scripts.etl.overlap_calculator import (
    _overlap_days,
    calculate_overlaps_for_patient,
    get_concurrent_drug_count,
)


def make_rx(
    edi: str,
    start: date,
    days: int,
    atc: str | None = None,
    patient: str = "P000001",
) -> PrescriptionRecord:
    return PrescriptionRecord(
        patient_id=patient,
        institution_id="INST0001",
        bill_no=f"BILL_{edi}",
        wk_compn_cd=edi,
        edi_code=edi,
        atc_code=atc,
        drug_name=edi,
        start_date=start,
        end_date=start + timedelta(days=days - 1),
        total_days=days,
        dose_once=1.0,
        dose_freq=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _overlap_days
# ─────────────────────────────────────────────────────────────────────────────

class TestOverlapDays:
    def test_no_overlap(self):
        a_start, a_end = date(2024, 1, 1), date(2024, 1, 10)
        b_start, b_end = date(2024, 1, 20), date(2024, 1, 30)
        assert _overlap_days(a_start, a_end, b_start, b_end) == 0

    def test_adjacent_no_overlap(self):
        """끝과 시작이 딱 붙어있으면 겹치지 않음."""
        a_start, a_end = date(2024, 1, 1), date(2024, 1, 10)
        b_start, b_end = date(2024, 1, 11), date(2024, 1, 20)
        assert _overlap_days(a_start, a_end, b_start, b_end) == 0

    def test_one_day_overlap(self):
        a_start, a_end = date(2024, 1, 1), date(2024, 1, 10)
        b_start, b_end = date(2024, 1, 10), date(2024, 1, 20)
        assert _overlap_days(a_start, a_end, b_start, b_end) == 1

    def test_full_overlap(self):
        """B가 A를 완전히 포함."""
        a_start, a_end = date(2024, 1, 5), date(2024, 1, 15)
        b_start, b_end = date(2024, 1, 1), date(2024, 1, 20)
        assert _overlap_days(a_start, a_end, b_start, b_end) == 11

    def test_symmetric(self):
        """순서 교환해도 결과 동일."""
        a_start, a_end = date(2024, 1, 1), date(2024, 1, 15)
        b_start, b_end = date(2024, 1, 10), date(2024, 1, 25)
        assert (
            _overlap_days(a_start, a_end, b_start, b_end)
            == _overlap_days(b_start, b_end, a_start, a_end)
        )


# ─────────────────────────────────────────────────────────────────────────────
# calculate_overlaps_for_patient
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateOverlaps:
    BASE = date(2024, 1, 1)

    def test_single_drug_no_pair(self):
        rxs = [make_rx("D001", self.BASE, 30)]
        pairs = calculate_overlaps_for_patient(rxs)
        assert pairs == []

    def test_two_drugs_overlapping(self):
        rxs = [
            make_rx("D001", self.BASE, 30),
            make_rx("D002", self.BASE + timedelta(days=10), 30),
        ]
        pairs = calculate_overlaps_for_patient(rxs, min_overlap=7)
        assert len(pairs) == 1
        assert pairs[0].overlap_days == 20  # 10~29일 = 20일

    def test_two_drugs_short_overlap(self):
        """중첩 2일 (MIN_OVERLAP_DAYS=7 미만) → 미탐지."""
        rxs = [
            make_rx("D001", self.BASE, 10),                       # 1/1 ~ 1/10
            make_rx("D002", self.BASE + timedelta(days=8), 10),   # 1/9 ~ 1/18
        ]
        # max(1/1,1/9)=1/9, min(1/10,1/18)=1/10 → 겹침 2일 → 탐지 안됨
        pairs = calculate_overlaps_for_patient(rxs, min_overlap=7)
        assert len(pairs) == 0

    def test_no_overlap_different_periods(self):
        rxs = [
            make_rx("D001", self.BASE, 10),
            make_rx("D002", self.BASE + timedelta(days=20), 10),
        ]
        pairs = calculate_overlaps_for_patient(rxs, min_overlap=7)
        assert pairs == []

    def test_same_drug_excluded(self):
        """동일 EDI 코드 쌍은 제외."""
        rxs = [
            make_rx("D001", self.BASE, 30),
            make_rx("D001", self.BASE + timedelta(days=5), 30),  # 동일 약
        ]
        pairs = calculate_overlaps_for_patient(rxs)
        assert pairs == []

    def test_three_drugs_three_pairs(self):
        """3종 약물 → 최대 3쌍 가능."""
        rxs = [
            make_rx("D001", self.BASE, 60),
            make_rx("D002", self.BASE, 60),
            make_rx("D003", self.BASE, 60),
        ]
        pairs = calculate_overlaps_for_patient(rxs, min_overlap=7)
        assert len(pairs) == 3

    def test_atc_codes_preserved(self):
        """ATC 코드가 쌍에 올바르게 전달되는지."""
        rxs = [
            make_rx("D001", self.BASE, 30, atc="B01AA03"),
            make_rx("D002", self.BASE, 30, atc="M01AE01"),
        ]
        pairs = calculate_overlaps_for_patient(rxs)
        assert len(pairs) == 1
        atcs = {pairs[0].drug_a_atc, pairs[0].drug_b_atc}
        assert atcs == {"B01AA03", "M01AE01"}

    def test_dedup_pairs(self):
        """같은 약물 쌍이 여러 윈도우에서 나와도 중복 제거."""
        rxs = [
            make_rx("D001", self.BASE, 90),
            make_rx("D002", self.BASE + timedelta(days=5), 90),
        ]
        pairs = calculate_overlaps_for_patient(rxs)
        assert len(pairs) == 1


# ─────────────────────────────────────────────────────────────────────────────
# get_concurrent_drug_count
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrentCount:
    BASE = date(2024, 1, 15)

    def test_all_active(self):
        rxs = [
            make_rx("D001", self.BASE - timedelta(days=5), 30),
            make_rx("D002", self.BASE - timedelta(days=3), 30),
            make_rx("D003", self.BASE - timedelta(days=1), 30),
        ]
        assert get_concurrent_drug_count(rxs, self.BASE) == 3

    def test_some_expired(self):
        rxs = [
            make_rx("D001", self.BASE - timedelta(days=20), 10),  # 만료됨
            make_rx("D002", self.BASE - timedelta(days=5), 30),   # 활성
        ]
        assert get_concurrent_drug_count(rxs, self.BASE) == 1

    def test_future_prescription(self):
        rxs = [
            make_rx("D001", self.BASE + timedelta(days=5), 30),  # 아직 시작 안함
            make_rx("D002", self.BASE - timedelta(days=5), 30),  # 활성
        ]
        assert get_concurrent_drug_count(rxs, self.BASE) == 1
