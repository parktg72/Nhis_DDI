"""Smoke prescription-history provider for DL serving verification."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from serving.hana_history import validate_history_frame


@dataclass(frozen=True)
class SmokeHistoryProvider:
    """Return deterministic D1/D2 history rows for smoke DL verification."""

    include_unknown: bool = False

    def fetch_patient_history(
        self,
        patient_id: str,
        reference_date: date,
        lookback_days: int,
    ) -> pd.DataFrame:
        del lookback_days
        events = [
            (str(patient_id), "D1", reference_date - timedelta(days=1)),
            (str(patient_id), "D2", reference_date),
        ]
        if self.include_unknown:
            events.append((str(patient_id), "UNKNOWN_SMOKE", reference_date))

        history = pd.DataFrame(
            [
                {
                    "patient_id": pid,
                    "drug_code": drug_code,
                    "prescription_date": when.strftime("%Y%m%d"),
                }
                for pid, drug_code, when in events
            ]
        )
        validate_history_frame(history, context="smoke history")
        return history
