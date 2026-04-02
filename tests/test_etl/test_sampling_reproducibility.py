from __future__ import annotations

from pathlib import Path

import pandas as pd

from hana_app.core.ml_runner import stratify_and_sample_patients
from hana_app.core.sas_reader import SASExtractor


def _write_mock_eligibility_csv(path: Path) -> None:
    rows: list[dict[str, object]] = []
    pid = 0
    for sex in ("M", "F"):
        for byear in (1948, 1958, 1972, 1988, 2004):
            for addr in ("11000", "26000", "41000"):
                for _ in range(18):
                    pid += 1
                    rows.append({
                        "INDI_DSCM_NO": f"P{pid:05d}",
                        "BYEAR": byear,
                        "SEX_TYPE": sex,
                        "RVSN_ADDR_CD": addr,
                        "STD_YYYY": "2023",
                    })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_sampling_reproducibility_with_sas_backend(tmp_path, monkeypatch):
    csv_path = tmp_path / "eligibility_mock.csv"
    sas_path = tmp_path / "eligibility_mock.sas7bdat"
    _write_mock_eligibility_csv(csv_path)
    sas_path.touch()

    def _mock_read_sas_chunks(file_path, encoding="cp949", usecols=None, chunksize=100_000):
        for chunk in pd.read_csv(csv_path, dtype=str, chunksize=chunksize):
            if usecols:
                chunk = chunk[[c for c in usecols if c in chunk.columns]]
            yield chunk

    monkeypatch.setattr("hana_app.core.sas_reader.read_sas_chunks", _mock_read_sas_chunks)

    cfg = {
        "seed": 17,
        "std_year": "2023",
        "reference_year": 2023,
        "addr_digits": 5,
        "prefetch_size": 120,
        "sample_size": 80,
    }
    extractor = SASExtractor(
        {
            "folder": str(tmp_path),
            "files": {"eligibility": sas_path.name},
            "encoding": "utf-8",
            "chunksize": 31,
        },
        {
            "eligibility": {
                "patient_id": "INDI_DSCM_NO",
                "byear": "BYEAR",
                "sex_type": "SEX_TYPE",
                "std_year": "STD_YYYY",
                "rvsn_addr_cd": "RVSN_ADDR_CD",
            }
        },
    )

    elig_df_1 = extractor.fetch_eligibility_for_sampling(
        std_year=cfg["std_year"],
        addr_digits=cfg["addr_digits"],
        sample_size=cfg["prefetch_size"],
        seed=cfg["seed"],
    )
    sampled_pids_1, _, _ = stratify_and_sample_patients(
        elig_df=elig_df_1,
        sample_size=cfg["sample_size"],
        reference_year=cfg["reference_year"],
        seed=cfg["seed"],
    )

    elig_df_2 = extractor.fetch_eligibility_for_sampling(
        std_year=cfg["std_year"],
        addr_digits=cfg["addr_digits"],
        sample_size=cfg["prefetch_size"],
        seed=cfg["seed"],
    )
    sampled_pids_2, _, _ = stratify_and_sample_patients(
        elig_df=elig_df_2,
        sample_size=cfg["sample_size"],
        reference_year=cfg["reference_year"],
        seed=cfg["seed"],
    )

    assert len(sampled_pids_1) == cfg["sample_size"]
    assert sorted(sampled_pids_1) == sorted(sampled_pids_2)
