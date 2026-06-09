import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.ops.extract_raw_subset import (
    InputError,
    PolicyError,
    QueryError,
    discover_raw_parquets,
    extract_subset,
    normalize_cli_path,
    parse_date_range,
    load_condition,
)


def _write_parquet(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _record(patient_id, day, wk="W1", edi="E1", inst="I1"):
    return {
        "patient_id": patient_id,
        "institution_id": inst,
        "bill_no": f"B-{patient_id}-{day}",
        "wk_compn_cd": wk,
        "edi_code": edi,
        "gnl_nm_cd": "G",
        "efmdc_clsf_no": "C",
        "start_date": day,
        "end_date": day,
        "total_days": 1,
        "dose_once": 1.0,
        "dose_freq": 1,
        "sick_code": "S",
        "sex": "M",
        "age_id": 70,
        "institution_type": "clinic",
        "source": "test",
    }


def _make_raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    _write_parquet(
        raw / "records_20240901.parquet",
        [_record("P001", "2024-09-01", "W1"), _record("P002", "2024-09-01", "W2")],
    )
    _write_parquet(raw / "records_20240902.parquet", [_record("P001", "2024-09-02", "W2")])
    _write_parquet(
        raw / "eligibility_demographics.parquet",
        [
            {"patient_id": "P001", "byear": 1950, "age": 74, "sex_type": "M", "addr_cd": "11"},
            {"patient_id": "P002", "byear": 1955, "age": 69, "sex_type": "F", "addr_cd": "26"},
            {"patient_id": "P999", "byear": 1960, "age": 64, "sex_type": "M", "addr_cd": "99"},
        ],
    )
    return raw


def _base_config():
    return {
        "version": 1,
        "project": "MODE_11_hana",
        "request_id": "test-run",
        "purpose": "analysis",
        "date_range": {"start": "2024-09-01", "end": "2024-09-02"},
        "filters": {},
        "outputs": {},
    }


def test_windows_path_normalization_under_wsl():
    got = normalize_cli_path(r"C:\model\MODE_11_hana")
    if sys.platform.startswith("linux"):
        assert got.as_posix() == "/mnt/c/model/MODE_11_hana"


def test_policy_rejects_jan_2025():
    cfg = _base_config()
    cfg["date_range"] = {"start": "2025-01-01", "end": "2025-01-01"}
    from scripts.ops.extract_raw_subset import enforce_policy

    with pytest.raises(PolicyError):
        enforce_policy(cfg, parse_date_range(cfg))


def test_policy_rejects_tuning_on_dec_holdout():
    cfg = _base_config()
    cfg["purpose"] = "tuning"
    cfg["date_range"] = {"start": "2024-12-01", "end": "2024-12-02"}
    from scripts.ops.extract_raw_subset import enforce_policy

    with pytest.raises(PolicyError):
        enforce_policy(cfg, parse_date_range(cfg))


def test_discover_raw_parquets_reports_missing_day(tmp_path):
    raw = _make_raw_dir(tmp_path)
    cfg = _base_config()
    cfg["date_range"] = {"start": "2024-09-01", "end": "2024-09-03"}
    with pytest.raises(InputError, match="records_20240903.parquet"):
        discover_raw_parquets(raw, parse_date_range(cfg))


def test_extract_subset_by_patient_and_drug_writes_records_demographics_and_manifest(tmp_path):
    raw = _make_raw_dir(tmp_path)
    cfg = _base_config()
    cfg["filters"] = {"patient_ids": ["P001"], "wk_compn_cd": ["W2"]}

    result = extract_subset(cfg, raw, tmp_path / "out", tmp_path, confirm_sensitive=True)

    assert result.row_count == 1
    records = pd.read_parquet(tmp_path / "out" / "raw" / "records_subset.parquet")
    assert records["patient_id"].tolist() == ["P001"]
    assert records["wk_compn_cd"].tolist() == ["W2"]
    demo = pd.read_parquet(tmp_path / "out" / "raw" / "eligibility_demographics.parquet")
    assert demo["patient_id"].tolist() == ["P001"]
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["output"]["row_count"] == 1
    assert manifest["privacy"]["patient_ids_logged"] is False
    assert "P001" not in json.dumps(manifest, ensure_ascii=False)


def test_empty_match_fails_without_allow_empty(tmp_path):
    raw = _make_raw_dir(tmp_path)
    cfg = _base_config()
    cfg["filters"] = {"patient_ids": ["P404"]}
    with pytest.raises(QueryError, match="zero rows"):
        extract_subset(cfg, raw, tmp_path / "out", tmp_path, confirm_sensitive=True)


def test_unknown_filter_key_fails_closed(tmp_path):
    raw = _make_raw_dir(tmp_path)
    cfg = _base_config()
    cfg["filters"] = {"patient_id_typo": ["P001"]}
    with pytest.raises(InputError, match="unsupported filter keys"):
        extract_subset(cfg, raw, tmp_path / "out", tmp_path, confirm_sensitive=True)


def test_row_level_date_filter_excludes_misfiled_rows(tmp_path):
    raw = tmp_path / "raw"
    _write_parquet(
        raw / "records_20240901.parquet",
        [_record("P001", "2024-09-01"), _record("P002", "2024-10-01")],
    )
    _write_parquet(
        raw / "eligibility_demographics.parquet",
        [
            {"patient_id": "P001", "byear": 1950, "age": 74, "sex_type": "M", "addr_cd": "11"},
            {"patient_id": "P002", "byear": 1955, "age": 69, "sex_type": "F", "addr_cd": "26"},
        ],
    )
    cfg = _base_config()
    cfg["date_range"] = {"start": "2024-09-01", "end": "2024-09-01"}

    result = extract_subset(cfg, raw, tmp_path / "out", tmp_path, confirm_sensitive=True)

    assert result.row_count == 1
    records = pd.read_parquet(tmp_path / "out" / "raw" / "records_subset.parquet")
    assert records["patient_id"].tolist() == ["P001"]


def test_invalid_json_config_returns_input_error(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text("{bad", encoding="utf-8")
    with pytest.raises(InputError, match="invalid JSON"):
        load_condition(cfg)


def test_dry_run_writes_manifest_but_no_records(tmp_path):
    raw = _make_raw_dir(tmp_path)
    cfg = _base_config()
    result = extract_subset(cfg, raw, tmp_path / "out", tmp_path, dry_run=True)
    assert result.row_count == 3
    assert (tmp_path / "out" / "manifest.json").exists()
    assert not (tmp_path / "out" / "raw" / "records_subset.parquet").exists()


def test_analysis_glob_copy_preserves_relative_path_and_allowlist(tmp_path):
    raw = _make_raw_dir(tmp_path)
    project = tmp_path / "project"
    report = project / "data" / "reports" / "summary.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"ok": true}', encoding="utf-8")
    (report.parent / "secret.exe").write_text("no", encoding="utf-8")
    cfg = _base_config()
    cfg["outputs"] = {"include_analysis_globs": ["data/reports/*"]}

    extract_subset(cfg, raw, tmp_path / "out", project, confirm_sensitive=True)

    assert (tmp_path / "out" / "analysis" / "data" / "reports" / "summary.json").exists()
    assert not (tmp_path / "out" / "analysis" / "data" / "reports" / "secret.exe").exists()


def test_analysis_glob_rejects_traversal(tmp_path):
    raw = _make_raw_dir(tmp_path)
    cfg = _base_config()
    cfg["outputs"] = {"include_analysis_globs": ["../*.json"]}
    with pytest.raises(InputError):
        extract_subset(cfg, raw, tmp_path / "out", tmp_path, confirm_sensitive=True)


def test_cli_runs_with_yaml_config(tmp_path):
    raw = _make_raw_dir(tmp_path)
    config = tmp_path / "condition.yaml"
    config.write_text(
        """
version: 1
project: MODE_11_hana
purpose: analysis
request_id: cli-test
date_range:
  start: '2024-09-01'
  end: '2024-09-01'
filters:
  institution_ids: ['I1']
outputs: {}
""".strip(),
        encoding="utf-8",
    )
    script = Path(__file__).parents[2] / "scripts" / "ops" / "extract_raw_subset.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--raw-dir",
            str(raw),
            "--out-dir",
            str(tmp_path / "out"),
            "--project-root",
            str(tmp_path),
            "--confirm-sensitive",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["row_count"] == 2


def test_cli_runs_with_inline_condition_json_dry_run(tmp_path):
    raw = _make_raw_dir(tmp_path)
    script = Path(__file__).parents[2] / "scripts" / "ops" / "extract_raw_subset.py"
    condition = json.dumps(_base_config())
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--condition-json",
            condition,
            "--raw-dir",
            str(raw),
            "--out-dir",
            str(tmp_path / "out"),
            "--project-root",
            str(tmp_path),
            "--dry-run",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["records"] is None
    assert payload["row_count"] == 3
