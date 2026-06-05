from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse


def _write_sparse_dataset(
    path: Path,
    *,
    metadata: dict | None = None,
    n_rows: int = 3,
    n_cols: int = 5,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path / "X_csr.npz", sparse.eye(n_rows, n_cols, dtype=np.float32, format="csr"))
    np.save(path / "y.npy", np.array([1] + [0] * (n_rows - 1), dtype=np.int8))
    (path / "metadata.json").write_text(
        json.dumps(metadata or {}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_lists_only_complete_sparse_datasets(tmp_path: Path) -> None:
    from hana_app.core.sparse_research import list_sparse_datasets

    _write_sparse_dataset(
        tmp_path / "future_t6",
        metadata={
            "label_source": "future_multi_institution_onset",
            "n_patients": 3,
            "input_dim": 5,
            "label_positive_rate_pct": 33.3333,
            "feature_window": {"start": "2024-11-01", "end": "2024-11-30"},
            "outcome_window": {"start": "2024-12-01", "end": "2024-12-31"},
        },
    )
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "metadata.json").write_text("{}", encoding="utf-8")

    summaries = list_sparse_datasets(tmp_path)

    assert [summary.name for summary in summaries] == ["future_t6"]
    assert summaries[0].dataset_dir == tmp_path / "future_t6"
    assert summaries[0].label_source == "future_multi_institution_onset"
    assert summaries[0].evaluation_context == "Future-Onset"
    assert summaries[0].n_patients == 3
    assert summaries[0].label_positive_rate_pct == 33.3333
    assert summaries[0].feature_window_label == "2024-11-01..2024-11-30"
    assert summaries[0].outcome_window_label == "2024-12-01..2024-12-31"


def test_dataset_rows_are_stable_for_streamlit_table(tmp_path: Path) -> None:
    from hana_app.core.sparse_research import dataset_display_rows, list_sparse_datasets

    _write_sparse_dataset(
        tmp_path / "multi_inst",
        metadata={
            "label_source": "multi_institution",
            "evaluation_context": "Same-Window",
            "n_patients": 10,
            "input_dim": 4,
            "label_positive_rate_pct": 22.5,
            "reference_date": "2024-12-31",
            "lookback_days": 29,
        },
        n_rows=10,
        n_cols=4,
    )

    rows = dataset_display_rows(list_sparse_datasets(tmp_path))

    assert rows == [
        {
            "dataset": "multi_inst",
            "label_source": "multi_institution",
            "evaluation_context": "Same-Window",
            "n_patients": 10,
            "positive_rate_pct": 22.5,
            "feature_window": "ref=2024-12-31, lookback=29",
            "outcome_window": "",
            "input_dim": 4,
            "status": "ok",
        }
    ]


def test_detects_reports_and_builds_smoke_command(tmp_path: Path) -> None:
    from hana_app.core.sparse_research import (
        build_smoke_command,
        default_smoke_output_dir,
        find_report_paths,
    )

    dataset_dir = tmp_path / "future_t6"
    _write_sparse_dataset(dataset_dir)
    output_dir = default_smoke_output_dir(dataset_dir, model="linear")
    output_dir.mkdir()
    (output_dir / "sparse_training_smoke_report.md").write_text("# Report", encoding="utf-8")
    (output_dir / "sparse_training_smoke_report.json").write_text("{}", encoding="utf-8")

    reports = find_report_paths(dataset_dir)
    command = build_smoke_command(
        dataset_dir,
        output_dir,
        python_executable=sys.executable,
        epochs=3,
        batch_size=128,
        seed=7,
        device="cpu",
    )

    assert reports.markdown == output_dir / "sparse_training_smoke_report.md"
    assert reports.json == output_dir / "sparse_training_smoke_report.json"
    assert command[:3] == [sys.executable, "-m", "scripts.ops.sparse_training_smoke"]
    assert "--dataset-dir" in command
    assert str(dataset_dir) in command
    assert "--output-dir" in command
    assert str(output_dir) in command
    assert command[-6:] == ["--batch-size", "128", "--seed", "7", "--device", "cpu"]


def test_log_tail_and_lock_paths_are_dataset_scoped(tmp_path: Path) -> None:
    from hana_app.core.sparse_research import lock_path_for, log_path_for, read_log_tail

    output_dir = tmp_path / "run"
    log_path = log_path_for(output_dir)
    lock_path = lock_path_for(output_dir)
    output_dir.mkdir()
    log_path.write_text("\n".join(f"line {i}" for i in range(5)), encoding="utf-8")

    assert log_path == output_dir / "sparse_training_smoke.log"
    assert lock_path == output_dir / "sparse_training_smoke.lock"
    assert read_log_tail(log_path, max_lines=2) == "line 3\nline 4"
    assert read_log_tail(output_dir / "missing.log") == ""


def test_dataset_listing_uses_metadata_without_loading_sparse_matrix(tmp_path: Path) -> None:
    from hana_app.core import sparse_research

    dataset_dir = tmp_path / "large_dataset"
    _write_sparse_dataset(
        dataset_dir,
        metadata={
            "label_source": "future_multi_institution_onset",
            "n_patients": 123,
            "input_dim": 14705,
            "label_positive_rate_pct": 11.9,
        },
    )
    (dataset_dir / "X_csr.npz").write_bytes(b"not a valid sparse matrix")
    (dataset_dir / "y.npy").write_bytes(b"not a valid numpy array")

    summaries = sparse_research.list_sparse_datasets(tmp_path)

    assert len(summaries) == 1
    assert summaries[0].n_patients == 123
    assert summaries[0].input_dim == 14705
    assert summaries[0].status == "ok"
