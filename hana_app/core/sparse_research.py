"""Helpers for prebuilt sparse research datasets.

The desktop app's regular Page 3 training path consumes feature DataFrames.
These helpers keep ops-scale sparse artifacts isolated under project-level
``data/datasets`` so the two contracts do not get mixed accidentally.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"
REQUIRED_DATASET_FILES = ("metadata.json", "X_csr.npz", "y.npy")


@dataclass(frozen=True)
class SparseReportPaths:
    json: Path | None
    markdown: Path | None


@dataclass(frozen=True)
class SparseDatasetSummary:
    name: str
    dataset_dir: Path
    label_source: str
    evaluation_context: str
    n_patients: int
    label_positive_rate_pct: float
    input_dim: int
    feature_window_label: str
    outcome_window_label: str
    status: str
    metadata: dict


def list_sparse_datasets(dataset_root: str | Path = DATASETS_ROOT) -> list[SparseDatasetSummary]:
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    summaries: list[SparseDatasetSummary] = []
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not _is_complete_dataset(dataset_dir):
            continue
        summary = _summary_from_dataset(dataset_dir)
        if summary is not None:
            summaries.append(summary)
    return summaries


def dataset_display_rows(summaries: Sequence[SparseDatasetSummary]) -> list[dict]:
    return [
        {
            "dataset": summary.name,
            "label_source": summary.label_source,
            "evaluation_context": summary.evaluation_context,
            "n_patients": summary.n_patients,
            "positive_rate_pct": summary.label_positive_rate_pct,
            "feature_window": summary.feature_window_label,
            "outcome_window": summary.outcome_window_label,
            "input_dim": summary.input_dim,
            "status": summary.status,
        }
        for summary in summaries
    ]


def find_report_paths(dataset_dir: str | Path) -> SparseReportPaths:
    dataset_path = Path(dataset_dir)
    candidates = [
        dataset_path,
        default_smoke_output_dir(dataset_path, model="linear"),
        dataset_path.parent / f"{dataset_path.name}_linear_smoke",
        dataset_path.parent / f"{dataset_path.name}_temporal",
    ]
    for candidate in candidates:
        json_path = candidate / "sparse_training_smoke_report.json"
        md_path = candidate / "sparse_training_smoke_report.md"
        if json_path.exists() or md_path.exists():
            return SparseReportPaths(
                json=json_path if json_path.exists() else None,
                markdown=md_path if md_path.exists() else None,
            )
    return SparseReportPaths(json=None, markdown=None)


def default_smoke_output_dir(dataset_dir: str | Path, *, model: str = "linear") -> Path:
    dataset_path = Path(dataset_dir)
    suffix = f"{model}_smoke"
    return dataset_path.parent / f"{dataset_path.name}_{suffix}"


def build_smoke_command(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    python_executable: str = sys.executable,
    epochs: int = 20,
    batch_size: int = 2048,
    seed: int = 42,
    device: str = "cpu",
) -> list[str]:
    return [
        python_executable,
        "-m",
        "scripts.ops.sparse_training_smoke",
        "--dataset-dir",
        str(Path(dataset_dir)),
        "--output-dir",
        str(Path(output_dir)),
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--seed",
        str(int(seed)),
        "--device",
        device,
    ]


def lock_path_for(output_dir: str | Path) -> Path:
    return Path(output_dir) / "sparse_training_smoke.lock"


def log_path_for(output_dir: str | Path) -> Path:
    return Path(output_dir) / "sparse_training_smoke.log"


def read_log_tail(path: str | Path, *, max_lines: int = 30, max_bytes: int = 64 * 1024) -> str:
    log_path = Path(path)
    if not log_path.exists():
        return ""
    with log_path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        data = handle.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max(0, int(max_lines)) :])


def _is_complete_dataset(dataset_dir: Path) -> bool:
    return all((dataset_dir / file_name).exists() for file_name in REQUIRED_DATASET_FILES)


def _summary_from_dataset(dataset_dir: Path) -> SparseDatasetSummary | None:
    try:
        metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    n_rows = int(metadata.get("n_patients") or 0)
    input_dim = int(metadata.get("input_dim") or 0)
    status = "ok" if n_rows > 0 and input_dim > 0 else "metadata_incomplete"

    return SparseDatasetSummary(
        name=dataset_dir.name,
        dataset_dir=dataset_dir,
        label_source=str(metadata.get("label_source") or ""),
        evaluation_context=_evaluation_context(metadata),
        n_patients=n_rows,
        label_positive_rate_pct=float(metadata.get("label_positive_rate_pct") or 0.0),
        input_dim=input_dim,
        feature_window_label=_feature_window_label(metadata),
        outcome_window_label=_window_label(metadata.get("outcome_window")),
        status=status,
        metadata=metadata,
    )


def _feature_window_label(metadata: dict) -> str:
    feature_window = _window_label(metadata.get("feature_window"))
    if feature_window:
        return feature_window
    if metadata.get("reference_date") and metadata.get("lookback_days") is not None:
        return f"ref={metadata['reference_date']}, lookback={metadata['lookback_days']}"
    return ""


def _evaluation_context(metadata: dict) -> str:
    label_source = str(metadata.get("label_source") or "")
    if label_source == "future_multi_institution_onset" or metadata.get("outcome_window"):
        return "Future-Onset"
    if label_source == "multi_institution":
        return "Same-Window"
    return "Unknown"


def _window_label(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    start = value.get("start")
    end = value.get("end")
    if start and end:
        return f"{start}..{end}"
    return ""
