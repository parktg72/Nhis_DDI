"""Data and artifact contracts for separated ML/DL training tracks.

The project has one shared raw extraction path. After extraction, tabular ML and
operational DL must produce different datasets and artifacts:

- ML uses patient-level tabular features.
- DL uses raw prescription history plus graph/sequence artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ML_TRACK = "ml"
DL_TRACK = "dl"

ML_DATASET_REQUIRED_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "drug_count",
    "drug_count_7d",
    "institution_count",
    "ddi_contraindicated",
    "ddi_major",
    "ddi_moderate",
    "ddi_minor",
    "risk_level",
)

DL_DATASET_REQUIRED_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "drug_code",
    "prescription_date",
)

DL_BUNDLE_REQUIRED_FILES: tuple[str, ...] = (
    "model.pt",
    "model_config.json",
    "drug_vocab.json",
    "edge_index.pt",
    "feature_normalizer.pkl",
    "schema_version.json",
)

DL_MANIFEST_FILE = "MANIFEST.json"
HASH_ALG_SHA256 = "sha256"

LOOKBACK_DAYS_DEFAULT = 365
LOOKBACK_DAYS_MIN = 7
LOOKBACK_DAYS_MAX = 1825


class LookbackMismatchError(ValueError):
    """Runtime lookback_days does not match the model artifact contract."""


class BundleHashMismatchError(ValueError):
    """An artifact file hash does not match MANIFEST.json."""


class BundleArtifactEmptyError(ValueError):
    """A required DL bundle artifact exists but is empty."""


@dataclass(frozen=True)
class DatasetContract:
    """Named dataset contract for pipeline and UI validation."""

    track: str
    required_columns: tuple[str, ...]
    description: str


ML_DATASET_CONTRACT = DatasetContract(
    track=ML_TRACK,
    required_columns=ML_DATASET_REQUIRED_COLUMNS,
    description="patient-level tabular features for XGBoost/LightGBM/classic ML",
)

DL_DATASET_CONTRACT = DatasetContract(
    track=DL_TRACK,
    required_columns=DL_DATASET_REQUIRED_COLUMNS,
    description="raw prescription sequence/graph events for operational DL",
)


def validate_required_columns(columns: Iterable[str], required: Iterable[str]) -> None:
    """Raise ValueError when a dataset is missing required columns."""
    col_set = set(columns)
    missing = [col for col in required if col not in col_set]
    if missing:
        raise ValueError(f"required columns missing: {missing}")


def validate_lookback_days(lookback_days: int) -> int:
    """Validate and return lookback_days for DL training/serving contracts."""
    value = int(lookback_days)
    if not LOOKBACK_DAYS_MIN <= value <= LOOKBACK_DAYS_MAX:
        raise ValueError(
            "lookback_days out of range: "
            f"{value} (allowed {LOOKBACK_DAYS_MIN}..{LOOKBACK_DAYS_MAX})"
        )
    return value


def validate_lookback_consistency(
    artifact_lookback_days: int,
    runtime_lookback_days: int,
    *,
    context: str = "",
) -> None:
    """STRICT policy: runtime lookback must match the trained artifact."""
    artifact_value = validate_lookback_days(artifact_lookback_days)
    runtime_value = validate_lookback_days(runtime_lookback_days)
    if artifact_value != runtime_value:
        ctx = f" ({context})" if context else ""
        raise LookbackMismatchError(
            "lookback mismatch"
            f"{ctx}: artifact={artifact_value}, runtime={runtime_value}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dl_bundle_manifest(
    bundle_dir: str | Path,
    run_id: str,
    schema_version: str,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
) -> dict:
    """Build a manifest dict for an operational DL artifact bundle.

    The manifest is intentionally plain JSON so serving can verify it before
    loading torch artifacts. The caller decides when to write it to disk.
    """
    root = Path(bundle_dir)
    files: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    lookback_value = validate_lookback_days(lookback_days)
    for name in DL_BUNDLE_REQUIRED_FILES:
        path = root / name
        if not path.exists():
            missing.append(name)
            continue
        size = path.stat().st_size
        if size <= 0:
            raise BundleArtifactEmptyError(f"DL bundle artifact is empty: {name}")
        files[name] = {
            "sha256": _sha256_file(path),
            "bytes": size,
        }
    if missing:
        raise FileNotFoundError(f"DL bundle missing required files: {missing}")

    return {
        "track": DL_TRACK,
        "run_id": run_id,
        "schema_version": schema_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash_alg": HASH_ALG_SHA256,
        "lookback_days": lookback_value,
        "drug_vocab_sha256": files["drug_vocab.json"]["sha256"],
        "edge_index_sha256": files["edge_index.pt"]["sha256"],
        "files": files,
    }


def write_dl_bundle_manifest(
    bundle_dir: str | Path,
    run_id: str,
    schema_version: str,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
) -> Path:
    """Create MANIFEST.json for a DL bundle and return its path."""
    root = Path(bundle_dir)
    manifest = build_dl_bundle_manifest(
        root,
        run_id=run_id,
        schema_version=schema_version,
        lookback_days=lookback_days,
    )
    out = root / DL_MANIFEST_FILE
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, out)
    validate_dl_bundle_manifest(root)
    return out


def validate_dl_bundle_manifest(bundle_dir: str | Path) -> dict:
    """Validate MANIFEST.json and required artifact hashes for a DL bundle."""
    root = Path(bundle_dir)
    manifest_path = root / DL_MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"DL bundle manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("track") != DL_TRACK:
        raise ValueError(f"manifest track must be 'dl', got {manifest.get('track')!r}")
    if not manifest.get("run_id"):
        raise ValueError("manifest missing run_id")
    if not manifest.get("schema_version"):
        raise ValueError("manifest missing schema_version")
    if manifest.get("hash_alg") != HASH_ALG_SHA256:
        raise ValueError(
            f"manifest hash_alg must be {HASH_ALG_SHA256!r}, "
            f"got {manifest.get('hash_alg')!r}"
        )
    if "created_at" not in manifest:
        raise ValueError("manifest missing created_at")
    validate_lookback_days(manifest.get("lookback_days"))

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("manifest files must be an object")

    missing = [name for name in DL_BUNDLE_REQUIRED_FILES if name not in files]
    if missing:
        raise ValueError(f"manifest missing required file entries: {missing}")

    for name in DL_BUNDLE_REQUIRED_FILES:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"DL bundle file missing: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise BundleArtifactEmptyError(f"DL bundle artifact is empty: {name}")
        expected = files[name].get("sha256")
        actual = _sha256_file(path)
        if actual != expected:
            raise BundleHashMismatchError(
                f"DL bundle hash mismatch for {name}: "
                f"expected={str(expected)[:16]}, actual={actual[:16]}"
            )
        expected_bytes = files[name].get("bytes")
        if expected_bytes != size:
            raise BundleHashMismatchError(
                f"DL bundle byte size mismatch for {name}: "
                f"expected={expected_bytes}, actual={size}"
            )

    if manifest.get("drug_vocab_sha256") != files["drug_vocab.json"]["sha256"]:
        raise BundleHashMismatchError("drug_vocab_sha256 top-level hash mismatch")
    if manifest.get("edge_index_sha256") != files["edge_index.pt"]["sha256"]:
        raise BundleHashMismatchError("edge_index_sha256 top-level hash mismatch")

    return manifest
