"""Semantic validator for operational DL bundles."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from scripts.datasets.contracts import validate_dl_bundle_manifest

SUPPORTED_ARCHITECTURES = {"linear", "gat", "gcn"}


@dataclass(frozen=True)
class VerificationReport:
    bundle_dir: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_json(path: Path, errors: list[str]) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        errors.append(f"{path.name} is not valid JSON: {e}")
        return {}
    if not isinstance(raw, dict):
        errors.append(f"{path.name} must be a JSON object")
        return {}
    return raw


def _validate_model_config(config: dict, errors: list[str], warnings: list[str]) -> int:
    architecture = config.get("architecture")
    if architecture is not None:
        arch = str(architecture).lower()
        if arch not in SUPPORTED_ARCHITECTURES:
            warnings.append(
                "model_config architecture is not recognized: "
                f"{architecture!r} (known={sorted(SUPPORTED_ARCHITECTURES)})"
            )

    try:
        input_dim = int(config.get("input_dim", 0))
    except (TypeError, ValueError):
        errors.append("model_config input_dim must be an integer")
        input_dim = 0
    if input_dim <= 0:
        errors.append("model_config input_dim must be positive")

    labels = config.get("output_labels")
    if not isinstance(labels, list) or not labels or not all(labels):
        errors.append("model_config output_labels must be a non-empty list")
        return input_dim
    normalized = [str(label) for label in labels]
    if len(set(normalized)) != len(normalized):
        errors.append("model_config output_labels contains duplicate labels")
    return input_dim


def _validate_drug_vocab(vocab: dict, input_dim: int, errors: list[str]) -> None:
    if not isinstance(vocab, dict) or not vocab:
        errors.append("drug_vocab must be a non-empty object")
        return
    for code, index in vocab.items():
        try:
            idx = int(index)
        except (TypeError, ValueError):
            errors.append(f"drug_vocab index for {code!r} must be an integer")
            continue
        if idx < 0 or idx >= input_dim:
            errors.append(
                f"drug_vocab index out of range for {code!r}: "
                f"{idx} (input_dim={input_dim})"
            )


def _validate_schema_sidecar(
    manifest: dict,
    sidecar: dict,
    warnings: list[str],
) -> None:
    manifest_schema = manifest.get("schema_version")
    sidecar_schema = sidecar.get("schema_version")
    if not sidecar_schema:
        warnings.append("schema_version.json missing schema_version")
        return
    if manifest_schema and sidecar_schema != manifest_schema:
        warnings.append(
            "schema_version.json schema_version does not match MANIFEST.json: "
            f"{sidecar_schema!r} != {manifest_schema!r}"
        )


def _check_torchscript_load(bundle_dir: Path, errors: list[str]) -> None:
    try:
        import torch
    except Exception as e:
        errors.append(f"--check-model requested but torch import failed: {e}")
        return
    try:
        torch.jit.load(str(bundle_dir / "model.pt"), map_location="cpu")
    except Exception as e:
        errors.append(f"torch.jit.load failed for model.pt: {e}")


def validate_bundle(
    bundle_dir: str | Path,
    *,
    check_model: bool = False,
) -> VerificationReport:
    """Validate manifest integrity and semantic DL bundle contracts."""
    root = Path(bundle_dir)
    errors: list[str] = []
    warnings: list[str] = []
    manifest: dict = {}
    try:
        manifest = validate_dl_bundle_manifest(root)
    except Exception as e:
        errors.append(f"manifest validation failed: {e}")

    model_config = _load_json(root / "model_config.json", errors)
    drug_vocab = _load_json(root / "drug_vocab.json", errors)
    schema_sidecar = _load_json(root / "schema_version.json", errors)

    input_dim = _validate_model_config(model_config, errors, warnings)
    _validate_drug_vocab(drug_vocab, input_dim, errors)
    if manifest:
        _validate_schema_sidecar(manifest, schema_sidecar, warnings)
    if check_model:
        _check_torchscript_load(root, errors)

    return VerificationReport(
        bundle_dir=root,
        errors=errors,
        warnings=warnings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate semantic contracts for an operational DL bundle.",
    )
    parser.add_argument("bundle_dir")
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="Also attempt torch.jit.load(model.pt) on CPU.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_bundle(args.bundle_dir, check_model=args.check_model)
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print(f"bundle_dir={report.bundle_dir}")
    print(f"status={'ok' if report.ok else 'failed'}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
