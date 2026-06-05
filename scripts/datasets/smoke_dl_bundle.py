"""Create a minimal TorchScript DL bundle for deployment smoke tests."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Sequence

from scripts.datasets.contracts import (
    LOOKBACK_DAYS_DEFAULT,
    write_dl_bundle_manifest,
)


DEFAULT_SCHEMA_VERSION = "dl.v1"
DEFAULT_RUN_ID = "smoke-dl"
# 운영 vocab(scripts/ops/build_drug_vocab)은 "_unk" 을 index 0 에 둔다. smoke 번들도
# 동일 관례를 따라 OOV→_unk 매핑 경로(serving._encode_history)를 실제로 행사한다.
DEFAULT_DRUG_VOCAB = {
    "_unk": 0,
    "D1": 1,
    "D2": 2,
    "D3": 3,
}
DEFAULT_OUTPUT_LABELS = ["low", "high"]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_torchscript_model(path: Path, *, input_dim: int) -> None:
    import torch

    model = torch.nn.Linear(input_dim, len(DEFAULT_OUTPUT_LABELS))
    with torch.no_grad():
        model.weight[0].fill_(-1.0)
        model.weight[1].fill_(1.0)
        model.bias.zero_()
    model.eval()

    example = torch.zeros((1, input_dim), dtype=torch.float32)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(path))


def _write_edge_index(path: Path) -> None:
    import torch

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    torch.save(edge_index, str(path))


def create_smoke_dl_bundle(
    bundle_dir: str | Path,
    *,
    run_id: str = DEFAULT_RUN_ID,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
) -> Path:
    """Create a reloadable DL smoke bundle and return MANIFEST.json path."""
    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)

    input_dim = len(DEFAULT_DRUG_VOCAB)
    _write_torchscript_model(root / "model.pt", input_dim=input_dim)
    _write_json(
        root / "model_config.json",
        {
            "architecture": "linear",
            "bundle_kind": "smoke",
            "device": "auto",
            "encoding_strategy": "multi_hot",
            "input_dim": input_dim,
            "output_labels": DEFAULT_OUTPUT_LABELS,
        },
    )
    _write_json(root / "drug_vocab.json", DEFAULT_DRUG_VOCAB)
    _write_edge_index(root / "edge_index.pt")
    (root / "feature_normalizer.pkl").write_bytes(
        pickle.dumps({"type": "identity"})
    )
    _write_json(
        root / "schema_version.json",
        {
            "schema_version": schema_version,
            "bundle_kind": "smoke",
        },
    )

    return write_dl_bundle_manifest(
        root,
        run_id=run_id,
        schema_version=schema_version,
        lookback_days=lookback_days,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a minimal TorchScript DL smoke bundle.",
    )
    parser.add_argument("bundle_dir", help="Output bundle directory.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--schema-version", default=DEFAULT_SCHEMA_VERSION)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=LOOKBACK_DAYS_DEFAULT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = create_smoke_dl_bundle(
        args.bundle_dir,
        run_id=args.run_id,
        schema_version=args.schema_version,
        lookback_days=args.lookback_days,
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
