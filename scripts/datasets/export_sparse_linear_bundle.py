"""Export a trained sparse-linear model into a servable DL bundle.

The same-window sparse-linear baseline trains an ``nn.Linear(input_dim, 1)``
binary head (score = ``sigmoid(logit)``; see
``scripts/ops/sparse_training_smoke.train_sparse_linear_temporal_smoke``).
The serving DL path (``serving/dl_predictor.DLModel``) instead expects a
multi-class TorchScript model and applies ``softmax`` over the output dim,
mapping to ``model_config["output_labels"]``.

This exporter bridges the two **without changing either side** by reconstructing
the single-logit head as a 2-output linear layer:

    logit_low  = 0
    logit_high = w · x + b              (w, b = trained weight/bias)

    softmax([0, z])[1] = e^z / (1 + e^z) = sigmoid(z)

so the served ``probabilities["high"]`` reproduces the training score exactly,
and ``probabilities["low"] = 1 - sigmoid(z)``.

Scope: this generalizes ``scripts/datasets/smoke_dl_bundle`` (hardcoded random
weights + toy vocab) into a real-weight exporter. It produces an *encoder-input*
servable bundle; exposing the proxy-label score as patient "risk" in serving is a
separate label-definition decision gated on cross-family sign-off and is **not**
performed here.

The drug vocabulary must be the exact one the training matrix ``X`` was built
with (drug-only multi-hot, ``input_dim == len(drug_vocab)``, ``"_unk"`` present),
otherwise serving would map drug codes to the wrong columns.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from scripts.datasets.contracts import (
    LOOKBACK_DAYS_DEFAULT,
    write_dl_bundle_manifest,
)

DEFAULT_SCHEMA_VERSION = "dl.v1"
DEFAULT_OUTPUT_LABELS = ("low", "high")


def _to_weight_bias(weight, bias):
    """Coerce trained Linear(in, 1) params to (1-D weight list, float bias).

    Accepts torch tensors, numpy arrays, or plain sequences. ``weight`` may be
    shape ``(input_dim,)`` or ``(1, input_dim)``; ``bias`` a scalar or length-1.
    """
    import numpy as np

    w = np.asarray(weight, dtype=np.float64)
    if w.ndim == 2:
        if w.shape[0] != 1:
            raise ValueError(
                f"weight must be Linear(in, 1) — got shape {w.shape}"
            )
        w = w.reshape(-1)
    elif w.ndim != 1:
        raise ValueError(f"weight must be 1-D or (1, in) — got shape {w.shape}")

    b = np.asarray(bias, dtype=np.float64).reshape(-1)
    if b.size != 1:
        raise ValueError(f"bias must be a scalar / length-1 — got size {b.size}")
    return w, float(b[0])


def _validate_drug_vocab(drug_vocab: dict, input_dim: int) -> None:
    if not isinstance(drug_vocab, dict) or not drug_vocab:
        raise ValueError("drug_vocab must be a non-empty {code: index} mapping")
    if "_unk" not in drug_vocab:
        raise ValueError(
            "drug_vocab must contain '_unk' (operational vocab convention); "
            "serving maps OOV drugs to this dimension"
        )
    indices = sorted(int(i) for i in drug_vocab.values())
    if indices != list(range(len(drug_vocab))):
        raise ValueError(
            "drug_vocab indices must be contiguous 0..N-1 with no gaps/dupes"
        )
    if len(drug_vocab) != input_dim:
        raise ValueError(
            "drug_vocab size must equal model input_dim "
            f"(len(drug_vocab)={len(drug_vocab)}, input_dim={input_dim}). "
            "The bundle vocab must be the exact vocab X was built with."
        )


def _build_two_output_linear(weight, bias, input_dim: int):
    """Reconstruct a 2-output linear so softmax(out)[1] == sigmoid(w·x + b)."""
    import torch

    model = torch.nn.Linear(input_dim, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
        # row 1 ("high") carries the trained head; row 0 ("low") stays at 0.
        model.weight[1] = torch.tensor(weight, dtype=model.weight.dtype)
        model.bias[1] = float(bias)
    model.eval()
    return model


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def export_sparse_linear_bundle(
    bundle_dir: str | Path,
    *,
    weight,
    bias,
    drug_vocab: dict,
    run_id: str,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
    output_labels=DEFAULT_OUTPUT_LABELS,
    device: str = "auto",
    feature_normalizer: object | None = None,
) -> Path:
    """Write a servable DL bundle from a trained sparse-linear head.

    Parameters
    ----------
    weight, bias
        Trained ``nn.Linear(input_dim, 1)`` parameters (any array-like).
    drug_vocab
        Drug-only multi-hot vocab used to build the training matrix. Must be a
        contiguous ``{code: 0..N-1}`` mapping containing ``"_unk"`` with
        ``len == input_dim``.
    output_labels
        Two labels; index 1 is the positive ("high") class whose served
        probability equals the training ``sigmoid`` score. Defaults to
        ``("low", "high")``.

    Returns the written ``MANIFEST.json`` path.
    """
    import torch

    labels = [str(x) for x in output_labels]
    if len(labels) != 2:
        raise ValueError(
            f"output_labels must have exactly 2 entries — got {labels}"
        )

    w, b = _to_weight_bias(weight, bias)
    input_dim = int(w.shape[0])
    if input_dim <= 0:
        raise ValueError("weight is empty (input_dim must be positive)")
    _validate_drug_vocab(drug_vocab, input_dim)

    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)

    model = _build_two_output_linear(w, b, input_dim)
    example = torch.zeros((1, input_dim), dtype=torch.float32)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(root / "model.pt"))

    _write_json(
        root / "model_config.json",
        {
            "architecture": "linear",
            "bundle_kind": "sparse_linear",
            "device": device,
            "encoding_strategy": "multi_hot",
            "input_dim": input_dim,
            "output_labels": labels,
        },
    )
    _write_json(root / "drug_vocab.json", {str(k): int(v) for k, v in drug_vocab.items()})

    # The linear head ignores graph structure; serving loads edge_index.pt
    # unconditionally, so write a minimal placeholder tensor (non-empty file).
    torch.save(torch.zeros((2, 1), dtype=torch.long), str(root / "edge_index.pt"))

    normalizer = feature_normalizer if feature_normalizer is not None else {"type": "identity"}
    (root / "feature_normalizer.pkl").write_bytes(pickle.dumps(normalizer))

    _write_json(
        root / "schema_version.json",
        {"schema_version": schema_version, "bundle_kind": "sparse_linear"},
    )

    return write_dl_bundle_manifest(
        root,
        run_id=run_id,
        schema_version=schema_version,
        lookback_days=lookback_days,
    )


def export_from_torch_linear(
    bundle_dir: str | Path,
    model,
    drug_vocab: dict,
    *,
    run_id: str,
    **kwargs,
) -> Path:
    """Convenience wrapper: export from an in-memory ``nn.Linear(input_dim, 1)``."""
    if getattr(model, "out_features", 1) != 1:
        raise ValueError(
            "export_from_torch_linear expects Linear(input_dim, 1); use "
            "export_sparse_linear_bundle for general weight/bias input"
        )
    weight = model.weight.detach().cpu().numpy()
    bias = model.bias.detach().cpu().numpy()
    return export_sparse_linear_bundle(
        bundle_dir,
        weight=weight,
        bias=bias,
        drug_vocab=drug_vocab,
        run_id=run_id,
        **kwargs,
    )
