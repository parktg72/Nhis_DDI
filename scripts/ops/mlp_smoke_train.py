"""Train a small MLP smoke model on multi-hot drug features."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops.label_audit import label_patient_histories as label_sick_code_histories
from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider
from scripts.ops.multi_institution_label import (
    MULTI_INSTITUTION_THRESHOLD,
    label_patient_histories as label_multi_institution_histories,
)
from scripts.ops.multihot_encoder import encode_batch
from scripts.ops.therapeutic_duplication_label import (
    THERAPEUTIC_DUP_THRESHOLD,
    label_therapeutic_duplication,
)


@dataclass(frozen=True)
class SmokeTrainResult:
    train_loss_final: float
    val_auc: float
    val_precision: float
    val_recall: float
    n_train: int
    n_val: int
    n_positive_train: int
    n_positive_val: int
    elapsed_sec: float


def _torch():
    import torch

    return torch


def _nn():
    import torch.nn as nn

    return nn


class MultiHotMLP(_nn().Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        nn = _nn()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_dataset(
    provider,
    vocab: dict[str, int],
    patient_ids: list[str],
    labels: dict[str, int],
    reference_date: date,
    lookback_days: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    X, _ = encode_batch(
        provider,
        patient_ids,
        vocab,
        reference_date=reference_date,
        lookback_days=lookback_days,
    )
    y = np.array([int(labels[str(patient_id)]) for patient_id in patient_ids], dtype=np.int64)
    return X, y


def build_training_labels(
    patient_ids: Sequence[str],
    histories,
    *,
    label_source: str = "sick_code",
    multi_institution_threshold: int = MULTI_INSTITUTION_THRESHOLD,
    therapeutic_dup_threshold: int = THERAPEUTIC_DUP_THRESHOLD,
) -> tuple[dict[str, int], dict]:
    if label_source == "sick_code":
        result = label_sick_code_histories(patient_ids, histories)
        return result.labels, {
            "label_source": "sick_code",
            "label_semantics": "sick_code ADR co-occurrence proxy, not retrospective ADR causality",
        }
    if label_source == "multi_institution":
        result = label_multi_institution_histories(
            patient_ids,
            histories,
            threshold=multi_institution_threshold,
        )
        return result.labels, {
            "label_source": "multi_institution",
            "multi_institution_threshold": multi_institution_threshold,
            "label_semantics": "multi-institution proxy: distinct institution_id count >= threshold within lookback window",
        }
    if label_source == "therapeutic_duplication":
        result = label_therapeutic_duplication(
            patient_ids,
            histories,
            min_duplicate_classes=therapeutic_dup_threshold,
        )
        return result.labels, {
            "label_source": "therapeutic_duplication",
            "therapeutic_dup_threshold": therapeutic_dup_threshold,
            "label_semantics": "therapeutic duplication proxy: same efmdc_clsf_no with distinct drug_code count >= 2; label positive when duplicate class count >= threshold",
        }
    raise ValueError(f"unsupported label_source: {label_source}")


def compute_pos_weight(y: np.ndarray) -> float:
    positives = int(np.asarray(y).sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return 1.0
    return round(negatives / positives, 6)


def stratified_train_val_indices(
    y: np.ndarray,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if len(positive) < 2 or len(negative) < 2:
        indices = np.arange(len(y))
        rng.shuffle(indices)
        n_val = max(1, int(round(len(y) * val_fraction)))
        return np.sort(indices[n_val:]), np.sort(indices[:n_val])

    rng.shuffle(positive)
    rng.shuffle(negative)
    n_val_pos = max(1, int(round(len(positive) * val_fraction)))
    n_val_neg = max(1, int(round(len(negative) * val_fraction)))
    val_idx = np.concatenate([positive[:n_val_pos], negative[:n_val_neg]])
    train_idx = np.concatenate([positive[n_val_pos:], negative[n_val_neg:]])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def train_mlp_smoke(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden_dims: tuple[int, ...] = (256, 64),
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
    return_model: bool = False,
):
    torch = _torch()
    nn = _nn()
    start = perf_counter()
    _seed_everything(seed)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    train_idx, val_idx = stratified_train_val_indices(
        y.astype(np.int64),
        val_fraction=val_fraction,
        seed=seed,
    )

    model = MultiHotMLP(X.shape[1], hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([compute_pos_weight(y[train_idx])], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_train = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[train_idx].reshape(-1, 1), dtype=torch.float32, device=device)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32, device=device)

    final_loss = 0.0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(train_idx), generator=generator)
        model.train()
        for start_idx in range(0, len(order), batch_size):
            batch_idx = order[start_idx : start_idx + batch_size].to(device)
            optimizer.zero_grad()
            logits = model(X_train[batch_idx])
            loss = criterion(logits, y_train[batch_idx])
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(X_val)).detach().cpu().numpy().reshape(-1)
    y_val = y[val_idx].astype(np.int64)
    result = SmokeTrainResult(
        train_loss_final=round(final_loss, 6),
        val_auc=round(_roc_auc(y_val, scores), 6),
        val_precision=round(_precision(y_val, scores), 6),
        val_recall=round(_recall(y_val, scores), 6),
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        n_positive_train=int(y[train_idx].sum()),
        n_positive_val=int(y[val_idx].sum()),
        elapsed_sec=round(perf_counter() - start, 3),
    )
    # return_model=True 면 학습된 모델(eval 모드)을 함께 반환해 저장에 쓸 수 있게 한다.
    # 기본 False 는 기존 호출부/테스트 하위호환(SmokeTrainResult 단일 반환).
    if return_model:
        return result, model
    return result


def run_raw_training_smoke(
    raw_dir: str | Path,
    vocab_path: str | Path,
    *,
    n_patients: int = 5000,
    reference_date: date | None = None,
    lookback_days: int = 60,
    epochs: int = 30,
    batch_size: int = 256,
    seed: int = 42,
    device: str = "cpu",
    label_source: str = "sick_code",
    multi_institution_threshold: int = MULTI_INSTITUTION_THRESHOLD,
    therapeutic_dup_threshold: int = THERAPEUTIC_DUP_THRESHOLD,
    save_model_path: str | Path | None = None,
) -> dict:
    raw_path = Path(raw_dir)
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    resolved_reference_date = reference_date or _latest_records_date(raw_path)
    patient_ids = _sample_patient_ids(raw_path, resolved_reference_date, n_patients, seed)
    provider = MultiDayParquetHistoryProvider(
        raw_path,
        extra_columns=[_extra_column_for_label_source(label_source)],
        deduplicate_keys=False,
    )
    histories = provider.get_history_batch(
        patient_ids,
        reference_date=resolved_reference_date,
        lookback_days=lookback_days,
    )
    labels, label_metadata = build_training_labels(
        patient_ids,
        histories,
        label_source=label_source,
        multi_institution_threshold=multi_institution_threshold,
        therapeutic_dup_threshold=therapeutic_dup_threshold,
    )
    X, y = build_dataset(
        provider,
        vocab,
        patient_ids,
        labels,
        reference_date=resolved_reference_date,
        lookback_days=lookback_days,
    )
    train_result, trained_model = train_mlp_smoke(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
        return_model=True,
    )
    if save_model_path is not None:
        _save_model_smoke(trained_model, X.shape[1], save_model_path, device=device)
    return {
        "reference_date": resolved_reference_date.isoformat(),
        "lookback_days": lookback_days,
        "n_patients": len(patient_ids),
        "input_dim": int(X.shape[1]),
        "label_positive": int(y.sum()),
        "label_positive_rate_pct": _pct(int(y.sum()), len(y)),
        **label_metadata,
        "train": asdict(train_result),
    }


def write_training_report(report: dict, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "mlp_smoke_report.json"
    md_path = output_path / "mlp_smoke_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def _seed_everything(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    comparisons = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(positives) * len(negatives)))


def _precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    predicted = scores >= 0.5
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    return 0.0 if tp + fp == 0 else tp / (tp + fp)


def _recall(y_true: np.ndarray, scores: np.ndarray) -> float:
    predicted = scores >= 0.5
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    return 0.0 if tp + fn == 0 else tp / (tp + fn)


def _sample_patient_ids(raw_dir: Path, reference_date: date, n_patients: int, seed: int) -> list[str]:
    path = raw_dir / f"records_{reference_date.strftime('%Y%m%d')}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"reference records file not found: {path}")
    df = pd_read_parquet(path, columns=["patient_id"])
    ids = df["patient_id"].dropna().astype(str).drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    if len(ids) <= n_patients:
        return ids
    selected = rng.choice(np.array(ids, dtype=object), size=n_patients, replace=False)
    return [str(value) for value in selected.tolist()]


def pd_read_parquet(path: Path, columns: list[str]):
    import pandas as pd

    return pd.read_parquet(path, columns=columns)


def _latest_records_date(raw_dir: Path) -> date:
    dates = []
    for path in raw_dir.glob("records_*.parquet"):
        try:
            dates.append(datetime.strptime(path.stem.removeprefix("records_"), "%Y%m%d").date())
        except ValueError:
            continue
    if not dates:
        raise FileNotFoundError(f"no records_YYYYMMDD.parquet files found in {raw_dir}")
    return max(dates)


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y%m%d").date()


def _save_model_smoke(model, input_dim: int, path: str | Path, *, device: str = "cpu") -> None:
    """학습된 모델을 TorchScript(model.pt)로 저장한다.

    서빙 로더(serving.dl_predictor.DLModel)가 ``torch.jit.load`` 로 model.pt 를
    로드하므로, state_dict 가 아니라 jit.trace 결과를 저장해야 한다. 또한 방금
    학습한 ``model`` 을 그대로 저장한다(과거엔 새 untrained 모델을 만들어 저장하던
    결함이 있었음). 입력은 (1, input_dim) multi-hot 벡터 형태.
    """
    torch = _torch()
    model.eval()
    example = torch.zeros(1, int(input_dim), dtype=torch.float32, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)


def _extra_column_for_label_source(label_source: str) -> str:
    if label_source == "sick_code":
        return "sick_code"
    if label_source == "multi_institution":
        return "institution_id"
    if label_source == "therapeutic_duplication":
        return "efmdc_clsf_no"
    raise ValueError(f"unsupported label_source: {label_source}")


def _markdown_report(report: dict) -> str:
    train = report["train"]
    return "\n".join([
        "# MLP Smoke Report",
        "",
        f"- reference_date: {report['reference_date']}",
        f"- lookback_days: {report['lookback_days']}",
        f"- n_patients: {report['n_patients']}",
        f"- input_dim: {report['input_dim']}",
        f"- label_source: {report['label_source']}",
        f"- label_semantics: {report['label_semantics']}",
        f"- label_positive: {report['label_positive']}",
        f"- label_positive_rate_pct: {report['label_positive_rate_pct']}",
        "",
        "| metric | value |",
        "|---|---:|",
        *[f"| {key} | {value} |" for key, value in train.items()],
        "",
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 1 MLP training smoke.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("--output-dir", default="data/vocab")
    parser.add_argument("--n-patients", type=int, default=5000)
    parser.add_argument("--reference-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--label-source", choices=["sick_code", "multi_institution", "therapeutic_duplication"], default="sick_code")
    parser.add_argument("--multi-institution-threshold", type=int, default=MULTI_INSTITUTION_THRESHOLD)
    parser.add_argument("--therapeutic-dup-threshold", type=int, default=THERAPEUTIC_DUP_THRESHOLD)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--model-path", default="data/models/mlp_smoke.pt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_raw_training_smoke(
        args.raw_dir,
        args.vocab_path,
        n_patients=args.n_patients,
        reference_date=_parse_date(args.reference_date),
        lookback_days=args.lookback_days,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        label_source=args.label_source,
        multi_institution_threshold=args.multi_institution_threshold,
        therapeutic_dup_threshold=args.therapeutic_dup_threshold,
        save_model_path=args.model_path if args.save_model else None,
    )
    json_path, md_path = write_training_report(report, args.output_dir)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    print(f"val_auc={report['train']['val_auc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
