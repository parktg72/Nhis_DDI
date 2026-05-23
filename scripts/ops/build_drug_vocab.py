"""Build deterministic drug_vocab.json from raw EDI frequency audit."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops.build_drug_vocab_audit import build_vocab_audit


SORT_POLICY = "freq_desc_lex_asc"
UNK_TOKEN = "_unk"
UNK_INDEX = 0


@dataclass(frozen=True)
class VocabBuildMeta:
    cutoff: int
    vocab_size: int
    input_dim: int
    unk_index: int
    sort_policy: str
    date_range: tuple[str | None, str | None]
    total_files: int


def build_drug_vocab(
    code_freq: dict[str, int],
    cutoff: int = 100,
) -> dict[str, int]:
    if cutoff < 0:
        raise ValueError("cutoff must be non-negative")

    included = [
        (str(code), int(freq))
        for code, freq in code_freq.items()
        if str(code) != UNK_TOKEN and int(freq) >= cutoff
    ]
    included.sort(key=lambda item: (-item[1], item[0]))

    vocab = {UNK_TOKEN: UNK_INDEX}
    for index, (code, _) in enumerate(included, start=1):
        vocab[code] = index
    return vocab


def write_vocab_outputs(
    vocab: dict[str, int],
    meta: VocabBuildMeta,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vocab_path = output_path / "drug_vocab.json"
    meta_path = output_path / "drug_vocab_meta.json"

    vocab_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="",
    )
    meta_path.write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="",
    )
    return vocab_path, meta_path


def build_vocab_from_raw(
    raw_dir: str | Path,
    *,
    cutoff: int = 100,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[dict[str, int], VocabBuildMeta]:
    audit = build_vocab_audit(
        raw_dir,
        date_from=date_from,
        date_to=date_to,
        cutoffs=(cutoff,),
    )
    code_freq = {
        code: stats.row_count
        for code, stats in audit.code_stats.items()
    }
    vocab = build_drug_vocab(code_freq, cutoff=cutoff)
    meta = VocabBuildMeta(
        cutoff=cutoff,
        vocab_size=len(vocab) - 1,
        input_dim=len(vocab),
        unk_index=UNK_INDEX,
        sort_policy=SORT_POLICY,
        date_range=audit.meta.date_range,
        total_files=audit.meta.total_files,
    )
    return vocab, meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic drug_vocab.json.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", default="data/vocab")
    parser.add_argument("--cutoff", type=int, default=100)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    vocab, meta = build_vocab_from_raw(
        args.raw_dir,
        cutoff=args.cutoff,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    vocab_path, meta_path = write_vocab_outputs(vocab, meta, args.output_dir)
    print(f"[OK] wrote {vocab_path}")
    print(f"[OK] wrote {meta_path}")
    print(f"vocab_size={meta.vocab_size} input_dim={meta.input_dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
