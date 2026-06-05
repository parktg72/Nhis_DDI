"""Operational DL bundle loader and minimal inference runner.

This module intentionally does not import torch at module import time. Bundle
validation remains lightweight; tensor/model loading is lazy and happens only
when DL prediction is requested.
"""
from __future__ import annotations

import copy
import importlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config import settings
from scripts.datasets.contracts import (
    validate_dl_bundle_manifest,
    validate_lookback_consistency,
    validate_lookback_days,
)
from serving.hana_history import validate_history_frame

logger = logging.getLogger(__name__)

# 학습측 인코더는 multi_hot 만 존재한다(scripts/ops/multihot_encoder). "count" 는
# 학습 경로가 전무한 dead infra 였어 오설정 번들이 조용히 수용되는 것을 막기 위해 제거.
_SUPPORTED_ENCODING_STRATEGIES = {"multi_hot"}
_GRAPH_ARCHITECTURES = {"gat", "gcn"}


class DLModel:
    """Validate and hold metadata for an operational DL artifact bundle."""

    def __init__(self, runtime_lookback_days: Optional[int] = None) -> None:
        if runtime_lookback_days is None:
            runtime_lookback_days = settings.HANA_HISTORY_LOOKBACK_DAYS
        self._runtime_lookback_days = validate_lookback_days(runtime_lookback_days)
        self._bundle_dir: Optional[Path] = None
        self._manifest: Optional[dict] = None
        self._runtime_loaded = False
        self._model: Any = None
        self._model_config: Optional[dict] = None
        self._drug_vocab: Optional[dict[str, int]] = None
        self._edge_index: Any = None
        self._feature_normalizer: Any = None

    @property
    def runtime_lookback_days(self) -> int:
        return self._runtime_lookback_days

    @property
    def is_loaded(self) -> bool:
        return self._bundle_dir is not None and self._manifest is not None

    @property
    def loaded(self) -> bool:
        return self.is_loaded

    @property
    def bundle_dir(self) -> Optional[Path]:
        return self._bundle_dir

    @property
    def manifest(self) -> Optional[dict]:
        if self._manifest is None:
            return None
        return copy.deepcopy(self._manifest)

    @property
    def runtime_loaded(self) -> bool:
        return self._runtime_loaded

    @property
    def run_id(self) -> Optional[str]:
        return None if self._manifest is None else str(self._manifest.get("run_id"))

    @property
    def schema_version(self) -> Optional[str]:
        if self._manifest is None:
            return None
        return str(self._manifest.get("schema_version"))

    @property
    def lookback_days(self) -> Optional[int]:
        if self._manifest is None:
            return None
        return int(self._manifest["lookback_days"])

    def validate_bundle(self, bundle_dir: str | Path) -> dict:
        """Validate manifest/hash integrity and STRICT lookback compatibility."""
        manifest = validate_dl_bundle_manifest(bundle_dir)
        validate_lookback_consistency(
            manifest["lookback_days"],
            self._runtime_lookback_days,
            context="dl bundle load",
        )
        return manifest

    def load(self, bundle_dir: str | Path) -> bool:
        """Load bundle metadata after validation.

        Invalid bundles raise their original validation exception. Instance
        state is updated only after every check passes.
        """
        root = Path(bundle_dir)
        manifest = self.validate_bundle(root)
        self._bundle_dir = root
        self._manifest = manifest
        self._clear_runtime()
        logger.info(
            "DL bundle validated: %s (run_id=%s, schema=%s, lookback_days=%s)",
            root,
            manifest.get("run_id"),
            manifest.get("schema_version"),
            manifest.get("lookback_days"),
        )
        return True

    def predict(self, history_df) -> dict[str, object]:
        """Run minimal DL inference for a normalized patient history frame.

        The first supported serving contract is fixed-size drug vector encoding.
        Training must record the same `encoding_strategy`, `input_dim`, and
        `output_labels` in `model_config.json`.
        """
        if not self.is_loaded or self._bundle_dir is None:
            raise RuntimeError("DL bundle is not loaded")

        history = validate_history_frame(
            history_df.copy(),
            context="dl predict history",
        )
        self._ensure_runtime_loaded()
        if (
            self._model_config is None
            or self._drug_vocab is None
            or self._model is None
        ):
            raise RuntimeError("DL runtime artifacts are not loaded")

        features, known_count, unknown_count = self._encode_history(history)
        features = self._apply_normalizer(features)
        torch = self._torch()
        device = self._resolve_device(torch, self._model_config)
        tensor = torch.tensor([features], dtype=torch.float32, device=device)

        with torch.no_grad():
            output = self._predict_forward(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            probabilities = self._to_probabilities(torch, output)

        labels = list(self._model_config["output_labels"])
        if len(probabilities) != len(labels):
            raise ValueError(
                "DL model output dimension mismatch: "
                f"labels={len(labels)}, probabilities={len(probabilities)}"
            )
        probability_map = {
            label: float(prob)
            for label, prob in zip(labels, probabilities)
        }
        predicted_label, score = max(
            probability_map.items(),
            key=lambda item: item[1],
        )
        return {
            "run_id": self.run_id,
            "encoding_strategy": self._model_config["encoding_strategy"],
            "predicted_label": predicted_label,
            "score": float(score),
            "probabilities": probability_map,
            "known_drug_count": known_count,
            "unknown_drug_count": unknown_count,
        }

    def _clear_runtime(self) -> None:
        self._runtime_loaded = False
        self._model = None
        self._model_config = None
        self._drug_vocab = None
        self._edge_index = None
        self._feature_normalizer = None

    def _ensure_runtime_loaded(self) -> None:
        if self._runtime_loaded:
            return
        if self._bundle_dir is None:
            raise RuntimeError("DL bundle is not loaded")

        # Re-validate before loading pickle/torch artifacts. This preserves the
        # existing hash gate and avoids loading tampered sidecars.
        self.validate_bundle(self._bundle_dir)
        torch = self._torch()
        model_config = self._load_model_config(self._bundle_dir / "model_config.json")
        drug_vocab = self._load_drug_vocab(
            self._bundle_dir / "drug_vocab.json",
            model_config["input_dim"],
        )
        device = self._resolve_device(torch, model_config)
        model = torch.jit.load(str(self._bundle_dir / "model.pt"), map_location=device)
        if hasattr(model, "eval"):
            model.eval()
        edge_index = torch.load(
            str(self._bundle_dir / "edge_index.pt"),
            map_location=device,
            weights_only=True,
        )
        feature_normalizer = pickle.loads(
            (self._bundle_dir / "feature_normalizer.pkl").read_bytes()
        )

        self._model_config = model_config
        self._drug_vocab = drug_vocab
        self._model = model
        self._edge_index = edge_index
        self._feature_normalizer = feature_normalizer
        self._runtime_loaded = True
        logger.info("DL runtime artifacts loaded: %s", self._bundle_dir)

    @staticmethod
    def _torch():
        return importlib.import_module("torch")

    @staticmethod
    def _resolve_device(torch, model_config: dict) -> str:
        configured = str(model_config.get("device", "auto")).lower()
        if configured != "auto":
            return configured
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _load_model_config(path: Path) -> dict:
        config = json.loads(path.read_text(encoding="utf-8"))
        strategy = str(config.get("encoding_strategy", ""))
        if strategy not in _SUPPORTED_ENCODING_STRATEGIES:
            raise ValueError(
                "unsupported DL encoding_strategy: "
                f"{strategy!r} (allowed={sorted(_SUPPORTED_ENCODING_STRATEGIES)})"
            )
        input_dim = int(config.get("input_dim", 0))
        if input_dim <= 0:
            raise ValueError("model_config input_dim must be positive")
        labels = config.get("output_labels")
        if not isinstance(labels, list) or not labels or not all(labels):
            raise ValueError("model_config output_labels must be a non-empty list")
        return {
            **config,
            "encoding_strategy": strategy,
            "input_dim": input_dim,
            "output_labels": [str(label) for label in labels],
        }

    @staticmethod
    def _load_drug_vocab(path: Path, input_dim: int) -> dict[str, int]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not raw:
            raise ValueError("drug_vocab must be a non-empty object")
        vocab: dict[str, int] = {}
        for code, index in raw.items():
            idx = int(index)
            if idx < 0 or idx >= input_dim:
                raise ValueError(
                    f"drug_vocab index out of range for {code!r}: {idx}"
                )
            vocab[str(code)] = idx
        # 운영 vocab(scripts/ops/build_drug_vocab)은 항상 "_unk"(index 0)를 포함한다.
        # _unk 가 없으면 _encode_history 가 OOV 약물을 조용히 드롭하므로(하위호환 경로),
        # 적어도 1회 경고를 남겨 silent train/serve skew 를 가시화한다. raise 하지 않는 이유:
        # _unk 없는 구형/토이 번들 하위호환을 유지(테스트 명시)하기 위함.
        if "_unk" not in vocab:
            logger.warning(
                "drug_vocab has no '_unk' token (%s): OOV drugs will be dropped "
                "silently instead of mapped to the _unk dimension",
                path,
            )
        return vocab

    def _encode_history(self, history_df) -> tuple[list[float], int, int]:
        assert self._model_config is not None
        assert self._drug_vocab is not None
        strategy = self._model_config["encoding_strategy"]
        features = [0.0] * int(self._model_config["input_dim"])
        known_count = 0
        unknown_count = 0
        # train/serve OOV 정합: 학습 인코더(scripts/ops/multihot_encoder.encode_patient_history)는
        # 미지 약물을 vocab["_unk"] 차원에 반영한다. _unk 가 vocab 에 있으면 서빙도 동일하게
        # 반영해 silent skew 를 막는다. _unk 없는 구형/토이 번들은 종전처럼 무시(하위호환).
        unk_idx = self._drug_vocab.get("_unk")
        for raw_code in history_df["drug_code"]:
            # 학습 인코더(_normalized_drug_codes)는 dropna 후 strip·빈값 제거한다.
            # None/np.nan/pd.NA 를 pd.isna 로 정확히 걸러 동일 동작을 보장한다.
            if pd.isna(raw_code):
                continue
            code = str(raw_code).strip()
            if not code:
                continue
            idx = self._drug_vocab.get(code)
            if idx is None:
                unknown_count += 1
                idx = unk_idx
                if idx is None:
                    continue  # 구형 번들(_unk 미포함): 미지 약물 무시
            else:
                known_count += 1
            if strategy == "multi_hot":
                features[idx] = 1.0
            else:  # defensive; _load_model_config 가 encoding_strategy 를 검증한다.
                raise ValueError(f"unsupported DL encoding_strategy: {strategy!r}")
        return features, known_count, unknown_count

    def _apply_normalizer(self, features: list[float]) -> list[float]:
        normalizer = self._feature_normalizer
        if not normalizer:
            return features
        if isinstance(normalizer, dict):
            if normalizer.get("type") in (None, "identity"):
                return features
            if normalizer.get("type") == "standard":
                mean = normalizer.get("mean", [0.0] * len(features))
                scale = normalizer.get("scale", [1.0] * len(features))
                if len(mean) != len(features) or len(scale) != len(features):
                    raise ValueError(
                        "feature_normalizer dimensions do not match input_dim"
                    )
                return [
                    (value - float(mu)) / (float(sigma) or 1.0)
                    for value, mu, sigma in zip(features, mean, scale)
                ]
        if hasattr(normalizer, "transform"):
            transformed = normalizer.transform([features])
            return [float(value) for value in transformed[0]]
        raise ValueError("unsupported feature_normalizer artifact")

    def _predict_forward(self, tensor):
        assert self._model_config is not None
        assert self._model is not None
        architecture = str(self._model_config.get("architecture", "linear")).lower()
        if architecture in _GRAPH_ARCHITECTURES:
            if self._edge_index is None:
                raise RuntimeError("DL graph architecture requires edge_index")
            return self._model(tensor, self._edge_index)
        return self._model(tensor)

    @staticmethod
    def _to_probabilities(torch, output) -> list[float]:
        values = output.detach().cpu().tolist()
        row = values[0] if values and isinstance(values[0], list) else values
        if len(row) == 1:
            prob_pos = float(torch.sigmoid(output).detach().cpu().tolist()[0][0])
            return [1.0 - prob_pos, prob_pos]
        probs = torch.softmax(output, dim=-1).detach().cpu().tolist()[0]
        return [float(value) for value in probs]
