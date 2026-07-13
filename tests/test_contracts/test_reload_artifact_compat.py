"""Characterize reload behavior and synthetic artifact compatibility."""
from __future__ import annotations

import ast
import hashlib
import inspect
import pickle
import pickletools
import threading
from collections.abc import Callable, Mapping, Sized
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from serving.predictor import HybridPredictor


class SyntheticProbabilityModel:
    """Picklable module-level model used only in trusted tmp_path artifacts."""

    def predict_proba(self, rows: Sized) -> list[list[float]]:
        return [[0.3, 0.7] for _ in range(len(rows))]


def _write_trusted_model(path: Path, feature_names: list[str]) -> Path:
    from scripts.etl.prescription_aggregator import DDI_FEATURE_SEMANTICS_VERSION

    payload = {
        "model": SyntheticProbabilityModel(),
        "best_threshold": 0.5,
        "trainer_class": "XGBoostTrainer",
        "feature_names": feature_names,
        "artifact_version": 2,
        "ddi_feature_semantics_version": DDI_FEATURE_SEMANTICS_VERSION,
    }
    content = pickle.dumps(payload)
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return path


def _hybrid() -> HybridPredictor:
    from serving.predictor import HybridPredictor, MLModel

    predictor = HybridPredictor.__new__(HybridPredictor)
    predictor._ml = MLModel()
    predictor._hierarchical = None
    predictor._dl = MagicMock(runtime_lookback_days=365)
    predictor._ml_lock = threading.RLock()
    predictor._hier_lock = threading.RLock()
    predictor._dl_lock = threading.RLock()
    return predictor


def _global_references_without_loading(blob: bytes) -> set[str]:
    """Inspect protocol 0 GLOBAL opcodes without executing pickle bytes."""
    references = set()
    for opcode, argument, _position in pickletools.genops(blob):
        if opcode.name == "GLOBAL":
            module, name = str(argument).split(" ", 1)
            references.add(f"{module}.{name}")
    return references


def _dict_string_keys(node: ast.Dict) -> list[str]:
    keys: list[str] = []
    for key in node.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise AssertionError("expected a string literal dict key")
        keys.append(key.value)
    return keys


def _assigned_dict_keys(
    function: Callable[..., object], variable_name: str
) -> list[str]:
    tree = ast.parse(inspect.getsource(function))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            assert isinstance(node.value, ast.Dict)
            return _dict_string_keys(node.value)
    raise AssertionError(f"dict assignment not found: {variable_name}")


def _joblib_dump_dict_keys(
    function: Callable[..., object], filename: str
) -> list[str]:
    tree = ast.parse(inspect.getsource(function))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "joblib"
            and node.func.attr == "dump"
        ):
            continue
        destination = node.args[1]
        if not (isinstance(destination, ast.Name) and destination.id == filename):
            continue
        payload = node.args[0]
        assert isinstance(payload, ast.Dict)
        return _dict_string_keys(payload)
    raise AssertionError(f"joblib payload not found: {filename}")


def _stamp_pipeline_metadata(
    trainer: object, metadata: Mapping[str, object]
) -> None:
    setattr(trainer, "_extra_meta", dict(metadata))


def test_reload_model_success_swaps_model(tmp_path: Path):
    predictor = _hybrid()
    old = predictor._ml
    model_path = _write_trusted_model(tmp_path / "model.pkl", ["drug_count", "age"])

    assert predictor.reload_model(model_path) is True
    assert predictor._ml is not old
    assert predictor._ml.loaded is True


def test_reload_model_failure_preserves_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    predictor = _hybrid()
    old = predictor._ml
    model_path = _write_trusted_model(
        tmp_path / "bad.pkl", ["drug_count", "unknown_feature"]
    )

    assert predictor.reload_model(model_path) is False
    assert predictor._ml is old


def test_reload_hierarchical_false_preserves_existing(
    monkeypatch: pytest.MonkeyPatch,
):
    import serving.predictor as predictor_module
    from serving.predictor import HierarchicalPredictor

    predictor = _hybrid()
    existing = MagicMock(spec=HierarchicalPredictor)
    predictor._hierarchical = existing
    candidate = MagicMock(spec=HierarchicalPredictor)
    candidate.load.return_value = False
    monkeypatch.setattr(
        predictor_module, "HierarchicalPredictor", MagicMock(return_value=candidate)
    )

    assert predictor.reload_hierarchical("synthetic") is False
    assert predictor._hierarchical is existing


def test_reload_hierarchical_rechecks_nonempty_feature_cols(
    monkeypatch: pytest.MonkeyPatch,
):
    import serving.predictor as predictor_module
    from serving.predictor import HierarchicalPredictor

    predictor = _hybrid()
    existing = MagicMock(spec=HierarchicalPredictor)
    predictor._hierarchical = existing
    candidate = MagicMock(spec=HierarchicalPredictor)
    candidate.feature_cols = []
    candidate.load.return_value = True
    monkeypatch.setattr(
        predictor_module, "HierarchicalPredictor", MagicMock(return_value=candidate)
    )

    assert predictor.reload_hierarchical("synthetic") is False
    assert predictor._hierarchical is existing


def test_reload_hierarchical_success_swaps_candidate(
    monkeypatch: pytest.MonkeyPatch,
):
    import serving.predictor as predictor_module
    from serving.predictor import _FEATURE_ALLOWED, HierarchicalPredictor

    predictor = _hybrid()
    candidate = MagicMock(spec=HierarchicalPredictor)
    candidate.load.return_value = True
    valid_feature_cols = ["drug_count"]
    assert set(valid_feature_cols).issubset(_FEATURE_ALLOWED)
    candidate.feature_cols = valid_feature_cols
    monkeypatch.setattr(
        predictor_module, "HierarchicalPredictor", MagicMock(return_value=candidate)
    )

    assert predictor.reload_hierarchical("synthetic") is True
    assert predictor._hierarchical is candidate


def test_reload_dl_invalid_bundle_raises_and_preserves_existing(tmp_path: Path):
    predictor = _hybrid()
    existing = predictor._dl
    empty_bundle = tmp_path / "empty_dl"
    empty_bundle.mkdir()

    with pytest.raises(FileNotFoundError):
        predictor.reload_dl(empty_bundle)
    assert predictor._dl is existing


def test_reload_dl_swaps_even_when_load_returns_false(
    monkeypatch: pytest.MonkeyPatch,
):
    import serving.predictor as predictor_module

    predictor = _hybrid()
    candidate = MagicMock()
    candidate.load.return_value = False
    factory = MagicMock(return_value=candidate)
    monkeypatch.setattr(predictor_module, "DLModel", factory)

    assert predictor.reload_dl("synthetic") is True
    assert predictor._dl is candidate
    candidate.load.assert_called_once_with("synthetic")


def test_base_trainer_payload_keys_match_current_writer(tmp_path: Path):
    from scripts.train.trainer import XGBoostTrainer

    trainer = XGBoostTrainer(params={}, config=None)
    _stamp_pipeline_metadata(
        trainer,
        {
            "artifact_version": 2,
            "feature_names": ["drug_count"],
            "scaler_path": "scaler.pkl",
            "selector_path": "selector.pkl",
            "ddi_feature_semantics_version": "ddi.v2",
        },
    )
    path = trainer.save(tmp_path / "single.pkl")
    payload = pickle.loads(path.read_bytes())

    assert set(payload) == {
        "model",
        "params",
        "feature_importances",
        "best_threshold",
        "trainer_class",
        "artifact_version",
        "feature_names",
        "scaler_path",
        "selector_path",
        "ddi_feature_semantics_version",
    }
    assert "partition" not in payload


def test_ensemble_top_level_payload_omits_model_and_params(tmp_path: Path):
    from scripts.train.trainer import EnsembleTrainer, LGBMTrainer, XGBoostTrainer

    trainer = EnsembleTrainer.__new__(EnsembleTrainer)
    trainer._xgb = XGBoostTrainer(params={}, config=None)
    trainer._lgb = LGBMTrainer(params={}, config=None)
    trainer.weights = (0.5, 0.5)
    trainer.best_threshold_ = 0.5
    trainer.feature_importances_ = None
    _stamp_pipeline_metadata(
        trainer, {"artifact_version": 2, "feature_names": ["drug_count"]}
    )
    path = trainer.save(tmp_path / "ensemble.pkl")
    payload = pickle.loads(path.read_bytes())

    assert set(payload) == {
        "trainer_class",
        "weights",
        "best_threshold",
        "feature_importances",
        "artifact_version",
        "feature_names",
    }
    assert "model" not in payload
    assert "params" not in payload
    assert path.with_suffix(".xgb.pkl").exists()
    assert path.with_suffix(".lgb.pkl").exists()


def test_hierarchical_writer_metadata_and_bundle_key_sets():
    from hana_app.core.hierarchical_runner import train_hierarchical

    assert _assigned_dict_keys(train_hierarchical, "meta") == [
        "clinical_standards_version",
        "ddi_feature_semantics_version",
        "feature_semantics_version",
        "feature_cols",
        "thresholds",
        "stage2_labels",
        "stage2_label_counts",
        "y_other_excluded_count",
        "stage1_sha256",
        "stage2_sha256",
        "cost_sensitive",
        "cost_ratio_by_class",
        "stage1_trained",
        "stage1_red_count",
    ]
    assert _joblib_dump_dict_keys(train_hierarchical, "p2") == [
        "model",
        "encoder",
        "stage2_classes_global",
        "classes_present",
    ]


def test_untrusted_pickle_module_path_inspection_does_not_execute():
    blob = b"ctests.test_contracts.test_reload_artifact_compat\nSyntheticProbabilityModel\n."

    assert _global_references_without_loading(blob) == {
        "tests.test_contracts.test_reload_artifact_compat.SyntheticProbabilityModel"
    }


def test_constant_stage1_pickle_records_stable_module_path():
    from hana_app.core.hierarchical_runner import _ConstantNegativeStage1

    trusted_blob = pickle.dumps(_ConstantNegativeStage1(), protocol=0)
    stable_reference = (
        "hana_app.core.hierarchical_runner._ConstantNegativeStage1"
    )
    references = _global_references_without_loading(trusted_blob)
    assert stable_reference in references
    assert references - {stable_reference} <= {
        "__builtin__.object",
        "builtins.object",
        "copy_reg._reconstructor",
        "copyreg._reconstructor",
    }
