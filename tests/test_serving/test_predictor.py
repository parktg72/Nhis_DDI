"""serving/predictor.py MLModel / _run_safety_net / HybridPredictor 단위 테스트."""
import hashlib
import pickle
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from serving.predictor import (
    MLModel, HybridPredictor, RequestFeatureBuilder,
    _run_safety_net, _run_duplicate_detector,
)
from serving.schemas import DrugItem, RiskLevel


# ─── 헬퍼 ───────────────────────────────────────────────────────────────────

class _FakeSklearnModel:
    """sklearn 없이 predict_proba를 흉내내는 최소 객체."""
    def predict_proba(self, X):
        n = len(X)
        prob = np.clip(X[:, 0] / (X[:, 0].max() + 1e-9), 0.01, 0.99)
        return np.column_stack([1 - prob, prob])


def _write_model_pkl(
    path: Path,
    model=None,
    trainer_class: str = "XGBoostTrainer",
    feature_names=None,
    threshold: float = 0.5,
    extra: dict = None,
) -> Path:
    """유효한 모델 pkl + sha256 사이드카 생성. 테스트 픽스처용."""
    payload = {
        "model": model or _FakeSklearnModel(),
        "best_threshold": threshold,
        "trainer_class": trainer_class,
        "feature_names": feature_names or ["f1", "f2"],
        "artifact_version": 2,
        **(extra or {}),
    }
    content = pickle.dumps(payload)
    path.write_bytes(content)
    sha256 = hashlib.sha256(content).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256}  {path.name}\n"
    )
    return path


# ─── MLModel.load 테스트 ──────────────────────────────────────────────────────

class TestMLModelLoad:

    def test_load_valid_model(self, tmp_path):
        """유효한 pkl + sha256 → loaded=True."""
        path = tmp_path / "model.pkl"
        _write_model_pkl(path)
        ml = MLModel()
        assert ml.load(path) is True
        assert ml.loaded is True
        assert ml._threshold == 0.5
        assert ml._feature_names == ["f1", "f2"]

    def test_load_missing_sha256_returns_false(self, tmp_path):
        """sha256 파일 없으면 로드 거부."""
        path = tmp_path / "model.pkl"
        payload = {"model": _FakeSklearnModel(), "best_threshold": 0.5,
                   "trainer_class": "X", "feature_names": []}
        path.write_bytes(pickle.dumps(payload))
        # sha256 파일 생성 안 함
        ml = MLModel()
        assert ml.load(path) is False
        assert ml.loaded is False

    def test_load_hash_mismatch_returns_false(self, tmp_path):
        """sha256 불일치 → 로드 거부."""
        path = tmp_path / "model.pkl"
        _write_model_pkl(path)
        # sha256를 오염시킴
        path.with_suffix(".pkl.sha256").write_text("deadbeef  model.pkl\n")
        ml = MLModel()
        assert ml.load(path) is False
        assert ml.loaded is False

    def test_load_non_dict_content_returns_false(self, tmp_path):
        """dict가 아닌 pkl → 로드 거부."""
        path = tmp_path / "model.pkl"
        content = pickle.dumps([1, 2, 3])
        path.write_bytes(content)
        sha256 = hashlib.sha256(content).hexdigest()
        path.with_suffix(".pkl.sha256").write_text(f"{sha256}  model.pkl\n")
        ml = MLModel()
        assert ml.load(path) is False

    def test_load_ensemble_loads_submodels(self, tmp_path):
        """EnsembleTrainer pkl + .xgb.pkl + .lgb.pkl + sha256 → loaded=True."""
        main_path = tmp_path / "model_ens.pkl"
        # 메인 파일 (model=None, trainer_class=EnsembleTrainer)
        payload = {
            "model": None,
            "trainer_class": "EnsembleTrainer",
            "weights": (0.5, 0.5),
            "best_threshold": 0.4,
            "feature_names": ["f1", "f2"],
            "artifact_version": 2,
        }
        content = pickle.dumps(payload)
        main_path.write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()
        main_path.with_suffix(".pkl.sha256").write_text(f"{sha}  model_ens.pkl\n")

        # 서브모델 생성
        for ext in (".xgb.pkl", ".lgb.pkl"):
            sub = tmp_path / f"model_ens{ext}"
            sub_payload = {"model": _FakeSklearnModel()}
            sub_content = pickle.dumps(sub_payload)
            sub.write_bytes(sub_content)
            sub_sha = hashlib.sha256(sub_content).hexdigest()
            sub.with_suffix(sub.suffix + ".sha256").write_text(
                f"{sub_sha}  model_ens{ext}\n"
            )

        ml = MLModel()
        assert ml.load(main_path) is True
        assert ml.loaded is True

    def test_load_ensemble_missing_submodel_returns_false(self, tmp_path):
        """EnsembleTrainer 서브모델 없으면 loaded=False."""
        main_path = tmp_path / "model_ens.pkl"
        payload = {
            "model": None,
            "trainer_class": "EnsembleTrainer",
            "weights": (0.5, 0.5),
            "best_threshold": 0.4,
            "feature_names": [],
            "artifact_version": 2,
        }
        content = pickle.dumps(payload)
        main_path.write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()
        main_path.with_suffix(".pkl.sha256").write_text(f"{sha}  model_ens.pkl\n")
        # 서브모델 파일 없음
        ml = MLModel()
        assert ml.load(main_path) is False


# ─── MLModel.classify 테스트 ──────────────────────────────────────────────────

class TestMLModelClassify:
    """MLModel.classify — 임계값 경계 조건 검증."""

    @pytest.fixture
    def ml_loaded(self, tmp_path):
        path = tmp_path / "model.pkl"
        _write_model_pkl(path, threshold=0.5)
        ml = MLModel()
        ml.load(path)
        return ml

    def test_classify_above_threshold_is_red(self, ml_loaded):
        """prob >= threshold → RED."""
        assert ml_loaded.classify(0.5) == RiskLevel.RED
        assert ml_loaded.classify(0.9) == RiskLevel.RED

    def test_classify_yellow_band(self, ml_loaded):
        """prob in [threshold*0.6, threshold) → YELLOW."""
        # threshold=0.5: yellow band = [0.30, 0.50)
        assert ml_loaded.classify(0.30) == RiskLevel.YELLOW
        assert ml_loaded.classify(0.49) == RiskLevel.YELLOW

    def test_classify_green_band(self, ml_loaded):
        """prob in [threshold*0.3, threshold*0.6) → GREEN."""
        # threshold=0.5: green band = [0.15, 0.30)
        assert ml_loaded.classify(0.15) == RiskLevel.GREEN
        assert ml_loaded.classify(0.29) == RiskLevel.GREEN

    def test_classify_below_green_is_normal(self, ml_loaded):
        """prob < threshold*0.3 → NORMAL."""
        # threshold=0.5: normal < 0.15
        assert ml_loaded.classify(0.0) == RiskLevel.NORMAL
        assert ml_loaded.classify(0.14) == RiskLevel.NORMAL

    def test_predict_proba_unloaded_returns_zero(self):
        """모델 미로드 시 predict_proba → 0.0."""
        ml = MLModel()
        X = np.array([[0.5, 0.5]])
        assert ml.predict_proba(X) == 0.0


# ─── _run_safety_net 테스트 ────────────────────────────────────────────────────

class TestRunSafetyNet:
    """_run_safety_net 실패 처리 검증."""

    @pytest.fixture
    def drugs(self):
        return [
            DrugItem(edi_code="A001", atc_code="B01AA03",
                     drug_name="warfarin", total_days=30),
        ]

    def test_import_error_degrades_gracefully(self, drugs):
        """SafetyNet 모듈 미설치(ImportError) → Normal 묵과 (선택적 기능)."""
        import sys
        # sys.modules에 None을 넣으면 해당 모듈 import 시 ImportError 발생
        with patch.dict(sys.modules, {"rules.safety_net": None}):
            level, reasons, alerts = _run_safety_net(drugs, sn_instance=None)
        assert level == RiskLevel.NORMAL
        assert reasons == []
        assert alerts == []

    def test_runtime_error_propagates_when_instance_provided(self, drugs):
        """sn_instance 제공 상태에서 assess() 런타임 오류 → RuntimeError 전파.

        SafetyNet이 초기화되었지만 assess()가 크래시하면
        묵과하지 않고 에러를 전파해야 DDI 탐지 실패를 숨기지 않는다.
        """
        mock_sn = MagicMock()
        mock_sn.assess.side_effect = RuntimeError("내부 오류")
        with pytest.raises(RuntimeError, match="내부 오류"):
            _run_safety_net(drugs, sn_instance=mock_sn)

    def test_no_instance_runtime_error_degrades(self, drugs):
        """sn_instance=None이고 SafetyNet() 초기화 실패 → Normal 묵과."""
        # rules.safety_net.SafetyNet 클래스 자체를 패치
        with patch("rules.safety_net.SafetyNet", side_effect=RuntimeError("로드 실패")):
            level, reasons, alerts = _run_safety_net(drugs, sn_instance=None)
        assert level == RiskLevel.NORMAL

    def test_attribute_error_propagates_when_instance_provided(self, drugs):
        """sn_instance 있을 때 assess()에서 AttributeError → 전파 (M-1).

        H-1 수정 이후: AttributeError는 outer except Exception에서 처리되어
        sn_instance is not None이면 전파됨.
        """
        mock_sn = MagicMock()
        mock_sn.assess.side_effect = AttributeError("assessment 객체 필드 누락")
        with pytest.raises(AttributeError, match="assessment 객체 필드 누락"):
            _run_safety_net(drugs, sn_instance=mock_sn)

    def test_instance_used_even_when_module_unavailable(self, drugs):
        """sn_instance 제공 + rules.safety_net 모듈 없음 → 인스턴스 정상 사용.

        Codex HIGH 수정: import는 sn_instance=None 경로에서만 실행되므로
        모듈이 sys.modules에서 제거되어도 sn_instance가 있으면 DDI 탐지 계속.
        """
        import sys
        mock_sn = MagicMock()
        mock_sn.assess.return_value = MagicMock(
            risk_grade="Normal",
            triggered_rules=[],
            ddi_pairs=[],
        )
        with patch.dict(sys.modules, {"rules.safety_net": None}):
            level, reasons, alerts = _run_safety_net(drugs, sn_instance=mock_sn)
        assert level == RiskLevel.NORMAL
        mock_sn.assess.assert_called_once()  # 인스턴스가 실제로 호출됨


# ─── HybridPredictor.reload_model 테스트 ─────────────────────────────────────

class TestHybridPredictorReloadModel:
    """HybridPredictor.reload_model — 핫스왑 단위 검증."""

    @pytest.fixture
    def predictor_no_model(self, tmp_path):
        """ML 모델 없는 빈 HybridPredictor (파일 경로 없이)."""
        pred = HybridPredictor.__new__(HybridPredictor)
        pred._start_time = 0.0
        pred._ml_lock = threading.RLock()
        pred._hier_lock = threading.RLock()
        pred._ml = MLModel()
        pred._hierarchical = None
        pred._ddi_matrix = None
        pred._cyp = None
        pred._std = None
        pred._builder = RequestFeatureBuilder()
        pred._safety_net = None
        pred._dup_detector = None
        return pred

    def test_reload_valid_model_sets_loaded(self, predictor_no_model, tmp_path):
        """유효한 pkl로 reload → ml.loaded=True."""
        path = tmp_path / "new_model.pkl"
        _write_model_pkl(path)
        assert predictor_no_model._ml.loaded is False
        ok = predictor_no_model.reload_model(path)
        assert ok is True
        assert predictor_no_model._ml.loaded is True

    def test_reload_invalid_model_keeps_old(self, predictor_no_model, tmp_path):
        """유효 모델 로드 후 깨진 파일로 reload 시도 → 기존 모델 유지."""
        # 먼저 유효 모델 로드
        valid_path = tmp_path / "valid.pkl"
        _write_model_pkl(valid_path)
        predictor_no_model.reload_model(valid_path)
        assert predictor_no_model._ml.loaded is True

        # 깨진 sha256로 재시도
        bad_path = tmp_path / "bad.pkl"
        payload = {"model": _FakeSklearnModel(), "best_threshold": 0.5,
                   "trainer_class": "X", "feature_names": []}
        bad_path.write_bytes(pickle.dumps(payload))
        bad_path.with_suffix(".pkl.sha256").write_text("badhash  bad.pkl\n")

        old_ml = predictor_no_model._ml
        ok = predictor_no_model.reload_model(bad_path)
        assert ok is False
        assert predictor_no_model._ml is old_ml  # 기존 참조 유지
        assert predictor_no_model._ml.loaded is True

    def test_reload_is_thread_safe(self, predictor_no_model, tmp_path):
        """동시 reload 시도에서 _ml 교체가 원자적이어야 함 (smoke test)."""
        import threading as _th

        path = tmp_path / "model.pkl"
        _write_model_pkl(path)
        errors = []

        def _reload():
            try:
                predictor_no_model.reload_model(path)
            except Exception as e:
                errors.append(e)

        threads = [_th.Thread(target=_reload) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"동시 reload 오류: {errors}"
        assert predictor_no_model._ml.loaded is True


# ─── _run_duplicate_detector 테스트 ──────────────────────────────────────────

class TestRunDuplicateDetector:
    """_run_duplicate_detector 실패 처리 검증."""

    @pytest.fixture
    def drugs(self):
        return [
            DrugItem(edi_code="A001", total_days=30),
            DrugItem(edi_code="A001", total_days=30),  # 동일 약물 중복
        ]

    def test_import_error_degrades_gracefully(self, drugs):
        """DuplicateDetector 모듈 미설치 → (0, []) 묵과."""
        import sys
        with patch.dict(sys.modules, {"rules.duplicate_detector": None}):
            count, reasons = _run_duplicate_detector(drugs, dd_instance=None)
        assert count == 0
        assert reasons == []

    def test_runtime_error_propagates_when_instance_provided(self, drugs):
        """dd_instance 제공 + detect() 런타임 오류 → 전파."""
        mock_dd = MagicMock()
        mock_dd.detect.side_effect = RuntimeError("탐지 내부 오류")
        with pytest.raises(RuntimeError, match="탐지 내부 오류"):
            _run_duplicate_detector(drugs, dd_instance=mock_dd)

    def test_instance_used_even_when_module_unavailable(self, drugs):
        """dd_instance 제공 + 모듈 없음 → 인스턴스 정상 사용."""
        import sys
        mock_dd = MagicMock()
        mock_dd.detect.return_value = MagicMock(
            duplicate_level1_count=1,
            duplicate_level2_count=0,
        )
        with patch.dict(sys.modules, {"rules.duplicate_detector": None}):
            count, reasons = _run_duplicate_detector(drugs, dd_instance=mock_dd)
        assert count == 1
        mock_dd.detect.assert_called_once()

    def test_no_instance_runtime_error_degrades(self, drugs):
        """dd_instance=None + DuplicateDetector() 초기화 실패 → (0, []) 묵과."""
        with patch("rules.duplicate_detector.DuplicateDetector",
                   side_effect=RuntimeError("초기화 실패")):
            count, reasons = _run_duplicate_detector(drugs, dd_instance=None)
        assert count == 0
        assert reasons == []
