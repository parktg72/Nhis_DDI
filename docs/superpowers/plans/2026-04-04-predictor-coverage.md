# serving/predictor.py 단위 테스트 보강 + 묵과형 실패 개선 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MLModel 로드/검증/분류, `_run_safety_net` 실패 구분, `HybridPredictor.reload_model` 단위 테스트 추가 및 SafetyNet 런타임 충돌 시 RuntimeError 전파로 묵과형 실패 제거.

**Architecture:** `tests/test_serving/test_predictor.py` (신규)에서 `serving/predictor.py` 내부 클래스를 직접 import해 단위 테스트. 실제 XGBoost/LightGBM 없이 pickle mock 모델 사용. `_run_safety_net` 시그니처·동작은 변경하되 기존 `test_serving.py` 통합 테스트는 유지.

**Tech Stack:** pytest, pickle, hashlib, unittest.mock, numpy, pandas

---

### Task 1: MLModel.load — 유효·해시불일치·형식오류·앙상블 단위 테스트

**Files:**
- Create: `tests/test_serving/test_predictor.py`

- [ ] **Step 1: 테스트 헬퍼 + 첫 4개 테스트 작성**

```python
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
```

- [ ] **Step 2: 테스트 실행 — RED 확인 (파일 없음)**

```bash
python3 -m pytest tests/test_serving/test_predictor.py::TestMLModelLoad -v
```
Expected: `ERROR` (파일 없음) or `ImportError`

- [ ] **Step 3: 테스트 실행 — 파일 생성 후 GREEN 확인**

```bash
python3 -m pytest tests/test_serving/test_predictor.py::TestMLModelLoad -v
```
Expected: `4 passed`

- [ ] **Step 4: 앙상블 로드 테스트 추가 (같은 파일에 이어서)**

```python
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
```

- [ ] **Step 5: 테스트 실행 — GREEN 확인**

```bash
python3 -m pytest tests/test_serving/test_predictor.py::TestMLModelLoad -v
```
Expected: `6 passed`

- [ ] **Step 6: 커밋**

```bash
git add tests/test_serving/test_predictor.py
git commit -m "test: MLModel.load 단위 테스트 6건 (유효·해시불일치·형식오류·앙상블)"
```

---

### Task 2: MLModel.classify 임계값 경계 테스트

**Files:**
- Modify: `tests/test_serving/test_predictor.py`

- [ ] **Step 1: classify 테스트 추가 (TestMLModelLoad 뒤에 이어서)**

```python
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
```

- [ ] **Step 2: 테스트 실행 — GREEN 확인**

```bash
python3 -m pytest tests/test_serving/test_predictor.py::TestMLModelClassify -v
```
Expected: `5 passed`

- [ ] **Step 3: 커밋**

```bash
git add tests/test_serving/test_predictor.py
git commit -m "test: MLModel.classify 임계값 경계 테스트 5건"
```

---

### Task 3: `_run_safety_net` 실패 구분 — ImportError 묵과 vs 런타임 충돌 전파

**Files:**
- Modify: `serving/predictor.py:93-131`
- Modify: `tests/test_serving/test_predictor.py`

- [ ] **Step 1: RED 테스트 먼저 작성**

```python
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
        # rules.safety_net.SafetyNet 클래스 자체를 패치 (from rules.safety_net import SafetyNet
        # 이 실행될 때 해당 모듈의 속성을 가져오므로 모듈 레벨을 패치해야 함)
        with patch("rules.safety_net.SafetyNet", side_effect=RuntimeError("로드 실패")):
            level, reasons, alerts = _run_safety_net(drugs, sn_instance=None)
        assert level == RiskLevel.NORMAL
```

- [ ] **Step 2: 테스트 실행 — RED 확인**

```bash
python3 -m pytest tests/test_serving/test_predictor.py::TestRunSafetyNet -v
```
Expected: `test_runtime_error_propagates_when_instance_provided` FAIL (현재 묵과함)

- [ ] **Step 3: `_run_safety_net` 수정 — sn_instance 제공 시 RuntimeError 전파**

`serving/predictor.py` line 93-131을 아래로 교체:

```python
def _run_safety_net(
    drugs: list[DrugItem],
    patient_age: Optional[int] = None,
    sn_instance=None,
) -> tuple[RiskLevel, list[str], list[DDIAlert]]:
    """
    rules/safety_net.py 실행 → (등급, 이유 목록, DDI 알림 목록).

    sn_instance 제공 + assess() 런타임 오류 → RuntimeError 전파
      (초기화된 SafetyNet이 충돌하면 DDI 탐지 실패를 숨겨선 안 됨)
    sn_instance 미제공 + 모듈 없음/초기화 실패 → Normal 묵과
      (선택적 기능 미설치 환경 지원)
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from rules.safety_net import SafetyNet

        has_renal, has_hepatic = _detect_risk_flags(drugs)

        sn = sn_instance or SafetyNet()
        drug_names = [d.drug_name or d.edi_code for d in drugs]
        assessment = sn.assess(
            drugs=drug_names,
            patient_age=patient_age,
            concurrent_drug_count=len(drugs),
            has_renal_risk=has_renal,
            has_hepatic_risk=has_hepatic,
        )

        level = RiskLevel(assessment.risk_grade)
        reasons = list(assessment.triggered_rules)

        alerts: list[DDIAlert] = []
        for ddi in assessment.ddi_pairs:
            try:
                severity = Severity(ddi.severity)
            except ValueError:
                severity = Severity.UNKNOWN
            alerts.append(DDIAlert(
                drug_a=ddi.drug_a,
                drug_b=ddi.drug_b,
                severity=severity,
                description=ddi.description,
                source=ddi.source,
            ))

        return level, reasons, alerts

    except (ImportError, AttributeError):
        # 모듈 미설치 또는 SafetyNet 초기화 실패 (선택적 기능) → 묵과
        if sn_instance is not None:
            # 인스턴스가 있는데 AttributeError라면 런타임 버그 → 전파
            raise
        logger.warning("Safety Net 미설치 또는 초기화 실패 (Normal 반환)")
        return RiskLevel.NORMAL, [], []
    except Exception as e:
        if sn_instance is not None:
            # 초기화된 Safety Net이 충돌 → DDI 탐지 실패, 전파
            logger.error("Safety Net assess() 런타임 오류: %s", e)
            raise
        logger.warning("Safety Net 실행 오류 (Normal 반환): %s", e)
        return RiskLevel.NORMAL, [], []
```

- [ ] **Step 4: 테스트 실행 — GREEN 확인**

```bash
python3 -m pytest tests/test_serving/test_predictor.py::TestRunSafetyNet -v
```
Expected: `3 passed`

- [ ] **Step 5: 전체 serving 테스트 통과 확인 (기존 테스트 영향 없음)**

```bash
python3 -m pytest tests/test_serving/ -v
```
Expected: 기존 테스트 + 신규 3건 포함 모두 PASS

- [ ] **Step 6: 커밋**

```bash
git add serving/predictor.py tests/test_serving/test_predictor.py
git commit -m "fix: _run_safety_net sn_instance 있을 때 RuntimeError 전파 + 테스트 3건"
```

---

### Task 4: `HybridPredictor.reload_model` 단위 테스트

**Files:**
- Modify: `tests/test_serving/test_predictor.py`

- [ ] **Step 1: reload_model 테스트 추가**

```python
class TestHybridPredictorReloadModel:
    """HybridPredictor.reload_model — 핫스왑 단위 검증."""

    @pytest.fixture
    def predictor_no_model(self, tmp_path):
        """ML 모델 없는 빈 HybridPredictor (파일 경로 없이)."""
        pred = HybridPredictor.__new__(HybridPredictor)
        pred._start_time = 0.0
        pred._ml_lock = threading.RLock()
        pred._ml = MLModel()
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
```

- [ ] **Step 2: 테스트 실행 — GREEN 확인**

```bash
python3 -m pytest tests/test_serving/test_predictor.py::TestHybridPredictorReloadModel -v
```
Expected: `3 passed`

- [ ] **Step 3: 전체 테스트 통과 확인**

```bash
python3 -m pytest --tb=short -q
```
Expected: 기존 + 신규 합산 통과

- [ ] **Step 4: 커밋**

```bash
git add tests/test_serving/test_predictor.py
git commit -m "test: HybridPredictor.reload_model 단위 테스트 3건 (성공·실패·스레드)"
```
