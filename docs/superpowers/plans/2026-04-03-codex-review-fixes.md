# Codex Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Codex 코드 리뷰에서 발견된 Critical 2건·High 4건·Medium 2건·Low 2건 총 10개 이슈를 순서대로 수정한다.

**Architecture:** Rule+ML 하이브리드 예측 서비스(FastAPI) + Airflow 재훈련 DAG + 학습 파이프라인(XGBoost/LightGBM). 수정은 아티팩트 계약 통일 → DAG 시그니처 수정 → 피처 계약 → 런타임 최적화 → 보안 → 데이터 품질 → 드리프트 감지 → 버그 패치 순으로 진행한다.

**Tech Stack:** Python 3.11, FastAPI, XGBoost/LightGBM, Airflow, pickle/joblib, scikit-learn scaler/selector, Docker

---

## ⚠️ Gemini + Codex 교차 검토 반영 사항 (2026-04-03)

### Gemini 제언 (반영 완료)
- **G1** Task 3: scaler/selector 경로를 절대경로 대신 모델 파일 기준 상대경로로 저장 → `scaler_path_mode` + `_resolve_artifact_path()` 도입

### Codex 추가 발견 (계획 수정 필요)
| # | Task | 추가 이슈 | 수정 방향 |
|---|------|-----------|-----------|
| C1 | 3 | `relative_to()` 가 형제 디렉터리에서 `ValueError` → G1 수정 무효 | `os.path.relpath()` 로 대체 |
| C2 | 3 | `scaler_path_mode` 하나를 두 경로에 재사용 → 분리 오류 가능 | `scaler_path_mode`, `selector_path_mode` 분리 |
| C3 | 3 | `FeatureNormalizer/Selector.transform()` 이 ndarray가 아닌 DataFrame 기대 | 온라인 경로도 DataFrame 래핑 후 transform |
| C4 | 1·3 | `EnsembleTrainer.save()` 가 `model` 키 미저장 → 앙상블 로드 후 예측 불가 | 앙상블 단일 예측 객체 저장 또는 `.xgb.pkl`/`.lgb.pkl` 로드 통합 |
| C5 | 5 | FastAPI `model_path: str` 는 query param인데 DAG가 JSON body로 전송 → 422 | 서버를 Pydantic body 모델로 변경 또는 DAG를 querystring으로 변경 |
| C6 | 5 | `DDI_MODEL_DIR`(DAG) vs `MODEL_DIR`(serving) 환경변수명 불일치 | 하나로 통일 |
| C7 | 7 | `evaluator.py` 단일 클래스 방어 코드 계획에 누락 | `len(np.unique(y_true)) < 2` 방어 추가 |
| C8 | 3 | 모델 포맷 변경 시 구버전 모델과 혼재 기간 발생 | `artifact_version` 필드 + 구포맷 fallback |

### 권장 실행 순서 (Codex 확정)
```
Task 2 → Task 1 → Task 3 (재학습 포함) → Task 5 → Task 6 → Task 4/7/8/9 → Task 10
```
> Task 10(Dockerfile)은 모든 Task 검증 완료 후 마지막에 적용

---

## 파일 맵

| 파일 | 변경 내용 |
|------|-----------|
| `scripts/train/trainer.py` | `save()` — SHA-256 사이드카 생성, feature_names 저장 |
| `scripts/train/pipeline.py` | `TrainPipeline.run()` — scaler/selector 경로 모델 pkl에 기록 |
| `dags/ddi_train_dag.py` | 스테이징 파일명 수정, `run_training` 호출 수정, `/admin/reload` 헤더 추가, `.sha256` 사이드카 복사 |
| `serving/predictor.py` | `MLModel.load()` scaler/selector 로드, `RequestFeatureBuilder` 피처 정렬·스케일링 적용, SafetyNet/DuplicateDetector 싱글턴화, `all_reasons` 중복 제거 |
| `serving/main.py` | CORS_ORIGINS 기본값 수정 |
| `scripts/train/dataset.py` | `_split_dataset` 층화 분할 적용 |
| `scripts/train/evaluator.py` | 단일 클래스 입력 방어 |
| `monitoring/drift_detector.py` | PSI overflow bin 추가 |
| `Dockerfile` | 비권한 사용자 추가 |

---

## Task 1: [Critical] SHA-256 사이드카 자동 생성

**Files:**
- Modify: `scripts/train/trainer.py:73-85`
- Test: `tests/test_train/test_trainer.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_train/test_trainer.py 에 추가
import hashlib
from pathlib import Path
from unittest.mock import MagicMock
import pickle
import pytest

def test_save_generates_sha256_sidecar(tmp_path):
    """save() 호출 후 .sha256 사이드카가 생성되어야 한다."""
    from scripts.train.trainer import XGBoostTrainer
    trainer = XGBoostTrainer.__new__(XGBoostTrainer)
    trainer.model = MagicMock()
    trainer.params = {}
    trainer.feature_importances_ = None
    trainer.best_threshold_ = 0.5
    trainer._trained = True
    trainer.config = None

    path = tmp_path / "model.pkl"
    trainer.save(path)

    sha_path = path.with_suffix(".pkl.sha256")
    assert sha_path.exists(), ".sha256 사이드카 없음"

    content = path.read_bytes()
    expected = hashlib.sha256(content).hexdigest()
    actual = sha_path.read_text().strip().split()[0]
    assert actual == expected, "SHA-256 불일치"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /Volumes/model/claude/MODE_11_hana
python -m pytest tests/test_train/test_trainer.py::test_save_generates_sha256_sidecar -v
```
Expected: `FAILED` — `.sha256 사이드카 없음`

- [ ] **Step 3: trainer.py save() 수정**

`scripts/train/trainer.py` 의 `BaseTrainer.save()` (line 73) 를 다음으로 교체:

```python
def save(self, path: str | Path) -> Path:
    import hashlib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": self.model,
        "params": self.params,
        "feature_importances": self.feature_importances_,
        "best_threshold": self.best_threshold_,
        "trainer_class": self.__class__.__name__,
    }
    content = pickle.dumps(payload)
    path.write_bytes(content)
    # SHA-256 사이드카 생성 (serving/predictor.py _verify_hash 요구사항)
    sha256 = hashlib.sha256(content).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256}  {path.name}\n")
    logger.info("모델 저장: %s (sha256=%s…)", path, sha256[:16])
    return path
```

`EnsembleTrainer.save()` (line 255) 도 동일하게 수정 — 메타 pkl 저장 후 SHA-256 생성:

```python
def save(self, path: str | Path) -> Path:
    import hashlib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    self._xgb.save(path.with_suffix(".xgb.pkl"))
    self._lgb.save(path.with_suffix(".lgb.pkl"))
    payload = {
        "weights": self.weights,
        "best_threshold": self.best_threshold_,
        "feature_importances": self.feature_importances_,
    }
    content = pickle.dumps(payload)
    path.write_bytes(content)
    sha256 = hashlib.sha256(content).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256}  {path.name}\n")
    logger.info("앙상블 모델 저장: %s (sha256=%s…)", path, sha256[:16])
    return path
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_train/test_trainer.py::test_save_generates_sha256_sidecar -v
```
Expected: `PASSED`

- [ ] **Step 5: DAG _deploy_model .sha256 복사 추가**

`dags/ddi_train_dag.py` `_deploy_model` (line 152):

```python
shutil.copy2(model_path, prod_path)
# .sha256 사이드카도 함께 복사
sha_src = model_path + ".sha256"
sha_dst = prod_path + ".sha256"
if os.path.exists(sha_src):
    shutil.copy2(sha_src, sha_dst)
else:
    import logging
    logging.warning(".sha256 사이드카 없음, 서빙 로드 실패 가능: %s", sha_src)
```

- [ ] **Step 6: 커밋**

```bash
git add scripts/train/trainer.py dags/ddi_train_dag.py tests/test_train/test_trainer.py
git commit -m "fix: trainer.save() SHA-256 사이드카 생성 + DAG 배포 시 복사"
```

---

## Task 2: [Critical] DAG run_training 시그니처 수정

**Files:**
- Modify: `dags/ddi_train_dag.py:54-82` (`_load_features`), `dags/ddi_train_dag.py:89-118` (`_run_training`)
- Test: `tests/test_dags/test_ddi_train_dag.py`

`run_training(partition: str, ...)` 인데 DAG이 `run_training(df, config)` 로 호출 중.
스테이징 파일을 `ml_features_staging.parquet` 으로 저장하고 `run_training(partition="staging", ...)` 으로 호출한다.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_dags/test_ddi_train_dag.py
import importlib
import sys

def test_run_training_receives_partition_str(monkeypatch):
    """_run_training 이 run_training(partition=str, ...) 형태로 호출해야 한다."""
    captured = {}
    def fake_run_training(partition, **kwargs):
        captured["partition"] = partition
        captured["kwargs"] = kwargs
        from scripts.train.pipeline import TrainResult
        from scripts.train.evaluator import EvalResult
        r = TrainResult(partition=partition, model_type="xgboost")
        r.model_path = "/tmp/model.pkl"
        r.eval_results = {"val": EvalResult("val", recall=0.95, auc_roc=0.90)}
        r.passed = True
        return r

    monkeypatch.setattr("scripts.train.pipeline.run_training", fake_run_training)

    # XCom mock
    class FakeTI:
        def xcom_pull(self, key, task_ids):
            if key == "staging_path":
                return "/tmp/ml_features_staging.parquet"
        def xcom_push(self, key, value):
            pass

    import pandas as pd, tempfile, os
    df = pd.DataFrame({"patient_id": [1], "is_high_risk": [0]})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp = f.name
    df.to_parquet(tmp)

    # 경로 조정
    class FakeTI2(FakeTI):
        def xcom_pull(self, key, task_ids):
            if key == "staging_path":
                return tmp

    from dags.ddi_train_dag import _run_training
    _run_training(ti=FakeTI2())
    os.unlink(tmp)

    assert isinstance(captured.get("partition"), str), (
        f"run_training 의 첫 인자가 str이어야 함, 실제: {type(captured.get('partition'))}"
    )
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_dags/test_ddi_train_dag.py::test_run_training_receives_partition_str -v
```
Expected: `FAILED`

- [ ] **Step 3: _load_features 스테이징 파일명 수정**

`dags/ddi_train_dag.py` `_load_features` (line 80):

```python
# 변경 전
staging_path = f"{FEATURES_DIR}/train_staging.parquet"
combined.to_parquet(staging_path, index=False)

# 변경 후
staging_path = f"{FEATURES_DIR}/ml_features_staging.parquet"
combined.to_parquet(staging_path, index=False)
```

- [ ] **Step 4: _run_training 호출 수정**

`dags/ddi_train_dag.py` `_run_training` (line 89-118):

```python
def _run_training(**context) -> None:
    """Optuna 하이퍼파라미터 튜닝 + 모델 훈련."""
    import sys
    sys.path.insert(0, "/app")
    from scripts.train.pipeline import run_training

    staging_path = context["ti"].xcom_pull(key="staging_path", task_ids="load_features")
    # staging_path 는 {FEATURES_DIR}/ml_features_staging.parquet
    # run_training(partition="staging", feature_base=FEATURES_DIR, ...) 로 호출
    result = run_training(
        partition="staging",
        model_type=MODEL_TYPE,
        feature_base=FEATURES_DIR,
        model_dir=MODEL_DIR,
        use_optuna=OPTUNA_TRIALS > 0,
        recall_threshold=RECALL_THRESHOLD,
        auc_threshold=AUC_THRESHOLD,
        optuna_trials=OPTUNA_TRIALS,
    )

    context["ti"].xcom_push(key="val_recall", value=result.val_recall)
    context["ti"].xcom_push(key="val_auc", value=result.val_auc)
    context["ti"].xcom_push(key="model_path", value=result.model_path)

    import logging
    logging.info(
        "훈련 완료 — val_recall=%.4f, val_auc=%.4f, model=%s",
        result.val_recall, result.val_auc, result.model_path,
    )
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
python -m pytest tests/test_dags/test_ddi_train_dag.py::test_run_training_receives_partition_str -v
```
Expected: `PASSED`

- [ ] **Step 6: 커밋**

```bash
git add dags/ddi_train_dag.py tests/test_dags/test_ddi_train_dag.py
git commit -m "fix: DAG run_training 시그니처 불일치 수정 — partition str 전달"
```

---

## Task 3: [High] 온라인-오프라인 피처 계약 통일

**Files:**
- Modify: `scripts/train/pipeline.py:136-138` (Step 6)
- Modify: `serving/predictor.py:160-208` (`MLModel`)
- Modify: `serving/predictor.py:240-337` (`RequestFeatureBuilder`)
- Test: `tests/test_serving/test_predictor_feature_contract.py`

학습 시 `scaler.pkl`, `selector.pkl` 을 저장하지만 서빙에서 로드하지 않는 문제.
모델 pkl 에 `feature_names`, `scaler_path`, `selector_path` 를 포함시키고, 서빙에서 동일 전처리 적용.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_serving/test_predictor_feature_contract.py
import pickle, hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

def _write_model_pkl(path: Path, feature_names, scaler_path=None, selector_path=None):
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    payload = {
        "model": mock_model,
        "params": {},
        "feature_importances": None,
        "best_threshold": 0.5,
        "trainer_class": "XGBoostTrainer",
        "feature_names": feature_names,
        "scaler_path": scaler_path,
        "selector_path": selector_path,
    }
    content = pickle.dumps(payload)
    path.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    path.with_suffix(".pkl.sha256").write_text(f"{sha}  {path.name}\n")


def test_mlmodel_loads_feature_names(tmp_path):
    feature_names = ["drug_count", "age", "ddi_major"]
    model_path = tmp_path / "model.pkl"
    _write_model_pkl(model_path, feature_names)

    from serving.predictor import MLModel
    ml = MLModel()
    ok = ml.load(model_path)
    assert ok
    assert ml._feature_names == feature_names, (
        f"feature_names 미로드: {ml._feature_names}"
    )
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_serving/test_predictor_feature_contract.py::test_mlmodel_loads_feature_names -v
```
Expected: `FAILED` — feature_names 미로드

- [ ] **Step 3: pipeline.py Step 6 수정 — feature_names 모델에 포함**

`scripts/train/pipeline.py` 의 Step 6 (line 136):

> **[G1 반영 - Gemini 제언]** scaler/selector 경로를 절대경로 대신 모델 파일 기준 상대경로로 저장.
> 학습 환경(`/app/data/features/scaler.pkl`)과 서빙 환경 경로가 달라도 동작하도록 보장.

```python
# ── Step 6: 모델 저장 ─────────────────────────────────────────────
model_path = self.model_dir / f"ddi_model_{partition}.pkl"
feature_base = Path(self.config.feature_base).resolve()
scaler_abs   = feature_base / "scaler.pkl"
selector_abs = feature_base / "selector.pkl"

# 상대경로 계산: 모델 파일 디렉토리 기준 (없으면 절대경로 폴백)
try:
    scaler_rel   = str(scaler_abs.relative_to(model_path.parent.resolve()))
    selector_rel = str(selector_abs.relative_to(model_path.parent.resolve()))
    _path_mode = "relative"
except ValueError:
    # 공통 상위 경로가 없을 때 (다른 드라이브 등) 절대경로 사용
    scaler_rel   = str(scaler_abs)
    selector_rel = str(selector_abs)
    _path_mode = "absolute"

trainer._extra_meta = {
    "feature_names": list(dataset.feature_names),
    "scaler_path": scaler_rel,
    "selector_path": selector_rel,
    "scaler_path_mode": _path_mode,   # 서빙에서 해석 방식 결정
}
trainer.save(model_path)
result.model_path = str(model_path)
tracker.log_artifact(model_path, "model")
```

- [ ] **Step 4: trainer.py save() — _extra_meta 포함**

`scripts/train/trainer.py` `BaseTrainer.save()` 의 payload dict 에 extra_meta 추가:

```python
payload = {
    "model": self.model,
    "params": self.params,
    "feature_importances": self.feature_importances_,
    "best_threshold": self.best_threshold_,
    "trainer_class": self.__class__.__name__,
    **getattr(self, "_extra_meta", {}),
}
```

- [ ] **Step 5: MLModel.load() — feature_names, scaler, selector 로드**

`serving/predictor.py` `MLModel` 클래스의 `__init__` 과 `load()` 수정:

```python
class MLModel:
    def __init__(self):
        self._model = None
        self._threshold: float = 0.5
        self._feature_names: list[str] = []
        self._partition: Optional[str] = None
        self._model_type: str = "none"
        self._scaler = None   # 추가
        self._selector = None # 추가

    # ... _verify_hash 그대로 ...

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        try:
            content = path.read_bytes()
            if not self._verify_hash(path, content):
                return False
            state = pickle.loads(content)
            if not isinstance(state, dict):
                logger.error("모델 파일 형식 오류: dict가 아님 (%s)", type(state))
                return False
            self._model = state.get("model")
            self._threshold = state.get("best_threshold", 0.5)
            self._model_type = state.get("trainer_class", "unknown")
            self._feature_names = state.get("feature_names", [])

            # scaler 로드 (G1: 상대경로 → 절대경로 변환 후 로드)
            def _resolve_artifact_path(stored: str, mode: str, model_dir: Path) -> Path:
                """모델 저장 시 경로 모드에 따라 실제 경로 반환."""
                if mode == "relative":
                    return (model_dir / stored).resolve()
                return Path(stored)

            path_mode = state.get("scaler_path_mode", "absolute")

            scaler_stored = state.get("scaler_path")
            if scaler_stored:
                scaler_resolved = _resolve_artifact_path(scaler_stored, path_mode, path.parent)
                if scaler_resolved.exists():
                    import pickle as _pk
                    with open(scaler_resolved, "rb") as f:
                        self._scaler = _pk.load(f)
                    logger.info("Scaler 로드: %s", scaler_resolved)
                else:
                    logger.warning("Scaler 없음 — 정규화 미적용: %s", scaler_resolved)

            selector_stored = state.get("selector_path")
            if selector_stored:
                selector_resolved = _resolve_artifact_path(selector_stored, path_mode, path.parent)
                if selector_resolved.exists():
                    import pickle as _pk
                    with open(selector_resolved, "rb") as f:
                        self._selector = _pk.load(f)
                    logger.info("Selector 로드: %s", selector_resolved)
                else:
                    logger.warning("Selector 없음 — 전체 피처 사용: %s", selector_resolved)

            logger.info("ML 모델 로드: %s (threshold=%.3f, features=%d)",
                        path, self._threshold, len(self._feature_names))
            return True
        except Exception as e:
            logger.warning("ML 모델 로드 실패: %s", e)
            return False
```

- [ ] **Step 6: RequestFeatureBuilder.build() — 피처 정렬 + 스케일 적용**

`serving/predictor.py` `RequestFeatureBuilder.build()` 의 마지막 반환 부분 수정:

```python
# 기존
vec = np.array(list(feat.values()), dtype=float)
return vec, feat

# 변경 후
def build(self, req: PredictRequest,
          feature_names: Optional[list[str]] = None,
          scaler=None, selector=None) -> tuple[np.ndarray, dict]:
    # ... (기존 feat 계산 코드 동일) ...

    # feature_names 기준으로 벡터 정렬 (불일치 피처는 0.0)
    if feature_names:
        vec = np.array([feat.get(name, 0.0) for name in feature_names], dtype=float)
    else:
        vec = np.array(list(feat.values()), dtype=float)

    # scaler 적용
    if scaler is not None:
        try:
            vec = scaler.transform(vec.reshape(1, -1)).flatten()
        except Exception as e:
            logger.warning("Scaler 적용 실패 (원본 사용): %s", e)

    # selector 적용
    if selector is not None:
        try:
            vec = selector.transform(vec.reshape(1, -1)).flatten()
        except Exception as e:
            logger.warning("Selector 적용 실패 (원본 사용): %s", e)

    return vec, feat
```

`HybridPredictor.predict()` 에서 `self._builder.build(req)` 호출을 다음으로 수정:

```python
feat_vec, _ = self._builder.build(
    req,
    feature_names=ml_snapshot._feature_names or None,
    scaler=ml_snapshot._scaler,
    selector=ml_snapshot._selector,
)
```

- [ ] **Step 7: 테스트 통과 확인**

```bash
python -m pytest tests/test_serving/test_predictor_feature_contract.py -v
```
Expected: `PASSED`

- [ ] **Step 8: 커밋**

```bash
git add scripts/train/trainer.py scripts/train/pipeline.py serving/predictor.py
git add tests/test_serving/test_predictor_feature_contract.py
git commit -m "fix: 온라인-오프라인 피처 계약 통일 — feature_names/scaler/selector 모델에 포함"
```

---

## Task 4: [High] SafetyNet/DuplicateDetector 싱글턴화

**Files:**
- Modify: `serving/predictor.py:84-151` (`_run_safety_net`, `_run_duplicate_detector`, `HybridPredictor.__init__`)
- Test: `tests/test_serving/test_predictor_singleton.py`

요청마다 `SafetyNet()`, `DuplicateDetector()` 를 생성해 YAML/parquet 를 재로딩하는 문제.
`HybridPredictor.__init__()` 에서 1회만 초기화한다.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_serving/test_predictor_singleton.py
from unittest.mock import patch, MagicMock, call
import pytest

def test_safety_net_instantiated_once_not_per_request():
    """SafetyNet() 이 요청마다 생성되지 않고 HybridPredictor 초기화 시 1회만 생성되어야 한다."""
    with patch("rules.safety_net.SafetyNet") as mock_sn_cls, \
         patch("rules.duplicate_detector.DuplicateDetector") as mock_dd_cls, \
         patch("pathlib.Path.exists", return_value=False):
        from serving.predictor import HybridPredictor
        pred = HybridPredictor()

        # 초기화 시 각 1회
        assert mock_sn_cls.call_count == 1
        assert mock_dd_cls.call_count == 1

        mock_sn = mock_sn_cls.return_value
        mock_sn.assess.return_value = MagicMock(
            risk_grade="Normal", triggered_rules=[], ddi_pairs=[]
        )
        mock_dd = mock_dd_cls.return_value
        mock_dd.detect.return_value = MagicMock(
            duplicate_level1_count=0, duplicate_level2_count=0
        )

        from serving.schemas import PredictRequest, DrugItem
        req = PredictRequest(
            patient_id="p1",
            drugs=[DrugItem(edi_code="123456", drug_name="aspirin", total_days=30)],
        )
        pred.predict(req)
        pred.predict(req)

        # 2번 요청해도 생성자 호출 횟수 변화 없음
        assert mock_sn_cls.call_count == 1
        assert mock_dd_cls.call_count == 1
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_serving/test_predictor_singleton.py::test_safety_net_instantiated_once_not_per_request -v
```
Expected: `FAILED`

- [ ] **Step 3: HybridPredictor.__init__ 에 SafetyNet/DuplicateDetector 추가**

`serving/predictor.py` `HybridPredictor.__init__()` 의 끝 부분 (line 413 근처) 에 추가:

```python
# Safety Net 싱글턴 (요청당 재생성 방지)
self._safety_net = None
self._dup_detector = None
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rules.safety_net import SafetyNet
    self._safety_net = SafetyNet()
    logger.info("SafetyNet 초기화 완료")
except Exception as e:
    logger.warning("SafetyNet 초기화 실패 (요청 시 재시도): %s", e)

try:
    from rules.duplicate_detector import DuplicateDetector
    self._dup_detector = DuplicateDetector()
    logger.info("DuplicateDetector 초기화 완료")
except Exception as e:
    logger.warning("DuplicateDetector 초기화 실패 (요청 시 재시도): %s", e)
```

- [ ] **Step 4: _run_safety_net, _run_duplicate_detector 를 메서드로 이동**

`serving/predictor.py` 에서 모듈 레벨 함수 `_run_safety_net`, `_run_duplicate_detector` 를 `HybridPredictor` 의 private 메서드로 변환.

`_run_safety_net` 함수 (line 84) 를 `HybridPredictor._run_safety_net` 메서드로:

```python
def _run_safety_net(
    self,
    drugs: list[DrugItem],
    patient_age: Optional[int] = None,
) -> tuple[RiskLevel, list[str], list[DDIAlert]]:
    try:
        sn = self._safety_net
        if sn is None:
            # 초기화 실패 시 재시도
            from rules.safety_net import SafetyNet
            sn = SafetyNet()

        has_renal, has_hepatic = _detect_risk_flags(drugs)
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
                drug_a=ddi.drug_a, drug_b=ddi.drug_b,
                severity=severity, description=ddi.description, source=ddi.source,
            ))
        return level, reasons, alerts
    except Exception as e:
        logger.warning("Safety Net 실행 오류 (Normal 반환): %s", e)
        return RiskLevel.NORMAL, [], []
```

`_run_duplicate_detector` 함수 (line 133) 를 `HybridPredictor._run_duplicate_detector` 메서드로:

```python
def _run_duplicate_detector(self, drugs: list[DrugItem]) -> tuple[int, list[str]]:
    try:
        dd = self._dup_detector
        if dd is None:
            from rules.duplicate_detector import DuplicateDetector
            dd = DuplicateDetector()

        drug_input = _drugs_to_dup_input(drugs)
        result = dd.detect(drug_input)
        dup_count = result.duplicate_level1_count + result.duplicate_level2_count
        reasons = []
        if result.duplicate_level1_count:
            reasons.append(f"동일성분중복 {result.duplicate_level1_count}건")
        if result.duplicate_level2_count:
            reasons.append(f"동일약리군중복 {result.duplicate_level2_count}건")
        return dup_count, reasons
    except Exception as e:
        logger.warning("DuplicateDetector 오류: %s", e)
        return 0, []
```

`HybridPredictor.predict()` 에서 호출부를 `self._run_safety_net(...)`, `self._run_duplicate_detector(...)` 로 수정.

- [ ] **Step 5: 테스트 통과 확인**

```bash
python -m pytest tests/test_serving/test_predictor_singleton.py -v
```
Expected: `PASSED`

- [ ] **Step 6: 커밋**

```bash
git add serving/predictor.py tests/test_serving/test_predictor_singleton.py
git commit -m "fix: SafetyNet/DuplicateDetector 싱글턴화 — 요청당 YAML/parquet 재로딩 제거"
```

---

## Task 5: [High] DAG /admin/reload X-Admin-Key 헤더 추가

**Files:**
- Modify: `dags/ddi_train_dag.py:154-167` (`_deploy_model`)
- Test: `tests/test_dags/test_ddi_train_dag.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_dags/test_ddi_train_dag.py 에 추가
from unittest.mock import patch, MagicMock

def test_deploy_sends_admin_key_header(monkeypatch, tmp_path):
    """_deploy_model 이 /admin/reload 에 X-Admin-Key 헤더를 전송해야 한다."""
    import os
    monkeypatch.setenv("DDI_ADMIN_API_KEY", "test-secret")
    monkeypatch.setenv("DDI_SERVING_URL", "http://localhost:8000")

    # 가짜 모델 파일 생성
    model_path = str(tmp_path / "model.pkl")
    (tmp_path / "model.pkl").write_bytes(b"fake")
    (tmp_path / "model.pkl.sha256").write_text("abc123  model.pkl\n")

    captured_headers = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured_headers.update(headers or {})
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"status": "ok"}
        return resp

    class FakeTI:
        def xcom_pull(self, key, task_ids):
            return model_path

    with patch("requests.post", side_effect=fake_post), \
         patch("shutil.copy2"), \
         patch("os.path.exists", return_value=True):
        from dags.ddi_train_dag import _deploy_model
        _deploy_model(ti=FakeTI())

    assert "X-Admin-Key" in captured_headers, "X-Admin-Key 헤더 없음"
    assert captured_headers["X-Admin-Key"] == "test-secret"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_dags/test_ddi_train_dag.py::test_deploy_sends_admin_key_header -v
```
Expected: `FAILED`

- [ ] **Step 3: _deploy_model requests.post 헤더 추가**

`dags/ddi_train_dag.py` `_deploy_model` (line 156):

```python
admin_key = os.environ.get("DDI_ADMIN_API_KEY", "")
resp = requests.post(
    f"{serving_url}/admin/reload",
    json={"model_path": prod_path},
    headers={"X-Admin-Key": admin_key},
    timeout=30,
)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_dags/test_ddi_train_dag.py::test_deploy_sends_admin_key_header -v
```
Expected: `PASSED`

- [ ] **Step 5: 커밋**

```bash
git add dags/ddi_train_dag.py tests/test_dags/test_ddi_train_dag.py
git commit -m "fix: DAG /admin/reload 에 X-Admin-Key 인증 헤더 추가"
```

---

## Task 6: [High] CORS_ORIGINS 기본값 수정

**Files:**
- Modify: `serving/main.py:85-86`
- Test: `tests/test_serving/test_main_cors.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_serving/test_main_cors.py
import importlib
import os
import sys

def test_cors_default_is_not_wildcard(monkeypatch):
    """CORS_ORIGINS 미설정 시 기본값이 '*' 이면 안 된다."""
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    # 모듈 재로드로 환경변수 반영
    if "serving.main" in sys.modules:
        del sys.modules["serving.main"]

    import serving.main as main_mod
    # CORSMiddleware 에 등록된 origins 확인
    cors_mw = None
    for mw in main_mod.app.user_middleware:
        if "CORSMiddleware" in str(mw):
            cors_mw = mw
            break

    # CORS 미들웨어의 allow_origins 이 "*" 를 포함하면 안 됨
    # FastAPI/Starlette 내부 구현: app.middleware_stack 이 아닌
    # app.user_middleware 에서 kwargs 확인
    # 간단히: 환경변수 없이 모듈 임포트 후 _cors_origins 가 [] 또는 빈 리스트여야 함
    assert main_mod._cors_origins_env != "*", (
        "CORS_ORIGINS 미설정 시 기본값이 '*' 임 — 보안 취약점"
    )
    assert "*" not in main_mod._cors_origins, "cors_origins 에 와일드카드 포함"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_serving/test_main_cors.py::test_cors_default_is_not_wildcard -v
```
Expected: `FAILED`

- [ ] **Step 3: serving/main.py 수정**

`serving/main.py` (line 83-86):

```python
# 변경 전
# CORS: 환경변수 CORS_ORIGINS로 허용 오리진 제한 가능 (쉼표 구분)
# 미설정 시 폐쇄망 기본값 "*" 적용
_cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]

# 변경 후
# CORS: CORS_ORIGINS 환경변수로 허용 오리진 지정 (쉼표 구분)
# 미설정 시 빈 목록 → 모든 외부 오리진 차단
_cors_origins_env = os.environ.get("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
if not _cors_origins:
    logger.info("CORS_ORIGINS 미설정 — 외부 오리진 차단 (폐쇄망 기본값)")
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_serving/test_main_cors.py::test_cors_default_is_not_wildcard -v
```
Expected: `PASSED`

- [ ] **Step 5: 커밋**

```bash
git add serving/main.py tests/test_serving/test_main_cors.py
git commit -m "fix: CORS_ORIGINS 기본값 '*' → '' (외부 오리진 차단)"
```

---

## Task 7: [Medium] 층화 분할 + 단일 클래스 평가 방어

**Files:**
- Modify: `scripts/train/dataset.py:160-220` (`_split_dataset`)
- Modify: `scripts/train/evaluator.py` (단일 클래스 방어)
- Test: `tests/test_train/test_dataset.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_train/test_dataset.py 에 추가
import numpy as np
import pandas as pd

def _make_imbalanced_df(n=200, red_ratio=0.05):
    rng = np.random.default_rng(42)
    n_red = int(n * red_ratio)
    n_non = n - n_red
    risk = ["Red"] * n_red + ["Green"] * n_non
    return pd.DataFrame({
        "patient_id": range(n),
        "drug_count": rng.integers(5, 15, n).astype(float),
        "age": rng.integers(40, 80, n).astype(float),
        "risk_level": risk,
    })


def test_split_is_stratified():
    """층화 분할 시 각 split 에 Red 클래스가 반드시 존재해야 한다."""
    df = _make_imbalanced_df(n=400, red_ratio=0.05)
    from scripts.train.dataset import _split_dataset
    ds = _split_dataset(df, val_ratio=0.15, test_ratio=0.15, random_state=42)
    assert ds.y_val.sum() > 0, "val split 에 Red(1) 없음 — 층화 미적용"
    assert ds.y_test.sum() > 0, "test split 에 Red(1) 없음 — 층화 미적용"
```

- [ ] **Step 2: 테스트 실패 확인 (때로는 랜덤으로 통과할 수 있으므로 10회 반복)**

```bash
python -m pytest tests/test_train/test_dataset.py::test_split_is_stratified -v --count=5
```
Expected: 5회 중 일부 FAILED (불균형 데이터에서 층화 없으면 불안정)

- [ ] **Step 3: _split_dataset 층화 분할 적용**

`scripts/train/dataset.py` `_split_dataset` (line 160):

```python
def _split_dataset(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> TrainDataset:
    rng = np.random.default_rng(random_state)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    if "is_high_risk" not in df.columns:
        if "risk_level" in df.columns:
            df = df.copy()
            df["is_high_risk"] = (df["risk_level"] == "Red").astype(int)
        else:
            raise ValueError("'is_high_risk' 또는 'risk_level' 컬럼 필요")
    else:
        df = df.copy()

    if "risk_level" in df.columns:
        df["risk_level_encoded"] = df["risk_level"].map(RISK_ORDER).fillna(0).astype(int)
    else:
        df["risk_level_encoded"] = df["is_high_risk"] * 3

    # 층화 분할: Red(1) vs Non-Red(0) 비율 유지
    y = df["is_high_risk"].values
    classes = np.unique(y)

    if len(classes) < 2:
        # 단일 클래스 — 층화 불가, 단순 랜덤
        logger.warning("단일 클래스 데이터 — 층화 분할 불가, 랜덤 분할 사용")
        idx = rng.permutation(len(df))
        df = df.iloc[idx].reset_index(drop=True)
        n = len(df)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        train_df = df.iloc[:n - n_val - n_test]
        val_df   = df.iloc[n - n_val - n_test:n - n_test]
        test_df  = df.iloc[n - n_test:]
    else:
        # 클래스별 인덱스 분리 후 각각 분할 → 합산 → 셔플
        split_dfs = {"train": [], "val": [], "test": []}
        for cls in classes:
            cls_df = df[y == cls].copy()
            cls_idx = rng.permutation(len(cls_df))
            cls_df = cls_df.iloc[cls_idx]
            n_c = len(cls_df)
            n_c_test = max(1, int(n_c * test_ratio))
            n_c_val  = max(1, int(n_c * val_ratio))
            split_dfs["train"].append(cls_df.iloc[:n_c - n_c_val - n_c_test])
            split_dfs["val"].append(cls_df.iloc[n_c - n_c_val - n_c_test:n_c - n_c_test])
            split_dfs["test"].append(cls_df.iloc[n_c - n_c_test:])

        import pandas as pd as _pd
        train_df = _pd.concat(split_dfs["train"]).sample(frac=1, random_state=int(random_state)).reset_index(drop=True)
        val_df   = _pd.concat(split_dfs["val"]).sample(frac=1, random_state=int(random_state)).reset_index(drop=True)
        test_df  = _pd.concat(split_dfs["test"]).sample(frac=1, random_state=int(random_state)).reset_index(drop=True)

    def _arrays(sub):
        X = sub[feature_cols].astype(float).values
        y_bin = sub["is_high_risk"].values.astype(int)
        y_multi = sub["risk_level_encoded"].values.astype(int)
        meta_cols = [c for c in ["patient_id", "risk_level", "window_start", "window_end"]
                     if c in sub.columns]
        meta = sub[meta_cols].reset_index(drop=True)
        return X, y_bin, y_multi, meta

    X_tr, y_tr, ym_tr, m_tr = _arrays(train_df)
    X_va, y_va, ym_va, m_va = _arrays(val_df)
    X_te, y_te, ym_te, m_te = _arrays(test_df)

    ds = TrainDataset(
        X_train=X_tr, X_val=X_va, X_test=X_te,
        y_train=y_tr, y_val=y_va, y_test=y_te,
        y_multi_train=ym_tr, y_multi_val=ym_va, y_multi_test=ym_te,
        feature_names=feature_cols,
        meta_train=m_tr, meta_val=m_va, meta_test=m_te,
    )
    ds.print_summary()
    return ds
```

> 참고: `import pandas as pd as _pd` 는 문법 오류 — 아래 Step 에서 수정.

- [ ] **Step 4: 코드 정정 (import 중복 제거)**

위 코드에서 `import pandas as pd as _pd` 를 제거하고, 파일 상단에 이미 `import pandas as pd` 가 있으므로 해당 라인을:

```python
train_df = pd.concat(split_dfs["train"]).sample(frac=1, random_state=int(random_state)).reset_index(drop=True)
val_df   = pd.concat(split_dfs["val"]).sample(frac=1, random_state=int(random_state)).reset_index(drop=True)
test_df  = pd.concat(split_dfs["test"]).sample(frac=1, random_state=int(random_state)).reset_index(drop=True)
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
python -m pytest tests/test_train/test_dataset.py::test_split_is_stratified -v
```
Expected: `PASSED`

- [ ] **Step 6: 커밋**

```bash
git add scripts/train/dataset.py tests/test_train/test_dataset.py
git commit -m "fix: _split_dataset 층화 분할 적용 — Red 클래스 각 split 보장"
```

---

## Task 8: [Medium] PSI overflow bin 추가

**Files:**
- Modify: `monitoring/drift_detector.py:104-133` (`compute_psi_continuous`)
- Test: `tests/test_monitoring/test_drift_detector.py`

기준 범위 밖의 현재 데이터 값이 histogram 에서 누락되어 PSI 과소평가.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_monitoring/test_drift_detector.py 에 추가
import numpy as np

def test_psi_counts_overflow_values():
    """기준 범위를 벗어난 현재 데이터도 PSI 계산에 포함되어야 한다."""
    from monitoring.drift_detector import compute_psi_continuous

    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20, dtype=float)
    # 현재 데이터에 기준 범위(1~5) 를 초과하는 값 다수 포함
    current = np.array([1.0, 2.0, 100.0, 200.0, 300.0] * 20, dtype=float)

    psi_no_overflow, *_ = compute_psi_continuous(reference, current, n_bins=5)
    # overflow 무시 시 PSI 가 낮게 계산됨 — 올바르게 계산하면 높아야 함
    assert psi_no_overflow > 0.25, (
        f"overflow 값이 무시되어 PSI 과소평가: {psi_no_overflow:.4f} (기대: >0.25)"
    )
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_monitoring/test_drift_detector.py::test_psi_counts_overflow_values -v
```
Expected: `FAILED`

- [ ] **Step 3: compute_psi_continuous overflow bin 추가**

`monitoring/drift_detector.py` `compute_psi_continuous` (line 104):

```python
def compute_psi_continuous(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = N_BINS,
) -> tuple[float, list[float], list[float], list[float]]:
    """연속형 피처의 PSI 계산. overflow bin 포함."""
    # 기준 데이터로 내부 bin edges 결정 (percentile)
    inner_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    inner_edges = np.unique(inner_edges)

    if len(inner_edges) < 2:
        return 0.0, [], [], list(inner_edges)

    # overflow bin: -inf ~ min, max ~ +inf 포함
    bin_edges = np.concatenate([[-np.inf], inner_edges[1:-1], [np.inf]])

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)

    ref_dist = (ref_counts / max(ref_counts.sum(), 1)).tolist()
    cur_dist = (cur_counts / max(cur_counts.sum(), 1)).tolist()

    psi = 0.0
    for r, c in zip(ref_dist, cur_dist):
        r = max(r, EPSILON)
        c = max(c, EPSILON)
        psi += (c - r) * np.log(c / r)

    # 직렬화 가능한 bin_edges 반환 (inf → float 표현)
    return float(psi), ref_dist, cur_dist, bin_edges.tolist()
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_monitoring/test_drift_detector.py::test_psi_counts_overflow_values -v
```
Expected: `PASSED`

- [ ] **Step 5: 커밋**

```bash
git add monitoring/drift_detector.py tests/test_monitoring/test_drift_detector.py
git commit -m "fix: PSI overflow bin 추가 — 기준 범위 밖 값 누락 방지"
```

---

## Task 9: [Low] all_reasons 중복 제거

**Files:**
- Modify: `serving/predictor.py:444,471` (`HybridPredictor.predict`)
- Test: `tests/test_serving/test_predictor_reasons.py`

`rule_reasons.extend(dup_reasons)` 후 `all_reasons = rule_reasons + dup_reasons` 로 `dup_reasons` 가 두 번 포함되는 버그.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_serving/test_predictor_reasons.py
from unittest.mock import patch, MagicMock

def test_dup_reasons_not_duplicated(monkeypatch):
    """중복약물 사유가 risk_reasons 에 두 번 포함되면 안 된다."""
    with patch("pathlib.Path.exists", return_value=False):
        from serving.predictor import HybridPredictor

    pred = HybridPredictor.__new__(HybridPredictor)
    pred._ml_lock = __import__("threading").RLock()
    pred._ml = MagicMock(loaded=False)
    pred._ddi_matrix = None
    pred._safety_net = None
    pred._dup_detector = None
    pred._builder = MagicMock()
    pred._start_time = __import__("time").time()

    # Safety Net: Normal, 중복약물: 1건
    with patch.object(pred, "_run_safety_net", return_value=(
        __import__("serving.schemas", fromlist=["RiskLevel"]).RiskLevel.NORMAL,
        [],
        [],
    )), patch.object(pred, "_run_duplicate_detector", return_value=(
        1, ["동일성분중복 1건"]
    )):
        from serving.schemas import PredictRequest, DrugItem
        req = PredictRequest(
            patient_id="p1",
            drugs=[DrugItem(edi_code="A001", drug_name="aspirin", total_days=30),
                   DrugItem(edi_code="A001", drug_name="aspirin", total_days=30)],
        )
        resp = pred.predict(req)

    # "동일성분중복 1건" 이 1번만 있어야 함
    count = resp.risk_reasons.count("동일성분중복 1건")
    assert count == 1, f"중복 사유가 {count}번 포함됨 (기대: 1번)"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_serving/test_predictor_reasons.py::test_dup_reasons_not_duplicated -v
```
Expected: `FAILED`

- [ ] **Step 3: predict() all_reasons 수정**

`serving/predictor.py` `HybridPredictor.predict()` 에서:

```python
# 변경 전 (line 444~471 근처)
if dup_count >= 1 and rule_level == RiskLevel.NORMAL:
    rule_level = RiskLevel.YELLOW
    rule_reasons.extend(dup_reasons)     # dup_reasons → rule_reasons 에 추가

...
all_reasons = rule_reasons + dup_reasons  # dup_reasons 재추가 → 중복!

# 변경 후
if dup_count >= 1 and rule_level == RiskLevel.NORMAL:
    rule_level = RiskLevel.YELLOW
    rule_reasons.extend(dup_reasons)

...
all_reasons = list(rule_reasons)          # dup_reasons 는 이미 rule_reasons 에 포함
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_serving/test_predictor_reasons.py::test_dup_reasons_not_duplicated -v
```
Expected: `PASSED`

- [ ] **Step 5: 커밋**

```bash
git add serving/predictor.py tests/test_serving/test_predictor_reasons.py
git commit -m "fix: all_reasons 에서 dup_reasons 중복 제거"
```

---

## Task 10: [Low] Dockerfile 비권한 사용자 설정

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Dockerfile 수정**

`Dockerfile` 의 `RUN mkdir -p /app/data /app/models` (line 42) 다음에 추가:

```dockerfile
# ── 비권한 사용자 (보안) ─────────────────────────────────────────────────────
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app/data /app/models
USER appuser
```

- [ ] **Step 2: 빌드 확인**

```bash
docker build -t ddi-serving:test . 2>&1 | tail -5
```
Expected: `Successfully built ...` 또는 `FINISHED`

- [ ] **Step 3: 컨테이너 사용자 확인**

```bash
docker run --rm ddi-serving:test whoami
```
Expected: `appuser`

- [ ] **Step 4: 커밋**

```bash
git add Dockerfile
git commit -m "fix: Dockerfile 비권한 사용자(appuser) 추가 — root 실행 방지"
```

---

## 전체 테스트 검증

- [ ] **전체 테스트 실행**

```bash
cd /Volumes/model/claude/MODE_11_hana
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```
Expected: 신규 추가 테스트 포함 전체 PASSED

- [ ] **최종 커밋 확인**

```bash
git log --oneline -10
```
