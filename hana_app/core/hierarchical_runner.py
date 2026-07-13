"""계층 분류 러너 (Stage 1 Red 이진 + Stage 2 Yellow 서브라벨 6-class).

라벨 상수, 인코딩/디코딩 헬퍼, 임계값 선택, sample_weight, 학습/추론.
Stage 1 / Stage 2 모델은 각각 독립 joblib 로 저장되고
predict_risk() 가 2단 임계값 (τ_red, τ_review) 으로 분기한다.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from .ml_runner import load_features_from_parquet, stratified_sample_from_parquet

# clinical_rules 는 scripts/etl/ 에 위치 — 절대 경로 import (모듈 최상단)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.etl.clinical_rules import CLINICAL_STANDARDS_VERSION  # noqa: E402

YELLOW_SUBTYPE_LABELS: tuple[str, ...] = (
    "Y_TRIPLE", "Y_DOUBLE", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG",
)
STAGE2_LABELS: tuple[str, ...] = YELLOW_SUBTYPE_LABELS + ("No_Alert",)

_VALID_RISK_LEVELS: frozenset[str] = frozenset({"Red", "Yellow", "Green", "Normal"})

_STAGE2_LABEL_TO_INT: dict[str, int] = {
    lbl: i for i, lbl in enumerate(STAGE2_LABELS)
}


class _ConstantNegativeStage1:
    """Red 표본이 없을 때 쓰는 '상수 비-Red' Stage 1 더미.

    predict_proba(X)[:, 1] 가 항상 0.0 (= 절대 Red 아님) → predict_risk 에서
    모든 샘플이 Stage 2(Yellow 세분화)로 분기된다. 번들 구조(stage1_red.joblib)와
    predict_risk·serving(HierarchicalModel) 인터페이스를 그대로 유지하기 위한 장치.

    ⚠️ 이 더미가 들어간 모델을 배포하면 Red 를 절대 탐지하지 못한다.
    stage_meta.stage1_trained=False 로 투명 기록되며, 운영 서빙 전 cross-family
    검토가 필요하다. (module-level 클래스라 joblib 직렬화/로드 가능.)
    """

    def predict_proba(self, X):
        n = len(np.asarray(X))
        proba = np.zeros((n, 2), dtype=float)
        proba[:, 0] = 1.0
        return proba

    def predict(self, X):
        return np.zeros(len(np.asarray(X)), dtype=int)


def build_stage2_label(risk_level: str, yellow_subtype: str | None) -> str:
    """Stage 2 학습용 라벨 변환.

    Red 는 Stage 2 대상이 아니므로 ValueError. Y_OTHER 는 학습셋에서 제외.
    알 수 없는 risk_level 도 ValueError (silent drift 방지).
    """
    if risk_level not in _VALID_RISK_LEVELS:
        raise ValueError(
            f"유효하지 않은 risk_level: {risk_level!r}. "
            f"허용 값: {sorted(_VALID_RISK_LEVELS)}"
        )
    if risk_level == "Red":
        raise ValueError("Red 는 Stage 1 에서 처리됨 — Stage 2 라벨 아님")
    if yellow_subtype == "Y_OTHER":
        raise ValueError("Y_OTHER 는 학습셋에서 제외 대상 — Stage 2 라벨로 쓸 수 없음")
    if risk_level == "Yellow":
        if yellow_subtype is None or yellow_subtype not in YELLOW_SUBTYPE_LABELS:
            raise ValueError(
                f"Yellow 는 yellow_subtype 이 {YELLOW_SUBTYPE_LABELS} 중 하나여야 함. "
                f"받은 값: {yellow_subtype!r}"
            )
        return yellow_subtype
    return "No_Alert"


def encode_stage2_labels(
    labels: Iterable[str],
) -> tuple[np.ndarray, LabelEncoder]:
    """Stage 2 라벨 문자열 → 정수 인코딩. classes_ 는 STAGE2_LABELS 순서 고정.

    Returns
    -------
    (y_int: np.ndarray, encoder: LabelEncoder)

    주의: 반환되는 encoder 에 .fit() 을 다시 호출하지 말 것.
    classes_ 가 STAGE2_LABELS 순서로 직접 할당되어 있으므로
    .fit() 이 알파벳 정렬로 덮어쓰면 정수 인덱스가 드리프트한다.
    """
    encoder = LabelEncoder()
    # LabelEncoder.fit() sorts classes_ alphabetically; override to fix STAGE2_LABELS order.
    encoder.classes_ = np.array(list(STAGE2_LABELS))
    y = encoder.transform(list(labels))
    return y, encoder


def decode_stage2_labels(y: np.ndarray, encoder) -> np.ndarray:
    """정수 → 문자열 역변환."""
    return encoder.inverse_transform(np.asarray(y))


def _normalize_stage2_local_predictions(raw_pred) -> np.ndarray:
    """Stage 2 predict 결과를 local class-index 1-D 배열로 정규화.

    일부 XGBoost/Windows 조합은 `multi:softprob` 모델의 `predict()` 결과로
    class index 벡터 대신 `(n_samples, n_classes)` 확률 행렬을 반환한다. Stage 2
    평가·CV 경로는 local class index를 기대하므로 확률 행렬은 argmax로 변환한다.
    """
    pred = np.asarray(raw_pred)
    if pred.ndim == 2:
        pred = np.argmax(pred, axis=1)
    else:
        pred = pred.reshape(-1)
    return pred.astype(int, copy=False)


def _stage2_sample_weight(
    y_train: np.ndarray,
    cost_sensitive: bool = False,
    cost_ratio_by_class: dict[str, float] | None = None,
    class_names: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Stage 2 6-class balanced sample_weight.

    balanced (클래스 불균형 역수) 가 기본 깔개. cost_sensitive=True 이고
    cost_ratio_by_class 가 주어지면 각 샘플에 해당 클래스 비용 배수를 곱한다.

    Parameters
    ----------
    y_train : np.ndarray
        정수 인코딩된 Stage 2 라벨 — encode_stage2_labels() 의 출력, 또는
        train_hierarchical 의 local→global 재매핑이 적용된 배열.
    cost_sensitive : bool
        True 면 cost_ratio_by_class 로 balanced 가중치를 곱함.
    cost_ratio_by_class : dict[str, float] | None
        클래스 이름(STAGE2_LABELS 중 하나) → 비용 배수.
        명시되지 않은 클래스는 1.0 (변경 없음) 으로 처리.
        STAGE2_LABELS 에 없는 키가 있으면 KeyError (오탈자 차단).
    class_names : tuple[str, ...] | None
        y_train 정수 인덱스 i → 클래스 이름 매핑.
        None 이면 기본 STAGE2_LABELS 사용 (전체 6-class 학습 가정).
        train_hierarchical 이 local→global 재매핑을 쓰는 경우에는
        local index 순서의 class_names 를 반드시 전달해야 cost_ratio 가
        올바른 클래스에 적용된다.
    """
    y_arr = np.asarray(y_train)
    if not np.issubdtype(y_arr.dtype, np.integer):
        raise TypeError(
            f"y_train 은 정수 인코딩 (encode_stage2_labels() 출력) 이어야 함 — "
            f"받은 dtype: {y_arr.dtype}"
        )

    balanced = compute_sample_weight("balanced", y_arr)
    if not cost_sensitive or cost_ratio_by_class is None:
        return balanced

    unknown = set(cost_ratio_by_class) - set(STAGE2_LABELS)
    if unknown:
        raise KeyError(
            f"cost_ratio_by_class 에 STAGE2 가 아닌 키 포함: {sorted(unknown)}"
        )

    _names = class_names if class_names is not None else STAGE2_LABELS
    label_strs = [_names[int(i)] for i in y_arr]
    cost_mult = np.array(
        [cost_ratio_by_class.get(s, 1.0) for s in label_strs],
        dtype=float,
    )
    return balanced * cost_mult


def stratified_sample_stage2(
    parquet_paths: list[Path] | str | Path,
    sample_size: int,
    seed: int = 42,
    memory_limit_mb: int = 512,
) -> pd.DataFrame:
    """Stage 2 용 층화 샘플링.

    전처리:
      1) risk_level != 'Red' prefilter (Red 는 Stage 1 영역)
      2) yellow_subtype == 'Y_OTHER' 제외 (학습 오염 방지)
      3) stage2_label 컬럼 derive — build_stage2_label() 로 validated
         (null yellow_subtype, 알 수 없는 risk_level 은 ValueError)
      4) stage2_label 기준 6-class 층화 추출

    내부적으로 ml_runner.stratified_sample_from_parquet 을 재사용한다
    (DuckDB numpy.int64 처리 등 호환 경로 보존).
    """
    df = load_features_from_parquet(parquet_paths, memory_limit_mb=memory_limit_mb)

    # DuckDB 가 null-only 컬럼을 Int32 로 읽을 수 있으므로 object 로 강제 변환
    df["yellow_subtype"] = df["yellow_subtype"].astype(object).where(
        df["yellow_subtype"].notna(), other=None
    )

    # prefilter
    df = df[df["risk_level"] != "Red"].copy()
    df = df[df["yellow_subtype"].fillna("") != "Y_OTHER"].copy()

    # stage2_label derive — build_stage2_label 이 validation 수행
    df["stage2_label"] = df.apply(
        lambda r: build_stage2_label(r["risk_level"], r["yellow_subtype"]),
        axis=1,
    )
    df["stage2_label_int"] = df["stage2_label"].map(_STAGE2_LABEL_TO_INT).astype("int64")

    # 임시 parquet → 층화 추출 (TemporaryDirectory 로 자동 정리)
    with tempfile.TemporaryDirectory(prefix="stage2_stratified_") as tmp_dir:
        tmp = Path(tmp_dir) / "stage2.parquet"
        df.to_parquet(tmp, index=False)
        sampled = stratified_sample_from_parquet(
            parquet_paths=tmp,
            target_col="stage2_label_int",
            sample_size=sample_size,
            seed=seed,
            memory_limit_mb=memory_limit_mb,
        )

    sampled = sampled.drop(columns=["stage2_label_int"], errors="ignore")
    return sampled


def select_thresholds_from_pr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    recall_floor: float = 0.90,
    review_recall_target: float = 0.98,
) -> dict[str, float]:
    """PR 곡선에서 τ_red, τ_review 2단 임계값 선택.

    τ_red:
      Recall ≥ recall_floor 제약 하에서 Precision 이 최대가 되는 임계값.

    τ_review:
      Recall ≥ review_recall_target (더 보수적) 을 만족하는 최대 임계값
      (더 높은 threshold = 더 엄격 = 더 적은 샘플 → 가장 엄격한 후보 채택).
      precondition 으로 review_recall_target > recall_floor 을 강제하고,
      PR 곡선 해상도 부족으로 동값이 나올 때는 τ_review 를 1e-6 만큼 내려
      τ_review < τ_red 를 항상 보장한다.

    Parameters
    ----------
    y_true : np.ndarray
        이진 {0, 1} 정답 라벨. 양/음 둘 다 최소 1건 이상 필요.
    y_proba : np.ndarray
        [0, 1] 범위의 양성 클래스 예측 확률.
    recall_floor : float
        τ_red 가 보장해야 할 Recall 하한. 범위 (0, 1].
    review_recall_target : float
        τ_review 가 보장해야 할 Recall. recall_floor 보다 엄격해야 함 (더 큰 값).

    Returns
    -------
    {"tau_red": float, "tau_review": float}

    Raises
    ------
    ValueError
        입력이 계약을 위반할 때. silent garbage 대신 명시적 실패.
    """
    y_true_arr = np.asarray(y_true)
    if not np.isin(y_true_arr, [0, 1]).all():
        raise ValueError("y_true 는 이진 {0, 1} 배열이어야 함")
    if np.unique(y_true_arr).size < 2:
        raise ValueError(
            "y_true 는 양성/음성 샘플을 모두 포함해야 함 (PR 곡선 비정의)"
        )

    y_proba_arr = np.asarray(y_proba, dtype=float)
    if y_proba_arr.size != y_true_arr.size:
        raise ValueError(
            f"y_true 와 y_proba 길이 불일치: {y_true_arr.size} vs {y_proba_arr.size}"
        )
    if y_proba_arr.min() < 0.0 or y_proba_arr.max() > 1.0:
        raise ValueError("y_proba 는 [0.0, 1.0] 범위여야 함")

    if not (0.0 < recall_floor <= 1.0):
        raise ValueError(f"recall_floor 은 (0, 1] 범위 — 받은 값: {recall_floor}")
    if not (0.0 < review_recall_target <= 1.0):
        raise ValueError(
            f"review_recall_target 은 (0, 1] 범위 — 받은 값: {review_recall_target}"
        )
    if review_recall_target <= recall_floor:
        raise ValueError(
            f"review_recall_target ({review_recall_target}) 은 "
            f"recall_floor ({recall_floor}) 보다 커야 함 (τ_review < τ_red 보장)"
        )

    precision, recall, thresholds = precision_recall_curve(
        y_true_arr.astype(int), y_proba_arr
    )
    # precision_recall_curve: precision/recall 은 len N+1, thresholds 는 len N

    # τ_red: recall ≥ recall_floor 을 만족하는 후보 중 최대 precision
    valid_red = recall[:-1] >= recall_floor
    if not valid_red.any():
        tau_red = float(thresholds.min())
    else:
        cand_idx = np.where(valid_red)[0]
        best = cand_idx[np.argmax(precision[:-1][cand_idx])]
        tau_red = float(thresholds[best])

    # τ_review: recall ≥ review_recall_target 을 만족하는 최대 threshold
    valid_review = recall[:-1] >= review_recall_target
    if not valid_review.any():
        tau_review = float(thresholds.min())
    else:
        cand_idx = np.where(valid_review)[0]
        tau_review = float(thresholds[cand_idx].max())

    # PR 곡선 해상도 부족(양성 수 적음) 으로 τ_review == τ_red 동값 가능 —
    # 불변식 τ_review < τ_red 를 유지하도록 미세 조정.
    if tau_review >= tau_red:
        tau_review = max(0.0, tau_red - 1e-6)
        # 밴드 붕괴 신호: stage-1 이 거의 완벽분리(과확신) → score 기반 review 구간이
        # 폭 1e-6 로 사실상 死(score 로는 "Red 의심" 거의 안 걸림). rulefeat 누수·
        # 확률보정 재검토 필요. 최종 Red 는 금기 결정적 백스톱으로만 좌우될 수 있음.
        print(
            f"[hierarchical] ⚠ PR band collapsed: tau_red={tau_red:.6f}, "
            f"review band forced to width 1e-6 — score-based review queue effectively inert "
            f"(stage-1 overconfident; check rulefeat leakage / calibration)",
            file=sys.stderr,
        )

    return {"tau_red": tau_red, "tau_review": tau_review}


def tau_sensitivity_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    recall_floors: Iterable[float],
    review_recall_target: float = 0.98,
) -> list[dict]:
    """recall_floor 값을 스윕하며 각 후보의 τ_red/τ_review + 운영 메트릭 계산.

    실데이터 접근 전 하이퍼파라미터 민감도 분석용 — 사용자가 폐쇄망에서
    stage1 모델로 (y_true, y_proba) 배열을 뽑아 본 함수에 넣으면 여러
    recall_floor 후보에 대한 τ 와 예상 운영 영향을 비교표로 받는다.

    각 row 는 precondition 실패(ValueError)를 crash 대신 error 필드에 기록해
    전체 스윕이 중단되지 않도록 한다.

    Parameters
    ----------
    y_true, y_proba
        Stage 1 검증 세트에서 추출한 이진 라벨 / Red 예측 확률.
        select_thresholds_from_pr 와 동일 계약.
    recall_floors : Iterable[float]
        τ_red 후보를 결정할 Red recall 하한 스윕 값 목록 (예: [0.85, 0.90, 0.95]).
    review_recall_target : float, default 0.98
        τ_review 가 보장할 recall (모든 row 공통).

    Returns
    -------
    list[dict]
        각 row 는 다음 키 포함:
        - recall_floor_requested : 입력된 recall_floor
        - tau_red, tau_review : 선택된 임계값 (error 시 None)
        - actual_red_recall : tau_red 에서의 실제 recall
        - actual_red_precision : tau_red 에서의 실제 precision
        - fallback_triggered : recall_floor 가 만족 불가해 min-threshold fallback 사용
        - n_red_confirmed : p_red ≥ tau_red 건수
        - n_review_band : tau_review ≤ p_red < tau_red 건수 (red_suspect 태그 대상)
        - n_clean_stage2 : p_red < tau_review 건수 (Stage 2 단독)
        - stage2_traffic_pct : Stage 2 로 라우팅되는 비율 (100 * (review_band + clean) / n)
        - red_missed_to_stage2 : y_true=1 인데 p_red < tau_red 로 Stage 1 에서 놓친 건수
        - red_lost_clean_stage2 : y_true=1 인데 p_red < tau_review — red_suspect 태그도
          못 받아 영구 유실된 건수 (가장 위험한 지표)
        - red_leakage_pct : 100 * red_lost_clean_stage2 / total_positives
        - error : precondition 실패 시 메시지, 정상 시 None
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    n = len(y_true_arr)
    total_positives = int(y_true_arr.sum())

    rows: list[dict] = []
    for floor in recall_floors:
        row: dict = {
            "recall_floor_requested": float(floor),
            "tau_red": None,
            "tau_review": None,
            "actual_red_recall": None,
            "actual_red_precision": None,
            "fallback_triggered": None,
            "n_red_confirmed": None,
            "n_review_band": None,
            "n_clean_stage2": None,
            "stage2_traffic_pct": None,
            "red_missed_to_stage2": None,
            "red_lost_clean_stage2": None,
            "red_leakage_pct": None,
            "error": None,
        }
        try:
            thr = select_thresholds_from_pr(
                y_true_arr, y_proba_arr,
                recall_floor=float(floor),
                review_recall_target=review_recall_target,
            )
        except ValueError as e:
            row["error"] = str(e)
            rows.append(row)
            continue

        tau_red = thr["tau_red"]
        tau_review = thr["tau_review"]

        # Stage 1 decision at tau_red
        pred_red = y_proba_arr >= tau_red
        tp = int(((pred_red == 1) & (y_true_arr == 1)).sum())
        fp = int(((pred_red == 1) & (y_true_arr == 0)).sum())
        fn = int(((pred_red == 0) & (y_true_arr == 1)).sum())
        actual_recall = tp / (tp + fn) if (tp + fn) else 0.0
        actual_precision = tp / (tp + fp) if (tp + fp) else 0.0

        n_red_confirmed = int(pred_red.sum())
        in_review = (y_proba_arr >= tau_review) & (~pred_red)
        in_clean = y_proba_arr < tau_review
        n_review_band = int(in_review.sum())
        n_clean_stage2 = int(in_clean.sum())

        stage2_traffic = n_review_band + n_clean_stage2
        stage2_pct = 100.0 * stage2_traffic / n if n else 0.0

        red_missed = fn  # 모든 y_true=1 중 tau_red 미달
        # red_suspect 태그도 못 받은 영구 유실 (clean_stage2 안의 실제 Red)
        red_lost = int(((y_proba_arr < tau_review) & (y_true_arr == 1)).sum())
        red_leakage_pct = 100.0 * red_lost / total_positives if total_positives else 0.0

        # Fallback 판정: recall_floor 이 만족 불가했다면 actual_recall 이 여전히 floor 미만
        fallback = actual_recall < float(floor) - 1e-9

        row.update({
            "tau_red": tau_red,
            "tau_review": tau_review,
            "actual_red_recall": actual_recall,
            "actual_red_precision": actual_precision,
            "fallback_triggered": fallback,
            "n_red_confirmed": n_red_confirmed,
            "n_review_band": n_review_band,
            "n_clean_stage2": n_clean_stage2,
            "stage2_traffic_pct": stage2_pct,
            "red_missed_to_stage2": red_missed,
            "red_lost_clean_stage2": red_lost,
            "red_leakage_pct": red_leakage_pct,
        })
        rows.append(row)
    return rows


def train_hierarchical(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_dir,
    seed: int = 42,
    stage1_params: dict | None = None,
    stage2_params: dict | None = None,
    recall_floor: float = 0.90,
    review_recall_target: float = 0.98,
    cost_sensitive: bool = False,
    cost_ratio_by_class: dict[str, float] | None = None,
    log_cb=None,
) -> dict:
    """Stage 1 (Red 이진) + Stage 2 (Yellow 서브라벨 6-class) 계층 학습.

    df 에는 risk_level, yellow_subtype, feature_cols 가 포함되어야 한다.
    Y_OTHER 는 Stage 2 학습셋에서 제외된다.

    저장 파일:
      {output_dir}/stage1_red.joblib
      {output_dir}/stage2_yellow.joblib
      {output_dir}/stage_meta.json  (임계값, feature_cols, 라벨 카운트, SHA-256)
    """
    import hashlib
    import json
    from collections import Counter

    import joblib
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Red 이진 ─────────────────────────────────────────────────
    X = df[feature_cols].to_numpy()
    y1 = (df["risk_level"] == "Red").astype(int).to_numpy()

    # Red(양성)·비Red(음성) 둘 다 충분해야 실제 Stage 1 학습·PR 임계값 선택이 가능하다.
    # 데이터에 Red 가 거의/전혀 없는 경우(예: 다운로드 Raw same-window 표본)는
    # 학습을 죽이는 대신 '상수 비-Red' Stage 1 더미로 degrade 하고 Stage 2(Yellow
    # 세분화) 만 학습한다. 번들 구조(stage1_red.joblib)·predict_risk·serving
    # 인터페이스를 그대로 유지하기 위함이며, stage_meta.stage1_trained=False 로
    # 투명하게 기록한다. (이 모델 배포 시 Red 미탐지 한계 — UI/메타에서 경고.)
    _n_red = int(y1.sum())
    _n_non = int(y1.size - _n_red)
    _dist = df["risk_level"].value_counts().to_dict()
    _MIN_RED_FOR_STAGE1 = 10

    def _degrade_stage1(reason: str) -> tuple:
        # p_red 가 항상 0 → 모두 Stage 2 로 분기. tau_red=1.0(아무도 Red 안 됨),
        # tau_review=0.5(p_red=0 < 0.5 → red_suspect=False). 불변식 τ_review<τ_red 유지.
        if log_cb:
            log_cb(f"⚠️ Stage 1(Red) 건너뜀 — {reason}. 상수 비-Red 더미로 대체, Stage 2 만 학습.")
        return _ConstantNegativeStage1(), {"tau_red": 1.0, "tau_review": 0.5}

    stage1_trained = False
    if _n_red >= _MIN_RED_FOR_STAGE1 and _n_non >= _MIN_RED_FOR_STAGE1:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y1, test_size=0.2, random_state=seed, stratify=y1,
        )
        if np.unique(y_val).size < 2:
            # 층화해도 양성이 적으면 검증셋이 단일 클래스가 될 수 있다 → degrade.
            m1, thresholds = _degrade_stage1(
                f"검증셋이 단일 클래스 (Red {_n_red}건으로 부족)"
            )
        else:
            pos = int(y_tr.sum())
            neg = int(len(y_tr) - pos)
            scale_pos_weight = neg / max(pos, 1)

            defaults1 = dict(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="logloss",
                scale_pos_weight=scale_pos_weight, random_state=seed, verbosity=0,
            )
            if stage1_params:
                defaults1.update(stage1_params)
            m1 = XGBClassifier(**defaults1)
            m1.fit(X_tr, y_tr)
            proba_val = m1.predict_proba(X_val)[:, 1]
            thresholds = select_thresholds_from_pr(
                y_val, proba_val,
                recall_floor=recall_floor,
                review_recall_target=review_recall_target,
            )
            stage1_trained = True
    else:
        m1, thresholds = _degrade_stage1(
            f"Red {_n_red}건 / 비Red {_n_non}건 (Red 최소 {_MIN_RED_FOR_STAGE1}건 필요) "
            f"— 위험도 분포 {_dist}"
        )

    # ── Stage 2: 6-class ─────────────────────────────────────────────────
    mask_non_red = df["risk_level"] != "Red"
    mask_not_other = df["yellow_subtype"].fillna("") != "Y_OTHER"
    y_other_excluded = int(((df["risk_level"] == "Yellow") &
                            (df["yellow_subtype"] == "Y_OTHER")).sum())
    df2 = df[mask_non_red & mask_not_other].copy()
    labels_str = [
        build_stage2_label(r["risk_level"], r["yellow_subtype"])
        for _, r in df2.iterrows()
    ]
    # global indices (STAGE2_LABELS 순서 고정): 0..5
    y2_global, encoder = encode_stage2_labels(labels_str)
    X2 = df2[feature_cols].to_numpy()

    # XGBoost 는 y 가 [0, num_class) 연속 범위여야 한다.
    # 학습셋에 없는 클래스(예: Y_FRAG)가 있으면 global → local 재매핑.
    classes_present = np.unique(y2_global)          # e.g. [0,1,2,3,5]
    if len(classes_present) < len(STAGE2_LABELS) or not (
        classes_present == np.arange(len(STAGE2_LABELS))
    ).all():
        global_to_local = {int(g): l for l, g in enumerate(classes_present)}
        y2 = np.array([global_to_local[int(g)] for g in y2_global], dtype=int)
        n_stage2_classes = len(classes_present)
    else:
        y2 = y2_global
        n_stage2_classes = len(STAGE2_LABELS)

    # local index → 전역 클래스 이름 매핑: cost_ratio 가 올바른 클래스에 적용되도록 보장
    local_class_names = tuple(STAGE2_LABELS[g] for g in classes_present)
    sw2 = _stage2_sample_weight(
        y2, cost_sensitive=cost_sensitive,
        cost_ratio_by_class=cost_ratio_by_class,
        class_names=local_class_names,
    )

    defaults2 = dict(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=n_stage2_classes,
        eval_metric="mlogloss", random_state=seed, verbosity=0,
    )
    if stage2_params:
        defaults2.update(stage2_params)
    m2 = XGBClassifier(**defaults2)
    m2.fit(X2, y2, sample_weight=sw2)

    # ── 저장 ─────────────────────────────────────────────────────────────
    p1 = out / "stage1_red.joblib"
    p2 = out / "stage2_yellow.joblib"
    joblib.dump(m1, p1)
    # stage2_classes_global: predict_risk 에서 로컬 인덱스 → 전역 STAGE2_LABELS 매핑에 사용
    joblib.dump({
        "model": m2,
        "encoder": encoder,
        "stage2_classes_global": classes_present.tolist(),
        "classes_present": classes_present.tolist(),
    }, p2)

    def _sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    label_counts = dict(Counter(labels_str))

    from scripts.etl.prescription_aggregator import (
        DDI_FEATURE_SEMANTICS_VERSION,
        FEATURE_SEMANTICS_VERSION,
    )
    meta = {
        "clinical_standards_version": CLINICAL_STANDARDS_VERSION,
        "ddi_feature_semantics_version": DDI_FEATURE_SEMANTICS_VERSION,
        "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
        "feature_cols": list(feature_cols),
        "thresholds": thresholds,
        "stage2_labels": list(STAGE2_LABELS),
        "stage2_label_counts": label_counts,
        "y_other_excluded_count": y_other_excluded,
        "stage1_sha256": _sha(p1),
        "stage2_sha256": _sha(p2),
        "cost_sensitive": cost_sensitive,
        "cost_ratio_by_class": cost_ratio_by_class,
        # Red 표본 부족으로 Stage 1 이 '상수 비-Red' 더미면 False — 배포 전 검토 신호.
        "stage1_trained": stage1_trained,
        "stage1_red_count": _n_red,
    }
    (out / "stage_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )

    # ── Stage 2 평가 지표 (in-sample, 학습셋 기준) ────────────────────────────
    # 실제 학습은 전체 X2/y2로 진행하므로 훈련 결과 불변.
    # 지표는 낙관적(in-sample)이지만 피처 중요도·클래스 분포 파악에 충분함.
    from sklearn.metrics import classification_report as _cr
    from sklearn.metrics import confusion_matrix as _cm_sk
    from sklearn.metrics import f1_score as _f1

    _fi_arr = m2.feature_importances_
    _feature_importance = sorted(
        [{"feature": fc, "importance": float(fi)}
         for fc, fi in zip(feature_cols, _fi_arr)],
        key=lambda x: -x["importance"],
    )

    _y2_pred_local = _normalize_stage2_local_predictions(m2.predict(X2))
    _cls_names = list(local_class_names)
    _y2_true_str = [_cls_names[i] for i in y2]
    _y2_pred_str = [_cls_names[int(i)] for i in _y2_pred_local]

    _f1_macro = float(_f1(_y2_true_str, _y2_pred_str, average="macro", zero_division=0))
    _cm_arr = _cm_sk(_y2_true_str, _y2_pred_str, labels=_cls_names).tolist()
    _cr_str = _cr(_y2_true_str, _y2_pred_str, labels=_cls_names, zero_division=0)

    return {
        "stage1_model": m1,
        "stage2_model": m2,
        "stage2_encoder": encoder,
        "thresholds": thresholds,
        "stage2_label_counts": label_counts,
        "y_other_excluded_count": y_other_excluded,
        "stage1_trained": stage1_trained,
        "stage1_red_count": _n_red,
        "meta_path": out / "stage_meta.json",
        # 평가 지표
        "feature_importance": _feature_importance,
        "f1_macro": _f1_macro,
        "confusion_matrix": _cm_arr,
        "stage2_class_names": _cls_names,
        "classification_report": _cr_str,
        "stage2_train_size": len(X2),
    }


# Red 권장 개입 — Red 는 stage2 라벨이 아니라 risk_level/Stage1 로 분기하므로
# ACTION_BY_LABEL 에 없고, _dispatch_result(확정 Red)와 결과분석 개입 분포가 공유한다.
RED_ACTION: str = "즉각 개입"

# Yellow 세부 라벨별 권장 개입 (2026-06-07 개입 위계 재설계).
# Red(금기)=즉각 개입 > Y_DDI_MAJOR(major DDI)=약사 전화 > Y_TRIPLE(중증: triple_whammy/
# 10drug+고위험/고령+장기 또는 3차원)=문자 안내 > Y_DOUBLE·단일차원(중등도DDI/중복/다기관)=
# 모니터링 > No_Alert·Green·Normal=관여 안 함.
ACTION_BY_LABEL: dict[str, str] = {
    "Y_DDI_MAJOR":  "약사 전화",
    "Y_TRIPLE":     "문자 안내",
    "Y_DOUBLE":     "모니터링",
    "Y_DDI_MOD":    "모니터링",
    "Y_DUP":        "모니터링",
    "Y_FRAG":       "모니터링",
    "No_Alert":     "관여 안 함",
}


def _dispatch_result(
    p_red: float,
    stage2_probs: np.ndarray | None,
    stage2_labels: tuple[str, ...],
    tau_red: float,
    tau_review: float,
) -> dict:
    """단일 환자에 대한 2단 임계값 분기 결과.

    p_red >= tau_red → Red 확정 (Stage 2 skip)
    tau_review <= p_red < tau_red → Stage 2 라벨 + red_suspect=True
    p_red < tau_review → Stage 2 라벨 단독 (red_suspect=False)
    """
    if p_red >= tau_red:
        return {
            "risk_level": "Red",
            "p_red": float(p_red),
            "stage2_probs": None,
            "red_suspect": False,
            "action": RED_ACTION,
        }
    if stage2_probs is None:
        raise ValueError(
            "_dispatch_result: p_red < tau_red 일 때 stage2_probs 는 None 일 수 없음. "
            "predict_risk() 를 호출하거나 유효한 probs 배열을 전달하세요."
        )
    stage2_idx = int(np.argmax(stage2_probs))
    stage2_label = stage2_labels[stage2_idx]
    red_suspect = bool(p_red >= tau_review)
    return {
        "risk_level": stage2_label,
        "p_red": float(p_red),
        "stage2_probs": {lbl: float(stage2_probs[i])
                         for i, lbl in enumerate(stage2_labels)},
        "red_suspect": red_suspect,
        "action": ACTION_BY_LABEL.get(stage2_label, "알림 없음"),
    }


def predict_risk(
    X: np.ndarray,
    stage1_model,
    stage2_model,
    stage2_encoder,
    thresholds: dict[str, float],
    classes_present: list[int] | None = None,
) -> list[dict]:
    """계층 추론 — 각 샘플에 대해 2단 분기 결과 리스트 반환.

    X : (n, n_features) 피처 배열 (열 순서는 학습 시 feature_cols 와 일치해야 함)

    Task 8 의 local→global remapping 을 역변환해 stage2_probs 를 항상
    STAGE2_LABELS (6-class) 순서의 벡터로 정렬한다. 누락된 클래스는 0.0.

    Note
    ----
    classes_present=None 일 때 encoder.classes_ 를 fallback 으로 사용하지만,
    stage2_model 이 일부 클래스만으로 학습된 경우 (local→global remapping 발동)
    predict_proba 출력 열 수가 encoder.classes_ 수와 맞지 않아 IndexError 가 된다.
    그 경우 classes_present 를 명시적으로 전달해야 한다 (저장된 bundle["classes_present"]).
    """
    X_arr = np.asarray(X)
    p_red = stage1_model.predict_proba(X_arr)[:, 1]

    # Stage 2 확률 — local 공간 → global STAGE2_LABELS 공간으로 reorder/pad
    local_probs = stage2_model.predict_proba(X_arr)  # shape (n, k)
    n = len(X_arr)
    stage2_probs_global = np.zeros((n, len(STAGE2_LABELS)), dtype=float)

    if classes_present is not None:
        # train_hierarchical 번들에서 저장된 global 인덱스 리스트 사용
        for local_i, global_i in enumerate(classes_present):
            stage2_probs_global[:, global_i] = local_probs[:, local_i]
    else:
        # classes_present 누락 시 encoder.classes_ 로 fallback
        # 주의: local→global remapping 이 발동한 경우 열 수 불일치 → ValueError
        if local_probs.shape[1] != len(stage2_encoder.classes_):
            raise ValueError(
                f"stage2_model 출력 클래스 수({local_probs.shape[1]})와 "
                f"encoder.classes_ 수({len(stage2_encoder.classes_)})가 불일치. "
                "일부 클래스가 학습에서 누락된 경우 classes_present 를 명시적으로 전달해야 함 "
                "(저장된 bundle['classes_present'] 사용)."
            )
        class_to_global = {c: i for i, c in enumerate(STAGE2_LABELS)}
        for local_i, cls_str in enumerate(stage2_encoder.classes_):
            if cls_str in class_to_global:
                stage2_probs_global[:, class_to_global[cls_str]] = local_probs[:, local_i]

    tau_red = thresholds["tau_red"]
    tau_review = thresholds["tau_review"]

    results = []
    for i in range(n):
        results.append(_dispatch_result(
            p_red=float(p_red[i]),
            stage2_probs=stage2_probs_global[i],
            stage2_labels=STAGE2_LABELS,
            tau_red=tau_red,
            tau_review=tau_review,
        ))
    return results
