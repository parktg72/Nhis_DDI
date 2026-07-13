"""계층(hierarchical) 모델 헤드리스 재학습 - same-window 분류기.

Page 3 (RAW → hierarchical) 데스크톱 흐름을 CLI 로 재현해, 동일 raw Parquet 에서
앱과 **같은 함수**(`build_patient_features_from_parquet` → `_patient_features_to_row`
→ `train_hierarchical`)로 7-class 계층 번들(stage1_red.joblib / stage2_yellow.joblib /
stage_meta.json)을 산출한다. 산출 번들을 `HIERARCHICAL_MODEL_DIR` 로 지정해 배포한다.

라벨 공간은 학습 시점의 STAGE2_LABELS 를 따른다(현재 7-class: Y_TRIPLE/Y_DOUBLE/
Y_DDI_MAJOR/Y_DDI_MOD/Y_DUP/Y_FRAG/No_Alert). 서빙 로드 가드가 이 라벨 공간과
불일치하는 구 번들을 거부한다(serving.predictor.HierarchicalPredictor.load).

freeze 주의: 이 모델은 **same-window 위험 분류기**(현재 피처→현재 위험)로 future-onset
Nov→Dec 홀드아웃 트랙이 아니다 → freeze-safe. 학습 데이터(raw)는 `--raw-dir` 로 사용자가
제어한다. 동결 future-outcome 데이터셋 빌더와는 무관하다.

예:
    python -m scripts.ops.retrain_hierarchical \
        --raw-dir data/Raw --output-dir hana_app/models/hierarchical/retrain_7class
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hana_app.core.hierarchical_runner import STAGE2_LABELS, train_hierarchical
from hana_app.core.ml_runner import (
    FEATURE_COLS,
    _patient_features_to_row,
    build_patient_features_from_parquet,
)
from scripts.etl.models import PatientFeatures


def collect_raw_paths(raw_dir: str | Path, glob="records_*.parquet") -> list[Path]:
    """raw_dir 에서 glob 패턴(들)에 맞는 Parquet 경로를 정렬·dedupe 해 반환 (flat).

    glob 은 단일 문자열 또는 패턴 리스트. 윈도우 학습(예: 07~11 = Dec 제외)에
    여러 패턴(records_20240[7-9]*.parquet, records_20241[01]*.parquet)을 합집합.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"raw-dir 없음: {raw_path}")
    patterns = [glob] if isinstance(glob, str) else list(glob)
    matched: set[Path] = set()
    for pat in patterns:
        matched.update(raw_path.glob(pat))
    paths = sorted(matched)
    if not paths:
        raise FileNotFoundError(f"raw-dir 에 {patterns} 매칭 파일 없음: {raw_path}")
    return paths


def retrain_hierarchical(
    raw_paths: list[Path],
    output_dir: str | Path,
    *,
    feature_cols: list[str] | None = None,
    window_days: int = 90,
    poly_threshold: int = 5,
    patient_batch_size: int = 5000,
    memory_limit_mb: int = 0,
    seed: int = 42,
    recall_floor: float = 0.90,
    review_recall_target: float = 0.98,
    cost_sensitive: bool = False,
    log_cb=print,
) -> dict:
    """raw Parquet → 피처 → df → train_hierarchical. 결과 dict(+output_dir/n_patients) 반환.

    앱과 동일 경로: build_patient_features_from_parquet(내부에서 DrugMaster 로드 →
    DDI 중증도·risk_level·yellow_subtype 산출) → _patient_features_to_row → train_hierarchical.
    """
    start = perf_counter()
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    out = Path(output_dir)

    log_cb(f"[1/3] 피처 계산 - raw 파일 {len(raw_paths)}개")
    # build_patient_features_from_parquet 는 실제로 **피처 배치 Parquet 경로(list[Path])**
    # 를 반환한다(소량·대량 분기 모두; -> list[PatientFeatures] 주석은 stale). 각 parquet 는
    # _patient_features_to_row 전체 스키마(risk_level/yellow_subtype/FEATURE_COLS)를 가진다.
    # 방어적으로 PatientFeatures 리스트 반환 경우도 처리.
    features = build_patient_features_from_parquet(
        parquet_paths=list(raw_paths),
        window_days=window_days,
        poly_threshold=poly_threshold,
        patient_batch_size=patient_batch_size,
        memory_limit_mb=memory_limit_mb,
        progress_cb=log_cb,
    )
    if not features:
        raise ValueError("피처 0건 - raw 데이터/필터(poly_threshold)를 확인하세요.")

    log_cb(f"[2/3] DataFrame 변환 - 피처 산출 {len(features)}건")
    if isinstance(features[0], PatientFeatures):
        df = pd.DataFrame([_patient_features_to_row(f) for f in features])
    else:  # list[Path] - 디스크 배치 parquet (실제 계약, Page3 와 동일 경로)
        df = pd.concat([pd.read_parquet(p) for p in features], ignore_index=True)

    log_cb(f"[3/3] 계층 학습 - feature_cols={len(cols)}, output={out}")
    result = train_hierarchical(
        df=df,
        feature_cols=cols,
        output_dir=out,
        seed=seed,
        recall_floor=recall_floor,
        review_recall_target=review_recall_target,
        cost_sensitive=cost_sensitive,
        log_cb=log_cb,
    )
    result["output_dir"] = str(out)
    result["n_patients"] = len(df)
    result["build_time_sec"] = round(perf_counter() - start, 2)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="계층 모델 헤드리스 재학습 (same-window, freeze-safe)")
    p.add_argument("--raw-dir", required=True, help="raw records_*.parquet 디렉터리 (예: data/Raw)")
    p.add_argument("--glob", nargs="+", default=["records_*.parquet"],
                   help="raw 파일 glob 패턴(들). 다중 지정 시 합집합(예: 07~11 윈도우)")
    p.add_argument("--output-dir", default=None,
                   help="번들 출력 디렉터리 (기본: hana_app/models/hierarchical/retrain_<ts>)")
    p.add_argument("--window-days", type=int, default=90)
    p.add_argument("--poly-threshold", type=int, default=5)
    p.add_argument("--patient-batch-size", type=int, default=5000)
    p.add_argument("--memory-limit-mb", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recall-floor", type=float, default=0.90)
    p.add_argument("--review-recall-target", type=float, default=0.98)
    p.add_argument("--cost-sensitive", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows 폐쇄망 콘솔(cp949)에서 한글/유니코드 로그가 UnicodeEncodeError 로 죽지
    # 않도록 stdout/stderr 를 UTF-8 로 재설정(chcp 65001 정합). 리다이렉트/구버전 대비 guard.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    raw_paths = collect_raw_paths(args.raw_dir, args.glob)
    output_dir = args.output_dir or (
        Path(__file__).resolve().parents[2] / "hana_app" / "models" / "hierarchical"
        / f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    result = retrain_hierarchical(
        raw_paths,
        output_dir,
        window_days=args.window_days,
        poly_threshold=args.poly_threshold,
        patient_batch_size=args.patient_batch_size,
        memory_limit_mb=args.memory_limit_mb,
        seed=args.seed,
        recall_floor=args.recall_floor,
        review_recall_target=args.review_recall_target,
        cost_sensitive=args.cost_sensitive,
    )
    print(f"[OK] 번들: {result['output_dir']}")
    print(f"n_patients={result['n_patients']} build_time={result['build_time_sec']}s")
    print(f"stage2_labels({len(STAGE2_LABELS)})={list(STAGE2_LABELS)}")
    print(f"stage2_label_counts={result.get('stage2_label_counts')}")
    print(f"stage1_trained={result.get('stage1_trained')} "
          f"stage1_red_count={result.get('stage1_red_count')}")
    print(f"thresholds={result.get('thresholds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
