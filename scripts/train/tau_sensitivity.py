"""τ 민감도 분석 CLI — Task 4a.

폐쇄망 내부에서 stage1 모델로 검증 세트 (y_true, y_proba) 배열을 저장한 뒤,
여러 recall_floor 후보에 대한 τ_red / τ_review + 운영 영향을 비교표로 출력한다.

[폐쇄망 내부 사용 예시]

    # 1) Stage 1 모델 + 검증 피처 로드 후 예측 확률 덤프 (한 번만)
    python -c "
    import joblib, numpy as np, pandas as pd
    m = joblib.load('models/stage1_red.joblib')
    val = pd.read_parquet('data/validation_features.parquet')
    X = val[FEATURE_COLS].to_numpy()
    y = val['is_red'].to_numpy().astype(int)
    np.save('y_true.npy', y)
    np.save('y_proba.npy', m.predict_proba(X)[:, 1])
    "

    # 2) τ 민감도 분석
    python scripts/train/tau_sensitivity.py \\
        --y-true y_true.npy \\
        --y-proba y_proba.npy \\
        --recall-floors 0.85,0.90,0.92,0.95 \\
        --review-recall-target 0.98 \\
        --output-dir reports/tau_sensitivity/

결과: reports/tau_sensitivity/ 에 tau_report.json + tau_report.md 저장.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 프로젝트 루트를 sys.path 에 추가 (폐쇄망 배포 시에도 동작하도록)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hana_app.core.hierarchical_runner import tau_sensitivity_sweep  # noqa: E402


def _parse_recall_floors(text: str) -> list[float]:
    """쉼표 구분 문자열 → float list."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--recall-floors 파싱 실패 ({text!r}): {e}"
        )


def _format_markdown_table(rows: list[dict]) -> str:
    """list[dict] → 사람이 읽을 수 있는 Markdown 테이블."""
    header = (
        "| recall_floor | tau_red | tau_review | actual_recall | actual_precision "
        "| stage2_traffic_% | red_leakage_% | fallback | error |"
    )
    sep = "|" + "|".join(["---"] * 9) + "|"
    lines = [header, sep]
    for r in rows:
        if r["error"]:
            lines.append(
                f"| {r['recall_floor_requested']:.3f} | — | — | — | — | — | — | — "
                f"| {r['error'][:60]} |"
            )
        else:
            lines.append(
                "| {:.3f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:.2f} | {} | — |"
                .format(
                    r["recall_floor_requested"],
                    r["tau_red"], r["tau_review"],
                    r["actual_red_recall"], r["actual_red_precision"],
                    r["stage2_traffic_pct"], r["red_leakage_pct"],
                    "YES" if r["fallback_triggered"] else "no",
                )
            )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description="τ 민감도 분석 — recall_floor 스윕",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--y-true", required=True, type=Path,
                   help="np.save() 로 저장한 이진 라벨 배열 경로 (.npy)")
    p.add_argument("--y-proba", required=True, type=Path,
                   help="np.save() 로 저장한 Red 예측 확률 배열 경로 (.npy)")
    p.add_argument("--recall-floors", required=True, type=_parse_recall_floors,
                   help="쉼표 구분 recall_floor 스윕 값 (예: 0.85,0.90,0.95)")
    p.add_argument("--review-recall-target", type=float, default=0.98,
                   help="τ_review 가 보장할 recall (기본 0.98)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="결과 JSON + Markdown 저장 디렉터리")
    args = p.parse_args()

    if not args.y_true.exists():
        print(f"error: y_true 파일 없음: {args.y_true}", file=sys.stderr)
        return 2
    if not args.y_proba.exists():
        print(f"error: y_proba 파일 없음: {args.y_proba}", file=sys.stderr)
        return 2

    y_true = np.load(args.y_true)
    y_proba = np.load(args.y_proba)

    rows = tau_sensitivity_sweep(
        y_true=y_true,
        y_proba=y_proba,
        recall_floors=args.recall_floors,
        review_recall_target=args.review_recall_target,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "tau_report.json"
    md_path = args.output_dir / "tau_report.md"

    json_path.write_text(json.dumps({
        "input": {
            "y_true": str(args.y_true),
            "y_proba": str(args.y_proba),
            "n_samples": int(len(y_true)),
            "n_positives": int(np.asarray(y_true).astype(int).sum()),
            "recall_floors": args.recall_floors,
            "review_recall_target": args.review_recall_target,
        },
        "results": rows,
    }, ensure_ascii=False, indent=2))

    md = (
        f"# τ 민감도 분석 보고서\n\n"
        f"- 샘플 수: {len(y_true):,}\n"
        f"- 양성 (Red) 수: {int(np.asarray(y_true).astype(int).sum()):,}\n"
        f"- review_recall_target: {args.review_recall_target}\n\n"
        f"## 스윕 결과\n\n"
        + _format_markdown_table(rows)
        + "\n\n"
        "**해석 가이드**\n\n"
        "- `fallback=YES`: 이 데이터에서 해당 recall_floor 는 도달 불가 → "
        "τ_red = min(threshold) 로 대체됨. 실전 설정값 후보에서 제외.\n"
        "- `red_leakage_%`: Stage 1 에서 놓치고 review band 에도 못 들어간 "
        "진짜 Red 비율. 가장 위험한 지표.\n"
        "- `stage2_traffic_%`: Stage 2 모델로 라우팅되는 환자 비율. "
        "운영 비용 (약사 전화 + 문자 알림) 의 상한.\n"
    )
    md_path.write_text(md)

    print(f"JSON 저장: {json_path}")
    print(f"Markdown 저장: {md_path}")
    print()
    print(_format_markdown_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
