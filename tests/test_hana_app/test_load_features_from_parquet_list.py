"""ml_runner.load_features_from_parquet — 리스트 입력 회귀 테스트.

배경: 다운로드 Raw / 디스크기반 피처빌드는 배치당 Parquet 한 개씩 만들어
`list[Path]` 를 돌려준다. Page 3 계층학습 경로(및 피처저장 버튼)가 이 리스트를
그대로 `load_features_from_parquet` 에 넘겼는데, DuckDB 분기가 `Path(list)` 에서
`TypeError` 로 터졌다(비-DuckDB 폴백만 list 를 처리). DuckDB `read_parquet([...])`
는 파일목록을 네이티브 지원하므로 함수가 list 를 받도록 수정했다.
"""
import pandas as pd
import pytest

from hana_app.core.ml_runner import load_features_from_parquet


def _write(path, df):
    df.to_parquet(path, index=False)
    return path


def test_accepts_list_of_paths_and_concatenates(tmp_path):
    p1 = _write(tmp_path / "features_batch_0000.parquet",
                pd.DataFrame({"patient_id": ["a", "b"], "risk_level": ["Red", "Green"]}))
    p2 = _write(tmp_path / "features_batch_0001.parquet",
                pd.DataFrame({"patient_id": ["c"], "risk_level": ["Yellow"]}))

    out = load_features_from_parquet([p1, p2])

    assert len(out) == 3
    assert set(out["patient_id"]) == {"a", "b", "c"}


def test_single_path_still_works(tmp_path):
    p1 = _write(tmp_path / "features_batch_0000.parquet",
                pd.DataFrame({"patient_id": ["a", "b"], "risk_level": ["Red", "Green"]}))

    out = load_features_from_parquet(p1)

    assert len(out) == 2


def test_columns_subset_with_list(tmp_path):
    p1 = _write(tmp_path / "f0.parquet",
                pd.DataFrame({"patient_id": ["a"], "risk_level": ["Red"], "age": [50]}))
    p2 = _write(tmp_path / "f1.parquet",
                pd.DataFrame({"patient_id": ["b"], "risk_level": ["Green"], "age": [60]}))

    out = load_features_from_parquet([p1, p2], columns=["patient_id", "age"])

    assert set(out.columns) == {"patient_id", "age"}
    assert len(out) == 2


def test_empty_list_raises():
    with pytest.raises(ValueError):
        load_features_from_parquet([])
