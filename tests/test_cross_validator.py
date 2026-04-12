"""cross_validator.py 단위 테스트 — 교차 검증 파이프라인 검증"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cross_validator import CrossValidator, _ALLOWED_COVARS, _DEFAULT_COVARS


# ---------------------------------------------------------------------------
# 공용 픽스처
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df_cr():
    """경쟁위험 분석에 쓰이는 최소 DataFrame."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        'follow_up_years': np.random.uniform(0.5, 10.0, n),
        'is_t1dm':         np.random.randint(0, 2, n),
        'is_t2dm_oha':     np.random.randint(0, 2, n),
        'age_at_index':    np.random.uniform(40, 80, n),
        'male':            np.random.randint(0, 2, n),
    })


@pytest.fixture
def sample_event_type(sample_df_cr):
    """0=검열, 1=관심사건, 2=경쟁위험."""
    n = len(sample_df_cr)
    et = np.zeros(n, dtype=int)
    et[:15] = 1
    et[15:25] = 2
    return et


@pytest.fixture
def cv():
    return CrossValidator()


# ---------------------------------------------------------------------------
# 1. export_csv_creates_file
# ---------------------------------------------------------------------------

class TestExportCsvForR:
    def test_export_csv_creates_file(self, cv, sample_df_cr, sample_event_type, tmp_path):
        """CSV 파일이 정상적으로 생성된다."""
        csv_path = cv.export_csv_for_r(
            sample_df_cr, sample_event_type, 'dementia_event', output_dir=tmp_path
        )
        assert csv_path.exists()
        assert csv_path.suffix == '.csv'

    def test_export_csv_contains_correct_columns(self, cv, sample_df_cr, sample_event_type, tmp_path):
        """CSV에 time, event_type, 공변량 컬럼이 포함된다."""
        csv_path = cv.export_csv_for_r(
            sample_df_cr, sample_event_type, 'dementia_event', output_dir=tmp_path
        )
        df = pd.read_csv(csv_path)
        assert 'time' in df.columns
        assert 'event_type' in df.columns
        # follow_up_years → time 으로 rename됐어야 함
        assert 'follow_up_years' not in df.columns

    def test_export_csv_no_patient_id(self, cv, sample_event_type, tmp_path):
        """INDI_DSCM_NO 등 환자 식별자가 포함된 경우 ValueError를 발생시킨다."""
        n = len(sample_event_type)
        df_with_pid = pd.DataFrame({
            'follow_up_years': np.ones(n),
            'INDI_DSCM_NO':    np.arange(n),  # 환자 식별자
        })
        with pytest.raises(ValueError, match="금지 컬럼"):
            cv.export_csv_for_r(
                df_with_pid, sample_event_type, 'test',
                covars=['INDI_DSCM_NO'], output_dir=tmp_path
            )

    def test_export_csv_filters_nonpositive_time(self, cv, sample_event_type, tmp_path):
        """time <= 0 행이 제거된다."""
        n = len(sample_event_type)
        df = pd.DataFrame({
            'follow_up_years': [-1.0, 0.0] + [1.0] * (n - 2),
            'age_at_index':    [50.0] * n,
            'male':            [1] * n,
        })
        csv_path = cv.export_csv_for_r(
            df, sample_event_type, 'test',
            covars=['age_at_index', 'male'], output_dir=tmp_path
        )
        result = pd.read_csv(csv_path)
        assert (result['time'] > 0).all()

    def test_export_csv_no_valid_covars_raises(self, cv, sample_event_type, tmp_path):
        """존재하지 않는 공변량만 지정하면 ValueError."""
        n = len(sample_event_type)
        df = pd.DataFrame({
            'follow_up_years': np.ones(n),
            'age_at_index':    np.ones(n),
        })
        with pytest.raises(ValueError, match="사용 가능한"):
            cv.export_csv_for_r(
                df, sample_event_type, 'test',
                covars=['is_t1dm'],  # df에 없는 컬럼
                output_dir=tmp_path
            )


# ---------------------------------------------------------------------------
# 2. generate_r_script_valid_syntax / reads_correct_csv
# ---------------------------------------------------------------------------

class TestGenerateRScript:
    def test_generate_r_script_creates_file(self, cv, tmp_path):
        """R 스크립트 파일이 생성된다."""
        dummy_csv = tmp_path / 'cv_data_test.csv'
        dummy_csv.write_text('time,event_type,age_at_index\n')
        r_path = cv.generate_r_script(dummy_csv, 'test', output_dir=tmp_path)
        assert r_path.exists()
        assert r_path.suffix == '.R'

    def test_generate_r_script_reads_correct_csv(self, cv, tmp_path):
        """R 스크립트에 CSV 경로가 포함된다."""
        dummy_csv = tmp_path / 'cv_data_dementia.csv'
        dummy_csv.write_text('time,event_type\n')
        r_path = cv.generate_r_script(dummy_csv, 'dementia', output_dir=tmp_path)
        script_text = r_path.read_text(encoding='utf-8')
        assert 'cv_data_dementia.csv' in script_text

    def test_generate_r_script_contains_crr_call(self, cv, tmp_path):
        """생성된 스크립트에 crr() 호출이 포함된다."""
        dummy_csv = tmp_path / 'cv_data_test.csv'
        dummy_csv.write_text('')
        r_path = cv.generate_r_script(dummy_csv, 'test', output_dir=tmp_path)
        script_text = r_path.read_text(encoding='utf-8')
        assert 'crr(' in script_text
        assert 'cmprsk' in script_text
        assert 'jsonlite' in script_text

    def test_generate_r_script_json_output_path(self, cv, tmp_path):
        """생성된 스크립트에 JSON 출력 경로가 포함된다."""
        dummy_csv = tmp_path / 'cv_data_mytest.csv'
        dummy_csv.write_text('')
        r_path = cv.generate_r_script(dummy_csv, 'mytest', output_dir=tmp_path)
        script_text = r_path.read_text(encoding='utf-8')
        assert 'cv_result_mytest.json' in script_text


# ---------------------------------------------------------------------------
# 3. run_r_script — R 미설치 / 타임아웃
# ---------------------------------------------------------------------------

class TestRunRScript:
    def test_run_r_script_no_r_installed(self, cv, tmp_path):
        """Rscript가 없으면 None을 반환한다."""
        dummy_r = tmp_path / 'cv_script_test.R'
        dummy_r.write_text('cat("hello")\n')
        with patch('shutil.which', return_value=None):
            result = cv.run_r_script(dummy_r)
        assert result is None

    def test_run_r_script_timeout(self, cv, tmp_path):
        """subprocess 타임아웃 발생 시 None을 반환한다."""
        import subprocess as _sp
        dummy_r = tmp_path / 'cv_script_test.R'
        dummy_r.write_text('Sys.sleep(999)\n')
        with patch('shutil.which', return_value='/usr/bin/Rscript'), \
             patch('subprocess.run', side_effect=_sp.TimeoutExpired('Rscript', 1)):
            result = cv.run_r_script(dummy_r, timeout=1)
        assert result is None

    def test_run_r_script_nonzero_return(self, cv, tmp_path):
        """R 스크립트가 비정상 종료 시 None을 반환한다."""
        import subprocess as _sp
        dummy_r = tmp_path / 'cv_script_test.R'
        dummy_r.write_text('stop("fail")\n')
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = 'Error: fail\n'
        with patch('shutil.which', return_value='/usr/bin/Rscript'), \
             patch('subprocess.run', return_value=mock_proc):
            result = cv.run_r_script(dummy_r)
        assert result is None

    def test_run_r_script_success_parses_json(self, cv, tmp_path):
        """R 성공 시 JSON을 파싱하여 dict를 반환한다."""
        import subprocess as _sp
        dummy_r = tmp_path / 'cv_script_test.R'
        dummy_r.write_text('')
        # JSON 결과 파일 미리 생성
        json_data = [
            {'covariate': 'age_at_index', 'hr': 1.25, 'ci_lower': 1.10,
             'ci_upper': 1.40, 'p_value': 0.001, 'coef': 0.22, 'se': 0.05,
             'n': 50, 'n_event': 15}
        ]
        json_file = tmp_path / 'cv_result_test.json'
        json_file.write_text(json.dumps(json_data), encoding='utf-8')

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = 'SUCCESS\n'
        mock_proc.stderr = ''
        with patch('shutil.which', return_value='/usr/bin/Rscript'), \
             patch('subprocess.run', return_value=mock_proc):
            result = cv.run_r_script(dummy_r)

        assert result is not None
        assert 'age_at_index' in result
        assert result['age_at_index']['hr'] == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# 4. compare_results — concordant / discrepant / missing_r
# ---------------------------------------------------------------------------

class TestCompareResults:
    def _make_py_summary(self, cov='age_at_index', hr=1.25, ci_lo=1.10, ci_hi=1.40, p=0.001):
        return pd.DataFrame({
            'exp(coef)':           [hr],
            'exp(coef) lower 95%': [ci_lo],
            'exp(coef) upper 95%': [ci_hi],
            'p':                   [p],
        }, index=[cov])

    def test_compare_results_concordant(self, cv):
        """HR 차이 ≤5% → concordant=True."""
        py_sum = self._make_py_summary(hr=1.25)
        r_res = {'age_at_index': {'hr': 1.26, 'ci_lower': 1.11, 'ci_upper': 1.41, 'p_value': 0.001}}
        df = cv.compare_results(py_sum, r_res, tolerance_pct=5.0)
        assert df.loc[0, 'concordant'] == True
        assert df.loc[0, 'hr_diff_pct'] < 5.0

    def test_compare_results_discrepant(self, cv):
        """HR 차이 >5% → concordant=False, note에 차이 표시."""
        py_sum = self._make_py_summary(hr=1.25)
        r_res = {'age_at_index': {'hr': 1.50, 'ci_lower': 1.30, 'ci_upper': 1.70, 'p_value': 0.001}}
        df = cv.compare_results(py_sum, r_res, tolerance_pct=5.0)
        assert df.loc[0, 'concordant'] == False
        assert '차이' in df.loc[0, 'note']

    def test_compare_results_missing_r(self, cv):
        """r_results=None → note='R 결과 없음'."""
        py_sum = self._make_py_summary()
        df = cv.compare_results(py_sum, r_results=None)
        assert df.loc[0, 'note'] == 'R 결과 없음'
        assert df.loc[0, 'concordant'] == False

    def test_compare_results_empty_python_summary(self, cv):
        """python_summary=None → R 결과만 표시된다."""
        r_res = {'male': {'hr': 0.85, 'ci_lower': 0.70, 'ci_upper': 1.00, 'p_value': 0.05}}
        df = cv.compare_results(None, r_res)
        assert len(df) == 1
        assert df.loc[0, 'covariate'] == 'male'
        assert np.isnan(df.loc[0, 'py_hr'])


# ---------------------------------------------------------------------------
# 5. validation_status
# ---------------------------------------------------------------------------

class TestValidationStatus:
    def test_r_not_available(self, cv):
        df = pd.DataFrame({'concordant': [True]})
        assert CrossValidator.validation_status(df, r_available=False) == 'R_NOT_AVAILABLE'

    def test_validated(self, cv):
        df = pd.DataFrame({'concordant': [True, True]})
        assert CrossValidator.validation_status(df, r_available=True) == 'VALIDATED'

    def test_discrepant(self, cv):
        df = pd.DataFrame({'concordant': [True, False]})
        assert CrossValidator.validation_status(df, r_available=True) == 'DISCREPANT'

    def test_empty_df(self, cv):
        df = pd.DataFrame()
        assert CrossValidator.validation_status(df, r_available=True) == 'NO_COMPARISON_DATA'


# ---------------------------------------------------------------------------
# 6. cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_cleanup_removes_files(self, cv, tmp_path):
        """존재하는 파일이 삭제된다."""
        f1 = tmp_path / 'a.csv'
        f2 = tmp_path / 'b.R'
        f1.write_text('x')
        f2.write_text('y')
        cv.cleanup_temp_files([f1, f2])
        assert not f1.exists()
        assert not f2.exists()

    def test_cleanup_missing_files_no_error(self, cv, tmp_path):
        """존재하지 않는 파일도 ignore_errors=True이면 예외 없이 통과."""
        missing = tmp_path / 'not_here.csv'
        cv.cleanup_temp_files([missing], ignore_errors=True)  # should not raise

    def test_cleanup_temp_dir_removes_directory(self, cv, tmp_path):
        """cleanup_temp_dir이 디렉토리 전체를 삭제한다."""
        subdir = tmp_path / 'cv_temp'
        subdir.mkdir()
        (subdir / 'file.csv').write_text('data')
        cv.cleanup_temp_dir(subdir)
        assert not subdir.exists()


# ---------------------------------------------------------------------------
# 7. 통합: full pipeline without R
# ---------------------------------------------------------------------------

class TestFullPipelineWithoutR:
    def test_full_pipeline_without_r(self, cv, sample_df_cr, sample_event_type, tmp_path):
        """Rscript 미설치 환경에서 파이프라인 전체가 에러 없이 완료된다."""
        py_summary = pd.DataFrame({
            'exp(coef)':           [1.25],
            'exp(coef) lower 95%': [1.10],
            'exp(coef) upper 95%': [1.40],
            'p':                   [0.001],
        }, index=['age_at_index'])

        with patch('shutil.which', return_value=None):
            csv_path = cv.export_csv_for_r(
                sample_df_cr, sample_event_type, 'dementia', output_dir=tmp_path
            )
            r_path = cv.generate_r_script(csv_path, 'dementia', output_dir=tmp_path)
            r_result = cv.run_r_script(r_path)          # R 없음 → None
            cmp_df   = cv.compare_results(py_summary, r_result)
            status   = CrossValidator.validation_status(cmp_df, r_available=False)

        assert csv_path.exists()
        assert r_path.exists()
        assert r_result is None
        assert status == 'R_NOT_AVAILABLE'
        assert len(cmp_df) >= 1
        cv.cleanup_temp_files([csv_path, r_path])
        assert not csv_path.exists()


# ---------------------------------------------------------------------------
# 8. 통합: full pipeline with mock R
# ---------------------------------------------------------------------------

class TestFullPipelineWithMockR:
    def test_full_pipeline_with_mock_r(self, cv, sample_df_cr, sample_event_type, tmp_path):
        """Mock R 실행 환경에서 VALIDATED 또는 DISCREPANT 상태를 올바르게 반환한다."""
        py_summary = pd.DataFrame({
            'exp(coef)':           [1.25, 0.90],
            'exp(coef) lower 95%': [1.10, 0.75],
            'exp(coef) upper 95%': [1.40, 1.05],
            'p':                   [0.001, 0.12],
        }, index=['age_at_index', 'male'])

        # R 결과와 Python 결과가 허용 범위 이내
        mock_r_json = [
            {'covariate': 'age_at_index', 'hr': 1.26, 'ci_lower': 1.11,
             'ci_upper': 1.41, 'p_value': 0.001, 'coef': 0.23, 'se': 0.06,
             'n': 50, 'n_event': 15},
            {'covariate': 'male', 'hr': 0.91, 'ci_lower': 0.76,
             'ci_upper': 1.06, 'p_value': 0.12, 'coef': -0.09, 'se': 0.08,
             'n': 50, 'n_event': 15},
        ]

        with patch('shutil.which', return_value=None):
            csv_path = cv.export_csv_for_r(
                sample_df_cr, sample_event_type, 'cv_mock', output_dir=tmp_path
            )
            r_path = cv.generate_r_script(csv_path, 'cv_mock', output_dir=tmp_path)

        # JSON 결과 파일 미리 생성하여 R 실행 모킹
        json_file = tmp_path / 'cv_result_cv_mock.json'
        json_file.write_text(json.dumps(mock_r_json), encoding='utf-8')

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = 'SUCCESS\n'
        mock_proc.stderr = ''
        with patch('shutil.which', return_value='/usr/bin/Rscript'), \
             patch('subprocess.run', return_value=mock_proc):
            r_result = cv.run_r_script(r_path)

        assert r_result is not None
        cmp_df = cv.compare_results(py_summary, r_result, tolerance_pct=5.0)
        status = CrossValidator.validation_status(cmp_df, r_available=True)

        assert status == 'VALIDATED'
        assert cmp_df['concordant'].all()
