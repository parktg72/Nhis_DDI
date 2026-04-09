# -*- mode: python ; coding: utf-8 -*-
# 이 파일은 build.bat 실행 시 자동 재생성됩니다.
# 직접 실행: pyinstaller NHIS_YOD_DM_Analyzer.spec
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = [
    # PyQt5
    'PyQt5.sip', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
    # 프로젝트 모듈 (모두 명시 — 누락 시 ImportError)
    'main_app', 'tabs', 'config', 'db_connector',
    'cohort_builder', 'statistical_analysis', 'analysis_runner',
    'analysis_checkpoint', 'visualization', 'results_exporter',
    'memory_manager', 'gpu_accelerator', 'utils', 'nhis_schema',
    'variable_generator',
    # DuckDB / 데이터 처리
    'duckdb', 'pyreadstat',
    'pyarrow', 'pyarrow.parquet', 'pyarrow.lib',
    # 생존분석
    'lifelines', 'lifelines.statistics', 'lifelines.fitters',
    'lifelines.fitters.coxph_fitter', 'lifelines.fitters.kaplan_meier_fitter',
    'lifelines.utils',
    'formulaic', 'autograd', 'autograd_gamma',
    # sklearn
    'sklearn', 'sklearn.linear_model', 'sklearn.linear_model._logistic',
    'sklearn.neighbors', 'sklearn.neighbors._ball_tree', 'sklearn.neighbors._kd_tree',
    'sklearn.utils._typedefs', 'sklearn.utils._param_validation',
    # scipy
    'scipy.stats', 'scipy.linalg', 'scipy.special', 'scipy.sparse',
    'scipy.optimize', 'scipy.integrate', 'scipy.interpolate',
    # matplotlib
    'matplotlib', 'matplotlib.backends.backend_agg', 'matplotlib.backends.backend_pdf',
    'matplotlib.figure', 'matplotlib.patches', 'matplotlib.font_manager',
    # pandas / openpyxl
    'pandas', 'pandas.io.formats.excel', 'pandas.io.excel._openpyxl',
    'openpyxl', 'openpyxl.workbook', 'openpyxl.styles',
    'openpyxl.styles.differential', 'openpyxl.cell',
    'openpyxl.utils', 'openpyxl.utils.dataframe',
    # 기타
    'numpy', 'psutil', 'win32timezone',
    # SAP HANA (선택)
    'hdbcli', 'hdbcli.dbapi',
]

for pkg in [
    'lifelines', 'duckdb', 'sklearn', 'scipy', 'formulaic',
    'pyreadstat', 'matplotlib', 'pandas', 'openpyxl', 'numpy',
    'psutil', 'autograd', 'hdbcli', 'pyarrow',
]:
    tmp_ret = collect_all(pkg)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['main_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NHIS_YOD_DM_Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NHIS_YOD_DM_Analyzer',
)
