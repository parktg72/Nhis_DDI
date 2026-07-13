from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


PACKAGE_CMD_TEXT_FILES = [
    "download_all.bat",
    "install_all.bat",
    "install_312.bat",
    "download_pywebview.bat",
    "install_pywebview.bat",
    "packages_win/download.bat",
    "packages_win/install.bat",
    "packages_win/download_cuda_cu126.bat",
    "packages_win/requirements.txt",
    "packages_win/requirements_cuda_cu126.txt",
    "hana/download.bat",
    "hana/install.bat",
    "hana/requirements.txt",
    "hana_app/run.bat",
    "hana_app/requirements.txt",
    "generate_smoke_dl_bundle.bat",
    "verify_smoke_dl.bat",
    "run_api_smoke_dl.bat",
    "inspect_parquet_history.bat",
]

PACKAGE_CMD_BAT_FILES = [
    path for path in PACKAGE_CMD_TEXT_FILES if path.endswith(".bat")
]

PACKAGE_CMD_PYTHON_BAT_FILES = [
    "install_all.bat",
    "install_312.bat",
    "download_pywebview.bat",
    "install_pywebview.bat",
    "packages_win/download.bat",
    "packages_win/install.bat",
    "packages_win/download_cuda_cu126.bat",
    "hana/download.bat",
    "hana/install.bat",
    "hana_app/run.bat",
    "generate_smoke_dl_bundle.bat",
    "verify_smoke_dl.bat",
    "run_api_smoke_dl.bat",
    "inspect_parquet_history.bat",
]


def test_package_cmd_files_are_utf8_crlf_without_bom() -> None:
    for relative_path in PACKAGE_CMD_TEXT_FILES:
        raw = (ROOT / relative_path).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), relative_path
        raw.decode("utf-8")
        assert b"\r\n" in raw, relative_path
        assert b"\n" not in raw.replace(b"\r\n", b""), relative_path


def test_package_cmd_bat_files_select_utf8_codepage() -> None:
    for relative_path in PACKAGE_CMD_BAT_FILES:
        script = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "chcp 65001" in script, relative_path


def test_package_cmd_python_bat_files_force_python_utf8_mode() -> None:
    for relative_path in PACKAGE_CMD_PYTHON_BAT_FILES:
        script = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "PYTHONUTF8=1" in script, relative_path


def test_hana_three_month_data_gate_is_documented() -> None:
    runbook = (ROOT / "docs/ops/dl-smoke-runbook.md").read_text(encoding="utf-8")
    report = (
        ROOT / "docs/reports/2026-05-18-cuda126-dl-next-steps.md"
    ).read_text(encoding="utf-8")

    assert "3개월" in runbook
    assert "HANA DB" in runbook
    assert "학습-서빙 계약" in runbook
    assert "10.1.67.115" in runbook
    assert "30015" in runbook
    assert "ID/PW" in runbook
    assert "3개월" in report
    assert "HANA DB" in report
    assert "10.1.67.115:30015" in report


def test_cuda_requirements_pin_operational_dl_wheel_set() -> None:
    req = (ROOT / "packages_win" / "requirements_cuda_cu126.txt").read_text(
        encoding="utf-8"
    )

    expected = {
        "torch==2.11.0+cu126",
        "torch-geometric==2.7.0",
        "pyg_lib==0.6.0+pt211cu126",
        "torch_scatter==2.1.2+pt211cu126",
        "torch_sparse==0.6.18+pt211cu126",
        "torch_cluster==1.6.3+pt211cu126",
    }
    actual = {
        line.strip()
        for line in req.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert expected <= actual
    assert "torch_spline_conv" not in actual


def test_cuda_downloader_uses_official_pytorch_and_pyg_indexes() -> None:
    script = (ROOT / "packages_win" / "download_cuda_cu126.bat").read_text(
        encoding="utf-8"
    )

    assert "https://download.pytorch.org/whl/cu126" in script
    assert "https://data.pyg.org/whl/torch-2.11.0+cu126.html" in script
    assert "torch==2.11.0+cu126" in script
    assert "pyg_lib==0.6.0+pt211cu126" in script


def test_install_312_has_cuda_dl_opt_in_hard_fail() -> None:
    script = (ROOT / "install_312.bat").read_text(encoding="utf-8")

    assert "requirements_cuda_cu126.txt" in script
    assert "DDI_REQUIRE_CUDA_DL" in script
    assert "torch.cuda.is_available()" in script
    assert "torch_geometric, pyg_lib, torch_scatter, torch_sparse, torch_cluster" in script
    final_fail_block = re.search(
        r'if "%FAIL%"=="0" \([\s\S]+?\) else \(([\s\S]+?)\)\s*echo\.',
        script,
    )
    assert final_fail_block is not None
    assert "exit /b 1" in final_fail_block.group(1)


def test_generate_smoke_dl_bundle_bat_uses_hana_venv_and_model_dir_default() -> None:
    script_path = ROOT / "generate_smoke_dl_bundle.bat"
    script = script_path.read_text(encoding="utf-8")
    raw = script_path.read_bytes()

    assert "chcp 65001" in script
    assert b"\r\n" in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")
    assert "PYTHONUTF8=1" in script
    assert ".venv_hana\\Scripts\\python.exe" in script
    assert ".venv\\Scripts\\python.exe" in script
    assert "models\\dl\\smoke" in script
    assert "-m scripts.datasets.smoke_dl_bundle" in script
    assert "--schema-version dl.v1.smoke" in script
    assert "exit /b 1" in script


def test_verify_smoke_dl_bat_uses_hana_venv_and_ops_module() -> None:
    script_path = ROOT / "verify_smoke_dl.bat"
    script = script_path.read_text(encoding="utf-8")
    raw = script_path.read_bytes()

    assert "chcp 65001" in script
    assert b"\r\n" in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")
    assert "PYTHONUTF8=1" in script
    assert ".venv_hana\\Scripts\\python.exe" in script
    assert ".venv\\Scripts\\python.exe" in script
    assert "-m scripts.ops.verify_smoke_dl" in script
    assert "--require-dl-prediction" in script
    assert "--skip-validation" in script
    assert "EXTRA_ARGS" in script
    assert "ADMIN_API_KEY" in script
    assert "MODEL_DIR" in script
    assert "exit /b 1" in script


def test_run_api_smoke_dl_bat_starts_uvicorn_with_smoke_env() -> None:
    script_path = ROOT / "run_api_smoke_dl.bat"
    script = script_path.read_text(encoding="utf-8")
    raw = script_path.read_bytes()

    assert "chcp 65001" in script
    assert b"\r\n" in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")
    assert "PYTHONUTF8=1" in script
    assert "DDI_SMOKE_HISTORY_PROVIDER=1" in script
    assert "ADMIN_API_KEY" in script
    assert ".venv_hana\\Scripts\\python.exe" in script
    assert ".venv\\Scripts\\python.exe" in script
    assert "MODEL_DIR" in script
    assert "serving.main:app" in script
    assert "-m uvicorn" in script
    assert "--host 127.0.0.1" in script
    assert "--port %PORT%" in script
    assert "Ctrl+C" in script
    assert "exit /b 1" in script


def test_inspect_parquet_history_bat_uses_hana_venv_and_ops_module() -> None:
    script_path = ROOT / "inspect_parquet_history.bat"
    script = script_path.read_text(encoding="utf-8")
    raw = script_path.read_bytes()
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "chcp 65001" in script
    assert b"\r\n" in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")
    assert "PYTHONUTF8=1" in script
    assert ".venv_hana\\Scripts\\python.exe" in script
    assert ".venv\\Scripts\\python.exe" in script
    assert "-m scripts.ops.inspect_parquet_history" in script
    assert "PARQUET_PATH" in script
    assert "--patient-id" in script
    assert "exit /b 1" in script
    assert "!inspect_parquet_history.bat" in gitignore
