"""Windows ML/DL 학습 패키지 다운로드·설치 계약 회귀 테스트."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read_rel(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_hana_app_requirements_include_phase3_training_dependencies():
    """hana_app 단독 설치도 Page3 Phase3 선택 모델을 import 가능하게 해야 한다."""
    text = _read_rel("hana_app/requirements.txt").lower()

    assert "torch" in text
    assert "pytorch-tabnet" in text


def test_download_all_fetches_python312_cuda_dl_wheel_set():
    """통합 다운로드가 Python 3.12 CUDA/PyG DL wheel set을 누락하면 폐쇄망 DL이 실패한다."""
    text = _read_rel("download_all.bat")

    assert "download_cuda_cu126.bat" in text
    assert "%%V" in text and "312" in text


def test_install_312_installs_and_verifies_phase3_dl_dependencies():
    """Python 3.12 설치 스크립트가 Page3 Phase3 DL 학습 의존성을 명시 설치·검증해야 한다."""
    text = _read_rel("install_312.bat")

    assert "DDI_REQUIRE_PHASE3_DL" in text
    assert "torch pytorch-tabnet" in text
    assert "pytorch_tabnet" in text
    assert "DDI_REQUIRE_PHASE3_DL=1 이므로 설치 검증 실패" in text


def test_install_all_installs_and_verifies_phase3_dl_dependencies():
    """통합 설치 스크립트도 Phase3 DL 학습 의존성을 명시 설치·검증해야 한다."""
    text = _read_rel("install_all.bat")

    assert "DDI_REQUIRE_PHASE3_DL" in text
    assert "torch pytorch-tabnet" in text
    assert "pytorch_tabnet" in text
    assert "DDI_REQUIRE_PHASE3_DL=1 이므로 설치 검증 실패" in text
    assert "torch.cuda.is_available()" in text
    assert "torch_geometric, pyg_lib, torch_scatter, torch_sparse, torch_cluster" in text
    assert "DDI_REQUIRE_CUDA_DL=1 이므로 설치 검증 실패" in text
