from __future__ import annotations

from pathlib import Path


def test_verify_smoke_dl_generates_bundle_reloads_and_checks_predict(tmp_path) -> None:
    from scripts.ops.validate_dl_bundle import VerificationReport
    from scripts.ops.verify_smoke_dl import run_verification

    calls: list[tuple[str, dict, dict[str, str]]] = []
    created: list[Path] = []
    validated: list[Path] = []

    def fake_create_bundle(bundle_dir) -> None:
        bundle_path = Path(bundle_dir)
        bundle_path.mkdir(parents=True)
        (bundle_path / "MANIFEST.json").write_text("{}", encoding="utf-8")
        created.append(bundle_path)

    def fake_post(url: str, payload: dict, headers: dict[str, str]) -> dict:
        calls.append((url, payload, headers))
        if url.endswith("/admin/reload/dl"):
            return {
                "status": "ok",
                "dl_loaded": True,
                "dl_bundle_run_id": "smoke-deploy",
            }
        if url.endswith("/predict"):
            return {
                "patient_id": payload["patient_id"],
                "dl_prediction": {
                    "predicted_label": "high",
                    "score": 0.75,
                },
            }
        raise AssertionError(f"unexpected url: {url}")

    def fake_validate_bundle(bundle_dir):
        validated.append(Path(bundle_dir))
        return VerificationReport(bundle_dir=Path(bundle_dir))

    result = run_verification(
        base_url="http://server.local:8000/",
        admin_key="secret",
        model_dir=tmp_path / "models",
        http_post=fake_post,
        create_bundle=fake_create_bundle,
        bundle_validator=fake_validate_bundle,
        require_dl_prediction=True,
    )

    assert result.ok is True
    assert result.bundle_dir == tmp_path / "models" / "dl" / "smoke"
    assert created == [tmp_path / "models" / "dl" / "smoke"]
    assert validated == [tmp_path / "models" / "dl" / "smoke"]
    assert (result.bundle_dir / "MANIFEST.json").exists()
    assert result.dl_prediction_present is True
    assert calls[0] == (
        "http://server.local:8000/admin/reload/dl",
        {"bundle_dir": str(tmp_path / "models" / "dl" / "smoke")},
        {"X-Admin-Key": "secret"},
    )
    assert calls[1][0] == "http://server.local:8000/predict"
    assert calls[1][1]["drugs"][0]["edi_code"] == "D1"


def test_verify_smoke_dl_requires_admin_key(tmp_path) -> None:
    from scripts.ops.verify_smoke_dl import run_verification

    result = run_verification(
        base_url="http://server.local:8000",
        admin_key="",
        model_dir=tmp_path / "models",
        http_post=lambda *_args, **_kwargs: {},
    )

    assert result.ok is False
    assert "ADMIN_API_KEY" in result.message


def test_verify_smoke_dl_stops_before_reload_when_bundle_validation_fails(tmp_path) -> None:
    from scripts.ops.validate_dl_bundle import VerificationReport
    from scripts.ops.verify_smoke_dl import run_verification

    calls: list[str] = []

    def fake_create_bundle(bundle_dir) -> None:
        calls.append("create")
        Path(bundle_dir).mkdir(parents=True)

    def fake_validate_bundle(bundle_dir):
        calls.append(f"validate:{Path(bundle_dir).name}")
        return VerificationReport(
            bundle_dir=Path(bundle_dir),
            errors=["model_config input_dim must be positive"],
        )

    def fake_post(_url: str, _payload: dict, _headers: dict[str, str]) -> dict:
        calls.append("http")
        return {}

    result = run_verification(
        base_url="http://server.local:8000",
        admin_key="secret",
        model_dir=tmp_path / "models",
        http_post=fake_post,
        create_bundle=fake_create_bundle,
        bundle_validator=fake_validate_bundle,
    )

    assert result.ok is False
    assert "bundle validation failed" in result.message
    assert "input_dim" in result.message
    assert calls == ["create", "validate:smoke"]


def test_verify_smoke_dl_can_warn_when_dl_prediction_is_absent(tmp_path) -> None:
    from scripts.ops.validate_dl_bundle import VerificationReport
    from scripts.ops.verify_smoke_dl import run_verification

    def fake_post(url: str, payload: dict, headers: dict[str, str]) -> dict:
        if url.endswith("/admin/reload/dl"):
            return {"status": "ok", "dl_loaded": True}
        return {"patient_id": payload["patient_id"], "dl_prediction": None}

    result = run_verification(
        base_url="http://server.local:8000",
        admin_key="secret",
        model_dir=Path(tmp_path / "models"),
        http_post=fake_post,
        create_bundle=lambda bundle_dir: Path(bundle_dir).mkdir(parents=True),
        bundle_validator=lambda bundle_dir: VerificationReport(bundle_dir=Path(bundle_dir)),
        require_dl_prediction=False,
    )

    assert result.ok is True
    assert result.dl_prediction_present is False
    assert "history provider" in result.message
