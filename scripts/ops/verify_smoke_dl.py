"""Verify smoke DL bundle reload and predict over an already-running API."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Sequence
from urllib import error, request

from scripts.datasets.smoke_dl_bundle import create_smoke_dl_bundle
from scripts.ops.validate_dl_bundle import VerificationReport, validate_bundle

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_RUN_ID = "smoke-deploy"
DEFAULT_SCHEMA_VERSION = "dl.v1.smoke"

HttpPost = Callable[[str, dict, dict[str, str]], dict]
BundleCreator = Callable[[str | Path], object]
BundleValidator = Callable[[str | Path], VerificationReport]


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    message: str
    bundle_dir: Path
    reload_response: dict | None = None
    predict_response: dict | None = None
    dl_prediction_present: bool = False


def _post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            **headers,
        },
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {url}: {detail}") from e
    except error.URLError as e:
        raise RuntimeError(f"HTTP request failed {url}: {e.reason}") from e
    return json.loads(raw) if raw else {}


def _predict_payload() -> dict:
    return {
        "patient_id": "SMOKE-P001",
        "reference_date": date.today().isoformat(),
        "drugs": [
            {
                "edi_code": "D1",
                "drug_name": "Smoke D1",
                "total_days": 7,
            },
            {
                "edi_code": "D2",
                "drug_name": "Smoke D2",
                "total_days": 7,
            },
        ],
    }


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _default_model_dir() -> Path:
    raw = os.environ.get("MODEL_DIR")
    return Path(raw) if raw else Path.cwd() / "models"


def _create_default_smoke_bundle(bundle_dir: str | Path) -> object:
    return create_smoke_dl_bundle(
        bundle_dir,
        run_id=DEFAULT_RUN_ID,
        schema_version=DEFAULT_SCHEMA_VERSION,
    )


def run_verification(
    *,
    base_url: str = DEFAULT_BASE_URL,
    admin_key: str | None = None,
    model_dir: str | Path | None = None,
    bundle_dir: str | Path | None = None,
    http_post: HttpPost = _post_json,
    create_bundle: BundleCreator = _create_default_smoke_bundle,
    bundle_validator: BundleValidator | None = validate_bundle,
    require_dl_prediction: bool = False,
) -> VerificationResult:
    """Generate a smoke bundle, reload it over HTTP, then call /predict."""
    root_model_dir = Path(model_dir) if model_dir is not None else _default_model_dir()
    target_bundle = (
        Path(bundle_dir)
        if bundle_dir is not None
        else root_model_dir / "dl" / "smoke"
    )
    if not admin_key:
        return VerificationResult(
            ok=False,
            message="ADMIN_API_KEY is required; pass --admin-key or set ADMIN_API_KEY.",
            bundle_dir=target_bundle,
        )

    try:
        create_bundle(target_bundle)
        if bundle_validator is not None:
            report = bundle_validator(target_bundle)
            if not report.ok:
                return VerificationResult(
                    ok=False,
                    message=f"bundle validation failed: {report.errors}",
                    bundle_dir=target_bundle,
                )
        reload_response = http_post(
            _join_url(base_url, "/admin/reload/dl"),
            {"bundle_dir": str(target_bundle)},
            {"X-Admin-Key": admin_key},
        )
        if reload_response.get("status") != "ok" or reload_response.get("dl_loaded") is not True:
            return VerificationResult(
                ok=False,
                message=f"DL reload did not report success: {reload_response}",
                bundle_dir=target_bundle,
                reload_response=reload_response,
            )

        predict_response = http_post(
            _join_url(base_url, "/predict"),
            _predict_payload(),
            {},
        )
    except Exception as e:
        return VerificationResult(
            ok=False,
            message=str(e),
            bundle_dir=target_bundle,
        )

    dl_prediction_present = bool(predict_response.get("dl_prediction"))
    if require_dl_prediction and not dl_prediction_present:
        return VerificationResult(
            ok=False,
            message=(
                "predict response has no dl_prediction; verify that a HANA "
                "history provider is configured for serving."
            ),
            bundle_dir=target_bundle,
            reload_response=reload_response,
            predict_response=predict_response,
            dl_prediction_present=False,
        )

    message = "smoke DL reload and /predict verification completed"
    if not dl_prediction_present:
        message += "; /predict returned no dl_prediction, likely no HANA history provider is configured"
    return VerificationResult(
        ok=True,
        message=message,
        bundle_dir=target_bundle,
        reload_response=reload_response,
        predict_response=predict_response,
        dl_prediction_present=dl_prediction_present,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a smoke DL bundle, hot-swap it, and call /predict.",
    )
    parser.add_argument(
        "base_url",
        nargs="?",
        default=os.environ.get("DDI_API_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--admin-key", default=os.environ.get("ADMIN_API_KEY", ""))
    parser.add_argument("--model-dir", default=str(_default_model_dir()))
    parser.add_argument("--bundle-dir", default="")
    parser.add_argument(
        "--require-dl-prediction",
        action="store_true",
        help="Fail if /predict does not include dl_prediction.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip local semantic validation before /admin/reload/dl.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_verification(
        base_url=args.base_url,
        admin_key=args.admin_key,
        model_dir=args.model_dir,
        bundle_dir=args.bundle_dir or None,
        bundle_validator=None if args.skip_validation else validate_bundle,
        require_dl_prediction=args.require_dl_prediction,
    )
    print(result.message)
    print(f"bundle_dir={result.bundle_dir}")
    if result.reload_response is not None:
        print(f"reload={json.dumps(result.reload_response, ensure_ascii=False, sort_keys=True)}")
    if result.predict_response is not None:
        print(
            "predict="
            f"{json.dumps(result.predict_response, ensure_ascii=False, sort_keys=True)}"
        )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
