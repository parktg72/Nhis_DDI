from __future__ import annotations

import logging


def test_build_model_warns_when_using_pseudo_dl_model(monkeypatch, caplog):
    from hana_app.core import ml_runner
    from hana_app.core import phase3_models

    calls = []

    def fake_build_phase3_model(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return {"model_name": model_name}

    monkeypatch.setattr(phase3_models, "build_phase3_model", fake_build_phase3_model)

    with caplog.at_level(logging.WARNING, logger=ml_runner.logger.name):
        model = ml_runner._build_model("gnn", target="risk_binary", params={"lr": 0.01})

    assert model == {"model_name": "gnn"}
    assert calls == [
        (
            "gnn",
            {
                "n_classes": 2,
                "params": {"lr": 0.01},
                "use_gpu": False,
                "n_jobs": -1,
            },
        )
    ]
    assert any(
        "not the operational DL inference path" in record.message
        for record in caplog.records
    )


def test_build_model_does_not_warn_for_tabnet(monkeypatch, caplog):
    from hana_app.core import ml_runner
    from hana_app.core import phase3_models

    monkeypatch.setattr(
        phase3_models,
        "build_phase3_model",
        lambda model_name, **kwargs: {"model_name": model_name},
    )

    with caplog.at_level(logging.WARNING, logger=ml_runner.logger.name):
        model = ml_runner._build_model("tabnet", target="risk_binary", params={})

    assert model == {"model_name": "tabnet"}
    assert not any(
        "not the operational DL inference path" in record.message
        for record in caplog.records
    )
