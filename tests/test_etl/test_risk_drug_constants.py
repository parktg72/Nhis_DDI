"""위험 약물 상수 단일 출처 동등성 — Codex 2026-05-06 ISSUE-3.

ETL ↔ serving ↔ drug_rules.yaml 모두 같은 정의를 사용해야 함.
rules/risk_drug_constants.py 가 단일 출처. 본 테스트가 회귀 가드.

분리된 단계:
  - 본 테스트는 ETL/serving 단일화만 검증 (커밋 A)
  - rules/safety_net.py 의 9개 hardcoded list 는 yaml 의 15개와 어긋난 결함이지만
    임상 영향 동작 변경이라 별도 단계 (커밋 B) 로 처리
"""
from __future__ import annotations

from pathlib import Path

import yaml

from rules.risk_drug_constants import (
    HEPATIC_RISK_ATC_PREFIXES,
    HEPATIC_RISK_KEYWORDS,
    HIGH_RISK_ATC_PREFIXES,
    HIGH_RISK_KEYWORDS,
    RENAL_RISK_ATC_PREFIXES,
    RENAL_RISK_KEYWORDS,
)


def test_etl_imports_from_single_source():
    """ETL prescription_aggregator 가 단일 출처에서 import 받는지."""
    from scripts.etl.prescription_aggregator import (
        _HEPATIC_RISK_ATC_PREFIXES as etl_hep_atc,
    )
    from scripts.etl.prescription_aggregator import (
        _HEPATIC_RISK_KEYWORDS as etl_hep_kw,
    )
    from scripts.etl.prescription_aggregator import (
        _HIGH_RISK_ATC_PREFIXES as etl_high_atc,
    )
    from scripts.etl.prescription_aggregator import (
        _HIGH_RISK_KEYWORDS as etl_high_kw,
    )
    from scripts.etl.prescription_aggregator import (
        _RENAL_RISK_ATC_PREFIXES as etl_renal_atc,
    )
    from scripts.etl.prescription_aggregator import (
        _RENAL_RISK_KEYWORDS as etl_renal_kw,
    )
    assert etl_high_kw is HIGH_RISK_KEYWORDS, (
        "ETL HIGH keywords 가 단일 출처와 객체 동일성 깨짐 — drift 가능"
    )
    assert etl_high_atc is HIGH_RISK_ATC_PREFIXES
    assert etl_renal_kw is RENAL_RISK_KEYWORDS
    assert etl_renal_atc is RENAL_RISK_ATC_PREFIXES
    assert etl_hep_kw is HEPATIC_RISK_KEYWORDS
    assert etl_hep_atc is HEPATIC_RISK_ATC_PREFIXES


def test_serving_imports_from_single_source():
    """serving predictor 가 단일 출처에서 import 받는지."""
    from serving.predictor import (
        _HEPATIC_RISK_ATC_PREFIXES as srv_hep_atc,
    )
    from serving.predictor import (
        _HEPATIC_RISK_KEYWORDS as srv_hep_kw,
    )
    from serving.predictor import (
        _HIGH_RISK_ATC_PREFIXES as srv_high_atc,
    )
    from serving.predictor import (
        _HIGH_RISK_KEYWORDS as srv_high_kw,
    )
    from serving.predictor import (
        _RENAL_RISK_ATC_PREFIXES as srv_renal_atc,
    )
    from serving.predictor import (
        _RENAL_RISK_KEYWORDS as srv_renal_kw,
    )
    assert srv_high_kw is HIGH_RISK_KEYWORDS, (
        "serving HIGH keywords 가 단일 출처와 객체 동일성 깨짐 — drift 가능"
    )
    assert srv_high_atc is HIGH_RISK_ATC_PREFIXES
    assert srv_renal_kw is RENAL_RISK_KEYWORDS
    assert srv_renal_atc is RENAL_RISK_ATC_PREFIXES
    assert srv_hep_kw is HEPATIC_RISK_KEYWORDS
    assert srv_hep_atc is HEPATIC_RISK_ATC_PREFIXES


def test_high_risk_matches_yaml_source_of_truth():
    """rules/risk_drug_constants.HIGH_RISK_KEYWORDS 가 drug_rules.yaml :123 과 일치.

    1차 자료: config/drug_rules.yaml > high_risk_drugs.name_keywords (CLINICAL_STANDARDS_v1.0).
    """
    yaml_path = Path(__file__).resolve().parent.parent.parent / "config" / "drug_rules.yaml"
    with yaml_path.open(encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    yaml_keywords = set(rules["drug_groups"]["high_risk_drugs"]["name_keywords"])
    assert HIGH_RISK_KEYWORDS == yaml_keywords, (
        f"yaml 1차 자료와 코드 상수 drift:\n"
        f"  yaml only: {yaml_keywords - HIGH_RISK_KEYWORDS}\n"
        f"  code only: {HIGH_RISK_KEYWORDS - yaml_keywords}"
    )


def test_high_risk_keyword_count_15():
    """drug_rules.yaml :123 정의는 15개 약물. 회귀 가드."""
    assert len(HIGH_RISK_KEYWORDS) == 15, (
        f"HIGH_RISK_KEYWORDS 카운트 변경: {len(HIGH_RISK_KEYWORDS)} (예상 15) — "
        f"drug_rules.yaml :123 과 동기 확인"
    )


def test_safety_net_uses_single_source():
    """rules/safety_net._has_high_risk_drug 가 단일 출처 import 사용 (ISSUE-3b 정렬 후).

    직전까지 hardcoded 9개 list — drug_rules.yaml :123 의 15개 정의와 어긋난 결함.
    yaml 1차 자료 동기로 정정. 회귀 시 본 테스트가 hardcoded fallback 또는 부분 list
    재출현을 잡는다.
    """
    import inspect

    from rules.safety_net import SafetyNet
    src = inspect.getsource(SafetyNet._has_high_risk_drug)
    # 1) HIGH_RISK_KEYWORDS 를 import 해서 사용해야 함
    assert "HIGH_RISK_KEYWORDS" in src, (
        "safety_net._has_high_risk_drug 가 HIGH_RISK_KEYWORDS 를 사용하지 않음 — "
        "yaml 정의와 drift 가능"
    )
    # 2) 더 이상 hardcoded list literal 이 함수 내부에 없어야 함 (회귀 가드)
    assert "high_risk_keywords = [" not in src, (
        "safety_net 에 hardcoded high_risk_keywords list 가 남아있음 — drift 위험"
    )
