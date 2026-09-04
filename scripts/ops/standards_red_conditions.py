#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M-3 — 임상 기준서 Red 7조건의 **측정 전용** 평가기 (개선계획 1단계 선행).

기준서(`CLINICAL_STANDARDS_v1.0.md` §4.1)가 규정한 Red 조건과 실제 코드가
1:1 이 아니다(차단 요인 B-2). 라벨 경로는 조건 1(금기)만 Red 로 두고 조건
3·6 을 즉시개입 하위 등급으로 강등했으며, 조건 2 는 별도 등급으로 옮겼고,
**조건 4·5·7 은 아예 없다.** 그래서 "조건 i 를 Red 로 되돌리면 개입 대상이
얼마가 되는가"(M1b)를 현 코드로는 셀 수 없다.

이 모듈은 **기준서 문언 그대로** 를 평가한다. 서빙·라벨 경로는 건드리지 않는다
— 1단계는 "되돌리면 얼마인가" 를 재는 것이지 되돌리는 것이 아니다. 되돌릴지는
2단계 D2(위계 재확정)의 결정이다.

**미측정을 미해당으로 읽지 않는다.** 판정은 세 값이다.

  fired         조건 성립
  not_fired     조건 불성립
  unmeasurable  입력이 없어 판정 불가 — **0 으로 세면 안 되는 값**

미측정이 나오는 자리는 둘이다.
  · 조건 7(QT) — `qt_risk_count` 는 dataclass 기본값 0 만 있고 ETL 집계에서
    대입되지 않는다(차단 요인 B-3). 코호트 피처로는 항상 판정 불가다. 규칙
    경로(이름 해소)로 잰 하루치 수치가 따로 있으나 그것은 이 모듈의 입력이 아니다.
  · 조건 6 — 연령 결측. pandas 는 결측을 NaN 으로 주고 `NaN >= 75` 는 조용히
    False 다. 그 False 는 "75세 미만" 이 아니라 "모름" 이다.

`has_high_risk_drug` 등은 기준서의 약효군 목록을 이 모듈이 다시 해석하지 않고
**기존 피처를 그대로 쓴다.** 키워드 커버리지 격차는 별도 측정(M5)의 대상이며,
여기서 다시 정의하면 두 개의 정답이 생긴다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

FIRED = "fired"
NOT_FIRED = "not_fired"
UNMEASURABLE = "unmeasurable"

_STATUSES = (FIRED, NOT_FIRED, UNMEASURABLE)

# 기준서 §4.1 고위험군(Red) 조건 — 문서에 적힌 순서 그대로.
CONDITION_IDS = (
    "STD_CONTRAINDICATED",   # 1. 금기 DDI 1건 이상
    "STD_MAJOR_3PLUS",       # 2. Major DDI 3건 이상
    "STD_TRIPLE_WHAMMY",     # 3. Triple Whammy
    "STD_10DRUG_MAJOR",      # 4. 10종 이상 + Major DDI 1건 이상
    "STD_HIGHRISK_MAJOR",    # 5. 고위험 약물 + Major DDI
    "STD_ELDERLY_ORGAN",     # 6. 75세 이상 + 5종 이상 + 신/간 위험 약물
    "STD_QT_3PLUS",          # 7. QT 연장 약물 3종 이상
)

# 강등된 것은 조건 1 을 뺀 여섯이다. 조건 1 은 지금도 Red 이므로 "되돌릴 규모"
# 에 섞으면 숫자가 부풀려진다.
DEMOTED_CONDITION_IDS = tuple(c for c in CONDITION_IDS if c != "STD_CONTRAINDICATED")

CONDITION_LABELS = {
    "STD_CONTRAINDICATED": "금기 DDI 1건 이상",
    "STD_MAJOR_3PLUS": "Major DDI 3건 이상",
    "STD_TRIPLE_WHAMMY": "Triple Whammy",
    "STD_10DRUG_MAJOR": "10종 이상 + Major DDI 1건 이상",
    "STD_HIGHRISK_MAJOR": "고위험 약물 + Major DDI",
    "STD_ELDERLY_ORGAN": "75세 이상 + 5종 이상 + 신/간 위험 약물",
    "STD_QT_3PLUS": "QT 연장 약물 3종 이상",
}


@dataclass(frozen=True)
class ConditionResult:
    condition_id: str
    ordinal: int          # 기준서 §4.1 의 순번 (1~7)
    status: str
    reason: str | None = None   # unmeasurable 일 때만 채운다


class _Missing:
    """피처가 없다는 사실 자체. None·0 과 구분한다."""


_MISSING = _Missing()


def _feature(f: Any, name: str):
    try:
        v = getattr(f, name)
    except AttributeError:
        return _MISSING
    return v


def _is_null(v) -> bool:
    if v is None:
        return True
    try:
        return bool(v != v)          # NaN != NaN
    except (TypeError, ValueError):
        # pandas 의 NA 처럼 비교 결과가 불리언이 아닌 값. 판정 불가로 둔다 —
        # 여기서 삼키면 결측이 "미해당" 으로 세어진다.
        return True


def _num(f: Any, name: str):
    """수치 피처. 없거나 결측이면 사유를 돌려준다."""
    v = _feature(f, name)
    if v is _MISSING:
        return None, f"feature_absent:{name}"
    if _is_null(v):
        return None, f"{name}_missing"
    return v, None


def _threshold(f: Any, name: str, minimum) -> "str | bool":
    """단일 임계 조건. 판정 불가면 불리언 대신 사유 문자열을 올린다."""
    v, reason = _num(f, name)
    if reason:
        return reason
    return v >= minimum


def _bool(f: Any, name: str) -> "str | bool":
    v, reason = _num(f, name)
    if reason:
        return reason
    return bool(v)


def _either(*parts):
    """논리합. 하나라도 판정 불가면 합 전체가 판정 불가다."""
    reasons = [p for p in parts if isinstance(p, str)]
    if reasons:
        return reasons[0]
    return any(parts)


def _resolve(*parts):
    """부분 판정을 모은다. 하나라도 사유면 미측정, 아니면 논리곱."""
    reasons = [p for p in parts if isinstance(p, str)]
    if reasons:
        return None, reasons[0]
    return all(parts), None


def evaluate(f: Any) -> dict[str, ConditionResult]:
    """행 하나에 대해 7조건을 평가한다.

    `f` 는 FEATURE_COLS 이름의 속성을 갖는 무엇이든 된다 — `PatientFeatures`,
    `DataFrame.itertuples()` 의 행, 동일 속성을 가진 객체.
    """
    out: dict[str, ConditionResult] = {}

    def put(cid: str, ordinal: int, verdict, reason=None):
        if isinstance(verdict, str):        # 사유가 올라온 경우
            verdict, reason = None, verdict
        if verdict is None:
            out[cid] = ConditionResult(cid, ordinal, UNMEASURABLE, reason)
        else:
            out[cid] = ConditionResult(cid, ordinal, FIRED if verdict else NOT_FIRED)

    put("STD_CONTRAINDICATED", 1, _threshold(f, "ddi_contraindicated", 1))
    put("STD_MAJOR_3PLUS", 2, _threshold(f, "ddi_major", 3))
    put("STD_TRIPLE_WHAMMY", 3, _bool(f, "triple_whammy"))

    verdict, reason = _resolve(
        _threshold(f, "drug_count", 10), _threshold(f, "ddi_major", 1),
    )
    put("STD_10DRUG_MAJOR", 4, verdict, reason)

    verdict, reason = _resolve(
        _bool(f, "has_high_risk_drug"), _threshold(f, "ddi_major", 1),
    )
    put("STD_HIGHRISK_MAJOR", 5, verdict, reason)

    # 신/간은 OR 이지만, 한쪽을 못 읽으면 **OR 전체가 판정 불가**다. 읽은 쪽이
    # False 라고 해서 "위험약 없음" 이 되지 않는다 — 못 읽은 쪽에 있었을 수 있다.
    organ = _either(_bool(f, "has_renal_risk_drug"), _bool(f, "has_hepatic_risk_drug"))
    verdict, reason = _resolve(
        _threshold(f, "age", 75), _threshold(f, "drug_count", 5), organ,
    )
    put("STD_ELDERLY_ORGAN", 6, verdict, reason)

    # 조건 7 은 코호트 피처로 판정할 수 없다. 값이 들어 있어도 신뢰하지 않는다 —
    # ETL 이 대입하지 않으므로 0 이 아닌 값은 출처가 불분명하다.
    put("STD_QT_3PLUS", 7, None, "qt_risk_count_never_assigned")

    return out


def summarize(rows: Iterable[Any]) -> dict:
    """코호트 집계. **모든 줄에 분모를 함께 낸다** (개선계획 S3.5).

    `any_demoted` 는 강등 6조건의 합성이며 미측정을 미해당보다 앞세운다 —
    하나라도 발화하면 발화, 아니면 미측정이 있으면 미측정, 둘 다 아니면 미해당.
    """
    counts = {cid: dict.fromkeys(_STATUSES, 0) for cid in CONDITION_IDS}
    any_demoted = dict.fromkeys(_STATUSES, 0)
    reasons: dict[str, dict[str, int]] = {cid: {} for cid in CONDITION_IDS}
    n = 0

    for row in rows:
        n += 1
        ev = evaluate(row)
        for cid, res in ev.items():
            counts[cid][res.status] += 1
            if res.status == UNMEASURABLE and res.reason:
                reasons[cid][res.reason] = reasons[cid].get(res.reason, 0) + 1
        demoted = [ev[cid].status for cid in DEMOTED_CONDITION_IDS]
        if FIRED in demoted:
            any_demoted[FIRED] += 1
        elif UNMEASURABLE in demoted:
            any_demoted[UNMEASURABLE] += 1
        else:
            any_demoted[NOT_FIRED] += 1

    return {
        "denominator": n,
        "conditions": counts,
        "unmeasurable_reasons": reasons,
        "any_demoted": any_demoted,
    }
