"""FEATURE_SCHEMA_LENIENT escape hatch sunset 회귀 가드 — Codex 2026-05-07 #6.

배경: FEATURE_SCHEMA_LENIENT=1 은 silent 0.0 drift 회피용 escape hatch. 영구
활성 시 학습/서빙 schema 정렬이 무한 미뤄질 위험. Codex 합의 design (A + 코드
default deadline):
  - env FEATURE_SCHEMA_LENIENT_SUNSET_DATE=YYYY-MM-DD
  - 미설정 시 코드 default (현재 2026-08-01)
  - today >= sunset → lenient 차단 (strict 강제), env=1 이어도 무시
  - invalid env date → 안전 측 차단 (운영 escape hatch 정책)
  - dup_efmdc allowlist sunset 은 별도 개념 (본 PR 범위 밖)

Codex 권고 5 시나리오 그대로:
  1. sunset 전 + LENIENT=1 → ok=True (lenient 효력)
  2. sunset 당일 (>=) → 차단
  3. sunset 후 + LENIENT=1 → ok=False
  4. invalid env date → 차단 (안전)
  5. LENIENT unset → strict 동작 그대로
"""
from __future__ import annotations

from datetime import date

from serving.predictor import (
    _FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT,
    _is_feature_schema_lenient_allowed,
    _validate_feature_schema,
)

# ─── _is_feature_schema_lenient_allowed 단위 ─────────────────────────────────


class TestSunsetHelper:

    def test_default_deadline_is_2026_08_01(self):
        """코드 default deadline 회귀 가드 (Codex 합의 값)."""
        assert _FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT == date(2026, 8, 1)

    def test_today_before_default_sunset_allows_lenient(self):
        """env 미설정 + default 이전 → 허용."""
        before = _FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT.replace(year=2026, month=7, day=31)
        assert _is_feature_schema_lenient_allowed(today=before) is True

    def test_today_equal_default_sunset_blocks_lenient(self):
        """today == sunset → 차단 (Codex 합의: today >= sunset_date 부터 차단)."""
        assert _is_feature_schema_lenient_allowed(
            today=_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT
        ) is False

    def test_today_after_default_sunset_blocks_lenient(self):
        """default 지난 날짜 → 차단."""
        after = date(2026, 12, 31)
        assert _is_feature_schema_lenient_allowed(today=after) is False

    def test_env_override_allows_extension(self, monkeypatch):
        """env 로 sunset 연장 → 그 날짜까지 허용."""
        monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "2027-12-31")
        assert _is_feature_schema_lenient_allowed(today=date(2027, 6, 1)) is True
        assert _is_feature_schema_lenient_allowed(today=date(2027, 12, 31)) is False

    def test_invalid_env_date_blocks_lenient(self, monkeypatch):
        """invalid env date → 안전 측 차단 (Codex: permissive 해석 금지)."""
        for invalid in ("not-a-date", "2026-13-40", "06/15/2026", ""):
            if invalid == "":
                # 빈 문자열은 default 으로 fallback 되므로 별도 케이스
                continue
            monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", invalid)
            assert _is_feature_schema_lenient_allowed(today=date(2026, 1, 1)) is False, (
                f"invalid env date={invalid!r} → 안전 차단 예상"
            )

    def test_noncanonical_env_date_blocks_lenient(self, monkeypatch):
        """zero padding 없는 env date → invalid 로 간주해 안전 차단."""
        monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "2027-1-1")
        assert _is_feature_schema_lenient_allowed(today=date(2026, 1, 1)) is False

    def test_empty_env_uses_default(self, monkeypatch):
        """env 빈 문자열 → default deadline 사용."""
        monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "")
        assert _is_feature_schema_lenient_allowed(
            today=date(2026, 7, 1)
        ) is True
        assert _is_feature_schema_lenient_allowed(
            today=_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT
        ) is False


# ─── _validate_feature_schema 통합 — Codex 권고 5 시나리오 ───────────────────


class TestValidateFeatureSchemaSunset:
    """sunset 적용 후 _validate_feature_schema 동작 검증."""

    UNKNOWN_FEATURES = ["drug_count", "fake_xyz_unknown"]

    def test_sunset_before_lenient_active_passes(self, monkeypatch):
        """1) sunset 전 + LENIENT=1 → unknown ok=True."""
        monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
        # default sunset 2026-08-01 이전 → 자연스럽게 lenient 효력
        # (직접 today 주입 안 됨 — _is_feature_schema_lenient_allowed 가 default 호출)
        # 대신 env override 로 미래 sunset 강제
        monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "2099-12-31")
        missing, ok = _validate_feature_schema(self.UNKNOWN_FEATURES, "test")
        assert ok is True, "sunset 전 + LENIENT=1 → lenient 효력으로 통과"
        assert "fake_xyz_unknown" in missing

    def test_sunset_today_blocks(self, monkeypatch):
        """2) sunset 당일 (today >= sunset) → LENIENT=1 이어도 차단."""
        monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
        # env 로 과거 날짜 sunset → 항상 차단
        monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "2020-01-01")
        missing, ok = _validate_feature_schema(self.UNKNOWN_FEATURES, "test")
        assert ok is False, "sunset 통과 후 LENIENT=1 무시되고 strict 강제"

    def test_sunset_after_blocks(self, monkeypatch):
        """3) sunset 후 + LENIENT=1 → ok=False (escape hatch 무력화)."""
        monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
        monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "2020-01-01")
        missing, ok = _validate_feature_schema(self.UNKNOWN_FEATURES, "test")
        assert ok is False
        assert "fake_xyz_unknown" in missing

    def test_invalid_sunset_env_blocks(self, monkeypatch):
        """4) env date invalid → 안전 차단 (escape hatch 무력화)."""
        monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
        monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "garbage-date")
        missing, ok = _validate_feature_schema(self.UNKNOWN_FEATURES, "test")
        assert ok is False, "invalid sunset env → 안전 차단 (Codex 정책)"

    def test_lenient_unset_keeps_strict(self, monkeypatch):
        """5) LENIENT unset → 기존 strict 동작 유지 (sunset 무관)."""
        monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
        # sunset env 도 미설정 → default 사용 (현 시점에선 default 안)
        monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)
        missing, ok = _validate_feature_schema(self.UNKNOWN_FEATURES, "test")
        assert ok is False, "LENIENT 미설정이면 sunset 무관하게 strict"

    def test_dup_efmdc_allowlist_unaffected_by_sunset(self, monkeypatch):
        """dup_efmdc allowlist sunset 은 본 PR 범위 밖 — LENIENT sunset 영향 없음."""
        monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
        monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "2020-01-01")
        # dup_efmdc 만 추가된 모델은 strict + sunset 통과 후에도 OK
        # (allowlist 가 별도 개념 — Codex 합의)
        missing, ok = _validate_feature_schema(["drug_count", "dup_efmdc"], "test")
        assert ok is True
        assert missing == []
