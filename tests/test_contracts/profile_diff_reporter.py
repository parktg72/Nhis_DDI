"""Report cross-profile feature differences without modifying them.

This adapter is test-only and is never imported by production runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ProfileDiff:
    profile_a: str
    profile_b: str
    only_in_a: frozenset[str]
    only_in_b: frozenset[str]
    shared: frozenset[str]


class ProfileDiffReporter:
    """Register profile feature snapshots and report their set differences."""

    def __init__(self) -> None:
        self._profiles: dict[str, frozenset[str]] = {}

    def register(self, name: str, features: Iterable[str]) -> None:
        self._profiles[name] = frozenset(features)

    def diff(self, profile_a: str, profile_b: str) -> ProfileDiff:
        a = self._profiles[profile_a]
        b = self._profiles[profile_b]
        return ProfileDiff(
            profile_a=profile_a,
            profile_b=profile_b,
            only_in_a=a - b,
            only_in_b=b - a,
            shared=a & b,
        )
