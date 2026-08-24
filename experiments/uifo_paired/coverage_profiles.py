"""Immutable structural specifications for sealed H100 coverage studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoverageProfileSpec:
    """Profile-bound counts and paths needed before outcome access."""

    profile: str
    panel_id: str
    panel_filename: str
    policy_id: str
    runs: int
    topologies: int
    seeds: tuple[int, int]
    session_wall_seconds: float
    worker_timeout_seconds: float
    archive_members: int
    criteria: int = 13

    @property
    def pairs(self) -> int:
        return self.runs // 2

    @property
    def panel_path(self) -> Path:
        return Path(__file__).with_name("panels") / self.panel_filename


COVERAGE_PROFILE_SPECS: dict[str, CoverageProfileSpec] = {
    "coverage-robustness-screen-v1": CoverageProfileSpec(
        profile="coverage-robustness-screen-v1",
        panel_id="coverage-robustness-v1",
        panel_filename="coverage-robustness-v1.json",
        policy_id="coverage-robustness-development-screen-v1",
        runs=48,
        topologies=12,
        seeds=(37, 41),
        session_wall_seconds=22 * 60 * 60.0,
        worker_timeout_seconds=2_100.0,
        archive_members=249,
    ),
    "coverage-triage-screen-v1": CoverageProfileSpec(
        profile="coverage-triage-screen-v1",
        panel_id="coverage-triage-v1",
        panel_filename="coverage-triage-v1.json",
        policy_id="coverage-triage-development-screen-v1",
        runs=32,
        topologies=8,
        seeds=(37, 41),
        session_wall_seconds=7 * 60 * 60.0,
        worker_timeout_seconds=1_200.0,
        archive_members=169,
        criteria=14,
    ),
}


def coverage_profile_spec(profile: object) -> CoverageProfileSpec:
    """Return one known frozen coverage specification, rejecting ambiguity."""
    if not isinstance(profile, str) or profile not in COVERAGE_PROFILE_SPECS:
        raise ValueError(f"unknown coverage study profile: {profile!r}")
    return COVERAGE_PROFILE_SPECS[profile]


def coverage_profile_names() -> frozenset[str]:
    return frozenset(COVERAGE_PROFILE_SPECS)
