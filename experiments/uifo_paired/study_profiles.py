"""Frozen paid-study profiles and their predeclared decision rules."""

from __future__ import annotations

import copy
import math
import re
from datetime import UTC, datetime

from experiments.uifo_paired.optimizer_settings import settings_with_patience


STUDY_PROFILES: dict[str, dict[str, object]] = {
    "development-v2": {
        "stage": "development",
        "required_configuration": {
            "allow_cpu": False,
            "arms": ["no_prior", "semantic_prior"],
            "evaluation_chunk_size": None,
            "max_evals": None,
            "max_idle_gpu_memory_mib": 1_000,
            "max_idle_gpu_utilization_percent": 5,
            "max_time_seconds": 600.0,
            "max_session_wall_seconds": 16 * 60 * 60,
            "max_worker_failures": 2,
            "minimum_free_disk_gib": 20.0,
            "minimum_gpu_memory_mib": 75_000,
            "n_frequencies": 50,
            "optimizer_seeds": [7, 11],
            "optimizer_telemetry": None,
            "population_size": 8,
            "require_a100": True,
            "target_losses": [4.0, 1.0, 0.5, 0.0],
            "worker_timeout_seconds": 1_200.0,
        },
        "required_panel": {
            "panel_id": "development-v1",
            "topology_count": 16,
            "archive_exclusion_verified": True,
        },
        "decision_policy": {
            "policy_id": "semantic-prior-development-v2",
            "action_if_passed": "advance_to_confirmation_v1",
            "action_if_failed": "retain_no_prior_candidate",
            "minimum_semantic_prior_topology_wins": 12,
            "minimum_practical_median_loss_reduction": 0.05,
            "minimum_semantic_prior_higher_finite_feasibility_topologies": 1,
            "maximum_no_prior_higher_finite_feasibility_topologies": 0,
            "maximum_no_prior_only_seed_pairs": 0,
            "maximum_neither_finite_feasible_seed_pairs": 0,
            "maximum_topology_p90_regret": 0.5,
            "require_bootstrap_mean_ci_upper_below_zero": False,
            "require_complete_uncensored_panel": True,
            "inference_unit": "topology",
        },
    },
    "confirmation-v1": {
        "stage": "confirmation",
        "required_configuration": {
            "allow_cpu": False,
            "arms": ["no_prior", "semantic_prior"],
            "evaluation_chunk_size": None,
            "max_evals": None,
            "max_idle_gpu_memory_mib": 1_000,
            "max_idle_gpu_utilization_percent": 5,
            "max_time_seconds": 1_800.0,
            "max_session_wall_seconds": 16 * 60 * 60,
            "max_worker_failures": 2,
            "minimum_free_disk_gib": 20.0,
            "minimum_gpu_memory_mib": 75_000,
            "n_frequencies": 50,
            "optimizer_seeds": [7, 11],
            "optimizer_telemetry": None,
            "population_size": 8,
            "require_a100": True,
            "target_losses": [4.0, 1.0, 0.5, 0.0],
            "worker_timeout_seconds": 3_000.0,
        },
        "required_panel": {
            "panel_id": "confirmation-v1",
            "topology_count": 12,
            "archive_exclusion_verified": True,
        },
        "decision_policy": {
            "policy_id": "semantic-prior-confirmation-v1",
            "action_if_passed": "keep_semantic_prior_candidate",
            "action_if_failed": "retain_no_prior_candidate",
            "minimum_semantic_prior_topology_wins": 10,
            "minimum_practical_median_loss_reduction": 0.05,
            "minimum_semantic_prior_higher_finite_feasibility_topologies": 1,
            "maximum_no_prior_higher_finite_feasibility_topologies": 0,
            "maximum_no_prior_only_seed_pairs": 0,
            "maximum_neither_finite_feasible_seed_pairs": 0,
            "maximum_topology_p90_regret": 0.5,
            "require_bootstrap_mean_ci_upper_below_zero": True,
            "require_complete_uncensored_panel": True,
            "inference_unit": "topology",
        },
    },
    "restart-mechanics-v1": {
        "stage": "optimizer_mechanics",
        "requires_provider_deadline": True,
        "required_configuration": {
            "allow_cpu": False,
            "arm_optimizer_settings": {
                "no_prior_p200": settings_with_patience(200),
            },
            "arms": ["no_prior_p200"],
            "evaluation_chunk_size": None,
            "max_evals": None,
            "max_idle_gpu_memory_mib": 1_000,
            "max_idle_gpu_utilization_percent": 5,
            "max_time_seconds": 600.0,
            "max_session_wall_seconds": 1_800,
            "max_worker_failures": 1,
            "minimum_free_disk_gib": 20.0,
            "minimum_gpu_memory_mib": 75_000,
            "n_frequencies": 50,
            "optimizer_seeds": [11],
            "optimizer_telemetry": "member-v1",
            "population_size": 8,
            "provider_deadline_maximum_horizon_seconds": 8 * 60 * 60.0,
            "provider_evacuation_reserve_seconds": 1_800.0,
            "require_a100": True,
            "target_losses": [4.0, 1.0, 0.5, 0.0],
            "worker_timeout_seconds": 1_200.0,
        },
        "required_panel": {
            "panel_id": "restart-mechanics-v1",
            "topology_count": 1,
            "source_sha256": (
                "2bc42026f52c09d85625ecce8d3ce0729c1efa06d0716511ed18d9d59c9f91c6"
            ),
            "archive_exclusion_verified": True,
        },
        "decision_policy": {
            "policy_id": "restart-mechanics-v1",
            "action_if_passed": "run_restart_screen_v1",
            "action_if_failed": "retain_patience_600",
            "exclude_loss_from_inference": True,
            "maximum_worker_wall_seconds": 825.0,
            "minimum_post_restart_evaluation_rows": 1,
            "minimum_restart_rows": 1,
            "inference_unit": "mechanics_only",
        },
    },
    "restart-screen-v1": {
        "stage": "optimizer_development_screen",
        "requires_mechanics_evidence": True,
        "requires_provider_deadline": True,
        "required_configuration": {
            "allow_cpu": False,
            "arm_optimizer_settings": {
                "no_prior_p600": settings_with_patience(600),
                "no_prior_p200": settings_with_patience(200),
            },
            "arms": ["no_prior_p600", "no_prior_p200"],
            "evaluation_chunk_size": None,
            "max_evals": None,
            "max_idle_gpu_memory_mib": 1_000,
            "max_idle_gpu_utilization_percent": 5,
            "max_time_seconds": 600.0,
            "max_session_wall_seconds": int(6.5 * 60 * 60),
            "max_worker_failures": 1,
            "minimum_free_disk_gib": 20.0,
            "minimum_gpu_memory_mib": 75_000,
            "n_frequencies": 50,
            "optimizer_seeds": [19, 23],
            "optimizer_telemetry": None,
            "pair_order_policy": "alternate_topology_and_seed",
            "population_size": 8,
            "provider_deadline_maximum_horizon_seconds": 8 * 60 * 60.0,
            "provider_evacuation_reserve_seconds": 1_800.0,
            "require_a100": True,
            "target_losses": [4.0, 1.0, 0.5, 0.0],
            "worker_timeout_seconds": 1_200.0,
        },
        "required_panel": {
            "panel_id": "restart-screen-v1",
            "topology_count": 8,
            "source_sha256": (
                "dd1404e7b260c93a141b303c1a7f88f9ef02ceba03f109523708b2a8ed54b5d3"
            ),
            "archive_exclusion_verified": True,
        },
        "decision_policy": {
            "policy_id": "patience-200-development-screen-v1",
            "action_if_passed": "plan_untouched_submission_like_gate",
            "action_if_failed": "retain_patience_600",
            "maximum_topology_median_difference": -0.05,
            "maximum_topology_p90_regret": 0.5,
            "minimum_patience_200_topology_wins": 6,
            "require_all_pairs_finite_comparable": True,
            "require_both_seed_mean_differences_below_zero": True,
            "require_no_control_only_finite_feasible_pairs": True,
            "require_no_topology_lower_treatment_feasibility": True,
            "require_topology_mean_difference_below_zero": True,
            "inference_unit": "topology",
            "optimizer_seeds_are_repeated_measurements": True,
        },
    },
}


def bind_study_profile(
    profile_name: str | None, configuration: dict[str, object]
) -> dict[str, object] | None:
    """Validate an exact frozen profile and return its decision policy."""
    if profile_name is None:
        return None
    if profile_name not in STUDY_PROFILES:
        raise ValueError(f"unknown study profile: {profile_name!r}")
    profile = STUDY_PROFILES[profile_name]
    required = profile["required_configuration"]
    assert isinstance(required, dict)
    for key, expected in required.items():
        observed = configuration.get(key)
        if not _same_value(observed, expected):
            raise ValueError(
                f"study profile {profile_name!r} requires {key}={expected!r}; "
                f"observed {observed!r}"
            )

    if profile.get("requires_provider_deadline"):
        _validate_provider_deadline(configuration.get("provider_stop_utc"))
    if profile.get("requires_mechanics_evidence"):
        _validate_mechanics_evidence(configuration.get("mechanics_evidence"))

    panel = configuration.get("topology_panel")
    if not isinstance(panel, dict):
        raise ValueError(f"study profile {profile_name!r} requires a named panel")
    required_panel = profile["required_panel"]
    assert isinstance(required_panel, dict)
    for key, expected in required_panel.items():
        observed = panel.get(key)
        if not _same_value(observed, expected):
            raise ValueError(
                f"study profile {profile_name!r} requires panel {key}={expected!r}; "
                f"observed {observed!r}"
            )
    topologies = configuration.get("topologies")
    if not isinstance(topologies, list) or len(topologies) != int(
        required_panel["topology_count"]
    ):
        raise ValueError(
            f"study profile {profile_name!r} requires exactly "
            f"{required_panel['topology_count']} topology specifications"
        )

    policy = copy.deepcopy(profile["decision_policy"])
    assert isinstance(policy, dict)
    policy["stage"] = profile["stage"]
    policy["study_profile"] = profile_name
    return policy


def profile_names() -> tuple[str, ...]:
    return tuple(STUDY_PROFILES)


def _same_value(observed: object, expected: object) -> bool:
    if isinstance(expected, float) and isinstance(observed, (int, float)):
        return math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
    return observed == expected


def _validate_provider_deadline(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("study profile requires a provider_stop_utc value")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("study profile provider_stop_utc is invalid") from error
    if parsed.utcoffset() != UTC.utcoffset(None):
        raise ValueError("study profile provider_stop_utc must be UTC")


def _validate_mechanics_evidence(value: object) -> None:
    required = {
        "format_version",
        "study_profile",
        "plan_id",
        "project_revision",
        "package_sha256",
        "package_manifest_sha256",
        "record_sha256",
        "history_sha256",
        "optimizer_telemetry_sha256",
        "decision_status",
        "decision_action",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            "restart-screen-v1 requires exact authenticated mechanics evidence"
        )
    expected = {
        "format_version": 1,
        "study_profile": "restart-mechanics-v1",
        "decision_status": "passed",
        "decision_action": "run_restart_screen_v1",
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"mechanics evidence has invalid {key}")
    if not re.fullmatch(r"[0-9a-f]{16}", str(value.get("plan_id", ""))):
        raise ValueError("mechanics evidence plan_id is invalid")
    if not re.fullmatch(
        r"[0-9a-f]{40}", str(value.get("project_revision", ""))
    ):
        raise ValueError("mechanics evidence project_revision is invalid")
    for key in (
        "package_sha256",
        "package_manifest_sha256",
        "record_sha256",
        "history_sha256",
        "optimizer_telemetry_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(value.get(key, ""))):
            raise ValueError(f"mechanics evidence {key} is invalid")
