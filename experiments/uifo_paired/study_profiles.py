"""Frozen paid-study profiles and their predeclared decision rules."""

from __future__ import annotations

import copy
import math


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
