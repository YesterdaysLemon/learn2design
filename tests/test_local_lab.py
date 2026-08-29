from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tools.run_local_lab as lab_controller

jax = pytest.importorskip("jax")
pytest.importorskip("jax.numpy")
pytest.importorskip("dfbench")
pytest.importorskip("numpy")

pytestmark = pytest.mark.integration

from experiments.local_lab.anchor_lane_stability import (
    REPOSITORY_ROOT,
    run_study,
)
from experiments.local_lab.contextual_bandit_toy_signal import (
    run_study as run_contextual_bandit_toy_signal_study,
)
from experiments.local_lab.two_step_delayed_credit import (
    run_study as run_two_step_delayed_credit_study,
)
from experiments.local_lab.feasible_progress_clock import (
    run_study as run_feasible_progress_study,
)
from experiments.local_lab.full_surface_prefix import (
    AUX_LEAF_PATHS as FULL_SURFACE_AUX_LEAF_PATHS,
    SIGNAL_CLASSES as FULL_SURFACE_SIGNAL_CLASSES,
    run_study as run_full_surface_prefix_study,
)
from experiments.local_lab.infeasible_prefix_indistinguishability import (
    run_study as run_prefix_boundary_study,
)
from experiments.local_lab.normal_path_jax_boundary import (
    AUX_LEAF_PATHS as JAX_BOUNDARY_AUX_LEAF_PATHS,
    TELEMETRY_LEAVES as JAX_BOUNDARY_TELEMETRY_LEAVES,
    run_study as run_normal_path_jax_boundary_study,
)
from experiments.local_lab.public_signal_surface import (
    AUX_LEAF_PATHS,
    DFBENCH_WHEEL_SHA256,
    run_study as run_public_signal_surface_study,
)
from experiments.local_lab.supervised_toy_signal import (
    run_study as run_supervised_toy_signal_study,
)
from tools.run_local_lab import (
    DuplicateStudyError,
    EXPECTED_SUBMISSION_SOURCE_SHA256,
    EXPECTED_SUBMISSION_TREE_OID,
    _acquire_lease,
    _begin_cycle,
    _git,
    _git_bytes,
    _load_state,
    _release_lease,
    _run_worker,
    _validate_study_result,
    _write_atomic,
)


def test_anchor_lane_stability_frozen_study_passes() -> None:
    result = run_study(include_process_isolation=True)

    assert result["study_id"] == "anchor-lane-stability-v1"
    assert result["status"] == "passed"
    assert result["action"] == "anchor_lane_mechanics_confirmed"
    assert set(result["cases"]) == {
        "diagnostics_disabled_control",
        "exact_twin",
        "exceptional_arithmetic_partial_tail",
        "forced_shared_state_boundary",
        "process_isolation",
        "suffix_invariance",
    }
    assert all(case["passed"] for case in result["cases"].values())
    assert result["environment"]["platform"] == "cpu"
    assert result["fixture"]["population_size"] == 8
    assert result["fixture"]["case_contract"]["forced_shared_state_boundary"][
        "max_evals"
    ] == 32
    assert result["cases"]["suffix_invariance"]["raw_random_hashes_equal"]
    assert result["cases"]["exceptional_arithmetic_partial_tail"][
        "batch_sizes"
    ] == [8, 8, 2]


def test_anchor_lane_result_is_sanitized() -> None:
    result = run_study(include_process_isolation=False)
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)

    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert result["cases"]["process_isolation"]["passed"] is None

    forbidden = (
        str(REPOSITORY_ROOT),
        "coverage-triage-v1",
        "coverage-robustness-v1",
        "optimization_pairs",
        "parameter_values",
        "raw_gradient",
    )
    assert all(value not in encoded for value in forbidden)


def test_feasible_progress_clock_frozen_study_passes() -> None:
    result = run_feasible_progress_study(include_process_isolation=True)

    assert result["study_id"] == "feasible-progress-clock-v1"
    assert result["status"] == "passed"
    assert result["action"] == (
        "finite_infeasible_progress_resets_clock_confirmed"
    )
    assert set(result["cases"]) == {
        "diagnostics_disabled_control",
        "finite_infeasible_descent",
        "finite_infeasible_improve_then_plateau",
        "finite_infeasible_plateau_control",
        "late_feasibility_crossing",
        "mixed_member_clock",
        "process_isolation",
    }
    assert all(case["passed"] for case in result["cases"].values())
    assert result["cases"]["finite_infeasible_descent"][
        "finite_infeasible_observations"
    ] == 32
    assert result["cases"]["mixed_member_clock"][
        "changed_members_after_boundary"
    ] == [1, 3, 5, 7]
    assert result["environment"]["platform"] == "cpu"


def test_feasible_progress_clock_result_is_sanitized() -> None:
    result = run_feasible_progress_study(include_process_isolation=False)
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)

    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert result["cases"]["process_isolation"]["passed"] is None
    forbidden = (
        str(REPOSITORY_ROOT),
        "optimization_pairs",
        "parameter_values",
        "raw_gradient",
        "topology",
    )
    assert all(value not in encoded for value in forbidden)


def test_infeasible_prefix_boundary_frozen_study_passes() -> None:
    result = run_prefix_boundary_study(include_process_isolation=True)

    assert result["study_id"] == "infeasible-prefix-indistinguishability-v1"
    assert result["status"] == "passed"
    assert result["action"] == (
        "synthetic_identical_prefix_obstruction_confirmed"
    )
    assert set(result["cases"]) == {
        "boundary_sweep",
        "extra_signal_positive_control",
        "action_vector_exhaustion",
        "process_isolation",
        "shared_prefix_identity",
        "witness_partition",
    }
    assert all(case["passed"] for case in result["cases"].values())
    assert result["cases"]["shared_prefix_identity"]["prefixes_identical"]
    assert result["cases"]["shared_prefix_identity"][
        "forever_prefix_sha256"
    ] == result["cases"]["shared_prefix_identity"]["late_prefix_sha256"]
    assert result["cases"]["action_vector_exhaustion"] == {
        "bound": 13,
        "joint_satisfiers": 0,
        "passed": True,
        "total_action_vectors": 8192,
    }
    assert result["cases"]["witness_partition"] == {
        "bounded_only_policies": 8191,
        "joint_satisfiers": 0,
        "partition_total": 8192,
        "passed": True,
        "preserve_only_policies": 1,
    }
    assert result["cases"]["boundary_sweep"][
        "joint_satisfiers_by_bound"
    ] == [0] * 6
    assert not result["cases"]["extra_signal_positive_control"][
        "prefixes_identical"
    ]
    assert result["environment"]["platform"] == "cpu"


def test_infeasible_prefix_boundary_result_is_sanitized() -> None:
    result = run_prefix_boundary_study(include_process_isolation=False)
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)

    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert result["cases"]["process_isolation"]["passed"] is None
    forbidden = (
        str(REPOSITORY_ROOT),
        "optimization_pairs",
        "parameter_values",
        "raw_gradient",
        "topology",
    )
    assert all(value not in encoded for value in forbidden)


def test_public_signal_surface_frozen_study_passes() -> None:
    result = run_public_signal_surface_study(include_process_isolation=True)

    assert result["study_id"] == "public-signal-surface-v1"
    assert result["status"] == "passed"
    assert result["action"] == "public_current_constraint_signals_confirmed"
    assert set(result["cases"]) == {
        "candidate_passthrough_modes",
        "consumer_boundary",
        "dependency_source_identity",
        "infeasible_magnitude_control",
        "no_aux_negative_control",
        "process_isolation",
        "scalar_batch_roundtrip",
        "uifo_aux_schema",
    }
    assert all(case["passed"] for case in result["cases"].values())
    assert result["cases"]["dependency_source_identity"][
        "dfbench_wheel_sha256"
    ] == DFBENCH_WHEEL_SHA256
    assert result["cases"]["uifo_aux_schema"]["top_level_fields"] == [
        "is_feasible",
        "penalty",
        "power_values",
        "sensitivity_loss",
        "violations",
    ]
    assert result["cases"]["candidate_passthrough_modes"]["aux_leaf_paths"] == (
        AUX_LEAF_PATHS
    )
    assert result["cases"]["consumer_boundary"]["consumed_aux_fields"] == [
        "is_feasible"
    ]
    assert result["cases"]["infeasible_magnitude_control"][
        "same_infeasible_boolean"
    ]
    assert not result["cases"]["no_aux_negative_control"][
        "rich_aux_universal"
    ]
    assert result["environment"]["platform"] == "cpu"


def test_public_signal_surface_result_is_sanitized() -> None:
    result = run_public_signal_surface_study(include_process_isolation=False)
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)

    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert result["cases"]["process_isolation"]["passed"] is None
    forbidden = (
        str(REPOSITORY_ROOT),
        "optimization_pairs",
        "parameter_values",
        "raw_gradient",
        "topology",
    )
    assert all(value not in encoded for value in forbidden)


def test_full_surface_prefix_frozen_study_passes() -> None:
    result = run_full_surface_prefix_study(include_process_isolation=True)

    assert result["study_id"] == "full-surface-prefix-indistinguishability-v1"
    assert result["status"] == "passed"
    assert result["action"] == "synthetic_full_surface_prefix_twin_confirmed"
    assert set(result["cases"]) == {
        "action_vector_exhaustion",
        "adapter_schema",
        "aux_leaf_negative_controls",
        "forbidden_extension_rejection",
        "normal_path_execution",
        "process_isolation",
        "shared_full_surface_prefix",
        "signal_class_negative_controls",
        "typed_array_metadata_boundary",
    }
    assert all(case["passed"] for case in result["cases"].values())
    assert result["cases"]["normal_path_execution"] == {
        "batches_per_world": 9,
        "evaluations_per_world": 36,
        "passed": True,
        "restart_events": 0,
        "rng_draws_per_world": 1,
        "scalar_calls": 0,
        "state_commitments_per_world": 9,
        "telemetry_events_per_world": 9,
        "worlds": 2,
    }
    assert result["cases"]["shared_full_surface_prefix"][
        "prefixes_identical"
    ]
    assert result["cases"]["shared_full_surface_prefix"][
        "prefix_all_finite"
    ]
    assert result["cases"]["shared_full_surface_prefix"][
        "prefix_all_infeasible"
    ]
    assert result["cases"]["shared_full_surface_prefix"][
        "prefix_strictly_improving"
    ]
    assert result["cases"]["shared_full_surface_prefix"][
        "next_evaluation_difference_paths"
    ] == ["aux.is_feasible.sha256"]
    assert result["cases"]["aux_leaf_negative_controls"][
        "aux_leaf_paths"
    ] == FULL_SURFACE_AUX_LEAF_PATHS
    assert result["cases"]["signal_class_negative_controls"][
        "control_names"
    ] == FULL_SURFACE_SIGNAL_CLASSES
    assert result["cases"]["action_vector_exhaustion"][
        "joint_satisfiers"
    ] == 0
    assert result["environment"]["platform"] == "cpu"


def test_full_surface_prefix_result_is_sanitized() -> None:
    result = run_full_surface_prefix_study(include_process_isolation=False)
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)

    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert result["cases"]["process_isolation"]["passed"] is None
    forbidden = (
        str(REPOSITORY_ROOT),
        "optimization_pairs",
        "parameter_values",
        "raw_gradient",
        "topology",
    )
    assert all(value not in encoded for value in forbidden)


def test_normal_path_jax_boundary_frozen_study_passes() -> None:
    result = run_normal_path_jax_boundary_study(include_process_isolation=True)

    assert result["study_id"] == "normal-path-jax-boundary-v1"
    assert result["status"] == "passed"
    assert result["action"] == (
        "synthetic_normal_path_jax_boundary_equivalent"
    )
    assert set(result["cases"]) == {
        "boundary_negative_controls",
        "dependency_source_identity",
        "explicit_jit_lowering",
        "normal_path_boundary_trace",
        "process_isolation",
        "pure_jax_transition_equivalence",
        "source_boundary_inventory",
    }
    assert all(case["passed"] for case in result["cases"].values())
    assert result["cases"]["source_boundary_inventory"][
        "objective_logging_device_host_sites"
    ] == 5
    assert result["cases"]["source_boundary_inventory"][
        "optimizer_device_scalar_sites"
    ] == 3
    assert result["cases"]["normal_path_boundary_trace"][
        "explicit_barriers"
    ] == 1
    assert result["cases"]["normal_path_boundary_trace"][
        "evaluation_count_after"
    ] == 4
    assert result["cases"]["pure_jax_transition_equivalence"][
        "exact_typed_matches"
    ] == 46
    assert result["cases"]["pure_jax_transition_equivalence"][
        "aux_leaves_checked"
    ] == len(JAX_BOUNDARY_AUX_LEAF_PATHS)
    assert result["cases"]["pure_jax_transition_equivalence"][
        "telemetry_leaves_checked"
    ] == len(JAX_BOUNDARY_TELEMETRY_LEAVES)
    assert result["cases"]["explicit_jit_lowering"]["eager_jit_exact"]
    assert result["cases"]["explicit_jit_lowering"][
        "eager_compiled_exact"
    ]
    assert result["environment"]["platform"] == "cpu"


def test_normal_path_jax_boundary_result_is_sanitized() -> None:
    result = run_normal_path_jax_boundary_study(
        include_process_isolation=False
    )
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)

    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert result["cases"]["process_isolation"]["passed"] is None
    forbidden = (
        str(REPOSITORY_ROOT),
        "optimization_pairs",
        "parameter_values",
        "raw_gradient",
        "topology",
    )
    assert all(value not in encoded for value in forbidden)


def _completed_supervised_toy_signal_result() -> dict[str, object]:
    result = run_supervised_toy_signal_study(include_process_isolation=False)
    result["cases"]["process_isolation"] = {
        "passed": True,
        "trace_sha256": "0" * 64,
    }
    result["status"] = "passed"
    result["action"] = (
        "synthetic_supervised_toy_signal_recovered_for_harness"
    )
    return result


def test_supervised_toy_signal_focused_projection_passes() -> None:
    result = run_supervised_toy_signal_study(include_process_isolation=False)

    assert result["study_id"] == "supervised-toy-signal-v1"
    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert set(result["cases"]) == {
        "baseline_replay",
        "generator_partition",
        "label_shuffle_control",
        "leakage_guards",
        "process_isolation",
        "signal_attribution_control",
        "supervised_recovery",
        "typed_task_contract",
    }
    assert all(
        case["passed"]
        for name, case in result["cases"].items()
        if name != "process_isolation"
    )
    assert result["cases"]["process_isolation"]["passed"] is None
    assert result["cases"]["supervised_recovery"]["test_macro_accuracy"] >= 0.99
    assert result["cases"]["supervised_recovery"][
        "test_gain_over_constant"
    ] >= 0.30
    assert result["cases"]["supervised_recovery"][
        "test_gain_over_random"
    ] >= 0.25
    assert result["cases"]["label_shuffle_control"]["test_macro_accuracy"] <= 0.55
    assert result["cases"]["generator_partition"][
        "expected_dataset_commitment"
    ]
    assert result["cases"]["generator_partition"][
        "within_split_keys_unique"
    ]
    assert result["cases"]["leakage_guards"][
        "fit_scope_sentinels_rejected"
    ] == 2
    assert result["cases"]["label_shuffle_control"][
        "heldout_commitment_unchanged"
    ]
    assert result["cases"]["signal_attribution_control"][
        "nuisance_only_test_macro_accuracy"
    ] <= 0.55
    assert result["cases"]["signal_attribution_control"][
        "only_signal_changed"
    ]
    assert result["environment"]["platform"] == "cpu"


def test_supervised_toy_signal_result_is_sanitized() -> None:
    result = run_supervised_toy_signal_study(include_process_isolation=False)
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)

    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert result["cases"]["process_isolation"]["passed"] is None
    forbidden = (
        str(REPOSITORY_ROOT),
        "optimization_pairs",
        "parameter_values",
        "raw_gradient",
        "topology",
    )
    assert all(value not in encoded for value in forbidden)

    def recursive_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                nested
                for child in value.values()
                for nested in recursive_keys(child)
            }
        if isinstance(value, list):
            return {
                nested for child in value for nested in recursive_keys(child)
            }
        return set()

    raw_result_keys = {
        "labels",
        "model_weights",
        "paths",
        "predictions",
        "raw_actions",
        "raw_observations",
        "trajectories",
    }
    assert recursive_keys(result).isdisjoint(raw_result_keys)


def _completed_contextual_bandit_toy_signal_result() -> dict[str, object]:
    result = run_contextual_bandit_toy_signal_study(
        include_process_isolation=False
    )
    result["cases"]["process_isolation"] = {
        "passed": True,
        "trace_sha256": "0" * 64,
    }
    result["status"] = "passed"
    result["action"] = (
        "synthetic_contextual_bandit_signal_recovered_for_harness"
    )
    return result


def test_contextual_bandit_toy_signal_focused_projection_passes() -> None:
    result = run_contextual_bandit_toy_signal_study(
        include_process_isolation=False
    )

    assert result["study_id"] == "contextual-bandit-toy-signal-v1"
    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert set(result["cases"]) == {
        "baseline_replay",
        "contextual_recovery",
        "generator_partition",
        "leakage_guards",
        "online_update_order",
        "process_isolation",
        "shuffled_context_control",
        "signal_attribution_control",
        "typed_bandit_contract",
    }
    assert all(
        case["passed"]
        for name, case in result["cases"].items()
        if name != "process_isolation"
    )
    assert result["cases"]["process_isolation"]["passed"] is None
    recovery = result["cases"]["contextual_recovery"]
    assert recovery["train_mean_reward"] == 0.75
    assert recovery["train_cumulative_regret"] == 256
    assert recovery["validation_macro_reward"] == 1.0
    assert recovery["test_macro_reward"] == 1.0
    assert recovery["test_gain_over_random"] >= 0.25
    assert result["cases"]["online_update_order"]["final_update_count"] == 1024
    assert result["cases"]["leakage_guards"]["heldout_updates"] == 0
    shuffled = result["cases"]["shuffled_context_control"]
    assert shuffled["validation_macro_reward"] <= 0.55
    assert shuffled["test_macro_reward"] <= 0.55
    assert shuffled["true_reward_gap"] >= 0.40
    attribution = result["cases"]["signal_attribution_control"]
    assert attribution["refit_test_macro_reward"] <= 0.55
    assert attribution["true_policy_zeroed_test_macro_reward"] <= 0.55
    assert result["environment"]["platform"] == "cpu"


def test_contextual_bandit_toy_signal_result_is_sanitized() -> None:
    result = run_contextual_bandit_toy_signal_study(
        include_process_isolation=False
    )
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)

    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert result["cases"]["process_isolation"]["passed"] is None
    forbidden = (
        str(REPOSITORY_ROOT),
        "credential",
        "optimization_pairs",
        "parameter_values",
        "raw_gradient",
        "topology",
    )
    assert all(value not in encoded for value in forbidden)

    def recursive_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                nested
                for child in value.values()
                for nested in recursive_keys(child)
            }
        if isinstance(value, list):
            return {
                nested for child in value for nested in recursive_keys(child)
            }
        return set()

    raw_result_keys = {
        "actions",
        "contexts",
        "logs",
        "paths",
        "policy_state",
        "preferred_actions",
        "rewards",
        "table_cells",
        "trajectories",
    }
    assert recursive_keys(result).isdisjoint(raw_result_keys)


def _completed_two_step_delayed_credit_result() -> dict[str, object]:
    result = run_two_step_delayed_credit_study(
        include_process_isolation=False
    )
    result["cases"]["process_isolation"] = {
        "passed": True,
        "trace_sha256": "0" * 64,
    }
    result["status"] = "passed"
    result["action"] = (
        "synthetic_two_step_delayed_credit_recovered_for_harness"
    )
    return result


def _completed_multistep_v3_result(entry: dict[str, object]) -> dict[str, object]:
    required = entry["case_required_fields"]
    cases = {}
    for case_name, fields in required.items():
        case = {}
        for field in fields:
            if field == "passed":
                case[field] = True
            elif field in lab_controller.V3_BOOLEAN_CASE_FIELDS:
                case[field] = True
            elif field.endswith("sha256"):
                case[field] = "0" * 64
            elif field.endswith(
                (
                    "_disjoint",
                    "_empty",
                    "_exact",
                    "_preserved",
                    "_unchanged",
                    "_unreachable",
                )
            ):
                case[field] = True
            else:
                case[field] = 0
        cases[case_name] = case
    return {
        "action": "synthetic_four_step_synchronous_td_propagation_confirmed_for_harness",
        "cases": cases,
        "environment": {
            "device_kind": "cpu",
            "jax_version": "test",
            "platform": "cpu",
            "python": "test",
        },
        "fixture": {
            "case_contract": entry["case_contract"],
            **entry["fixture_identity"],
        },
        "schema_version": entry["result_schema_version"],
        "status": "passed",
        "study_id": "multistep-td-action-prefix-v3",
    }


def test_two_step_delayed_credit_focused_projection_passes() -> None:
    result = run_two_step_delayed_credit_study(
        include_process_isolation=False
    )

    assert result["study_id"] == "two-step-delayed-credit-v1"
    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert set(result["cases"]) == {
        "action_dependent_transition",
        "baseline_replay",
        "delayed_credit_recovery",
        "delayed_update_order",
        "generator_partition",
        "leakage_guards",
        "process_isolation",
        "reward_delay_control",
        "signal_attribution_control",
        "transition_shuffle_control",
        "typed_episodic_contract",
    }
    assert all(
        case["passed"]
        for name, case in result["cases"].items()
        if name != "process_isolation"
    )
    assert result["cases"]["process_isolation"]["passed"] is None
    recovery = result["cases"]["delayed_credit_recovery"]
    assert recovery["behavior_train_mean_return"] == 0.25
    assert recovery["behavior_train_regret"] == 96
    assert recovery["postfit_train_macro_return"] == 1.0
    assert recovery["validation_macro_return"] == 1.0
    assert recovery["test_macro_return"] == 1.0
    assert recovery["test_gain_over_random"] >= 0.30
    assert result["cases"]["delayed_update_order"][
        "episode_update_count"
    ] == 128
    assert result["cases"]["delayed_update_order"][
        "cell_update_count"
    ] == 256
    assert result["cases"]["leakage_guards"]["heldout_updates"] == 0
    assert result["cases"]["leakage_guards"][
        "scoring_sentinels_rejected"
    ] == 9
    transition_control = result["cases"]["transition_shuffle_control"]
    assert transition_control["donor_mapping_exact"]
    assert transition_control["validation_macro_return"] <= 0.05
    assert transition_control["test_macro_return"] <= 0.05
    assert transition_control["true_test_gap"] >= 0.90
    reward_control = result["cases"]["reward_delay_control"]
    assert reward_control["origin_assignment_exact"]
    assert reward_control["reward_queue_empty_at_boundary"]
    assert reward_control["validation_macro_return"] <= 0.55
    assert reward_control["test_macro_return"] <= 0.55
    assert reward_control["true_test_gap"] >= 0.40
    attribution = result["cases"]["signal_attribution_control"]
    assert attribution["refit_test_macro_return"] <= 0.55
    assert attribution["true_policy_ablated_test_macro_return"] <= 0.55
    assert result["environment"]["platform"] == "cpu"


def test_two_step_delayed_credit_result_is_sanitized() -> None:
    result = run_two_step_delayed_credit_study(
        include_process_isolation=False
    )
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)

    assert result["status"] == "incomplete"
    assert result["action"] == "no_decision_incomplete_study"
    assert result["cases"]["process_isolation"]["passed"] is None
    forbidden = (
        str(REPOSITORY_ROOT),
        "credential",
        "optimization_pairs",
        "parameter_values",
        "raw_gradient",
        "topology",
    )
    assert all(value not in encoded for value in forbidden)

    def recursive_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                nested
                for child in value.values()
                for nested in recursive_keys(child)
            }
        if isinstance(value, list):
            return {
                nested for child in value for nested in recursive_keys(child)
            }
        return set()

    raw_result_keys = {
        "actions",
        "logs",
        "observations",
        "paths",
        "policy_state",
        "q_values",
        "returns",
        "rewards",
        "states",
        "targets",
        "trajectories",
        "transitions",
        "value_table",
    }
    assert recursive_keys(result).isdisjoint(raw_result_keys)


def test_result_validator_requires_exact_sanitized_contract() -> None:
    registry = lab_controller._load_study_registry()
    entry = lab_controller._study_entry(registry, "feasible-progress-clock-v1")
    result = run_feasible_progress_study(include_process_isolation=False)
    result["cases"]["process_isolation"] = {
        "passed": True,
        "trace_sha256": "0" * 64,
    }
    result["status"] = "passed"
    result["action"] = "finite_infeasible_progress_resets_clock_confirmed"

    _validate_study_result("feasible-progress-clock-v1", entry, result)
    result["cases"]["mixed_member_clock"]["parameter_values"] = [[1.0]]
    with pytest.raises(RuntimeError, match="forbidden field"):
        _validate_study_result("feasible-progress-clock-v1", entry, result)


@pytest.mark.parametrize(
    "forbidden_key,forbidden_value",
    [
        ("observations", [[0.0]]),
        ("observation", [0.0]),
        ("raw_actions", [0, 1]),
        ("action", [0]),
        ("actions", [0, 1]),
        ("rewards", [1.0]),
        ("reward", [1.0]),
        ("return", [1.0]),
        ("targets", [0, 1]),
        ("q_table", [[0.0]]),
        ("paths", [{"step": 0}]),
        ("path", [{"step": 0}]),
        ("transition", [{"step": 0}]),
        ("trajectory", [{"step": 0}]),
        ("log", [{"step": 0}]),
        ("weight", [1.0]),
        ("gradient", [1.0]),
        ("q_value", [1.0]),
        ("private_evidence_alias", "x"),
        ("policy_parameters_copy", [0.0]),
        ("raw_trajectory_payload", [{"x": 1}]),
        ("successor_payload", [[0.0]]),
        ("successor", [[0.0]]),
        ("successors", [[0.0]]),
        ("rows", [{"x": 1}]),
        ("data", [1]),
        ("records", [{"x": 1}]),
        ("raw_state", [0.0]),
        ("donor_array", [1]),
        ("origin_array", [1]),
        ("gradients", [[1.0]]),
    ],
)
def test_controller_sanitizer_rejects_raw_result_variants(
    forbidden_key: str, forbidden_value: object
) -> None:
    with pytest.raises(RuntimeError, match="forbidden field"):
        lab_controller._validate_sanitized_value(  # noqa: SLF001
            {forbidden_key: forbidden_value}
        )


@pytest.mark.parametrize(
    "absolute_path",
    [
        r"C:\\private\\result.json",
        r"\private\result.json",
        "/private/result.json",
        r"\\\\server\\share\\result.json",
    ],
)
def test_controller_sanitizer_rejects_absolute_paths(absolute_path: str) -> None:
    with pytest.raises(RuntimeError, match="absolute path"):
        lab_controller._validate_sanitized_value({"safe": absolute_path})  # noqa: SLF001


@pytest.mark.parametrize(
    "case_name,field_name",
    [
        ("typed_episodic_contract", "immutable_observations"),
        ("complete_family_replay", "action_balance"),
        ("complete_family_replay", "target_balance"),
        ("realized_path_disjointness", "identity_fields_excluded"),
        ("pending_transition_authentication", "cross_episode_rejected"),
        ("synchronous_td_order", "exact_positive_sets"),
        ("transition_target_control", "positive_gate_rejected"),
        ("signal_attribution_control", "only_signal_changed"),
        ("control_difference_whitelists", "reward_origin_bijection"),
        ("control_difference_whitelists", "transition_involution"),
    ],
)
def test_v3_result_validator_requires_explicit_boolean_invariants(
    case_name: str, field_name: str
) -> None:
    registry = lab_controller._load_study_registry()
    study_id = "multistep-td-action-prefix-v3"
    entry = lab_controller._study_entry(registry, study_id)
    result = _completed_multistep_v3_result(entry)
    result["cases"][case_name][field_name] = [True]

    with pytest.raises(RuntimeError, match="forbidden field|non-Boolean V3 invariant"):
        lab_controller._validate_study_result(study_id, entry, result)


@pytest.mark.parametrize("field_name", ["device_kind", "jax_version", "python"])
def test_v3_result_validator_requires_scalar_environment_strings(
    field_name: str,
) -> None:
    registry = lab_controller._load_study_registry()
    study_id = "multistep-td-action-prefix-v3"
    entry = lab_controller._study_entry(registry, study_id)
    result = _completed_multistep_v3_result(entry)
    result["environment"][field_name] = ["cpu"]

    with pytest.raises(RuntimeError, match="CPU backend"):
        lab_controller._validate_study_result(study_id, entry, result)


def test_prefix_boundary_validator_requires_exact_fixture_identity() -> None:
    registry = lab_controller._load_study_registry()
    entry = lab_controller._study_entry(
        registry, "infeasible-prefix-indistinguishability-v1"
    )
    result = run_prefix_boundary_study(include_process_isolation=False)
    result["cases"]["process_isolation"] = {
        "passed": True,
        "trace_sha256": "0" * 64,
    }
    result["status"] = "passed"
    result["action"] = "synthetic_identical_prefix_obstruction_confirmed"

    _validate_study_result(
        "infeasible-prefix-indistinguishability-v1", entry, result
    )
    result["fixture"]["max_bound"] = 12
    with pytest.raises(RuntimeError, match="wrong frozen fixture identity"):
        _validate_study_result(
            "infeasible-prefix-indistinguishability-v1", entry, result
        )


def test_public_signal_surface_validator_requires_exact_contract() -> None:
    registry = lab_controller._load_study_registry()
    entry = lab_controller._study_entry(registry, "public-signal-surface-v1")
    result = run_public_signal_surface_study(include_process_isolation=False)
    result["cases"]["process_isolation"] = {
        "passed": True,
        "trace_sha256": "0" * 64,
    }
    result["status"] = "passed"
    result["action"] = "public_current_constraint_signals_confirmed"

    _validate_study_result("public-signal-surface-v1", entry, result)
    result["fixture"]["dfbench_wheel_sha256"] = "1" * 64
    with pytest.raises(RuntimeError, match="wrong frozen fixture identity"):
        _validate_study_result("public-signal-surface-v1", entry, result)


def test_full_surface_prefix_validator_requires_exact_contract() -> None:
    registry = lab_controller._load_study_registry()
    entry = lab_controller._study_entry(
        registry, "full-surface-prefix-indistinguishability-v1"
    )
    result = run_full_surface_prefix_study(include_process_isolation=False)
    result["cases"]["process_isolation"] = {
        "passed": True,
        "trace_sha256": "0" * 64,
    }
    result["status"] = "passed"
    result["action"] = "synthetic_full_surface_prefix_twin_confirmed"

    _validate_study_result(
        "full-surface-prefix-indistinguishability-v1", entry, result
    )
    wrong_identity = json.loads(json.dumps(result))
    wrong_identity["fixture"]["max_bound"] = 7
    with pytest.raises(RuntimeError, match="wrong frozen fixture identity"):
        _validate_study_result(
            "full-surface-prefix-indistinguishability-v1",
            entry,
            wrong_identity,
        )

    wrong_contract = json.loads(json.dumps(result))
    wrong_contract["fixture"]["case_contract"]["shared_full_surface_prefix"][
        "bound"
    ] = 7
    with pytest.raises(RuntimeError, match="wrong frozen case contract"):
        _validate_study_result(
            "full-surface-prefix-indistinguishability-v1",
            entry,
            wrong_contract,
        )


def test_normal_path_jax_boundary_validator_requires_exact_contract() -> None:
    registry = lab_controller._load_study_registry()
    study_id = "normal-path-jax-boundary-v1"
    entry = lab_controller._study_entry(registry, study_id)
    result = run_normal_path_jax_boundary_study(
        include_process_isolation=False
    )
    result["cases"]["process_isolation"] = {
        "passed": True,
        "trace_sha256": "0" * 64,
    }
    result["status"] = "passed"
    result["action"] = "synthetic_normal_path_jax_boundary_equivalent"

    _validate_study_result(study_id, entry, result)
    result["fixture"]["batch_microseconds"] = 3999
    with pytest.raises(RuntimeError, match="wrong frozen fixture identity"):
        _validate_study_result(study_id, entry, result)


def test_supervised_toy_signal_validator_requires_exact_contract() -> None:
    registry = lab_controller._load_study_registry()
    study_id = "supervised-toy-signal-v1"
    entry = lab_controller._study_entry(registry, study_id)
    result = _completed_supervised_toy_signal_result()

    _validate_study_result(study_id, entry, result)
    result["fixture"]["baseline_seed"] += 1
    with pytest.raises(RuntimeError, match="wrong frozen fixture identity"):
        _validate_study_result(study_id, entry, result)


def test_contextual_bandit_toy_signal_validator_requires_exact_contract() -> None:
    registry = lab_controller._load_study_registry()
    study_id = "contextual-bandit-toy-signal-v1"
    entry = lab_controller._study_entry(registry, study_id)
    result = _completed_contextual_bandit_toy_signal_result()

    _validate_study_result(study_id, entry, result)
    result["fixture"]["baseline_seed"] += 1
    with pytest.raises(RuntimeError, match="wrong frozen fixture identity"):
        _validate_study_result(study_id, entry, result)


def test_two_step_delayed_credit_validator_requires_exact_contract() -> None:
    registry = lab_controller._load_study_registry()
    study_id = "two-step-delayed-credit-v1"
    entry = lab_controller._study_entry(registry, study_id)
    result = _completed_two_step_delayed_credit_result()

    _validate_study_result(study_id, entry, result)
    result["fixture"]["random_baseline_seed"] += 1
    with pytest.raises(RuntimeError, match="wrong frozen fixture identity"):
        _validate_study_result(study_id, entry, result)


def test_protected_submission_canonical_digest_matches_pin() -> None:
    committed_source = _git_bytes("show", "HEAD:submission/submission.py")

    assert hashlib.sha256(committed_source).hexdigest() == (
        EXPECTED_SUBMISSION_SOURCE_SHA256
    )
    assert _git("rev-parse", "HEAD:submission") == EXPECTED_SUBMISSION_TREE_OID


def test_local_lab_atomic_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    first_digest = _write_atomic(output, {"status": "passed"})

    assert len(first_digest) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "passed"
    }
    with pytest.raises(FileExistsError):
        _write_atomic(output, {"status": "failed"})


def test_local_lab_lease_is_single_writer_and_identity_bound(tmp_path: Path) -> None:
    lock_directory, lease_id = _acquire_lease(
        tmp_path,
        cycle_id="cycle-test",
        revision="a" * 40,
        study="anchor-lane-stability-v1",
    )

    with pytest.raises(RuntimeError, match="lease already exists"):
        _acquire_lease(
            tmp_path,
            cycle_id="cycle-test-2",
            revision="a" * 40,
            study="anchor-lane-stability-v1",
        )
    with pytest.raises(RuntimeError, match="identity changed"):
        _release_lease(lock_directory, "different")

    _release_lease(lock_directory, lease_id)
    assert not lock_directory.exists()


def test_local_lab_state_refuses_overlap(tmp_path: Path) -> None:
    snapshot = {"revision": "a" * 40}
    output = tmp_path / "cycles" / "cycle-test" / "result.json"

    state = _begin_cycle(
        tmp_path,
        cycle_id="cycle-test",
        output=output,
        snapshot=snapshot,
        study="anchor-lane-stability-v1",
    )

    assert state["status"] == "active"
    assert _load_state(tmp_path)["active_cycle"]["cycle_id"] == "cycle-test"
    with pytest.raises(RuntimeError, match="state is not idle"):
        _begin_cycle(
            tmp_path,
            cycle_id="cycle-test-2",
            output=tmp_path / "cycles" / "cycle-test-2" / "result.json",
            snapshot=snapshot,
            study="anchor-lane-stability-v1",
        )


def test_completed_study_refusal_does_not_park_state(tmp_path: Path) -> None:
    original = lab_controller._default_state()
    original["status"] = "awaiting_study"
    original["completed_studies"] = {
        "anchor-lane-stability-v1": {
            "status": "passed",
        }
    }
    lab_controller._write_mutable_json(tmp_path / "lab-state.json", original)

    with pytest.raises(DuplicateStudyError, match="terminal record"):
        _begin_cycle(
            tmp_path,
            cycle_id="duplicate-test",
            output=tmp_path / "cycles" / "duplicate-test" / "result.json",
            snapshot={"revision": "a" * 40},
            study="anchor-lane-stability-v1",
        )

    after = _load_state(tmp_path)
    assert after == original


def test_worker_policy_probe_blocks_network_and_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "must-not-cross-worker-boundary")

    result, _receipt = _run_worker(
        "policy-probe",
        cycle_id="policy-probe",
        heartbeat=lambda _pid, _elapsed: None,
    )

    assert result == {
        "network_blocked": True,
        "sensitive_environment_names": [],
    }


def test_worker_hard_timeout_terminates_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(lab_controller, "CYCLE_TIMEOUT_SECONDS", 0.05)

    with pytest.raises(TimeoutError, match="exceeded"):
        _run_worker(
            "anchor-lane-stability-v1",
            cycle_id="timeout-probe",
            heartbeat=lambda _pid, _elapsed: None,
        )

    assert not list((tmp_path / "worker-tmp").glob("timeout-probe.*"))


def test_controller_end_to_end_terminal_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = lab_controller._load_study_registry()
    entry = lab_controller._study_entry(registry, "anchor-lane-stability-v1")
    snapshot = {
        "committed_file_sha256": entry["approved_file_sha256"],
        "committed_source_paths": entry["source_paths"],
        "revision": "a" * 40,
    }
    output = tmp_path / "cycles" / "controller-test" / "result.json"
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(
        lab_controller, "_repository_snapshot", lambda _entry: snapshot
    )
    monkeypatch.setattr(lab_controller, "_git", lambda *_args: "a" * 40)
    monkeypatch.setattr(
        lab_controller.sys,
        "argv",
        [
            "run_local_lab.py",
            "--study",
            "anchor-lane-stability-v1",
            "--output",
            str(output),
        ],
    )

    lab_controller.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    sidecar = output.with_name("result.json.sha256").read_text(encoding="ascii")
    state = _load_state(tmp_path)
    events = [
        json.loads(line)
        for line in (tmp_path / "lab-events.jsonl").read_text().splitlines()
    ]
    assert payload["result"]["status"] == "passed"
    assert sidecar.startswith(
        hashlib.sha256(output.read_bytes()).hexdigest() + "  result.json"
    )
    assert state["status"] == "idle"
    assert "anchor-lane-stability-v1" in state["completed_studies"]
    assert {event["event"] for event in events} >= {
        "cycle_completed",
        "cycle_started",
        "heartbeat",
    }
    assert not (tmp_path / "lab.lock").exists()


def test_second_study_end_to_end_leaves_third_study_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = lab_controller._load_study_registry()
    entry = lab_controller._study_entry(registry, "feasible-progress-clock-v1")
    snapshot = {
        "committed_file_sha256": entry["approved_file_sha256"],
        "committed_source_paths": entry["source_paths"],
        "revision": "b" * 40,
    }
    initial_state = lab_controller._default_state()
    initial_state["status"] = "awaiting_study"
    initial_state["completed_studies"] = {
        "anchor-lane-stability-v1": {
            "cycle_id": "prior-cycle",
            "result_sha256": "c" * 64,
            "revision": "d" * 40,
            "status": "passed",
        }
    }
    lab_controller._write_mutable_json(tmp_path / "lab-state.json", initial_state)
    output = tmp_path / "cycles" / "second-study-test" / "result.json"
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(
        lab_controller, "_repository_snapshot", lambda _entry: snapshot
    )
    monkeypatch.setattr(lab_controller, "_git", lambda *_args: "b" * 40)
    monkeypatch.setattr(
        lab_controller.sys,
        "argv",
        [
            "run_local_lab.py",
            "--study",
            "feasible-progress-clock-v1",
            "--output",
            str(output),
        ],
    )

    lab_controller.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    state = _load_state(tmp_path)
    assert payload["result"]["status"] == "passed"
    assert state["status"] == "idle"
    assert state["stop_reason"] is None
    assert set(state["completed_studies"]) == {
        "anchor-lane-stability-v1",
        "feasible-progress-clock-v1",
    }
    assert not (tmp_path / "lab.lock").exists()


def test_third_study_end_to_end_leaves_fourth_study_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = lab_controller._load_study_registry()
    entry = lab_controller._study_entry(
        registry, "infeasible-prefix-indistinguishability-v1"
    )
    snapshot = {
        "committed_file_sha256": entry["approved_file_sha256"],
        "committed_source_paths": entry["source_paths"],
        "revision": "e" * 40,
    }
    prior_anchor = {
        "cycle_id": "anchor-cycle",
        "result_sha256": "a" * 64,
        "revision": "b" * 40,
        "status": "passed",
    }
    prior_clock = {
        "cycle_id": "clock-cycle",
        "result_sha256": "c" * 64,
        "revision": "d" * 40,
        "status": "passed",
    }
    initial_state = lab_controller._default_state()
    initial_state["status"] = "idle"
    initial_state["completed_studies"] = {
        "anchor-lane-stability-v1": prior_anchor,
        "feasible-progress-clock-v1": prior_clock,
    }
    lab_controller._write_mutable_json(tmp_path / "lab-state.json", initial_state)
    output = tmp_path / "cycles" / "third-study-test" / "result.json"
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(
        lab_controller, "_repository_snapshot", lambda _entry: snapshot
    )
    monkeypatch.setattr(lab_controller, "_git", lambda *_args: "e" * 40)
    monkeypatch.setattr(
        lab_controller.sys,
        "argv",
        [
            "run_local_lab.py",
            "--study",
            "infeasible-prefix-indistinguishability-v1",
            "--output",
            str(output),
        ],
    )

    lab_controller.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    state = _load_state(tmp_path)
    assert payload["result"]["status"] == "passed"
    assert state["status"] == "idle"
    assert state["stop_reason"] is None
    assert state["completed_studies"]["anchor-lane-stability-v1"] == prior_anchor
    assert state["completed_studies"]["feasible-progress-clock-v1"] == prior_clock
    assert set(state["completed_studies"]) == {
        "anchor-lane-stability-v1",
        "feasible-progress-clock-v1",
        "infeasible-prefix-indistinguishability-v1",
    }
    assert not (tmp_path / "lab.lock").exists()


def test_fourth_study_end_to_end_leaves_fifth_study_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = lab_controller._load_study_registry()
    entry = lab_controller._study_entry(registry, "public-signal-surface-v1")
    snapshot = {
        "committed_file_sha256": entry["approved_file_sha256"],
        "committed_source_paths": entry["source_paths"],
        "revision": "f" * 40,
    }
    initial_state = lab_controller._default_state()
    initial_state["status"] = "idle"
    initial_state["completed_studies"] = {
        "anchor-lane-stability-v1": {
            "cycle_id": "anchor-cycle",
            "result_sha256": "a" * 64,
            "revision": "b" * 40,
            "status": "passed",
        },
        "feasible-progress-clock-v1": {
            "cycle_id": "clock-cycle",
            "result_sha256": "c" * 64,
            "revision": "d" * 40,
            "status": "passed",
        },
        "infeasible-prefix-indistinguishability-v1": {
            "cycle_id": "prefix-cycle",
            "result_sha256": "e" * 64,
            "revision": "f" * 40,
            "status": "passed",
        },
    }
    lab_controller._write_mutable_json(tmp_path / "lab-state.json", initial_state)
    output = tmp_path / "cycles" / "fourth-study-test" / "result.json"
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(
        lab_controller, "_repository_snapshot", lambda _entry: snapshot
    )
    monkeypatch.setattr(lab_controller, "_git", lambda *_args: "f" * 40)
    monkeypatch.setattr(
        lab_controller.sys,
        "argv",
        [
            "run_local_lab.py",
            "--study",
            "public-signal-surface-v1",
            "--output",
            str(output),
        ],
    )

    lab_controller.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    state = _load_state(tmp_path)
    assert payload["result"]["status"] == "passed"
    assert state["status"] == "idle"
    assert state["stop_reason"] is None
    assert set(state["completed_studies"]) == {
        "anchor-lane-stability-v1",
        "feasible-progress-clock-v1",
        "infeasible-prefix-indistinguishability-v1",
        "public-signal-surface-v1",
    }
    assert not (tmp_path / "lab.lock").exists()


def test_fifth_study_end_to_end_leaves_sixth_study_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = lab_controller._load_study_registry()
    study_id = "full-surface-prefix-indistinguishability-v1"
    entry = lab_controller._study_entry(registry, study_id)
    snapshot = {
        "committed_file_sha256": entry["approved_file_sha256"],
        "committed_source_paths": entry["source_paths"],
        "revision": "1" * 40,
    }
    initial_state = lab_controller._default_state()
    initial_state["status"] = "idle"
    initial_state["completed_studies"] = {
        "anchor-lane-stability-v1": {
            "cycle_id": "anchor-cycle",
            "result_sha256": "a" * 64,
            "revision": "b" * 40,
            "status": "passed",
        },
        "feasible-progress-clock-v1": {
            "cycle_id": "clock-cycle",
            "result_sha256": "c" * 64,
            "revision": "d" * 40,
            "status": "passed",
        },
        "infeasible-prefix-indistinguishability-v1": {
            "cycle_id": "prefix-cycle",
            "result_sha256": "e" * 64,
            "revision": "f" * 40,
            "status": "passed",
        },
        "public-signal-surface-v1": {
            "cycle_id": "signal-cycle",
            "result_sha256": "2" * 64,
            "revision": "3" * 40,
            "status": "passed",
        },
    }
    lab_controller._write_mutable_json(tmp_path / "lab-state.json", initial_state)
    output = tmp_path / "cycles" / "fifth-study-test" / "result.json"
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(
        lab_controller, "_repository_snapshot", lambda _entry: snapshot
    )
    monkeypatch.setattr(lab_controller, "_git", lambda *_args: "1" * 40)
    monkeypatch.setattr(
        lab_controller.sys,
        "argv",
        [
            "run_local_lab.py",
            "--study",
            study_id,
            "--output",
            str(output),
        ],
    )

    lab_controller.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    state = _load_state(tmp_path)
    assert payload["result"]["status"] == "passed"
    assert state["status"] == "idle"
    assert state["stop_reason"] is None
    assert set(state["completed_studies"]) == {
        "anchor-lane-stability-v1",
        "feasible-progress-clock-v1",
        "full-surface-prefix-indistinguishability-v1",
        "infeasible-prefix-indistinguishability-v1",
        "public-signal-surface-v1",
    }
    assert not (tmp_path / "lab.lock").exists()


def test_sixth_study_end_to_end_leaves_seventh_study_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = lab_controller._load_study_registry()
    study_id = "normal-path-jax-boundary-v1"
    entry = lab_controller._study_entry(registry, study_id)
    snapshot = {
        "committed_file_sha256": entry["approved_file_sha256"],
        "committed_source_paths": entry["source_paths"],
        "revision": "4" * 40,
    }
    initial_state = lab_controller._default_state()
    initial_state["status"] = "awaiting_study"
    initial_state["stop_reason"] = "no_approved_study_pending"
    initial_state["completed_studies"] = {
        "anchor-lane-stability-v1": {
            "cycle_id": "anchor-cycle",
            "result_sha256": "a" * 64,
            "revision": "b" * 40,
            "status": "passed",
        },
        "feasible-progress-clock-v1": {
            "cycle_id": "clock-cycle",
            "result_sha256": "c" * 64,
            "revision": "d" * 40,
            "status": "passed",
        },
        "full-surface-prefix-indistinguishability-v1": {
            "cycle_id": "full-surface-cycle",
            "result_sha256": "4" * 64,
            "revision": "5" * 40,
            "status": "passed",
        },
        "infeasible-prefix-indistinguishability-v1": {
            "cycle_id": "prefix-cycle",
            "result_sha256": "e" * 64,
            "revision": "f" * 40,
            "status": "passed",
        },
        "public-signal-surface-v1": {
            "cycle_id": "signal-cycle",
            "result_sha256": "2" * 64,
            "revision": "3" * 40,
            "status": "passed",
        },
    }
    lab_controller._write_mutable_json(tmp_path / "lab-state.json", initial_state)
    output = tmp_path / "cycles" / "sixth-study-test" / "result.json"
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(
        lab_controller, "_repository_snapshot", lambda _entry: snapshot
    )
    monkeypatch.setattr(lab_controller, "_git", lambda *_args: "4" * 40)
    monkeypatch.setattr(
        lab_controller.sys,
        "argv",
        [
            "run_local_lab.py",
            "--study",
            study_id,
            "--output",
            str(output),
        ],
    )

    lab_controller.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    state = _load_state(tmp_path)
    assert payload["result"]["status"] == "passed"
    assert state["status"] == "idle"
    assert state["stop_reason"] is None
    assert set(state["completed_studies"]) == {
        "anchor-lane-stability-v1",
        "feasible-progress-clock-v1",
        "full-surface-prefix-indistinguishability-v1",
        "infeasible-prefix-indistinguishability-v1",
        "normal-path-jax-boundary-v1",
        "public-signal-surface-v1",
    }
    assert not (tmp_path / "lab.lock").exists()


def test_seventh_study_end_to_end_leaves_eighth_study_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = lab_controller._load_study_registry()
    study_id = "supervised-toy-signal-v1"
    entry = lab_controller._study_entry(registry, study_id)
    snapshot = {
        "committed_file_sha256": entry["approved_file_sha256"],
        "committed_source_paths": entry["source_paths"],
        "revision": "6" * 40,
    }
    initial_state = lab_controller._default_state()
    initial_state["status"] = "idle"
    initial_state["completed_studies"] = {
        "anchor-lane-stability-v1": {
            "cycle_id": "anchor-cycle",
            "result_sha256": "a" * 64,
            "revision": "b" * 40,
            "status": "passed",
        },
        "feasible-progress-clock-v1": {
            "cycle_id": "clock-cycle",
            "result_sha256": "c" * 64,
            "revision": "d" * 40,
            "status": "passed",
        },
        "full-surface-prefix-indistinguishability-v1": {
            "cycle_id": "full-surface-cycle",
            "result_sha256": "4" * 64,
            "revision": "5" * 40,
            "status": "passed",
        },
        "infeasible-prefix-indistinguishability-v1": {
            "cycle_id": "prefix-cycle",
            "result_sha256": "e" * 64,
            "revision": "f" * 40,
            "status": "passed",
        },
        "normal-path-jax-boundary-v1": {
            "cycle_id": "jax-cycle",
            "result_sha256": "7" * 64,
            "revision": "8" * 40,
            "status": "passed",
        },
        "public-signal-surface-v1": {
            "cycle_id": "signal-cycle",
            "result_sha256": "2" * 64,
            "revision": "3" * 40,
            "status": "passed",
        },
    }
    lab_controller._write_mutable_json(tmp_path / "lab-state.json", initial_state)
    output = tmp_path / "cycles" / "seventh-study-test" / "result.json"
    complete_result = _completed_supervised_toy_signal_result()
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(
        lab_controller, "_repository_snapshot", lambda _entry: snapshot
    )
    monkeypatch.setattr(lab_controller, "_git", lambda *_args: "6" * 40)
    monkeypatch.setattr(
        lab_controller,
        "_run_worker",
        lambda *_args, **_kwargs: (
            complete_result,
            {
                "stderr_bytes": 0,
                "stderr_sha256": "0" * 64,
                "stdout_bytes": 0,
            },
        ),
    )
    monkeypatch.setattr(
        lab_controller.sys,
        "argv",
        [
            "run_local_lab.py",
            "--study",
            study_id,
            "--output",
            str(output),
        ],
    )

    lab_controller.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    state = _load_state(tmp_path)
    assert payload["result"]["status"] == "passed"
    assert state["status"] == "idle"
    assert state["stop_reason"] is None
    assert set(state["completed_studies"]) == (
        set(registry["studies"])
        - {
            "contextual-bandit-toy-signal-v1",
            "two-step-delayed-credit-v1",
        }
    )
    assert not (tmp_path / "lab.lock").exists()


def test_eighth_study_end_to_end_leaves_ninth_study_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = lab_controller._load_study_registry()
    study_id = "contextual-bandit-toy-signal-v1"
    entry = lab_controller._study_entry(registry, study_id)
    snapshot = {
        "committed_file_sha256": entry["approved_file_sha256"],
        "committed_source_paths": entry["source_paths"],
        "revision": "9" * 40,
    }
    initial_state = lab_controller._default_state()
    initial_state["status"] = "idle"
    initial_state["completed_studies"] = {
        name: {
            "cycle_id": f"prior-{index}",
            "result_sha256": format(index + 1, "x") * 64,
            "revision": format(index + 2, "x") * 40,
            "status": "passed",
        }
        for index, name in enumerate(registry["studies"])
        if name not in {study_id, "two-step-delayed-credit-v1"}
    }
    lab_controller._write_mutable_json(tmp_path / "lab-state.json", initial_state)
    output = tmp_path / "cycles" / "eighth-study-test" / "result.json"
    complete_result = _completed_contextual_bandit_toy_signal_result()
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(
        lab_controller, "_repository_snapshot", lambda _entry: snapshot
    )
    monkeypatch.setattr(lab_controller, "_git", lambda *_args: "9" * 40)
    monkeypatch.setattr(
        lab_controller,
        "_run_worker",
        lambda *_args, **_kwargs: (
            complete_result,
            {
                "stderr_bytes": 0,
                "stderr_sha256": "0" * 64,
                "stdout_bytes": 0,
            },
        ),
    )
    monkeypatch.setattr(
        lab_controller.sys,
        "argv",
        [
            "run_local_lab.py",
            "--study",
            study_id,
            "--output",
            str(output),
        ],
    )

    lab_controller.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    state = _load_state(tmp_path)
    assert payload["result"]["status"] == "passed"
    assert state["status"] == "idle"
    assert state["stop_reason"] is None
    assert set(state["completed_studies"]) == (
        set(registry["studies"]) - {"two-step-delayed-credit-v1"}
    )
    assert not (tmp_path / "lab.lock").exists()


def test_ninth_study_end_to_end_leaves_tenth_study_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = lab_controller._load_study_registry()
    study_id = "two-step-delayed-credit-v1"
    entry = lab_controller._study_entry(registry, study_id)
    snapshot = {
        "committed_file_sha256": entry["approved_file_sha256"],
        "committed_source_paths": entry["source_paths"],
        "revision": "a" * 40,
    }
    initial_state = lab_controller._default_state()
    initial_state["status"] = "idle"
    initial_state["completed_studies"] = {
        name: {
            "cycle_id": f"prior-{index}",
            "result_sha256": format(index + 1, "x") * 64,
            "revision": format(index + 2, "x") * 40,
            "status": "passed",
        }
        for index, name in enumerate(registry["studies"])
        if name not in {study_id, "multistep-td-action-prefix-v3"}
    }
    lab_controller._write_mutable_json(tmp_path / "lab-state.json", initial_state)
    output = tmp_path / "cycles" / "ninth-study-test" / "result.json"
    complete_result = _completed_two_step_delayed_credit_result()
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(
        lab_controller, "_repository_snapshot", lambda _entry: snapshot
    )
    monkeypatch.setattr(lab_controller, "_git", lambda *_args: "a" * 40)
    monkeypatch.setattr(
        lab_controller,
        "_run_worker",
        lambda *_args, **_kwargs: (
            complete_result,
            {
                "stderr_bytes": 0,
                "stderr_sha256": "0" * 64,
                "stdout_bytes": 0,
            },
        ),
    )
    monkeypatch.setattr(
        lab_controller.sys,
        "argv",
        [
            "run_local_lab.py",
            "--study",
            study_id,
            "--output",
            str(output),
        ],
    )

    lab_controller.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    state = _load_state(tmp_path)
    assert payload["result"]["status"] == "passed"
    assert state["status"] == "idle"
    assert state["stop_reason"] is None
    assert set(state["completed_studies"]) == (
        set(registry["studies"]) - {"multistep-td-action-prefix-v3"}
    )
    assert not (tmp_path / "lab.lock").exists()


def test_tenth_study_end_to_end_closes_pending_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = lab_controller._load_study_registry()
    study_id = "multistep-td-action-prefix-v3"
    entry = lab_controller._study_entry(registry, study_id)
    snapshot = {
        "committed_file_sha256": entry["approved_file_sha256"],
        "committed_source_paths": entry["source_paths"],
        "revision": "b" * 40,
    }
    initial_state = lab_controller._default_state()
    initial_state["status"] = "idle"
    initial_state["completed_studies"] = {
        name: {
            "cycle_id": f"prior-{index}",
            "result_sha256": format(index + 1, "x") * 64,
            "revision": format(index + 2, "x") * 40,
            "status": "passed",
        }
        for index, name in enumerate(registry["studies"])
        if name != study_id
    }
    lab_controller._write_mutable_json(tmp_path / "lab-state.json", initial_state)
    output = tmp_path / "cycles" / "tenth-study-test" / "result.json"
    complete_result = _completed_multistep_v3_result(entry)
    monkeypatch.setattr(lab_controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(
        lab_controller, "_repository_snapshot", lambda _entry: snapshot
    )
    monkeypatch.setattr(lab_controller, "_git", lambda *_args: "b" * 40)
    monkeypatch.setattr(
        lab_controller,
        "_run_worker",
        lambda *_args, **_kwargs: (
            complete_result,
            {
                "stderr_bytes": 0,
                "stderr_sha256": "0" * 64,
                "stdout_bytes": 0,
            },
        ),
    )
    monkeypatch.setattr(
        lab_controller.sys,
        "argv",
        [
            "run_local_lab.py",
            "--study",
            study_id,
            "--output",
            str(output),
        ],
    )

    lab_controller.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    state = _load_state(tmp_path)
    assert payload["result"]["status"] == "passed"
    assert state["status"] == "awaiting_study"
    assert state["stop_reason"] == "no_approved_study_pending"
    assert set(state["completed_studies"]) == set(registry["studies"])
    assert not (tmp_path / "lab.lock").exists()
