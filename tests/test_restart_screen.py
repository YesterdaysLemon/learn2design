from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.uifo_paired.optimizer_settings import (
    BATCHED_SETTINGS,
    settings_with_patience,
    validate_batched_settings,
)
from experiments.uifo_paired.plan import build_plan
from experiments.uifo_paired.restart_analysis import summarize_restart_records
from experiments.uifo_paired.runner import (
    _require_provider_time,
    _run_config,
    _validate_mechanics_revision,
    _validate_resolved_topology_identities,
    orchestrate,
    parse_arm_patience,
)


ROOT = Path(__file__).parents[1]
PANEL_ROOT = ROOT / "experiments" / "uifo_paired" / "panels"
PROVIDER_STOP = "2099-01-01T00:00:00Z"


def _mechanics_evidence() -> dict[str, object]:
    return {
        "format_version": 1,
        "study_profile": "restart-mechanics-v1",
        "plan_id": "1" * 16,
        "project_revision": "2" * 40,
        "package_sha256": "3" * 64,
        "package_manifest_sha256": "4" * 64,
        "record_sha256": "5" * 64,
        "history_sha256": "6" * 64,
        "optimizer_telemetry_sha256": "7" * 64,
        "decision_status": "passed",
        "decision_action": "run_restart_screen_v1",
    }


def _panel(name: str) -> tuple[dict[str, object], dict[str, object]]:
    path = PANEL_ROOT / f"{name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = {
        "source_kind": "json_topology_panel",
        "source_name": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "archive_exclusion_verified": True,
        "official_dataset_sha256": "test-only",
        "panel_id": payload["panel_id"],
        "topology_count": len(payload["topologies"]),
    }
    return payload, metadata


def _mechanics_plan() -> dict[str, object]:
    panel, metadata = _panel("restart-mechanics-v1")
    return build_plan(
        topology_seeds=None,
        topologies=panel["topologies"],
        optimizer_seeds=[11],
        arms=["no_prior_p200"],
        max_time_seconds=600.0,
        max_evals=None,
        population_size=8,
        n_frequencies=50,
        target_losses=[4.0, 1.0, 0.5, 0.0],
        worker_timeout_seconds=1_200,
        topology_panel=metadata,
        require_a100=True,
        minimum_gpu_memory_mib=75_000,
        max_idle_gpu_memory_mib=1_000,
        max_idle_gpu_utilization_percent=5,
        minimum_free_disk_gib=20,
        max_session_wall_seconds=1_800,
        max_worker_failures=1,
        study_profile="restart-mechanics-v1",
        optimizer_telemetry="member-v1",
        arm_patience={"no_prior_p200": 200},
        provider_stop_utc=PROVIDER_STOP,
        provider_evacuation_reserve_seconds=1_800,
    )


def _screen_plan(*, include_mechanics_evidence: bool = True) -> dict[str, object]:
    panel, metadata = _panel("restart-screen-v1")
    return build_plan(
        topology_seeds=None,
        topologies=panel["topologies"],
        optimizer_seeds=[19, 23],
        arms=["no_prior_p600", "no_prior_p200"],
        max_time_seconds=600,
        max_evals=None,
        population_size=8,
        n_frequencies=50,
        target_losses=[4.0, 1.0, 0.5, 0.0],
        worker_timeout_seconds=1_200,
        topology_panel=metadata,
        require_a100=True,
        minimum_gpu_memory_mib=75_000,
        max_idle_gpu_memory_mib=1_000,
        max_idle_gpu_utilization_percent=5,
        minimum_free_disk_gib=20,
        max_session_wall_seconds=int(6.5 * 60 * 60),
        max_worker_failures=1,
        study_profile="restart-screen-v1",
        arm_patience={"no_prior_p600": 600, "no_prior_p200": 200},
        pair_order_policy="alternate_topology_and_seed",
        mechanics_evidence=(
            _mechanics_evidence() if include_mechanics_evidence else None
        ),
        provider_stop_utc=PROVIDER_STOP,
        provider_evacuation_reserve_seconds=1_800,
    )


def _expected_configs(plan: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        run["run_id"]: _run_config(run, plan["configuration"])
        for run in plan["runs"]
    }


def _record(
    config: dict[str, object],
    *,
    loss: float,
    status: str = "complete",
) -> dict[str, object]:
    topology = str(config["topology"]["value"])
    return {
        "run_id": config["run_id"],
        "status": status,
        "started_utc": (
            f"2026-08-21T12:{int(config['planned_run_index']):02d}:00+00:00"
        ),
        "config": config,
        "metrics": {
            "has_finite_feasible": True,
            "best_feasible_loss": loss,
        },
        "objective_accounting": {
            "eval_count": 400 + int(config["planned_run_index"]),
        },
        "problem": {
            "topology_sha256": hashlib.sha256(topology.encode()).hexdigest(),
            "topology_string": topology,
        },
    }


def _passing_screen_records(
    expected: dict[str, dict[str, object]],
    *,
    topology_differences: list[float] | None = None,
    seed_adjustments: dict[int, float] | None = None,
) -> list[dict[str, object]]:
    topology_order = []
    for config in expected.values():
        topology = str(config["topology"]["value"])
        if topology not in topology_order:
            topology_order.append(topology)
    differences = topology_differences or [
        -0.20,
        -0.18,
        -0.14,
        -0.12,
        -0.10,
        -0.08,
        0.01,
        0.02,
    ]
    by_topology = dict(zip(topology_order, differences))
    records = []
    for config in expected.values():
        topology = str(config["topology"]["value"])
        loss = 1.0
        if config["arm"] == "no_prior_p200":
            loss += by_topology[topology]
            if seed_adjustments:
                loss += seed_adjustments[int(config["optimizer_seed"])]
        records.append(_record(config, loss=loss))
    return records


def test_restart_settings_are_exact_and_validated() -> None:
    p200 = settings_with_patience(200)
    assert p200 == {**BATCHED_SETTINGS, "patience": 200}
    assert validate_batched_settings(p200) == p200
    with pytest.raises(ValueError, match="exact schema"):
        validate_batched_settings({"patience": 200})
    with pytest.raises(ValueError, match="positive integer"):
        settings_with_patience(True)


def test_historical_development_v2_plan_id_remains_exact() -> None:
    panel, _ = _panel("development-v1")
    topology_panel = {
        "source_kind": "json_topology_panel",
        "source_name": "development-v1.json",
        "source_sha256": (
            "d5f660261e413f59b179d4fadf1f157b30f117aa265fd230d1d130bd6d69246b"
        ),
        "archive_exclusion_verified": True,
        "official_dataset_sha256": (
            "149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7"
        ),
        "panel_id": "development-v1",
        "topology_count": 16,
        "archive_exclusion_audit": {
            "method": "exact topology-string set intersection",
            "official_dataset": {
                "entries": 29650,
                "overlap_count": 0,
                "sha256": (
                    "149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7"
                ),
                "size_bytes": 74920439,
                "source_name": "dataset.h5",
                "unique_topologies": 12437,
            },
            "panel_identity_sha256": (
                "e5c93e125a66c8ccf7d1da997110ff59f04f5dd776a84d2a5e25a9ec34ca697c"
            ),
            "panel_topology_count": 16,
            "prior_panels": [],
        },
    }
    plan = build_plan(
        topology_seeds=None,
        topologies=panel["topologies"],
        optimizer_seeds=[7, 11],
        arms=["no_prior", "semantic_prior"],
        max_time_seconds=600.0,
        max_evals=None,
        population_size=8,
        n_frequencies=50,
        target_losses=[4.0, 1.0, 0.5, 0.0],
        worker_timeout_seconds=1_200,
        topology_panel=topology_panel,
        require_a100=True,
        minimum_gpu_memory_mib=75_000,
        max_idle_gpu_memory_mib=1_000,
        max_idle_gpu_utilization_percent=5,
        minimum_free_disk_gib=20.0,
        max_session_wall_seconds=float(16 * 60 * 60),
        max_worker_failures=2,
        study_profile="development-v2",
    )
    assert plan["plan_id"] == "49ff0e783f4f6a10"


def test_restart_mechanics_profile_is_bound_and_loss_blind() -> None:
    plan = _mechanics_plan()
    assert len(plan["runs"]) == 1
    assert plan["configuration"]["optimizer_telemetry"] == "member-v1"
    assert plan["configuration"]["arm_optimizer_settings"]["no_prior_p200"][
        "patience"
    ] == 200
    expected = _expected_configs(plan)
    config = next(iter(expected.values()))
    record = _record(config, loss=999.0)
    record["metrics"] = {
        "has_finite_feasible": False,
        "best_feasible_loss": None,
    }
    record["optimizer_telemetry"] = {
        "mode": "member-v1",
        "summary": {
            "restart_rows": 1,
            "post_restart_evaluation_rows": 1,
        },
    }
    record["worker_process"] = {"full_wall_seconds": 700.0}
    summary = summarize_restart_records([record], expected)
    assert summary["predeclared_decision"] == {
        "status": "passed",
        "passed": True,
        "action": "run_restart_screen_v1",
    }
    assert summary["mechanics"]["loss_excluded_from_inference"] is True
    assert "best_feasible_loss" not in summary["mechanics"]


def test_restart_screen_requires_authenticated_mechanics_evidence() -> None:
    with pytest.raises(ValueError, match="authenticated mechanics evidence"):
        _screen_plan(include_mechanics_evidence=False)

    evidence = _mechanics_evidence()
    evidence["decision_status"] = "failed"
    panel, metadata = _panel("restart-screen-v1")
    with pytest.raises(ValueError, match="invalid decision_status"):
        build_plan(
            topology_seeds=None,
            topologies=panel["topologies"],
            optimizer_seeds=[19, 23],
            arms=["no_prior_p600", "no_prior_p200"],
            max_time_seconds=600,
            max_evals=None,
            population_size=8,
            n_frequencies=50,
            target_losses=[4.0, 1.0, 0.5, 0.0],
            worker_timeout_seconds=1_200,
            topology_panel=metadata,
            require_a100=True,
            minimum_gpu_memory_mib=75_000,
            max_idle_gpu_memory_mib=1_000,
            max_idle_gpu_utilization_percent=5,
            minimum_free_disk_gib=20,
            max_session_wall_seconds=int(6.5 * 60 * 60),
            max_worker_failures=1,
            study_profile="restart-screen-v1",
            arm_patience={"no_prior_p600": 600, "no_prior_p200": 200},
            pair_order_policy="alternate_topology_and_seed",
            mechanics_evidence=evidence,
            provider_stop_utc=PROVIDER_STOP,
            provider_evacuation_reserve_seconds=1_800,
        )


def test_restart_screen_plan_uses_fresh_seeds_and_exact_order_balance() -> None:
    plan = _screen_plan()
    assert len(plan["runs"]) == 32
    assert plan["configuration"]["optimizer_seeds"] == [19, 23]
    assert "optimizer_telemetry" not in plan["configuration"]
    assert plan["primary_pair_order"] == {
        "complete_primary_pairs": 16,
        "no_prior_p600_first": 8,
        "no_prior_p200_first": 8,
        "absolute_imbalance": 0,
    }
    by_seed = {19: [0, 0], 23: [0, 0]}
    for run in plan["runs"]:
        if run["run_order_within_pair"] == 0:
            arm_index = 0 if run["arm"] == "no_prior_p600" else 1
            by_seed[run["optimizer_seed"]][arm_index] += 1
    assert by_seed == {19: [4, 4], 23: [4, 4]}


def test_restart_screen_panel_recomputes_outcome_blind_selection() -> None:
    development, _ = _panel("development-v1")
    mechanics, _ = _panel("restart-mechanics-v1")
    screen, _ = _panel("restart-screen-v1")
    submission_like, _ = _panel("submission-like-v1")
    domain = screen["generation"]["selection_domain"]
    excluded = mechanics["topologies"][0]
    ranked = sorted(
        (
            hashlib.sha256(f"{domain}\0{topology}".encode()).hexdigest(),
            topology,
        )
        for topology in development["topologies"]
        if topology != excluded
    )
    assert screen["topologies"] == [topology for _, topology in ranked[:8]]
    assert [member["selection_sha256"] for member in screen["members"]] == [
        digest for digest, _ in ranked[:8]
    ]
    development_seeds = {
        member["topology"]: member["topology_seed"]
        for member in development["members"]
    }
    assert {
        member["topology"]: member["topology_seed"]
        for member in screen["members"]
    } == {
        topology: development_seeds[topology] for topology in screen["topologies"]
    }
    assert mechanics["members"][0]["topology_seed"] == development_seeds[
        mechanics["topologies"][0]
    ]
    assert not (set(screen["topologies"]) & set(submission_like["topologies"]))


def test_restart_plan_rejects_partial_or_mixed_arm_settings() -> None:
    kwargs = {
        "topology_seeds": [1],
        "topologies": None,
        "optimizer_seeds": [19],
        "arms": ["no_prior_p600", "no_prior_p200"],
        "max_time_seconds": 600,
        "max_evals": None,
        "population_size": 8,
        "n_frequencies": 50,
    }
    with pytest.raises(ValueError, match="exact per-arm patience"):
        build_plan(**kwargs, arm_patience={"no_prior_p200": 200})
    with pytest.raises(ValueError, match="cannot be mixed"):
        build_plan(
            **{**kwargs, "arms": ["no_prior", "no_prior_p200"]},
            arm_patience={"no_prior": 600, "no_prior_p200": 200},
        )


def test_restart_screen_summary_uses_topology_as_inference_unit() -> None:
    plan = _screen_plan()
    expected = _expected_configs(plan)
    summary = summarize_restart_records(_passing_screen_records(expected), expected)
    assert summary["complete_optimizer_seed_pairs"] == 16
    assert summary["finite_comparable_topologies"] == 8
    assert summary["wins_ties_losses"] == {
        "p200_wins": 6,
        "ties": 0,
        "p200_losses": 2,
    }
    assert summary["exact_sign_flip_assignments"] == 256
    assert summary["predeclared_decision"]["status"] == "passed"
    assert summary["predeclared_decision"]["action"] == (
        "plan_untouched_submission_like_gate"
    )
    assert len(summary["topology_differences"]) == 8
    assert len(summary["optimizer_seed_pair_rows"]) == 16

    provisional = summarize_restart_records(
        _passing_screen_records(expected), expected, compute_bootstrap=False
    )
    assert provisional["predeclared_decision"]["status"] == "pending"
    assert provisional["predeclared_decision"]["action"] is None
    assert provisional["predeclared_decision"]["criteria"][
        "complete_records_revalidated"
    ] is False


def test_restart_screen_summary_fails_closed_on_drift_or_worker_error() -> None:
    plan = _screen_plan()
    expected = _expected_configs(plan)
    records = _passing_screen_records(expected)
    records[0] = {**records[0], "status": "error"}
    summary = summarize_restart_records(records, expected)
    assert summary["predeclared_decision"] == {
        "status": "failed",
        "passed": False,
        "action": "retain_patience_600",
        "criteria": summary["predeclared_decision"]["criteria"],
    }

    records = _passing_screen_records(expected)
    treatment = next(
        record for record in records if record["config"]["arm"] == "no_prior_p200"
    )
    treatment["config"] = {
        **treatment["config"],
        "optimizer_settings": {
            **treatment["config"]["optimizer_settings"],
            "learning_rate_low": 0.04,
        },
    }
    with pytest.raises(ValueError, match="differ beyond patience"):
        summarize_restart_records(records, expected)


@pytest.mark.parametrize("censored_arms", [("no_prior_p200",), ("no_prior_p600",), ("no_prior_p600", "no_prior_p200")])
def test_restart_screen_censoring_is_explicitly_not_promotable(
    censored_arms: tuple[str, ...],
) -> None:
    plan = _screen_plan()
    expected = _expected_configs(plan)
    records = _passing_screen_records(expected)
    pair_id = records[0]["config"]["pair_id"]
    for record in records:
        if record["config"]["pair_id"] == pair_id and record["config"]["arm"] in censored_arms:
            record["metrics"] = {
                "has_finite_feasible": False,
                "best_feasible_loss": None,
            }
    summary = summarize_restart_records(records, expected)
    assert summary["predeclared_decision"]["status"] == "failed"
    assert summary["predeclared_decision"]["action"] == "retain_patience_600"
    assert summary["predeclared_decision"]["criteria"][
        "all_pairs_finite_comparable"
    ] is False


def test_restart_screen_frozen_decision_boundaries() -> None:
    plan = _screen_plan()
    expected = _expected_configs(plan)

    five_wins = summarize_restart_records(
        _passing_screen_records(
            expected,
            topology_differences=[-0.2] * 5 + [0.01] * 3,
        ),
        expected,
    )
    assert five_wins["predeclared_decision"]["criteria"][
        "minimum_topology_wins_met"
    ] is False

    median_boundary = summarize_restart_records(
        _passing_screen_records(
            expected,
            topology_differences=[-0.05] * 6 + [0.01, 0.02],
        ),
        expected,
    )
    assert median_boundary["predeclared_decision"]["criteria"][
        "median_difference_at_most_negative_0_05"
    ] is True
    median_above = summarize_restart_records(
        _passing_screen_records(
            expected,
            topology_differences=[-0.049] * 6 + [0.01, 0.02],
        ),
        expected,
    )
    assert median_above["predeclared_decision"]["criteria"][
        "median_difference_at_most_negative_0_05"
    ] is False

    mean_zero = summarize_restart_records(
        _passing_screen_records(
            expected,
            topology_differences=[-0.125] * 6 + [0.375, 0.375],
        ),
        expected,
    )
    assert mean_zero["predeclared_decision"]["criteria"][
        "mean_difference_below_zero"
    ] is False

    high_regret = summarize_restart_records(
        _passing_screen_records(
            expected,
            topology_differences=[-0.3] * 6 + [0.6, 0.6],
        ),
        expected,
    )
    assert high_regret["predeclared_decision"]["criteria"][
        "p90_regret_at_most_0_5"
    ] is False
    regret_boundary = summarize_restart_records(
        _passing_screen_records(
            expected,
            topology_differences=[-0.3] * 6 + [0.5, 0.5],
        ),
        expected,
    )
    assert regret_boundary["topology_p90_regret"] == pytest.approx(0.5)
    assert regret_boundary["predeclared_decision"]["criteria"][
        "p90_regret_at_most_0_5"
    ] is True

    seed_inconsistent = summarize_restart_records(
        _passing_screen_records(
            expected,
            seed_adjustments={19: 0.2, 23: -0.2},
        ),
        expected,
    )
    assert seed_inconsistent["predeclared_decision"]["criteria"][
        "both_seed_mean_differences_below_zero"
    ] is False


def test_restart_screen_ties_missing_rows_and_exploratory_outputs() -> None:
    plan = _screen_plan()
    expected = _expected_configs(plan)
    records = _passing_screen_records(
        expected,
        topology_differences=[-0.2] * 6 + [0.0, 0.1],
    )
    first = summarize_restart_records(records, expected)
    second = summarize_restart_records(records, expected)
    assert first["wins_ties_losses"] == {
        "p200_wins": 6,
        "ties": 1,
        "p200_losses": 1,
    }
    assert first["topology_bootstrap_mean_difference_ci_95"] == second[
        "topology_bootstrap_mean_difference_ci_95"
    ]
    exploratory = first["exploratory_sensitivity"]
    assert len(exploratory["leave_one_topology_out"]) == 8
    assert len(exploratory["arm_first"]["topology_contrasts"]) == 8
    assert exploratory["serial_order"] is not None
    assert exploratory["evaluation_throughput"] is not None
    assert exploratory["changes_frozen_decision"] is False
    first_loo = exploratory["leave_one_topology_out"][0]
    assert first_loo["mean_difference"] == pytest.approx(-0.15714285714285708)
    assert first_loo["median_difference"] == pytest.approx(-0.2)
    assert first_loo["p90_regret"] == pytest.approx(-0.08)
    assert exploratory["arm_first"]["mean_contrast"] == pytest.approx(0.0)
    assert exploratory["arm_first"][
        "exact_sign_flip_mean_pvalue_two_sided"
    ] == pytest.approx(1.0)
    assert exploratory["serial_order"][
        "spearman_planned_run_index_vs_difference"
    ] == pytest.approx(0.7637626158259733)
    assert exploratory["serial_order"][
        "spearman_session_start_vs_difference"
    ] == pytest.approx(0.7637626158259733)
    assert exploratory["evaluation_throughput"]["mean"] == pytest.approx(
        2.4286551279800337e-08
    )
    assert exploratory["evaluation_throughput"]["topology_values"][0] == (
        pytest.approx(2.398767636961727e-06)
    )

    missing = summarize_restart_records(records[:-1], expected)
    assert missing["predeclared_decision"]["status"] == "pending"
    assert missing["predeclared_decision"]["action"] is None


def test_parse_arm_patience_is_strict() -> None:
    assert parse_arm_patience(["no_prior_p600=600", "no_prior_p200=200"]) == {
        "no_prior_p600": 600,
        "no_prior_p200": 200,
    }
    with pytest.raises(ValueError, match="duplicate"):
        parse_arm_patience(["x=1", "x=2"])
    with pytest.raises(ValueError, match="ARM=INTEGER"):
        parse_arm_patience(["x"])


def test_restart_profiles_are_non_resumable_and_deadline_guarded(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="non-resumable"):
        orchestrate(_mechanics_plan(), tmp_path / "study", resume=True)
    with pytest.raises(RuntimeError, match="insufficient provider time"):
        _require_provider_time(
            {"provider_stop_utc": "2000-01-01T00:00:00Z"},
            1.0,
            "start test",
        )
    with pytest.raises(RuntimeError, match="eight-hour horizon"):
        _require_provider_time(
            {
                "provider_stop_utc": "2099-01-01T00:00:00Z",
                "provider_deadline_maximum_horizon_seconds": 8 * 60 * 60,
            },
            1.0,
            "start test",
        )
    with pytest.raises(RuntimeError, match="differs from passed mechanics"):
        _validate_mechanics_revision(
            {"mechanics_evidence": _mechanics_evidence()}, "f" * 40
        )


def test_one_configured_topology_cannot_resolve_to_multiple_hashes() -> None:
    topology = {"kind": "string", "value": "AAAAAAAAA-LLLLLLLLLLLD"}
    records = [
        {
            "status": "complete",
            "config": {"topology": topology},
            "problem": {"topology_sha256": digest},
        }
        for digest in ("a" * 64, "b" * 64)
    ]
    with pytest.raises(RuntimeError, match="multiple topology identities"):
        _validate_resolved_topology_identities(records)
