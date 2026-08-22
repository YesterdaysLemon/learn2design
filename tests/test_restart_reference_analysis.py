from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.uifo_paired.plan import build_plan
from experiments.uifo_paired.restart_analysis import summarize_restart_records
from experiments.uifo_paired.restart_reference_analysis import (
    reference_restart_screen,
)
from experiments.uifo_paired.restart_posthoc_analysis import (
    analyze_restart_posthoc,
    create_restart_plots,
)
from experiments.uifo_paired.restart_results_ingestion import (
    load_restart_summary_after_reproduction,
)
from experiments.uifo_paired.results_ingestion import (
    SourcePaths,
    StudyValidationError,
    ValidatedStudy,
)
from experiments.uifo_paired.results_workflow import (
    compare_restart_production_and_reference,
)
from experiments.uifo_paired.runner import _run_config
from tools.analyze_restart_screen import _next_gate_text


ROOT = Path(__file__).parents[1]


def _plan() -> dict[str, object]:
    panel_path = ROOT / "experiments/uifo_paired/panels/restart-screen-v1.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    mechanics = {
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
    return build_plan(
        topology_seeds=None,
        topologies=list(panel["topologies"]),
        optimizer_seeds=[19, 23],
        arms=["no_prior_p600", "no_prior_p200"],
        max_time_seconds=600.0,
        max_evals=None,
        population_size=8,
        n_frequencies=50,
        target_losses=[4.0, 1.0, 0.5, 0.0],
        allow_cpu=False,
        worker_timeout_seconds=1_200.0,
        topology_panel={
            "source_kind": "json_topology_panel",
            "source_name": panel_path.name,
            "source_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
            "archive_exclusion_verified": True,
            "official_dataset_sha256": "test-only",
            "panel_id": "restart-screen-v1",
            "topology_count": 8,
        },
        require_a100=True,
        minimum_gpu_memory_mib=75_000,
        max_idle_gpu_memory_mib=1_000,
        max_idle_gpu_utilization_percent=5,
        minimum_free_disk_gib=20.0,
        max_session_wall_seconds=23_400.0,
        max_worker_failures=1,
        study_profile="restart-screen-v1",
        arm_patience={"no_prior_p600": 600, "no_prior_p200": 200},
        pair_order_policy="alternate_topology_and_seed",
        mechanics_evidence=mechanics,
        provider_stop_utc="2099-01-01T00:00:00Z",
        provider_evacuation_reserve_seconds=1_800.0,
    )


def _study(
    differences: list[float] | None = None,
) -> ValidatedStudy:
    plan = _plan()
    configs = {
        str(run["run_id"]): _run_config(run, plan["configuration"])
        for run in plan["runs"]
    }
    topology_order = [
        str(topology["value"]) for topology in plan["configuration"]["topologies"]
    ]
    values = differences or [-0.20, -0.18, -0.14, -0.12, -0.10, -0.08, 0.01, 0.02]
    by_topology = dict(zip(topology_order, values))
    records = []
    histories: dict[str, list[dict[str, object]]] = {}
    for config in configs.values():
        topology = str(config["topology"]["value"])
        loss = 1.0 + (
            by_topology[topology] if config["arm"] == "no_prior_p200" else 0.0
        )
        run_id = str(config["run_id"])
        histories[run_id] = [
            {
                "call_index": 0,
                "candidate_index": 0,
                "eval_count_after_call": 8,
                "time_seconds": 1.0,
                "loss": loss,
                "sensitivity_loss": 0.0,
                "penalty": 0.0,
                "is_feasible": True,
            }
        ]
        records.append(
            {
                "run_id": run_id,
                "status": "complete",
                "started_utc": "2026-08-21T12:00:00+00:00",
                "config": config,
                "metrics": {
                    "has_feasible": True,
                    "has_finite_feasible": True,
                    "best_feasible_loss": loss,
                    "last_logged_time_seconds": 1.0,
                    "last_logged_eval_count": 8,
                    "targets": {
                        format(target, ".12g"): {
                            "time_seconds": 1.0 if loss <= target else None,
                            "eval_count": 8 if loss <= target else None,
                        }
                        for target in (4.0, 1.0, 0.5, 0.0)
                    },
                },
                "objective_accounting": {"eval_count": 8},
                "problem": {
                    "topology_sha256": hashlib.sha256(topology.encode()).hexdigest(),
                    "topology_string": topology,
                },
            }
        )
    records.sort(key=lambda record: str(record["run_id"]))
    return ValidatedStudy(
        sources=SourcePaths(*(Path(name) for name in ("a", "b", "c", "d"))),
        source_hashes={},
        archive_members=(),
        plan=plan,
        manifest=plan,
        package_state={},
        session={"started_utc": "2026-08-21T12:00:00+00:00"},
        configs=configs,
        records=records,
        history_rows=histories,
        integrity={"summary_content_opened": False},
    )


def test_reference_matches_production_on_passing_fixture() -> None:
    study = _study()
    production = summarize_restart_records(study.records, study.configs)
    reference = reference_restart_screen(study)
    agreement = compare_restart_production_and_reference(production, reference)
    assert agreement == {
        "status": "matched",
        "topology_values_compared": 8,
        "seed_pairs_compared": 16,
        "frozen_criteria_compared": 11,
        "relative_tolerance": 1e-12,
        "absolute_tolerance": 1e-12,
    }
    assert reference["topology_macro_mean_difference"] == pytest.approx(-0.09875)
    assert reference["topology_macro_median_difference"] == pytest.approx(-0.11)
    assert reference["topology_p90_regret"] == pytest.approx(0.013)
    assert reference["wins_ties_losses"] == {
        "p200_wins": 6,
        "ties": 0,
        "p200_losses": 2,
    }
    assert reference["predeclared_decision"]["action"] == (
        "plan_untouched_submission_like_gate"
    )


def test_frozen_replay_defers_exploratory_tests() -> None:
    study = _study()
    production = summarize_restart_records(
        study.records,
        study.configs,
        include_exploratory=False,
    )
    reference = reference_restart_screen(study, include_exploratory=False)
    assert production["topology_bootstrap_mean_difference_ci_95"] is not None
    assert production["exact_sign_flip_mean_pvalue_two_sided"] is None
    assert production["exact_sign_test_pvalue_two_sided"] is None
    assert production["exploratory_sensitivity"] == {
        "ready": False,
        "deferred_until_frozen_replay_match": True,
        "changes_frozen_decision": False,
    }
    assert reference["exact_sign_flip_mean_pvalue_two_sided"] is None
    assert reference["exact_sign_test_pvalue_two_sided"] is None
    agreement = compare_restart_production_and_reference(production, reference)
    assert agreement["status"] == "matched"


def test_reference_uses_history_and_rejects_metric_drift() -> None:
    study = _study()
    run_id = next(iter(study.history_rows))
    study.history_rows[run_id][0]["loss"] = 999.0
    with pytest.raises(StudyValidationError, match="best loss mismatch"):
        reference_restart_screen(study)


def test_reference_rejects_broken_pair_hierarchy() -> None:
    study = _study()
    treatment = next(
        record
        for record in study.records
        if record["config"]["arm"] == "no_prior_p200"
    )
    treatment["config"] = {
        **treatment["config"],
        "pair_id": "crossed-pair",
    }
    study.configs[str(treatment["run_id"])] = treatment["config"]
    with pytest.raises(StudyValidationError, match="16 seed pairs"):
        reference_restart_screen(study)


def test_reference_tie_tolerance_is_numerical_not_equivalence() -> None:
    study = _study([0.0, 0.5e-12, -0.5e-12, 1.1e-12, -1.1e-12, -0.1, 0.1, 0.2])
    result = reference_restart_screen(study)
    assert result["wins_ties_losses"] == {
        "p200_wins": 2,
        "ties": 3,
        "p200_losses": 3,
    }


def test_three_way_comparison_fails_closed_on_discrepancy() -> None:
    study = _study()
    production = summarize_restart_records(study.records, study.configs)
    reference = reference_restart_screen(study)
    mutated = copy.deepcopy(reference)
    mutated["topology_p90_regret"] = 123.0
    with pytest.raises(StudyValidationError, match="topology_p90_regret"):
        compare_restart_production_and_reference(production, mutated)
    missing = copy.deepcopy(reference)
    del missing["topology_p90_regret"]
    with pytest.raises(StudyValidationError, match="topology_p90_regret is missing"):
        compare_restart_production_and_reference(production, missing)


def test_restart_summary_stays_locked_without_full_replay_agreement() -> None:
    study = _study()
    with pytest.raises(StudyValidationError, match="summary remains locked"):
        load_restart_summary_after_reproduction(
            study,
            {
                "status": "matched",
                "topology_values_compared": 8,
                "seed_pairs_compared": 15,
                "frozen_criteria_compared": 11,
            },
        )


def test_reference_module_has_no_banned_analysis_imports() -> None:
    path = ROOT / "experiments/uifo_paired/restart_reference_analysis.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    banned = {
        "experiments.uifo_paired.analysis",
        "experiments.uifo_paired.restart_analysis",
        "experiments.uifo_paired.reference_analysis",
        "experiments.uifo_paired.metrics",
    }
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not (imported & banned)


def test_generated_report_next_gate_follows_frozen_action() -> None:
    failed = _next_gate_text({"action": "retain_patience_600"})
    passed = _next_gate_text({"action": "plan_untouched_submission_like_gate"})
    assert "Retain the packaged patience-600" in failed
    assert "authorizes planning, but not launching" in passed
    assert "Retain the packaged patience-600" not in passed
    with pytest.raises(StudyValidationError, match="unsupported frozen restart action"):
        _next_gate_text({"action": "invented"})


@pytest.mark.integration
def test_restart_posthoc_and_plots_use_eight_topology_blocks(tmp_path: Path) -> None:
    pytest.importorskip("scipy", reason="restart post-hoc analysis dependency")
    pytest.importorskip("matplotlib", reason="restart plot dependency")
    study = _study()
    production = summarize_restart_records(study.records, study.configs)
    reference = reference_restart_screen(study)
    receipt = {
        "status": "matched",
        "archived_summary_compared": True,
        "topology_values_compared": 8,
        "seed_pairs_compared": 16,
        "frozen_criteria_compared": 11,
    }
    posthoc = analyze_restart_posthoc(
        study,
        production,
        reference,
        frozen_agreement=receipt,
    )
    assert posthoc["topology_count"] == 8
    assert posthoc["exact_mean_sign_flip"]["assignments_enumerated"] == 256
    assert posthoc["changes_frozen_decision"] is False
    assert len(posthoc["leave_one_topology_out"]) == 8
    paths = create_restart_plots(posthoc, tmp_path)
    assert len(paths) == 4
    assert all(path.stat().st_size > 1_000 for path in paths)


def test_restart_posthoc_requires_frozen_three_way_receipt() -> None:
    study = _study()
    production = summarize_restart_records(study.records, study.configs)
    reference = reference_restart_screen(study)
    with pytest.raises(StudyValidationError, match="matched frozen production"):
        analyze_restart_posthoc(
            study,
            production,
            reference,
            frozen_agreement={"status": "matched"},
        )


def test_restart_posthoc_rejects_missing_finite_loss_defensively() -> None:
    study = _study()
    study.records[0]["metrics"]["best_feasible_loss"] = None
    receipt = {
        "status": "matched",
        "archived_summary_compared": True,
        "topology_values_compared": 8,
        "seed_pairs_compared": 16,
        "frozen_criteria_compared": 11,
    }
    with pytest.raises(StudyValidationError, match="finite feasible loss"):
        analyze_restart_posthoc(
            study,
            {},
            {},
            frozen_agreement=receipt,
        )
