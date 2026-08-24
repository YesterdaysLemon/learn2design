from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.uifo_paired.results_ingestion import (
    SourcePaths,
    StudyValidationError,
    ValidatedStudy,
)
from experiments.uifo_paired.submission_like_reference_analysis import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FROZEN_POLICY,
    SUBMISSION_LIKE_TOPOLOGIES,
    TARGET_LOSSES,
    reference_submission_like_screen,
)
from experiments.uifo_paired.study_profiles import STUDY_PROFILES
from experiments.uifo_paired.submission_like_analysis import (
    summarize_submission_like_records,
)
from experiments.uifo_paired.submission_like_evidence import (
    compare_submission_like_archived_summary,
    compare_submission_like_replays,
)
from experiments.uifo_paired.submission_like_posthoc_analysis import (
    analyze_submission_like_posthoc,
    create_submission_like_plots,
    safe_submission_like_posthoc,
)
from experiments.uifo_paired.submission_like_trajectory_analysis import (
    analyze_submission_like_trajectories,
)


ROOT = Path(__file__).parents[1]
SEEDS = (29, 31)


def _archived_agreement() -> dict[str, object]:
    return {
        "status": "matched",
        "archived_summary_compared": True,
        "fields_compared": 21,
        "absolute_tolerance": 1e-12,
        "relative_tolerance": 1e-12,
    }


def _target_hits(loss: float | None) -> dict[str, dict[str, float | int | None]]:
    return {
        format(target, ".12g"): {
            "time_seconds": 2.0 if loss is not None and loss <= target else None,
            "eval_count": 16 if loss is not None and loss <= target else None,
        }
        for target in TARGET_LOSSES
    }


def _study(*, seed_gap: float = 0.2) -> ValidatedStudy:
    project_revision = "a" * 40
    upstream_reference = "dfbench@pinned-test-reference"
    package_evidence = {
        "format_version": 1,
        "archive_name": "submission.zip",
        "archive_sha256": "b" * 64,
        "builder_manifest_name": "submission.manifest.json",
        "builder_manifest_sha256": "c" * 64,
        "project_revision": project_revision,
        "source_files": [
            {
                "path": "submission/solution.py",
                "sha256": "d" * 64,
                "size_bytes": 123,
            }
        ],
        "upstream_reference": upstream_reference,
    }
    topology_specs = [
        {"kind": "string", "value": topology}
        for topology in SUBMISSION_LIKE_TOPOLOGIES
    ]
    configuration = {
        "study_profile": "submission-like-screen-v1",
        "arms": ["no_prior"],
        "optimizer_seeds": [29, 31],
        "target_losses": [4.0, 1.0, 0.5, 0.0],
        "topologies": topology_specs,
        "topology_panel": {
            "panel_id": "submission-like-v1",
            "topology_count": 10,
        },
        "decision_policy": dict(FROZEN_POLICY),
        "candidate_package_evidence": copy.deepcopy(package_evidence),
        "execution_mode": "serial",
        "resource_budget": {
            "currency": "USD",
            "gpu_count": 1,
            "maximum_gpu_hourly_price": 1.6,
            "maximum_provider_charge": 16.0,
            "maximum_provider_hours": 10.0,
            "planned_runs": 20,
            "scored_objective_seconds": 24_000,
        },
    }
    configs: dict[str, dict[str, object]] = {}
    records = []
    histories: dict[str, list[dict[str, object]]] = {}
    for topology_index, topology in enumerate(SUBMISSION_LIKE_TOPOLOGIES):
        topology_mean = float(topology_index + 1)
        for seed_index, seed in enumerate(SEEDS):
            loss = topology_mean + (-seed_gap / 2.0 if seed_index == 0 else seed_gap / 2.0)
            run_id = f"submission-like-{topology_index:02d}-{seed}"
            config = {
                "run_id": run_id,
                "pair_id": f"submission-like-pair-{topology_index:02d}-{seed}",
                "planned_run_index": len(configs),
                "run_order_within_pair": 0,
                "topology": {"kind": "string", "value": topology},
                "optimizer_seed": seed,
                "arm": "no_prior",
                "study_profile": "submission-like-screen-v1",
                "target_losses": [4.0, 1.0, 0.5, 0.0],
                "decision_policy": dict(FROZEN_POLICY),
                "candidate_package_evidence": copy.deepcopy(package_evidence),
            }
            rows = [
                {
                    "call_index": 0,
                    "candidate_index": 0,
                    "eval_count_after_call": 8,
                    "time_seconds": 1.0,
                    "loss": loss + 100.0,
                    "sensitivity_loss": 0.0,
                    "penalty": 1.0,
                    "is_feasible": False,
                },
                {
                    "call_index": 1,
                    "candidate_index": 0,
                    "eval_count_after_call": 16,
                    "time_seconds": 2.0,
                    "loss": loss,
                    "sensitivity_loss": 0.0,
                    "penalty": 0.0,
                    "is_feasible": True,
                },
            ]
            configs[run_id] = config
            histories[run_id] = rows
            records.append(
                {
                    "run_id": run_id,
                    "status": "complete",
                    "started_utc": (
                        f"2026-08-22T00:{config['planned_run_index']:02d}:00+00:00"
                    ),
                    "completed_utc": (
                        f"2026-08-22T00:{config['planned_run_index']:02d}:30+00:00"
                    ),
                    "config": config,
                    "metrics": {
                        "has_feasible": True,
                        "has_finite_feasible": True,
                        "best_feasible_loss": loss,
                        "targets": _target_hits(loss),
                    },
                    "problem": {
                        "topology_string": topology,
                        "topology_sha256": hashlib.sha256(topology.encode()).hexdigest(),
                    },
                }
            )
    records.sort(key=lambda record: str(record["run_id"]))
    plan = {"configuration": copy.deepcopy(configuration)}
    manifest = {
        "configuration": copy.deepcopy(configuration),
        "project_revision": project_revision,
        "upstream_reference": upstream_reference,
    }
    return ValidatedStudy(
        sources=SourcePaths(*(Path(name) for name in ("a", "b", "c", "d"))),
        source_hashes={},
        archive_members=(),
        plan=plan,
        manifest=manifest,
        package_state={},
        session={
            "status": "complete",
            "started_utc": "2026-08-22T00:00:00+00:00",
            "completed_utc": "2026-08-22T00:20:00+00:00",
        },
        configs=configs,
        records=records,
        history_rows=histories,
        integrity={"summary_content_opened": False},
    )


def test_reference_module_has_only_the_allowed_project_import() -> None:
    path = (
        ROOT
        / "experiments/uifo_paired/submission_like_reference_analysis.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    project_imports = []
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "experiments.uifo_paired"
        ):
            project_imports.append(node.module)
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            project_imports.extend(
                alias.name
                for alias in node.names
                if alias.name.startswith("experiments.uifo_paired")
            )
    assert project_imports == ["experiments.uifo_paired.results_ingestion"]
    assert imported_names == {"StudyValidationError", "ValidatedStudy"}


def test_reference_topologies_are_exactly_the_submission_like_panel() -> None:
    panel = json.loads(
        (
            ROOT / "experiments/uifo_paired/panels/submission-like-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert tuple(panel["topologies"]) == SUBMISSION_LIKE_TOPOLOGIES


def test_reference_policy_is_the_exact_frozen_profile_policy() -> None:
    assert FROZEN_POLICY == STUDY_PROFILES["submission-like-screen-v1"][
        "decision_policy"
    ]


def test_reference_recomputes_history_first_and_preserves_target_censoring() -> None:
    result = reference_submission_like_screen(_study())
    assert result["completed_runs"] == 20
    assert result["complete_topology_blocks"] == 10
    assert result["physical_feasible_runs"] == 20
    assert result["finite_feasible_runs"] == 20
    assert result["topology_macro_mean_best_feasible_loss"] == pytest.approx(5.5)
    assert result["topology_macro_median_best_feasible_loss"] == pytest.approx(5.5)
    assert result["topology_macro_p90_best_feasible_loss"] == pytest.approx(9.1)
    assert result["topology_p90_absolute_seed_gap"] == pytest.approx(0.2)
    assert result["predeclared_decision"]["status"] == "passed"
    assert result["predeclared_decision"]["action"] == (
        "candidate_evidence_complete_for_submission_review"
    )
    assert result["predeclared_decision"]["criteria"][
        "candidate_package_bound"
    ] == {"passed": True}
    target = result["target_hitting"]["4"]
    assert target["run_categories"] == {
        "reached": 7,
        "right_censored": 13,
        "incomplete": 0,
    }
    assert target["topology_categories"] == {
        "both_seeds_reached": 3,
        "one_seed_reached": 1,
        "neither_seed_reached": 6,
        "incomplete": 0,
    }


def test_reference_rejects_record_metrics_instead_of_using_them_as_input() -> None:
    study = _study()
    run_id = next(iter(study.history_rows))
    study.history_rows[run_id][-1]["loss"] = 999.0
    with pytest.raises(StudyValidationError, match="best loss mismatch"):
        reference_submission_like_screen(study)


@pytest.mark.parametrize("seed_gap", [0.5, 0.5001, 5.0])
def test_seed_gap_is_descriptive_and_does_not_change_frozen_status(
    seed_gap: float,
) -> None:
    result = reference_submission_like_screen(_study(seed_gap=seed_gap))
    assert result["topology_p90_absolute_seed_gap"] == pytest.approx(seed_gap)
    assert result["predeclared_decision"]["status"] == "passed"
    assert "topology_p90_absolute_seed_gap" not in result[
        "predeclared_decision"
    ]["criteria"]


@pytest.mark.parametrize(
    "corruption, message",
    [
        ("path", "source path is unsafe"),
        ("digest", "source digest is invalid"),
        ("revision", "project revision is not manifest-bound"),
        ("config", "config candidate package evidence mismatch"),
    ],
)
def test_candidate_package_provenance_is_strictly_plan_and_config_bound(
    corruption: str, message: str
) -> None:
    study = _study()
    evidence = study.plan["configuration"]["candidate_package_evidence"]
    assert isinstance(evidence, dict)
    if corruption == "path":
        evidence["source_files"][0]["path"] = "../submission/solution.py"
        study.manifest["configuration"] = copy.deepcopy(study.plan["configuration"])
    elif corruption == "digest":
        evidence["source_files"][0]["sha256"] = "not-a-digest"
        study.manifest["configuration"] = copy.deepcopy(study.plan["configuration"])
    elif corruption == "revision":
        evidence["project_revision"] = "e" * 40
        study.manifest["configuration"] = copy.deepcopy(study.plan["configuration"])
    else:
        first_config = next(iter(study.configs.values()))
        first_config["candidate_package_evidence"]["archive_sha256"] = "e" * 64
        record = next(
            record
            for record in study.records
            if record["run_id"] == first_config["run_id"]
        )
        record["config"] = copy.deepcopy(first_config)
    with pytest.raises(StudyValidationError, match=message):
        reference_submission_like_screen(study)


def test_missing_attempt_is_not_evaluable_and_censoring_fails_closed() -> None:
    missing = _study()
    run_id = next(iter(missing.configs))
    missing.configs.pop(run_id)
    missing.records = [record for record in missing.records if record["run_id"] != run_id]
    missing.history_rows.pop(run_id)
    result = reference_submission_like_screen(missing)
    assert result["predeclared_decision"] == {
        "policy_id": "no-prior-submission-like-screen-v1",
        "status": "not_evaluable",
        "passed": False,
        "action": "retain_candidate_attempt_not_evaluable",
        "criteria": result["predeclared_decision"]["criteria"],
    }
    assert result["completed_runs"] == 19
    assert result["complete_topology_blocks"] == 9

    censored = _study()
    record = censored.records[0]
    run_id = str(record["run_id"])
    censored.history_rows[run_id][-1]["loss"] = None
    record["metrics"] = {
        "has_feasible": True,
        "has_finite_feasible": False,
        "best_feasible_loss": None,
        "targets": _target_hits(None),
    }
    result = reference_submission_like_screen(censored)
    assert result["completed_runs"] == 20
    assert result["finite_feasible_runs"] == 19
    assert result["topology_p90_absolute_seed_gap"] is None
    assert result["predeclared_decision"]["status"] == "failed"
    assert result["predeclared_decision"]["action"] == (
        "retain_candidate_and_investigate_submission_like_reliability"
    )


@pytest.mark.parametrize("corruption", ["duplicate_seed", "topology_identity"])
def test_reference_rejects_hierarchy_and_topology_identity_corruption(
    corruption: str,
) -> None:
    study = _study()
    if corruption == "duplicate_seed":
        record = next(
            record
            for record in study.records
            if record["config"]["optimizer_seed"] == 31
        )
        run_id = str(record["run_id"])
        changed = {**study.configs[run_id], "optimizer_seed": 29}
        study.configs[run_id] = changed
        record["config"] = changed
        expected_message = "duplicate topology/seed cell"
    else:
        study.records[0]["problem"]["topology_sha256"] = "0" * 64
        expected_message = "topology identity mismatch"
    with pytest.raises(StudyValidationError, match=expected_message):
        reference_submission_like_screen(study)


def test_bootstrap_is_deterministic_and_uses_frozen_parameters() -> None:
    first = reference_submission_like_screen(_study())[
        "topology_bootstrap_mean_best_feasible_loss_ci_95"
    ]
    second = reference_submission_like_screen(_study())[
        "topology_bootstrap_mean_best_feasible_loss_ci_95"
    ]
    assert first == second
    assert first["seed"] == BOOTSTRAP_SEED
    assert first["resamples"] == BOOTSTRAP_RESAMPLES
    assert first["lower"] < 5.5 < first["upper"]


def test_reference_requires_summary_to_remain_sealed() -> None:
    study = _study()
    study.integrity["summary_content_opened"] = True
    with pytest.raises(StudyValidationError, match="summary sealed"):
        reference_submission_like_screen(study)


def test_production_reference_and_archived_comparison_is_fail_closed() -> None:
    study = _study()
    production = summarize_submission_like_records(study.records, study.configs)
    reference = reference_submission_like_screen(study)
    agreement = compare_submission_like_replays(production, reference)
    assert agreement == {
        "status": "matched",
        "runs_compared": 20,
        "topology_values_compared": 10,
        "target_thresholds_compared": 4,
        "frozen_criteria_compared": 5,
        "absolute_tolerance": 1e-12,
        "relative_tolerance": 1e-12,
    }
    assert compare_submission_like_archived_summary(production, copy.deepcopy(production))[
        "status"
    ] == "matched"

    changed = copy.deepcopy(reference)
    changed["topology_rows"][0]["topology_mean_best_feasible_loss"] += 1e-6
    with pytest.raises(StudyValidationError, match="replay mismatch"):
        compare_submission_like_replays(production, changed)


def test_failed_complete_replays_match_none_bootstrap_and_archive_is_key_strict() -> None:
    study = _study()
    record = study.records[0]
    run_id = str(record["run_id"])
    study.history_rows[run_id][-1]["loss"] = None
    record["metrics"] = {
        "has_feasible": True,
        "has_finite_feasible": False,
        "best_feasible_loss": None,
        "targets": _target_hits(None),
    }
    production = summarize_submission_like_records(study.records, study.configs)
    reference = reference_submission_like_screen(study)
    assert production["predeclared_decision"]["status"] == "failed"
    assert reference["predeclared_decision"]["status"] == "failed"
    assert production["topology_bootstrap_mean_loss_ci_95"] is None
    assert reference["topology_bootstrap_mean_best_feasible_loss_ci_95"] is None
    assert compare_submission_like_replays(production, reference)["status"] == "matched"

    archived = copy.deepcopy(production)
    archived.pop("topology_bootstrap_mean_loss_ci_95")
    with pytest.raises(StudyValidationError, match="summary schema mismatch"):
        compare_submission_like_archived_summary(production, archived)

    archived = copy.deepcopy(production)
    archived["predeclared_decision"]["passed"] = 0
    with pytest.raises(StudyValidationError, match=r"passed\.type"):
        compare_submission_like_archived_summary(production, archived)


def test_replay_compares_physical_feasibility_independently() -> None:
    study = _study()
    production = summarize_submission_like_records(study.records, study.configs)
    reference = reference_submission_like_screen(study)
    reference["run_rows"][0]["physical_feasible"] = False
    with pytest.raises(StudyValidationError, match=r"\.physical"):
        compare_submission_like_replays(production, reference)


def _posthoc(seed_gap: float = 0.2) -> tuple[dict[str, object], dict[str, object]]:
    pytest.importorskip("scipy", reason="submission-like analysis dependency")
    study = _study(seed_gap=seed_gap)
    production = summarize_submission_like_records(study.records, study.configs)
    reference = reference_submission_like_screen(study)
    return analyze_submission_like_posthoc(
        study,
        production,
        reference,
        agreement=_archived_agreement(),
    ), production


def test_posthoc_uses_topology_blocks_and_preserves_censor_bounds() -> None:
    posthoc, _production = _posthoc()
    assert posthoc["inference_unit"] == "topology (n=10)"
    assert posthoc["serial_drift"]["topology_pairs"] == 10
    assert len(posthoc["topology_rows"]) == 10
    assert len(posthoc["run_rows"]) == 20
    assert len(posthoc["leave_one_topology_out"]) == 10
    assert all(
        row["topologies_remaining"] == 9
        and row["decision_recomputed"] is False
        for row in posthoc["leave_one_topology_out"]
    )
    assert "serial_run_order_vs_loss" not in posthoc["serial_drift"]
    assert "scored_time_proxy_gap_vs_later_minus_earlier_loss" not in posthoc[
        "serial_drift"
    ]
    assert (
        "actual_start_gap_vs_later_minus_earlier_loss"
        in posthoc["serial_drift"]
    )
    assert posthoc["serial_drift"]["causal_effect_identified"] is False
    trajectory = posthoc["trajectory_alignment"]
    assert trajectory["inference_unit"] == "topology (n=10)"
    assert trajectory["seed_and_sweep_phase_confounded"] is True
    assert trajectory["changes_frozen_decision"] is False
    assert trajectory["evaluation_aligned"]["checkpoints"][-1][
        "complete_topologies"
    ] == 10
    assert trajectory["wall_time_aligned"]["checkpoints"][-1][
        "complete_topologies"
    ] == 10
    assert trajectory["evaluation_aligned"]["checkpoints"][-1][
        "topology_mean_seed_31_minus_seed_29_loss"
    ] == pytest.approx(0.2)

    target = posthoc["target_hitting"]["4"]
    assert target["topology_categories"] == {
        "both_seeds_reached": 3,
        "one_seed_reached": 1,
        "neither_seed_reached": 6,
        "incomplete": 0,
    }
    assert target["seed_29_only_topologies"] == 1
    assert target["seed_31_only_topologies"] == 0
    censored = next(
        hit
        for row in target["topology_rows"]
        for hit in row["seed_hits"].values()
        if not hit["event_reached"]
    )
    assert censored["observed_or_censor_time_seconds"] == 2.0
    assert censored["observed_or_censor_eval_count"] == 16


def test_posthoc_distinguishes_seed_31_only_target_topology() -> None:
    posthoc, _production = _posthoc(seed_gap=-0.2)
    target = posthoc["target_hitting"]["4"]
    assert target["seed_29_only_topologies"] == 0
    assert target["seed_31_only_topologies"] == 1


def test_trajectory_diagnostic_rejects_duplicate_candidate_rows() -> None:
    study = _study()
    run_id = next(iter(study.history_rows))
    duplicate = copy.deepcopy(study.history_rows[run_id][0])
    study.history_rows[run_id].insert(1, duplicate)

    with pytest.raises(StudyValidationError, match="candidate indexes"):
        analyze_submission_like_trajectories(study)


def test_trajectory_diagnostic_requires_common_evaluation_support() -> None:
    study = _study()
    run_id = next(iter(study.history_rows))
    study.history_rows[run_id][0]["eval_count_after_call"] = 7
    study.history_rows[run_id][1]["eval_count_after_call"] = 15

    with pytest.raises(StudyValidationError, match="no common evaluation count"):
        analyze_submission_like_trajectories(study)


@pytest.mark.parametrize(
    ("record_index", "field", "value", "error"),
    [
        (0, "started_utc", "not-a-timestamp", "malformed"),
        (0, "started_utc", "2026-08-22T00:00:00", "timezone-aware"),
        (
            0,
            "completed_utc",
            "2026-08-21T23:59:59+00:00",
            "precedes",
        ),
        (
            0,
            "started_utc",
            "2026-08-21T23:59:59+00:00",
            "outside the session",
        ),
        (
            1,
            "started_utc",
            "2026-08-22T00:00:15+00:00",
            "nonmonotone|overlap",
        ),
    ],
)
def test_posthoc_rejects_invalid_session_timestamps(
    record_index: int,
    field: str,
    value: str,
    error: str,
) -> None:
    pytest.importorskip("scipy", reason="submission-like analysis dependency")
    study = _study()
    run_id = next(
        run_id
        for run_id, config in study.configs.items()
        if config["planned_run_index"] == record_index
    )
    record = next(row for row in study.records if row["run_id"] == run_id)
    record[field] = value
    production = summarize_submission_like_records(study.records, study.configs)
    reference = reference_submission_like_screen(study)
    with pytest.raises(StudyValidationError, match=error):
        analyze_submission_like_posthoc(
            study,
            production,
            reference,
            agreement=_archived_agreement(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "mismatch"),
        ("archived_summary_compared", False),
        ("fields_compared", 20),
        ("absolute_tolerance", 1e-9),
        ("relative_tolerance", 1e-9),
    ],
)
def test_posthoc_rejects_forged_agreement(field: str, value: object) -> None:
    pytest.importorskip("scipy", reason="submission-like analysis dependency")
    study = _study()
    production = summarize_submission_like_records(study.records, study.configs)
    reference = reference_submission_like_screen(study)
    agreement = _archived_agreement()
    agreement[field] = value
    with pytest.raises(StudyValidationError, match="three-way agreement"):
        analyze_submission_like_posthoc(
            study,
            production,
            reference,
            agreement=agreement,
        )


def test_posthoc_rejects_wrong_frozen_action() -> None:
    pytest.importorskip("scipy", reason="submission-like analysis dependency")
    study = _study()
    production = summarize_submission_like_records(study.records, study.configs)
    reference = reference_submission_like_screen(study)
    production["predeclared_decision"]["action"] = "wrong"
    with pytest.raises(StudyValidationError, match="exact frozen action"):
        analyze_submission_like_posthoc(
            study,
            production,
            reference,
            agreement=_archived_agreement(),
        )


def test_safe_posthoc_allowlist_excludes_private_rows_and_identifiers() -> None:
    posthoc, _production = _posthoc()
    safe = safe_submission_like_posthoc(posthoc)
    serialized = json.dumps(safe, sort_keys=True)
    for forbidden in (
        "run_id",
        "pair_id",
        "topology_sha256",
        "topology_string",
        "started_utc",
        "provider",
        "gpu",
        "source_path",
        "history_rows",
    ):
        assert forbidden not in serialized
    assert safe["no_new_action_authorized"] is True
    assert safe["leave_one_topology_out"]["omissions"] == 10
    assert "private_topology_checkpoint_rows" not in serialized
    assert safe["trajectory_alignment"]["changes_frozen_decision"] is False
    assert "recommended_next_evidence_gate" not in serialized
    for topology in SUBMISSION_LIKE_TOPOLOGIES:
        assert topology not in serialized


def test_posthoc_plots_are_the_five_private_diagnostics(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib", reason="submission-like plot dependency")
    posthoc, _production = _posthoc()
    paths = create_submission_like_plots(posthoc, tmp_path)
    assert [path.name for path in paths] == [
        "topology_seed_outcomes.png",
        "run_order_and_throughput.png",
        "target_hitting_outcomes.png",
        "leave_one_topology_out.png",
        "trajectory_alignment.png",
    ]
    assert all(path.stat().st_size > 10_000 for path in paths)
