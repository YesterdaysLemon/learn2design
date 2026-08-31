from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration
np = pytest.importorskip("numpy")

from experiments.local_lab import constraint_aware_progress_toy as fixture
from experiments.local_lab import constraint_aware_progress_toy_worker as worker
from tools import run_local_lab as controller


ROOT = Path(__file__).parents[1]
STUDY_ID = "constraint-aware-progress-toy-v1"
WORKER_RECEIPT = {
    "stderr_bytes": 0,
    "stderr_sha256": hashlib.sha256(b"").hexdigest(),
    "stdout_bytes": 100,
}


def test_fixture_import_is_inert_and_worker_network_gate_precedes_runtime() -> None:
    fixture_source = (ROOT / "experiments/local_lab/constraint_aware_progress_toy.py").read_text(
        encoding="utf-8"
    )
    worker_source = (
        ROOT / "experiments/local_lab/constraint_aware_progress_toy_worker.py"
    ).read_text(encoding="utf-8")
    assert "submission" not in fixture_source
    assert "submission" not in worker_source
    tree = ast.parse(worker_source)
    module_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert module_calls == []
    main_source = ast.get_source_segment(
        worker_source,
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        ),
    )
    assert main_source is not None
    assert main_source.index("_await_child_job_gate(args.mode)") < main_source.index(
        "_disable_network()"
    )
    assert main_source.index("_disable_network()") < main_source.index("_load_runtime()")
    child_source = inspect.getsource(worker._run_child)
    assert child_source.index("job.assign(process)") < child_source.index(
        "writer.start()"
    )
    assert child_source.index("started = time.monotonic()") < child_source.index(
        "subprocess.Popen("
    )


def test_family_reference_and_transcript_commitments_replay_without_optimizer() -> None:
    worker._load_runtime()
    worlds, implementation_root, oracle_root = worker._world_replay()
    assert len(worlds) == 48
    assert implementation_root == oracle_root
    assert implementation_root == controller._constraint_root(
        "WorldRecord", controller._constraint_world_rows()
    )
    primary, oracle, observed_root = worker._transcript_replay()
    assert len(primary) == len(oracle) == 4
    assert observed_root == fixture.TRANSCRIPT_ROOT_SHA256
    assert all(not item.suffix.flags.writeable for item in primary)
    assert all(not item.fresh.flags.writeable for item in primary)
    assert all(not item.perturb.flags.writeable for item in primary)


def test_optimizer_packet_and_oracle_keep_the_physical_information_boundary() -> None:
    assert tuple(fixture.OptimizerPacket.__dataclass_fields__) == (
        "u",
        "learning_rates",
        "m",
        "v",
        "ages",
        "stalls",
        "progress",
        "losses",
        "gradients",
        "decisions",
        "incumbent_present",
        "incumbent_center",
        "restart_round",
        "budget_fraction",
    )
    oracle_source = inspect.getsource(worker._ReplayOracle)
    for forbidden in (
        "fx.build_world_records",
        "fx.evaluate",
        "fx.progress_decision",
        "fx.OptimizerAdapter",
        "fx._update_incumbent",
        "fx._trajectory_roots",
    ):
        assert forbidden not in oracle_source
    implementation_rows, oracle_rows = worker._intervention_whitelist_rows()
    assert implementation_rows == oracle_rows


def test_restart_provider_releases_only_committed_hashes_before_seal() -> None:
    transcript = fixture.build_transcript(fixture.SEEDS[0])
    provider = fixture.RestartDrawProvider(transcript)
    prepared = fixture.PreparedStep(
        u=np.zeros((8, 3), dtype=np.float64),
        m=np.zeros((8, 3), dtype=np.float64),
        v=np.zeros((8, 3), dtype=np.float64),
        ages=np.ones(8, dtype=np.int64),
        stalls=np.zeros(8, dtype=np.int64),
        progress=tuple(
            fixture.ProgressState("lex", False, False, 0.0, 0.0)
            for _ in range(8)
        ),
        restart_mask=(False,) * 8,
    )
    authorization = provider.authorize(prepared.restart_mask, 0)
    assert tuple(authorization.__dataclass_fields__) == (
        "batch",
        "restart_round",
        "restart_mask",
        "fresh_draw_sha256",
        "perturb_draw_sha256",
    )
    assert not any("array" in name or "values" in name for name in authorization.__dataclass_fields__)
    receipt = fixture.BatchReceipt(
        family="canonical",
        world=0,
        seed=fixture.SEEDS[0],
        order="forward",
        arm="constraint_lexicographic_progress",
        batch=0,
        incumbent_present=True,
        incumbent_sensitivity=0.0,
        incumbent_source_batch=0,
        incumbent_source_member=0,
        incumbent_center_sha256=fixture.incumbent_center_hash(True, (0.0, 0.0, 0.0)),
        restart_round_before=0,
        restart_round_after=0,
        restart_mask=(False,) * 8,
        fresh_draw_sha256=authorization.fresh_draw_sha256,
        perturb_draw_sha256=authorization.perturb_draw_sha256,
    )
    sealed = provider.seal(authorization, receipt)
    outcome = provider.apply(
        sealed,
        prepared,
        receipt=receipt,
        incumbent_present=True,
        incumbent_center=(0.0, 0.0, 0.0),
        budget_fraction=1.0 / 64.0,
    )
    assert outcome.restart_mask == (False,) * 8
    assert provider._next_batch == 1


def test_runtime_sentinel_capability_and_malformed_consumers_are_live() -> None:
    worker._load_runtime()
    # The failed V1 worker and runtime probe are now policy-quarantined.  Unit
    # consumers remain testable, but no process path may launch that worker.
    with pytest.raises(controller.QuarantinedStudyError, match="cannot be invoked"):
        controller._constraint_progress_runtime_identity()
    with pytest.raises(fixture.ContractError, match="runtime-identity"):
        worker._runtime_identity()
    outcomes, capabilities, implementation_root, oracle_root = (
        worker._sentinel_and_capabilities()
    )
    assert outcomes == ((True, True), (False, False), (True, False))
    assert len(capabilities) == 22
    assert implementation_root == oracle_root
    assert implementation_root == controller._constraint_intervention_root()
    attacks, implementation_attack_root, oracle_attack_root = worker._malformed_attacks()
    valid_observations, valid_transition = worker._minimal_envelope()
    assert valid_observations[0]["canonical_is_feasible"] is True
    assert valid_observations[0]["loss"] == pytest.approx(0.56585)
    assert valid_observations[0]["gradient"] == [0.1, 0.5, 0.25]
    progress_after = valid_transition["progress_after"]
    assert progress_after["mode"] == "lex"
    assert progress_after["observed"] is True
    assert progress_after["feasible"] is True
    assert progress_after["first"] == pytest.approx(0.56585)
    assert progress_after["second"] == 0.0
    assert valid_transition["state_before_sha256"] != "0" * 64
    assert valid_transition["state_after_sha256"] != "1" * 64
    assert [item["attack_id"] for item in attacks] == [
        item[0] for item in fixture.MALFORMED_ATTACKS
    ]
    assert all(item["consumer_reached"] for item in attacks)
    assert all(item["state_mutations"] == 0 for item in attacks)
    assert implementation_attack_root == oracle_attack_root
    assert implementation_attack_root == controller._constraint_attack_root()


def _aggregate_value(family: str, arm: str) -> float:
    if family == "impossible":
        return 1.0
    if family == "canonical":
        return {
            "protected_raw_progress": 0.8,
            "constraint_lexicographic_progress": 0.6,
            "shuffled_progress_control": 0.79,
            "ablated_progress_control": 0.8,
            "no_restart_comparator": 0.9,
        }[arm]
    return {
        "protected_raw_progress": 0.5,
        "constraint_lexicographic_progress": 0.51,
        "shuffled_progress_control": 0.5,
        "ablated_progress_control": 0.5,
        "no_restart_comparator": 0.5,
    }[arm]


def _valid_result(entry: dict[str, object], revision: str, contract: str):
    fixture_identity = entry["fixture_identity"]
    assert isinstance(fixture_identity, dict)
    families = fixture_identity["families"]
    arms = fixture_identity["arms"]
    world_aggregates = []
    for family in families:
        for world in range(16):
            for arm in arms:
                value = float(_aggregate_value(family, arm))
                world_aggregates.append(
                    {
                        "family": family,
                        "world": world,
                        "arm": arm,
                        "seed_gaps": [value, value, value, value],
                        "mean_gap": value,
                    }
                )

    root = "a" * 64
    family_root = controller._constraint_root(
        "WorldRecord", controller._constraint_world_rows()
    )
    intervention_root = controller._constraint_intervention_root()
    source_root = controller._constraint_source_root(world_aggregates)
    attack_root = controller._constraint_attack_root()
    cases_by_id = {
        "family_replay": {
            "world_records": 48,
            "constrained_references": 32,
            "reference_exclusions": 16,
            "development_worlds_per_family": 8,
            "heldout_worlds_per_family": 8,
            "formula_mismatches": 0,
            "reference_mismatches": 0,
            "nonpositive_denominators": 0,
            "duplicate_world_keys": 0,
            "implementation_root_sha256": family_root,
            "oracle_root_sha256": family_root,
            "roots_equal": True,
        },
        "transcript_commitment": {
            "transcripts": 4,
            "values_per_transcript": 3093,
            "transcript_values": 12372,
            "trajectories": 1920,
            "evaluations": 983040,
            "unequal_arm_counts": 0,
            "order_twin_mismatches": 0,
            "committed_root_sha256": entry["transcript_commitment"]["root_sha256"],
            "observed_root_sha256": entry["transcript_commitment"]["root_sha256"],
            "roots_equal": True,
        },
        "typed_aux_and_intervention": {
            "observations": 983040,
            "schema_valid_observations": 983040,
            "join_failures": 0,
            "capability_attacks": 22,
            "capability_rejected": 22,
            "capability_state_mutations": 0,
            "canonical_decisions": [True, True],
            "donor_decisions": [False, False],
            "ablated_decisions": [True, False],
            "implementation_schema_root_sha256": root,
            "oracle_schema_root_sha256": root,
            "implementation_intervention_root_sha256": intervention_root,
            "oracle_intervention_root_sha256": intervention_root,
            "roots_equal": True,
        },
        "chronology_replay": {
            "batches": 122880,
            "batch_receipts": 122880,
            "transitions": 983040,
            "replay_mismatches": 0,
            "order_mismatches": 0,
            "reset_mismatches": 0,
            "incumbent_tie_mismatches": 0,
            "incumbent_state_mismatches": 0,
            "restart_events": 1,
            "implementation_state_root_sha256": root,
            "oracle_state_root_sha256": root,
            "roots_equal": True,
        },
        "development_and_source_isolation": {
            "development_aggregates": 120,
            "development_receipts": 3,
            "heldout_receipts": 3,
            "forbidden_reads": 0,
            "heldout_source_in_development": 0,
            "development_outputs_in_heldout": 0,
            "implementation_source_root_sha256": source_root,
            "oracle_source_root_sha256": source_root,
            "roots_equal": True,
        },
        "impossible_control": {
            "trajectories": 640,
            "observations": 327680,
            "feasible_observations": 0,
            "nonunit_gaps": 0,
            "references_used": 0,
            "false_feasible_joins": 0,
        },
        "process_and_sanitizer": {
            "launches": 2,
            "projections_equal": True,
            "maximum_stdout_bytes": 100,
            "stderr_bytes": 0,
            "surviving_children": 0,
            "attacks": 12,
            "attacks_rejected": 12,
            "attack_state_mutations": 0,
            "implementation_attack_root_sha256": attack_root,
            "oracle_attack_root_sha256": attack_root,
            "roots_equal": True,
        },
    }
    treatment, baseline, differences, wins, ties, losses = controller._constraint_comparison(
        world_aggregates,
        "canonical",
        "constraint_lexicographic_progress",
        "protected_raw_progress",
    )
    improvement = baseline - treatment
    harm = max(differences)
    cases_by_id["heldout_primary"] = {
        "treatment_mean_gap": treatment,
        "baseline_mean_gap": baseline,
        "mean_improvement": improvement,
        "heldout_wins": wins,
        "heldout_ties": ties,
        "heldout_losses": losses,
        "maximum_signed_world_harm": harm,
        "mean_gate": improvement >= 0.05,
        "win_gate": wins >= 6,
        "harm_gate": harm <= 0.15,
    }
    no_restart, treatment_again, _differences, *_ = controller._constraint_comparison(
        world_aggregates,
        "canonical",
        "no_restart_comparator",
        "constraint_lexicographic_progress",
    )
    assert treatment_again == treatment
    cases_by_id["restart_comparators"] = {
        "treatment_mean_gap": treatment,
        "no_restart_mean_gap": no_restart,
        "mean_improvement": no_restart - treatment,
        "minimum_arm_evaluations": 512,
        "maximum_arm_evaluations": 512,
        "evaluation_parity": True,
        "transcript_parity": True,
        "comparator_gate": no_restart - treatment >= 0.05,
    }
    for case_id, arm in (
        ("shuffled_signal_control", "shuffled_progress_control"),
        ("ablated_signal_control", "ablated_progress_control"),
    ):
        control, control_baseline, control_diff, control_wins, *_ = (
            controller._constraint_comparison(
                world_aggregates, "canonical", arm, "protected_raw_progress"
            )
        )
        control_improvement = control_baseline - control
        control_harm = max(control_diff)
        mean_gate = control_improvement >= 0.05
        win_gate = control_wins >= 6
        harm_gate = control_harm <= 0.15
        cases_by_id[case_id] = {
            "control_mean_gap": control,
            "baseline_mean_gap": control_baseline,
            "mean_improvement": control_improvement,
            "heldout_wins": control_wins,
            "maximum_signed_world_harm": control_harm,
            "substituted_mean_gate": mean_gate,
            "substituted_win_gate": win_gate,
            "substituted_harm_gate": harm_gate,
            "positive_gate_recovered": mean_gate and win_gate and harm_gate,
        }
    aligned, aligned_baseline, aligned_diff, *_ = controller._constraint_comparison(
        world_aggregates,
        "aligned",
        "constraint_lexicographic_progress",
        "protected_raw_progress",
    )
    aligned_abs = abs(aligned - aligned_baseline)
    aligned_harm = max(aligned_diff)
    cases_by_id["aligned_control"] = {
        "treatment_mean_gap": aligned,
        "baseline_mean_gap": aligned_baseline,
        "absolute_mean_difference": aligned_abs,
        "maximum_signed_world_harm": aligned_harm,
        "trajectories": 640,
        "mean_gate": aligned_abs <= 0.03,
        "harm_gate": aligned_harm <= 0.10,
    }
    case_ids = entry["case_ids"]
    cases = [
        {"case_id": case_id, "passed": True, "metrics": cases_by_id[case_id]}
        for case_id in case_ids
    ]
    return {
        "study_id": STUDY_ID,
        "plan_revision": entry["plan_revision"],
        "study_revision": revision,
        "contract_sha256": contract,
        "transcript_root_sha256": entry["transcript_commitment"]["root_sha256"],
        "status": "passed",
        "action": entry["success_action"],
        "world_aggregates": world_aggregates,
        "cases": cases,
    }


def test_controller_recomputes_the_closed_result_and_rejects_tampering() -> None:
    registry = json.loads(
        (ROOT / "experiments/local_lab/studies.json").read_text(encoding="utf-8")
    )
    entry = registry["studies"][STUDY_ID]
    revision = "b" * 40
    contract = "c" * 64
    result = _valid_result(entry, revision, contract)
    controller._validate_study_result(
        STUDY_ID,
        entry,
        result,
        study_revision=revision,
        contract_sha256=contract,
        worker_receipt=WORKER_RECEIPT,
    )

    wrong_mean = copy.deepcopy(result)
    wrong_mean["world_aggregates"][0]["mean_gap"] += 0.01
    with pytest.raises(RuntimeError, match="aggregate value"):
        controller._validate_study_result(
            STUDY_ID,
            entry,
            wrong_mean,
            study_revision=revision,
            contract_sha256=contract,
            worker_receipt=WORKER_RECEIPT,
        )

    wrong_type = copy.deepcopy(result)
    wrong_type["cases"][0]["metrics"]["roots_equal"] = 1
    with pytest.raises(RuntimeError, match="metric type"):
        controller._validate_study_result(
            STUDY_ID,
            entry,
            wrong_type,
            study_revision=revision,
            contract_sha256=contract,
            worker_receipt=WORKER_RECEIPT,
        )

    wrong_pass = copy.deepcopy(result)
    wrong_pass["cases"][0]["passed"] = False
    with pytest.raises(RuntimeError, match="case pass"):
        controller._validate_study_result(
            STUDY_ID,
            entry,
            wrong_pass,
            study_revision=revision,
            contract_sha256=contract,
            worker_receipt=WORKER_RECEIPT,
        )

    fabricated_root = copy.deepcopy(result)
    fabricated_root["cases"][0]["metrics"]["implementation_root_sha256"] = "a" * 64
    fabricated_root["cases"][0]["metrics"]["oracle_root_sha256"] = "a" * 64
    with pytest.raises(RuntimeError, match="derived metric disagrees"):
        controller._validate_study_result(
            STUDY_ID,
            entry,
            fabricated_root,
            study_revision=revision,
            contract_sha256=contract,
            worker_receipt=WORKER_RECEIPT,
        )

    negative_inner_output = copy.deepcopy(result)
    negative_inner_output["cases"][-1]["metrics"]["maximum_stdout_bytes"] = -1
    with pytest.raises(RuntimeError, match="inner projection"):
        controller._validate_study_result(
            STUDY_ID,
            entry,
            negative_inner_output,
            study_revision=revision,
            contract_sha256=contract,
            worker_receipt=WORKER_RECEIPT,
        )

    balanced_but_nonunit_impossible = copy.deepcopy(result)
    impossible_row = next(
        row
        for row in balanced_but_nonunit_impossible["world_aggregates"]
        if row["family"] == "impossible"
    )
    impossible_row["seed_gaps"] = [0.0, 0.0, 2.0, 2.0]
    impossible_row["mean_gap"] = 1.0
    changed_source_root = controller._constraint_source_root(
        balanced_but_nonunit_impossible["world_aggregates"]
    )
    source_metrics = balanced_but_nonunit_impossible["cases"][4]["metrics"]
    source_metrics["implementation_source_root_sha256"] = changed_source_root
    source_metrics["oracle_source_root_sha256"] = changed_source_root
    with pytest.raises(RuntimeError, match="impossible aggregate"):
        controller._validate_study_result(
            STUDY_ID,
            entry,
            balanced_but_nonunit_impossible,
            study_revision=revision,
            contract_sha256=contract,
            worker_receipt=WORKER_RECEIPT,
        )


def test_duplicate_json_keys_fail_closed_at_worker_and_controller_boundaries() -> None:
    worker._load_runtime()
    with pytest.raises(fixture.ContractError, match="duplicate-json-key"):
        worker._loads_json('{"x":1,"x":2}')
    with pytest.raises(RuntimeError, match="duplicate JSON object key"):
        controller._loads_json('{"x":1,"x":2}')


def test_registry_preserves_historical_sources_and_controller_quarantine() -> None:
    encoded = (ROOT / "experiments/local_lab/studies.json").read_bytes().replace(
        b"\r\n", b"\n"
    )
    registry = json.loads(encoded)
    entry = registry["studies"][STUDY_ID]
    assert entry["source_paths"] == {
        "dependency_lock": "uv.lock",
        "fixture_source": "experiments/local_lab/constraint_aware_progress_toy.py",
        "lab_protocol": "docs/AUTONOMOUS_LAB.md",
        "study_plan": "research/2026-08-30-round1-feedback-and-round2-program.md",
        "worker_source": (
            "experiments/local_lab/constraint_aware_progress_toy_worker.py"
        ),
    }
    # The CI checkout is intentionally shallow, so an ancestor commit is not a
    # portable source of truth here.  Pin the retired manifest itself instead;
    # the controller refuses this study before it can inspect or approve files.
    assert entry["approved_file_sha256"] == {
        "dependency_lock": (
            "5aa38f61873af4713dd88514227eb28aceaaade949215bef65d8125ab45834d0"
        ),
        "fixture_source": (
            "691c81fa0e960c307266d5963685067a71803bc2a1e9c4dfe0561d53508e63d1"
        ),
        "lab_protocol": (
            "d062a71131533d3d26ae33c95ae155a73d2477914c48024339bf6d94fd8c8472"
        ),
        "study_plan": (
            "9a9c4536a28ee6fdea8f74387be5975943eaab551a3c959f41e4d3d49ba86c96"
        ),
        "worker_source": (
            "2f59c44fc3c98fd5e90431ed7b3d52946815b170d8b2bead6689953e463d3994"
        ),
    }
    assert STUDY_ID in controller.QUARANTINED_STUDIES
    assert entry["worker_module"] in controller.WORKER_MODULE_PATHS
    assert (
        controller.WORKER_MODULE_PATHS[entry["worker_module"]]
        == entry["source_paths"]["worker_source"]
    )
    assert len(entry["case_ids"]) == 12
    assert set(entry["case_ids"]) == set(entry["case_metric_schema"])
    assert hashlib.sha256(encoded).hexdigest() == controller.EXPECTED_STUDY_REGISTRY_SHA256


def test_controller_refuses_quarantined_study_before_private_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cycles" / "forbidden-retry" / "result.json"
    monkeypatch.setattr(controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(
        controller.sys,
        "argv",
        [
            "run_local_lab.py",
            "--study",
            STUDY_ID,
            "--output",
            str(output),
        ],
    )

    with pytest.raises(controller.QuarantinedStudyError, match="cannot be invoked"):
        controller.main()

    with pytest.raises(controller.QuarantinedStudyError, match="cannot be invoked"):
        controller._run_worker(
            STUDY_ID,
            cycle_id="forbidden-retry",
            heartbeat=lambda _pid, _elapsed: None,
            worker_module="experiments.local_lab.constraint_aware_progress_toy_worker",
        )

    assert list(tmp_path.iterdir()) == []
