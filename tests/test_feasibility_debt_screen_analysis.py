from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytestmark = pytest.mark.integration

from experiments.feasibility_debt_candidate_screen.analysis import (
    AnalysisError,
    evaluate_stage2,
    load_history_npz,
    project_history,
    select_stage1_finalist,
)
from experiments.feasibility_debt_candidate_screen.contract import (
    ARM_ORDER,
    STUDY_ID,
    arm_spec,
    run_id,
    stage1_order,
    stage2_order,
)
from experiments.feasibility_debt_candidate_screen.evidence import compare_replays
from experiments.feasibility_debt_candidate_screen.detached_analysis import (
    DetachedError,
    detached_stage1,
    detached_stage2,
)
from experiments.feasibility_debt_candidate_screen.reference_analysis import (
    ReferenceError,
    reference_stage1,
    reference_stage2,
)


def _history(loss: float | None, *, feasible: bool = True) -> list[dict[str, object]]:
    return [
        {
            "call_index": 0,
            "candidate_index": 0,
            "eval_count_after_call": 1,
            "time_seconds": 600.0,
            "loss": loss,
            "sensitivity_loss": 0.0,
            "penalty": 0.0,
            "is_feasible": feasible,
        }
    ]


def _document(config: dict[str, object], loss: float) -> dict[str, object]:
    projection = project_history(_history(loss))
    metrics = projection.as_dict()
    metrics.pop("rows")
    metrics.pop("logged_calls")
    return {
        "run_id": config["run_id"],
        "config": config,
        "history_rows": _history(loss),
        "metrics": metrics,
        "objective_accounting": {
            "eval_count": projection.evaluation_count,
            "log_call_count": projection.logged_calls,
        },
        "runtime": {
            "returncode": 0,
            "timed_out": False,
            "wall_seconds": 610.0,
            "stdout_bytes": 200,
            "stderr_bytes": 0,
            "root_pid": 10,
            "parent_pid": 9,
            "process_group_id": 10,
            "start_ticks": 123,
            "executable_sha256": "a" * 64,
            "command_line_sha256": "b" * 64,
            "timeout_tree_killed": False,
            "zero_descendants_after_exit": True,
        },
    }


def _config(
    *, stage: int, member: int, arm: str, position: int, within: int
) -> dict[str, object]:
    topology = f"topology-{member}"
    package = "9" * 64
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "run_id": run_id(stage, member, within, arm),
        "stage": stage,
        "member_index": member,
        "execution_position": position,
        "arm_id": arm,
        "optimizer_seed": 20260901 if stage == 1 else 20260902,
        "topology": topology,
        "topology_sha256": __import__("hashlib").sha256(
            topology.encode("utf-8")
        ).hexdigest(),
        "panel_sha256": "1" * 64,
        "panel_commitment_sha256": "2" * 64,
        "split_receipt_sha256": "3" * 64,
        "selection_receipt_sha256": None if stage == 1 else "4" * 64,
        "source_lock_sha256": "5" * 64,
        "runtime_lock_sha256": "6" * 64,
        "revision": "7" * 40,
        "arm_profile": arm_spec(arm).lock_row(package),
        "max_time_seconds": 600.0,
        "max_evals": None,
        "population_size": 8,
        "n_frequencies": 50,
        "allow_cpu": False,
    }


def _write_history(path: Path, loss: float, *, time_seconds: float = 600.0) -> None:
    np.savez_compressed(
        path,
        call_index=np.asarray([0], dtype=np.int32),
        candidate_index=np.asarray([0], dtype=np.int16),
        eval_count_after_call=np.asarray([1], dtype=np.int64),
        time_seconds=np.asarray([time_seconds], dtype=np.float64),
        loss=np.asarray([loss], dtype=np.float64),
        sensitivity_loss=np.asarray([0.0], dtype=np.float64),
        penalty=np.asarray([0.0], dtype=np.float64),
        is_feasible=np.asarray([True], dtype=np.bool_),
        initial_params_unbounded=np.zeros((8, 2), dtype=np.float64),
        raw_params_unbounded=np.zeros((8, 2), dtype=np.float64),
    )


def _stage1_fixture(tmp_path: Path):
    indices = [0, 2, 4, 6]
    differences = {
        "A_round1_control": [0.0, 0.0, 0.0, 0.0],
        "B_round1_warmup": [-0.20, -0.20, -0.20, 0.10],
        "C_v3_random": [-0.30, -0.30, -0.30, 0.20],
        "D_v3_coverage": [-0.10, 0.10, 0.10, 0.10],
    }
    documents = []
    reference = []
    within: dict[int, int] = {}
    for position, (member, arm) in enumerate(stage1_order(indices)):
        local = within.get(member, 0)
        within[member] = local + 1
        config = _config(
            stage=1, member=member, arm=arm, position=position, within=local
        )
        member_offset = indices.index(member)
        loss = 1.0 + differences[arm][member_offset]
        documents.append(_document(config, loss))
        history_path = tmp_path / f"{config['run_id']}.npz"
        _write_history(history_path, loss)
        reference.append({"config": config, "history_bytes": history_path.read_bytes()})
    return indices, documents, reference


def test_history_projection_enforces_types_chronology_and_row_rule() -> None:
    rows = _history(None)
    rows.append(
        {
            "call_index": 1,
            "candidate_index": 0,
            "eval_count_after_call": 2,
            "time_seconds": 601.0,
            "loss": -0.5,
            "sensitivity_loss": None,
            "penalty": None,
            "is_feasible": True,
        }
    )
    with pytest.raises(AnalysisError, match="budget"):
        project_history(rows)
    rows[1]["time_seconds"] = 600.0
    projection = project_history(rows)
    assert projection.has_feasible is True
    assert projection.has_finite_feasible is True
    assert projection.best_feasible_loss == -0.5
    assert projection.evaluation_count == 2
    assert projection.evaluation_rate == pytest.approx(2 / 600)

    bad = [dict(rows[0])]
    bad[0]["is_feasible"] = 1
    with pytest.raises(AnalysisError, match="strict Boolean"):
        project_history(bad)
    bad = [dict(rows[0])]
    bad[0]["time_seconds"] = math.inf
    with pytest.raises(AnalysisError, match="finite"):
        project_history(bad)
    bad = [dict(rows[0]), dict(rows[1])]
    bad[1]["call_index"] = 2
    with pytest.raises(AnalysisError, match="missing or reordered"):
        project_history(bad)


def test_history_loader_requires_and_authenticates_both_population_arrays(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.npz"
    _write_history(valid, 1.0)
    assert len(load_history_npz(valid)) == 1

    missing = tmp_path / "missing-raw.npz"
    np.savez_compressed(
        missing,
        call_index=np.asarray([0], dtype=np.int32),
        candidate_index=np.asarray([0], dtype=np.int16),
        eval_count_after_call=np.asarray([1], dtype=np.int64),
        time_seconds=np.asarray([600.0], dtype=np.float64),
        loss=np.asarray([1.0], dtype=np.float64),
        sensitivity_loss=np.asarray([0.0], dtype=np.float64),
        penalty=np.asarray([0.0], dtype=np.float64),
        is_feasible=np.asarray([True], dtype=np.bool_),
        initial_params_unbounded=np.zeros((8, 2), dtype=np.float64),
    )
    with pytest.raises(AnalysisError, match="member schema"):
        load_history_npz(missing)

    nonfinite = tmp_path / "nonfinite-raw.npz"
    _write_history(nonfinite, 1.0)
    with np.load(nonfinite, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["raw_params_unbounded"] = np.full((8, 2), np.nan, dtype=np.float64)
    np.savez_compressed(nonfinite, **arrays)
    with pytest.raises(AnalysisError, match="population artifact"):
        load_history_npz(nonfinite)


def test_stage1_selects_only_eligible_best_and_matches_independent_replay(
    tmp_path: Path,
) -> None:
    indices, documents, packets = _stage1_fixture(tmp_path)
    production = select_stage1_finalist(documents, indices)
    reference = reference_stage1(packets, indices)
    detached = detached_stage1(packets, indices)
    token = compare_replays(
        production,
        reference,
        stage=1,
        archive_sha256="a" * 64,
        ordered_run_ids=[document["run_id"] for document in documents],
        panel_sha256="1" * 64,
        split_receipt_sha256="3" * 64,
        source_lock_sha256="5" * 64,
        runtime_lock_sha256="6" * 64,
    )
    assert token.stage == 1
    assert production["eligible_ids"] == ["B_round1_warmup", "C_v3_random"]
    assert production["finalist"] == "C_v3_random"
    assert production["action"] == "advance_selected_finalist_to_stage2"
    assert production["stage2_outcome_opened"] is False
    assert detached == production


def test_all_stage1_replays_reject_boolean_integer_and_over_budget_rows(
    tmp_path: Path,
) -> None:
    indices, documents, packets = _stage1_fixture(tmp_path)
    bad_documents = copy.deepcopy(documents)
    bad_documents[0]["config"]["stage"] = True
    with pytest.raises(AnalysisError, match="integer type"):
        select_stage1_finalist(bad_documents, indices)

    bad_packets = copy.deepcopy(packets)
    bad_packets[0]["config"]["stage"] = True
    with pytest.raises(ReferenceError, match="integer type"):
        reference_stage1(bad_packets, indices)
    with pytest.raises(DetachedError, match="integer type"):
        detached_stage1(bad_packets, indices)

    over_budget_path = tmp_path / "over-budget.npz"
    _write_history(over_budget_path, 1.0, time_seconds=600.000001)
    over_budget_packets = copy.deepcopy(packets)
    over_budget_packets[0]["history_bytes"] = over_budget_path.read_bytes()
    with pytest.raises(ReferenceError, match="budget"):
        reference_stage1(over_budget_packets, indices)
    with pytest.raises(DetachedError, match="terminal history"):
        detached_stage1(over_budget_packets, indices)


def test_stage1_failed_requires_all_sixteen_valid_runs(tmp_path: Path) -> None:
    indices, documents, _packets = _stage1_fixture(tmp_path)
    for document in documents:
        arm = document["config"]["arm_id"]
        if arm != "A_round1_control":
            document["history_rows"][0]["loss"] = 2.0
            projection = project_history(document["history_rows"])
            metrics = projection.as_dict()
            metrics.pop("rows")
            metrics.pop("logged_calls")
            document["metrics"] = metrics
    result = select_stage1_finalist(documents, indices)
    assert result["finalist"] is None
    assert result["action"] == "retain_round1_control_stage1_failed"
    with pytest.raises(AnalysisError, match="run count"):
        select_stage1_finalist(documents[:-1], indices)


def test_stage1_tie_within_tolerance_uses_b_c_d_priority(tmp_path: Path) -> None:
    indices, documents, _packets = _stage1_fixture(tmp_path)
    for document in documents:
        arm = document["config"]["arm_id"]
        member = document["config"]["member_index"]
        baseline = 1.0
        if arm == "A_round1_control":
            loss = baseline
        elif arm in {"B_round1_warmup", "C_v3_random"}:
            values = [-0.2, -0.2, -0.2, 0.1]
            loss = baseline + values[indices.index(member)]
            if arm == "C_v3_random":
                loss += 5e-13
        else:
            loss = 2.0
        replacement = _document(document["config"], loss)
        document.clear()
        document.update(replacement)
    result = select_stage1_finalist(documents, indices)
    assert result["finalist"] == "B_round1_warmup"


def test_stage2_requires_four_wins_and_mean_at_most_minus_point_zero_five(
    tmp_path: Path,
) -> None:
    indices = [1, 3, 5, 7]
    finalist = "C_v3_random"

    def build(differences: list[float]):
        documents = []
        packets = []
        within: dict[int, int] = {}
        for position, (member, arm) in enumerate(stage2_order(indices, finalist)):
            local = within.get(member, 0)
            within[member] = local + 1
            config = _config(
                stage=2, member=member, arm=arm, position=position, within=local
            )
            delta = differences[indices.index(member)]
            loss = 1.0 if arm == "A_round1_control" else 1.0 + delta
            documents.append(_document(config, loss))
            path = tmp_path / f"{config['run_id']}-{len(documents)}.npz"
            _write_history(path, loss)
            packets.append({"config": config, "history_bytes": path.read_bytes()})
        return documents, packets

    passing_docs, passing_packets = build([-0.05, -0.05, -0.05, -0.05])
    production = evaluate_stage2(passing_docs, indices, finalist, "4" * 64)
    reference = reference_stage2(
        passing_packets, indices, finalist, "4" * 64
    )
    compare_replays(
        production,
        reference,
        stage=2,
        archive_sha256="a" * 64,
        ordered_run_ids=[document["run_id"] for document in passing_docs],
        panel_sha256="1" * 64,
        split_receipt_sha256="3" * 64,
        source_lock_sha256="5" * 64,
        runtime_lock_sha256="6" * 64,
    )
    assert production["wins"] == 4
    assert production["mean_difference"] == pytest.approx(-0.05)
    assert production["passed"] is True
    assert production["action"] == (
        "review_selected_bundle_for_round2_candidate_integration"
    )

    failing_docs, _ = build([-0.1, -0.1, -0.1, 0.0])
    failed = evaluate_stage2(failing_docs, indices, finalist, "4" * 64)
    assert failed["wins"] == 3
    assert failed["passed"] is False
    assert failed["action"] == "retain_round1_control"

    with pytest.raises(AnalysisError, match="selection"):
        evaluate_stage2(passing_docs, indices, finalist, "9" * 64)
    with pytest.raises(ReferenceError, match="selection"):
        reference_stage2(passing_packets, indices, finalist, "9" * 64)
    with pytest.raises(DetachedError, match="selection"):
        detached_stage2(passing_packets, indices, finalist, "9" * 64)
