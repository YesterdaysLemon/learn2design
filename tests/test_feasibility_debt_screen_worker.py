from __future__ import annotations

from pathlib import Path
import io
import sys
import types

import pytest

from experiments.feasibility_debt_candidate_screen.canonical import sha256_bytes
from experiments.feasibility_debt_candidate_screen.archive import (
    ArchiveError,
    HISTORY_SCHEMA,
    _validate_population_and_algorithm,
)
from experiments.feasibility_debt_candidate_screen.contract import (
    STUDY_ID,
    arm_spec,
)
from experiments.feasibility_debt_candidate_screen.worker import (
    WorkerError,
    _Instrumentation,
    _flatten,
    load_arm_class,
    validate_config,
    verify_warmup_source,
)


ROOT = Path(__file__).parents[1]


def _config(stage: int = 1) -> dict[str, object]:
    topology = "AAAAAAAAA-DLLLLLLLLLLL"
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "run_id": "s1-m00-p0-A_round1_control",
        "stage": stage,
        "member_index": 0,
        "execution_position": 0,
        "arm_id": "A_round1_control",
        "optimizer_seed": 20260901 if stage == 1 else 20260902,
        "topology": topology,
        "topology_sha256": sha256_bytes(topology.encode("utf-8")),
        "panel_sha256": "a" * 64,
        "panel_commitment_sha256": "f" * 64,
        "split_receipt_sha256": "b" * 64,
        "selection_receipt_sha256": None if stage == 1 else "c" * 64,
        "source_lock_sha256": "d" * 64,
        "runtime_lock_sha256": "e" * 64,
        "revision": "1" * 40,
        "arm_profile": arm_spec("A_round1_control").lock_row("2" * 64),
        "max_time_seconds": 600.0,
        "max_evals": None,
        "population_size": 8,
        "n_frequencies": 50,
        "allow_cpu": False,
    }


def test_worker_config_is_exact_and_stage2_requires_selection() -> None:
    assert validate_config(_config())["stage"] == 1
    stage2 = _config(2)
    assert validate_config(stage2)["selection_receipt_sha256"] == "c" * 64
    stage2["selection_receipt_sha256"] = None
    with pytest.raises(WorkerError, match="lacks a selection"):
        validate_config(stage2)
    extra = _config()
    extra["unexpected"] = True
    with pytest.raises(WorkerError, match="schema"):
        validate_config(extra)
    for field in (
        "schema_version",
        "stage",
        "member_index",
        "execution_position",
        "optimizer_seed",
        "population_size",
        "n_frequencies",
    ):
        boolean_integer = _config()
        boolean_integer[field] = True
        with pytest.raises(WorkerError):
            validate_config(boolean_integer)
    boolean_time = _config()
    boolean_time["max_time_seconds"] = True
    with pytest.raises(WorkerError, match="time budget"):
        validate_config(boolean_time)


def test_round1_control_loads_from_exact_archive_in_isolated_namespace() -> None:
    cls = load_arm_class(
        "A_round1_control",
        repository_root=ROOT,
        round1_archive=ROOT / "artifacts/generated/submission.zip",
        round1_manifest=ROOT / "artifacts/generated/submission.manifest.json",
    )
    assert cls.__module__ == "l2d_round1_control_submission"
    assert cls.__name__ == "BatchedRestartAdam"
    assert cls.algorithm_str == "batched_restart_adam"


def test_current_arms_load_from_verified_bytes_and_reject_poisoned_namespace() -> None:
    for arm in ("B_round1_warmup", "C_v3_random", "D_v3_coverage"):
        cls = load_arm_class(
            arm,
            repository_root=ROOT,
            round1_archive=ROOT / "artifacts/generated/submission.zip",
            round1_manifest=ROOT / "artifacts/generated/submission.manifest.json",
        )
        assert cls.__name__ == arm_spec(arm).class_name
        assert cls.algorithm_str == arm_spec(arm).algorithm_str
    poisoned = "l2d_screen_B_round1_warmup"
    sys.modules[poisoned] = types.ModuleType(poisoned)
    try:
        with pytest.raises(WorkerError, match="already occupied"):
            load_arm_class(
                "B_round1_warmup",
                repository_root=ROOT,
                round1_archive=ROOT / "artifacts/generated/submission.zip",
                round1_manifest=ROOT
                / "artifacts/generated/submission.manifest.json",
            )
    finally:
        sys.modules.pop(poisoned, None)


class _FakeObjective:
    def __init__(self) -> None:
        self.eval_count = 0
        self.log_call_count = 0
        self.time_elapsed = 0.0
        self.time_steps = []
        self.loss_history = []
        self.sensitivity_loss_history = []
        self.penalty_history = []
        self.is_feasible_history = []
        self.budget_exceeded = False
        self.time_exceeded = False
        self.evals_exceeded = False
        self._seed = 20260901
        self._rng_key = [1, 2]
        self._max_time = 600.0
        self._max_evals = None
        self._time_offset = 0.0
        self._start_time = None
        self.started = False

    @property
    def time_left(self):
        return 600.0

    @property
    def evals_left(self):
        return None

    def random_params_unbounded(self, count: int):
        return [[float(index), 0.0] for index in range(count)]

    def warmup_value_and_grad_aux(self):
        return None

    def warmup_vmap_value_and_grad_aux(self, count: int):
        return None

    def start_logging(self):
        self.started = True


def test_population_and_warmup_instrumentation_is_fail_closed() -> None:
    objective = _FakeObjective()
    proof = {
        "source_sha256": "a" * 64,
        "calls": ["objective.warmup_vmap_value_and_grad_aux"],
        "random_sampling_calls": 0,
    }
    capture = _Instrumentation(
        objective, expect_warmup=True, warmup_source_proof=proof
    )
    raw = objective.random_params_unbounded(8)
    capture.capture_raw(raw)
    capture.capture_initial(raw)
    objective.warmup_vmap_value_and_grad_aux(8)
    objective.start_logging()
    receipt = capture.receipt()
    assert receipt["warmup_enabled"] is True
    assert receipt["before_warmup"] == receipt["after_warmup"]
    assert receipt["before_warmup"]["random_draw_calls"] == 1
    assert receipt["before_warmup"]["parameter_sha256"] == receipt[
        "initial_population_sha256"
    ]

    changed = _FakeObjective()
    capture = _Instrumentation(
        changed, expect_warmup=True, warmup_source_proof=proof
    )
    raw = changed.random_params_unbounded(8)
    capture.capture_initial(raw)
    changed.warmup_vmap_value_and_grad_aux(8)
    changed.eval_count = 1
    changed.start_logging()
    with pytest.raises(WorkerError, match="warmup mutated"):
        capture.receipt()


def test_archive_replays_raw_population_bytes_instead_of_trusting_receipt() -> None:
    np = pytest.importorskip("numpy")
    objective = _FakeObjective()
    proof = {
        "source_sha256": "a" * 64,
        "calls": ["no_warmup"],
        "random_sampling_calls": 0,
    }
    capture = _Instrumentation(
        objective, expect_warmup=False, warmup_source_proof=proof
    )
    raw = np.asarray(objective.random_params_unbounded(8), dtype=np.float64)
    capture.capture_raw(raw)
    capture.capture_initial(raw)
    objective.start_logging()
    population = capture.receipt()
    config = _config()
    spec = arm_spec("A_round1_control")

    def history_bytes(raw_bytes: object) -> bytes:
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            call_index=np.asarray([0], dtype=np.int32),
            candidate_index=np.asarray([0], dtype=np.int16),
            eval_count_after_call=np.asarray([1], dtype=np.int64),
            time_seconds=np.asarray([600.0], dtype=np.float64),
            loss=np.asarray([1.0], dtype=np.float64),
            sensitivity_loss=np.asarray([0.0], dtype=np.float64),
            penalty=np.asarray([0.0], dtype=np.float64),
            is_feasible=np.asarray([True], dtype=np.bool_),
            initial_params_unbounded=raw,
            raw_params_unbounded=raw_bytes,
        )
        return buffer.getvalue()

    valid_history = history_bytes(raw)
    record = {
        "run_id": config["run_id"],
        "config": config,
        "algorithm": {
            "logical_module_id": spec.logical_module_id,
            "python_module_name": spec.python_module_name,
            "class_name": spec.class_name,
            "algorithm_str": spec.algorithm_str,
            "source_sha256": spec.source_sha256,
            "kwargs": {**spec.fixed_kwargs(), "random_seed": 20260901},
        },
        "topology_sha256": config["topology_sha256"],
        "initial_population": population,
        "history": {
            "sha256": sha256_bytes(valid_history),
            "rows": 1,
            "schema": HISTORY_SCHEMA,
        },
        "metrics": {},
        "objective_accounting": {},
        "worker_measurement": {},
    }
    _validate_population_and_algorithm(record, valid_history)
    tampered = history_bytes(raw + 1.0)
    with pytest.raises(ArchiveError, match="differs from NPZ"):
        _validate_population_and_algorithm(record, tampered)


def test_warmup_ast_proof_and_history_flattening_fail_closed() -> None:
    cls = load_arm_class(
        "B_round1_warmup",
        repository_root=ROOT,
        round1_archive=ROOT / "artifacts/generated/submission.zip",
        round1_manifest=ROOT / "artifacts/generated/submission.manifest.json",
    )
    proof = verify_warmup_source(cls)
    assert proof["random_sampling_calls"] == 0
    objective = _FakeObjective()
    objective.loss_history = [[1.0]]
    objective.sensitivity_loss_history = [[0.0]]
    objective.penalty_history = [[0.0]]
    objective.is_feasible_history = [[1]]
    objective.time_steps = [1.0]
    with pytest.raises(WorkerError, match="strict Boolean"):
        _flatten(objective)
