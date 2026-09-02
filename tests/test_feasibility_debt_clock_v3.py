from __future__ import annotations

import ast
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("dfbench")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

from experiments.candidates import feasibility_debt_clock_v3_fixture as fixture
from experiments.candidates.feasibility_debt_clock_v3 import (
    FeasibilityDebtBatchedRestartAdamV3,
)
from experiments.candidates.feasibility_debt_clock_v3_source import (
    EXPECTED_OPTIMIZE_AST_SHA256,
    EXPECTED_TRANSITION_AST_SHA256,
    EXPECTED_VALIDATOR_AST_SHA256,
    source_projection,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CASE_SPECS = (
    ("protected_composite_trace_identity", 93503),
    ("pre_feasible_penalty_routing", 93521),
    ("first_feasible_irreversible_handoff", 93529),
    ("post_handoff_infeasible_reentry", 93553),
    ("masked_restart_state_rng_alignment", 93559),
    ("chunk_partition_trace_equivalence", 93581),
    ("partial_tail_transactionality", 93607),
    ("auxiliary_nonfinite_fail_closed", 93629),
    ("source_delta_and_process_seal", 93637),
)
EXPECTED_TRACE_KEYS = (
    "case",
    "seed",
    "batch",
    "admitted_count",
    "complete_population",
    "loss",
    "penalty",
    "feasible",
    "selected_progress",
    "improvement_mask",
    "stall_before",
    "stall_after",
    "restart_mask",
    "generation_before",
    "generation_after",
    "latch_before",
    "latch_after",
    "best_before",
    "best_after",
    "adam_age_before",
    "adam_age_after",
    "rng_before",
    "rng_after",
    "callback_count",
    "objective_eval_count",
    "update_applied",
)
EXPECTED_FIXTURE_AST = {
    "_case_aux_nonfinite": "c6301db7b4f6f4425b561e9888f208e888f732b3e2a1e4a65a3c49dce68f4db5",
    "_case_chunk_equivalence": "f8b274d923c783f12b1f7ed19c7112e3c1f52640ed7acbae52567dbdd580fba3",
    "_case_first_feasible": "b86051951c6270834023958ea1cfa2103bb1d3db3af085d2a0ee681582ec8fc3",
    "_case_infeasible_reentry": "da2effd535b1282839c989c8dafe3ce9b65790c2c3dd9981ac46b10b69e50bcd",
    "_case_masked_rng": "0f5b6353c494a2cc40959c4c0742226cd5ce6987573e90c2dd592d76c0627593",
    "_case_partial_tail": "c725278a4bbf6e5c35cca28bbc3d11c8d77d1c9084c5aee490094b739bf23c27",
    "_case_penalty_routing": "828d63271bf36bcbc1219ebd909fcbe1343c796b12fe6faa705678bfe6e48cf7",
    "_case_projection": "f0af9e1eaf18ae6af99a6e21ac50f3989921631f9b118528ac67999afb943988",
    "_case_protected_identity": "7207ecf3ef16874e2dd9b14c0cfd8b8a3143fe6377f5ba6d8e0faad0a41972cb",
    "_case_source_seal": "6dea9bcd1988861c8b8679b3422e546cb4d28e195d849f7c7073fda9c16d6d8c",
    "_chunk_rows": "4ff764116960c3af28c041b91bcea1f862967ebab6e4b3c894d0ee611df93daf",
    "_default_rows": "a4d120aca8be3f08704247730063248400fa4233e729fc2aedfdf020a2c0fdb5",
    "_simulate": "5d4b57687a2a629ce31d523d66c2ff083a0945347ff9d6e954087473f5b8f95a",
}


def _valid_aux(population: int = 3) -> dict[str, object]:
    values = jnp.arange(population, dtype=jnp.float32)
    return {
        "is_feasible": jnp.zeros((population,), dtype=bool),
        "penalty": values + 1.0,
        "sensitivity_loss": values,
        "violations": jnp.zeros((population, 2), dtype=jnp.float32),
        "power_values": {
            "hard": jnp.zeros((population, 1), dtype=jnp.float32),
            "soft": jnp.zeros((population, 1), dtype=jnp.float32),
            "detector": jnp.zeros((population, 1), dtype=jnp.float32),
        },
    }


def _probe_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["JAX_PLATFORMS"] = "cpu"
    return environment


def test_v3_plan_identity_and_complete_case_table_are_frozen() -> None:
    assert fixture.STUDY_ID == "feasibility-debt-clock-v3"
    assert fixture.PLAN_REVISION == "a61ba6003ec7cc5de5f41fc0c4349e62364ebd89"
    assert fixture.PLAN_SHA256 == (
        "1bf96ddd42c95dd9aa4ea516b1813929b6835f3949c4feb516fd2d7db62f57b8"
    )
    assert fixture.CASE_SPECS == EXPECTED_CASE_SPECS
    assert fixture.CASE_KEYS == tuple(name for name, _seed in EXPECTED_CASE_SPECS)
    assert fixture.TRACE_KEYS == EXPECTED_TRACE_KEYS
    assert fixture._sha256(fixture.PLAN_PATH.read_bytes()) == fixture.PLAN_SHA256


def test_v3_frozen_fixture_tables_and_cases_have_exact_ast() -> None:
    import hashlib

    tree = ast.parse(
        Path(fixture.__file__).read_text(encoding="utf-8")
    )
    observed = {
        node.name: hashlib.sha256(
            ast.dump(
                node, annotate_fields=True, include_attributes=False
            ).encode("utf-8")
        ).hexdigest()
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in EXPECTED_FIXTURE_AST
    }

    assert observed == EXPECTED_FIXTURE_AST


def test_v3_candidate_abi_requires_only_progress_mode() -> None:
    from submission.submission import BatchedRestartAdam

    protected = inspect.signature(BatchedRestartAdam.optimize).parameters
    candidate = inspect.signature(
        FeasibilityDebtBatchedRestartAdamV3.optimize
    ).parameters

    assert tuple(candidate) == tuple(protected)[:-1] + ("progress_mode", "kwargs")
    assert candidate["progress_mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert candidate["progress_mode"].default is inspect.Parameter.empty
    assert candidate["kwargs"].kind is inspect.Parameter.VAR_KEYWORD


def test_v3_source_boundary_is_exactly_pinned_without_running_cases() -> None:
    projection = source_projection()

    assert EXPECTED_OPTIMIZE_AST_SHA256 == (
        "85adb3dec2cfa83e0ed3ce6a6b826fd45fbcfe3d2c806dfdc44dd2711f2e990b"
    )
    assert EXPECTED_VALIDATOR_AST_SHA256 == (
        "40b6552a65518c97cff5cbb4d87647c6ee7cc7c303a0f5699eeff3315616db21"
    )
    assert EXPECTED_TRANSITION_AST_SHA256 == (
        "c4d1173d3235bc7dbb45439b1ff09c39503cedf84144fe8bdf5361fa16fb7f55"
    )
    assert set(projection) == {
        "protected_source_sha256",
        "candidate_source_sha256",
        "fixture_source_sha256",
        "worker_source_sha256",
        "plan_sha256",
        "normalized_delta_sha256",
        "source_boundary_root_sha256",
    }
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in projection.values()
    )


def test_v3_auxiliary_contract_accepts_exact_public_shape() -> None:
    FeasibilityDebtBatchedRestartAdamV3._validate_feasibility_debt_aux(
        _valid_aux(), 3
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda aux: aux.pop("penalty"),
        lambda aux: aux.__setitem__("extra", jnp.zeros((3,))),
        lambda aux: aux.__setitem__("is_feasible", jnp.zeros((3,), dtype=jnp.int32)),
        lambda aux: aux.__setitem__("penalty", jnp.zeros((3,), dtype=jnp.int32)),
        lambda aux: aux.__setitem__("penalty", jnp.asarray(1.0, dtype=jnp.float32)),
        lambda aux: aux.__setitem__("penalty", jnp.asarray([1.0, -1.0, 2.0])),
        lambda aux: aux["power_values"].__setitem__("extra", jnp.zeros((3, 1))),
    ],
)
def test_v3_auxiliary_contract_rejects_nonfrozen_smoke_attacks(mutator) -> None:
    auxiliary = copy.deepcopy(_valid_aux())
    mutator(auxiliary)

    with pytest.raises((TypeError, ValueError)):
        FeasibilityDebtBatchedRestartAdamV3._validate_feasibility_debt_aux(
            auxiliary, 3
        )


def test_v3_nonfrozen_transition_smoke_authenticates_handoff() -> None:
    auxiliary = _valid_aux()
    auxiliary["is_feasible"] = jnp.asarray([False, True, False])
    auxiliary["penalty"] = jnp.asarray([4.0, 3.0, 2.0])
    transition = FeasibilityDebtBatchedRestartAdamV3._progress_transition(
        jnp.asarray([9.0, 8.0, 7.0]),
        auxiliary,
        jnp.asarray([False, False, False]),
        jnp.asarray([5.0, jnp.inf, 1.0]),
        jnp.asarray([0, 4, 2]),
        0.0,
        "feasibility_debt",
    )

    assert np.asarray(transition[0]).tolist() == [4.0, 8.0, 2.0]
    assert np.asarray(transition[1]).tolist() == [True, True, False]
    assert np.asarray(transition[2]).tolist() == [False, True, False]
    assert np.asarray(transition[3]).tolist() == [4.0, 8.0, 1.0]
    assert np.asarray(transition[4]).tolist() == [0, 0, 3]

    auxiliary["is_feasible"] = jnp.asarray([False, False, False])
    auxiliary["penalty"] = jnp.asarray([100.0, 100.0, 100.0])
    later = FeasibilityDebtBatchedRestartAdamV3._progress_transition(
        jnp.asarray([3.0, 7.0, 6.0]),
        auxiliary,
        jnp.asarray([False, True, False]),
        jnp.asarray([4.0, 8.0, 1.0]),
        jnp.asarray([0, 0, 3]),
        0.0,
        "feasibility_debt",
    )
    assert float(np.asarray(later[0])[1]) == 7.0
    assert bool(np.asarray(later[1])[1]) is True
    assert bool(np.asarray(later[2])[1]) is True


def test_v3_total_loss_transition_ignores_penalty_bytes() -> None:
    first = _valid_aux()
    second = _valid_aux()
    first["penalty"] = jnp.asarray([1.0, 2.0, 3.0])
    second["penalty"] = jnp.asarray([300.0, jnp.nan, jnp.inf])
    arguments = (
        jnp.asarray([7.0, 6.0, 5.0]),
        jnp.asarray([False, False, False]),
        jnp.asarray([8.0, 5.0, jnp.inf]),
        jnp.asarray([1, 2, 3]),
        0.0,
        "total_loss",
    )

    left = FeasibilityDebtBatchedRestartAdamV3._progress_transition(
        arguments[0], first, *arguments[1:]
    )
    right = FeasibilityDebtBatchedRestartAdamV3._progress_transition(
        arguments[0], second, *arguments[1:]
    )

    assert all(
        np.array_equal(np.asarray(left_item), np.asarray(right_item))
        for left_item, right_item in zip(left, right, strict=True)
    )


def test_v3_closed_parent_failure_is_valid_and_complete() -> None:
    payload = fixture._closed_failure("1" * 40)

    assert fixture._validate_parent(payload) is True
    assert set(payload["case_outcomes"]) == set(fixture.CASE_KEYS)
    assert set(payload["transport_outcomes"]) == set(fixture.TRANSPORT_KEYS)
    assert len(payload["transport_outcomes"]) == 14
    assert not any(payload["case_outcomes"].values())
    assert not any(payload["transport_outcomes"].values())
    assert payload["action"] == "park_feasibility_debt_v3"


def test_v3_worker_schema_is_canonical_and_relational() -> None:
    outcomes = {key: True for key in sorted(fixture.CASE_KEYS)}
    payload = {
        "study_id": fixture.STUDY_ID,
        "invocation_revision": "a" * 40,
        "plan_revision": fixture.PLAN_REVISION,
        "plan_sha256": fixture.PLAN_SHA256,
        "protected_source_sha256": "b" * 64,
        "candidate_source_sha256": "c" * 64,
        "fixture_source_sha256": "d" * 64,
        "worker_source_sha256": "e" * 64,
        "case_count": 9,
        "case_outcomes": outcomes,
        "case_roots": {key: "f" * 64 for key in sorted(fixture.CASE_KEYS)},
        "all_cases_passed": True,
        "stdout_sealed": True,
        "source_boundary_root_sha256": "1" * 64,
    }
    payload["core_root_sha256"] = fixture._sha256(
        fixture._canonical_json(payload)
    )
    raw = fixture._canonical_json(payload) + b"\n"

    assert fixture._validate_worker_payload(payload, raw) is True
    expected_sources = {
        key: payload[key]
        for key in (
            "protected_source_sha256",
            "candidate_source_sha256",
            "fixture_source_sha256",
            "worker_source_sha256",
            "source_boundary_root_sha256",
        )
    }
    assert (
        fixture._worker_identity_valid(
            payload, payload["invocation_revision"], expected_sources
        )
        is True
    )

    wrong_revision = dict(payload)
    wrong_revision["invocation_revision"] = "9" * 40
    assert (
        fixture._worker_identity_valid(
            wrong_revision, payload["invocation_revision"], expected_sources
        )
        is False
    )

    wrong_source = dict(payload)
    wrong_source["candidate_source_sha256"] = "9" * 64
    assert (
        fixture._worker_identity_valid(
            wrong_source, payload["invocation_revision"], expected_sources
        )
        is False
    )

    inconsistent = dict(payload)
    inconsistent["all_cases_passed"] = False
    assert (
        fixture._validate_worker_payload(
            inconsistent, fixture._canonical_json(inconsistent) + b"\n"
        )
        is False
    )


def test_v3_environment_scrubs_secrets_and_forces_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAMPLE_API_KEY", "do-not-forward")
    monkeypatch.setenv("FDC_V3_IMPORT_NOISE_MODULE", "must-not-forward")
    monkeypatch.setenv("SAFE_MARKER", "retained")

    environment = fixture._scrubbed_environment()

    assert "EXAMPLE_API_KEY" not in environment
    assert "FDC_V3_IMPORT_NOISE_MODULE" not in environment
    assert environment["SAFE_MARKER"] == "retained"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["JAX_PLATFORMS"] == "cpu"
    assert environment["NO_PROXY"] == "*"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONPATH"] == ""
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_v3_worker_and_parent_process_boundaries_are_static_and_bounded() -> None:
    from experiments.candidates import feasibility_debt_clock_v3_worker as worker

    worker_tree = ast.parse(Path(worker.__file__).read_text(encoding="utf-8"))
    top_level_imports = {
        node.module.split(".", 1)[0]
        if isinstance(node, ast.ImportFrom) and node.module
        else alias.name.split(".", 1)[0]
        for node in worker_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (
            node.names if isinstance(node, ast.Import) else [node.names[0]]
        )
    }
    assert top_level_imports <= {
        "__future__",
        "argparse",
        "importlib",
        "json",
        "os",
        "sys",
        "typing",
    }
    main = next(
        node
        for node in worker_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    seal_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_seal_stdout"
    )
    result_import = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.ImportFrom)
        and node.module
        == "experiments.candidates.feasibility_debt_clock_v3_fixture"
    )
    assert seal_call.lineno < result_import.lineno
    assert worker.MAX_ENVELOPE_BYTES == 262_144
    assert fixture.MAX_WORKER_BYTES == 262_144
    assert len(fixture.TRANSPORT_KEYS) == 14

    fixture_tree = ast.parse(
        Path(fixture.__file__).read_text(encoding="utf-8")
    )
    run_worker = next(
        node
        for node in fixture_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_worker"
    )
    rendered = ast.unparse(run_worker)
    assert "--child" in rendered
    assert "timeout=180" in rendered
    assert "stdin=subprocess.DEVNULL" in rendered
    assert "stdout=subprocess.PIPE" in rendered
    assert "stderr=subprocess.PIPE" in rendered


def test_v3_transport_probe_physically_seals_python_and_fd_noise(
    tmp_path: Path,
) -> None:
    noise_module = tmp_path / "v3_import_noise.py"
    noise_module.write_text(
        "import os\n"
        "print('import-python-noise', flush=True)\n"
        "os.write(1, b'import-fd-noise\\n')\n",
        encoding="utf-8",
    )
    environment = _probe_environment()
    environment["FDC_V3_IMPORT_NOISE_MODULE"] = "v3_import_noise"
    environment["PYTHONPATH"] = str(tmp_path) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.candidates.feasibility_debt_clock_v3_worker",
            "--transport-probe",
        ],
        cwd=ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0
    assert completed.stderr == b""
    assert completed.stdout == (
        b'{"probe":"feasibility-debt-clock-v3","stdout_sealed":true}\n'
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "experiments.candidates.feasibility_debt_clock_v3_fixture",
        "experiments.candidates.feasibility_debt_clock_v3_worker",
        "experiments.candidates.feasibility_debt_clock_v3_source",
        "experiments.candidates.feasibility_debt_clock_v3",
    ],
)
def test_v3_module_import_is_quiet_in_fresh_process(module_name: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib,sys; importlib.import_module(sys.argv[1])",
            module_name,
        ],
        cwd=ROOT,
        env=_probe_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0
    assert completed.stdout == b""
    assert completed.stderr == b""


def test_v3_tests_cannot_call_a_terminal_path() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden_functions = {
        "_case_projection",
        "_worker_projection",
        "run_terminal_projection",
    }
    forbidden_modes = {"--child", "--run"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = None
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        assert function_name not in forbidden_functions
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(
                argument.value, str
            ):
                assert argument.value not in forbidden_modes


def test_v3_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        fixture._strict_json_loads(b'{"value":1,"value":2}')

    assert fixture._strict_json_loads(json.dumps({"value": 1}).encode()) == {
        "value": 1
    }
