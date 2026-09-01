from __future__ import annotations

import ast
import copy
import hashlib
import io
import importlib.util
import inspect
import json
import os
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration
np = pytest.importorskip("numpy")

from experiments.local_lab import constraint_aware_progress_toy as v1_fixture
from experiments.local_lab import constraint_aware_progress_toy_v2 as science
from experiments.local_lab import constraint_aware_progress_toy_v2_worker as worker
from tools import run_local_lab as controller


ROOT = Path(__file__).parents[1]
STUDY_ID = "constraint-aware-progress-toy-v2"
WORKER_MODULE = "experiments.local_lab.constraint_aware_progress_toy_v2_worker"
WORKER_PATH = ROOT / "experiments/local_lab/constraint_aware_progress_toy_v2_worker.py"
PLAN_REVISION = "c5314afaa50490e39c53669d971114d280e43c07"


def _registry() -> dict[str, object]:
    return json.loads(
        (ROOT / "experiments/local_lab/studies.json").read_text(encoding="utf-8")
    )


def _parked_state(registry: dict[str, object]) -> dict[str, object]:
    studies = registry["studies"]
    assert isinstance(studies, dict)
    state = controller._default_state()
    state["status"] = "parked"
    state["failure_streak"] = 1
    state["stop_reason"] = controller.CONSTRAINT_PROGRESS_V1_PARK_REASON
    state["completed_studies"] = {
        name: {
            "cycle_id": f"prior-{index}",
            "result_sha256": format(index + 1, "x") * 64,
            "revision": format(index + 2, "x") * 40,
            "status": "passed",
        }
        for index, name in enumerate(studies)
        if name not in controller.CONSTRAINT_PROGRESS_STUDIES
    }
    return state


def _prepare_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], bytes, bytes]:
    registry = _registry()
    entry = registry["studies"][STUDY_ID]
    assert isinstance(entry, dict)
    state = _parked_state(registry)
    controller._write_mutable_json(tmp_path / "lab-state.json", state)
    controller._append_event(
        tmp_path,
        {
            "event": "cycle_parked",
            "study": controller.CONSTRAINT_PROGRESS_V1,
            "utc": "2026-08-31T00:00:00Z",
        },
    )
    state_bytes = (tmp_path / "lab-state.json").read_bytes()
    event_bytes = (tmp_path / "lab-events.jsonl").read_bytes()
    revision = "a" * 40
    monkeypatch.setattr(controller, "_git", lambda *_args: revision)
    monkeypatch.setattr(controller, "_load_study_registry", lambda: registry)
    monkeypatch.setattr(
        controller,
        "_repository_snapshot",
        lambda _entry: {
            "committed_file_sha256": entry["approved_file_sha256"],
            "committed_source_paths": entry["source_paths"],
            "revision": revision,
        },
    )
    monkeypatch.setattr(controller, "_validate_study_approval", lambda *_args: None)
    return registry, state_bytes, event_bytes


def test_scientific_constants_preserve_the_frozen_v1_mechanics() -> None:
    assert science.STUDY_ID == STUDY_ID
    assert science.PLAN_REVISION == PLAN_REVISION
    assert science.PLAN_PATH == "research/2026-08-31-constraint-aware-progress-toy-v2-plan.md"
    for name in (
        "SCHEMA_DOMAIN",
        "TRANSCRIPT_DOMAIN",
        "FAMILIES",
        "SPLITS",
        "ORDERS",
        "ARMS",
        "SEEDS",
        "POPULATION",
        "BATCHES",
        "PATIENCE",
        "EVALUATIONS_PER_TRAJECTORY",
        "TRANSITIONS_PER_TRAJECTORY",
        "VALUES_PER_TRANSCRIPT",
        "TOTAL_TRANSCRIPT_VALUES",
        "IMPROVEMENT_TOLERANCE",
        "TRANSCRIPT_HASHES",
        "TRANSCRIPT_ROOT_SHA256",
        "RUNTIME_IDENTITY",
        "CAPABILITY_ATTACKS",
        "CAPABILITY_PATHS",
        "MALFORMED_ATTACKS",
        "OBSERVATION_FIELDS",
        "TRANSITION_FIELDS",
    ):
        assert getattr(science, name) == getattr(v1_fixture, name)
    assert science.fx is science


def test_every_v1_fixture_function_and_class_is_ast_identical_in_v2() -> None:
    def definitions(path: Path) -> dict[str, ast.AST]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }

    v1 = definitions(ROOT / "experiments/local_lab/constraint_aware_progress_toy.py")
    v2 = definitions(ROOT / "experiments/local_lab/constraint_aware_progress_toy_v2.py")
    assert v1
    assert set(v1) <= set(v2)
    for name, node in v1.items():
        assert ast.dump(v2[name], include_attributes=False) == ast.dump(
            node, include_attributes=False
        ), name


def test_worker_oracle_translation_is_exact_outside_the_four_frozen_deltas() -> None:
    def definitions(path: Path) -> dict[str, ast.AST]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }

    predecessor = definitions(
        ROOT / "experiments/local_lab/constraint_aware_progress_toy_worker.py"
    )
    translated = definitions(
        ROOT / "experiments/local_lab/constraint_aware_progress_toy_v2.py"
    )
    process_boundary = {
        "_ChildJob",
        "_NetworkDisabledSocket",
        "_await_child_job_gate",
        "_deny_network",
        "_disable_network",
        "_load_runtime",
        "_phase_payload",
        "_run_child",
        "_source_environment",
        "_terminate_child",
        "main",
    }
    wrappers = {"_phase", "_projection", "_full"}
    renamed = {
        "_positive_zero": "_oracle_positive_zero",
        "_sigmoid": "_oracle_sigmoid",
    }

    class BackTranslate(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):
            node = self.generic_visit(node)
            if node.id == "_oracle_positive_zero":
                node.id = "_positive_zero"
            elif node.id == "_oracle_sigmoid":
                node.id = "_sigmoid"
            elif node.id == "run_child":
                node.id = "_run_child"
            return node

        def visit_Constant(self, node: ast.Constant):
            node = self.generic_visit(node)
            if isinstance(node.value, str):
                node.value = node.value.replace(
                    "constraint-aware-progress-toy-v2",
                    "constraint-aware-progress-toy-v1",
                )
            return node

    for name in sorted(set(predecessor) - process_boundary - wrappers):
        observed_name = renamed.get(name, name)
        assert observed_name in translated, name
        observed = BackTranslate().visit(copy.deepcopy(translated[observed_name]))
        if isinstance(observed, (ast.FunctionDef, ast.ClassDef)):
            observed.name = name
        assert ast.dump(observed, include_attributes=False) == ast.dump(
            predecessor[name], include_attributes=False
        ), name

    old_phase = predecessor["_phase"]
    new_phase = translated["run_phase"]
    assert isinstance(old_phase, ast.FunctionDef) and isinstance(new_phase, ast.FunctionDef)
    assert ast.dump(
        ast.Module(body=old_phase.body[2:], type_ignores=[]), include_attributes=False
    ) == ast.dump(
        ast.Module(body=new_phase.body[2:], type_ignores=[]), include_attributes=False
    )
    for old_name, new_name in (("_projection", "run_projection"), ("_full", "run_full")):
        old = copy.deepcopy(predecessor[old_name])
        new = BackTranslate().visit(copy.deepcopy(translated[new_name]))
        assert isinstance(old, ast.FunctionDef) and isinstance(new, ast.FunctionDef)
        new.name = old.name
        new.args = copy.deepcopy(old.args)
        assert ast.dump(new, include_attributes=False) == ast.dump(
            old, include_attributes=False
        ), old_name


def test_transcript_world_and_action_preimages_are_byte_identical() -> None:
    predecessor_worlds = v1_fixture.build_world_records()
    translated_worlds = science.build_world_records()
    assert [item.to_dict() for item in translated_worlds] == [
        item.to_dict() for item in predecessor_worlds
    ]
    predecessor_transcripts = tuple(
        v1_fixture.build_transcript(seed) for seed in science.SEEDS
    )
    translated_transcripts = tuple(
        science.build_transcript(seed) for seed in science.SEEDS
    )
    for old, new in zip(predecessor_transcripts, translated_transcripts, strict=True):
        assert old.seed == new.seed
        assert old.sha256 == new.sha256
        assert old.fresh_hashes == new.fresh_hashes
        assert old.perturb_hashes == new.perturb_hashes
        assert np.array_equal(old.suffix, new.suffix)
        assert np.array_equal(old.fresh, new.fresh)
        assert np.array_equal(old.perturb, new.perturb)
    assert (
        v1_fixture.transcript_root(predecessor_transcripts)
        == science.transcript_root(translated_transcripts)
        == science.TRANSCRIPT_ROOT_SHA256
    )


def test_bootstrap_imports_and_dispatch_chronology_are_closed() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    assert "submission" not in source
    assert "constraint_aware_progress_toy_worker" not in source
    tree = ast.parse(source)
    imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert imports == {"__future__", "os", "struct", "sys"}
    module_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert module_calls == []
    assert worker.STUDY_ID == science.STUDY_ID == STUDY_ID
    assert worker.PLAN_REVISION == science.PLAN_REVISION == PLAN_REVISION
    assert worker.TRANSCRIPT_ROOT_SHA256 == science.TRANSCRIPT_ROOT_SHA256
    assert worker.MAX_PACKET_BYTES == science.MAX_PACKET_BYTES == 1_048_576
    assert worker.WORLD_FIELDS == tuple(science.WorldRecord.__dataclass_fields__)
    assert worker.PHASE_FIELDS == (
        "family",
        "split",
        "contract_sha256",
        "transcript_root_sha256",
        "worlds",
    )
    main_source = inspect.getsource(worker.main)
    assert main_source.index("_read_frame()") < main_source.index(
        "_authenticate_environment()"
    )
    assert main_source.index("_authenticate_environment()") < main_source.index(
        "_late_imports()"
    )
    assert main_source.index("_late_imports()") < main_source.index(
        "_validate_payload(mode, payload)"
    )
    assert main_source.index("_validate_payload(mode, payload)") < main_source.index(
        "_disable_network()"
    )
    assert main_source.index("_disable_network()") < main_source.index(
        "_prove_network_disabled()"
    )
    assert main_source.index("_prove_network_disabled()") < main_source.index(
        "_dispatch(mode, payload)"
    )
    assert main_source.index("_disable_network()") < main_source.index(
        "_dispatch(mode, payload)"
    )
    child_source = inspect.getsource(worker._run_child)
    assert child_source.index("job.assign_and_verify(process)") < child_source.index(
        "writer.start()"
    )
    assert child_source.index("started = time.monotonic()") < child_source.index(
        "subprocess.Popen("
    )


@pytest.mark.parametrize(
    ("payload", "stage"),
    (
        (b"", "gate"),
        (worker.GATE[:-1], "gate"),
        (worker.GATE + b"\x00\x00", "length"),
        (worker.GATE + struct.pack("<I", worker.MAX_PACKET_BYTES + 1), "payload"),
        (worker.GATE + struct.pack("<I", 0) + b"x", "payload"),
    ),
)
def test_recursive_bootstrap_framing_failures_are_sanitized(
    payload: bytes, stage: str
) -> None:
    environment = controller._worker_environment(
        {
            "L2D_CONTRACT_SHA256": "a" * 64,
            "L2D_PLAN_REVISION": PLAN_REVISION,
            "L2D_STUDY_REVISION": "b" * 40,
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-P",
            str(WORKER_PATH),
            "--mode",
            worker.PROJECTION_MODE,
        ],
        cwd=ROOT,
        env=environment,
        input=payload,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 70
    assert completed.stderr == b""
    assert completed.stdout == worker._failure_bytes(worker.PROJECTION_MODE, stage)


def test_controller_accepts_only_canonical_v2_failure_receipts() -> None:
    encoded = worker._failure_bytes(STUDY_ID, "dispatch")
    assert controller._constraint_progress_failure_stage(encoded, STUDY_ID) == "dispatch"
    with pytest.raises(RuntimeError, match="malformed"):
        controller._constraint_progress_failure_stage(
            encoded.replace(b'"dispatch"', b'"unknown"'), STUDY_ID
        )
    with pytest.raises(RuntimeError, match="noncanonical"):
        controller._constraint_progress_failure_stage(encoded[:-1] + b" \n", STUDY_ID)
    for malformed_version in (True, 1.0, "1", None):
        malformed = json.dumps(
            {
                "schema_version": malformed_version,
                "study_id": STUDY_ID,
                "mode": STUDY_ID,
                "stage": "dispatch",
            },
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        with pytest.raises(RuntimeError, match="malformed"):
            controller._constraint_progress_failure_stage(malformed, STUDY_ID)


def test_controller_refuses_retired_v2_before_worker_pairing() -> None:
    with pytest.raises(controller.QuarantinedStudyError, match="quarantined"):
        controller._run_worker(
            STUDY_ID,
            cycle_id="wrong-worker",
            heartbeat=lambda *_args: None,
            worker_module="experiments.local_lab.constraint_aware_progress_toy_worker",
        )
    with pytest.raises(RuntimeError, match="bound to another mode"):
        controller._run_worker(
            "policy-probe",
            cycle_id="wrong-mode",
            heartbeat=lambda *_args: None,
            worker_module=WORKER_MODULE,
        )


@pytest.mark.skipif(os.name != "nt", reason="V2 freezes the Windows CPython runtime")
def test_retired_v2_runtime_probe_is_not_reexecuted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen_called = False

    def forbidden_popen(*_args, **_kwargs):
        nonlocal popen_called
        popen_called = True
        raise AssertionError("retired V2 runtime child became reachable")

    monkeypatch.setattr(controller.subprocess, "Popen", forbidden_popen)
    with pytest.raises(controller.QuarantinedStudyError, match="quarantined"):
        controller._constraint_progress_runtime_identity(
            _registry()["studies"][STUDY_ID],
            study_revision="b" * 40,
            contract_sha256="a" * 64,
        )
    assert popen_called is False


@pytest.mark.skipif(os.name != "nt", reason="V2 freezes the Windows CPython runtime")
def test_controller_runtime_probe_uses_computed_v2_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    entry = registry["studies"][STUDY_ID]
    assert isinstance(entry, dict)
    contract = controller._constraint_progress_contract_sha256(
        registry, entry, STUDY_ID
    )
    popen_called = False

    def forbidden_popen(*_args, **_kwargs):
        nonlocal popen_called
        popen_called = True
        raise AssertionError("retired V2 runtime child became reachable")

    monkeypatch.setattr(controller.subprocess, "Popen", forbidden_popen)
    with pytest.raises(controller.QuarantinedStudyError, match="quarantined"):
        controller._constraint_progress_runtime_identity(
            entry,
            study_revision="b" * 40,
            contract_sha256=contract,
        )
    assert popen_called is False


@pytest.mark.parametrize(
    ("failure_kind", "expected_stage"),
    ((None, None), ("import", "import"), ("dispatch", "dispatch"), ("output", "output")),
)
def test_authenticated_recursive_path_gates_network_and_seals_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str | None,
    expected_stage: str | None,
) -> None:
    frame = worker.GATE + struct.pack("<I", 0)
    input_stream = types.SimpleNamespace(buffer=io.BytesIO(frame))
    output_buffer = io.BytesIO()
    error_buffer = io.BytesIO()
    output_stream = types.SimpleNamespace(buffer=output_buffer)
    error_stream = types.SimpleNamespace(buffer=error_buffer)
    fake_socket = types.SimpleNamespace(
        socket=lambda: None,
        create_connection=lambda *_args, **_kwargs: None,
        getaddrinfo=lambda *_args, **_kwargs: None,
        gethostbyaddr=lambda *_args, **_kwargs: None,
        gethostbyname=lambda *_args, **_kwargs: None,
        gethostbyname_ex=lambda *_args, **_kwargs: None,
    )
    original_late_imports = worker._late_imports
    events: list[str] = []

    def late_imports() -> None:
        original_late_imports()
        worker.socket = fake_socket
        events.append("late-imports")

    class FakeScience:
        @staticmethod
        def validate_runtime_identity() -> dict[str, str]:
            events.append("runtime-identity")
            with pytest.raises(worker.BootstrapFailure, match="environment"):
                fake_socket.socket()
            if failure_kind == "import":
                raise RuntimeError("secret-import-detail")
            return {"identity": "authenticated"}

        @staticmethod
        def run_projection(_runner):
            events.append("dispatch")
            if failure_kind == "dispatch":
                raise RuntimeError("secret-dispatch-detail")
            if failure_kind == "output":
                return {"forbidden": {1, 2, 3}}
            return {"ok": True}

    environment = controller._worker_environment(
        {
            "L2D_CONTRACT_SHA256": "a" * 64,
            "L2D_PLAN_REVISION": PLAN_REVISION,
            "L2D_STUDY_REVISION": "b" * 40,
        }
    )
    monkeypatch.setattr(worker.sys, "argv", ["worker.py", "--mode", worker.PROJECTION_MODE])
    monkeypatch.setattr(worker.sys, "stdin", input_stream)
    monkeypatch.setattr(worker.sys, "stdout", output_stream)
    monkeypatch.setattr(worker.sys, "stderr", error_stream)
    monkeypatch.setattr(worker.os, "environ", environment)
    monkeypatch.setattr(worker, "_late_imports", late_imports)
    monkeypatch.setattr(worker, "_load_scientific_module", lambda: FakeScience)

    if expected_stage is None:
        worker.main()
        assert output_buffer.getvalue() == b'{"ok":true}\n'
    else:
        with pytest.raises(SystemExit) as stopped:
            worker.main()
        assert stopped.value.code == 70
        assert output_buffer.getvalue() == worker._failure_bytes(
            worker.PROJECTION_MODE, expected_stage
        )
        assert b"secret" not in output_buffer.getvalue()
    assert error_buffer.getvalue() == b""
    assert events[0] == "late-imports"
    assert "runtime-identity" in events


def test_authenticated_recursive_dispatch_exercises_real_socket_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker._late_imports()
    real_socket = worker.socket
    names = (
        "socket",
        "create_connection",
        "getaddrinfo",
        "gethostbyaddr",
        "gethostbyname",
        "gethostbyname_ex",
    )
    originals = {name: getattr(real_socket, name) for name in names}
    frame = worker.GATE + struct.pack("<I", 0)
    output = io.BytesIO()

    class FakeScience:
        @staticmethod
        def validate_runtime_identity() -> dict[str, str]:
            worker._prove_network_disabled()
            return {"identity": "authenticated"}

        @staticmethod
        def run_projection(_runner):
            return {"network_gate": "denied"}

    environment = controller._worker_environment(
        {
            "L2D_CONTRACT_SHA256": "a" * 64,
            "L2D_PLAN_REVISION": PLAN_REVISION,
            "L2D_STUDY_REVISION": "b" * 40,
        }
    )
    monkeypatch.setattr(worker.sys, "argv", ["worker.py", "--mode", worker.PROJECTION_MODE])
    monkeypatch.setattr(
        worker.sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(frame))
    )
    monkeypatch.setattr(worker.sys, "stdout", types.SimpleNamespace(buffer=output))
    monkeypatch.setattr(
        worker.sys, "stderr", types.SimpleNamespace(buffer=io.BytesIO())
    )
    monkeypatch.setattr(worker.os, "environ", environment)
    monkeypatch.setattr(worker, "_load_scientific_module", lambda: FakeScience)
    try:
        worker.main()
        assert output.getvalue() == b'{"network_gate":"denied"}\n'
        for name in names:
            with pytest.raises(worker.BootstrapFailure, match="environment"):
                getattr(real_socket, name)()
    finally:
        for name, value in originals.items():
            setattr(real_socket, name, value)


def test_partial_result_write_never_appends_a_failure_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ShortBuffer(io.BytesIO):
        def write(self, value: bytes) -> int:
            super().write(value[:3])
            return 3

    frame = worker.GATE + struct.pack("<I", 0)
    output_buffer = ShortBuffer()
    fake_socket = types.SimpleNamespace(
        socket=lambda: None,
        create_connection=lambda *_args, **_kwargs: None,
        getaddrinfo=lambda *_args, **_kwargs: None,
        gethostbyaddr=lambda *_args, **_kwargs: None,
        gethostbyname=lambda *_args, **_kwargs: None,
        gethostbyname_ex=lambda *_args, **_kwargs: None,
    )
    original_late_imports = worker._late_imports

    def late_imports() -> None:
        original_late_imports()
        worker.socket = fake_socket

    fake_science = types.SimpleNamespace(
        validate_runtime_identity=lambda: {},
        run_projection=lambda _runner: {"ok": True},
    )
    environment = controller._worker_environment(
        {
            "L2D_CONTRACT_SHA256": "a" * 64,
            "L2D_PLAN_REVISION": PLAN_REVISION,
            "L2D_STUDY_REVISION": "b" * 40,
        }
    )
    monkeypatch.setattr(worker.sys, "argv", ["worker.py", "--mode", worker.PROJECTION_MODE])
    monkeypatch.setattr(
        worker.sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(frame))
    )
    monkeypatch.setattr(
        worker.sys, "stdout", types.SimpleNamespace(buffer=output_buffer)
    )
    monkeypatch.setattr(
        worker.sys, "stderr", types.SimpleNamespace(buffer=io.BytesIO())
    )
    monkeypatch.setattr(worker.os, "environ", environment)
    monkeypatch.setattr(worker, "_late_imports", late_imports)
    monkeypatch.setattr(worker, "_load_scientific_module", lambda: fake_science)
    with pytest.raises(SystemExit) as stopped:
        worker.main()
    assert stopped.value.code == 70
    assert output_buffer.getvalue() == b'{"o'
    assert b"schema_version" not in output_buffer.getvalue()


@pytest.mark.skipif(os.name != "nt", reason="V2 uses a frozen Windows Job boundary")
def test_real_recursive_child_reaches_network_gate_then_inert_import_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker._late_imports()
    temporary_root = Path(worker.tempfile.gettempdir())
    before_directories = set(temporary_root.glob("l2d-constraint-v2-*"))
    before_threads = {
        thread.ident
        for thread in worker.threading.enumerate()
        if thread.name == "constraint-v2-frame-writer"
    }
    launched = []
    real_popen = worker.subprocess.Popen

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        launched.append(process)
        return process

    # Execute an exact copy of the real bootstrap, but place it in a temporary
    # package whose scientific module is an inert import refusal.  The child
    # must therefore authenticate its frame and environment, install and
    # self-test every real socket denial, and only then reach the sealed import
    # failure.  It can never construct a world, learner, optimizer, or result,
    # even when the host happens to match the frozen scientific runtime.
    sandbox_root = tmp_path / "inert-bootstrap"
    package_root = sandbox_root / "experiments" / "local_lab"
    package_root.mkdir(parents=True)
    (sandbox_root / "experiments" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    inert_worker = package_root / WORKER_PATH.name
    inert_worker.write_bytes(WORKER_PATH.read_bytes())
    (package_root / "constraint_aware_progress_toy_v2.py").write_text(
        'raise RuntimeError("inert scientific module must not load")\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(worker.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(worker, "__file__", str(inert_worker))
    monkeypatch.setattr(
        worker,
        "_source_environment",
        lambda: controller._worker_environment(
            {
                "L2D_CONTRACT_SHA256": "a" * 64,
                "L2D_PLAN_REVISION": PLAN_REVISION,
                "L2D_STUDY_REVISION": "b" * 40,
            }
        ),
    )
    with pytest.raises(worker.BootstrapFailure, match="import"):
        worker._run_child(worker.PROJECTION_MODE)
    assert len(launched) == 1
    assert launched[0].returncode == 70
    assert launched[0].poll() == 70
    assert set(temporary_root.glob("l2d-constraint-v2-*")) == before_directories
    assert {
        thread.ident
        for thread in worker.threading.enumerate()
        if thread.name == "constraint-v2-frame-writer"
    } == before_threads


@pytest.mark.skipif(os.name != "nt", reason="V2 uses a frozen Windows Job boundary")
@pytest.mark.parametrize(
    ("script", "limit", "expected_stage"),
    (
        (
            "import sys;sys.stdin.buffer.read();"
            "sys.stdout.buffer.write(b'{\"ok\":true}\\n');sys.stdout.flush()",
            1_048_576,
            None,
        ),
        (
            "import sys;sys.stdin.buffer.read();"
            "sys.stdout.buffer.write(b'x'*2048);sys.stdout.flush()",
            1024,
            "output",
        ),
        (
            "import sys;sys.stdin.buffer.read();"
            "sys.stdout.buffer.write(b'{\"ok\": true}\\n');sys.stdout.flush()",
            1_048_576,
            "output",
        ),
        (
            "import sys;sys.stdin.buffer.read();"
            "sys.stderr.buffer.write(b'closed-error');sys.stderr.flush()",
            1_048_576,
            "dispatch",
        ),
        ("import sys;sys.stdin.buffer.read();raise SystemExit(9)", 1_048_576, "dispatch"),
    ),
)
def test_real_job_supervisor_bounds_success_output_stderr_and_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    script: str,
    limit: int,
    expected_stage: str | None,
) -> None:
    worker._late_imports()
    real_popen = worker.subprocess.Popen
    launched = []
    temporary_root = Path(worker.tempfile.gettempdir())
    before_directories = set(temporary_root.glob("l2d-constraint-v2-*"))

    def scripted_popen(_command, *args, **kwargs):
        process = real_popen([sys.executable, "-S", "-P", "-c", script], *args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(worker.subprocess, "Popen", scripted_popen)
    monkeypatch.setattr(worker, "MAX_PACKET_BYTES", limit)
    if expected_stage is None:
        stdout, returncode, receipt = worker._run_child(worker.PROJECTION_MODE)
        assert (stdout, returncode, receipt) == (
            b'{"ok":true}\n',
            0,
            {"stdout_bytes": 12, "stderr_bytes": 0},
        )
    else:
        with pytest.raises(worker.BootstrapFailure, match=expected_stage):
            worker._run_child(worker.PROJECTION_MODE)
    assert len(launched) == 1
    assert launched[0].poll() is not None
    assert set(temporary_root.glob("l2d-constraint-v2-*")) == before_directories
    assert not any(
        thread.name == "constraint-v2-frame-writer" and thread.is_alive()
        for thread in worker.threading.enumerate()
    )


@pytest.mark.skipif(os.name != "nt", reason="V2 uses a frozen Windows Job boundary")
def test_real_job_supervisor_bounds_timeout_and_writer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker._late_imports()
    real_popen = worker.subprocess.Popen
    launched = []
    temporary_root = Path(worker.tempfile.gettempdir())
    before_directories = set(temporary_root.glob("l2d-constraint-v2-*"))

    def sleeping_popen(_command, *args, **kwargs):
        process = real_popen(
            [sys.executable, "-S", "-P", "-c", "import time;time.sleep(60)"],
            *args,
            **kwargs,
        )
        launched.append(process)
        return process

    monkeypatch.setattr(worker.subprocess, "Popen", sleeping_popen)
    monkeypatch.setattr(worker, "CHILD_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(worker.BootstrapFailure, match="cleanup"):
        worker._run_child(worker.PROJECTION_MODE)
    assert launched[-1].poll() is not None
    assert set(temporary_root.glob("l2d-constraint-v2-*")) == before_directories
    assert not any(
        thread.name == "constraint-v2-frame-writer" and thread.is_alive()
        for thread in worker.threading.enumerate()
    )


@pytest.mark.skipif(os.name != "nt", reason="V2 uses a frozen Windows Job boundary")
def test_real_job_timeout_kills_a_spawned_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes
    from ctypes import wintypes

    worker._late_imports()
    real_popen = worker.subprocess.Popen
    pid_path = tmp_path / "descendant.pid"
    temporary_root = Path(worker.tempfile.gettempdir())
    before_directories = set(temporary_root.glob("l2d-constraint-v2-*"))
    before_threads = {
        thread.ident
        for thread in worker.threading.enumerate()
        if thread.name == "constraint-v2-frame-writer"
    }
    script = (
        "import pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-S','-P','-c',"
        "'import time;time.sleep(60)']);"
        f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid),encoding='ascii');"
        "time.sleep(60)"
    )

    def descendant_popen(_command, *args, **kwargs):
        return real_popen([sys.executable, "-S", "-P", "-c", script], *args, **kwargs)

    monkeypatch.setattr(worker.subprocess, "Popen", descendant_popen)
    monkeypatch.setattr(worker, "CHILD_TIMEOUT_SECONDS", 1.0)
    with pytest.raises(worker.BootstrapFailure, match="cleanup"):
        worker._run_child(worker.PROJECTION_MODE)

    assert pid_path.is_file()
    descendant_pid = int(pid_path.read_text(encoding="ascii"))
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    process_handle = kernel32.OpenProcess(0x1000, False, descendant_pid)
    if process_handle:
        try:
            exit_code = wintypes.DWORD()
            assert kernel32.GetExitCodeProcess(process_handle, ctypes.byref(exit_code))
            assert exit_code.value != 259
        finally:
            assert kernel32.CloseHandle(process_handle)
    assert set(temporary_root.glob("l2d-constraint-v2-*")) == before_directories
    assert {
        thread.ident
        for thread in worker.threading.enumerate()
        if thread.name == "constraint-v2-frame-writer"
    } == before_threads


@pytest.mark.skipif(os.name != "nt", reason="V2 uses a frozen Windows Job boundary")
def test_real_job_supervisor_bounds_writer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker._late_imports()
    real_popen = worker.subprocess.Popen
    launched = []
    temporary_root = Path(worker.tempfile.gettempdir())
    before_directories = set(temporary_root.glob("l2d-constraint-v2-*"))
    before_threads = {
        thread.ident
        for thread in worker.threading.enumerate()
        if thread.name == "constraint-v2-frame-writer"
    }

    class FailingStdin:
        def __init__(self, original):
            self.original = original

        def write(self, _value):
            raise OSError("injected-writer-failure")

        def close(self):
            self.original.close()

        def fileno(self):
            return self.original.fileno()

    def failing_writer_popen(_command, *args, **kwargs):
        process = real_popen(
            [sys.executable, "-S", "-P", "-c", "import time;time.sleep(60)"],
            *args,
            **kwargs,
        )
        process.stdin = FailingStdin(process.stdin)
        launched.append(process)
        return process

    monkeypatch.setattr(worker.subprocess, "Popen", failing_writer_popen)
    monkeypatch.setattr(worker, "CHILD_TIMEOUT_SECONDS", 60)
    with pytest.raises(worker.BootstrapFailure, match="cleanup"):
        worker._run_child(worker.PROJECTION_MODE)
    assert launched[-1].poll() is not None
    assert set(temporary_root.glob("l2d-constraint-v2-*")) == before_directories
    assert {
        thread.ident
        for thread in worker.threading.enumerate()
        if thread.name == "constraint-v2-frame-writer"
    } == before_threads


@pytest.mark.skipif(os.name != "nt", reason="V2 uses a frozen Windows Job boundary")
def test_terminate_failure_still_drains_job_thread_and_temporary_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker._late_imports()
    real_popen = worker.subprocess.Popen
    real_terminate = worker._Job.terminate
    temporary_root = Path(worker.tempfile.gettempdir())
    before_directories = set(temporary_root.glob("l2d-constraint-v2-*"))
    launched = []

    def sleeping_popen(_command, *args, **kwargs):
        process = real_popen(
            [sys.executable, "-S", "-P", "-c", "import time;time.sleep(60)"],
            *args,
            **kwargs,
        )
        launched.append(process)
        return process

    def terminate_then_fail(self):
        real_terminate(self)
        raise worker.BootstrapFailure("cleanup")

    monkeypatch.setattr(worker.subprocess, "Popen", sleeping_popen)
    monkeypatch.setattr(worker._Job, "terminate", terminate_then_fail)
    monkeypatch.setattr(worker, "CHILD_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(worker.BootstrapFailure, match="cleanup"):
        worker._run_child(worker.PROJECTION_MODE)
    assert launched[-1].poll() is not None
    assert set(temporary_root.glob("l2d-constraint-v2-*")) == before_directories
    assert not any(
        thread.name == "constraint-v2-frame-writer" and thread.is_alive()
        for thread in worker.threading.enumerate()
    )


@pytest.mark.skipif(os.name != "nt", reason="V2 uses a frozen Windows Job boundary")
def test_real_job_supervisor_fails_closed_on_job_boundary_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker._late_imports()
    real_popen = worker.subprocess.Popen
    real_assign = worker._Job.assign_and_verify
    real_active = worker._Job.active_processes
    real_close = worker._Job.close
    launched = []
    temporary_root = Path(worker.tempfile.gettempdir())
    before_directories = set(temporary_root.glob("l2d-constraint-v2-*"))

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(worker.subprocess, "Popen", recording_popen)

    def reject_assignment(self, process):
        real_assign(self, process)
        raise worker.BootstrapFailure("environment")

    monkeypatch.setattr(worker._Job, "assign_and_verify", reject_assignment)
    with pytest.raises(worker.BootstrapFailure, match="environment"):
        worker._run_child(worker.PROJECTION_MODE)
    assert launched[-1].poll() is not None

    monkeypatch.setattr(worker._Job, "assign_and_verify", real_assign)
    calls = {"active": 0}

    def reject_query(self):
        calls["active"] += 1
        if calls["active"] >= 1:
            raise worker.BootstrapFailure("cleanup")
        return real_active(self)

    monkeypatch.setattr(worker._Job, "active_processes", reject_query)
    with pytest.raises(worker.BootstrapFailure, match="cleanup"):
        worker._run_child(worker.PROJECTION_MODE)
    assert launched[-1].poll() is not None
    assert set(temporary_root.glob("l2d-constraint-v2-*")) == before_directories
    assert not any(
        thread.name == "constraint-v2-frame-writer" and thread.is_alive()
        for thread in worker.threading.enumerate()
    )

    monkeypatch.setattr(worker._Job, "active_processes", real_active)

    def reject_close(self):
        real_close(self)
        raise worker.BootstrapFailure("cleanup")

    monkeypatch.setattr(worker._Job, "close", reject_close)
    with pytest.raises(worker.BootstrapFailure, match="cleanup"):
        worker._run_child(worker.PROJECTION_MODE)
    assert launched[-1].poll() is not None


@pytest.mark.skipif(os.name != "nt", reason="V2 uses a frozen Windows Job boundary")
def test_job_api_false_returns_fail_closed_without_leaking_handles() -> None:
    worker._late_imports()
    job = worker._Job()
    kernel32 = job._kernel32
    try:
        job._kernel32 = types.SimpleNamespace(
            TerminateJobObject=lambda *_args: 0,
        )
        with pytest.raises(worker.BootstrapFailure, match="cleanup"):
            job.terminate()
    finally:
        job._kernel32 = kernel32
        job.close()

    job = worker._Job()
    kernel32 = job._kernel32
    try:
        job._kernel32 = types.SimpleNamespace(CloseHandle=lambda *_args: 0)
        with pytest.raises(worker.BootstrapFailure, match="cleanup"):
            job.close()
        assert job.handle is not None
    finally:
        job._kernel32 = kernel32
        job.close()


def test_bounded_reader_and_packet_types_fail_closed(tmp_path: Path) -> None:
    worker._late_imports()
    oversized = tmp_path / "oversized"
    oversized.write_bytes(b"x" * 9)
    with pytest.raises(worker.BootstrapFailure, match="output"):
        worker._read_bounded(oversized, 8)
    with pytest.raises(RuntimeError, match="retention cap"):
        controller._read_bounded_file(oversized, 8)
    base = {
        "family": "canonical",
        "world": 0,
        "bits": [0, 0, 0, 0],
        "split": "development",
        "a": 1.0,
        "b": 1.0,
        "k": 1.0,
        "t": 1.0,
        "c": 1.0,
        "threshold": 1.0,
        "reference_x0": None,
        "reference_sensitivity": None,
        "denominator": None,
    }
    for field in ("family", "split"):
        malformed = dict(base)
        malformed[field] = []
        with pytest.raises(worker.BootstrapFailure, match="payload"):
            worker._validate_world_packet(malformed)


def test_controller_cleanup_removes_files_when_tree_termination_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProcess:
        pid = 4242
        returncode = None

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(controller, "CYCLE_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(
        controller,
        "_terminate_process_tree",
        lambda _process: (_ for _ in ()).throw(RuntimeError("injected-termination")),
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        controller._run_worker(
            "policy-probe",
            cycle_id="cleanup-failure",
            heartbeat=lambda *_args: None,
        )
    assert not list((tmp_path / "worker-tmp").glob("cleanup-failure.*"))


def test_registry_retains_both_failed_contracts_as_quarantined_history() -> None:
    registry = _registry()
    studies = registry["studies"]
    assert isinstance(studies, dict)
    v1 = studies[controller.CONSTRAINT_PROGRESS_V1]
    v2 = studies[STUDY_ID]
    assert isinstance(v1, dict) and isinstance(v2, dict)
    assert controller.CONSTRAINT_PROGRESS_V1 in controller.QUARANTINED_STUDIES
    assert STUDY_ID in controller.QUARANTINED_STUDIES
    assert v2["worker_module"] == WORKER_MODULE
    assert controller.WORKER_MODULE_PATHS[WORKER_MODULE] == v2["source_paths"][
        "worker_source"
    ]
    assert v2["plan_revision"] == PLAN_REVISION
    assert v2["case_ids"] == v1["case_ids"]
    assert v2["case_metric_schema"] == v1["case_metric_schema"]
    assert v2["fixture_identity"] == v1["fixture_identity"]
    assert v2["result_top_level_fields"] == v1["result_top_level_fields"]
    assert v2["runtime_identity"] == v1["runtime_identity"]
    assert v2["transcript_commitment"] == v1["transcript_commitment"]
    assert v2["success_action"] == v1["success_action"]
    assert v2["failure_action"] == v1["failure_action"]


def test_v2_controller_recomputes_result_identity_and_rejects_tampering() -> None:
    spec = importlib.util.spec_from_file_location(
        "historical_constraint_v1_tests",
        ROOT / "tests/test_constraint_aware_progress_toy.py",
    )
    assert spec is not None and spec.loader is not None
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    entry = _registry()["studies"][STUDY_ID]
    assert isinstance(entry, dict)
    revision = "b" * 40
    contract = "c" * 64
    result = historical._valid_result(entry, revision, contract)
    result["study_id"] = STUDY_ID
    controller._validate_study_result(
        STUDY_ID,
        entry,
        result,
        study_revision=revision,
        contract_sha256=contract,
        worker_receipt=historical.WORKER_RECEIPT,
    )
    tampered = copy.deepcopy(result)
    tampered["world_aggregates"][0]["mean_gap"] += 0.01
    with pytest.raises(RuntimeError, match="aggregate value"):
        controller._validate_study_result(
            STUDY_ID,
            entry,
            tampered,
            study_revision=revision,
            contract_sha256=contract,
            worker_receipt=historical.WORKER_RECEIPT,
        )
    for field, malformed_value in (
        ("stderr_bytes", False),
        ("stderr_bytes", 0.0),
        ("stderr_sha256", 0),
        ("stderr_sha256", "not-a-sha256"),
    ):
        malformed_receipt = dict(historical.WORKER_RECEIPT)
        malformed_receipt[field] = malformed_value
        with pytest.raises(RuntimeError, match="outer worker receipt"):
            controller._validate_study_result(
                STUDY_ID,
                entry,
                result,
                study_revision=revision,
                contract_sha256=contract,
                worker_receipt=malformed_receipt,
            )


def test_outer_contract_domain_changes_only_with_the_versioned_study(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    studies = registry["studies"]
    assert isinstance(studies, dict)
    plan = b"frozen-plan\n"
    monkeypatch.setattr(controller, "_git_bytes", lambda *_args: plan)
    normalized = json.dumps(
        registry,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    for study_id, version in (
        (controller.CONSTRAINT_PROGRESS_V1, "v1"),
        (controller.CONSTRAINT_PROGRESS_V2, "v2"),
    ):
        entry = studies[study_id]
        assert isinstance(entry, dict)
        expected = hashlib.sha256(
            f"L2D-constraint-progress-{version}/contract\0".encode("ascii")
            + plan
            + b"\0"
            + normalized
        ).hexdigest()
        assert (
            controller._constraint_progress_contract_sha256(
                registry, entry, study_id
            )
            == expected
        )


def test_resume_transaction_preserves_ledger_and_appends_exact_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry_value, state_before, events_before = _prepare_resume(tmp_path, monkeypatch)
    before = json.loads(state_before)
    observed = controller._resume_constraint_progress_v2(tmp_path)
    assert observed["status"] == "awaiting_study"
    assert observed["stop_reason"] is None
    assert observed["failure_streak"] == before["failure_streak"]
    assert observed["completed_studies"] == before["completed_studies"]
    event_delta = (tmp_path / "lab-events.jsonl").read_bytes()[len(events_before) :]
    event = json.loads(event_delta)
    assert event == {
        "event": "controller_resumed",
        "event_schema_version": 1,
        "from_status": "parked",
        "reason": "owner_authorized_v1_quarantine_recovery",
        "retired_study": controller.CONSTRAINT_PROGRESS_V1,
        "to_status": "awaiting_study",
        "utc": event["utc"],
    }
    assert not (tmp_path / "lab.lock").exists()


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("missing-cycle", None),
        ("extra-field", True),
        ("bad-result", "g" * 64),
        ("bad-revision", "a" * 39),
        ("bad-status-type", True),
        ("bad-cycle-type", 1),
    ),
)
def test_resume_requires_exact_prior_completion_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    value: object,
) -> None:
    _registry_value, _state_before, events_before = _prepare_resume(tmp_path, monkeypatch)
    state = controller._load_state(tmp_path)
    completed = state["completed_studies"]
    assert isinstance(completed, dict)
    first = next(iter(completed.values()))
    assert isinstance(first, dict)
    if mutation == "missing-cycle":
        first.pop("cycle_id")
    elif mutation == "extra-field":
        first["unexpected"] = value
    elif mutation == "bad-result":
        first["result_sha256"] = value
    elif mutation == "bad-revision":
        first["revision"] = value
    elif mutation == "bad-status-type":
        first["status"] = value
    elif mutation == "bad-cycle-type":
        first["cycle_id"] = value
    controller._write_mutable_json(tmp_path / "lab-state.json", state)
    malformed_state = (tmp_path / "lab-state.json").read_bytes()
    with pytest.raises(RuntimeError, match="preconditions"):
        controller._resume_constraint_progress_v2(tmp_path)
    assert (tmp_path / "lab-state.json").read_bytes() == malformed_state
    assert (tmp_path / "lab-events.jsonl").read_bytes() == events_before
    assert not (tmp_path / "lab.lock").exists()


def test_resume_rejects_noncanonical_event_history_and_retains_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry_value, state_before, events_before = _prepare_resume(tmp_path, monkeypatch)
    corrupted = events_before.removesuffix(b"\n")
    (tmp_path / "lab-events.jsonl").write_bytes(corrupted)
    with pytest.raises(controller.ControlLedgerIntegrityError, match="final newline"):
        controller._resume_constraint_progress_v2(tmp_path)
    assert (tmp_path / "lab-state.json").read_bytes() == state_before
    assert (tmp_path / "lab-events.jsonl").read_bytes() == corrupted
    assert (tmp_path / "lab.lock/lease.json").is_file()


def test_resume_rejects_mismatched_v2_identity_before_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, state_before, events_before = _prepare_resume(tmp_path, monkeypatch)
    entry = registry["studies"][STUDY_ID]
    assert isinstance(entry, dict)
    entry["worker_mode"] = "policy-probe"
    monkeypatch.setattr(
        controller,
        "_repository_snapshot",
        lambda *_args: (_ for _ in ()).throw(AssertionError("snapshot reached")),
    )
    with pytest.raises(RuntimeError, match="identity"):
        controller._resume_constraint_progress_v2(tmp_path)
    assert (tmp_path / "lab-state.json").read_bytes() == state_before
    assert (tmp_path / "lab-events.jsonl").read_bytes() == events_before
    assert not (tmp_path / "lab.lock").exists()


def test_resume_postcommit_failure_retains_lease_and_committed_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry_value, _state_before, events_before = _prepare_resume(tmp_path, monkeypatch)
    original_heartbeat = controller._heartbeat_lease

    def heartbeat_then_fail(lock_directory: Path, lease_id: str, *, phase: str) -> None:
        original_heartbeat(lock_directory, lease_id, phase=phase)
        if phase == "resume-verified":
            raise OSError("injected-postcommit-heartbeat")

    monkeypatch.setattr(controller, "_heartbeat_lease", heartbeat_then_fail)
    with pytest.raises(OSError, match="postcommit"):
        controller._resume_constraint_progress_v2(tmp_path)
    state = controller._load_state(tmp_path)
    assert state["status"] == "awaiting_study"
    assert state["stop_reason"] is None
    assert (tmp_path / "lab-events.jsonl").read_bytes().startswith(events_before)
    assert (tmp_path / "lab.lock/lease.json").is_file()


def test_resume_cli_refuses_retired_v2_before_resume_or_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        controller,
        "_resume_constraint_progress_v2",
        lambda: calls.append("resume"),
    )
    monkeypatch.setattr(
        controller,
        "_run_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("resume dispatched worker")
        ),
    )
    monkeypatch.setattr(
        controller.sys,
        "argv",
        ["run_local_lab.py", "--resume-constraint-progress-v2"],
    )
    with pytest.raises(controller.QuarantinedStudyError):
        controller.main()
    assert calls == []


@pytest.mark.parametrize("failure_point", ("state", "event"))
def test_resume_write_failures_restore_exact_parked_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    _registry_value, state_before, events_before = _prepare_resume(tmp_path, monkeypatch)
    if failure_point == "state":
        original_save = controller._save_state

        def failing_save(root: Path, state: dict[str, object]) -> None:
            original_save(root, state)
            raise OSError("injected-state-write")

        monkeypatch.setattr(controller, "_save_state", failing_save)
    else:
        monkeypatch.setattr(
            controller,
            "_append_event",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected-event-append")
            ),
        )
    with pytest.raises(OSError, match="injected"):
        controller._resume_constraint_progress_v2(tmp_path)
    assert (tmp_path / "lab-state.json").read_bytes() == state_before
    assert (tmp_path / "lab-events.jsonl").read_bytes() == events_before
    assert not (tmp_path / "lab.lock").exists()


def test_resume_rollback_failure_retains_the_global_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_resume(tmp_path, monkeypatch)
    monkeypatch.setattr(
        controller,
        "_append_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("event-failure")),
    )
    monkeypatch.setattr(
        controller,
        "_replace_mutable_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("rollback-failure")),
    )
    with pytest.raises(RuntimeError, match="lease retained"):
        controller._resume_constraint_progress_v2(tmp_path)
    assert (tmp_path / "lab.lock/lease.json").is_file()
    assert controller._load_state(tmp_path)["status"] == "parked"


def test_resume_rejects_noncanonical_or_missing_active_state_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry_value, state_before, events_before = _prepare_resume(tmp_path, monkeypatch)
    state = json.loads(state_before)
    noncanonical = json.dumps(state, separators=(",", ":")).encode("utf-8")
    (tmp_path / "lab-state.json").write_bytes(noncanonical)
    with pytest.raises(RuntimeError, match="canonical"):
        controller._resume_constraint_progress_v2(tmp_path)
    assert (tmp_path / "lab-state.json").read_bytes() == noncanonical
    assert (tmp_path / "lab-events.jsonl").read_bytes() == events_before

    controller._write_mutable_json(tmp_path / "lab-state.json", state)
    state.pop("active_cycle")
    controller._write_mutable_json(tmp_path / "lab-state.json", state)
    missing_before = (tmp_path / "lab-state.json").read_bytes()
    with pytest.raises(RuntimeError, match="preconditions"):
        controller._resume_constraint_progress_v2(tmp_path)
    assert (tmp_path / "lab-state.json").read_bytes() == missing_before
    assert (tmp_path / "lab-events.jsonl").read_bytes() == events_before


def test_resume_stop_marker_race_rolls_back_event_before_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry_value, state_before, events_before = _prepare_resume(tmp_path, monkeypatch)
    original_append = controller._append_event

    def append_then_stop(root: Path, event: dict[str, object]) -> None:
        original_append(root, event)
        controller._write_mutable_json(
            root / "stop.request.json", {"reason": "injected-owner-stop"}
        )

    monkeypatch.setattr(controller, "_append_event", append_then_stop)
    with pytest.raises(RuntimeError, match="owner stop request appeared"):
        controller._resume_constraint_progress_v2(tmp_path)
    assert (tmp_path / "lab-state.json").read_bytes() == state_before
    assert (tmp_path / "lab-events.jsonl").read_bytes() == events_before
    assert not (tmp_path / "lab.lock").exists()


def test_retired_v2_cannot_reenter_the_live_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry()
    studies = registry["studies"]
    assert isinstance(studies, dict)
    entry = studies[STUDY_ID]
    assert isinstance(entry, dict)
    revision = "c" * 40
    state = controller._default_state()
    state["status"] = "awaiting_study"
    state["completed_studies"] = {
        name: {
            "cycle_id": f"prior-{index}",
            "result_sha256": format(index + 1, "x") * 64,
            "revision": format(index + 2, "x") * 40,
            "status": "passed",
        }
        for index, name in enumerate(studies)
        if name not in controller.CONSTRAINT_PROGRESS_STUDIES
    }
    controller._write_mutable_json(tmp_path / "lab-state.json", state)
    state_before = (tmp_path / "lab-state.json").read_bytes()
    output = tmp_path / "cycles/v2-state-test/result.json"
    monkeypatch.setattr(controller, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(controller, "_load_study_registry", lambda: registry)
    monkeypatch.setattr(controller, "_git", lambda *_args: revision)
    monkeypatch.setattr(
        controller,
        "_repository_snapshot",
        lambda _entry: {
            "committed_file_sha256": entry["approved_file_sha256"],
            "committed_source_paths": entry["source_paths"],
            "revision": revision,
        },
    )
    monkeypatch.setattr(controller, "_validate_study_approval", lambda *_args: None)
    monkeypatch.setattr(controller, "_validate_constraint_progress_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(controller, "_constraint_progress_contract_sha256", lambda *_args, **_kwargs: "d" * 64)
    monkeypatch.setattr(
        controller,
        "_run_worker",
        lambda *_args, **_kwargs: (
            {"status": "passed", "action": entry["success_action"]},
            {
                "stderr_bytes": 0,
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "stdout_bytes": 1,
            },
        ),
    )
    monkeypatch.setattr(controller, "_validate_study_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        controller.sys,
        "argv",
        ["run_local_lab.py", "--study", STUDY_ID, "--output", str(output)],
    )
    with pytest.raises(controller.QuarantinedStudyError, match="quarantined"):
        controller.main()
    assert (tmp_path / "lab-state.json").read_bytes() == state_before
    assert not output.exists()
    assert not (tmp_path / "lab.lock").exists()
