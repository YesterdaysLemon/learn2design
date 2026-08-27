"""Run one frozen, unpaid local mechanics study and retain private evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).parents[1].resolve()
PRIVATE_ROOT = ROOT.with_name(f"{ROOT.name}-local-lab").resolve()
STUDY_REGISTRY_PATH = ROOT / "experiments" / "local_lab" / "studies.json"
EXPECTED_STUDY_REGISTRY_SHA256 = (
    "88c52111c379c3178ff26f238131a0043ff58237c79d8d449dff0172c24d2039"
)
EXPECTED_SUBMISSION_SOURCE_SHA256 = (
    "34ba5a1403d22a8f9861851c2ddfb77a6ed57cc33554249f38bb9bf7b6bc1176"
)
EXPECTED_SUBMISSION_TREE_OID = "e2b495b6f1bf9f5c8f0b36ae5bc095f6df8e7588"
PROTECTED_LOCAL_ARTIFACTS = {
    "artifacts/generated/submission.manifest.json": (
        "99e18299ba6dd232f203cc2e59131cea61cd83ff8b0a9bceab6c56c764ddd86a"
    ),
    "artifacts/generated/submission.zip": (
        "4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b"
    ),
}
REQUIRED_SOURCE_KEYS = {
    "dependency_lock",
    "fixture_source",
    "lab_protocol",
    "study_plan",
    "worker_source",
}
WORKER_MODULE_PATHS = {
    "experiments.local_lab.full_surface_prefix_worker": (
        "experiments/local_lab/full_surface_prefix_worker.py"
    ),
    "experiments.local_lab.normal_path_jax_boundary_worker": (
        "experiments/local_lab/normal_path_jax_boundary_worker.py"
    ),
    "experiments.local_lab.feasible_progress_clock_worker": (
        "experiments/local_lab/feasible_progress_clock_worker.py"
    ),
    "experiments.local_lab.infeasible_prefix_indistinguishability_worker": (
        "experiments/local_lab/infeasible_prefix_indistinguishability_worker.py"
    ),
    "experiments.local_lab.public_signal_surface_worker": (
        "experiments/local_lab/public_signal_surface_worker.py"
    ),
    "experiments.local_lab.worker": "experiments/local_lab/worker.py",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ALLOWED_BRANCH = "codex/autonomous-local-lab"
ALLOWED_BRANCH_PREFIX = "codex/lab-"
CYCLE_TIMEOUT_SECONDS = 60 * 60
HEARTBEAT_SECONDS = 30
MAX_WORKER_OUTPUT_BYTES = 5 * 1024 * 1024
OUTPUT_POLL_SECONDS = 1
STATE_SCHEMA_VERSION = 1


class DuplicateStudyError(RuntimeError):
    """A terminal study was requested again; refuse without parking state."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _load_study_registry() -> dict[str, object]:
    encoded = STUDY_REGISTRY_PATH.read_bytes().replace(b"\r\n", b"\n")
    if _sha256_bytes(encoded) != EXPECTED_STUDY_REGISTRY_SHA256:
        raise RuntimeError("the approved local-lab study registry changed")
    registry = json.loads(encoded)
    if registry.get("schema_version") != 1:
        raise RuntimeError("unsupported local-lab study registry")
    studies = registry.get("studies")
    if not isinstance(studies, dict) or not studies:
        raise RuntimeError("the local-lab study registry is empty or malformed")
    return registry


def _protected_artifact_snapshot() -> dict[str, dict[str, object]]:
    snapshot = {}
    for relative_path, expected_digest in sorted(PROTECTED_LOCAL_ARTIFACTS.items()):
        path = ROOT / relative_path
        present = path.is_file()
        digest = _sha256(path) if present else None
        if present and digest != expected_digest:
            raise RuntimeError(f"protected local artifact changed: {relative_path}")
        snapshot[relative_path] = {
            "present": present,
            "sha256": digest,
        }
    return snapshot


def _source_paths(entry: dict[str, object]) -> dict[str, str]:
    source_paths = entry.get("source_paths")
    if not isinstance(source_paths, dict) or set(source_paths) != REQUIRED_SOURCE_KEYS:
        raise RuntimeError("study source paths do not match the frozen source set")
    validated = {}
    for name, relative_path in source_paths.items():
        if not isinstance(name, str) or not isinstance(relative_path, str):
            raise RuntimeError("malformed approved study source path")
        pure_path = PurePosixPath(relative_path)
        if (
            pure_path.is_absolute()
            or ".." in pure_path.parts
            or relative_path != pure_path.as_posix()
        ):
            raise RuntimeError(f"unsafe approved study source path: {name}")
        validated[name] = relative_path
    worker_module = entry.get("worker_module")
    if not isinstance(worker_module, str) or worker_module not in WORKER_MODULE_PATHS:
        raise RuntimeError("study worker module is not allowlisted")
    if validated["worker_source"] != WORKER_MODULE_PATHS[worker_module]:
        raise RuntimeError("study worker module disagrees with its frozen source")
    return validated


def _repository_snapshot(entry: dict[str, object]) -> dict[str, object]:
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("the local lab refuses a dirty worktree")
    branch = _git("branch", "--show-current")
    if branch != ALLOWED_BRANCH and not branch.startswith(ALLOWED_BRANCH_PREFIX):
        raise RuntimeError(f"the local lab refuses unapproved branch: {branch}")

    revision = _git("rev-parse", "HEAD")
    submission_tree_oid = _git("rev-parse", "HEAD:submission")
    if submission_tree_oid != EXPECTED_SUBMISSION_TREE_OID:
        raise RuntimeError("the protected submission tree changed")
    committed_source = _git_bytes("show", "HEAD:submission/submission.py")
    if _sha256_bytes(committed_source) != EXPECTED_SUBMISSION_SOURCE_SHA256:
        raise RuntimeError("the protected submission source digest changed")

    source_paths = _source_paths(entry)
    committed_hashes = {
        name: _sha256_bytes(_git_bytes("show", f"HEAD:{relative_path}"))
        for name, relative_path in sorted(source_paths.items())
    }
    return {
        "branch": branch,
        "committed_file_sha256": committed_hashes,
        "committed_source_paths": source_paths,
        "protected_local_artifacts": _protected_artifact_snapshot(),
        "revision": revision,
        "submission_source_sha256": EXPECTED_SUBMISSION_SOURCE_SHA256,
        "submission_tree_oid": submission_tree_oid,
        "working_tree_dirty": False,
    }


def _study_entry(
    registry: dict[str, object], study: str
) -> dict[str, object]:
    studies = registry["studies"]
    if not isinstance(studies, dict) or study not in studies:
        raise RuntimeError(f"study is not present in the approved registry: {study}")
    entry = studies[study]
    if not isinstance(entry, dict):
        raise RuntimeError(f"malformed approved study entry: {study}")
    return entry


def _validate_study_approval(
    entry: dict[str, object], snapshot: dict[str, object]
) -> None:
    approved = entry.get("approved_file_sha256")
    committed = snapshot.get("committed_file_sha256")
    committed_paths = snapshot.get("committed_source_paths")
    if (
        not isinstance(approved, dict)
        or not isinstance(committed, dict)
        or not isinstance(committed_paths, dict)
    ):
        raise RuntimeError("malformed approved study source manifest")
    if set(approved) != set(committed) or set(approved) != set(committed_paths):
        raise RuntimeError("approved study source manifest is incomplete")
    for name, expected_digest in approved.items():
        if (
            not isinstance(expected_digest, str)
            or SHA256_PATTERN.fullmatch(expected_digest) is None
        ):
            raise RuntimeError(f"malformed approved study source digest: {name}")
        if committed.get(name) != expected_digest:
            raise RuntimeError(f"approved study source changed: {name}")


def _validate_study_result(
    study: str,
    entry: dict[str, object],
    result: dict[str, object],
) -> None:
    expected_top_level = {
        "action",
        "cases",
        "environment",
        "fixture",
        "schema_version",
        "status",
        "study_id",
    }
    if set(result) != expected_top_level:
        raise RuntimeError("worker returned an unexpected top-level result field")
    _validate_sanitized_value(result)
    if result.get("study_id") != study:
        raise RuntimeError("worker returned the wrong study identity")
    if result.get("schema_version") != entry.get("result_schema_version"):
        raise RuntimeError("worker returned the wrong result schema")

    required_case_fields = entry.get("case_required_fields")
    cases = result.get("cases")
    if not isinstance(required_case_fields, dict) or not isinstance(cases, dict):
        raise RuntimeError("worker returned a malformed case collection")
    if set(cases) != set(required_case_fields):
        raise RuntimeError("worker returned the wrong frozen case set")

    case_passes = []
    for case_name, required_fields in required_case_fields.items():
        case = cases.get(case_name)
        if not isinstance(case, dict) or not isinstance(required_fields, list):
            raise RuntimeError(f"worker returned malformed case: {case_name}")
        if set(case) != set(required_fields):
            raise RuntimeError(
                f"worker returned the wrong frozen fields for case: {case_name}"
            )
        if not isinstance(case.get("passed"), bool):
            raise RuntimeError(f"worker returned non-terminal case: {case_name}")
        case_passes.append(case["passed"])

    passed = all(case_passes)
    expected_status = "passed" if passed else "failed"
    expected_action = (
        entry.get("success_action") if passed else entry.get("failure_action")
    )
    if result.get("status") != expected_status:
        raise RuntimeError("worker status disagrees with the frozen case results")
    if result.get("action") != expected_action:
        raise RuntimeError("worker action disagrees with the frozen decision rule")

    fixture = result.get("fixture")
    expected_fixture = entry.get("fixture_identity")
    if not isinstance(fixture, dict) or not isinstance(expected_fixture, dict):
        raise RuntimeError("worker returned malformed fixture identity")
    if set(fixture) != set(expected_fixture) | {"case_contract"}:
        raise RuntimeError("worker returned unexpected fixture fields")
    if any(fixture.get(name) != value for name, value in expected_fixture.items()):
        raise RuntimeError("worker returned the wrong frozen fixture identity")
    case_contract = fixture.get("case_contract")
    expected_case_contract = entry.get("case_contract")
    if (
        not isinstance(case_contract, dict)
        or not isinstance(expected_case_contract, dict)
        or case_contract != expected_case_contract
        or set(case_contract) != set(cases)
    ):
        raise RuntimeError("worker returned the wrong frozen case contract")

    environment = result.get("environment")
    if (
        not isinstance(environment, dict)
        or set(environment) != {"device_kind", "jax_version", "platform", "python"}
        or environment.get("platform") != "cpu"
    ):
        raise RuntimeError("worker did not authenticate the CPU backend")


def _validate_sanitized_value(value: object, *, depth: int = 0) -> None:
    if depth > 8:
        raise RuntimeError("worker result exceeded the sanitized nesting limit")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("worker result contained a non-finite number")
        return
    if isinstance(value, str):
        if len(value) > 512:
            raise RuntimeError("worker result contained an oversized string")
        if str(ROOT).lower() in value.lower():
            raise RuntimeError("worker result exposed a repository path")
        return
    if isinstance(value, list):
        if len(value) > 64:
            raise RuntimeError("worker result contained an oversized list")
        for item in value:
            _validate_sanitized_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 64:
            raise RuntimeError("worker result contained an oversized object")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise RuntimeError("worker result contained a malformed key")
            lowered = key.lower()
            if any(
                fragment in lowered
                for fragment in (
                    "credential",
                    "history",
                    "parameter_values",
                    "raw_gradient",
                    "secret",
                    "topology",
                )
            ):
                raise RuntimeError("worker result contained a forbidden field")
            if lowered.endswith("sha256") and (
                not isinstance(item, str)
                or SHA256_PATTERN.fullmatch(item) is None
            ):
                raise RuntimeError("worker result contained a malformed SHA-256")
            _validate_sanitized_value(item, depth=depth + 1)
        return
    raise RuntimeError("worker result contained a non-JSON value")


def _validate_output(output: Path) -> tuple[Path, Path]:
    resolved = output.resolve()
    if resolved == PRIVATE_ROOT or not resolved.is_relative_to(PRIVATE_ROOT):
        raise RuntimeError(
            f"local-lab output must be a file beneath the private root: {PRIVATE_ROOT}"
        )
    sidecar = resolved.with_name(f"{resolved.name}.sha256")
    for path in (resolved, sidecar):
        if path.exists():
            raise RuntimeError(f"local-lab evidence already exists: {path.name}")
    return resolved, sidecar


def _write_atomic_bytes(path: Path, encoded: bytes) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite local-lab output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-volume hard link is an atomic create-if-absent operation. Unlike
        # os.replace(), it cannot overwrite an output created by another process.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_bytes(encoded)


def _write_atomic(path: Path, payload: dict[str, object]) -> str:
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return _write_atomic_bytes(path, encoded)


def _write_mutable_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_event(private_root: Path, event: dict[str, object]) -> None:
    private_root.mkdir(parents=True, exist_ok=True)
    payload = {"event_schema_version": 1, **event}
    encoded = (
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with (private_root / "lab-events.jsonl").open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _acquire_lease(
    private_root: Path,
    *,
    cycle_id: str,
    revision: str,
    study: str,
) -> tuple[Path, str]:
    private_root.mkdir(parents=True, exist_ok=True)
    lock_directory = private_root / "lab.lock"
    try:
        lock_directory.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            "a local-lab lease already exists; inspect it manually and do not "
            "auto-recover it"
        ) from error

    lease_id = uuid.uuid4().hex
    now = _utc_now()
    lease = {
        "cycle_id": cycle_id,
        "heartbeat_utc": now,
        "hostname": socket.gethostname(),
        "lease_id": lease_id,
        "phase": "preflight",
        "pid": os.getpid(),
        "process_started_utc": now,
        "project_revision": revision,
        "schema_version": 1,
        "study": study,
    }
    try:
        _write_mutable_json(lock_directory / "lease.json", lease)
    except BaseException:
        if not any(lock_directory.iterdir()):
            lock_directory.rmdir()
        raise
    return lock_directory, lease_id


def _read_lease(lock_directory: Path, lease_id: str) -> dict[str, object]:
    lease_path = lock_directory / "lease.json"
    if not lease_path.is_file():
        raise RuntimeError("local-lab lease metadata disappeared")
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    if lease.get("lease_id") != lease_id:
        raise RuntimeError("local-lab lease identity changed during execution")
    return lease


def _heartbeat_lease(lock_directory: Path, lease_id: str, *, phase: str) -> None:
    lease = _read_lease(lock_directory, lease_id)
    lease["heartbeat_utc"] = _utc_now()
    lease["phase"] = phase
    _write_mutable_json(lock_directory / "lease.json", lease)


def _release_lease(lock_directory: Path, lease_id: str) -> None:
    _read_lease(lock_directory, lease_id)
    (lock_directory / "lease.json").unlink()
    lock_directory.rmdir()


def _default_state() -> dict[str, object]:
    return {
        "active_cycle": None,
        "completed_studies": {},
        "failure_streak": 0,
        "schema_version": STATE_SCHEMA_VERSION,
        "status": "idle",
        "stop_reason": None,
        "updated_utc": _utc_now(),
    }


def _load_state(private_root: Path) -> dict[str, object]:
    state_path = private_root / "lab-state.json"
    if not state_path.exists():
        return _default_state()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise RuntimeError("unsupported or malformed local-lab state")
    if not isinstance(state.get("completed_studies"), dict):
        raise RuntimeError("malformed local-lab completed-study ledger")
    return state


def _save_state(private_root: Path, state: dict[str, object]) -> None:
    state["updated_utc"] = _utc_now()
    _write_mutable_json(private_root / "lab-state.json", state)


def _worker_environment() -> dict[str, str]:
    safe_names = {
        "COMSPEC",
        "LD_LIBRARY_PATH",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "VIRTUAL_ENV",
        "WINDIR",
    }
    environment = {
        name: value for name, value in os.environ.items() if name.upper() in safe_names
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "LEARN2DESIGN_LOCAL_LAB_NETWORK": "disabled",
            "PYTHONHASHSEED": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    return environment


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        os.killpg(process.pid, signal.SIGKILL)
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


def _run_worker(
    worker_mode: str,
    *,
    cycle_id: str,
    heartbeat,
    worker_module: str = "experiments.local_lab.worker",
) -> tuple[dict[str, object], dict[str, object]]:
    if worker_module not in WORKER_MODULE_PATHS:
        raise RuntimeError("local-lab worker module is not allowlisted")
    command = [
        sys.executable,
        "-m",
        worker_module,
        "--mode",
        worker_mode,
    ]
    creationflags = (
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    )
    temporary_root = PRIVATE_ROOT / "worker-tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    stdout_path = temporary_root / f"{cycle_id}.stdout"
    stderr_path = temporary_root / f"{cycle_id}.stderr"
    process = None
    try:
        with stdout_path.open("xb") as stdout_handle, stderr_path.open(
            "xb"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=_worker_environment(),
                stdout=stdout_handle,
                stderr=stderr_handle,
                creationflags=creationflags,
                start_new_session=os.name != "nt",
            )
            started = time.monotonic()
            next_heartbeat = started + HEARTBEAT_SECONDS
            heartbeat(process.pid, 0.0)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                remaining = CYCLE_TIMEOUT_SECONDS - elapsed
                if remaining <= 0:
                    _terminate_process_tree(process)
                    raise TimeoutError(
                        f"local-lab worker exceeded {CYCLE_TIMEOUT_SECONDS} seconds"
                    )
                if stdout_path.stat().st_size + stderr_path.stat().st_size > (
                    MAX_WORKER_OUTPUT_BYTES
                ):
                    _terminate_process_tree(process)
                    raise RuntimeError(
                        "local-lab worker output exceeded the retention cap"
                    )
                if time.monotonic() >= next_heartbeat:
                    heartbeat(process.pid, elapsed)
                    next_heartbeat = time.monotonic() + HEARTBEAT_SECONDS
                try:
                    process.wait(timeout=min(float(OUTPUT_POLL_SECONDS), remaining))
                except subprocess.TimeoutExpired:
                    pass

        stdout_bytes = stdout_path.read_bytes()
        stderr_bytes = stderr_path.read_bytes()
        if len(stdout_bytes) + len(stderr_bytes) > MAX_WORKER_OUTPUT_BYTES:
            raise RuntimeError("local-lab worker output exceeded the retention cap")
        if process.returncode != 0:
            tail = stderr_bytes[-2000:].decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"local-lab worker exited {process.returncode}: "
                f"{tail or 'no stderr'}"
            )
        result = json.loads(stdout_bytes.decode("utf-8"))
        if not isinstance(result, dict):
            raise RuntimeError("local-lab worker returned a non-object result")
        return result, {
            "stderr_bytes": len(stderr_bytes),
            "stderr_sha256": _sha256_bytes(stderr_bytes),
            "stdout_bytes": len(stdout_bytes),
        }
    finally:
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)
        for temporary in (stdout_path, stderr_path):
            if temporary.exists():
                temporary.unlink()


def _begin_cycle(
    private_root: Path,
    *,
    cycle_id: str,
    output: Path,
    snapshot: dict[str, object],
    study: str,
) -> dict[str, object]:
    if (private_root / "stop.request.json").exists():
        raise RuntimeError(
            "owner stop request is present; the local lab remains stopped"
        )
    state = _load_state(private_root)
    if state.get("status") not in {"idle", "awaiting_study"}:
        raise RuntimeError(f"local-lab state is not idle: {state.get('status')}")
    completed = state["completed_studies"]
    assert isinstance(completed, dict)
    if study in completed:
        raise DuplicateStudyError(
            f"the frozen study already has a terminal record: {study}"
        )
    state["active_cycle"] = {
        "cycle_id": cycle_id,
        "output": output.relative_to(private_root).as_posix(),
        "revision": snapshot["revision"],
        "started_utc": _utc_now(),
        "study": study,
    }
    state["status"] = "active"
    state["stop_reason"] = None
    _save_state(private_root, state)
    _append_event(
        private_root,
        {
            "cycle_id": cycle_id,
            "event": "cycle_started",
            "revision": snapshot["revision"],
            "study": study,
            "utc": _utc_now(),
        },
    )
    return state


def _park_cycle(
    private_root: Path,
    state: dict[str, object],
    *,
    cycle_id: str,
    error: BaseException,
    study: str,
) -> None:
    state["active_cycle"] = None
    state["failure_streak"] = int(state.get("failure_streak", 0)) + 1
    state["status"] = "parked"
    state["stop_reason"] = f"{type(error).__name__}: {str(error)[:500]}"
    _save_state(private_root, state)
    _append_event(
        private_root,
        {
            "cycle_id": cycle_id,
            "error_type": type(error).__name__,
            "event": "cycle_parked",
            "study": study,
            "utc": _utc_now(),
        },
    )


def _park_preflight(
    private_root: Path,
    *,
    cycle_id: str,
    error: BaseException,
    study: str,
) -> None:
    state = _load_state(private_root)
    if state.get("status") not in {"idle", "awaiting_study"}:
        return
    owner_stopped = (private_root / "stop.request.json").exists()
    state["active_cycle"] = None
    state["failure_streak"] = (
        int(state.get("failure_streak", 0))
        if owner_stopped
        else int(state.get("failure_streak", 0)) + 1
    )
    state["status"] = "stopped" if owner_stopped else "parked"
    state["stop_reason"] = f"{type(error).__name__}: {str(error)[:500]}"
    _save_state(private_root, state)
    _append_event(
        private_root,
        {
            "cycle_id": cycle_id,
            "error_type": type(error).__name__,
            "event": "preflight_stopped" if owner_stopped else "preflight_parked",
            "study": study,
            "utc": _utc_now(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output, sidecar = _validate_output(args.output)
    cycle_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
    prelease_revision = _git("rev-parse", "HEAD")
    lock_directory, lease_id = _acquire_lease(
        PRIVATE_ROOT,
        cycle_id=cycle_id,
        revision=prelease_revision,
        study=args.study,
    )
    state = None
    exit_code = 0
    try:
        registry = _load_study_registry()
        entry = _study_entry(registry, args.study)
        snapshot = _repository_snapshot(entry)
        if snapshot["revision"] != prelease_revision:
            raise RuntimeError("repository revision changed while acquiring the lease")
        _validate_study_approval(entry, snapshot)
        _heartbeat_lease(lock_directory, lease_id, phase="preflight-complete")
        state = _begin_cycle(
            PRIVATE_ROOT,
            cycle_id=cycle_id,
            output=output,
            snapshot=snapshot,
            study=args.study,
        )

        def heartbeat(worker_pid: int, elapsed_seconds: float) -> None:
            _heartbeat_lease(lock_directory, lease_id, phase="worker-running")
            _append_event(
                PRIVATE_ROOT,
                {
                    "cycle_id": cycle_id,
                    "elapsed_seconds": round(float(elapsed_seconds), 3),
                    "event": "heartbeat",
                    "study": args.study,
                    "utc": _utc_now(),
                    "worker_pid": worker_pid,
                },
            )

        worker_mode = entry.get("worker_mode")
        if not isinstance(worker_mode, str):
            raise RuntimeError("approved study has no worker mode")
        worker_module = entry.get("worker_module")
        if not isinstance(worker_module, str):
            raise RuntimeError("approved study has no worker module")
        result, worker_receipt = _run_worker(
            worker_mode,
            cycle_id=cycle_id,
            heartbeat=heartbeat,
            worker_module=worker_module,
        )
        _validate_study_result(args.study, entry, result)
        post_snapshot = _repository_snapshot(entry)
        if post_snapshot != snapshot:
            raise RuntimeError(
                "repository or protected-artifact drift during the cycle"
            )

        payload = {
            "artifact_format_version": 2,
            "cycle_id": cycle_id,
            "execution_contract": {
                "cycle_timeout_seconds": CYCLE_TIMEOUT_SECONDS,
                "device": "cpu",
                "network": "disabled-in-worker",
                "worker_output_cap_bytes": MAX_WORKER_OUTPUT_BYTES,
            },
            "provenance": snapshot,
            "result": result,
            "worker_receipt": worker_receipt,
        }
        digest = _write_atomic(output, payload)
        _write_atomic_bytes(
            sidecar,
            f"{digest}  {output.name}\n".encode("ascii"),
        )

        completed = state["completed_studies"]
        assert isinstance(completed, dict)
        completed[args.study] = {
            "cycle_id": cycle_id,
            "result_sha256": digest,
            "revision": snapshot["revision"],
            "status": result["status"],
        }
        state["active_cycle"] = None
        state["failure_streak"] = 0 if result["status"] == "passed" else 1
        if result["status"] == "passed":
            studies = registry["studies"]
            assert isinstance(studies, dict)
            pending = set(studies) - set(completed)
            state["status"] = "idle" if pending else "awaiting_study"
            state["stop_reason"] = (
                None if pending else "no_approved_study_pending"
            )
        else:
            state["status"] = "parked"
            state["stop_reason"] = result.get("action")
        _save_state(PRIVATE_ROOT, state)
        _append_event(
            PRIVATE_ROOT,
            {
                "cycle_id": cycle_id,
                "event": "cycle_completed",
                "result_sha256": digest,
                "status": result["status"],
                "study": args.study,
                "utc": _utc_now(),
            },
        )
        _heartbeat_lease(lock_directory, lease_id, phase="terminal-recorded")
        print(f"wrote {output} ({digest})")
        if result["status"] != "passed":
            exit_code = 1
    except BaseException as error:
        if state is not None and state.get("status") == "active":
            _park_cycle(
                PRIVATE_ROOT,
                state,
                cycle_id=cycle_id,
                error=error,
                study=args.study,
            )
        elif state is None and not isinstance(error, DuplicateStudyError):
            _park_preflight(
                PRIVATE_ROOT,
                cycle_id=cycle_id,
                error=error,
                study=args.study,
            )
        raise
    finally:
        _release_lease(lock_directory, lease_id)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
