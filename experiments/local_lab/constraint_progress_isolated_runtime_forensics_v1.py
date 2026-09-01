"""Single-shot, non-scientific isolated-runtime forensics probe."""

from __future__ import annotations

import os
import sys


CHECKPOINT_ID = "constraint-progress-isolated-runtime-forensics-v1"
PLAN_REVISION = "b6efe5cfaca849fdab4531fb4dcdea04823f0a2a"
FAILED_ATTEMPT_REVISION = "9413cd4982cab74887fa8c7dc3dd4bf9c4d8508a"
FAILED_PLAN_REVISION = "c5314afaa50490e39c53669d971114d280e43c07"
FAILED_CONTRACT_SHA256 = (
    "621ade24962abd16ea4c3902691ae1781067572618c0639785fcadbfcb5b585f"
)

EXPECTED_RUNTIME_IDENTITY = {
    "machine": "AMD64",
    "numpy_init_sha256": (
        "a6958cb364663b7acce81ccfd58eeb65a2b34d5376157f924777b97211a73be4"
    ),
    "numpy_metadata_sha256": (
        "6ae45122ee97050e48849438320430d05f01814f72e66e69cbeed027d2c6a1e8"
    ),
    "numpy_version": "2.5.1",
    "pcg64_identity": "numpy.random._pcg64.PCG64",
    "pcg64_module_sha256": (
        "210bd962e911039f1639d0137f6e41444e37db23aba1622635d9dba8abc6a1c9"
    ),
    "python_architecture": "64bit",
    "python_executable_sha256": (
        "ad169f4cb4bfb78c7a5c030a4529c19d6643276778e33994c93e145b6191c3ec"
    ),
    "python_implementation": "CPython",
    "python_version": "3.13.14",
    "seed_sequence_identity": "numpy.random.bit_generator.SeedSequence",
    "seed_sequence_module_sha256": (
        "08355a330efec79a840b5767bb5356ad21e3b0f14acce9a3c969208626daad7f"
    ),
}

STAGES = (
    "argv_bootstrap",
    "contract_environment",
    "late_stdlib_imports",
    "network_denial",
    "site_discovery",
    "numpy_import",
    "runtime_identity",
    "canonical_output",
    "composite",
)
CASE_IDS = STAGES
SAFE_ENVIRONMENT = {
    "COMSPEC",
    "LD_LIBRARY_PATH",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "VIRTUAL_ENV",
    "WINDIR",
}
ENVIRONMENT_OVERRIDES = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_PLATFORMS": "cpu",
    "L2D_CONTRACT_SHA256": FAILED_CONTRACT_SHA256,
    "L2D_PLAN_REVISION": FAILED_PLAN_REVISION,
    "L2D_STUDY_REVISION": FAILED_ATTEMPT_REVISION,
    "LEARN2DESIGN_LOCAL_LAB_NETWORK": "disabled",
    "PYTHONHASHSEED": "0",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
ALLOWED_ENVIRONMENT = SAFE_ENVIRONMENT | set(ENVIRONMENT_OVERRIDES)
NETWORK_PROBES = 6
TIMEOUT_SECONDS = 60.0
MAX_OUTPUT_BYTES = 16_384
RESULT_DIRECTORY_PREFIX = "l2d-constraint-runtime-forensics-v1-"
PLAN_PATH_POSIX = (
    "research/2026-09-01-constraint-progress-isolated-runtime-forensics-v1-plan.md"
)
SOURCE_PATH_POSIX = (
    "experiments/local_lab/constraint_progress_isolated_runtime_forensics_v1.py"
)

CHILD_FIELDS = (
    "schema_version",
    "checkpoint_id",
    "case_id",
    "target_stage",
    "reached_stage",
    "status",
    "error_code",
    "environment_keys_sha256",
    "site_commitment_sha256",
    "identity_sha256",
    "identity_matches",
    "network_attempts_rejected",
)
OBSERVATION_FIELDS = (
    "case_id",
    "stdout_bytes",
    "stderr_bytes",
    "return_code",
    "child_receipt",
    "surviving_processes",
    "error_code",
)
RUN_FIELDS = (
    "cases",
    "first_failure",
    "child_launches",
    "stderr_bytes",
    "surviving_processes",
    "prefix_valid",
    "passed",
    "body_sha256",
)
RESULT_FIELDS = (
    "schema_version",
    "checkpoint_id",
    "plan_revision",
    "probe_revision",
    "plan_sha256",
    "probe_source_sha256",
    "failed_attempt_revision",
    "failed_contract_sha256",
    "expected_runtime_identity",
    "runs",
    "runs_equal",
    "identified_stage",
    "diagnostic_status",
    "action",
    "receipt_root_sha256",
)
PARENT_ERROR_CODES = {
    "spawn",
    "timeout",
    "output_cap",
    "stderr",
    "exit",
    "schema",
    "relation",
    "cleanup",
}


class StageFailure(RuntimeError):
    def __init__(self, stage: str):
        self.stage = stage if stage in STAGES else "composite"
        super().__init__(self.stage)


class ProbeFailure(RuntimeError):
    pass


def _discover_mode() -> tuple[str, str | None]:
    arguments = tuple(sys.argv[1:])
    if arguments == ("--run",):
        return "run", None
    if len(arguments) == 2 and arguments[0] == "--child" and arguments[1] in CASE_IDS:
        return "child", arguments[1]
    return "invalid", None


def _late_imports() -> None:
    global hashlib, importlib, json, platform, shutil, socket, subprocess
    global tempfile, time, Path
    import hashlib
    import importlib
    import json
    import platform
    import shutil
    import socket
    import subprocess
    import tempfile
    import time
    from pathlib import Path


def _reject_duplicate_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ProbeFailure("duplicate-json-key")
        value[key] = item
    return value


def _reject_constant(_value):
    raise ProbeFailure("nonfinite-json-value")


def _loads(value: bytes | str):
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except ProbeFailure:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeFailure("malformed-json") from error


def _canonical_line(value: dict[str, object]) -> bytes:
    try:
        encoded = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ProbeFailure("noncanonical-json") from error
    if not (0 < len(encoded) <= MAX_OUTPUT_BYTES):
        raise ProbeFailure("output-cap")
    return encoded


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_domain(domain: bytes, value: bytes) -> str:
    return _sha256_bytes(domain + b"\0" + value)


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hex(value: object, length: int = 64) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _probe_environment(source=None) -> dict[str, str]:
    source = os.environ if source is None else source
    environment = {
        name.upper(): value
        for name, value in source.items()
        if name.upper() in SAFE_ENVIRONMENT
    }
    environment.update(ENVIRONMENT_OVERRIDES)
    return environment


def _environment_keys_sha256(environment=None) -> str:
    environment = os.environ if environment is None else environment
    encoded = ("\n".join(sorted(name.upper() for name in environment)) + "\n").encode(
        "ascii"
    )
    return _sha256_domain(b"L2D-runtime-forensics-v1/environment-keys", encoded)


def _check_environment() -> None:
    names = {name.upper() for name in os.environ}
    if not names.issubset(ALLOWED_ENVIRONMENT):
        raise StageFailure("contract_environment")
    for name, expected in ENVIRONMENT_OVERRIDES.items():
        if os.environ.get(name) != expected:
            raise StageFailure("contract_environment")
    l2d_names = {name for name in names if name.startswith("L2D_")}
    if l2d_names != {
        "L2D_CONTRACT_SHA256",
        "L2D_PLAN_REVISION",
        "L2D_STUDY_REVISION",
    }:
        raise StageFailure("contract_environment")


def _deny_network(*_args, **_kwargs):
    raise StageFailure("network_denial")


def _disable_and_prove_network() -> int:
    socket.socket = _deny_network
    socket.create_connection = _deny_network
    socket.getaddrinfo = _deny_network
    socket.gethostbyaddr = _deny_network
    socket.gethostbyname = _deny_network
    socket.gethostbyname_ex = _deny_network
    probes = (
        (socket.socket, ()),
        (socket.create_connection, (("127.0.0.1", 9),)),
        (socket.getaddrinfo, ("localhost", 9)),
        (socket.gethostbyaddr, ("127.0.0.1",)),
        (socket.gethostbyname, ("localhost",)),
        (socket.gethostbyname_ex, ("localhost",)),
    )
    rejected = 0
    for probe, arguments in probes:
        try:
            probe(*arguments)
        except StageFailure as error:
            if error.stage == "network_denial":
                rejected += 1
                continue
        except BaseException:
            pass
        raise StageFailure("network_denial")
    if rejected != NETWORK_PROBES:
        raise StageFailure("network_denial")
    return rejected


def _site_discovery() -> str:
    site = importlib.import_module("site")
    root = Path(__file__).parents[2].resolve()
    user_value = site.getusersitepackages()
    if type(user_value) is not str:
        raise StageFailure("site_discovery")
    user_packages = Path(user_value).resolve()
    root_text = str(root)
    user_text = str(user_packages)
    for entry in (root_text, user_text):
        if entry not in sys.path:
            sys.path.append(entry)
    projection = {
        "root_exists": root.is_dir(),
        "root_present": root_text in sys.path,
        "user_exists": user_packages.is_dir(),
        "user_present": user_text in sys.path,
    }
    return _sha256_domain(
        b"L2D-runtime-forensics-v1/site",
        _canonical_line(projection),
    )


def _numpy_imports():
    np = importlib.import_module("numpy")
    pcg64_module = importlib.import_module("numpy.random._pcg64")
    bit_generator_module = importlib.import_module("numpy.random.bit_generator")
    return np, pcg64_module, bit_generator_module


def _process_executable_path():
    if os.name != "nt":
        return Path(sys.executable).resolve()
    ctypes = importlib.import_module("ctypes")
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetModuleFileNameW(None, buffer, len(buffer))
    if length <= 0 or length >= len(buffer):
        raise StageFailure("runtime_identity")
    return Path(buffer.value)


def _runtime_identity(np, pcg64_module, bit_generator_module) -> dict[str, str]:
    metadata = importlib.import_module("importlib.metadata")
    numpy_init = Path(np.__file__).resolve()
    metadata_root = metadata.distribution("numpy")._path
    metadata_path = Path(metadata_root) / "METADATA"
    return {
        "machine": platform.machine(),
        "numpy_init_sha256": _sha256_file(numpy_init),
        "numpy_metadata_sha256": _sha256_file(metadata_path),
        "numpy_version": np.__version__,
        "pcg64_identity": f"{np.random.PCG64.__module__}.{np.random.PCG64.__name__}",
        "pcg64_module_sha256": _sha256_file(Path(pcg64_module.__file__).resolve()),
        "python_architecture": platform.architecture()[0],
        "python_executable_sha256": _sha256_file(_process_executable_path()),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "seed_sequence_identity": (
            f"{np.random.SeedSequence.__module__}.{np.random.SeedSequence.__name__}"
        ),
        "seed_sequence_module_sha256": _sha256_file(
            Path(bit_generator_module.__file__).resolve()
        ),
    }


def _child_receipt(case_id: str) -> dict[str, object]:
    target_index = STAGES.index(case_id)
    reached_stage = None
    error_code = None
    site_commitment = None
    identity_sha256 = None
    identity_matches = None
    network_attempts = 0
    identity = None
    current_stage = "argv_bootstrap"
    try:
        if (
            tuple(sys.argv[1:]) != ("--child", case_id)
            or not sys.flags.safe_path
            or not sys.flags.no_site
        ):
            raise StageFailure("argv_bootstrap")
        reached_stage = "argv_bootstrap"
        if target_index == 0:
            raise StopIteration

        current_stage = "contract_environment"
        _check_environment()
        reached_stage = "contract_environment"
        if target_index == 1:
            raise StopIteration

        current_stage = "late_stdlib_imports"
        _late_imports()
        reached_stage = "late_stdlib_imports"
        if target_index == 2:
            raise StopIteration

        current_stage = "network_denial"
        network_attempts = _disable_and_prove_network()
        reached_stage = "network_denial"
        if target_index == 3:
            raise StopIteration

        current_stage = "site_discovery"
        site_commitment = _site_discovery()
        reached_stage = "site_discovery"
        if target_index == 4:
            raise StopIteration

        current_stage = "numpy_import"
        np, pcg64_module, bit_generator_module = _numpy_imports()
        reached_stage = "numpy_import"
        if target_index == 5:
            raise StopIteration

        current_stage = "runtime_identity"
        identity = _runtime_identity(np, pcg64_module, bit_generator_module)
        identity_sha256 = _sha256_domain(
            b"L2D-runtime-forensics-v1/runtime-identity",
            _canonical_line(identity),
        )
        identity_matches = identity == EXPECTED_RUNTIME_IDENTITY
        if not identity_matches:
            raise StageFailure("runtime_identity")
        reached_stage = "runtime_identity"
        if target_index == 6:
            raise StopIteration

        current_stage = "canonical_output"
        encoded_identity = _canonical_line(identity)
        if len(encoded_identity) > MAX_OUTPUT_BYTES:
            raise StageFailure("canonical_output")
        reached_stage = "canonical_output"
        if target_index == 7:
            raise StopIteration

        current_stage = "composite"
        repeated = _runtime_identity(np, pcg64_module, bit_generator_module)
        if repeated != identity or _canonical_line(repeated) != encoded_identity:
            raise StageFailure("composite")
        reached_stage = "composite"
    except StopIteration:
        pass
    except StageFailure as error:
        error_code = error.stage
    except BaseException:
        error_code = current_stage

    try:
        _late_imports()
        receipt = {
            "schema_version": 1,
            "checkpoint_id": CHECKPOINT_ID,
            "case_id": case_id,
            "target_stage": case_id,
            "reached_stage": reached_stage,
            "status": "failed" if error_code is not None else "passed",
            "error_code": error_code,
            "environment_keys_sha256": _environment_keys_sha256(),
            "site_commitment_sha256": site_commitment,
            "identity_sha256": identity_sha256,
            "identity_matches": identity_matches,
            "network_attempts_rejected": network_attempts,
        }
        return receipt
    except BaseException:
        raise SystemExit(71) from None


def _validate_child_receipt(value: object, case_id: str) -> dict[str, object]:
    if type(value) is not dict or tuple(value) != CHILD_FIELDS:
        raise ProbeFailure("child-schema")
    if (
        value["schema_version"] != 1
        or value["checkpoint_id"] != CHECKPOINT_ID
        or value["case_id"] != case_id
        or value["target_stage"] != case_id
        or value["status"] not in {"passed", "failed"}
        or not _is_hex(value["environment_keys_sha256"])
        or value["environment_keys_sha256"]
        != _environment_keys_sha256(_probe_environment())
        or type(value["network_attempts_rejected"]) is not int
        or value["network_attempts_rejected"] not in range(NETWORK_PROBES + 1)
    ):
        raise ProbeFailure("child-schema")
    reached = value["reached_stage"]
    error = value["error_code"]
    if reached is not None and reached not in STAGES:
        raise ProbeFailure("child-schema")
    target_index = STAGES.index(case_id)
    if value["status"] == "passed":
        if error is not None or reached != case_id:
            raise ProbeFailure("child-relation")
    else:
        if error not in STAGES or STAGES.index(error) > target_index:
            raise ProbeFailure("child-relation")
        expected_reached = (
            None if STAGES.index(error) == 0 else STAGES[STAGES.index(error) - 1]
        )
        if reached != expected_reached:
            raise ProbeFailure("child-relation")
    site_commitment = value["site_commitment_sha256"]
    if site_commitment is not None and not _is_hex(site_commitment):
        raise ProbeFailure("child-schema")
    identity_sha256 = value["identity_sha256"]
    if identity_sha256 is not None and not _is_hex(identity_sha256):
        raise ProbeFailure("child-schema")
    if value["identity_matches"] is not None and type(value["identity_matches"]) is not bool:
        raise ProbeFailure("child-schema")
    reached_index = -1 if reached is None else STAGES.index(reached)
    if reached_index >= STAGES.index("network_denial"):
        if value["network_attempts_rejected"] != NETWORK_PROBES:
            raise ProbeFailure("child-relation")
    elif error != "network_denial" and value["network_attempts_rejected"] != 0:
        raise ProbeFailure("child-relation")
    if reached_index >= STAGES.index("site_discovery"):
        if site_commitment is None or site_commitment != _expected_site_commitment():
            raise ProbeFailure("child-relation")
    elif site_commitment is not None:
        raise ProbeFailure("child-relation")
    if reached_index >= STAGES.index("runtime_identity"):
        if (
            identity_sha256 != _expected_identity_commitment()
            or value["identity_matches"] is not True
        ):
            raise ProbeFailure("child-relation")
    elif error == "runtime_identity":
        if identity_sha256 is None or type(value["identity_matches"]) is not bool:
            raise ProbeFailure("child-relation")
    elif identity_sha256 is not None or value["identity_matches"] is not None:
        raise ProbeFailure("child-relation")
    return value


def _expected_site_commitment() -> str:
    site = importlib.import_module("site")
    root = Path(__file__).parents[2].resolve()
    user_packages = Path(site.getusersitepackages()).resolve()
    projection = {
        "root_exists": root.is_dir(),
        "root_present": True,
        "user_exists": user_packages.is_dir(),
        "user_present": True,
    }
    return _sha256_domain(
        b"L2D-runtime-forensics-v1/site",
        _canonical_line(projection),
    )


def _expected_identity_commitment() -> str:
    return _sha256_domain(
        b"L2D-runtime-forensics-v1/runtime-identity",
        _canonical_line(EXPECTED_RUNTIME_IDENTITY),
    )


def _validate_observation(value: object, case_id: str) -> dict[str, object]:
    if type(value) is not dict or tuple(value) != OBSERVATION_FIELDS:
        raise ProbeFailure("observation-schema")
    if (
        value["case_id"] != case_id
        or type(value["stdout_bytes"]) is not int
        or type(value["stderr_bytes"]) is not int
        or type(value["return_code"]) is not int
        or type(value["surviving_processes"]) is not int
        or value["stderr_bytes"] != 0
        or value["return_code"] != 0
        or value["surviving_processes"] != 0
        or value["error_code"] is not None
    ):
        raise ProbeFailure("observation-schema")
    receipt = _validate_child_receipt(value["child_receipt"], case_id)
    if value["stdout_bytes"] != len(_canonical_line(receipt)):
        raise ProbeFailure("observation-relation")
    return value


def _terminate_process_tree(process) -> None:
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=10,
            )
        except BaseException:
            process.kill()
    else:
        process.kill()
    try:
        process.wait(timeout=10)
    except BaseException:
        pass


def _read_capture(path) -> bytes:
    if not path.is_file():
        return b""
    with path.open("rb") as handle:
        return handle.read(MAX_OUTPUT_BYTES + 1)


def _observe_case(case_id: str, scratch, run_index: int) -> dict[str, object]:
    stdout_path = scratch / f"run-{run_index}-{case_id}.stdout"
    stderr_path = scratch / f"run-{run_index}-{case_id}.stderr"
    process = None
    error_code = None
    child_receipt = None
    stdout = b""
    stderr = b""
    return_code = None
    surviving_processes = 0
    try:
        try:
            with stdout_path.open("xb") as stdout_handle, stderr_path.open(
                "xb"
            ) as stderr_handle:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-S",
                        "-P",
                        str(Path(__file__).resolve()),
                        "--child",
                        case_id,
                    ],
                    cwd=Path(__file__).parents[2].resolve(),
                    env=_probe_environment(),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    creationflags=(
                        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        if os.name == "nt"
                        else 0
                    ),
                    start_new_session=os.name != "nt",
                )
        except BaseException:
            error_code = "spawn"
        if process is not None:
            started = time.monotonic()
            while process.poll() is None:
                if time.monotonic() - started > TIMEOUT_SECONDS:
                    error_code = "timeout"
                    _terminate_process_tree(process)
                    break
                size = sum(
                    path.stat().st_size for path in (stdout_path, stderr_path) if path.exists()
                )
                if size > MAX_OUTPUT_BYTES:
                    error_code = "output_cap"
                    _terminate_process_tree(process)
                    break
                time.sleep(0.02)
            return_code = process.poll()
            if return_code is None:
                _terminate_process_tree(process)
                return_code = process.poll()
            stdout = _read_capture(stdout_path)
            stderr = _read_capture(stderr_path)
            if error_code is None and len(stdout) + len(stderr) > MAX_OUTPUT_BYTES:
                error_code = "output_cap"
            if error_code is None and stderr:
                error_code = "stderr"
            if error_code is None and return_code != 0:
                error_code = "exit"
            if error_code is None:
                try:
                    parsed = _loads(stdout)
                    if _canonical_line(parsed) != stdout:
                        raise ProbeFailure("child-canonical")
                    child_receipt = _validate_child_receipt(parsed, case_id)
                except BaseException:
                    error_code = "schema"
        surviving_processes = 0 if process is None or process.poll() is not None else 1
        if surviving_processes:
            error_code = error_code or "cleanup"
    finally:
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)
        cleanup_failed = False
        for path in (stdout_path, stderr_path):
            try:
                if path.exists():
                    path.unlink()
            except BaseException:
                cleanup_failed = True
        if cleanup_failed:
            error_code = "cleanup"
            child_receipt = None
    return {
        "case_id": case_id,
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "return_code": return_code,
        "child_receipt": child_receipt,
        "surviving_processes": surviving_processes,
        "error_code": error_code,
    }


def _derive_prefix(observations: list[dict[str, object]]) -> tuple[bool, str | None]:
    if len(observations) != len(CASE_IDS):
        return False, None
    receipts = []
    for expected_case, observation in zip(CASE_IDS, observations, strict=True):
        try:
            _validate_observation(observation, expected_case)
        except ProbeFailure:
            return False, None
        receipts.append(observation["child_receipt"])
    failures = [receipt["error_code"] for receipt in receipts if receipt["status"] == "failed"]
    identified = failures[0] if failures else None
    if any(failure != identified for failure in failures):
        return False, None
    if identified is None:
        return all(receipt["status"] == "passed" for receipt in receipts), None
    failure_index = STAGES.index(identified)
    for target_index, receipt in enumerate(receipts):
        expected_status = "passed" if target_index < failure_index else "failed"
        if receipt["status"] != expected_status:
            return False, None
        if expected_status == "failed" and receipt["error_code"] != identified:
            return False, None
    return True, identified


def _run_receipt(scratch, run_index: int) -> dict[str, object]:
    observations = [
        _observe_case(case_id, scratch, run_index) for case_id in CASE_IDS
    ]
    prefix_valid, first_failure = _derive_prefix(observations)
    core = {
        "cases": observations,
        "first_failure": first_failure,
        "child_launches": sum(
            observation["return_code"] is not None for observation in observations
        ),
        "stderr_bytes": sum(observation["stderr_bytes"] for observation in observations),
        "surviving_processes": sum(
            observation["surviving_processes"] for observation in observations
        ),
        "prefix_valid": prefix_valid,
        "passed": (
            prefix_valid
            and all(observation["error_code"] is None for observation in observations)
            and all(observation["stderr_bytes"] == 0 for observation in observations)
            and all(observation["surviving_processes"] == 0 for observation in observations)
        ),
    }
    body_sha256 = _sha256_domain(
        b"L2D-runtime-forensics-v1/run",
        _canonical_line(core),
    )
    return {**core, "body_sha256": body_sha256}


def _validate_run(value: object) -> dict[str, object]:
    if type(value) is not dict or tuple(value) != RUN_FIELDS:
        raise ProbeFailure("run-schema")
    cases = value["cases"]
    if type(cases) is not list:
        raise ProbeFailure("run-schema")
    prefix_valid, first_failure = _derive_prefix(cases)
    core = {key: value[key] for key in RUN_FIELDS[:-1]}
    expected_hash = _sha256_domain(
        b"L2D-runtime-forensics-v1/run",
        _canonical_line(core),
    )
    if (
        value["first_failure"] != first_failure
        or value["prefix_valid"] is not prefix_valid
        or type(value["passed"]) is not bool
        or value["passed"] is not prefix_valid
        or value["child_launches"] != len(CASE_IDS)
        or value["stderr_bytes"] != 0
        or value["surviving_processes"] != 0
        or value["body_sha256"] != expected_hash
    ):
        raise ProbeFailure("run-relation")
    return value


def _git(*arguments: str, binary: bool = False):
    completed = subprocess.run(
        ["git", "-C", str(Path(__file__).parents[2].resolve()), *arguments],
        check=True,
        capture_output=True,
    )
    return completed.stdout if binary else completed.stdout.decode("utf-8").strip()


def _build_result(scratch) -> dict[str, object]:
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ProbeFailure("dirty-worktree")
    revision = _git("rev-parse", "HEAD")
    _git("merge-base", "--is-ancestor", PLAN_REVISION, revision)
    plan_blob = _git("show", f"{PLAN_REVISION}:{PLAN_PATH_POSIX}", binary=True)
    if _git("show", f"HEAD:{PLAN_PATH_POSIX}", binary=True) != plan_blob:
        raise ProbeFailure("plan-drift")
    source_blob = _git("show", f"HEAD:{SOURCE_PATH_POSIX}", binary=True)
    runs = [_run_receipt(scratch, index) for index in range(2)]
    runs_equal = _canonical_line(runs[0]) == _canonical_line(runs[1])
    diagnostic_passed = runs_equal and all(run["passed"] for run in runs)
    identified_stage = runs[0]["first_failure"] if diagnostic_passed else None
    if diagnostic_passed and identified_stage is None:
        action = "freeze_constraint_progress_v3_reproducible_runtime_plan"
    elif diagnostic_passed:
        action = "freeze_constraint_progress_v3_stage_bounded_runtime_plan"
    else:
        action = "park_constraint_progress_runtime_research"
    receipt_root = _sha256_domain(
        b"L2D-runtime-forensics-v1/receipt",
        _canonical_line(runs[0]) + b"\0" + _canonical_line(runs[1]),
    )
    return {
        "schema_version": 1,
        "checkpoint_id": CHECKPOINT_ID,
        "plan_revision": PLAN_REVISION,
        "probe_revision": revision,
        "plan_sha256": _sha256_bytes(plan_blob),
        "probe_source_sha256": _sha256_bytes(source_blob),
        "failed_attempt_revision": FAILED_ATTEMPT_REVISION,
        "failed_contract_sha256": FAILED_CONTRACT_SHA256,
        "expected_runtime_identity": EXPECTED_RUNTIME_IDENTITY,
        "runs": runs,
        "runs_equal": runs_equal,
        "identified_stage": identified_stage,
        "diagnostic_status": "passed" if diagnostic_passed else "failed",
        "action": action,
        "receipt_root_sha256": receipt_root,
    }


def _validate_result(value: object) -> dict[str, object]:
    if type(value) is not dict or tuple(value) != RESULT_FIELDS:
        raise ProbeFailure("result-schema")
    if (
        value["schema_version"] != 1
        or value["checkpoint_id"] != CHECKPOINT_ID
        or value["plan_revision"] != PLAN_REVISION
        or not _is_hex(value["probe_revision"], 40)
        or not _is_hex(value["plan_sha256"])
        or not _is_hex(value["probe_source_sha256"])
        or value["failed_attempt_revision"] != FAILED_ATTEMPT_REVISION
        or value["failed_contract_sha256"] != FAILED_CONTRACT_SHA256
        or value["expected_runtime_identity"] != EXPECTED_RUNTIME_IDENTITY
        or type(value["runs"]) is not list
        or len(value["runs"]) != 2
        or type(value["runs_equal"]) is not bool
        or value["identified_stage"] not in {*STAGES, None}
        or value["diagnostic_status"] not in {"passed", "failed"}
        or value["action"]
        not in {
            "freeze_constraint_progress_v3_stage_bounded_runtime_plan",
            "freeze_constraint_progress_v3_reproducible_runtime_plan",
            "park_constraint_progress_runtime_research",
        }
        or not _is_hex(value["receipt_root_sha256"])
    ):
        raise ProbeFailure("result-schema")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ProbeFailure("result-provenance")
    expected_revision = _git("rev-parse", "HEAD")
    _git("merge-base", "--is-ancestor", PLAN_REVISION, expected_revision)
    expected_plan_sha256 = _sha256_bytes(
        _git("show", f"{PLAN_REVISION}:{PLAN_PATH_POSIX}", binary=True)
    )
    if _git("show", f"HEAD:{PLAN_PATH_POSIX}", binary=True) != _git(
        "show", f"{PLAN_REVISION}:{PLAN_PATH_POSIX}", binary=True
    ):
        raise ProbeFailure("result-provenance")
    expected_source_sha256 = _sha256_bytes(
        _git("show", f"HEAD:{SOURCE_PATH_POSIX}", binary=True)
    )
    if (
        value["probe_revision"] != expected_revision
        or value["plan_sha256"] != expected_plan_sha256
        or value["probe_source_sha256"] != expected_source_sha256
    ):
        raise ProbeFailure("result-provenance")
    runs = [_validate_run(run) for run in value["runs"]]
    runs_equal = _canonical_line(runs[0]) == _canonical_line(runs[1])
    passed = runs_equal and all(run["passed"] for run in runs)
    identified = runs[0]["first_failure"] if passed else None
    expected_action = (
        "freeze_constraint_progress_v3_reproducible_runtime_plan"
        if passed and identified is None
        else "freeze_constraint_progress_v3_stage_bounded_runtime_plan"
        if passed
        else "park_constraint_progress_runtime_research"
    )
    expected_root = _sha256_domain(
        b"L2D-runtime-forensics-v1/receipt",
        _canonical_line(runs[0]) + b"\0" + _canonical_line(runs[1]),
    )
    if (
        value["runs_equal"] is not runs_equal
        or value["diagnostic_status"] != ("passed" if passed else "failed")
        or value["identified_stage"] != identified
        or value["action"] != expected_action
        or value["receipt_root_sha256"] != expected_root
    ):
        raise ProbeFailure("result-relation")
    return value


def _inside(path, root) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def run_once() -> dict[str, object]:
    _late_imports()
    checkout = Path(__file__).parents[2].resolve()
    private_root = checkout.with_name(f"{checkout.name}-local-lab").resolve()
    result_root = Path(tempfile.mkdtemp(prefix=RESULT_DIRECTORY_PREFIX)).resolve()
    if _inside(result_root, checkout) or _inside(result_root, private_root):
        shutil.rmtree(result_root, ignore_errors=True)
        raise ProbeFailure("unsafe-result-root")
    try:
        scratch = result_root / "capture"
        scratch.mkdir()
        value = _build_result(scratch)
        shutil.rmtree(scratch)
        result_path = result_root / "result.json"
        sidecar_path = result_root / "result.json.sha256"
        encoded = _canonical_line(value)
        digest = _sha256_bytes(encoded)
        with result_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        with sidecar_path.open("xb") as handle:
            handle.write(f"{digest}  result.json\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        observed = result_path.read_bytes()
        sidecar = sidecar_path.read_bytes()
        if observed != encoded or sidecar != f"{digest}  result.json\n".encode("ascii"):
            raise ProbeFailure("result-sidecar")
        verified = _validate_result(_loads(observed))
        result_path.unlink()
        sidecar_path.unlink()
        result_root.rmdir()
        return verified
    except BaseException:
        shutil.rmtree(result_root, ignore_errors=True)
        raise


def _summary(value: dict[str, object]) -> dict[str, object]:
    return {
        "checkpoint_id": CHECKPOINT_ID,
        "plan_revision": value["plan_revision"],
        "probe_revision": value["probe_revision"],
        "plan_sha256": value["plan_sha256"],
        "probe_source_sha256": value["probe_source_sha256"],
        "runs_equal": value["runs_equal"],
        "identified_stage": value["identified_stage"],
        "diagnostic_status": value["diagnostic_status"],
        "action": value["action"],
        "receipt_root_sha256": value["receipt_root_sha256"],
    }


def main() -> None:
    mode, case_id = _discover_mode()
    if mode == "child":
        try:
            receipt = _child_receipt(case_id)
            _late_imports()
            encoded = _canonical_line(receipt)
            written = sys.stdout.buffer.write(encoded)
            sys.stdout.buffer.flush()
            raise SystemExit(0 if written == len(encoded) else 71)
        except SystemExit:
            raise
        except BaseException:
            raise SystemExit(71) from None
    if mode == "run":
        try:
            value = run_once()
            encoded = _canonical_line(_summary(value))
            written = sys.stdout.buffer.write(encoded)
            sys.stdout.buffer.flush()
            raise SystemExit(0 if written == len(encoded) else 70)
        except SystemExit:
            raise
        except BaseException:
            try:
                _late_imports()
                failure = {
                    "checkpoint_id": CHECKPOINT_ID,
                    "diagnostic_status": "failed",
                    "action": "park_constraint_progress_runtime_research",
                    "error_code": "runner",
                }
                sys.stdout.buffer.write(_canonical_line(failure))
                sys.stdout.buffer.flush()
            except BaseException:
                pass
            raise SystemExit(70) from None
    raise SystemExit(64)


if __name__ == "__main__":
    main()
