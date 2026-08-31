"""Framed standard-library bootstrap for constraint-aware-progress-toy-v2."""

from __future__ import annotations

import os
import struct
import sys


STUDY_ID = "constraint-aware-progress-toy-v2"
FULL_MODE = STUDY_ID
PROJECTION_MODE = f"{STUDY_ID}-projection"
PHASE_MODE = f"{STUDY_ID}-phase"
RUNTIME_MODE = f"{STUDY_ID}-runtime-probe"
MODES = (FULL_MODE, PROJECTION_MODE, PHASE_MODE, RUNTIME_MODE)
RECURSIVE_MODES = (PROJECTION_MODE, PHASE_MODE)
FAILURE_STAGES = (
    "gate",
    "length",
    "payload",
    "environment",
    "import",
    "dispatch",
    "output",
    "cleanup",
)
GATE = b"L2D-CONSTRAINT-PROGRESS-V2\n"
MAX_PACKET_BYTES = 1_048_576
CHILD_TIMEOUT_SECONDS = 60 * 60
PLAN_REVISION = "c5314afaa50490e39c53669d971114d280e43c07"
TRANSCRIPT_ROOT_SHA256 = (
    "9c250412d296b7e60a5ab0e02f4cf69925d165bb6c3f61e3a29e00b475d99edd"
)
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
CONTRACT_ENVIRONMENT = {
    "L2D_CONTRACT_SHA256",
    "L2D_PLAN_REVISION",
    "L2D_STUDY_REVISION",
}
WORLD_FIELDS = (
    "family",
    "world",
    "bits",
    "split",
    "a",
    "b",
    "k",
    "t",
    "c",
    "threshold",
    "reference_x0",
    "reference_sensitivity",
    "denominator",
)
PHASE_FIELDS = (
    "family",
    "split",
    "contract_sha256",
    "transcript_root_sha256",
    "worlds",
)


class BootstrapFailure(RuntimeError):
    def __init__(self, stage: str):
        self.stage = stage if stage in FAILURE_STAGES else "dispatch"
        super().__init__(self.stage)


def _discover_mode() -> str:
    arguments = tuple(sys.argv[1:])
    for mode in MODES:
        if arguments == ("--mode", mode):
            return mode
    return "invalid"


def _failure_bytes(mode: str, stage: str) -> bytes:
    safe_mode = mode if mode in MODES else "invalid"
    safe_stage = stage if stage in FAILURE_STAGES else "dispatch"
    return (
        '{"schema_version":1,"study_id":"'
        + STUDY_ID
        + '","mode":"'
        + safe_mode
        + '","stage":"'
        + safe_stage
        + '"}\n'
    ).encode("ascii")


def _emit_failure(mode: str, stage: str) -> None:
    try:
        sys.stdout.buffer.write(_failure_bytes(mode, stage))
        sys.stdout.buffer.flush()
    except BaseException:
        pass
    raise SystemExit(70)


def _read_exact(handle, length: int, stage: str) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        item = handle.read(length - len(chunks))
        if not item:
            raise BootstrapFailure(stage)
        chunks.extend(item)
    return bytes(chunks)


def _read_frame() -> bytes:
    handle = sys.stdin.buffer
    if _read_exact(handle, len(GATE), "gate") != GATE:
        raise BootstrapFailure("gate")
    raw_length = _read_exact(handle, 4, "length")
    length = struct.unpack("<I", raw_length)[0]
    if length > MAX_PACKET_BYTES:
        raise BootstrapFailure("payload")
    payload = _read_exact(handle, length, "payload")
    if handle.read(1) != b"":
        raise BootstrapFailure("payload")
    return payload


def _late_imports() -> None:
    global hashlib, importlib, json, math, Path, platform, socket
    global subprocess, tempfile, threading, time
    import hashlib
    import importlib
    import json
    import math
    import platform
    import socket
    import subprocess
    import tempfile
    import threading
    import time
    from pathlib import Path


def _reject_duplicate_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise BootstrapFailure("payload")
        value[key] = item
    return value


def _reject_constant(_value):
    raise BootstrapFailure("payload")


def _loads(value: bytes):
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except BootstrapFailure:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BootstrapFailure("payload") from None


def _is_hex(value, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_world_packet(value) -> None:
    if not isinstance(value, dict) or tuple(value) != WORLD_FIELDS:
        raise BootstrapFailure("payload")
    if type(value["family"]) is not str or value["family"] not in {
        "canonical",
        "aligned",
        "impossible",
    }:
        raise BootstrapFailure("payload")
    if type(value["split"]) is not str or value["split"] not in {
        "development",
        "heldout",
    }:
        raise BootstrapFailure("payload")
    if type(value["world"]) is not int or value["world"] not in range(16):
        raise BootstrapFailure("payload")
    bits = value["bits"]
    if (
        type(bits) is not list
        or len(bits) != 4
        or any(type(item) is not int or item not in {0, 1} for item in bits)
    ):
        raise BootstrapFailure("payload")
    for name in ("a", "b", "k", "t", "c", "threshold"):
        if type(value[name]) is not float or not math.isfinite(value[name]):
            raise BootstrapFailure("payload")
    for name in ("reference_x0", "reference_sensitivity", "denominator"):
        if value[name] is not None and (
            type(value[name]) is not float or not math.isfinite(value[name])
        ):
            raise BootstrapFailure("payload")


def _authenticate_environment() -> None:
    if os.environ.get("LEARN2DESIGN_LOCAL_LAB_NETWORK") != "disabled":
        raise BootstrapFailure("environment")
    inherited = {
        name.upper() for name in os.environ if name.upper().startswith("L2D_")
    }
    if inherited != CONTRACT_ENVIRONMENT:
        raise BootstrapFailure("environment")
    if os.environ.get("L2D_PLAN_REVISION") != PLAN_REVISION:
        raise BootstrapFailure("environment")
    if not _is_hex(os.environ.get("L2D_STUDY_REVISION"), 40):
        raise BootstrapFailure("environment")
    if not _is_hex(os.environ.get("L2D_CONTRACT_SHA256"), 64):
        raise BootstrapFailure("environment")


def _validate_payload(mode: str, payload: bytes):
    if mode == PROJECTION_MODE:
        if payload != b"":
            raise BootstrapFailure("payload")
        return None
    if mode != PHASE_MODE:
        if payload not in {b"", None}:
            raise BootstrapFailure("payload")
        return None
    value = _loads(payload)
    if not isinstance(value, dict) or tuple(value) != PHASE_FIELDS:
        raise BootstrapFailure("payload")
    if type(value["family"]) is not str or value["family"] not in {
        "canonical",
        "aligned",
        "impossible",
    }:
        raise BootstrapFailure("payload")
    if type(value["split"]) is not str or value["split"] not in {
        "development",
        "heldout",
    }:
        raise BootstrapFailure("payload")
    if value["contract_sha256"] != os.environ["L2D_CONTRACT_SHA256"]:
        raise BootstrapFailure("environment")
    if value["transcript_root_sha256"] != TRANSCRIPT_ROOT_SHA256:
        raise BootstrapFailure("environment")
    worlds = value["worlds"]
    if type(worlds) is not list or len(worlds) != 8:
        raise BootstrapFailure("payload")
    for world in worlds:
        _validate_world_packet(world)
    canonical = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if canonical != payload:
        raise BootstrapFailure("payload")
    return value


def _deny_network(*_args, **_kwargs):
    raise BootstrapFailure("environment")


def _disable_network() -> None:
    socket.socket = _deny_network
    socket.create_connection = _deny_network
    socket.getaddrinfo = _deny_network
    socket.gethostbyaddr = _deny_network
    socket.gethostbyname = _deny_network
    socket.gethostbyname_ex = _deny_network
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "LEARN2DESIGN_LOCAL_LAB_NETWORK": "disabled",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )


def _prove_network_disabled() -> None:
    probes = (
        (socket.socket, ()),
        (socket.create_connection, (("127.0.0.1", 9),)),
        (socket.getaddrinfo, ("localhost", 9)),
        (socket.gethostbyaddr, ("127.0.0.1",)),
        (socket.gethostbyname, ("localhost",)),
        (socket.gethostbyname_ex, ("localhost",)),
    )
    for probe, arguments in probes:
        try:
            probe(*arguments)
        except BootstrapFailure as error:
            if error.stage == "environment":
                continue
        except BaseException:
            pass
        raise BootstrapFailure("environment")


def _load_scientific_module():
    import site

    root = str(Path(__file__).parents[2].resolve())
    user_packages = str(Path(site.getusersitepackages()).resolve())
    for entry in (root, user_packages):
        if entry not in sys.path:
            sys.path.append(entry)
    return importlib.import_module(
        "experiments.local_lab.constraint_aware_progress_toy_v2"
    )


def _normalize(value):
    if isinstance(value, float):
        if not (value == value and value not in {float("inf"), float("-inf")}):
            raise BootstrapFailure("output")
        return 0.0 if value == 0.0 else value
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value


def _canonical_line(value) -> bytes:
    if not isinstance(value, dict):
        raise BootstrapFailure("output")
    try:
        encoded = (
            json.dumps(
                _normalize(value),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise BootstrapFailure("output") from None
    if not (0 < len(encoded) <= MAX_PACKET_BYTES):
        raise BootstrapFailure("output")
    return encoded


def _source_environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper() in SAFE_ENVIRONMENT or name.upper() in CONTRACT_ENVIRONMENT
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "LEARN2DESIGN_LOCAL_LAB_NETWORK": "disabled",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    return environment


class _Job:
    """A fresh kill-on-close Windows Job with exact membership checks."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise BootstrapFailure("environment")
        import ctypes
        from ctypes import wintypes

        class LARGE_INTEGER(ctypes.Structure):
            _fields_ = [("QuadPart", ctypes.c_longlong)]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", LARGE_INTEGER),
                ("PerJobUserTimeLimit", LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC_LIMIT),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class BASIC_ACCOUNTING(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", LARGE_INTEGER),
                ("TotalKernelTime", LARGE_INTEGER),
                ("ThisPeriodTotalUserTime", LARGE_INTEGER),
                ("ThisPeriodTotalKernelTime", LARGE_INTEGER),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        kernel32.IsProcessInJob.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._kernel32 = kernel32
        self._accounting_type = BASIC_ACCOUNTING
        self.handle = kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            raise BootstrapFailure("environment")
        limits = EXTENDED_LIMIT()
        limits.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            self.close()
            raise BootstrapFailure("environment")

    def assign_and_verify(self, process) -> None:
        if not self._kernel32.AssignProcessToJobObject(
            self.handle, int(process._handle)
        ):
            raise BootstrapFailure("environment")
        contained = self._wintypes.BOOL()
        if not self._kernel32.IsProcessInJob(
            int(process._handle),
            self.handle,
            self._ctypes.byref(contained),
        ) or not bool(contained.value):
            raise BootstrapFailure("environment")

    def active_processes(self) -> int:
        value = self._accounting_type()
        if not self._kernel32.QueryInformationJobObject(
            self.handle,
            1,
            self._ctypes.byref(value),
            self._ctypes.sizeof(value),
            None,
        ):
            raise BootstrapFailure("cleanup")
        return int(value.ActiveProcesses)

    def terminate(self) -> None:
        if self.handle is not None:
            if not self._kernel32.TerminateJobObject(self.handle, 1):
                raise BootstrapFailure("cleanup")

    def close(self) -> None:
        if self.handle is not None:
            if not self._kernel32.CloseHandle(self.handle):
                raise BootstrapFailure("cleanup")
            self.handle = None


def _force_close_stdin(process) -> None:
    if process is None or process.stdin is None:
        return
    try:
        os.close(process.stdin.fileno())
    except (OSError, ValueError):
        pass


def _terminate(process, job) -> None:
    cleanup_failed = False
    if job is not None:
        try:
            job.terminate()
        except BaseException:
            cleanup_failed = True
    elif process is not None and process.poll() is None:
        try:
            process.kill()
        except BaseException:
            cleanup_failed = True
    if process is not None and process.poll() is None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=10)
            except BaseException:
                cleanup_failed = True
    if process is not None and process.poll() is None:
        cleanup_failed = True
    if cleanup_failed:
        raise BootstrapFailure("cleanup")


def _drain_job(job) -> None:
    if job is None:
        return
    if job.active_processes() > 0:
        job.terminate()
        deadline = time.monotonic() + 10
        while job.active_processes() > 0 and time.monotonic() < deadline:
            time.sleep(0.01)
    if job.active_processes() != 0:
        raise BootstrapFailure("cleanup")


def _read_bounded(path, limit: int) -> bytes:
    if type(limit) is not int or limit < 0:
        raise BootstrapFailure("output")
    with path.open("rb") as handle:
        value = handle.read(limit + 1)
    if len(value) > limit:
        raise BootstrapFailure("output")
    return value


def _parse_failure(value: bytes, expected_mode: str) -> str:
    parsed = _loads(value)
    if (
        not isinstance(parsed, dict)
        or tuple(parsed) != ("schema_version", "study_id", "mode", "stage")
        or type(parsed["schema_version"]) is not int
        or parsed["schema_version"] != 1
        or type(parsed["study_id"]) is not str
        or parsed["study_id"] != STUDY_ID
        or type(parsed["mode"]) is not str
        or parsed["mode"] != expected_mode
        or type(parsed["stage"]) is not str
        or parsed["stage"] not in FAILURE_STAGES
        or _failure_bytes(expected_mode, parsed["stage"]) != value
    ):
        raise BootstrapFailure("dispatch")
    return parsed["stage"]


def _run_child(mode: str, payload=None):
    if mode not in RECURSIVE_MODES:
        raise BootstrapFailure("dispatch")
    if payload is None:
        encoded = b""
    else:
        try:
            encoded = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise BootstrapFailure("payload") from None
    if len(encoded) > MAX_PACKET_BYTES:
        raise BootstrapFailure("payload")
    frame = GATE + struct.pack("<I", len(encoded)) + encoded
    command = [sys.executable, "-S", "-P", str(Path(__file__).resolve()), "--mode", mode]
    started = time.monotonic()
    job = _Job()
    process = None
    writer = None
    writer_failed = []
    temporary = None
    root = None
    cleanup_failed = False
    try:
        temporary = tempfile.TemporaryDirectory(prefix="l2d-constraint-v2-")
        root = Path(temporary.name)
        stdout_path = root / "stdout"
        stderr_path = root / "stderr"
        with stdout_path.open("xb") as stdout_handle, stderr_path.open(
            "xb"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).parents[2],
                env=_source_environment(),
                stdin=subprocess.PIPE,
                stdout=stdout_handle,
                stderr=stderr_handle,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            job.assign_and_verify(process)

            def write_frame() -> None:
                try:
                    written = process.stdin.write(frame)
                    if written not in {None, len(frame)}:
                        raise BootstrapFailure("payload")
                    process.stdin.close()
                except BaseException:
                    writer_failed.append(True)

            writer = threading.Thread(
                target=write_frame,
                name="constraint-v2-frame-writer",
                daemon=False,
            )
            writer.start()
            while process.poll() is None:
                if writer_failed:
                    raise BootstrapFailure("cleanup")
                if time.monotonic() - started > CHILD_TIMEOUT_SECONDS:
                    raise BootstrapFailure("cleanup")
                if stdout_path.stat().st_size + stderr_path.stat().st_size > MAX_PACKET_BYTES:
                    raise BootstrapFailure("output")
                time.sleep(0.05)
            writer.join(timeout=10)
            if writer.is_alive() or writer_failed:
                raise BootstrapFailure("cleanup")
        stdout = _read_bounded(stdout_path, MAX_PACKET_BYTES)
        stderr = _read_bounded(stderr_path, MAX_PACKET_BYTES - len(stdout))
        if stderr:
            raise BootstrapFailure("dispatch")
        if job.active_processes() != 0:
            raise BootstrapFailure("cleanup")
        if process.returncode == 70:
            raise BootstrapFailure(_parse_failure(stdout, mode))
        if process.returncode != 0:
            raise BootstrapFailure("dispatch")
        if not stdout.endswith(b"\n") or stdout.count(b"\n") != 1:
            raise BootstrapFailure("output")
        parsed = _loads(stdout)
        if not isinstance(parsed, dict) or _canonical_line(parsed) != stdout:
            raise BootstrapFailure("output")
        return stdout, 0, {
            "stdout_bytes": len(stdout),
            "stderr_bytes": 0,
        }
    finally:
        try:
            _force_close_stdin(process)
        except BaseException:
            cleanup_failed = True
        if writer is not None and writer.is_alive():
            try:
                writer.join(timeout=10)
            except BaseException:
                cleanup_failed = True
            cleanup_failed = cleanup_failed or writer.is_alive()
        if process is not None and process.poll() is None:
            try:
                _terminate(process, job)
            except BaseException:
                cleanup_failed = True
        try:
            _drain_job(job)
        except BaseException:
            cleanup_failed = True
        finally:
            try:
                job.close()
            except BaseException:
                cleanup_failed = True
        if writer is not None and writer.is_alive():
            writer.join(timeout=10)
            cleanup_failed = cleanup_failed or writer.is_alive()
        if process is not None and process.poll() is None:
            cleanup_failed = True
        if temporary is not None:
            try:
                temporary.cleanup()
            except BaseException:
                cleanup_failed = True
        if root is not None and root.exists():
            cleanup_failed = True
        if cleanup_failed:
            raise BootstrapFailure("cleanup")


def _dispatch(mode: str, payload):
    try:
        scientific = _load_scientific_module()
        identity = scientific.validate_runtime_identity()
    except BaseException:
        raise BootstrapFailure("import") from None
    try:
        if mode == RUNTIME_MODE:
            return identity
        if mode == FULL_MODE:
            return scientific.run_full(_run_child)
        if mode == PROJECTION_MODE:
            return scientific.run_projection(_run_child)
        if mode == PHASE_MODE:
            return scientific.run_phase(payload)
    except BootstrapFailure:
        raise
    except BaseException:
        raise BootstrapFailure("dispatch") from None
    raise BootstrapFailure("dispatch")


def main() -> None:
    mode = _discover_mode()
    if mode == "invalid":
        _emit_failure(mode, "environment")
    payload = None
    output_started = False
    try:
        if mode in RECURSIVE_MODES:
            payload = _read_frame()
        _authenticate_environment()
        try:
            _late_imports()
        except BaseException:
            raise BootstrapFailure("environment") from None
        payload = _validate_payload(mode, payload)
        _disable_network()
        _prove_network_disabled()
        result = _dispatch(mode, payload)
        encoded = _canonical_line(result)
        try:
            output_started = True
            written = sys.stdout.buffer.write(encoded)
            if written != len(encoded):
                raise BootstrapFailure("output")
            sys.stdout.buffer.flush()
        except BaseException:
            raise BootstrapFailure("output") from None
    except BootstrapFailure as error:
        if output_started:
            raise SystemExit(70) from None
        _emit_failure(mode, error.stage)
    except BaseException:
        if output_started:
            raise SystemExit(70) from None
        _emit_failure(mode, "dispatch")


if __name__ == "__main__":
    main()
