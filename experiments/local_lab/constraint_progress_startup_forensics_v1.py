"""Bounded, standard-library-only Windows startup forensics probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Mapping


CHECKPOINT_ID = "constraint-progress-startup-forensics-v1"
PLAN_REVISION = "59f3c5a5ab5a985b1a67477b68eb6eb9976c2f3f"
EXPECTED_EXECUTABLE_SHA256 = (
    "ad169f4cb4bfb78c7a5c030a4529c19d6643276778e33994c93e145b6191c3ec"
)
EXPECTED_RUNTIME_IDENTITY = {
    "python_executable_sha256": EXPECTED_EXECUTABLE_SHA256,
    "python_version": "3.13.14",
    "python_architecture": "64bit",
    "machine": "AMD64",
}
ROOT = Path(__file__).parents[2]
PLAN_PATH = ROOT / "research/2026-08-31-constraint-progress-startup-forensics-v1-plan.md"
SOURCE_PATH = Path(__file__)
PRIVATE_ROOT = ROOT.parent / "learn2design-local-lab"
RESULT_DIRECTORY_PREFIX = "l2d-constraint-progress-startup-forensics-v1-"

CHILD_GATE = b"L2D-STARTUP-FORENSICS-V1\n"
OUTER_GATE = b"L2D-STARTUP-FORENSICS-OUTER-V1\n"
MAX_PAYLOAD_BYTES = 32_768
MAX_OUTPUT_BYTES = 65_536
TIMEOUT_SECONDS = 60.0
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

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
    "LEARN2DESIGN_LOCAL_LAB_NETWORK": "disabled",
    "PYTHONHASHSEED": "0",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}

CASE_SPECS = (
    ("gate_only_empty", False, "contiguous", 0, "accepted", None),
    ("nested_empty", True, "contiguous", 0, "accepted", None),
    ("nested_fragmented", True, "fragmented", 4_097, "accepted", None),
    ("nested_large", True, "contiguous", 32_768, "accepted", None),
    ("nested_wrong_gate", True, "wrong_gate", 0, "rejected", "gate_read"),
    (
        "nested_truncated_gate",
        True,
        "truncated_gate",
        0,
        "rejected",
        "gate_read",
    ),
    ("nested_no_gate", True, "no_gate", 0, "rejected", "gate_read"),
    (
        "nested_short_length",
        True,
        "short_length",
        0,
        "rejected",
        "length_read",
    ),
    (
        "nested_short_payload",
        True,
        "short_payload",
        0,
        "rejected",
        "payload_read",
    ),
    (
        "nested_oversized_length",
        True,
        "oversized_length",
        0,
        "rejected",
        "payload_cap",
    ),
    (
        "nested_trailing_input",
        True,
        "trailing_input",
        0,
        "rejected",
        "trailing_input",
    ),
)
CASE_IDS = tuple(item[0] for item in CASE_SPECS)
CASE_BY_ID = {item[0]: item for item in CASE_SPECS}
CHILD_ERROR_CODES = {
    "gate_read",
    "length_read",
    "payload_read",
    "payload_cap",
    "trailing_input",
}
PARENT_ERROR_CODES = {
    "outer_gate_read",
    "job_create",
    "job_limit",
    "child_spawn",
    "job_assign",
    "job_membership",
    "gate_write",
    "timeout",
    "output_cap",
    "writer_join",
    "child_stderr",
    "child_exit",
    "child_schema",
    "child_relation",
    "job_query",
    "survivor",
}
RUNNER_ERROR_CODES = {
    "runtime_identity",
    "dirty_worktree",
    "outer_job_create",
    "outer_job_limit",
    "outer_spawn",
    "outer_job_assign",
    "outer_job_membership",
    "outer_gate_write",
    "outer_timeout",
    "outer_output_cap",
    "outer_stderr",
    "outer_exit",
    "outer_schema",
    "outer_relation",
    "outer_job_query",
    "outer_survivor",
}


class ProbeError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class FrameError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _reject_duplicate_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate-key")
        value[key] = item
    return value


def _loads(value: bytes | str):
    return json.loads(value, object_pairs_hook=_reject_duplicate_pairs)


def _canonical(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def _process_executable_path() -> Path:
    if os.name != "nt":
        return Path(sys.executable).resolve()
    import ctypes

    buffer = ctypes.create_unicode_buffer(32_768)
    length = ctypes.windll.kernel32.GetModuleFileNameW(None, buffer, len(buffer))
    if length <= 0 or length >= len(buffer):
        raise ProbeError("runtime_identity")
    return Path(buffer.value)


def _runtime_identity() -> dict[str, str]:
    value = {
        "python_executable_sha256": _sha256_file(_process_executable_path()),
        "python_version": platform.python_version(),
        "python_architecture": platform.architecture()[0],
        "machine": platform.machine(),
    }
    if value != EXPECTED_RUNTIME_IDENTITY:
        raise ProbeError("runtime_identity")
    return value


def _probe_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if source is None else source
    selected: dict[str, str] = {}
    seen: set[str] = set()
    for name, value in source.items():
        upper = name.upper()
        if upper in SAFE_ENVIRONMENT:
            if upper in seen:
                raise ProbeError("runtime_identity")
            selected[name] = value
            seen.add(upper)
    for name, value in ENVIRONMENT_OVERRIDES.items():
        for existing in tuple(selected):
            if existing.upper() == name:
                del selected[existing]
        selected[name] = value
    return selected


def _environment_receipt(environment: Mapping[str, str] | None = None) -> tuple[int, str]:
    environment = os.environ if environment is None else environment
    rows = sorted((name.upper(), value) for name, value in environment.items())
    if len({name for name, _value in rows}) != len(rows):
        raise ProbeError("runtime_identity")
    digest = hashlib.sha256(b"L2D-startup-forensics-v1/environment\0")
    for name, value in rows:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return len(rows), digest.hexdigest()


def _payload(length: int) -> bytes:
    value = bytearray()
    index = 0
    while len(value) < length:
        value.extend(
            hashlib.sha256(
                b"L2D-startup-forensics-v1/payload\0" + struct.pack("<I", index)
            ).digest()
        )
        index += 1
    return bytes(value[:length])


def _frame(case_id: str) -> bytes:
    _name, _inner, mode, payload_length, _status, _error = CASE_BY_ID[case_id]
    payload = _payload(payload_length)
    if mode in {"contiguous", "fragmented"}:
        return CHILD_GATE + struct.pack("<I", payload_length) + payload
    if mode == "wrong_gate":
        return CHILD_GATE[:-1] + b"!" + struct.pack("<I", 0)
    if mode == "truncated_gate":
        return CHILD_GATE[:-1]
    if mode == "no_gate":
        return b""
    if mode == "short_length":
        return CHILD_GATE + b"\0\0\0"
    if mode == "short_payload":
        return CHILD_GATE + struct.pack("<I", 5) + _payload(4)
    if mode == "oversized_length":
        return CHILD_GATE + struct.pack("<I", MAX_PAYLOAD_BYTES + 1)
    if mode == "trailing_input":
        return CHILD_GATE + struct.pack("<I", 0) + b"\x01"
    raise ProbeError("child_relation")


def _read_exact(handle, length: int, code: str) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        item = handle.read(length - len(chunks))
        if not item:
            raise FrameError(code)
        chunks.extend(item)
    return bytes(chunks)


class _Job:
    """Ephemeral kill-on-close Windows Job."""

    def __init__(self, *, create_code: str, limit_code: str):
        if os.name != "nt":
            raise ProbeError(create_code)
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
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
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
            ctypes.c_void_p,
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
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ProbeError(create_code)
        limits = EXTENDED_LIMIT()
        limits.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            kernel32.CloseHandle(handle)
            raise ProbeError(limit_code)
        self.handle = handle

    def assign(self, process: subprocess.Popen, code: str) -> None:
        if not self._kernel32.AssignProcessToJobObject(
            self.handle, int(process._handle)  # type: ignore[attr-defined]
        ):
            raise ProbeError(code)

    def contains(self, process: subprocess.Popen, code: str) -> bool:
        contained = self._wintypes.BOOL()
        if not self._kernel32.IsProcessInJob(
            int(process._handle),  # type: ignore[attr-defined]
            self.handle,
            self._ctypes.byref(contained),
        ):
            raise ProbeError(code)
        if not bool(contained.value):
            raise ProbeError(code)
        return True

    def active_processes(self, code: str) -> int:
        value = self._accounting_type()
        if not self._kernel32.QueryInformationJobObject(
            self.handle,
            1,
            self._ctypes.byref(value),
            self._ctypes.sizeof(value),
            None,
        ):
            raise ProbeError(code)
        return int(value.ActiveProcesses)

    def terminate(self) -> None:
        self._kernel32.TerminateJobObject(self.handle, 1)

    def close(self) -> None:
        if self.handle is not None:
            self._kernel32.CloseHandle(self.handle)
            self.handle = None


def _current_process_in_job() -> bool:
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.IsProcessInJob.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.BOOL),
    ]
    kernel32.IsProcessInJob.restype = wintypes.BOOL
    contained = wintypes.BOOL()
    if not kernel32.IsProcessInJob(
        kernel32.GetCurrentProcess(), None, ctypes.byref(contained)
    ):
        raise ProbeError("outer_gate_read")
    return bool(contained.value)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _checked_scratch_directory(path: str | Path, error_code: str) -> Path:
    resolved = Path(path).resolve()
    if _inside(resolved, ROOT.resolve()) or _inside(resolved, PRIVATE_ROOT.resolve()):
        raise ProbeError(error_code)
    return resolved


def _capture_files(
    stdout_path: Path, stderr_path: Path, error_code: str
) -> tuple[bytes, bytes, int, int]:
    stdout = b""
    stderr = b""
    if stdout_path.exists():
        with stdout_path.open("rb") as handle:
            stdout = handle.read(MAX_OUTPUT_BYTES + 1)
    remaining = max(0, MAX_OUTPUT_BYTES - min(len(stdout), MAX_OUTPUT_BYTES))
    if stderr_path.exists():
        with stderr_path.open("rb") as handle:
            stderr = handle.read(remaining + 1)
    stdout_size = stdout_path.stat().st_size if stdout_path.exists() else 0
    stderr_size = stderr_path.stat().st_size if stderr_path.exists() else 0
    if stdout_size + stderr_size > MAX_OUTPUT_BYTES:
        return b"", b"", stdout_size, stderr_size
    if len(stdout) != stdout_size or len(stderr) != stderr_size:
        raise ProbeError(error_code)
    return stdout, stderr, stdout_size, stderr_size


def _post_close_survivors(process: subprocess.Popen | None) -> int:
    if process is None:
        return 0
    return int(process.poll() is None)


def _time_remaining(started: float) -> float:
    return max(0.0, TIMEOUT_SECONDS - (time.monotonic() - started))


def _force_close_stdin(process: subprocess.Popen | None) -> None:
    if process is None or process.stdin is None:
        return
    try:
        os.close(process.stdin.fileno())
    except (OSError, ValueError):
        pass


def _join_writer_after_cleanup(
    writer: threading.Thread | None, process: subprocess.Popen | None
) -> bool:
    if writer is None or not writer.is_alive():
        return True
    _force_close_stdin(process)
    writer.join(timeout=10)
    return not writer.is_alive()


def _terminate_process(process: subprocess.Popen | None, job: _Job | None) -> bool:
    if process is None or process.poll() is not None:
        return True
    if job is not None:
        job.terminate()
    else:
        process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return False
    return process.poll() is not None


def _write_frame(handle, frame: bytes, mode: str, errors: list[str]) -> None:
    try:
        if mode == "fragmented":
            sizes = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89)
            offset = 0
            index = 0
            while offset < len(frame):
                next_offset = min(len(frame), offset + sizes[index % len(sizes)])
                handle.write(frame[offset:next_offset])
                handle.flush()
                offset = next_offset
                index += 1
        elif frame:
            handle.write(frame)
            handle.flush()
        handle.close()
    except (BrokenPipeError, OSError, ValueError):
        errors.append("gate_write")


def _child_receipt(case_id: str) -> dict[str, object]:
    status = "accepted"
    error_code = None
    payload = b""
    try:
        gate = _read_exact(sys.stdin.buffer, len(CHILD_GATE), "gate_read")
        if gate != CHILD_GATE:
            raise FrameError("gate_read")
        raw_length = _read_exact(sys.stdin.buffer, 4, "length_read")
        length = struct.unpack("<I", raw_length)[0]
        if length > MAX_PAYLOAD_BYTES:
            raise FrameError("payload_cap")
        payload = _read_exact(sys.stdin.buffer, length, "payload_read")
        if sys.stdin.buffer.read(1) != b"":
            raise FrameError("trailing_input")
    except FrameError as error:
        status = "rejected"
        error_code = error.code
        payload = b""
    _pairs, environment_sha256 = _environment_receipt(_probe_environment())
    return {
        "schema_version": 1,
        "case_id": case_id,
        "environment_sha256": environment_sha256,
        "status": status,
        "error_code": error_code,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validate_child_receipt(
    value: object, case_id: str, environment_sha256: str
) -> dict[str, object]:
    if not isinstance(value, dict) or tuple(value) != (
        "schema_version",
        "case_id",
        "environment_sha256",
        "status",
        "error_code",
        "payload_bytes",
        "payload_sha256",
    ):
        raise ProbeError("child_schema")
    if value["schema_version"] != 1 or value["case_id"] != case_id:
        raise ProbeError("child_schema")
    if value["environment_sha256"] != environment_sha256:
        raise ProbeError("child_relation")
    status = value["status"]
    error = value["error_code"]
    if status not in {"accepted", "rejected"}:
        raise ProbeError("child_schema")
    if not ((status == "accepted" and error is None) or (status == "rejected" and error in CHILD_ERROR_CODES)):
        raise ProbeError("child_schema")
    if type(value["payload_bytes"]) is not int or not (0 <= value["payload_bytes"] <= MAX_PAYLOAD_BYTES):
        raise ProbeError("child_schema")
    digest = value["payload_sha256"]
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ProbeError("child_schema")
    return value


def _child_relation_matches(receipt: Mapping[str, object], case_id: str) -> bool:
    expected_status = CASE_BY_ID[case_id][4]
    expected_error = CASE_BY_ID[case_id][5]
    expected_payload = (
        _payload(CASE_BY_ID[case_id][3]) if expected_status == "accepted" else b""
    )
    return (
        receipt["status"] == expected_status
        and receipt["error_code"] == expected_error
        and receipt["payload_bytes"] == len(expected_payload)
        and receipt["payload_sha256"] == hashlib.sha256(expected_payload).hexdigest()
    )


def _empty_observation(case_id: str) -> dict[str, object]:
    _name, inner_required, mode, payload_length, expected_status, _error = CASE_BY_ID[case_id]
    expected_payload = _payload(payload_length) if expected_status == "accepted" else b""
    return {
        "case_id": case_id,
        "inner_job_required": inner_required,
        "inner_job_assigned": False,
        "membership_before_gate": False,
        "write_mode": mode,
        "expected_payload_bytes": len(expected_payload),
        "expected_payload_sha256": hashlib.sha256(expected_payload).hexdigest(),
        "child_receipt": None,
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "return_code": None,
        "surviving_descendants": 0,
        "passed": False,
        "error_code": None,
    }


def _observe_child(case_id: str, environment: Mapping[str, str], environment_sha256: str):
    started = time.monotonic()
    observation = _empty_observation(case_id)
    _name, inner_required, mode, _payload_length, _expected_status, _expected_error = CASE_BY_ID[case_id]
    job = None
    process = None
    writer = None
    writer_errors: list[str] = []
    stdout = b""
    stderr = b""
    stdout_size = 0
    stderr_size = 0
    error_code = None
    writer_joined = True
    process_cleanup_ok = True
    try:
        temporary_root = _checked_scratch_directory(
            tempfile.gettempdir(), "child_spawn"
        )
    except ProbeError as error:
        observation["error_code"] = error.code
        return observation
    with tempfile.TemporaryDirectory(
        prefix="l2d-startup-child-", dir=temporary_root
    ) as directory:
        root = Path(directory).resolve()
        stdout_path = root / "stdout"
        stderr_path = root / "stderr"
        try:
            _checked_scratch_directory(root, "child_spawn")
            if inner_required:
                job = _Job(create_code="job_create", limit_code="job_limit")
            with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-S",
                            "-P",
                            str(SOURCE_PATH),
                            "--mode",
                            "child",
                            "--case",
                            case_id,
                        ],
                        cwd=ROOT,
                        env=dict(environment),
                        stdin=subprocess.PIPE,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                    )
                except OSError as error:
                    raise ProbeError("child_spawn") from error
                if job is not None:
                    job.assign(process, "job_assign")
                    observation["inner_job_assigned"] = True
                    job.contains(process, "job_membership")
                    observation["membership_before_gate"] = True
                if _time_remaining(started) == 0:
                    raise ProbeError("timeout")
                frame = _frame(case_id)
                writer = threading.Thread(
                    target=_write_frame,
                    args=(process.stdin, frame, mode, writer_errors),
                    name="startup-forensics-input",
                    daemon=False,
                )
                writer.start()
                while process.poll() is None:
                    if _time_remaining(started) == 0:
                        raise ProbeError("timeout")
                    if stdout_path.stat().st_size + stderr_path.stat().st_size > MAX_OUTPUT_BYTES:
                        raise ProbeError("output_cap")
                    time.sleep(0.01)
                remaining = _time_remaining(started)
                writer.join(timeout=min(10.0, remaining))
                if writer.is_alive():
                    raise ProbeError("timeout" if remaining == 0 else "writer_join")
                if writer_errors:
                    raise ProbeError("gate_write")
                if _time_remaining(started) == 0:
                    raise ProbeError("timeout")
        except ProbeError as error:
            error_code = error.code
            _terminate_process(process, job)
            writer_joined = _join_writer_after_cleanup(writer, process)
        finally:
            try:
                stdout, stderr, stdout_size, stderr_size = _capture_files(
                    stdout_path, stderr_path, "output_cap"
                )
            except ProbeError as error:
                if error_code is None:
                    error_code = error.code
            if stdout_size + stderr_size > MAX_OUTPUT_BYTES and error_code is None:
                error_code = "output_cap"
            observation["stdout_bytes"] = stdout_size
            observation["stderr_bytes"] = stderr_size
            observation["return_code"] = None if process is None else process.returncode
            if job is not None:
                try:
                    observation["surviving_descendants"] = job.active_processes("job_query")
                except ProbeError:
                    observation["surviving_descendants"] = -1
                    # A missing cleanup count invalidates every earlier partial;
                    # ``-1`` is legal only under this integrity error.
                    error_code = "job_query"
                job.close()
                post_close_survivors = _post_close_survivors(process)
                if observation["surviving_descendants"] >= 0:
                    observation["surviving_descendants"] = max(
                        observation["surviving_descendants"], post_close_survivors
                    )
            process_cleanup_ok = _terminate_process(process, None)

    if not writer_joined or not process_cleanup_ok:
        # Returning with a non-daemon writer or live child would violate the
        # frozen bound. Scratch and Job cleanup have completed at this point.
        os._exit(1)

    if error_code is None:
        if stdout_size + stderr_size > MAX_OUTPUT_BYTES:
            error_code = "output_cap"
        elif stderr_size:
            error_code = "child_stderr"
        elif process is None or process.returncode != 0:
            error_code = "child_exit"
        elif observation["surviving_descendants"] != 0:
            error_code = "survivor"
        else:
            try:
                receipt = _validate_child_receipt(_loads(stdout), case_id, environment_sha256)
                observation["child_receipt"] = receipt
                if not _child_relation_matches(receipt, case_id):
                    raise ProbeError("child_relation")
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                error_code = "child_schema"
            except ProbeError as error:
                error_code = error.code
    observation["passed"] = error_code is None
    observation["error_code"] = error_code
    return observation


def _parent_receipt() -> dict[str, object]:
    children = []
    error_code = None
    outer_membership = False
    try:
        if _read_exact(sys.stdin.buffer, len(OUTER_GATE), "outer_gate_read") != OUTER_GATE:
            raise ProbeError("outer_gate_read")
        if sys.stdin.buffer.read(1) != b"":
            raise ProbeError("outer_gate_read")
    except FrameError:
        error_code = "outer_gate_read"
    except ProbeError as error:
        error_code = error.code

    try:
        projected_environment = _probe_environment()
        environment_pairs, environment_sha256 = _environment_receipt(
            projected_environment
        )
    except ProbeError:
        projected_environment = _probe_environment({})
        environment_pairs, environment_sha256 = _environment_receipt(
            projected_environment
        )
        if error_code is None:
            error_code = "outer_gate_read"
    if error_code is None:
        try:
            outer_membership = _current_process_in_job()
            if not outer_membership:
                raise ProbeError("outer_gate_read")
        except ProbeError as error:
            error_code = error.code

    if error_code is None:
        environment = projected_environment
        for case_id in CASE_IDS:
            observation = _observe_child(case_id, environment, environment_sha256)
            children.append(observation)
            if not observation["passed"]:
                error_code = observation["error_code"]
                break

    child_launches = sum(item["return_code"] is not None for item in children)
    assignments = sum(item["inner_job_assigned"] for item in children)
    memberships = sum(item["membership_before_gate"] for item in children)
    accepted = sum(
        isinstance(item["child_receipt"], dict)
        and item["child_receipt"]["status"] == "accepted"
        for item in children
    )
    rejected = sum(
        isinstance(item["child_receipt"], dict)
        and item["child_receipt"]["status"] == "rejected"
        for item in children
    )
    stderr_bytes = sum(int(item["stderr_bytes"]) for item in children)
    survivors = sum(max(0, int(item["surviving_descendants"])) for item in children)
    passed = (
        error_code is None
        and outer_membership
        and len(children) == 11
        and child_launches == 11
        and assignments == memberships == 10
        and accepted == 4
        and rejected == 7
        and stderr_bytes == 0
        and survivors == 0
        and all(item["passed"] is True for item in children)
    )
    if not passed and error_code is None:
        error_code = "child_relation"
    return {
        "schema_version": 1,
        "checkpoint_id": CHECKPOINT_ID,
        "environment_pairs": environment_pairs,
        "environment_sha256": environment_sha256,
        "outer_membership_before_gate": outer_membership,
        "children": children,
        "child_launches": child_launches,
        "inner_assignments": assignments,
        "inner_memberships_before_gate": memberships,
        "accepted_frames": accepted,
        "rejected_frames": rejected,
        "child_stderr_bytes": stderr_bytes,
        "surviving_descendants": survivors,
        "passed": passed,
        "error_code": error_code,
    }


def _validate_observation(
    observation: object, case_id: str, environment_sha256: str
) -> dict[str, object]:
    if not isinstance(observation, dict) or tuple(observation) != (
        "case_id",
        "inner_job_required",
        "inner_job_assigned",
        "membership_before_gate",
        "write_mode",
        "expected_payload_bytes",
        "expected_payload_sha256",
        "child_receipt",
        "stdout_bytes",
        "stderr_bytes",
        "return_code",
        "surviving_descendants",
        "passed",
        "error_code",
    ):
        raise ProbeError("outer_schema")
    expected = _empty_observation(case_id)
    for key in (
        "case_id",
        "inner_job_required",
        "write_mode",
        "expected_payload_bytes",
        "expected_payload_sha256",
    ):
        if observation[key] != expected[key]:
            raise ProbeError("outer_relation")
    for key in (
        "inner_job_required",
        "inner_job_assigned",
        "membership_before_gate",
        "passed",
    ):
        if type(observation[key]) is not bool:
            raise ProbeError("outer_schema")
    for key in ("stdout_bytes", "stderr_bytes"):
        if type(observation[key]) is not int or observation[key] < 0:
            raise ProbeError("outer_schema")
    if type(observation["surviving_descendants"]) is not int:
        raise ProbeError("outer_schema")
    if observation["return_code"] is not None and type(observation["return_code"]) is not int:
        raise ProbeError("outer_schema")
    error_code = observation["error_code"]
    if error_code not in PARENT_ERROR_CODES | {None}:
        raise ProbeError("outer_schema")
    receipt = observation["child_receipt"]
    if receipt is not None:
        receipt = _validate_child_receipt(receipt, case_id, environment_sha256)

    inner_required = CASE_BY_ID[case_id][1]
    if not inner_required and error_code in {
        "job_create",
        "job_limit",
        "job_assign",
        "job_membership",
        "job_query",
        "survivor",
    }:
        raise ProbeError("outer_relation")
    if observation["passed"]:
        if (
            error_code is not None
            or observation["inner_job_assigned"] is not inner_required
            or observation["membership_before_gate"] is not inner_required
            or observation["stderr_bytes"] != 0
            or observation["return_code"] != 0
            or observation["surviving_descendants"] != 0
            or receipt is None
        ):
            raise ProbeError("outer_relation")
        if (
            not _child_relation_matches(receipt, case_id)
            or observation["stdout_bytes"] != len(_canonical(receipt))
        ):
            raise ProbeError("outer_relation")
        return observation

    if error_code is None:
        raise ProbeError("outer_relation")
    total_bytes = observation["stdout_bytes"] + observation["stderr_bytes"]
    topology_complete = (
        observation["inner_job_assigned"] is inner_required
        and observation["membership_before_gate"] is inner_required
        and type(observation["return_code"]) is int
    )
    prelaunch = {"job_create", "job_limit", "child_spawn"}
    if error_code in prelaunch:
        expected_partial = (
            observation["inner_job_assigned"] is False
            and observation["membership_before_gate"] is False
            and receipt is None
            and observation["stdout_bytes"] == 0
            and observation["stderr_bytes"] == 0
            and observation["return_code"] is None
            and observation["surviving_descendants"] == 0
        )
    elif error_code == "job_assign":
        expected_partial = (
            inner_required
            and observation["inner_job_assigned"] is False
            and observation["membership_before_gate"] is False
            and receipt is None
            and observation["stdout_bytes"] == 0
            and observation["stderr_bytes"] == 0
            and type(observation["return_code"]) is int
            and observation["surviving_descendants"] >= 0
        )
    elif error_code == "job_membership":
        expected_partial = (
            inner_required
            and observation["inner_job_assigned"] is True
            and observation["membership_before_gate"] is False
            and receipt is None
            and observation["stdout_bytes"] == 0
            and observation["stderr_bytes"] == 0
            and type(observation["return_code"]) is int
            and observation["surviving_descendants"] >= 0
        )
    elif error_code == "child_relation":
        expected_partial = (
            topology_complete
            and receipt is not None
            and not _child_relation_matches(receipt, case_id)
            and observation["stdout_bytes"] == len(_canonical(receipt))
            and observation["stderr_bytes"] == 0
            and observation["return_code"] == 0
            and observation["surviving_descendants"] == 0
            and total_bytes <= MAX_OUTPUT_BYTES
        )
    elif error_code == "job_query":
        expected_partial = (
            topology_complete
            and receipt is None
            and observation["surviving_descendants"] == -1
        )
    elif error_code == "survivor":
        expected_partial = (
            topology_complete
            and receipt is None
            and observation["stderr_bytes"] == 0
            and observation["return_code"] == 0
            and observation["surviving_descendants"] > 0
            and total_bytes <= MAX_OUTPUT_BYTES
        )
    elif error_code == "output_cap":
        expected_partial = (
            topology_complete
            and receipt is None
            and observation["surviving_descendants"] >= 0
            and total_bytes > MAX_OUTPUT_BYTES
        )
    elif error_code == "child_stderr":
        expected_partial = (
            topology_complete
            and receipt is None
            and observation["stderr_bytes"] > 0
            and observation["surviving_descendants"] >= 0
            and total_bytes <= MAX_OUTPUT_BYTES
        )
    elif error_code == "child_exit":
        expected_partial = (
            topology_complete
            and receipt is None
            and observation["stderr_bytes"] == 0
            and observation["return_code"] != 0
            and observation["surviving_descendants"] >= 0
            and total_bytes <= MAX_OUTPUT_BYTES
        )
    elif error_code == "child_schema":
        expected_partial = (
            topology_complete
            and receipt is None
            and observation["stdout_bytes"] > 0
            and observation["stderr_bytes"] == 0
            and observation["return_code"] == 0
            and observation["surviving_descendants"] == 0
            and total_bytes <= MAX_OUTPUT_BYTES
        )
    else:
        expected_partial = (
            topology_complete
            and receipt is None
            and observation["surviving_descendants"] >= 0
        )
    if not expected_partial:
        raise ProbeError("outer_relation")
    return observation


def _validate_parent(
    value: object,
    environment_pairs: int,
    environment_sha256: str,
    outer_membership_before_gate: bool,
):
    if not isinstance(value, dict) or tuple(value) != (
        "schema_version",
        "checkpoint_id",
        "environment_pairs",
        "environment_sha256",
        "outer_membership_before_gate",
        "children",
        "child_launches",
        "inner_assignments",
        "inner_memberships_before_gate",
        "accepted_frames",
        "rejected_frames",
        "child_stderr_bytes",
        "surviving_descendants",
        "passed",
        "error_code",
    ):
        raise ProbeError("outer_schema")
    if (
        value["schema_version"] != 1
        or value["checkpoint_id"] != CHECKPOINT_ID
        or value["environment_pairs"] != environment_pairs
        or value["environment_sha256"] != environment_sha256
        or value["outer_membership_before_gate"] is not outer_membership_before_gate
        or not isinstance(value["children"], list)
        or len(value["children"]) > 11
        or type(value["passed"]) is not bool
        or value["error_code"] not in PARENT_ERROR_CODES | {None}
    ):
        raise ProbeError("outer_schema")
    for key in (
        "environment_pairs",
        "child_launches",
        "inner_assignments",
        "inner_memberships_before_gate",
        "accepted_frames",
        "rejected_frames",
        "child_stderr_bytes",
        "surviving_descendants",
    ):
        if type(value[key]) is not int:
            raise ProbeError("outer_schema")

    children = value["children"]
    for index, observation in enumerate(children):
        _validate_observation(observation, CASE_IDS[index], environment_sha256)
        if not observation["passed"] and index != len(children) - 1:
            raise ProbeError("outer_relation")
    launches = sum(item["return_code"] is not None for item in children)
    assignments = sum(item["inner_job_assigned"] for item in children)
    memberships = sum(item["membership_before_gate"] for item in children)
    accepted = sum(
        isinstance(item["child_receipt"], dict)
        and item["child_receipt"]["status"] == "accepted"
        for item in children
    )
    rejected = sum(
        isinstance(item["child_receipt"], dict)
        and item["child_receipt"]["status"] == "rejected"
        for item in children
    )
    stderr_bytes = sum(item["stderr_bytes"] for item in children)
    survivors = sum(max(0, item["surviving_descendants"]) for item in children)
    if (
        value["child_launches"] != launches
        or value["inner_assignments"] != assignments
        or value["inner_memberships_before_gate"] != memberships
        or value["accepted_frames"] != accepted
        or value["rejected_frames"] != rejected
        or value["child_stderr_bytes"] != stderr_bytes
        or value["surviving_descendants"] != survivors
    ):
        raise ProbeError("outer_relation")
    expected_pass = (
        value["outer_membership_before_gate"] is True
        and len(children) == 11
        and all(item["passed"] for item in children)
        and launches == 11
        and assignments == memberships == 10
        and accepted == 4
        and rejected == 7
        and stderr_bytes == 0
        and survivors == 0
    )
    if value["passed"] is not expected_pass:
        raise ProbeError("outer_relation")
    expected_error = None if expected_pass else (
        children[-1]["error_code"] if children else "outer_gate_read"
    )
    if value["error_code"] != expected_error:
        raise ProbeError("outer_relation")
    return value


def _outer_observation(environment: Mapping[str, str], pairs: int, environment_sha256: str):
    started = time.monotonic()
    body = None
    stdout = b""
    stderr = b""
    stdout_size = 0
    stderr_size = 0
    process = None
    job = None
    writer = None
    writer_errors: list[str] = []
    error_code = None
    survivors = 0
    outer_membership_before_gate = False
    writer_joined = True
    process_cleanup_ok = True
    try:
        temporary_root = _checked_scratch_directory(
            tempfile.gettempdir(), "outer_job_create"
        )
    except ProbeError as error:
        observation = {
            "body_sha256": EMPTY_SHA256,
            "body_bytes": 0,
            "stderr_bytes": 0,
            "return_code": None,
            "surviving_processes": 0,
            "body": None,
        }
        return observation, b"", error.code
    with tempfile.TemporaryDirectory(
        prefix="l2d-startup-outer-", dir=temporary_root
    ) as directory:
        root = Path(directory).resolve()
        stdout_path = root / "stdout"
        stderr_path = root / "stderr"
        try:
            _checked_scratch_directory(root, "outer_job_create")
            job = _Job(create_code="outer_job_create", limit_code="outer_job_limit")
            with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-S", "-P", str(SOURCE_PATH), "--mode", "parent"],
                        cwd=ROOT,
                        env=dict(environment),
                        stdin=subprocess.PIPE,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                    )
                except OSError as error:
                    raise ProbeError("outer_spawn") from error
                job.assign(process, "outer_job_assign")
                outer_membership_before_gate = job.contains(
                    process, "outer_job_membership"
                )
                if _time_remaining(started) == 0:
                    raise ProbeError("outer_timeout")
                writer = threading.Thread(
                    target=_write_frame,
                    args=(process.stdin, OUTER_GATE, "contiguous", writer_errors),
                    name="startup-forensics-outer-gate",
                    daemon=False,
                )
                writer.start()
                while process.poll() is None:
                    if _time_remaining(started) == 0:
                        raise ProbeError("outer_timeout")
                    if stdout_path.stat().st_size + stderr_path.stat().st_size > MAX_OUTPUT_BYTES:
                        raise ProbeError("outer_output_cap")
                    time.sleep(0.01)
                remaining = _time_remaining(started)
                writer.join(timeout=min(10.0, remaining))
                if writer.is_alive():
                    raise ProbeError(
                        "outer_timeout" if remaining == 0 else "outer_gate_write"
                    )
                if writer_errors:
                    raise ProbeError("outer_gate_write")
                if _time_remaining(started) == 0:
                    raise ProbeError("outer_timeout")
        except ProbeError as error:
            error_code = error.code
            _terminate_process(process, job)
            writer_joined = _join_writer_after_cleanup(writer, process)
        finally:
            try:
                stdout, stderr, stdout_size, stderr_size = _capture_files(
                    stdout_path, stderr_path, "outer_output_cap"
                )
            except ProbeError as error:
                if error_code is None:
                    error_code = error.code
            if stdout_size + stderr_size > MAX_OUTPUT_BYTES and error_code is None:
                error_code = "outer_output_cap"
            if job is not None:
                try:
                    survivors = job.active_processes("outer_job_query")
                except ProbeError:
                    survivors = -1
                    # A missing cleanup count invalidates every earlier partial;
                    # ``-1`` is legal only under this integrity error.
                    error_code = "outer_job_query"
                job.close()
                post_close_survivors = _post_close_survivors(process)
                if survivors >= 0:
                    survivors = max(survivors, post_close_survivors)
            process_cleanup_ok = _terminate_process(process, None)

    if not writer_joined or not process_cleanup_ok:
        # The root probe cannot return while a non-daemon writer or outer
        # process remains. Its owned scratch has already been removed.
        os._exit(1)

    if error_code is None:
        if stdout_size + stderr_size > MAX_OUTPUT_BYTES:
            error_code = "outer_output_cap"
        elif stderr_size:
            error_code = "outer_stderr"
        elif process is None or process.returncode != 0:
            error_code = "outer_exit"
        elif survivors != 0:
            error_code = "outer_survivor"
        else:
            try:
                body = _validate_parent(
                    _loads(stdout),
                    pairs,
                    environment_sha256,
                    outer_membership_before_gate,
                )
                if body["passed"] is not True:
                    error_code = "outer_relation"
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                error_code = "outer_schema"
            except ProbeError as error:
                error_code = error.code
    body_bytes = _canonical(body) if isinstance(body, dict) else b""
    observation = {
        "body_sha256": hashlib.sha256(body_bytes).hexdigest(),
        "body_bytes": stdout_size,
        "stderr_bytes": stderr_size,
        "return_code": None if process is None else process.returncode,
        "surviving_processes": survivors,
        "body": body,
    }
    return observation, body_bytes, error_code


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _runner_envelope() -> dict[str, object]:
    plan_bytes = _normalized_bytes(PLAN_PATH)
    source_bytes = _normalized_bytes(SOURCE_PATH)
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    contract_sha256 = hashlib.sha256(
        b"L2D-startup-forensics-v1/contract\0"
        + plan_bytes
        + b"\0"
        + source_bytes
    ).hexdigest()
    probe_revision = _git("rev-parse", "HEAD")
    runs = []
    bodies = []
    error_code = None
    runtime = dict(EXPECTED_RUNTIME_IDENTITY)
    try:
        _runtime_identity()
        if _git("status", "--porcelain=v1", "--untracked-files=all"):
            raise ProbeError("dirty_worktree")
        environment = _probe_environment()
        pairs, environment_sha256 = _environment_receipt(environment)
        for _index in range(2):
            observation, body_bytes, run_error = _outer_observation(
                environment, pairs, environment_sha256
            )
            runs.append(observation)
            bodies.append(body_bytes)
            if run_error is not None:
                raise ProbeError(run_error)
    except ProbeError as error:
        error_code = error.code
    runs_equal = len(bodies) == 2 and bodies[0] == bodies[1] and bodies[0] != b""
    passed = error_code is None and runs_equal and len(runs) == 2
    if not passed and error_code is None:
        error_code = "outer_relation"
    first = bodies[0] if len(bodies) > 0 else b""
    second = bodies[1] if len(bodies) > 1 else b""
    receipt_root = hashlib.sha256(
        b"L2D-startup-forensics-v1/receipt\0" + first + b"\0" + second
    ).hexdigest()
    if error_code not in RUNNER_ERROR_CODES | {None}:
        error_code = "outer_relation"
        passed = False
    return {
        "schema_version": 1,
        "checkpoint_id": CHECKPOINT_ID,
        "plan_revision": PLAN_REVISION,
        "probe_revision": probe_revision,
        "plan_sha256": plan_sha256,
        "probe_source_sha256": source_sha256,
        "contract_sha256": contract_sha256,
        "python_executable_sha256": runtime["python_executable_sha256"],
        "python_version": runtime["python_version"],
        "python_architecture": runtime["python_architecture"],
        "machine": runtime["machine"],
        "runs": runs,
        "runs_equal": runs_equal,
        "passed": passed,
        "error_code": error_code,
        "receipt_root_sha256": receipt_root,
    }


def _is_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _complete_outer_run(run: Mapping[str, object]) -> bool:
    return (
        isinstance(run["body"], dict)
        and run["body"]["passed"] is True
        and run["stderr_bytes"] == 0
        and run["return_code"] == 0
        and run["surviving_processes"] == 0
        and run["body_bytes"] + run["stderr_bytes"] <= MAX_OUTPUT_BYTES
    )


def _validate_failed_outer_run(run: Mapping[str, object], error_code: str) -> None:
    body = run["body"]
    body_bytes = run["body_bytes"]
    stderr_bytes = run["stderr_bytes"]
    return_code = run["return_code"]
    survivors = run["surviving_processes"]
    total_bytes = body_bytes + stderr_bytes
    prelaunch = {"outer_job_create", "outer_job_limit", "outer_spawn"}
    if error_code in prelaunch:
        legal = (
            body is None
            and body_bytes == 0
            and stderr_bytes == 0
            and return_code is None
            and survivors == 0
        )
    elif error_code in {"outer_job_assign", "outer_job_membership"}:
        legal = (
            body is None
            and body_bytes == 0
            and stderr_bytes == 0
            and type(return_code) is int
            and survivors >= 0
        )
    elif error_code in {"outer_gate_write", "outer_timeout"}:
        legal = (
            body is None
            and type(return_code) is int
            and survivors >= 0
        )
    elif error_code == "outer_job_query":
        legal = body is None and type(return_code) is int and survivors == -1
    elif error_code == "outer_output_cap":
        legal = (
            body is None
            and type(return_code) is int
            and survivors >= 0
            and total_bytes > MAX_OUTPUT_BYTES
        )
    elif error_code == "outer_stderr":
        legal = (
            body is None
            and type(return_code) is int
            and survivors >= 0
            and stderr_bytes > 0
            and total_bytes <= MAX_OUTPUT_BYTES
        )
    elif error_code == "outer_exit":
        legal = (
            body is None
            and type(return_code) is int
            and return_code != 0
            and survivors >= 0
            and stderr_bytes == 0
            and total_bytes <= MAX_OUTPUT_BYTES
        )
    elif error_code == "outer_schema":
        legal = (
            body is None
            and body_bytes > 0
            and return_code == 0
            and survivors == 0
            and stderr_bytes == 0
            and total_bytes <= MAX_OUTPUT_BYTES
        )
    elif error_code == "outer_relation":
        legal = (
            isinstance(body, dict)
            and body["passed"] is False
            and return_code == 0
            and survivors == 0
            and stderr_bytes == 0
            and total_bytes <= MAX_OUTPUT_BYTES
        ) or (
            body is None
            and body_bytes > 0
            and return_code == 0
            and survivors == 0
            and stderr_bytes == 0
            and total_bytes <= MAX_OUTPUT_BYTES
        )
    elif error_code == "outer_survivor":
        legal = (
            body is None
            and return_code == 0
            and survivors > 0
            and stderr_bytes == 0
            and total_bytes <= MAX_OUTPUT_BYTES
        )
    else:
        legal = False
    if not legal:
        raise ProbeError("outer_relation")


def _validate_runner_envelope(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or tuple(value) != (
        "schema_version",
        "checkpoint_id",
        "plan_revision",
        "probe_revision",
        "plan_sha256",
        "probe_source_sha256",
        "contract_sha256",
        "python_executable_sha256",
        "python_version",
        "python_architecture",
        "machine",
        "runs",
        "runs_equal",
        "passed",
        "error_code",
        "receipt_root_sha256",
    ):
        raise ProbeError("outer_relation")
    plan_bytes = _normalized_bytes(PLAN_PATH)
    source_bytes = _normalized_bytes(SOURCE_PATH)
    expected_contract = hashlib.sha256(
        b"L2D-startup-forensics-v1/contract\0"
        + plan_bytes
        + b"\0"
        + source_bytes
    ).hexdigest()
    if (
        value["schema_version"] != 1
        or value["checkpoint_id"] != CHECKPOINT_ID
        or value["plan_revision"] != PLAN_REVISION
        or value["probe_revision"] != _git("rev-parse", "HEAD")
        or value["plan_sha256"] != hashlib.sha256(plan_bytes).hexdigest()
        or value["probe_source_sha256"] != hashlib.sha256(source_bytes).hexdigest()
        or value["contract_sha256"] != expected_contract
        or value["python_executable_sha256"]
        != EXPECTED_RUNTIME_IDENTITY["python_executable_sha256"]
        or value["python_version"] != EXPECTED_RUNTIME_IDENTITY["python_version"]
        or value["python_architecture"]
        != EXPECTED_RUNTIME_IDENTITY["python_architecture"]
        or value["machine"] != EXPECTED_RUNTIME_IDENTITY["machine"]
        or not isinstance(value["runs"], list)
        or len(value["runs"]) > 2
        or type(value["runs_equal"]) is not bool
        or type(value["passed"]) is not bool
        or value["error_code"] not in RUNNER_ERROR_CODES | {None}
        or not _is_hex(value["receipt_root_sha256"], 64)
    ):
        raise ProbeError("outer_relation")
    environment = _probe_environment()
    pairs, environment_sha256 = _environment_receipt(environment)
    bodies = []
    complete_runs = []
    for run in value["runs"]:
        if not isinstance(run, dict) or tuple(run) != (
            "body_sha256",
            "body_bytes",
            "stderr_bytes",
            "return_code",
            "surviving_processes",
            "body",
        ):
            raise ProbeError("outer_relation")
        if (
            not _is_hex(run["body_sha256"], 64)
            or type(run["body_bytes"]) is not int
            or run["body_bytes"] < 0
            or type(run["stderr_bytes"]) is not int
            or run["stderr_bytes"] < 0
            or (run["return_code"] is not None and type(run["return_code"]) is not int)
            or type(run["surviving_processes"]) is not int
        ):
            raise ProbeError("outer_relation")
        if run["body"] is None:
            body_bytes = b""
            if run["body_sha256"] != EMPTY_SHA256:
                raise ProbeError("outer_relation")
        else:
            body = _validate_parent(
                run["body"], pairs, environment_sha256, True
            )
            body_bytes = _canonical(body)
            if (
                run["body_sha256"] != hashlib.sha256(body_bytes).hexdigest()
                or run["body_bytes"] != len(body_bytes)
            ):
                raise ProbeError("outer_relation")
        bodies.append(body_bytes)
        complete_runs.append(_complete_outer_run(run))
    expected_equal = len(bodies) == 2 and bodies[0] == bodies[1] and bodies[0] != b""
    if value["runs_equal"] is not expected_equal:
        raise ProbeError("outer_relation")
    expected_root = hashlib.sha256(
        b"L2D-startup-forensics-v1/receipt\0"
        + (bodies[0] if len(bodies) > 0 else b"")
        + b"\0"
        + (bodies[1] if len(bodies) > 1 else b"")
    ).hexdigest()
    if value["receipt_root_sha256"] != expected_root:
        raise ProbeError("outer_relation")
    if value["passed"]:
        if (
            value["error_code"] is not None
            or not expected_equal
            or len(value["runs"]) != 2
            or not all(complete_runs)
        ):
            raise ProbeError("outer_relation")
    else:
        error_code = value["error_code"]
        if error_code is None:
            raise ProbeError("outer_relation")
        if error_code in {"runtime_identity", "dirty_worktree"}:
            if value["runs"]:
                raise ProbeError("outer_relation")
        else:
            if not value["runs"] or not all(complete_runs[:-1]):
                raise ProbeError("outer_relation")
            if (
                error_code == "outer_relation"
                and len(value["runs"]) == 2
                and all(complete_runs)
                and not expected_equal
            ):
                pass
            else:
                if complete_runs[-1]:
                    raise ProbeError("outer_relation")
                _validate_failed_outer_run(value["runs"][-1], error_code)
    return value


def _owned_result_directory(name: str) -> Path:
    if (
        type(name) is not str
        or not name.startswith(RESULT_DIRECTORY_PREFIX)
        or Path(name).name != name
    ):
        raise ProbeError("outer_relation")
    temporary_root = _checked_scratch_directory(
        tempfile.gettempdir(), "outer_relation"
    )
    directory = (temporary_root / name).resolve()
    if directory.parent != temporary_root:
        raise ProbeError("outer_relation")
    return _checked_scratch_directory(directory, "outer_relation")


def _verify_runner_result(name: str) -> tuple[dict[str, object], Path]:
    directory = _owned_result_directory(name)
    if not directory.is_dir():
        raise ProbeError("outer_relation")
    entries = {item.name for item in directory.iterdir()}
    if entries != {"result.json", "result.json.sha256"}:
        raise ProbeError("outer_relation")
    result_path = directory / "result.json"
    sidecar_path = directory / "result.json.sha256"
    if not result_path.is_file() or not sidecar_path.is_file():
        raise ProbeError("outer_relation")
    if (
        result_path.stat().st_size > MAX_OUTPUT_BYTES
        or sidecar_path.stat().st_size != 65
    ):
        raise ProbeError("outer_relation")
    with result_path.open("rb") as handle:
        encoded = handle.read(MAX_OUTPUT_BYTES + 1)
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ProbeError("outer_relation")
    value = _validate_runner_envelope(_loads(encoded))
    if encoded != _canonical(value):
        raise ProbeError("outer_relation")
    digest = hashlib.sha256(encoded).hexdigest()
    if sidecar_path.read_text(encoding="ascii") != digest + "\n":
        raise ProbeError("outer_relation")
    return value, directory


def _discard_owned_result_directory(directory: Path) -> None:
    temporary_root = Path(tempfile.gettempdir()).resolve()
    resolved = directory.resolve()
    if (
        resolved.parent != temporary_root
        or not resolved.name.startswith(RESULT_DIRECTORY_PREFIX)
        or _inside(resolved, ROOT.resolve())
        or _inside(resolved, PRIVATE_ROOT.resolve())
    ):
        raise ProbeError("outer_relation")
    for name in ("result.json", "result.json.sha256"):
        path = resolved / name
        if path.exists():
            if not path.is_file():
                raise ProbeError("outer_relation")
            path.unlink()
    resolved.rmdir()


def _remove_verified_runner_result(name: str) -> dict[str, object]:
    value, directory = _verify_runner_result(name)
    _discard_owned_result_directory(directory)
    if directory.exists():
        raise ProbeError("outer_relation")
    return value


def _write_runner_result(value: Mapping[str, object]) -> bytes:
    value = _validate_runner_envelope(value)
    temporary_root = _checked_scratch_directory(
        tempfile.gettempdir(), "outer_relation"
    )
    directory = Path(
        tempfile.mkdtemp(prefix=RESULT_DIRECTORY_PREFIX, dir=temporary_root)
    ).resolve()
    try:
        if _owned_result_directory(directory.name) != directory:
            raise ProbeError("outer_relation")
        encoded = _canonical(value)
        result_path = directory / "result.json"
        sidecar_path = directory / "result.json.sha256"
        with result_path.open("xb") as handle:
            handle.write(encoded)
        with sidecar_path.open("xb") as handle:
            handle.write(hashlib.sha256(encoded).hexdigest().encode("ascii") + b"\n")
        observed, observed_directory = _verify_runner_result(directory.name)
        if observed != value or observed_directory != directory:
            raise ProbeError("outer_relation")
        encoded = _canonical(observed)
        _remove_verified_runner_result(directory.name)
        return encoded
    except BaseException:
        if directory.exists():
            _discard_owned_result_directory(directory)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("runner", "parent", "child"), required=True)
    parser.add_argument("--case", choices=CASE_IDS)
    args = parser.parse_args()
    if args.mode == "child":
        if args.case is None:
            raise SystemExit(2)
        sys.stdout.buffer.write(_canonical(_child_receipt(args.case)))
        return
    if args.case is not None:
        raise SystemExit(2)
    if args.mode == "parent":
        sys.stdout.buffer.write(_canonical(_parent_receipt()))
        return
    try:
        sys.stdout.buffer.write(_write_runner_result(_runner_envelope()))
    except BaseException:
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
