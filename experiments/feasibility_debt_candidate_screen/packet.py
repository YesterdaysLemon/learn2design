"""Binary worker-packet parsing and closed parse-failure receipts."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from .canonical import canonical_json_bytes, sha256_bytes, write_receipt
from .contract import STUDY_ID


MAX_PACKET_BYTES = 1024 * 1024


class PacketError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class BoundedTextCapture(io.TextIOBase):
    """In-memory text sink whose UTF-8 byte cap is enforced on each write."""

    def __init__(self, *, max_bytes: int) -> None:
        super().__init__()
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("text capture cap must be positive")
        self.max_bytes = max_bytes
        self._parts: list[str] = []
        self._size = 0

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError("text capture accepts strings only")
        size = len(value.encode("utf-8"))
        if self._size + size > self.max_bytes:
            raise PacketError(
                "captured_output_too_large",
                "captured scientific output exceeds the byte cap",
            )
        self._parts.append(value)
        self._size += size
        return len(value)

    def getvalue(self) -> str:
        return "".join(self._parts)


def _pairs_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PacketError("duplicate_key", "worker packet contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PacketError("nonfinite_number", f"worker packet contains {value}")


def parse_worker_packet(
    raw_stdout: bytes,
    *,
    raw_stderr: bytes,
    returncode: int,
    timed_out: bool,
    expected_keys: set[str],
) -> dict[str, Any]:
    if timed_out:
        raise PacketError("timeout", "worker exceeded the kill timeout")
    if returncode != 0:
        raise PacketError("nonzero_exit", "worker exited unsuccessfully")
    if raw_stderr:
        raise PacketError("stderr_not_empty", "worker emitted forbidden stderr")
    if len(raw_stdout) > MAX_PACKET_BYTES:
        raise PacketError("stdout_too_large", "worker packet exceeds byte cap")
    if not raw_stdout:
        raise PacketError("stdout_empty", "worker emitted no packet")
    if raw_stdout.startswith(b"\xef\xbb\xbf") or raw_stdout.endswith((b"\n", b"\r")):
        raise PacketError("noncanonical_framing", "worker packet framing is invalid")
    try:
        value = json.loads(
            raw_stdout.decode("utf-8"),
            object_pairs_hook=_pairs_object,
            parse_constant=_reject_constant,
        )
    except PacketError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PacketError("invalid_json", "worker packet is not strict JSON") from error
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise PacketError("schema_mismatch", "worker packet key schema mismatch")
    if canonical_json_bytes(value) != raw_stdout:
        raise PacketError("noncanonical_json", "worker packet bytes are not canonical")
    return value


def write_closed_failure_receipt(
    path: Path,
    *,
    raw_stdout: bytes,
    raw_stderr: bytes,
    returncode: int | None,
    timed_out: bool,
    error_code: str,
    stdout_sha256: str | None = None,
    stdout_size_bytes: int | None = None,
    stderr_sha256: str | None = None,
    stderr_size_bytes: int | None = None,
) -> str:
    """Persist hashes and sizes only; never include raw process output."""
    stdout_digest = sha256_bytes(raw_stdout) if stdout_sha256 is None else stdout_sha256
    stderr_digest = sha256_bytes(raw_stderr) if stderr_sha256 is None else stderr_sha256
    stdout_size = len(raw_stdout) if stdout_size_bytes is None else stdout_size_bytes
    stderr_size = len(raw_stderr) if stderr_size_bytes is None else stderr_size_bytes
    for digest in (stdout_digest, stderr_digest):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(token not in "0123456789abcdef" for token in digest)
        ):
            raise PacketError("invalid_capture_digest", "capture digest is invalid")
    for size in (stdout_size, stderr_size):
        if type(size) is not int or size < 0:
            raise PacketError("invalid_capture_size", "capture size is invalid")
    return write_receipt(
        path,
        study_id=STUDY_ID,
        receipt_type="worker_failure",
        payload={
            "stdout_sha256": stdout_digest,
            "stdout_size_bytes": stdout_size,
            "stderr_sha256": stderr_digest,
            "stderr_size_bytes": stderr_size,
            "returncode": returncode,
            "timed_out": timed_out,
            "error_code": error_code,
            "raw_output_included": False,
        },
    )
