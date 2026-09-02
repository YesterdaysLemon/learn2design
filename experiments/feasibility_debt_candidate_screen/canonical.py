"""Canonical, append-only receipt primitives for the candidate screen."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


class ReceiptError(RuntimeError):
    """A receipt, sidecar, or append-only boundary is invalid."""


RECEIPT_KEYS = {"payload", "receipt_type", "schema_version", "study_id"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize using the frozen path-free JSON byte contract."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ReceiptError("value is not strict canonical JSON") from error


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt_value(study_id: str, receipt_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(study_id, str) or not study_id:
        raise ReceiptError("study_id must be a non-empty string")
    if not isinstance(receipt_type, str) or not receipt_type:
        raise ReceiptError("receipt_type must be a non-empty string")
    if not isinstance(payload, dict):
        raise ReceiptError("receipt payload must be an object")
    return {
        "payload": payload,
        "receipt_type": receipt_type,
        "schema_version": 1,
        "study_id": study_id,
    }


def sidecar_bytes(digest: str, basename: str) -> bytes:
    if SHA256_RE.fullmatch(digest) is None:
        raise ReceiptError("sidecar digest is invalid")
    if not basename or Path(basename).name != basename or "\n" in basename:
        raise ReceiptError("sidecar basename is invalid")
    return f"{digest}  {basename}\n".encode("ascii")


def _exclusive_write(path: Path, content: bytes) -> None:
    if path.exists():
        raise ReceiptError(f"append-only target already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_write_bytes(path: Path, content: bytes) -> None:
    """Public append-only, fsynced byte write for generated evidence."""
    _exclusive_write(path, content)


def write_receipt(
    path: Path,
    *,
    study_id: str,
    receipt_type: str,
    payload: dict[str, Any],
) -> str:
    """Write a receipt and fixed sibling sidecar exactly once."""
    value = receipt_value(study_id, receipt_type, payload)
    content = canonical_json_bytes(value)
    digest = sha256_bytes(content)
    sidecar = path.with_name(path.name + ".sha256")
    if path.exists() or sidecar.exists():
        raise ReceiptError("receipt or sidecar already exists")
    _exclusive_write(path, content)
    try:
        _exclusive_write(sidecar, sidecar_bytes(digest, path.name))
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return digest


def parse_canonical_json(content: bytes) -> Any:
    if content.startswith(b"\xef\xbb\xbf") or content.endswith((b"\n", b"\r")):
        raise ReceiptError("canonical JSON has a BOM or trailing newline")
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError("receipt is not valid UTF-8 JSON") from error
    if canonical_json_bytes(value) != content:
        raise ReceiptError("JSON bytes are not canonical")
    return value


def read_receipt(
    path: Path,
    *,
    expected_study_id: str,
    expected_receipt_type: str,
    expected_payload_keys: set[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """Authenticate canonical bytes, sidecar, envelope, and payload schema."""
    content = path.read_bytes()
    digest = sha256_bytes(content)
    expected_sidecar = sidecar_bytes(digest, path.name)
    sidecar_path = path.with_name(path.name + ".sha256")
    if not sidecar_path.is_file() or sidecar_path.read_bytes() != expected_sidecar:
        raise ReceiptError("receipt sidecar mismatch")
    value = parse_canonical_json(content)
    if not isinstance(value, dict) or set(value) != RECEIPT_KEYS:
        raise ReceiptError("receipt envelope schema mismatch")
    if value.get("schema_version") != 1:
        raise ReceiptError("receipt schema version mismatch")
    if value.get("study_id") != expected_study_id:
        raise ReceiptError("receipt study ID mismatch")
    if value.get("receipt_type") != expected_receipt_type:
        raise ReceiptError("receipt type mismatch")
    payload = value.get("payload")
    if not isinstance(payload, dict):
        raise ReceiptError("receipt payload is not an object")
    if expected_payload_keys is not None and set(payload) != expected_payload_keys:
        raise ReceiptError("receipt payload schema mismatch")
    return payload, digest


def assert_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ReceiptError(f"{label} must be a lowercase SHA-256")
    return value
