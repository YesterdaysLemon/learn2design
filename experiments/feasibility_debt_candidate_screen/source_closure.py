"""Path-free source closure used by both preflight and scored workers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

from .canonical import sha256_file
from .contract import (
    PRIOR_PANEL_SHA256,
    ROUND1_ARCHIVE_SHA256,
    ROUND1_MANIFEST_SHA256,
    ROUND1_MEMBER_SHA256,
)


class SourceClosureError(RuntimeError):
    pass


def round1_virtual_sources(
    archive_path: Path, manifest_path: Path
) -> dict[str, bytes]:
    if sha256_file(archive_path) != ROUND1_ARCHIVE_SHA256:
        raise SourceClosureError("Round-1 archive digest mismatch")
    if sha256_file(manifest_path) != ROUND1_MANIFEST_SHA256:
        raise SourceClosureError("Round-1 manifest digest mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if [info.filename for info in infos] != list(ROUND1_MEMBER_SHA256):
            raise SourceClosureError("Round-1 archive member schema mismatch")
        if any(info.is_dir() or info.flag_bits & 0x1 for info in infos):
            raise SourceClosureError("Round-1 archive member is invalid")
        values = {name: archive.read(name) for name in ROUND1_MEMBER_SHA256}
    if any(
        hashlib.sha256(values[name]).hexdigest() != expected
        for name, expected in ROUND1_MEMBER_SHA256.items()
    ):
        raise SourceClosureError("Round-1 archive member digest mismatch")
    return {f"round1_zip::{name}": value for name, value in values.items()}


def logical_source_closure(
    repository_root: Path,
    round1_archive: Path,
    round1_manifest: Path,
) -> dict[str, Path | bytes]:
    package = repository_root / "experiments/feasibility_debt_candidate_screen"
    sources: dict[str, Path | bytes] = {
        path.relative_to(repository_root).as_posix(): path
        for path in sorted(package.glob("*.py"))
    }
    for relative in (
        "tools/build_feasibility_debt_candidate_panel.py",
        "submission/submission.py",
        "experiments/candidates/feasibility_debt_clock_v3.py",
        "experiments/uifo_paired/optimizer_settings.py",
        "tools/build_topology_panels.py",
        "pyproject.toml",
        "uv.lock",
    ):
        sources[relative] = repository_root / relative
    for name in PRIOR_PANEL_SHA256:
        relative = f"experiments/uifo_paired/panels/{name}"
        sources[relative] = repository_root / relative
    sources["round1_zip"] = round1_archive
    sources["round1_manifest"] = round1_manifest
    sources.update(round1_virtual_sources(round1_archive, round1_manifest))
    return dict(sorted(sources.items()))
