"""Create a path-free source lock after packaging, without reading outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROFILE = "submission-like-screen-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inside_git(path: Path) -> bool:
    resolved = path.resolve()
    current = resolved if resolved.is_dir() else resolved.parent
    return any((parent / ".git").exists() for parent in (current, *current.parents))


def build_source_lock(
    *,
    archive: Path,
    checksum: Path,
    package_manifest: Path,
    plan: Path,
    terminal_attempt_receipt: Path,
) -> dict[str, object]:
    paths = (archive, checksum, package_manifest, plan, terminal_attempt_receipt)
    if any(not path.is_file() for path in paths):
        raise ValueError("all source-lock inputs must be regular files")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("source-lock inputs must have distinct basenames")
    try:
        package = json.loads(package_manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("package manifest is malformed") from error
    if not isinstance(package, dict):
        raise ValueError("package manifest must be a JSON object")
    plan_id = package.get("study_plan_id")
    revision = package.get("study_project_revision")
    if not isinstance(plan_id, str) or len(plan_id) != 16:
        raise ValueError("package manifest plan ID is invalid")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("package manifest project revision is invalid")
    return {
        "format_version": 1,
        "study_profile": PROFILE,
        "plan_id": plan_id,
        "project_revision": revision,
        "files": {
            path.name: {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in paths
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--terminal-attempt-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite source lock: {args.output}")
    if inside_git(args.output):
        raise ValueError("source lock must be written outside every Git checkout")
    payload = build_source_lock(
        archive=args.archive,
        checksum=args.checksum,
        package_manifest=args.package_manifest,
        plan=args.plan,
        terminal_attempt_receipt=args.terminal_attempt_receipt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"source_lock_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
