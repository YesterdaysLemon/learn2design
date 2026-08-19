"""Build and validate a deterministic Learn2Design submission archive."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path


REQUIRED_FILES = ("submission.py", "requirements.txt")
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
UPSTREAM_REFERENCE = "d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c"
TEXT_SUFFIXES = {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_bytes(path: Path) -> bytes:
    """Return stable bytes for text while leaving weights/binaries untouched."""
    if path.suffix.lower() in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8")
        return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return path.read_bytes()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_submission_source(source_dir: Path) -> list[Path]:
    missing = [name for name in REQUIRED_FILES if not (source_dir / name).is_file()]
    if missing:
        raise ValueError(f"missing required root files: {', '.join(missing)}")

    source_path = source_dir / "submission.py"
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    subclasses = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = {
            base.id
            for base in node.bases
            if isinstance(base, ast.Name)
        }
        if "OptimizationAlgorithm" in base_names:
            subclasses.append(node.name)
    if len(subclasses) != 1:
        raise ValueError(
            "submission.py must define exactly one direct OptimizationAlgorithm "
            f"subclass; found {subclasses}"
        )

    forbidden_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            forbidden_imports.extend(
                alias.name for alias in node.names if alias.name.split(".")[0] == "differometor"
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "differometor":
                forbidden_imports.append(node.module)
    if forbidden_imports:
        raise ValueError("direct differometor imports are forbidden by the rules")

    files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    return files


def build_archive(source_dir: Path, output_path: Path) -> dict[str, object]:
    files = validate_submission_source(source_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(source_dir).as_posix()
            info = zipfile.ZipInfo(relative, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, archive_bytes(path))

    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())
        for required in REQUIRED_FILES:
            if required not in names:
                raise AssertionError(f"archive root is missing {required}")
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise AssertionError("archive contains an unsafe path")

    project_root = source_dir.resolve().parent
    revision = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip() or None
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    return {
        "archive": output_path.name,
        "archive_sha256": sha256(output_path),
        "created_utc": datetime.now(UTC).isoformat(),
        "project_revision": revision,
        "source_files": [
            {
                "path": path.relative_to(source_dir).as_posix(),
                "sha256": bytes_sha256(archive_bytes(path)),
                "size_bytes": len(archive_bytes(path)),
            }
            for path in files
        ],
        "upstream_reference": UPSTREAM_REFERENCE,
        "working_tree_dirty": dirty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("submission"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/generated/submission.zip"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/generated/submission.manifest.json"),
    )
    args = parser.parse_args()

    manifest = build_archive(args.source, args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"built {args.output} ({manifest['archive_sha256']})")


if __name__ == "__main__":
    main()
