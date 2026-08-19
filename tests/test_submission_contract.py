from __future__ import annotations

import ast
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tools.build_submission import build_archive


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "submission" / "submission.py"


def test_exactly_one_algorithm_subclass_and_no_forbidden_import() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    subclasses = []
    imported_roots = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any(
                isinstance(base, ast.Name) and base.id == "OptimizationAlgorithm"
                for base in node.bases
            ):
                subclasses.append(node.name)
        elif isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert subclasses == ["BatchedRestartAdam"]
    assert "differometor" not in imported_roots


def test_builder_creates_flat_deterministic_archive(tmp_path: Path) -> None:
    archive = tmp_path / "submission.zip"
    manifest = tmp_path / "manifest.json"
    command = [
        sys.executable,
        str(ROOT / "tools" / "build_submission.py"),
        "--source",
        str(ROOT / "submission"),
        "--output",
        str(archive),
        "--manifest",
        str(manifest),
    ]
    subprocess.run(command, check=True, cwd=ROOT)

    first_bytes = archive.read_bytes()
    subprocess.run(command, check=True, cwd=ROOT)
    assert archive.read_bytes() == first_bytes

    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {"requirements.txt", "submission.py"}
        assert bundle.getinfo("submission.py").date_time == (2026, 1, 1, 0, 0, 0)

    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(recorded["archive_sha256"]) == 64
    assert {item["path"] for item in recorded["source_files"]} == {
        "requirements.txt",
        "submission.py",
    }


def test_builder_normalizes_cross_platform_text_newlines(tmp_path: Path) -> None:
    source_lf = tmp_path / "lf"
    source_crlf = tmp_path / "crlf"
    source_lf.mkdir()
    source_crlf.mkdir()
    code = (
        "from dfbench import OptimizationAlgorithm\n\n"
        "class Candidate(OptimizationAlgorithm):\n"
        "    pass\n"
    )
    (source_lf / "submission.py").write_bytes(code.encode())
    (source_crlf / "submission.py").write_bytes(code.replace("\n", "\r\n").encode())
    (source_lf / "requirements.txt").write_bytes(b"# none\n")
    (source_crlf / "requirements.txt").write_bytes(b"# none\r\n")

    archive_lf = tmp_path / "lf.zip"
    archive_crlf = tmp_path / "crlf.zip"
    build_archive(source_lf, archive_lf)
    build_archive(source_crlf, archive_crlf)

    assert archive_lf.read_bytes() == archive_crlf.read_bytes()
