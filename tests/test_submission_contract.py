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
PRIOR = ROOT / "submission" / "semantic_prior.json"


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


def test_packaged_candidate_defaults_to_no_semantic_prior() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    algorithm = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchedRestartAdam"
    )
    optimize = next(
        node
        for node in algorithm.body
        if isinstance(node, ast.FunctionDef) and node.name == "optimize"
    )
    positional = optimize.args.posonlyargs + optimize.args.args
    names = [argument.arg for argument in positional]
    defaults = dict(
        zip(names[-len(optimize.args.defaults) :], optimize.args.defaults, strict=True)
    )

    assert isinstance(defaults["use_semantic_prior"], ast.Constant)
    assert defaults["use_semantic_prior"].value is False


def test_semantic_prior_has_pinned_provenance_and_support() -> None:
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))

    assert prior["format_version"] == 1
    assert prior["dataset_sha256"] == (
        "149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7"
    )
    assert prior["records"] == 11_678
    assert len(prior["key_medians"]) == 247
    assert min(prior["key_support"].values()) >= 400
    assert set(prior["property_medians"]) == {
        "angle",
        "db",
        "length",
        "mass",
        "power",
        "reflectivity",
        "tuning",
    }


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
        assert set(bundle.namelist()) == {
            "requirements.txt",
            "semantic_prior.json",
            "submission.py",
        }
        assert bundle.getinfo("submission.py").date_time == (2026, 1, 1, 0, 0, 0)
        assert all(info.create_system == 0 for info in bundle.infolist())

    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(recorded["archive_sha256"]) == 64
    assert {item["path"] for item in recorded["source_files"]} == {
        "requirements.txt",
        "semantic_prior.json",
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


def test_builder_pins_creator_platform_without_changing_evaluated_hash(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "submission.zip"
    manifest = build_archive(ROOT / "submission", archive)

    assert manifest["archive_sha256"] == (
        "4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b"
    )
    with zipfile.ZipFile(archive) as bundle:
        assert {info.create_system for info in bundle.infolist()} == {0}
