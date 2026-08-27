from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytestmark = pytest.mark.integration

from experiments.uifo_paired.coverage_analysis import summarize_coverage_records
from experiments.uifo_paired.coverage_evidence import (
    compare_coverage_archived_summary,
    compare_coverage_replays,
)
from experiments.uifo_paired.coverage_reference_analysis import (
    CoverageReferenceError,
    reference_coverage_screen,
)
from experiments.uifo_paired.results_ingestion import (
    SourcePaths,
    StudyValidationError,
    ValidatedStudy,
)
from tests.test_coverage_robustness import _plan, _records


ROOT = Path(__file__).parents[1]


def _study() -> ValidatedStudy:
    plan = _plan()
    records, configs = _records(plan)
    histories = {}
    for record in records:
        run_id = str(record["run_id"])
        histories[run_id] = [
            {
                "call_index": 0,
                "candidate_index": 0,
                "eval_count_after_call": int(
                    record["objective_accounting"]["eval_count"]
                ),
                "time_seconds": 1_199.0,
                "loss": float(record["metrics"]["best_feasible_loss"]),
                "sensitivity_loss": 0.0,
                "penalty": 0.0,
                "is_feasible": True,
            }
        ]
    return ValidatedStudy(
        sources=SourcePaths(
            archive=Path("coverage.zip"),
            checksum=Path("coverage.zip.sha256"),
            package_manifest=Path("coverage.package.json"),
            plan=Path("coverage.plan.json"),
        ),
        source_hashes={},
        archive_members=(),
        plan=plan,
        manifest={},
        package_state={},
        session={},
        configs=configs,
        records=records,
        history_rows=histories,
        integrity={"summary_content_opened": False},
    )


def test_reference_recomputes_history_first_and_matches_production() -> None:
    study = _study()
    production = summarize_coverage_records(study.records, study.configs)
    reference = reference_coverage_screen(study)

    agreement = compare_coverage_replays(production, reference, study=study)

    receipt = agreement.as_dict()
    assert receipt == {
        "status": "matched",
        "runs_compared": 48,
        "topology_values_compared": 12,
        "optimizer_seed_pairs_compared": 24,
        "frozen_criteria_compared": 13,
        "study_identity_sha256": receipt["study_identity_sha256"],
    }
    assert len(receipt["study_identity_sha256"]) == 64
    assert reference["predeclared_decision"]["status"] == "passed"


def test_reference_rejects_record_metric_drift() -> None:
    study = _study()
    study.records[0]["metrics"]["best_feasible_loss"] = 99.0

    with pytest.raises(CoverageReferenceError, match="best-loss drift"):
        reference_coverage_screen(study)


def test_reference_rejects_raw_draw_and_summary_seal_drift() -> None:
    raw_drift = _study()
    treatment = next(
        record
        for record in raw_drift.records
        if record["config"]["arm"] == "coverage_balanced"
    )
    treatment["raw_suffix_parameter_hashes"][0] = "f" * 64
    with pytest.raises(CoverageReferenceError, match="raw draw mismatch"):
        reference_coverage_screen(raw_drift)

    opened = _study()
    opened.integrity["summary_content_opened"] = True
    with pytest.raises(CoverageReferenceError, match="summary sealed"):
        reference_coverage_screen(opened)


def test_archived_summary_comparison_is_fail_closed() -> None:
    study = _study()
    production = summarize_coverage_records(study.records, study.configs)
    reference = reference_coverage_screen(study)
    archived = copy.deepcopy(reference)

    assert compare_coverage_archived_summary(
        production, reference, archived
    )["status"] == "matched"

    archived["topology_macro_mean_difference"] = 0.0
    with pytest.raises(StudyValidationError, match="numeric mismatch"):
        compare_coverage_archived_summary(production, reference, archived)


def test_reference_module_has_no_project_imports() -> None:
    path = (
        ROOT
        / "experiments"
        / "uifo_paired"
        / "coverage_reference_analysis.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    project_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (
            node.module or ""
        ).startswith("experiments.uifo_paired"):
            project_imports.append(
                (node.module, {alias.name for alias in node.names})
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("experiments.uifo_paired"):
                    project_imports.append((alias.name, set()))

    assert project_imports == []
