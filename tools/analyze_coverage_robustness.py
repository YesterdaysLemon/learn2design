"""Authenticate and replay a sealed H100 coverage screen outside Git."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from experiments.uifo_paired.coverage_analysis import summarize_coverage_records
from experiments.uifo_paired.coverage_evidence import (
    compare_coverage_archived_summary,
    compare_coverage_replays,
)
from experiments.uifo_paired.coverage_reference_analysis import (
    reference_coverage_screen,
)
from experiments.uifo_paired.coverage_results_ingestion import (
    authenticate_coverage_source_lock,
    coverage_package_is_complete,
    load_coverage_summary_after_reproduction,
    validate_coverage_archive,
    validate_coverage_terminal_partial,
)
from experiments.uifo_paired.results_ingestion import SourcePaths
from experiments.uifo_paired.results_ingestion import sha256_path
from tools.create_submission_like_source_lock import inside_git


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_analysis(
    *,
    archive: Path,
    checksum: Path,
    package_manifest: Path,
    plan: Path,
    terminal_attempt_receipt: Path,
    source_lock: Path,
    expected_source_lock_sha256: str,
    output: Path,
    summary_release: Path | None = None,
    provider_billing_receipt: Path | None = None,
) -> dict[str, object]:
    """Run authenticate, sealed replay, unlock, and archive-agreement gates."""
    if inside_git(output):
        raise ValueError("private coverage analysis output must remain outside Git")
    output.mkdir(parents=True, exist_ok=False)
    sources = SourcePaths(
        archive=archive,
        checksum=checksum,
        package_manifest=package_manifest,
        plan=plan,
    )
    expected = authenticate_coverage_source_lock(
        source_lock,
        expected_source_lock_sha256=expected_source_lock_sha256,
        sources=sources,
        terminal_attempt_receipt=terminal_attempt_receipt,
        provider_billing_receipt=provider_billing_receipt,
    )
    if not coverage_package_is_complete(sources, expected):
        partial = validate_coverage_terminal_partial(
            sources,
            expected=expected,
            terminal_attempt_receipt=terminal_attempt_receipt,
        )
        partial["expected_source_lock_sha256"] = expected_source_lock_sha256
        partial["source_lock_sha256"] = sha256_path(source_lock)
        _write_json(output / "terminal_partial.json", partial)
        return partial

    study = validate_coverage_archive(
        sources,
        expected=expected,
        terminal_attempt_receipt=terminal_attempt_receipt,
    )
    production = summarize_coverage_records(study.records, study.configs)
    reference = reference_coverage_screen(study)
    replay = compare_coverage_replays(production, reference, study=study)
    archived = load_coverage_summary_after_reproduction(
        study, replay, summary_release_path=summary_release
    )
    agreement = compare_coverage_archived_summary(
        production, reference, archived
    )
    receipt = {
        "status": "validated",
        "study_profile": expected.study_profile,
        "plan_id": expected.plan_id,
        "project_revision": expected.project_revision,
        "expected_source_lock_sha256": expected_source_lock_sha256,
        "source_lock_sha256": sha256_path(source_lock),
        "authenticated_source_hashes": study.source_hashes,
        "candidate_package_evidence": study.plan["configuration"][
            "candidate_package_evidence"
        ],
        "archive_integrity": study.integrity,
        "replay_agreement": replay.as_dict(),
        "archived_summary_agreement": agreement,
        "predeclared_decision": production["predeclared_decision"],
        "summary_content_opened_after_replay": True,
    }
    _write_json(output / "production_replay.json", production)
    _write_json(output / "independent_reference.json", reference)
    _write_json(output / "archived_summary.json", archived)
    _write_json(output / "validation_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--terminal-attempt-receipt", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--expected-source-lock-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-release", type=Path)
    parser.add_argument("--provider-billing-receipt", type=Path)
    args = parser.parse_args()
    receipt = run_analysis(
        archive=args.archive,
        checksum=args.checksum,
        package_manifest=args.package_manifest,
        plan=args.plan,
        terminal_attempt_receipt=args.terminal_attempt_receipt,
        source_lock=args.source_lock,
        expected_source_lock_sha256=args.expected_source_lock_sha256,
        output=args.output,
        summary_release=args.summary_release,
        provider_billing_receipt=args.provider_billing_receipt,
    )
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
