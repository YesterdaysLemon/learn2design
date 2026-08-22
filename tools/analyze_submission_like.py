"""Authenticate and replay the sealed submission-like screen outside Git."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from experiments.uifo_paired.results_ingestion import SourcePaths
from experiments.uifo_paired.submission_like_analysis import (
    summarize_submission_like_records,
)
from experiments.uifo_paired.submission_like_evidence import (
    compare_submission_like_archived_summary,
    compare_submission_like_replays,
)
from experiments.uifo_paired.submission_like_reference_analysis import (
    reference_submission_like_screen,
)
from experiments.uifo_paired.submission_like_results_ingestion import (
    authenticate_submission_like_source_lock,
    load_submission_like_summary_after_reproduction,
    submission_like_package_is_complete,
    validate_submission_like_archive,
    validate_submission_like_terminal_partial,
)


def _inside_git(path: Path) -> bool:
    resolved = path.resolve()
    current = resolved if resolved.is_dir() else resolved.parent
    return any((parent / ".git").exists() for parent in (current, *current.parents))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty normalized table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    *,
    sources: SourcePaths,
    source_lock: Path,
    expected_source_lock_sha256: str,
    terminal_attempt_receipt: Path,
    output: Path,
) -> dict[str, object]:
    if _inside_git(output):
        raise ValueError("analysis output must remain outside every Git checkout")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"analysis output is not empty: {output}")

    # Gate 1: external authentication, ZIP safety, exact hierarchy, and raw
    # history metric recomputation. summary.json is not parsed here.
    expected = authenticate_submission_like_source_lock(
        source_lock,
        expected_source_lock_sha256=expected_source_lock_sha256,
        sources=sources,
        terminal_attempt_receipt=terminal_attempt_receipt,
    )
    if not submission_like_package_is_complete(sources, expected):
        partial = validate_submission_like_terminal_partial(
            sources,
            expected=expected,
            terminal_attempt_receipt=terminal_attempt_receipt,
        )
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "terminal_partial_integrity.json", partial)
        return {
            "status": "not_evaluable",
            "study_profile": "submission-like-screen-v1",
            "plan_id": expected.plan_id,
            "project_revision": expected.project_revision,
            "action": "retain_candidate_attempt_not_evaluable",
            "summary_content_opened": False,
            "output": str(output.resolve()),
        }
    study = validate_submission_like_archive(
        sources,
        expected=expected,
        terminal_attempt_receipt=terminal_attempt_receipt,
    )

    # Gate 2: distinct record-based production and history-first no-import
    # calculations while the archived summary remains sealed.
    production = summarize_submission_like_records(study.records, study.configs)
    reference = reference_submission_like_screen(study)
    replay = compare_submission_like_replays(production, reference)

    # Gate 3: only the matched raw replays unlock summary.json.
    archived = load_submission_like_summary_after_reproduction(study, replay)
    archived_agreement = compare_submission_like_archived_summary(
        production, archived
    )

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "integrity.json", study.integrity)
    _write_json(output / "production_replay.json", production)
    _write_json(output / "independent_reference.json", reference)
    _write_json(output / "raw_replay_agreement.json", replay)
    _write_json(output / "archived_summary.json", archived)
    _write_json(output / "archived_summary_agreement.json", archived_agreement)

    run_rows = []
    for topology in reference["topology_rows"]:
        topology_hash = topology["topology_sha256"]
        topology_name = topology["topology"]
        for seed in (29, 31):
            outcome = next(
                row
                for row in reference["run_rows"]
                if row["topology"] == topology_name
                and row["optimizer_seed"] == seed
            )
            run_rows.append(
                {
                    "run_id": outcome["run_id"],
                    "topology_sha256": topology_hash,
                    "optimizer_seed": seed,
                    "physical_feasible": outcome["physical_feasible"],
                    "finite_feasible": outcome["finite_feasible"],
                    "best_feasible_loss": outcome["best_feasible_loss"],
                }
            )
    topology_rows = [
        {
            "topology_sha256": row["topology_sha256"],
            "optimizer_seeds": "29;31",
            "complete": row["complete"],
            "finite_feasible": row["finite_feasible"],
            "topology_mean_best_feasible_loss": row[
                "topology_mean_best_feasible_loss"
            ],
            "absolute_seed_gap": row["absolute_seed_gap"],
        }
        for row in reference["topology_rows"]
    ]
    target_rows = []
    for target, result in reference["target_hitting"].items():
        for row in result["topology_rows"]:
            target_rows.append(
                {
                    "target_loss": target,
                    "topology_sha256": next(
                        item["topology_sha256"]
                        for item in reference["topology_rows"]
                        if item["topology"] == row["topology"]
                    ),
                    "category": row["category"],
                    "seed_29_time_seconds": row["seed_hits"]["29"][
                        "time_seconds"
                    ],
                    "seed_29_eval_count": row["seed_hits"]["29"]["eval_count"],
                    "seed_31_time_seconds": row["seed_hits"]["31"][
                        "time_seconds"
                    ],
                    "seed_31_eval_count": row["seed_hits"]["31"]["eval_count"],
                }
            )
    _write_csv(output / "runs.csv", run_rows)
    _write_csv(output / "topologies.csv", topology_rows)
    _write_csv(output / "target_hits.csv", target_rows)
    _write_json(
        output / "data_dictionary.json",
        {
            "inference_unit": "topology (n=10)",
            "optimizer_seeds": "repeated measurements nested within topology",
            "runs.csv": "One row per run; not an independent inference table.",
            "topologies.csv": "Primary normalized inference table, one row per topology.",
            "target_hits.csv": (
                "Censor-aware topology/target rows; null hit fields remain right-censored."
            ),
            "raw_histories_exported": False,
        },
    )
    return {
        "status": "validated",
        "study_profile": "submission-like-screen-v1",
        "plan_id": expected.plan_id,
        "project_revision": expected.project_revision,
        "raw_replay": replay["status"],
        "archived_summary": archived_agreement["status"],
        "output": str(output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--expected-source-lock-sha256", required=True)
    parser.add_argument("--terminal-attempt-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_analysis(
        sources=SourcePaths(
            archive=args.archive,
            checksum=args.checksum,
            package_manifest=args.package_manifest,
            plan=args.plan,
        ),
        source_lock=args.source_lock,
        expected_source_lock_sha256=args.expected_source_lock_sha256,
        terminal_attempt_receipt=args.terminal_attempt_receipt,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
