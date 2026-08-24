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
from experiments.uifo_paired.submission_like_posthoc_analysis import (
    analyze_submission_like_posthoc,
    create_submission_like_plots,
    safe_submission_like_posthoc,
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


def _render_html(markdown_text: str) -> str:
    import markdown

    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>Learn2Design submission-like screen</title>
<style>
body {{ color:#1f2933; font:16px/1.55 system-ui,sans-serif; max-width:980px;
margin:2rem auto; padding:0 1.5rem 4rem; }}
h1,h2,h3 {{ color:#102a43; line-height:1.25; }}
h1 {{ border-bottom:3px solid #2563eb; padding-bottom:.4rem; }}
h2 {{ border-bottom:1px solid #bcccdc; padding-bottom:.25rem; margin-top:2rem; }}
code {{ background:#f0f4f8; padding:.1rem .25rem; border-radius:3px; }}
table {{ border-collapse:collapse; width:100%; }}
th,td {{ border:1px solid #d9e2ec; padding:.45rem; text-align:left; }}
</style></head><body>{body}</body></html>"""


def _fmt(value: object) -> str:
    if value is None:
        return "not estimable"
    return format(float(value), ".8g")


def _report(
    study,
    production: dict[str, object],
    posthoc: dict[str, object],
    archived_agreement: dict[str, object],
) -> str:
    decision = production["predeclared_decision"]
    ci = production["topology_bootstrap_mean_loss_ci_95"]
    descriptive = posthoc["descriptive_loss"]
    median_ci = descriptive["exploratory_median_bootstrap_ci_95"]
    seed = posthoc["seed_consistency"]
    drift = posthoc["serial_drift"]
    throughput = posthoc["evaluation_throughput"]
    trajectory = posthoc["trajectory_alignment"]
    evaluation_checkpoints = trajectory["evaluation_aligned"]["checkpoints"]
    wall_time_checkpoints = trajectory["wall_time_aligned"]["checkpoints"]
    evaluation_final = next(
        row
        for row in reversed(evaluation_checkpoints)
        if row["complete_topologies"] == 10
    )
    wall_time_final = next(
        row
        for row in reversed(wall_time_checkpoints)
        if row["complete_topologies"] == 10
    )
    axis_comparison = trajectory["axis_comparison"]
    mirrored = drift["mirrored_topology_contrasts"]
    loo = posthoc["leave_one_topology_out"]
    loo_ranges = loo["ranges"]
    target_lines = []
    for target in sorted(
        posthoc["target_hitting"].values(),
        key=lambda item: float(item["target_loss"]),
        reverse=True,
    ):
        run_counts = target["run_categories"]
        topology_counts = target["topology_categories"]
        target_lines.append(
            f"- `{format(float(target['target_loss']), '.12g')}`: runs reached "
            f"{run_counts['reached']}/20; topology blocks both/one/neither "
            f"{topology_counts['both_seeds_reached']}/"
            f"{topology_counts['one_seed_reached']}/"
            f"{topology_counts['neither_seed_reached']}. Conditional "
            f"both-reached-topology median time "
            f"`{_fmt(target['both_reached_topology_time_seconds_median'])}` "
            f"seconds and median evaluations "
            f"`{_fmt(target['both_reached_topology_eval_count_median'])}`; "
            f"seed-29-only/seed-31-only topology counts "
            f"{target['seed_29_only_topologies']}/"
            f"{target['seed_31_only_topologies']}. Unreached runs retain their "
            "observed censor bounds."
        )
    gap_loss = drift["pair_gap_runs_vs_later_minus_earlier_loss"]
    actual_gap_loss = drift[
        "actual_start_gap_vs_later_minus_earlier_loss"
    ]
    gap_throughput = drift["pair_gap_runs_vs_log10_throughput_ratio"]
    loss_throughput = throughput[
        "loss_contrast_vs_log10_throughput_contrast"
    ]
    pearson = seed["pearson_across_topologies"]
    spearman = seed["spearman_across_topologies"]
    return f"""# Learn2Design submission-like A100 screen (generated)

This private generated report contains aggregates and anonymous topology labels
only. Raw histories, topology strings, candidate arrays, logs, run identifiers,
GPU identifiers, secrets, balances, and provider-local paths are excluded.

## 1. Integrity validation

- Authenticated the source lock before parsing it, then verified the exact ZIP,
  SHA-256 sidecar contents, package manifest, reviewed plan, and terminal-attempt
  receipt.
- Validated {study.integrity['entries']} ZIP members, safe paths/types, bounded
  entry and total sizes, compression ratios, all CRCs, 20 pickle-free NPZ
  histories, 20 records, 20 configs, and 40 worker logs.
- Recomputed physical/finite feasibility, best loss, target hits, timing, and
  evaluation counts from histories while `summary.json` remained sealed.
- The production record replay and the independent history-first evaluator
  matched before the archived summary was opened.

## 2. Frozen-policy reproduction

Production, independent reference, and archived summary status:
`{archived_agreement['status']}` at absolute/relative tolerance
`{archived_agreement['absolute_tolerance']}`.

- Complete runs/topologies: {production['completed_runs']}/20 and
  {production['complete_topologies']}/10.
- Physical/finite-feasible runs: {production['physical_feasible_runs']}/20 and
  {production['finite_feasible_runs']}/20.
- Topology mean best finite feasible loss:
  `{_fmt(production['topology_arithmetic_mean_best_feasible_loss'])}`.
- Median and linear p90: `{_fmt(production['topology_median_best_feasible_loss'])}`
  and `{_fmt(production['topology_p90_best_feasible_loss'])}`.
- Frozen topology-block bootstrap mean interval:
  `[{_fmt(ci[0])}, {_fmt(ci[1])}]`, seed `{production['bootstrap_seed']}`,
  {production['bootstrap_resamples']} resamples.
- p90 absolute within-topology seed gap:
  `{_fmt(production['topology_p90_absolute_seed_gap'])}`.
- Frozen status/action: `{decision['status']}` /
  `{decision['action']}`; all five operational criteria passed.

This pass authorizes final package/evidence review only. It is not evidence of
leaderboard competitiveness, does not reopen patience 600, and does not change
the packaged candidate.

## 3. Post-hoc sensitivity and exploratory reliability analysis

The exploratory topology-bootstrap median interval is
`[{_fmt(median_ci['lower'])}, {_fmt(median_ci['upper'])}]`. Complete topology
blocks were resampled; the two optimizer seeds remain repeated measurements.

Seed-29 and seed-31 mean losses were
`{_fmt(seed['seed_29_mean_best_feasible_loss'])}` and
`{_fmt(seed['seed_31_mean_best_feasible_loss'])}`. Seed 29 was lower on
{seed['seed_29_lower']} topologies, seed 31 on {seed['seed_31_lower']}, with
{seed['ties']} ties. Across-topology Pearson r was
`{_fmt(pearson['statistic'])}` (exploratory p=`{_fmt(pearson['two_sided_p_value'])}`)
and Spearman rho was `{_fmt(spearman['statistic'])}`
(p=`{_fmt(spearman['two_sided_p_value'])}`). These correlations and p-values
diagnose seed consistency; they are not promotion criteria.

Leave-one-topology-out means ranged from
`{_fmt(loo_ranges['topology_mean_best_feasible_loss']['minimum'])}` to
`{_fmt(loo_ranges['topology_mean_best_feasible_loss']['maximum'])}`, medians
from `{_fmt(loo_ranges['topology_median_best_feasible_loss']['minimum'])}` to
`{_fmt(loo_ranges['topology_median_best_feasible_loss']['maximum'])}`, p90 loss
from `{_fmt(loo_ranges['topology_p90_best_feasible_loss']['minimum'])}` to
`{_fmt(loo_ranges['topology_p90_best_feasible_loss']['maximum'])}`, and p90 seed
gap from `{_fmt(loo_ranges['topology_p90_absolute_seed_gap']['minimum'])}` to
`{_fmt(loo_ranges['topology_p90_absolute_seed_gap']['maximum'])}`. Each omission
keeps nine topology units and does not recompute the frozen action.

### Censor-aware target attainment

{chr(10).join(target_lines)}

Time and evaluation counts are kept separate. Conditional both-reached
summaries are selection-biased and descriptive; censored observations are not
converted into arbitrary finite values.

### Serial order and throughput

Across ten topology blocks, planned pair gap versus later-minus-earlier loss
had Spearman rho `{_fmt(gap_loss['statistic'])}` (exploratory p=
`{_fmt(gap_loss['two_sided_p_value'])}`); actual run-start gap versus the loss
contrast had rho `{_fmt(actual_gap_loss['statistic'])}`
(p=`{_fmt(actual_gap_loss['two_sided_p_value'])}`). Pair gap versus log10
throughput contrast had rho `{_fmt(gap_throughput['statistic'])}`
(p=`{_fmt(gap_throughput['two_sided_p_value'])}`). Within mirrored topology
blocks, the mean later-minus-earlier loss was
`{_fmt(mirrored['mean_later_minus_earlier_loss'])}`; later runs were lower/higher/tied
on {mirrored['later_lower']}/{mirrored['later_higher']}/{mirrored['ties']}
topologies. Mean log10 later/earlier throughput was
`{_fmt(throughput['mean_log10_later_over_earlier_throughput'])}`; topology-macro
evaluations per second were
`{_fmt(throughput['topology_macro_evaluations_per_second'])}`. Loss contrast
versus log10 throughput contrast had rho
`{_fmt(loss_throughput['statistic'])}`
(p=`{_fmt(loss_throughput['two_sided_p_value'])}`).

These are drift diagnostics only. Seed 29 is the first sweep and seed 31 the
second, so seed and sweep phase are not identifiable. Strict timezone-aware
run timestamps were validated within the completed session and actual start
gaps include controller and compilation intervals. This one-arm screen has no
arm-first contrast. No causal drift claim is made.

### Matched-resource trajectory diagnostic

The history-only diagnostic is now complete. At the final common evaluation
checkpoint of `{_fmt(evaluation_final['checkpoint'])}` evaluations, the
topology-mean seed-31-minus-seed-29 loss contrast was
`{_fmt(evaluation_final['topology_mean_seed_31_minus_seed_29_loss'])}`; seed 29
was lower on {evaluation_final['seed_29_lower']} topologies and seed 31 on
{evaluation_final['seed_31_lower']}. At the final common wall-time checkpoint
of `{_fmt(wall_time_final['checkpoint'])}` seconds, the corresponding contrast
was `{_fmt(wall_time_final['topology_mean_seed_31_minus_seed_29_loss'])}` with
seed-29/seed-31 lower counts {wall_time_final['seed_29_lower']}/
{wall_time_final['seed_31_lower']}.

The contrast direction agreed across the evaluation and wall-time views at
{axis_comparison['same_direction_fractions']} of
{axis_comparison['comparable_fractions']} comparable progress fractions. Their
mean absolute contrast difference was
`{_fmt(axis_comparison['mean_absolute_contrast_difference'])}`. Positive
contrasts mean the second sweep was worse. Persistence after matching
evaluation count is inconsistent with a throughput-only explanation, but seed
and sweep phase are still perfectly confounded. This is a diagnostic lead, not
a causal finding or a new promotion rule.

## 4. Limitations

There are ten independent topology units and only two repeated optimizer seeds.
The 1,200-second budget is shorter than the official competition budget. This
single-arm study has no optimizer comparator, and failure to reach a target is
right-censoring rather than a finite time. The operational pass cannot establish
equivalence, non-inferiority, or expected leaderboard rank.

## 5. Recommended next evidence gate

The history-only seed-divergence diagnostic is complete. It rules out a simple
evaluation-throughput explanation but cannot distinguish random-seed
sensitivity from sweep/session order. Finish package review, then formulate one
pre-result search-robustness change and evaluate it on a new disjoint panel.
Do not reuse this screen as confirmation or launch `confirmation-v1`.
"""


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

    # Gate 4: descriptive reliability work begins only after the frozen
    # production/reference/archive agreement is complete.
    posthoc = analyze_submission_like_posthoc(
        study,
        production,
        reference,
        agreement=archived_agreement,
    )
    safe_posthoc = safe_submission_like_posthoc(posthoc)

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "integrity.json", study.integrity)
    _write_json(output / "production_replay.json", production)
    _write_json(output / "independent_reference.json", reference)
    _write_json(output / "raw_replay_agreement.json", replay)
    _write_json(output / "archived_summary.json", archived)
    _write_json(output / "archived_summary_agreement.json", archived_agreement)
    _write_json(output / "private_posthoc_diagnostics.json", posthoc)
    _write_json(output / "posthoc_analysis.json", safe_posthoc)

    run_rows = []
    posthoc_runs = {
        (row["topology_sha256"], row["optimizer_seed"]): row
        for row in posthoc["run_rows"]
    }
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
                    "planned_run_index": posthoc_runs[
                        (topology_hash, seed)
                    ]["planned_run_index"],
                    "terminal_time_seconds": posthoc_runs[
                        (topology_hash, seed)
                    ]["terminal_time_seconds"],
                    "terminal_eval_count": posthoc_runs[
                        (topology_hash, seed)
                    ]["terminal_eval_count"],
                    "evaluations_per_second": posthoc_runs[
                        (topology_hash, seed)
                    ]["evaluations_per_second"],
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
    for target, result in posthoc["target_hitting"].items():
        for row in result["topology_rows"]:
            seed_29_hit = row["seed_hits"]["29"]
            seed_31_hit = row["seed_hits"]["31"]
            target_rows.append(
                {
                    "target_loss": target,
                    "topology_label": row["topology_label"],
                    "topology_sha256": row["topology_sha256"],
                    "category": row["category"],
                    "seed_29_event_reached": seed_29_hit["event_reached"],
                    "seed_29_observed_or_censor_time_seconds": seed_29_hit[
                        "observed_or_censor_time_seconds"
                    ],
                    "seed_29_observed_or_censor_eval_count": seed_29_hit[
                        "observed_or_censor_eval_count"
                    ],
                    "seed_31_event_reached": seed_31_hit["event_reached"],
                    "seed_31_observed_or_censor_time_seconds": seed_31_hit[
                        "observed_or_censor_time_seconds"
                    ],
                    "seed_31_observed_or_censor_eval_count": seed_31_hit[
                        "observed_or_censor_eval_count"
                    ],
                }
            )
    _write_csv(output / "runs.csv", run_rows)
    _write_csv(output / "topologies.csv", topology_rows)
    _write_csv(output / "target_hits.csv", target_rows)
    _write_csv(
        output / "leave_one_topology_out.csv",
        posthoc["leave_one_topology_out"],
    )
    _write_csv(output / "drift_diagnostics.csv", posthoc["topology_rows"])
    _write_csv(
        output / "trajectory_diagnostics.csv",
        posthoc["trajectory_alignment"]["private_topology_checkpoint_rows"],
    )
    _write_json(
        output / "data_dictionary.json",
        {
            "inference_unit": "topology (n=10)",
            "optimizer_seeds": "repeated measurements nested within topology",
            "runs.csv": "One row per run; not an independent inference table.",
            "topologies.csv": "Primary normalized inference table, one row per topology.",
            "target_hits.csv": (
                "Private topology/target rows with event flags and observed-or-censor "
                "time/evaluation values; no censor value is treated as a hit."
            ),
            "leave_one_topology_out.csv": (
                "Ten n=9 sensitivity rows; the frozen action is not recomputed."
            ),
            "drift_diagnostics.csv": (
                "Ten paired topology rows; seed and sweep phase are confounded."
            ),
            "trajectory_diagnostics.csv": (
                "Private topology/checkpoint rows at matched evaluation counts and "
                "matched wall times; missing feasible values are not imputed."
            ),
            "posthoc_analysis.json": (
                "Allowlisted safe aggregates only; no row identifiers or hashes."
            ),
            "private_posthoc_diagnostics.json": (
                "Private normalized rows outside Git; never a committed report."
            ),
            "raw_histories_exported": False,
        },
    )
    plot_paths = create_submission_like_plots(posthoc, output)
    report = _report(study, production, safe_posthoc, archived_agreement)
    (output / "analysis_report.md").write_text(report, encoding="utf-8")
    (output / "analysis_report.html").write_text(
        _render_html(report), encoding="utf-8"
    )
    _write_json(
        output / "handoff.json",
        {
            "format_version": 1,
            "study_profile": "submission-like-screen-v1",
            "inference_unit": "topology (n=10)",
            "report": "analysis_report.md",
            "rendered_report": "analysis_report.html",
            "data_dictionary": "data_dictionary.json",
            "figures": [path.relative_to(output).as_posix() for path in plot_paths],
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
        "frozen_action": production["predeclared_decision"]["action"],
        "figures": len(plot_paths),
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
