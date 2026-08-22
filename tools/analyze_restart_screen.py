#!/usr/bin/env python3
"""Outcome-blind three-way replay and analysis of restart-screen-v1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.uifo_paired.restart_analysis import summarize_restart_records
from experiments.uifo_paired.restart_posthoc_analysis import (
    analyze_restart_posthoc,
    create_restart_plots,
)
from experiments.uifo_paired.restart_reference_analysis import (
    reference_restart_screen,
)
from experiments.uifo_paired.restart_results_ingestion import (
    RESTART_SCREEN_V1_SOURCES,
    load_restart_summary_after_reproduction,
    validate_restart_screen_archive,
)
from experiments.uifo_paired.results_ingestion import (
    ExpectedSources,
    SourcePaths,
    StudyValidationError,
    ValidatedStudy,
)
from experiments.uifo_paired.results_workflow import (
    compare_restart_archived_summary,
    compare_restart_production_and_reference,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise StudyValidationError(f"refusing to write empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _inside_git_checkout(path: Path) -> bool:
    current = path.resolve()
    if not current.is_dir():
        current = current.parent
    return any((parent / ".git").exists() for parent in (current, *current.parents))


def _normalized_tables(
    study: ValidatedStudy,
    reference: dict[str, object],
    output_dir: Path,
) -> dict[str, int]:
    output_dir.mkdir(parents=False, exist_ok=False)
    run_rows = []
    history_rows = []
    target_rows = []
    for record in study.records:
        config = record["config"]
        metrics = record["metrics"]
        run_rows.append(
            {
                "run_id": record["run_id"],
                "pair_id": config["pair_id"],
                "topology_sha256": record["problem"]["topology_sha256"],
                "optimizer_seed": config["optimizer_seed"],
                "arm": config["arm"],
                "planned_run_index": config["planned_run_index"],
                "run_order_within_pair": config["run_order_within_pair"],
                "has_feasible": metrics["has_feasible"],
                "has_finite_feasible": metrics["has_finite_feasible"],
                "best_feasible_loss": metrics["best_feasible_loss"],
                "logged_calls": metrics["logged_calls"],
                "logged_candidates": metrics["logged_candidates"],
                "last_logged_time_seconds": metrics["last_logged_time_seconds"],
                "last_logged_eval_count": metrics["last_logged_eval_count"],
            }
        )
        for history_row in study.history_rows[str(record["run_id"])]:
            history_rows.append(
                {
                    "run_id": record["run_id"],
                    "pair_id": config["pair_id"],
                    "topology_sha256": record["problem"]["topology_sha256"],
                    "optimizer_seed": config["optimizer_seed"],
                    "arm": config["arm"],
                    "call_index": history_row["call_index"],
                    "candidate_index": history_row["candidate_index"],
                    "eval_count_after_call": history_row["eval_count_after_call"],
                    "time_seconds": history_row["time_seconds"],
                    "loss": history_row["loss"],
                    "sensitivity_loss": history_row["sensitivity_loss"],
                    "penalty": history_row["penalty"],
                    "is_feasible": history_row["is_feasible"],
                }
            )
        for target, hit in metrics["targets"].items():
            target_rows.append(
                {
                    "run_id": record["run_id"],
                    "pair_id": config["pair_id"],
                    "topology_sha256": record["problem"]["topology_sha256"],
                    "optimizer_seed": config["optimizer_seed"],
                    "arm": config["arm"],
                    "target_loss": target,
                    "reached": hit["time_seconds"] is not None,
                    "time_seconds": hit["time_seconds"],
                    "eval_count": hit["eval_count"],
                }
            )
    pair_rows = [
        {
            key: row[key]
            for key in (
                "pair_id",
                "optimizer_seed",
                "topology_sha256",
                "control_finite_feasible",
                "treatment_finite_feasible",
                "difference_p200_minus_p600",
                "p200_first",
                "control_eval_count",
                "treatment_eval_count",
            )
        }
        for row in reference["optimizer_seed_pair_rows"]
    ]
    topology_rows = [
        {
            "topology_sha256": row["topology_sha256"],
            "optimizer_seeds": ";".join(str(seed) for seed in row["optimizer_seeds"]),
            "replication_complete": row["replication_complete"],
            "inference_complete": row["inference_complete"],
            "p600_finite_feasible_seeds": row["control_finite_feasible_seeds"],
            "p200_finite_feasible_seeds": row["treatment_finite_feasible_seeds"],
            "mean_difference_p200_minus_p600": row[
                "mean_seed_difference_p200_minus_p600"
            ],
        }
        for row in reference["topology_differences"]
    ]
    _write_csv(output_dir / "runs.csv", run_rows)
    _write_csv(output_dir / "history_rows.csv", history_rows)
    _write_csv(output_dir / "seed_pairs.csv", pair_rows)
    _write_csv(output_dir / "topologies.csv", topology_rows)
    _write_csv(output_dir / "target_hitting.csv", target_rows)
    dictionary = {
        "format_version": 1,
        "hierarchy": (
            "run -> topology-seed pair -> topology. The primary inference unit is "
            "the topology (n=8); 32 runs and 16 seed pairs are not independent units."
        ),
        "difference_sign": "p200 minus p600; negative favors patience 200",
        "censoring": (
            "target reached=false retains null time/eval values; no finite values "
            "are imputed for censored observations"
        ),
        "tables": {
            "runs.csv": {
                "unit": "one optimizer run",
                "primary_key": "run_id",
                "rows": len(run_rows),
                "columns": {
                    "run_id": "plan-bound run identifier",
                    "pair_id": "topology-seed pair identifier",
                    "topology_sha256": "resolved topology identity digest",
                    "optimizer_seed": "repeated optimizer seed within topology",
                    "arm": "frozen patience arm",
                    "planned_run_index": "zero-based serial execution order",
                    "run_order_within_pair": "zero-based arm order within pair",
                    "has_feasible": "whether any logged candidate was feasible",
                    "has_finite_feasible": "whether any feasible loss was finite",
                    "best_feasible_loss": "minimum finite feasible logged loss",
                    "logged_calls": "number of logged objective calls",
                    "logged_candidates": "number of logged candidate evaluations",
                    "last_logged_time_seconds": "last cumulative worker time in seconds",
                    "last_logged_eval_count": "last cumulative evaluation count",
                },
            },
            "history_rows.csv": {
                "unit": (
                    "one logged candidate evaluation nested within a run; these "
                    "rows are repeated measurements and are not inference units"
                ),
                "primary_key": ["run_id", "call_index", "candidate_index"],
                "rows": len(history_rows),
                "columns": {
                    "run_id": "parent run identifier",
                    "pair_id": "parent topology-seed pair identifier",
                    "topology_sha256": "parent topology identity digest",
                    "optimizer_seed": "parent optimizer seed",
                    "arm": "parent frozen patience arm",
                    "call_index": "zero-based objective-call index",
                    "candidate_index": "zero-based candidate index within call",
                    "eval_count_after_call": "cumulative evaluations after the call",
                    "time_seconds": "cumulative worker time at the call",
                    "loss": "logged candidate objective loss",
                    "sensitivity_loss": "logged sensitivity component",
                    "penalty": "logged feasibility penalty component",
                    "is_feasible": "candidate feasibility flag",
                },
            },
            "seed_pairs.csv": {
                "unit": "one topology and optimizer-seed paired contrast",
                "primary_key": "pair_id",
                "rows": len(pair_rows),
                "columns": {
                    "pair_id": "topology-seed pair identifier",
                    "optimizer_seed": "repeated optimizer seed",
                    "topology_sha256": "resolved topology identity digest",
                    "control_finite_feasible": "p600 produced a finite feasible loss",
                    "treatment_finite_feasible": "p200 produced a finite feasible loss",
                    "difference_p200_minus_p600": (
                        "best feasible loss difference; negative favors p200"
                    ),
                    "p200_first": "whether p200 ran before p600 in the pair",
                    "control_eval_count": "p600 terminal cumulative evaluations",
                    "treatment_eval_count": "p200 terminal cumulative evaluations",
                },
            },
            "topologies.csv": {
                "unit": "one topology after averaging both optimizer seeds",
                "primary_key": "topology_sha256",
                "rows": len(topology_rows),
                "columns": {
                    "topology_sha256": "resolved topology identity digest",
                    "optimizer_seeds": "semicolon-separated repeated seeds",
                    "replication_complete": "both frozen optimizer seeds are present",
                    "inference_complete": "both seed contrasts are finite/comparable",
                    "p600_finite_feasible_seeds": "p600 finite-feasible seed count",
                    "p200_finite_feasible_seeds": "p200 finite-feasible seed count",
                    "mean_difference_p200_minus_p600": (
                        "mean of the two seed differences; primary topology effect"
                    ),
                },
            },
            "target_hitting.csv": {
                "unit": "one run and target-loss combination",
                "primary_key": ["run_id", "target_loss"],
                "rows": len(target_rows),
                "columns": {
                    "run_id": "parent run identifier",
                    "pair_id": "parent topology-seed pair identifier",
                    "topology_sha256": "parent topology identity digest",
                    "optimizer_seed": "parent optimizer seed",
                    "arm": "parent frozen patience arm",
                    "target_loss": "predeclared loss threshold",
                    "reached": "whether the run reached the threshold",
                    "time_seconds": "first-hit cumulative seconds; null when censored",
                    "eval_count": "first-hit cumulative evaluations; null when censored",
                },
            },
        },
    }
    _write_json(output_dir / "data_dictionary.json", dictionary)
    return {
        "runs": len(run_rows),
        "history_rows": len(history_rows),
        "seed_pairs": len(pair_rows),
        "topologies": len(topology_rows),
        "target_hitting": len(target_rows),
    }


def _support_statement(posthoc: dict[str, object]) -> dict[str, str]:
    effect = posthoc["effect_sizes"]
    mean_ci = effect["mean_bootstrap_ci_95"]
    median_ci = effect["median_bootstrap_ci_95"]
    topology_ready_targets = []
    for target, payload in posthoc["target_hitting"].items():
        if payload["topology_inference_ready"]:
            topology_ready_targets.append(target)
    return {
        "average_final_loss": (
            "exploratory support"
            if float(effect["mean_difference"]) < 0 and float(mean_ci["upper"]) < 0
            else "not supported by this screen"
        ),
        "median_topology": (
            "exploratory support"
            if float(effect["median_difference"]) < 0
            and float(median_ci["upper"]) < 0
            else "not supported by this screen"
        ),
        "finite_feasibility_rate": (
            "evaluate from the explicit p200-only/p600-only/both/neither counts; "
            "no equivalence claim"
        ),
        "target_hitting_time": (
            "complete topology-level contrasts are available for targets "
            f"{', '.join(topology_ready_targets)}; interpret the complete target "
            "family above, with time and evaluation counts kept separate"
            if topology_ready_targets
            else "no target has a complete topology-level comparison; the full "
            "censoring pattern is reported above"
        ),
        "topology_dependent_benefit": (
            "descriptive heterogeneity only; no post-hoc subgroup is confirmed"
        ),
    }


def _next_gate_text(decision: dict[str, object]) -> str:
    action = decision.get("action")
    if action == "retain_patience_600":
        return (
            "Retain the packaged patience-600/no-prior candidate and stop tuning "
            "patience on this development panel. The recommended next evidence "
            "gate is a separately frozen, no-prior-only submission-like evaluation "
            "on the existing disjoint panel, with a hard cost cap and no algorithm "
            "changes during the gate."
        )
    if action == "plan_untouched_submission_like_gate":
        return (
            "The screen authorizes planning, but not launching, an untouched "
            "submission-like gate on the existing disjoint panel. Freeze that "
            "design and its candidate-alignment rule before changing the packaged "
            "submission or spending additional GPU budget."
        )
    raise StudyValidationError(f"unsupported frozen restart action: {action!r}")


def _report(
    study: ValidatedStudy,
    production: dict[str, object],
    reference: dict[str, object],
    posthoc: dict[str, object],
    agreement: dict[str, object],
) -> str:
    decision = production["predeclared_decision"]
    ci = production["topology_bootstrap_mean_difference_ci_95"]
    sign_flip = posthoc["exact_mean_sign_flip"]
    sign = posthoc["exact_direction_sign_test"]
    wilcoxon = posthoc["wilcoxon_signed_rank_sensitivity"]
    support = _support_statement(posthoc)
    seed = posthoc["seed_consistency"]
    drift = posthoc["drift_diagnostics"]
    heterogeneity = posthoc["heterogeneity"]
    loo = posthoc["leave_one_topology_out"]
    loo_mean = [float(row["mean_difference"]) for row in loo]
    loo_median = [float(row["median_difference"]) for row in loo]
    loo_wins = [int(row["p200_wins"]) for row in loo]
    loo_p90 = [float(row["p90_regret"]) for row in loo]
    target_lines = []
    for target, payload in posthoc["target_hitting"].items():
        counts = payload["seed_pair_outcomes"]
        time_ci = payload["topology_bootstrap_time_log10_ratio_ci_95"]
        eval_ci = payload["topology_bootstrap_eval_log10_ratio_ci_95"]
        time_result = (
            "not estimable"
            if time_ci is None
            else (
                f"mean log10(p200/p600) "
                f"{payload['topology_macro_mean_log10_time_ratio']}, 95% "
                f"topology-bootstrap CI [{time_ci['lower']}, {time_ci['upper']}]"
            )
        )
        eval_result = (
            "not estimable"
            if eval_ci is None
            else (
                f"mean log10(p200/p600) "
                f"{payload['topology_macro_mean_log10_eval_ratio']}, 95% "
                f"topology-bootstrap CI [{eval_ci['lower']}, {eval_ci['upper']}]"
            )
        )
        target_lines.append(
            f"- `{target}`: both {counts['both_reached']}, p200-only "
            f"{counts['p200_only']}, p600-only {counts['p600_only']}, neither "
            f"{counts['neither_reached']}; complete topology comparison "
            f"`{str(payload['topology_inference_ready']).lower()}` "
            f"({payload['finite_comparable_topologies']}/8 topologies). Time: "
            f"{time_result}. Evaluations: {eval_result}."
        )
    feasibility = reference["feasibility_pair_outcomes"]
    next_gate = _next_gate_text(decision)
    return f"""# Learn2Design patience-200 development screen (generated)

This private generated report contains aggregates only. Raw histories, candidate
arrays, logs, GPU identifiers, secrets, and provider-local paths are excluded.

## 1. Integrity validation

- Authenticated archive, sidecar, package manifest, and external plan hashes.
- Validated {study.integrity['entries']} ZIP members, paths/types, bounded sizes
  and compression ratios, all CRCs, 32 pickle-free NPZ histories, 32 records,
  32 configs, and 64 worker logs.
- Exact hierarchy: 32 runs, 16 paired optimizer seeds, 8 topology inference
  units. All workers and the serial session completed cleanly.
- The archived summary remained sealed until both raw-data replays agreed.

## 2. Frozen-policy reproduction

Production replay, independent history-only reference calculation, and archived
summary status: `{agreement['status']}` at absolute/relative tolerance
`{agreement['absolute_tolerance']}`.

- p200 wins/ties/losses: {production['wins_ties_losses']['p200_wins']}/{production['wins_ties_losses']['ties']}/{production['wins_ties_losses']['p200_losses']}.
- Mean p200-minus-p600 loss: `{production['topology_macro_mean_difference']}`.
- Median difference: `{production['topology_macro_median_difference']}`.
- p90 regret: `{production['topology_p90_regret']}`.
- Frozen topology-bootstrap CI: `[{ci[0]}, {ci[1]}]`, seed
  `{production['bootstrap_seed']}`, {production['bootstrap_resamples']} resamples.
- Feasibility pairs: both {feasibility['both_finite_feasible']}, p200-only
  {feasibility['p200_only_finite_feasible']}, p600-only
  {feasibility['p600_only_finite_feasible']}, neither
  {feasibility['neither_finite_feasible']}.
- Frozen status/action: `{decision['status']}` / `{decision['action']}`.

The frozen action is not revised by any post-hoc p-value.

## 3. Phase 3 — post-hoc sensitivity and exploratory analysis

The exact sign-flip statistic is the absolute topology-level mean difference.
All {sign_flip['assignments_enumerated']} sign assignments were enumerated;
two-sided p=`{sign_flip['two_sided_p_value']}`. The separate exact sign test of
direction only gave p=`{sign['two_sided_p_value']}`.

Wilcoxon signed-rank sensitivity gave p=`{wilcoxon['two_sided_p_value']}` using
`{wilcoxon['method']}`, with {wilcoxon['zeros']} zeros and
{wilcoxon['absolute_rank_ties']} absolute-rank ties. This analysis assumes a
symmetric topology-difference distribution and is not a decision rule.

The exploratory mean bootstrap interval is
`[{posthoc['effect_sizes']['mean_bootstrap_ci_95']['lower']},
{posthoc['effect_sizes']['mean_bootstrap_ci_95']['upper']}]`; the median interval
is `[{posthoc['effect_sizes']['median_bootstrap_ci_95']['lower']},
{posthoc['effect_sizes']['median_bootstrap_ci_95']['upper']}]`. Complete topology
blocks were resampled, preserving both seeds.

Seed 19 and seed 23 mean differences were
`{seed['seed_19_mean_difference']}` and `{seed['seed_23_mean_difference']}`.
Across topologies, {seed['pattern_counts'].get('both_help', 0)} had both seeds
favor p200, {seed['pattern_counts'].get('both_harm', 0)} had both favor p600,
and {seed['pattern_counts'].get('opposite_signs', 0)} had opposite signs. The
cross-seed correlation was `{seed['pearson_seed_19_vs_23']['statistic']}`
(exploratory p=`{seed['pearson_seed_19_vs_23']['two_sided_p_value']}`).

Leave-one-topology-out means ranged from `{min(loo_mean)}` to `{max(loo_mean)}`;
medians from `{min(loo_median)}` to `{max(loo_median)}`; p200 wins from
`{min(loo_wins)}` to `{max(loo_wins)}`; and p90 regret from `{min(loo_p90)}` to
`{max(loo_p90)}`. Descriptive topology effects ranged from
`{heterogeneity['minimum_difference']}` to `{heterogeneity['maximum_difference']}`;
{heterogeneity['topologies_with_at_least_0_05_help']} topologies showed at least
0.05 benefit and {heterogeneity['topologies_with_at_least_0_05_harm']} at least
0.05 harm. These are not confirmed subgroups.

### Censor-aware target hitting

{chr(10).join(target_lines)}

Time and evaluation counts are analyzed separately. Unreached targets remain
censored; no arbitrary finite times are substituted.

### Narrow conclusions

- Average final feasible loss improvement: {support['average_final_loss']}.
- Median-topology improvement: {support['median_topology']}.
- Finite-feasibility improvement: not supported; both arms were finite-feasible
  in all 16 pairs, with no discordance. This is not an equivalence claim.
- Earlier useful-target attainment: {support['target_hitting_time']}.
- Meaningful topology-dependent benefits: {support['topology_dependent_benefit']}.

Serial run order and session time each had Spearman rho
`{drift['serial_run_order']['statistic']}` (p=`{drift['serial_run_order']['two_sided_p_value']}`).
The mean within-topology arm-first contrast was
`{drift['arm_first_order']['mean_contrast']}` (exact sign-flip p=
`{drift['arm_first_order']['exact_mean_sign_flip']['two_sided_p_value']}`). The
mean topology log10 evaluation-count ratio was
`{drift['evaluation_throughput']['mean_topology_log10_evaluation_ratio_p200_over_p600']}`
with bootstrap interval
`[{drift['evaluation_throughput']['bootstrap_mean_log10_evaluation_ratio_ci_95']['lower']},
{drift['evaluation_throughput']['bootstrap_mean_log10_evaluation_ratio_ci_95']['upper']}]`.
These are drift diagnostics only. Failure to reject is not evidence of
equivalence or non-inferiority; no equivalence margin was preregistered.

## 4. Limitations

There are eight independent development topologies and only two optimizer seeds
within each. This panel was used to screen patience and cannot serve as its own
confirmation. Censoring can prevent topology-level target-time inference. Any
apparent subgroup or order association is post-hoc.

## 5. Recommended next experiment

{next_gate} This workflow does not launch that evaluation, confirmation-v1, or
any GPU experiment.
"""


def _render_html(markdown_text: str) -> str:
    import markdown

    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>Learn2Design restart screen</title>
<style>
body {{ color:#1f2933; font:16px/1.55 system-ui,sans-serif; max-width:980px;
margin:2rem auto; padding:0 1.5rem 4rem; }}
h1,h2,h3 {{ color:#102a43; line-height:1.25; }}
h1 {{ border-bottom:3px solid #2a9d8f; padding-bottom:.4rem; }}
h2 {{ border-bottom:1px solid #bcccdc; padding-bottom:.25rem; margin-top:2rem; }}
code {{ background:#f0f4f8; padding:.1rem .25rem; border-radius:3px; }}
</style></head><body>{body}</body></html>"""


def run_analysis(
    sources: SourcePaths,
    output: Path,
    *,
    expected: ExpectedSources = RESTART_SCREEN_V1_SOURCES,
) -> dict[str, object]:
    """Execute the sealed validation/replay/unlock sequence and write outputs."""
    output = output.resolve()
    if _inside_git_checkout(output):
        raise StudyValidationError("generated output must be outside every Git checkout")
    if output.exists():
        raise StudyValidationError(f"refusing to overwrite output directory: {output}")

    # Gate 1: authenticate bytes, inspect both ZIP layers, and revalidate records.
    study = validate_restart_screen_archive(sources, expected)

    # Gate 2: two genuinely distinct frozen-policy calculations. Phase 3 tests
    # remain disabled while the archived summary is sealed.
    production_frozen = summarize_restart_records(
        study.records,
        study.configs,
        include_exploratory=False,
    )
    reference_frozen = reference_restart_screen(
        study,
        include_exploratory=False,
    )
    replay = compare_restart_production_and_reference(
        production_frozen,
        reference_frozen,
    )

    # Gate 3: unlock and compare only the frozen fields before any post-hoc work.
    archived = load_restart_summary_after_reproduction(study, replay)
    frozen_agreement = compare_restart_archived_summary(
        production_frozen,
        reference_frozen,
        archived,
    )

    # Gate 4: only the matched frozen three-way receipt enables Phase 3.
    production = summarize_restart_records(study.records, study.configs)
    reference = reference_restart_screen(study)
    agreement = compare_restart_archived_summary(
        production,
        reference,
        archived,
        include_exploratory=True,
    )
    posthoc = analyze_restart_posthoc(
        study,
        production,
        reference,
        frozen_agreement=frozen_agreement,
    )
    output.mkdir(parents=True, exist_ok=False)
    normalized_counts = _normalized_tables(study, reference, output / "normalized")
    plots = create_restart_plots(posthoc, output)
    validation = {
        "format_version": 1,
        "source_sha256": study.source_hashes,
        "plan_id": study.manifest["plan_id"],
        "project_revision": study.manifest["project_revision"],
        "study_profile": study.manifest["configuration"]["study_profile"],
        "integrity": {
            **study.integrity,
            "summary_content_opened_after_raw_replays": True,
        },
    }
    comparison = {
        "format_version": 1,
        "three_way_status": "matched",
        **agreement,
    }
    _write_json(output / "validation.json", validation)
    _write_json(output / "frozen_replay_receipt.json", frozen_agreement)
    _write_json(output / "production_replay.json", production)
    _write_json(output / "independent_reference.json", reference)
    _write_json(output / "archived_summary.json", archived)
    _write_json(output / "three_way_comparison.json", comparison)
    _write_json(output / "exploratory_analysis.json", posthoc)
    _write_csv(output / "leave_one_topology_out.csv", posthoc["leave_one_topology_out"])
    _write_csv(
        output / "drift_diagnostics.csv",
        posthoc["drift_diagnostics"]["topology_rows"],
    )
    report = _report(study, production, reference, posthoc, agreement)
    (output / "analysis_report.md").write_text(report, encoding="utf-8")
    (output / "analysis_report.html").write_text(
        _render_html(report), encoding="utf-8"
    )
    handoff = {
        "format_version": 1,
        "output_directory": ".",
        "validation": "validation.json",
        "frozen_replay_receipt": "frozen_replay_receipt.json",
        "three_way_comparison": "three_way_comparison.json",
        "report": "analysis_report.md",
        "rendered_report": "analysis_report.html",
        "normalized_data_dictionary": "normalized/data_dictionary.json",
        "normalized_row_counts": normalized_counts,
        "figures": [path.name for path in plots],
    }
    _write_json(output / "handoff.json", handoff)
    return handoff


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and analyze the sealed restart-screen-v1 package."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--checksum", type=Path)
    parser.add_argument("--package-manifest", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    sources = SourcePaths(
        archive=archive,
        checksum=(
            args.checksum.resolve()
            if args.checksum
            else archive.with_suffix(archive.suffix + ".sha256")
        ),
        package_manifest=(
            args.package_manifest.resolve()
            if args.package_manifest
            else archive.with_suffix(archive.suffix + ".manifest.json")
        ),
        plan=args.plan.resolve(),
    )
    output = (
        args.output.resolve()
        if args.output
        else archive.parent / f"{archive.stem}-analysis"
    )
    try:
        handoff = run_analysis(sources, output)
    except StudyValidationError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {"resolved_output_directory": str(output), **handoff},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
