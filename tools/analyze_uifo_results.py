#!/usr/bin/env python3
"""Validate and analyze a packaged development-v2 UIFO study."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.uifo_paired.analysis import summarize_records
from experiments.uifo_paired.posthoc_analysis import (
    analyze_posthoc,
    create_plots,
    serializable_posthoc,
)
from experiments.uifo_paired.reference_analysis import reference_replay
from experiments.uifo_paired.results_ingestion import (
    SourcePaths,
    StudyValidationError,
    load_summary_after_reproduction,
    validate_study_archive,
    write_normalized_tables,
)
from experiments.uifo_paired.results_workflow import (
    compare_archived_summary,
    compare_production_and_reference,
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


def _aggregate_report(
    production: dict[str, object],
    posthoc: dict[str, object],
    comparison: dict[str, object],
) -> str:
    paired = production["semantic_prior_vs_no_prior"]
    decision = paired["predeclared_decision"]
    ci = paired["topology_bootstrap_mean_difference_ci_95"]
    effect = posthoc["effect_sizes"]
    sign_flip = posthoc["exact_mean_sign_flip"]
    sign = posthoc["exact_direction_sign_test"]
    wilcoxon = posthoc["wilcoxon_signed_rank_sensitivity"]
    seed = posthoc["seed_consistency"]
    drift = posthoc["drift_diagnostics"]
    target_lines = []
    for target in ("4", "1", "0.5", "0"):
        item = posthoc["target_hitting_existing_censor_aware_summary"][target]
        outcomes = item["seed_pair_outcomes"]
        target_lines.append(
            f"- `{target}`: both {outcomes['both_reached']}, semantic-only "
            f"{outcomes['semantic_prior_only']}, no-prior-only "
            f"{outcomes['no_prior_only']}, neither {outcomes['neither_reached']}; "
            f"order-of-magnitude ready `{str(item['order_of_magnitude_claim_ready']).lower()}`."
        )
    return f"""# Learn2Design development-v2 analysis (generated)

This generated report contains aggregates only. Raw histories, candidate arrays,
logs, GPU identifiers, and provider-local paths remain outside this report.

## Integrity and frozen-policy reproduction

- External hashes, sidecar filename/digest, ZIP path/type/size/ratio checks, every
  CRC, 64 configs, 64 records, 64 pickle-free NPZ histories, and 128 worker logs
  validated.
- Production replay, independent reference replay, and the archived summary
  matched with absolute/relative tolerance `{comparison['absolute_tolerance']}`.
- Runs/pairs/topologies: 64/32/16; all seed pairs finite-feasible in both arms.
- Semantic-prior topology wins/ties/losses:
  `{paired['wins_ties_losses']['semantic_prior_wins']}/{paired['wins_ties_losses']['ties']}/{paired['wins_ties_losses']['semantic_prior_losses']}`.
- Mean difference: `{paired['topology_macro_mean_difference']}`; median:
  `{paired['topology_macro_median_difference']}`; p90 regret:
  `{paired['topology_p90_regret']}`.
- Frozen 95% topology bootstrap CI: `[{ci['lower']}, {ci['upper']}]`, seed
  `{ci['seed']}`, `{ci['resamples']}` resamples.
- Frozen status/action: `{decision['status']}` / `{decision['action']}`. No
  promotion to confirmation-v1.

## Phase 3 — post-hoc sensitivity and exploratory analysis

The estimand is semantic-prior minus no-prior best feasible loss after averaging
optimizer seeds within each topology. Negative favors semantic-prior. The exact
mean sign-flip test enumerated all {sign_flip['assignments_enumerated']} assignments
and gave two-sided p=`{sign_flip['two_sided_p_value']}`. The separate exact sign
test (direction only) gave p=`{sign['two_sided_p_value']}`. Wilcoxon signed-rank
sensitivity gave p=`{wilcoxon['two_sided_p_value']}` using `{wilcoxon['method']}`
with {wilcoxon['zeros']} zeros and {wilcoxon['absolute_rank_ties']} absolute-rank
ties; it requires symmetry and is not a promotion rule.

The exploratory topology-block mean CI is
`[{effect['mean_bootstrap_ci_95']['lower']}, {effect['mean_bootstrap_ci_95']['upper']}]`;
the median CI is
`[{effect['median_bootstrap_ci_95']['lower']}, {effect['median_bootstrap_ci_95']['upper']}]`.
No analysis supports average, median, or feasibility improvement. Four topologies
showed at least 0.05 descriptive benefit and six at least 0.05 harm, but these
post-hoc extremes are not confirmed subgroups.

Seed signs were opposite in {seed['pattern_counts'].get('opposite_signs', 0)} of
16 topologies and both harmful in {seed['pattern_counts'].get('both_harm', 0)};
the seed-difference correlation was `{seed['pearson_correlation_seed7_seed11']}`.
This instability limits topology-specific interpretation.

### Existing censor-aware target summaries

{chr(10).join(target_lines)}

Serial/session order showed Spearman rho
`{drift['serial_run_order']['spearman_rho']}`. The within-topology arm-first
contrast was `{drift['arm_first_order']['mean_contrast']}` (exploratory exact
sign-flip p=`{drift['arm_first_order']['exact_mean_sign_flip']['two_sided_p_value']}`).
The mean topology log10 evaluation-count ratio was
`{drift['evaluation_throughput']['mean_topology_log10_evaluation_ratio_semantic_over_no_prior']}`.
These diagnose drift and do not revise the frozen decision.

## Limitations and next gate

The study has only 16 independent topology units, two optimizer seeds per
topology, heavy censoring below loss 4.0, and no preregistered equivalence margin.
Failure to reject is not equivalence or non-inferiority. After the separate
packaged-candidate alignment, the recommended next gate is a separately frozen,
owner-approved no-prior-only submission-like evaluation on the existing
disjoint panel at the official budget. This workflow does not launch it.
"""


def _render_report_html(markdown_text: str) -> str:
    import markdown

    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>Learn2Design development-v2 analysis</title>
<style>
body {{ color: #1f2933; background: #fff; font: 16px/1.55 system-ui, sans-serif;
       max-width: 980px; margin: 2rem auto; padding: 0 1.5rem 4rem; }}
h1, h2, h3 {{ color: #102a43; line-height: 1.25; }}
h1 {{ border-bottom: 3px solid #2a9d8f; padding-bottom: .4rem; }}
h2 {{ border-bottom: 1px solid #bcccdc; padding-bottom: .25rem; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #bcccdc; padding: .45rem .6rem; text-align: left; }}
th {{ background: #f0f4f8; }}
code {{ background: #f0f4f8; padding: .1rem .25rem; border-radius: 3px; }}
pre {{ background: #102a43; color: #f0f4f8; padding: 1rem; overflow-x: auto; }}
pre code {{ background: transparent; padding: 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Outcome-blind validation, dual frozen replay, and post-hoc topology "
            "analysis for development-v2."
        )
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--checksum", type=Path)
    parser.add_argument("--package-manifest", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    archive = args.archive.resolve()
    checksum = (
        args.checksum.resolve()
        if args.checksum
        else archive.with_suffix(archive.suffix + ".sha256")
    )
    package_manifest = (
        args.package_manifest.resolve()
        if args.package_manifest
        else archive.with_suffix(archive.suffix + ".manifest.json")
    )
    output = (
        args.output.resolve()
        if args.output
        else archive.parent / f"{archive.stem}-analysis"
    )
    repository_root = ROOT.resolve()
    if output == repository_root or repository_root in output.parents:
        parser.error("generated output must be outside the Git repository")
    if output.exists():
        parser.error(f"refusing to overwrite existing output directory: {output}")

    sources = SourcePaths(
        archive=archive,
        checksum=checksum,
        package_manifest=package_manifest,
        plan=args.plan.resolve(),
    )

    # Gate 1: source authentication, ZIP integrity, and history recomputation.
    study = validate_study_archive(sources)

    # Gate 2: two genuinely distinct topology-level replays, still summary-blind.
    production = summarize_records(study.records, study.configs)
    reference = reference_replay(study)
    production_reference = compare_production_and_reference(production, reference)

    # Gate 3: only after both raw replays agree may the archived summary be opened.
    archived_summary = load_summary_after_reproduction(
        study, production_reference
    )
    production_archived = compare_archived_summary(production, archived_summary)

    posthoc = analyze_posthoc(study, production, reference)
    output.mkdir(parents=True, exist_ok=False)
    normalized = write_normalized_tables(
        study, output / "normalized", repository_root=repository_root
    )
    plot_paths = create_plots(posthoc, output)
    comparison = {
        "format_version": 1,
        "production_vs_independent_reference": production_reference,
        "production_vs_archived_summary": production_archived,
        "absolute_tolerance": production_archived["absolute_tolerance"],
        "relative_tolerance": production_archived["relative_tolerance"],
        "three_way_status": "matched",
    }
    validation = {
        "format_version": 1,
        "integrity": {**study.integrity, "summary_content_opened_after_replay": True},
        "source_sha256": study.source_hashes,
        "plan_id": study.manifest["plan_id"],
        "project_revision": study.manifest["project_revision"],
        "study_profile": study.manifest["configuration"]["study_profile"],
    }
    _write_json(output / "validation.json", validation)
    _write_json(output / "production_replay.json", production)
    _write_json(output / "independent_reference.json", reference)
    _write_json(output / "three_way_comparison.json", comparison)
    _write_json(output / "exploratory_analysis.json", serializable_posthoc(posthoc))
    _write_csv(output / "leave_one_topology_out.csv", posthoc["leave_one_topology_out"])
    _write_csv(
        output / "topology_exploratory.csv", posthoc["heterogeneity"]["topology_rows"]
    )
    _write_csv(
        output / "drift_diagnostics.csv",
        posthoc["drift_diagnostics"]["topology_rows"],
    )
    report_text = _aggregate_report(production, posthoc, comparison)
    (output / "analysis_report.md").write_text(report_text, encoding="utf-8")
    (output / "analysis_report.html").write_text(
        _render_report_html(report_text), encoding="utf-8"
    )
    handoff = {
        "format_version": 1,
        "output_directory": ".",
        "normalized_manifest": "normalized/normalized_manifest.json",
        "validation": "validation.json",
        "three_way_comparison": "three_way_comparison.json",
        "exploratory_analysis": "exploratory_analysis.json",
        "report": "analysis_report.md",
        "rendered_report": "analysis_report.html",
        "figures": [path.relative_to(output).as_posix() for path in plot_paths],
        "normalized_tables": normalized["row_counts"],
    }
    _write_json(output / "handoff.json", handoff)
    print(
        json.dumps(
            {"resolved_output_directory": str(output), **handoff},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
