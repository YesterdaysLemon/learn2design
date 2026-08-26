# Frozen topology panels

The original development, confirmation, and submission-like size-3 UIFO panels
were frozen before live UIFO results were observed. The restart panels are a
later, explicitly post-hoc follow-up with a separate evidence boundary.

- `development-v1.json`: 16 topologies for the short initializer screen;
- `confirmation-v1.json`: 12 disjoint topologies for longer confirmation;
- `submission-like-v1.json`: 10 disjoint topologies originally reserved for an
  official-budget rehearsal; the current cost-bounded screen does not make that
  claim;
- `coverage-robustness-v1.json`: 12 archive-disjoint and prior-panel-disjoint
  topologies from the historical conditional confirmation design; Stage A
  failed, so this still-unobserved panel was never unlocked and must not run;
- `coverage-triage-v1.json`: 8 archive-disjoint and all-prior-panel-disjoint
  topologies used by the completed, terminal lower-cost Stage-A coverage
  screen; that failed gate is closed and cannot be rerun or topped up;
- `restart-mechanics-v1.json`: one outcome-selected restart mechanics case whose
  loss is excluded from inference;
- `restart-screen-v1.json`: eight development topologies selected by a committed
  outcome-blind SHA-256 rank after excluding the mechanics case;
- `audit.json`: source hashes, exact archive-overlap results, panel-overlap results,
  coarse balance summaries, and explicit provenance records for the post-hoc
  restart panels.

The audit records the historical dataset/generator default revision
`d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c` and the explicit current
topology-generation/API override `1bb7f54737dec6a08b59879a8831d125f08f8a0b`
used for `coverage-robustness-v1` and `coverage-triage-v1`. Exact archive
exclusion is independently bound to the dataset SHA-256 in `audit.json`; all
older panel bytes remain frozen.

The original panel generator resolves topology metadata only. It does not
construct `UIFOProblem`, run the simulator, or inspect losses. Its selection
round-robins across detector/homodyne readout, squeezer-count, and
directional-interior strata. The 8-member coverage-triage panel is readout
balanced (D/H `4/4`) but its observed complexity marginals are squeezer
low/middle/high `4/2/2` and directional low/middle/high `2/4/2`; it is a triage
panel, not a full complexity-balanced panel. The restart mechanics case is
deliberately outcome-selected for mechanism observability and excluded from
inference. The restart screen uses the documented SHA-256 ranking after that
exclusion. Their source hashes, inherited archive exclusion, and exact overlaps
are recorded under `posthoc_panels` in `audit.json`.

Rebuild and verify against the pinned official archive:

```bash
uv run --group integration python tools/build_topology_panels.py /path/to/dataset.h5
```

Confirmation must also pass the prior development panel to the run harness with
`--exclude-prior-panel`. The coverage triage screen must pass every earlier
generated panel (`development-v1`, `confirmation-v1`, `submission-like-v1`,
and `coverage-robustness-v1`) plus any explicitly supplied post-hoc panel. The
harness recomputes exclusion from the actual dataset and panel bytes rather
than trusting `audit.json`.
