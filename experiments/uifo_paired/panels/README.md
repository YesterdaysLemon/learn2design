# Frozen topology panels

The original development, confirmation, and submission-like size-3 UIFO panels
were frozen before live UIFO results were observed. The restart panels are a
later, explicitly post-hoc follow-up with a separate evidence boundary.

- `development-v1.json`: 16 topologies for the short initializer screen;
- `confirmation-v1.json`: 12 disjoint topologies for longer confirmation;
- `submission-like-v1.json`: 10 disjoint topologies originally reserved for an
  official-budget rehearsal; the current cost-bounded screen does not make that
  claim;
- `restart-mechanics-v1.json`: one outcome-selected restart mechanics case whose
  loss is excluded from inference;
- `restart-screen-v1.json`: eight development topologies selected by a committed
  outcome-blind SHA-256 rank after excluding the mechanics case;
- `audit.json`: source hashes, exact archive-overlap results, panel-overlap results, and coarse balance summaries.

The original panel generator resolves topology metadata only. It does not
construct `UIFOProblem`, run the simulator, or inspect losses. Its selection
round-robins across detector/homodyne readout, squeezer-count, and
directional-interior strata. The restart mechanics case is deliberately
outcome-selected for mechanism observability and excluded from inference. The
restart screen uses the documented SHA-256 ranking after that exclusion.

Rebuild and verify against the pinned official archive:

```bash
uv run --group integration python tools/build_topology_panels.py /path/to/dataset.h5
```

Confirmation must also pass the prior development panel to the run harness with `--exclude-prior-panel`; the harness recomputes exclusion from the actual dataset and panel bytes rather than trusting `audit.json`.
