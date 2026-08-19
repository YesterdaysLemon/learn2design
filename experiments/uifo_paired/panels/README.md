# Frozen topology panels

These size-3 UIFO panels were frozen before live UIFO results were observed.

- `development-v1.json`: 16 topologies for the short initializer screen;
- `confirmation-v1.json`: 12 disjoint topologies for longer confirmation;
- `submission-like-v1.json`: 10 disjoint topologies for official-budget rehearsal;
- `audit.json`: source hashes, exact archive-overlap results, panel-overlap results, and coarse balance summaries.

The generator resolves topology metadata only. It does not construct `UIFOProblem`, run the simulator, inspect losses, or select topologies by optimizer performance. Selection round-robins across detector/homodyne readout, squeezer-count, and directional-interior strata.

Rebuild and verify against the pinned official archive:

```bash
uv run --group integration python tools/build_topology_panels.py /path/to/dataset.h5
```

Confirmation must also pass the prior development panel to the run harness with `--exclude-prior-panel`; the harness recomputes exclusion from the actual dataset and panel bytes rather than trusting `audit.json`.
