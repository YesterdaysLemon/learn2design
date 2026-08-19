# ML initializer experiment

This experiment asks a narrow question: can topology-conditioned archive information propose better starting points than semantic medians or random sampling?

The target is not a surrogate for the physics simulator. Every proposed design must still be evaluated and refined through the official Objective.

## First screen

The loader keeps the best stored-loss size-3 row for each exact topology, reconstructs its topology-specific semantic parameter layout without constructing a simulator, and normalizes every bounded value to its property range. The split is deterministic and grouped by topology identity.

```bash
uv run --group integration python -m experiments.ml_initializer.screen /path/to/dataset.h5
```

The screen compares:

- exact semantic-key/property medians;
- semantic transfer from the nearest topology by component-token Hamming distance;
- a shuffled-topology control;
- random unit-space starts.

The metric is topology-macro, property-balanced unit-space MAE. It is only an offline debug gate. A method must still improve paired best-feasible loss and anytime performance through the live Objective before entering the submission.

## Conditional decoder

If simple transfer fails, the next screen trains a four-head topology-token encoder with a shared semantic-key decoder and an exact semantic-median skip connection:

```bash
uv run --group integration python -m experiments.ml_initializer.train /path/to/dataset.h5
```

The same model is trained again with topology tokens shuffled between training examples. The learned prior is licensed for live testing only if its held-out four-head error beats semantic medians by 15%, beats the shuffled control by 10%, and every head wins at least 5% of held-out topology identities.

The neural gate failed. The simpler semantic median passed the comparative screen and is refit on all unique size-3 topology identities for one live-test population slot:

```bash
uv run --group integration python -m experiments.ml_initializer.export_prior /path/to/dataset.h5
```

The exporter refuses any dataset whose SHA-256 does not match the pinned official archive. The checked-in JSON records its source revision, checksum, sample policy, and support for each semantic key.
