# ML initializer screen — 2026-08-19

## Decision

A neural topology-conditioned initializer is **not licensed for live testing** by this first screen. The model improved held-out reconstruction over semantic medians by 5.18%, but a control trained with topology tokens shuffled improved slightly more. The measurable gain came from producing several generic semantic parameter modes, not from learning useful topology structure.

We will not add the network or its 104 KB weight file to the submission. The next live-test candidate is the much smaller exact-semantic median prior, occupying at most one population slot alongside the feasibility anchor and random starts.

This is an offline reconstruction decision, not a claim about simulator loss.

## Data and split

- Official Differometor-30k SHA-256: `149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7`
- Size-3 topology identities: 11,678
- Training topologies: 9,351
- Held-out topologies: 2,327
- Target row: lowest stored loss per exact topology
- Split: deterministic SHA-256 topology identity split
- Metric: topology-macro, property-balanced unit-space MAE

The semantic layout reconstruction matched the archive parameter length for every unique size-3 topology. A raw padded-vector target was rejected before training: same-index slots share the same property only 36.28% of the time across topology layouts.

## Controls

| Initializer | Held-out mean MAE | Relative to semantic median |
|---|---:|---:|
| Random, one candidate | 0.43159 | +142.51% |
| Semantic median | 0.17797 | reference |
| Semantic quantiles, best of four | 0.17283 | -2.89% |
| Nearest topology, best of four | 0.19412 | +9.08% |
| Shuffled-topology transfer, best of four | 0.19366 | +8.82% |
| Neural decoder, best of four | 0.16876 | -5.18% |
| Neural decoder trained with shuffled topology | 0.16798 | -5.61% |

Nearest-topology transfer is indistinguishable from its shuffled control and is rejected. Semantic quantile heads add too little to justify four population slots.

## Neural screen

The 20-epoch JAX model used:

- an 8-dimensional embedding for each of 21 topology tokens;
- a 16-dimensional exact semantic-key embedding;
- a 32-dimensional topology context;
- a 64-dimensional shared decoder;
- four multiple-choice output heads;
- a semantic-key median logit skip connection.

All heads were used on at least 5% of held-out topologies, but the topology-signal gate failed: model-vs-shuffled improvement was -0.46%, versus a required +10%.

## What this says about the hammer

ML remains appropriate as a possible amortized initializer, but this dataset does not yet justify a topology-conditioned neural network. The archive strongly supports semantic, property-aware priors; it does not show held-out predictive topology signal under this target and control.

The safe next experiment is one semantic-median candidate plus the unchanged exact-gradient portfolio. A live Objective must decide whether that prior reaches a feasible strong basin earlier. We should only revisit a graph model if live evidence shows that generic semantic initialization helps but leaves systematic topology-dependent regret.

## Reproduction

```bash
uv run --group integration python -m experiments.ml_initializer.screen /path/to/dataset.h5
uv run --group integration python -m experiments.ml_initializer.train /path/to/dataset.h5 --epochs 20
```

Machine-readable results are stored under `research/results/`.
