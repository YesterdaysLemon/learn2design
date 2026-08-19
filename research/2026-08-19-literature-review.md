# Literature review and experiment hypotheses — 2026-08-19

## Bottom line

The strongest near-term direction is a feasibility-aware, batched multistart gradient method, followed by topology-conditioned archive starts if the simpler method earns that complexity. A global Gaussian-process optimizer over roughly 200 raw coordinates is not the first bet: this challenge supplies exact gradients, tens of thousands of potential evaluations, and heterogeneous topology-dependent parameter meanings.

This note proposes experiments; it does not claim that any method below improves the competition score.

## Why this direction is plausible

The organizer's published benchmark favors gradient methods: Adam is currently listed ahead of noisy Adam, BFGS, CMA-ES, and random search. The comparison is useful as a directional signal, but it lacks enough public run metadata to be treated as reproduced evidence.

The prior Urania gravitational-wave instrument search combined a large pool of starts, local BFGS searches, biased reselection, transferred solutions, and perturbations. That supports a local-global pool design in this domain, although its compute scale was far beyond the four-hour competition budget.

Adam offers cheap scale-adaptive local search. L-BFGS and basin hopping supply established exploitation and restart patterns. Transfer-optimization work shows that prior task configurations can improve initialization when source and target are related; it also makes the failure mode clear—negative transfer—so every archive-based method needs random-start controls.

## Ordered hypotheses

### H1 — batched restart Adam reduces initialization risk

Run several Adam trajectories through the public vmapped Objective API. Diversify learning rates, preserve the best feasible incumbent, and replace stalled members with alternating random and elite-perturbed starts.

Compare on identical topology and random-seed pairs:

1. organizer-style single-start Adam;
2. equal-budget independent Adam restarts;
3. batched Adam with fixed restarts;
4. batched Adam with adaptive restarts.

Primary outcome: mean best feasible loss. Secondary outcomes: infeasible-run count, time to first feasible point, time to best point, and anytime best-feasible curves.

### H2 — topology-conditioned archive starts beat random starts

Build an offline index over size-3 archive entries. Match topology strings by weighted component/orientation distance, transfer only coordinates whose component/property identities agree, and fill unmatched coordinates from robust property-specific distributions. Evaluate transferred, perturbed-transfer, and random starts together.

Controls must include random starts, property-median starts without topology matching, and globally best archive rows. Split by topology string and keep `initialized_from` lineages together; a row-random split would leak related reoptimizations.

### H3 — feasibility-aware selection improves the scored metric

Preserve feasible candidates separately from the lowest penalized-loss candidate. Compare the default squashed-ReLU power penalty with a small, predeclared set of alternatives on held-out public topologies. Do not select a penalty from the same topology/seed pairs used for the final local comparison.

### H4 — quasi-Newton polishing helps only after profiling

Reserve a final budget fraction to polish a few feasible elites with a public-API-compatible L-BFGS implementation. This is lower priority because the starter kit's current SciPy and L-BFGS examples use lifecycle patterns that conflict with `dfbench 0.3.3`, and custom compilation consumes timed budget.

### Stretch — learned low-dimensional initialization

Only pursue a latent archive model or high-dimensional Bayesian optimization after measurements show a stable low effective dimension. TuRBO and SAASBO are relevant references, but both target expensive derivative-free settings; neither establishes that a surrogate should replace exact gradients here.

## Evaluation protocol

- Use constrained Voyager for mechanics only; choose algorithms on multiple size-3 UIFO topologies.
- Pair topology seeds and optimizer seeds across methods.
- Predeclare staged budgets and stopping rules before the sweep.
- Report the competition metric, paired uncertainty intervals, infeasible-run count, time to first feasible point, and device/runtime metadata.
- Reevaluate archived candidates through the live Objective. Do not treat stored loss as interchangeable with the current scoring implementation.
- Re-audit the untagged starter kit before every submission cutoff.

## Primary sources

- [Learn2Design 2026 starter kit and baseline table](https://github.com/artificial-scientist-lab/Learn2Design-2026)
- [Krenn, Drori, and Adhikari, *Digital Discovery of Interferometric Gravitational Wave Detectors*, Physical Review X 15, 021012 (2025)](https://doi.org/10.1103/PhysRevX.15.021012)
- [Kingma and Ba, *Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980)
- [Byrd et al., *A Limited Memory Algorithm for Bound Constrained Optimization*](https://doi.org/10.1137/0916069)
- [Wales and Doye, *Global Optimization by Basin-Hopping*](https://doye.chem.ox.ac.uk/abstracts/jpc97.html)
- [Feurer et al., *Efficient and Robust Automated Machine Learning*](https://doi.org/10.1609/aaai.v29i1.9354)
- [Perrone et al., *Learning Search Spaces for Bayesian Optimization*](https://proceedings.neurips.cc/paper/2019/hash/6ea3f1874b188558fafbab78e8c3a968-Abstract.html)
- [Nomura et al., *Warm Starting CMA-ES for Hyperparameter Optimization*](https://doi.org/10.1609/aaai.v35i10.17109)
- [Eriksson et al., *Scalable Global Optimization via Local Bayesian Optimization*](https://papers.nips.cc/paper_files/paper/2019/hash/6c990b7aca7bc7058f5e98ea909e924b-Abstract.html)
- [Eriksson and Jankowiak, *High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces*](https://proceedings.mlr.press/v161/eriksson21a.html)
- [MODE Collaboration, *Toward Machine-Learned Optimization of Experimental Design*](https://doi.org/10.1016/j.revip.2023.100085)
