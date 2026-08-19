# Competition brief — 2026-08-19

## Verified target

The official competition page and starter kit describe a method-level optimization challenge. For each hidden UIFO topology, the submitted algorithm tunes roughly 200 continuous detector parameters. The official evaluation allows four hours after `Objective.start_logging()` on one NVIDIA A100 GPU and AMD EPYC 7302 CPU. Each leaderboard score is the arithmetic mean of the best physically feasible loss over ten hidden topologies; lower is better.

The submission is a flat ZIP containing `submission.py` and `requirements.txt`. The Python file must expose exactly one `dfbench.OptimizationAlgorithm` subclass. Evaluation has no network access, and direct imports of `differometor` are forbidden.

The final deadline is October 15, 2026, Anywhere on Earth. Optional public-leaderboard deadlines are August 26, September 12, and September 29.

## Reproducibility anchor

Starter-kit revision inspected: `artificial-scientist-lab/Learn2Design-2026@d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c`.

The upstream README reports these reference means across its benchmark set:

| Baseline | Mean loss | SEM |
|---|---:|---:|
| Adam | 1.1 | 0.3 |
| Noisy Adam | 1.2 | 0.4 |
| Momentum SGD | 1.2 | 0.4 |
| BFGS | 1.8 | 0.2 |
| L-BFGS | 2.9 | 0.2 |
| CMA-ES | 4.1 | 0.1 |
| Random search | 4.8 | 0.03 |

These are upstream reference numbers, not results reproduced in this repository.

## Initial hypothesis

The first candidate is deliberately modest: parallel Adam trajectories in unbounded coordinates, diversified learning rates, and deterministic restarts. The testable claim is that a batched portfolio reduces sensitivity to initialization while retaining the strong local performance of the organizer's Adam baseline. Until multi-topology measurements support that claim, it remains only a hypothesis.

## Primary sources

- [Competition page](https://www.learn2design2026.com/)
- [Official starter kit](https://github.com/artificial-scientist-lab/Learn2Design-2026)
- [Submission rules](https://github.com/artificial-scientist-lab/Learn2Design-2026/blob/d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c/docs/submission.md)
- [Scoring rules](https://github.com/artificial-scientist-lab/Learn2Design-2026/blob/d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c/docs/scoring.md)
