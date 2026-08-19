# Learn2Design 2026 experiment lab

This repository is our open, reproducible attempt at the [Learn2Design 2026 competition](https://www.learn2design2026.com/): building an optimizer for differentiable gravitational-wave detector design.

The competition asks for an algorithm—not one fixed detector. On each hidden topology, the algorithm must tune roughly 200 continuous parameters within a four-hour budget. Lower average best loss is better.

## Current status

**Bootstrap stage.** We are auditing the official starter kit, reproducing its baselines, and turning promising ideas into measured experiments. No leaderboard claim has been made yet.

## Where things live

- `src/` — submission code and reusable implementation
- `tests/` — fast correctness and packaging checks
- `research/` — literature notes, experiment records, and longer technical reports
- `artifacts/` — generated submission bundles and evaluation summaries (not hand-edited)
- `docs/` — short project plans and contributor-facing guides

The repository root is intentionally small. Durable evidence goes in the folders above rather than in a growing root-level ledger.

## Reproducibility rule

Every reported result should identify the code revision, configuration, topology split, random seeds, budget, hardware, and raw output location. Public and hidden leaderboard results must be labeled separately.

## Official resources

- [Competition page](https://www.learn2design2026.com/)
- [Official starter kit](https://github.com/artificial-scientist-lab/Learn2Design-2026)
- [Submission portal](https://submit.learn2design2026.com/)

## License

Our original code and documentation are released under the [MIT License](LICENSE). Upstream datasets, simulator code, and competition materials retain their own licenses and terms.
