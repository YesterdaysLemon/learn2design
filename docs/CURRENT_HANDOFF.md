# Current handoff

Updated: 2026-08-20

This is the authority for the project's next action. Dated files in `research/`
are the evidence record, not a competing task list.

## Current state

- The submission candidate is deterministic and packages successfully.
- The topology-conditioned neural initializer failed its offline control gate.
- The semantic archive prior is only a live-test candidate.
- No competition-aligned multi-topology A100 result exists yet.

## Next decision

Run the frozen three-arm `development-v1` screen on one idle A100 80 GB GPU.
The causal comparison is `semantic_prior` versus `no_prior`; `adam` is an
orientation arm. Follow [`A100_RENTAL_RUNBOOK.md`](A100_RENTAL_RUNBOOK.md)
without changing the arms, panels, seeds, budgets, thresholds, cache policy, or
run order after observing results.

Do not start paid compute without explicit owner approval. The current harness
is serial and has no validated multi-GPU shard/merge path.

## What can contribute now

Without an A100: improve correctness tests, artifact integrity, rule audits, or
documentation without changing the frozen algorithmic comparison. With an
approved A100: execute and preserve the runbook's deployment ladder, study, and
checksummed artifact package.

After the development result, apply the promotion rule in
[`2026-08-19-accelerator-evaluation-plan.md`](../research/2026-08-19-accelerator-evaluation-plan.md):
either test the surviving prior comparison on the disjoint confirmation panel
or retain the no-prior candidate. Do not claim an order-of-magnitude gain unless
the predeclared time-and-evaluation threshold rule is satisfied.
