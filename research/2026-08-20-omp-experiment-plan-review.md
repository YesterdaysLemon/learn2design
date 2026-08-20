# Independent experiment-plan review — 2026-08-20

## Question

Before renting an A100, we asked two external models to attack the frozen UIFO
experiment design independently:

- `anthropic/claude-opus-5` as a hostile experimental-design and competition
  methodology reviewer;
- `alibaba-token-plan/qwen3.8-max` as a decision-theoretic compute-efficiency
  and reproducibility auditor.

The review used the local OMP harness in read-only mode. The models could read
only listed public repository paths, had no shell or write tools, saw no local
artifacts or credentials, and did not see each other's reports. The first
exhaustive Qwen attempt reached its time limit; the same exact model completed
an independent retry from an outcome-free compressed design dossier. No GPU,
simulator, or live result was used.

## Consensus

Both reviewers returned **go only after specific repairs**. They agreed that the
paired equal-wall-clock causal comparison is credible, but that the paid design
was not yet decision-complete:

- the development and confirmation actions needed numeric precommitment;
- 16 topology units can screen only a large, consistent effect;
- the Adam orientation arm consumed one third of the rental without entering
  the promotion decision;
- threshold censoring needed topology-level analysis before any speedup claim;
- feasibility needed an explicit co-primary or lexicographic decision route;
- one outcome-independent timing pilot should precede the panel;
- partial recovery and repeated validation overhead deserved paid-run guards.

## Local adjudication

The external reports were treated as attack ideas, not evidence. Local checks
resolved their disagreements:

- The three-arm plan balanced marginal arm positions but not the primary
  comparison: `no_prior` preceded `semantic_prior` in 22 of 32 pairs, while the
  reverse occurred in 10. The two-arm plan is exactly 16:16.
- The semantic prior does not collapse to fallbacks on the development panel:
  all 3,058 parameter slots have exact semantic keys.
- The pinned dfbench runtime uses 50 frequencies and starts the scored time at
  `Objective.start_logging()`.
- Rebuilding indexes after every worker revalidated completed history artifacts
  4,656 times across the old 96-run plan. The new loop still emits durable
  indexes but reserves full historical revalidation for startup, resume,
  finalization, and packaging.
- With 16 topology units, 12 or more same-direction wins has a two-sided exact
  sign-test value of 0.076812744140625; 13 or more has 0.021270751953125. The
  development rule is explicitly a screening gate, not a significance claim.

Qwen's claim that the three-arm ordering was sound was rejected because it
looked only at marginal positions. Its statement that two seeds provide no
within-topology variance information was also too strong; two provide extremely
weak information. Opus's concern about prior fallback coverage was retired by
the exact slot audit. Neither review licenses a claim about live optimizer
quality.

## Adopted design

The updated `development-v2` profile contains 64 runs and 10 hours 40 minutes
of scored Objective time. Its complete configuration and decision policy are
bound into the plan ID. A disjoint 12-topology, 1,800-second `confirmation-v1`
profile is also frozen, but it runs only after a machine-reported development
pass. The decision treats strict finite-feasibility dominance as a guarded
lexicographic route and otherwise requires the complete paired-loss rule.

The primary report remains the repository's dated accelerator evaluation plan;
the rental commands and recovery procedure remain the A100 runbook. This note
records why those authorities changed before any panel outcome existed.
