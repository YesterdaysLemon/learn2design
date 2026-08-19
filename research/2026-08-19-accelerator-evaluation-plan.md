# Accelerator UIFO evaluation plan — 2026-08-19

## Decision

Live, paired size-3 UIFO evidence is the next gate. No further initializer model or optimizer feature should enter the submission until the current semantic prior is compared with its exact no-prior ablation on suitable accelerator hardware.

The experimental unit is one topology identity. Optimizer seeds are repeated measurements within a topology, not independent samples.

## Frozen initial arms

- `adam`: organizer-style single-start Adam for orientation;
- `no_prior`: current batched restart Adam with the semantic slot left random;
- `semantic_prior`: the identical batched method with that slot replaced by the checked-in prior.

The causal initializer comparison is `semantic_prior` versus `no_prior`. Both arms draw the same seeded population and install the same feasibility anchor; only member 1 differs.

## Stages

1. **Deployment smoke:** one fresh topology, population sizes 2, 4, then 8 if memory permits. Establish warmup completion, batch latency, throughput, feasibility, and a conservative end-of-budget safety margin.
2. **Initializer screen:** paired short wall-clock runs on a frozen development panel absent from the archive. Keep every optimizer setting fixed.
3. **Confirmation:** repeat the surviving comparison on a disjoint topology panel and longer wall-clock budget.
4. **Submission-like run:** ten disjoint topologies at the official budget. Report the arithmetic mean best feasible loss and every no-feasible run.

Equal-evaluation runs may diagnose initialization and the first restart, but they do not license a competition-performance claim because batching changes throughput. The primary comparison uses equal wall-clock budgets.

The pinned dfbench 0.3.3 warmup helpers discard asynchronous device outputs, so neither the candidate nor local Adam baseline uses them. Compilation is conservatively counted inside the Objective clock until a public synchronous helper is available; deployment reports must state this policy.

## Required outputs

- full batched loss and feasibility histories;
- best feasible loss and anytime curve;
- first-feasible and best-feasible times/evaluation counts;
- feasible candidate and feasible-call fractions;
- frozen loss-threshold hitting times with censoring visible;
- topology-macro paired differences, wins/ties/losses, and topology-bootstrap intervals;
- device, versions, code revision, prior hash, exact budgets, topology identities, seeds, and run order.

`Objective.best_loss` is not the score: it can describe an infeasible candidate. No-feasible runs remain `null` locally because only the organizers possess the official random-search replacement score.

## Promotion rule

The semantic prior stays in the submission only if it lowers paired best feasible loss at equal wall-clock time on the disjoint confirmation panel, does not increase no-feasible runs, and does not create harmful upper-tail regret. An order-of-magnitude statement is reserved for a frozen loss threshold reached at least ten times sooner in both time and Objective evaluations with a topology-clustered confidence bound; final loss is never expressed as a multiplicative improvement because it can cross zero.
