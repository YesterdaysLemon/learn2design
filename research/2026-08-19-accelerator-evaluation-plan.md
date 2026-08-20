# Accelerator UIFO evaluation plan — 2026-08-19

Updated before live development outcomes on 2026-08-20 after independent OMP
review. The development panel itself is unchanged; the paid design is now bound
as the `development-v2` study profile.

## Decision

Live, paired size-3 UIFO evidence is the next gate. No further initializer model
or optimizer feature should enter the submission until the current semantic
prior is compared with its exact no-prior ablation on suitable accelerator
hardware.

The experimental unit is one topology identity. Optimizer seeds are repeated
measurements within a topology, not independent samples. Development is a
screen for a large, consistent effect; failure to pass is not evidence that all
semantic priors are ineffective.

## Frozen profiles

The primary arms are:

- `no_prior`: current batched restart Adam with the semantic slot left random;
- `semantic_prior`: the identical batched method with that slot replaced by the
  checked-in prior.

Both arms draw the same seeded population and install the same feasibility
anchor; only member 1 differs. Organizer-style scalar Adam remains available as
a diagnostic but is not part of either paid causal panel.

| Profile | Panel | Seeds | Budget | Runs | Scored time |
| --- | --- | --- | ---: | ---: | ---: |
| `development-v2` | 16 development topologies | 7, 11 | 600 s | 64 | 10 h 40 m |
| `confirmation-v1` | 12 disjoint topologies | 7, 11 | 1,800 s | 48 | 12 h |

Both profiles use population 8, 50 frequencies, full-vmap evaluation, one idle
A100 80 GB with MIG disabled, cache-disabled isolated workers, and target losses
4.0, 1.0, 0.5, and 0.0. Development uses a 1,200-second worker timeout and
16-hour session cap. Confirmation uses a 3,000-second worker timeout and the
same session cap. One isolated worker error is tolerated so the rental can
preserve other pairs; the second error stops the session. Resume reruns every
non-complete record.

The profile name and complete decision policy are embedded in the plan ID and
every run configuration. The development plan must contain 16 `no_prior`-first
and 16 `semantic_prior`-first pairs. Any configuration drift fails plan
construction before a worker starts.

## Predeclared decisions

The smallest practically interesting median reduction is 0.05 absolute loss
units. It is one tenth of the smallest 0.5-unit separation among the frozen
diagnostic loss thresholds. Because the competition loss can cross zero,
relative final-loss improvements are not used.

Development advances to confirmation only when the complete, uncensored panel
satisfies every condition:

- semantic prior wins at least 12 of 16 topology-macro comparisons;
- topology-macro median `(semantic_prior - no_prior)` is at most -0.05;
- no optimizer-seed pair is physically feasible only under `no_prior`;
- the 90th-percentile topology regret is at most 0.5.

This is intentionally a liberal screening rule, not a confirmatory p-value. With
16 topology units, 12 wins has a two-sided exact sign-test value of about 0.077;
13 wins would be about 0.021.

Confirmation keeps the semantic prior only when the complete, uncensored panel
satisfies every condition:

- semantic prior wins at least 10 of 12 topology-macro comparisons;
- topology-macro median difference is at most -0.05;
- no optimizer-seed pair is physically feasible only under `no_prior`;
- the 90th-percentile topology regret is at most 0.5;
- the upper 95% topology-bootstrap bound for the mean difference is below zero.

Otherwise the honest action is to retain the no-prior candidate. There is no
interim stopping, rule adjustment, topology removal, threshold replacement, or
seed addition in response to observed outcomes. Missing runs are
`not_evaluable`; a completed panel with a no-feasible/censored comparison fails
rather than being converted into an outcome-conditional complete-case analysis.

## Deployment and execution stages

1. **Deployment ladder:** one fresh non-panel topology, population sizes 2, 4,
   then 8. Establish warmup completion, memory fit, feasibility logging, and the
   default full-vmap path.
2. **Timing pilot:** one outcome-independent 600-second `no_prior` run on the
   same non-panel topology. Require complete artifacts and at most 825 seconds
   of full worker wall time. The result is operational evidence only and never
   enters algorithm selection.
3. **Initializer screen:** execute the exact `development-v2` profile.
4. **Confirmation:** only after a machine-reported development pass, execute
   the exact `confirmation-v1` profile on the disjoint panel.
5. **Submission-like run:** ten disjoint topologies at the official budget.
   Report the arithmetic mean best feasible loss and every no-feasible run.

Equal-evaluation runs may diagnose initialization and the first restart, but
they do not license a competition-performance claim because batching changes
throughput. The primary comparison uses equal wall-clock budgets.

The pinned dfbench 0.3.3 warmup helpers discard asynchronous device outputs, so
neither the candidate nor local Adam baseline uses them. Compilation is counted
inside the Objective clock. Persistent JAX compilation caching is disabled and
recorded for every scored worker: sharing compiled executables would make later
arms warm and change scored evaluation throughput.

The current harness is serial. Parallel rental requires a separately validated
parent-plan shard and merge format; independently rebuilding subset plans is
prohibited because it changes plan identity and arm order.

## Required outputs

- full batched loss and feasibility histories;
- best feasible loss and anytime curve;
- first-feasible and best-feasible times/evaluation counts;
- feasible candidate and feasible-call fractions;
- frozen loss-threshold hitting times with censoring visible;
- topology-macro paired differences, wins/ties/losses, upper-tail regret, and
  topology-bootstrap intervals;
- the machine-evaluated predeclared decision and every criterion;
- device, versions, code revision, prior hash, exact budgets, topology
  identities, seeds, and run order.

`Objective.best_loss` is not the score: it can describe an infeasible candidate.
No-feasible runs remain `null` locally because only the organizers possess the
official random-search replacement score.

An order-of-magnitude statement is reserved for one frozen threshold where
every predeclared topology and seed pair supplies either two observed hits or a
conservative upper ratio bound from a semantic-only hit and the no-prior arm's
last logged censoring horizon. The upper 95% topology-bootstrap bounds for both
`log10(time ratio)` and `log10(evaluation ratio)` must be at most -1. A
no-prior-only or neither-reached pair cannot pass. Final loss is never expressed
as a multiplicative improvement because it can cross zero.
