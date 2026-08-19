# Optimizer state and budget-tail repair — 2026-08-19

## Decision

Reset Adam's bias-correction age together with each restarted population
member's first and second moments. This is a state-consistency repair, not a
new optimizer-performance claim. The frozen `semantic_prior` and `no_prior`
arms receive the identical repair, so their causal initializer contrast is
unchanged.

The same correctness pass also preserves otherwise-discarded evaluation-budget
tails and avoids launching a batch when its observed runtime no longer fits in
the remaining wall-clock budget.

## Failure mode

The previous loop reset a stalled member's moments to zero but continued using
one global Adam step counter. A fresh member therefore combined first-step
moments with late-run bias-correction denominators. With the configured
`beta1=0.9`, `beta2=0.999`, and `patience=600`, the first update after the
earliest possible restart was about 2.12 times the correctly bias-corrected
first Adam update. The factor approaches about 3.16 for sufficiently late
restarts.

The repair maintains one integer age per population member, increments ages
with each gradient update, and resets an age to zero whenever that member's
parameters and moments restart. Non-restarted members retain their ages.

## Budget-tail failures

The public Objective admits a batch only when the entire batch fits the
remaining evaluation budget. The old loop always submitted the full population,
so a non-divisible cap evaluated and counted the tail but logged none of its
results. The repaired loop submits only the remaining prefix and stops after
Objective records it.

Time enforcement also happens after an evaluation returns. A fixed two-second
guard could therefore launch a batch that finishes after the logging deadline.
The loop now measures device-complete batch durations and requires the larger
of the configured guard or `batch_time_safety_factor` times the slowest of the
last `batch_time_window` batches before starting another. Their defaults are
1.5 and eight, and the paired harness records both. The first compilation-heavy
call is deliberately excluded; the learned guard activates after the second,
same-shape call supplies a steady-state observation. The first compilation
remains timed and cannot be predicted in advance. This policy protects the
competition tail without carrying one-off compilation latency through a short
screen or using a private API.

## Verification and evidence boundary

A deterministic analytic test constructs one just-restarted member and one
uninterrupted 600-step member under a constant gradient. Their next updates
both equal the configured learning rate, while their ages become 1 and 601.
The existing lifecycle test exercises the complete optimizer loop with forced
restarts. Additional analytic tests prove that a six-evaluation budget with a
four-member population records batch sizes 4 and 2 without overshoot, and that
an observed batch duration can stop the next call even when the static guard
is zero. A signature contract keeps the harness's recorded settings equal to
the submission defaults.

This mathematical correction does not establish a lower UIFO loss. The next
performance gate remains the predeclared equal-wall-clock, full-vmap A100
comparison on frozen topology panels.
