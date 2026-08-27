# Supervised toy-signal learning contract - validated result

Date: 2026-08-27

Study ID: `supervised-toy-signal-v1`

Study revision: `09389b5a71d14ead88da3f8d5afd368fda9906eb`

Private immutable result SHA-256:
`5184bf927d3d1b301dc977a31af32352bf2f9a3543e23e04a11458fc8b8d2129`

## Decision

All eight frozen deterministic CPU cases passed. The authenticated action is:

```text
synthetic_supervised_toy_signal_recovered_for_harness
```

The small deterministic ridge surrogate recovered the deliberately exposed
binary signal on every train, validation, and test regime. It beat both frozen
held-out baselines, while the identically scored within-block label-shuffle and
signal-ablation controls stayed at chance.

This validates the synthetic learning harness and its intended toy signal. It
does not establish that supervised learning, a contextual bandit, or RL can
improve the submitted optimizer or any competition objective.

## Frozen-case results

- `typed_task_contract`: the no-structure, horizon-one task used a float64
  four-field observation, binary int8 action and target, float64 zero-or-one
  reward, and terminal one-step trajectory. The policy boundary accepted only
  the observation field.
- `generator_partition`: all 1,024 trajectories across eight generator regimes
  matched the frozen formulas, row order, regime order, sample keys, dtypes,
  and committed dataset SHA-256. All 256 four-row blocks validated; train,
  validation, and test regimes, keys, and observation rows were disjoint.
- `leakage_guards`: all five forbidden policy fields and all three overlap
  sentinels were rejected. Validation and test fit attempts were rejected from
  their authenticated sample identities. Reversing each held-out evaluation
  stream preserved the key-aligned scores and commitments exactly. All three
  legitimate fits authenticated train rows only.
- `baseline_replay`: the constant-zero baseline scored `0.5` on validation and
  test. PCG64 seed `2026082707` replayed exactly and scored `0.51953125` on
  validation and `0.54296875` on test, on the same committed held-out rows.
- `supervised_recovery`: train, validation, test, and minimum held-out-regime
  macro accuracy were all `1.0`. Validation gain over constant was `0.5`; test
  gains over constant and random were `0.5` and `0.45703125`.
- `label_shuffle_control`: the frozen nonidentity permutation preserved the
  label marginal and feature and held-out commitments. Its test macro accuracy
  was `0.5`, it missed the positive gate, and the true-minus-shuffled gap was
  `0.5`.
- `signal_attribution_control`: zeroing only the declared signal coordinate
  preserved every nuisance coordinate and held-out row/label commitment. Both
  the refit nuisance-only policy and the true policy evaluated without the
  signal scored `0.5` on test.
- `process_isolation`: two fresh credential-scrubbed, network-disabled CPU
  workers reproduced the complete non-process projection byte-for-byte.

## Claim boundary

The result is synthetic, deterministic, fixed-formula, topology-independent,
one-step, linear-surrogate, local-CPU, and locked-runtime only. The positive
signal was deliberately easy: the target is the sign of one observed feature.
The result therefore supports only that the harness can learn an exposed toy
signal, score held-out generator regimes, detect its declared negative
controls, and enforce the frozen split boundary.

It does not establish:

- predictive signal in UIFO observations, optimizer trajectories, or public
  constraint diagnostics;
- sequential credit assignment, exploration, contextual-bandit, RL, or
  meta-RL competence;
- a useful restart, initializer, budget-allocation, or candidate policy;
- runtime, accelerator, native-rewrite, or competition-score improvement;
- transfer to official or hidden structures.

No official problem, dataset, structure panel, private trajectory, generated
competition panel, GPU, provider, paid endpoint, or portal was used. The
protected submission tree and patience-600/no-prior random-start defaults are
unchanged. The owner-uploaded ZIP remained SHA-256
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.

## Repository verification

The focused learning-contract and controller tests passed before execution.
The single full repository verification pass then completed with the expected
two skips, `git diff --check` was clean, and a fresh scratch-only package build
produced SHA-256
`4b7384dd6d401918b9b46ace4e65c3f116c34feeeca825ae25745dfb9e7908bd`.
The controller authenticated the frozen registry, five committed source
hashes, exact result schema, protected source/tree/artifacts, clean revision,
CPU worker, network block, immutable output, and SHA sidecar. It returned to
`awaiting_study` with failure streak zero and released its lease.

## Next research rung

The next admissible learning checkpoint is a contextual-bandit mechanics test,
not candidate integration or production RL. Before running it, freeze a new
topology-independent sequential toy family, typed context/action/reward/logging
contract, untouched generator regimes, online update order, deterministic
seeds, constant and random baselines, regret or reward metric, and a
context-shuffle negative control.

The narrow question should be whether a small deterministic two-action bandit
can learn an observation-dependent choice online, beat the frozen baselines on
held-out generator regimes, and fail its identically evaluated shuffled-context
control. A pass would validate only the bandit harness and toy online signal.
Meta-RL, official-data training, candidate integration, native rewrites,
accelerator benchmarking, and paid training remain separate owner gates.
