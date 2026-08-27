# Contextual-bandit toy-signal mechanics - validated result

Date: 2026-08-27

Study ID: `contextual-bandit-toy-signal-v1`

Study revision: `6fd4d560e2896902a72c4cfe5a9a6b62b5301aa8`

Private immutable result SHA-256:
`3396ec8c8203c86659b21d2beaa4c5dfee91bbcdfdd9b1245d3057086aa0fc9b`

## Decision

All nine frozen deterministic CPU cases passed. The authenticated action is:

```text
synthetic_contextual_bandit_signal_recovered_for_harness
```

The fixed two-bin, two-action empirical-reward policy learned the deliberately
exposed context sign from chosen-action reward alone. It preserved its final
train state without held-out updates, beat the frozen constant and random
policies, and returned to chance under both the within-trajectory
shuffled-context and signal-ablation controls.

This validates the synthetic contextual-bandit harness, exact online update
order, and intended toy signal. It does not establish useful RL, optimizer, or
competition performance.

## Frozen-case results

- `typed_bandit_contract`: every generated trajectory used eight ordered
  steps, a finite float64 three-field context, scalar binary int8 preferred
  and selected actions, scalar float64 zero-or-one reward, and the exact
  nonterminal/terminal boundary. The selector accepted only the immutable
  context record.
- `generator_partition`: all 2,048 steps across 256 complete trajectories and
  eight generator regimes matched the frozen formulas, regime/key order,
  dtypes, boundary pattern, and committed dataset SHA-256. Train, validation,
  and test regimes, evaluator keys, and context rows were disjoint.
- `online_update_order`: all 1,024 train steps produced one context, select,
  reward, update, and log event in that order. Exactly 512 selections were
  forced exploration and 512 were greedy exploitation; all 128 terminal
  rewards were incorporated, only the selected table cell changed, and all
  seven malformed-order/value sentinels were rejected.
- `leakage_guards`: all ten forbidden pre-selection fields and both held-out
  update attempts were rejected. Validation and test produced zero updates and
  left the train-state commitment unchanged. Reversing completed logs
  preserved key-aligned scores, while duplicate-key and wrong-reward records
  were rejected and swapped actions changed the score.
- `baseline_replay`: constant-zero and constant-one each scored `0.5` on both
  held-out splits. PCG64 seed `2026082713` replayed exactly and scored
  `0.505859375` on validation and `0.51171875` on test on the same evaluator
  rows.
- `contextual_recovery`: the online train reward was exactly `768 / 1024 =
  0.75`, with cumulative regret `256`. Validation, test, and minimum held-out
  regime macro reward were all `1.0`; test gains over the better constant and
  random policies were `0.5` and `0.48828125`.
- `shuffled_context_control`: the frozen permutation preserved the context
  multiset, preferred-action/key metadata, held-out rows, and exact
  per-trajectory independence table while changing rowwise context assignment.
  Validation and test macro reward were both `0.5`, the positive gate failed,
  and the true-minus-shuffled test gap was `0.5`.
- `signal_attribution_control`: zeroing only the declared signal coordinate
  preserved both nuisance coordinates and all evaluator metadata. The freshly
  trained context-free policy and the true policy evaluated on zeroed test
  contexts both scored `0.5`.
- `process_isolation`: two fresh credential-scrubbed, network-disabled CPU
  workers reproduced the complete timing-free non-process projection
  byte-for-byte.

## Claim boundary

The result is synthetic, deterministic, fixed-formula, topology-independent,
two-action, immediate-reward, local-CPU, and locked-runtime only. The positive
signal was deliberately easy: one visible sign determines the rewarding arm,
and the exploration schedule guarantees balanced coverage. The result supports
only that this harness can process chosen-action reward in the declared order,
learn that exposed online signal, score untouched generator regimes without
updating, and detect the declared negative controls.

It does not establish:

- predictive or causal signal in UIFO observations, optimizer trajectories, or
  public constraint diagnostics;
- delayed credit assignment, action-dependent state transitions, production
  RL, meta-RL, or useful policy transfer;
- a useful restart, initializer, budget allocator, or candidate treatment;
- runtime, accelerator, native-rewrite, or competition-score improvement;
- transfer to official or hidden structures.

No official problem, dataset, structure panel, private trajectory, generated
competition panel, GPU, provider, paid endpoint, or portal was used. The
protected submission tree and patience-600/no-prior random-start defaults are
unchanged. The owner-uploaded ZIP remained SHA-256
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.

## Repository verification

The focused fixture, exact-result-validator, sanitizer, and seventh/eighth
controller-transition tests passed before execution. The single full
repository verification pass completed with the expected two skips,
`git diff --check` was clean, and a fresh scratch-only package build produced
SHA-256
`4b7384dd6d401918b9b46ace4e65c3f116c34feeeca825ae25745dfb9e7908bd`.
The submitted ZIP, manifest, and submission tree retained their protected
hashes.

The controller authenticated the frozen registry, five committed source
hashes, exact result schema, clean branch, protected tree/artifacts, CPU worker,
network block, immutable output, and SHA sidecar. It returned to
`awaiting_study` with failure streak zero and released its global lease.

## Next research rung

The next admissible learning-mechanics question is delayed credit assignment
in a synthetic action-dependent state process, not production RL or candidate
integration. Before executing it, freeze a new topology-independent episodic
family and typed state, action, transition, delayed-reward, trajectory,
train/validation/test, update-order, attribution, and leakage contracts.

One narrow successor could ask whether a small deterministic tabular or linear
value learner recovers a deliberately learnable two-step delayed choice on
held-out generator regimes, beats frozen myopic, constant, and seeded-random
baselines, and fails identically evaluated transition-shuffle and reward-delay
controls. A pass would validate only the delayed-credit harness and toy signal.
Meta-RL, official-data training, candidate integration, native rewrites,
accelerator benchmarking, and paid training remain separate owner gates.
