# Two-step delayed-credit mechanics result

Date: 2026-08-28  
Study ID: `two-step-delayed-credit-v1`  
Terminal status: **passed**  
Terminal action: `synthetic_two_step_delayed_credit_recovered_for_harness`

## Decision

The ninth guarded local study passed all eleven frozen cases. The fixed
four-state, two-action empirical-return learner recovered the deliberately
exposed two-step terminal-return signal on untouched synthetic validation and
test regimes. It beat the frozen constant, myopic, and seeded-random baselines,
and the signal disappeared under the precommitted transition-assignment,
reward-origin-misalignment, and signal-ablation interventions.

This closes only the synthetic two-step delayed-credit harness rung. It does
not authorize production RL, meta-RL, a submission change, official-data
training, candidate integration, a native rewrite, accelerator benchmarking,
or paid compute.

## Provenance

- frozen plan:
  [`2026-08-28-two-step-delayed-credit-plan.md`](2026-08-28-two-step-delayed-credit-plan.md)
- terminal revision:
  `13c3111275dd199ef839c4acee8f53907d74fd9f`
- immutable private result SHA-256:
  `a46b6fafd08531b48fc8505dae2cc45544594dfe3bd2e9e448c91167bb741e13`
- generated-family commitment:
  `fdeed53ae38fed818dba8ec5d3aa203d982ee0f3d92e6bd179cfa87d47970b89`
- pinned registry SHA-256:
  `b14d7e20fd2b06dce8911254f5110cb445fae36287e07dc7d5f1da7a6b9b56e3`
- fixture source SHA-256:
  `a65a8499ca7509b99c60e538a482d11caf670f7d6f6c9b1d01c5d35228b0fbb9`
- worker source SHA-256:
  `9b5c2ab937be7fbf5675ea80aa916f41118a8aefa1d02a1b7827835ff94da03b`

The controller ran the committed worker on local CPU with networking disabled,
credential-scrubbed environment variables, a single-writer lease, a one-hour
process-tree deadline, bounded output, immutable result creation, and a SHA
sidecar. The state ledger records the terminal pass and returned to
`awaiting_study` with no approved study pending, no active cycle, no lease, and
failure streak zero. No raw private result, trajectory, state, action, reward,
target, table, or path is committed here.

## Frozen family and learner

The generator contained eight topology-independent regimes: four train, two
validation, and two test, each with 32 two-action episodes. The policy saw only
an immutable `float64[4]` observation. Target, reward, transition provenance,
keys, split, regime, done, donor identity, and reward origin remained evaluator
only.

The train behavior schedule covered all four action pairs in every eight
episodes. Exactly one terminal scalar arrived after both actions. The learner
then updated the phase-one table cell followed by the phase-zero table cell.
Validation and test used a frozen greedy train state and performed no updates.

## Sanitized aggregate result

| quantity | frozen expectation or gate | observed |
|---|---:|---:|
| train behavior mean return | exactly `0.25` | `0.25` |
| train behavior regret | exactly `96` | `96` |
| completed episode updates | exactly `128` | `128` |
| ordered table-cell updates | exactly `256` | `256` |
| post-fit train macro return | at least `0.99` | `1.0` |
| validation macro return | at least `0.99` | `1.0` |
| test macro return | at least `0.99` | `1.0` |
| minimum held-out-regime return | at least `0.98` | `1.0` |
| best constant validation/test | frozen baseline | `0.5 / 0.5` |
| myopic validation/test | frozen baseline | `0.5 / 0.5` |
| seeded-random validation/test | replay only | `0.203125 / 0.265625` |
| transition-shuffle validation/test | at most `0.05` | `0.0 / 0.0` |
| true minus transition-shuffle test | at least `0.90` | `1.0` |
| reward-misalignment validation/test | at most `0.55` | `0.5 / 0.5` |
| true minus reward-misalignment test | at least `0.40` | `0.5` |
| refit signal-ablation test | at most `0.55` | `0.5` |
| true-policy signal-ablation test | at most `0.55` | `0.5` |

All cases passed:

1. `typed_episodic_contract`
2. `generator_partition`
3. `action_dependent_transition`
4. `delayed_update_order`
5. `leakage_guards`
6. `baseline_replay`
7. `delayed_credit_recovery`
8. `transition_shuffle_control`
9. `reward_delay_control`
10. `signal_attribution_control`
11. `process_isolation`

The guards included exact formula replay, disjoint phase-zero generator rows,
authenticated keyed scoring, independent component reordering, strict action
types, wrong/missing/duplicate record rejection, nine scoring attacks, thirteen
timing/action/reward/queue attacks, forbidden-field probes, train-only scope,
frozen held-out state, exact full-row transition donors, exact reward origins,
empty delayed queue at the split boundary, signal-only ablation, and identical
fresh-process projections.

## Interpretation

The positive result supports three narrow mechanics statements:

- the harness can withhold all feedback until the end of a two-action episode;
- the fixed learner can apply one terminal return to the two visited state-action
  cells in the frozen order and retain that mapping on held-out generator
  regimes;
- the frozen controls remove that learner's usable association when successor
  rows or terminal-reward origins are reassigned.

The result is deliberately model-specific. The target is a simple function of
the visible toy signal, and an oracle hand-coded with the generator formula
could choose the correct actions without learning. The evidence is therefore
that this blank empirical table learned the mapping from terminal feedback—not
that delayed feedback is information-theoretically necessary for every policy.
Likewise, the transition control reassigns the public successor observation
while preserving the hidden realized branch and canonical terminal rule; it
tests this learner's transition attribution, not every possible representation.

The reward-delay control is a no-fixed-point reward-origin derangement, not a
claim that a correctly associated longer transport delay is unlearnable. Every
table cell received the same assigned mean return `0.25`, and the frozen tie
policy returned to `0.5` on canonical held-out episodes.

## Verification and protected-submission boundary

Before terminal execution:

- the focused delayed-credit, sanitizer, validator, and queue-transition tests
  passed;
- the single full repository verification pass completed green;
- a submission artifact built successfully only to a fresh scratch path;
- the submitted local ZIP remained
  `4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`;
- the protected submission tree remained
  `e2b495b6f1bf9f5c8f0b36ae5bc095f6df8e7588`.

No file under `submission/` changed, no protected artifact was overwritten, no
portal or leaderboard was accessed, no official dataset or private topology
panel was used, and no GPU, provider, paid endpoint, Docker, SSH, or money was
involved.

## Next gate

The next admissible unpaid learning rung is bootstrapped multi-step value
propagation, not meta-RL. A future study must first freeze a new
topology-independent multi-step family and exact TD target, bootstrap,
update-order, held-out, attribution, and leakage contracts. The narrow question
should be whether a deterministic tabular TD learner propagates a terminal
signal backward across more than one bootstrap boundary, beats frozen
constant, myopic, no-bootstrap, and seeded-random baselines, and loses the
signal under precommitted transition-target and reward-origin controls.

That future plan requires a new unique study ID and a fresh pre-result commit.
There is currently no approved pending study, and this terminal fixture must
never be rerun.
