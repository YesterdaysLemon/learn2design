# Multi-step TD propagation v1 preflight rejection

Date: 2026-08-28

Study ID: `multistep-td-propagation-v1`

Disposition: `rejected_before_terminal_execution`

Next action: `freeze_target_independent_multistep_td_v2`

## What happened

The v1 plan and self-contained synthetic fixture were committed before local
preflight. The deterministic non-process projection passed its twelve
implemented cases, and a direct isolated-worker replay also matched locally.
Those checks are development diagnostics only. The guarded controller was
never invoked, no terminal result or result sidecar was written, and the study
was not added to the approved registry.

Two independent read-only audits then found that the fixture did not support
its intended claim cleanly enough. The checkpoint was rejected before the
clean approval commit rather than weakening a rule or treating passing
preflight numbers as evidence.

## Blocking findings

1. The public `alive` state was computed from whether the previous action
   matched the evaluator-only target. Later observations therefore exposed a
   partial correctness signal before terminal reward. Even though the TD
   updater received zero nonterminal reward, this state channel confounded the
   intended terminal-signal-only propagation claim.
2. The frozen dataset commitment covered phase-zero observations and metadata,
   not the complete legal transition and terminal-outcome family. A successor
   or reward-formula change could therefore evade that commitment.
3. The held-out isolation case rejected direct validation/test fitting but did
   not exercise the promised absent and exploding held-out-source sentinels.
4. Keyed scoring recomputed canonical reward and transition state, but it did
   not authenticate every training-only donor/origin field or independently
   permute every component. Several malformed and cross-episode recombination
   attacks were missing.
5. The reward-origin derangement preserved the reward multiset but did not
   prove equal assigned-return counts and means in every target/action cell.
   It therefore did not establish destruction of the learnable association.
6. The terminal-dependency, reward-timing, and signal-ablation cases did not
   cover every sentinel or every generated trajectory observation promised by
   the plan.

## Frozen boundary

No terminal pass/fail claim exists for v1. Its passing preflight projection
does not validate multi-step bootstrapping, delayed credit, RL, an optimizer,
candidate performance, hidden structure, accelerator behavior, or a
competition score. The nine earlier terminal local studies remain the entire
approved evidence set.

## Requirements for a successor

A successor must use a new versioned study ID and a newly frozen plan before
executing its learner. At minimum it must:

- make every public successor target-independent, for example by exposing only
  an action-prefix state while keeping terminal success evaluator-only;
- commit and independently replay every legal observation/action successor,
  terminal reward, and `done` outcome across all regimes;
- prove train-only execution with both absent and exploding held-out sources;
- authenticate action, observation, predecessor, successor, reward,
  update-reward, donor, origin, and Boolean fields under independently
  reordered components and malformed/cross-episode attacks;
- construct the reward-origin control outcome-blindly with exact balanced
  counts and means in every frozen target/action cell, materializing its scalar
  feedback only at the terminal evaluator boundary; and
- test terminal-scalar dependency, every frozen timing attack, and
  signal-only ablation across the complete generated trajectory family.

The thresholds, regimes, seeds, and controls for that successor must be frozen
without selecting against v1's observed preflight metrics.
