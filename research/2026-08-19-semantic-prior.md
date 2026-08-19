# Semantic archive prior — 2026-08-19

## Candidate change

The submission now reserves at most one population member for a deterministic semantic median computed from the official Differometor-30k archive. It keeps the feasibility anchor, user-supplied starts, remaining random starts, exact simulator gradients, and restart logic unchanged.

This is the smallest candidate licensed by the offline initializer screen. It is not described as a learned topology model and it is not yet a competition-performance claim.

## Artifact

`submission/semantic_prior.json` contains 247 unit-space semantic-key medians and seven property fallbacks. It was refit on the lowest stored-loss row for each of 11,678 unique size-3 topology identities after the grouped holdout decision was frozen.

- Dataset SHA-256: `149f6aac17aff2e33750b4e1b6cebd3cef1c39d47ae49a3a7ed77315cb7838a7`
- Canonical LF prior SHA-256: `c08cefb94f0285d9681ab8125c23545cc93c7231d0b5aefe849a80c74a4f4312`
- Upstream revision: `d9b1bd7d6f2c4df335bc7725755b02aa5f6f942c`
- Minimum distinct-topology support for any exact semantic key: 453
- Canonical LF size: 18,300 bytes

The submission reads runtime `optimization_pairs`, canonicalizes simple and coupled slots, uses an exact key when known, falls back to the matching property median, validates every value, and maps the unit-space vector through the same logit transform as dfbench's unbounded mode. A missing or malformed file safely leaves that population slot random.

## Runtime allocation

With the default population of eight and no caller-supplied starts:

1. one low-power/low-reflectivity feasibility anchor;
2. one semantic archive median;
3. six seeded random starts.

Caller-supplied starts retain precedence. The prior is controlled by the `use_semantic_prior` optimizer argument so a paired ablation can run the identical optimizer with the slot left random.

## Evidence boundary

The grouped archive screen showed that semantic medians reconstruct held-out stored designs much better than random parameters. That does not establish lower live physics loss. Promotion beyond a candidate requires paired, equal-budget Objective runs on unseen topology identities and accelerator hardware. The relevant target is fewer evaluations or less time to reach a fixed feasible loss; because the competition loss can cross zero, a multiplicative final-loss claim is not meaningful.

## Local smoke boundary

The available JAX runtime reports only `CpuDevice(id=0)`. A three-second scored Constrained Voyager smoke was stopped during pre-clock warmup after more than two wall-clock minutes and roughly 27 aggregate CPU-minutes; it had not reached a simulator evaluation or produced a result artifact. This is evidence that the local machine cannot provide a proportionate live comparison, not evidence for or against the prior. The analytic lifecycle test remains the local mechanics check; the paired prior-on/prior-off physics gate remains open for accelerator execution.
