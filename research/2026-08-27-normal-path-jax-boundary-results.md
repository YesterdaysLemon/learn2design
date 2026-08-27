# Normal-path JAX boundary audit - validated result

Date: 2026-08-27

Study ID: `normal-path-jax-boundary-v1`

Study revision: `ddb119049a195ad9c708c0d3f292b3b342b30f04`

Private immutable result SHA-256:
`eb89f4361611b74865366b4add2663d214a147c5c2c9759eca2423b1081560ce`

## Decision

All seven frozen deterministic CPU cases passed. The authenticated action is:

```text
synthetic_normal_path_jax_boundary_equivalent
```

On the locked Python 3.12.13, JAX/JAXlib 0.9.0.1, x64-enabled CPU runtime,
the experiment-only pure-JAX no-restart transition exactly reproduced the
frozen synthetic batch's typed public observation tree, protected Adam
outputs, member progress, incumbent selection, and complete telemetry event.
Its eager, `jax.jit`, and explicitly lowered/compiled projections were also
byte-identical.

This establishes a systems-mechanics option, not a performance result. It
shows that this one static transition can cross one explicit experiment-owned
JIT boundary without changing its frozen typed projection. It does not show
that a fused submission would run faster, compile less often, use one kernel,
or improve the competition objective.

## Frozen-case results

- `dependency_source_identity`: authenticated `dfbench==0.3.3`, its locked
  wheel, the installed Objective source, protected submission source,
  JAX/JAXlib 0.9.0.1, CPU backend, and x64-enabled configuration.
- `source_boundary_inventory`: the protected optimizer contains two syntactic
  `jax.block_until_ready` sites, two performance-clock sites, three
  device-dependent scalar conversions, four callback invocation sites, and no
  explicit JIT around `optimize` or `_adam_step`. The finite full-batch
  `dfbench.Objective` logging path contains five additional source-visible
  device-dependent host decisions or conversions. These are source sites, not
  an observed kernel or backend-compilation count.
- `normal_path_boundary_trace`: one population-four, dimension-three batch
  made one public transformed evaluation call, one explicit runtime ready
  barrier, two deterministic performance-clock reads, three deterministic
  Objective wall-clock reads, one initial Objective RNG draw, one raw-initial
  callback, one final-initial callback, and one telemetry callback. It used no
  scalar objective call, warmup, partial tail, or restart and consumed exactly
  four evaluations without overshoot.
- `pure_jax_transition_equivalence`: all 46 typed leaves matched exactly: ten
  public observation leaves, nine next-state leaves, and all 27 full-batch
  telemetry leaves. The four Adam outputs were referenced directly through
  the checked-in `_adam_step` on the same frozen inputs. The pure transition's
  source and recursive JAXPR contained no host scalarization, device transfer,
  clock, RNG, NumPy, or callback operation.
- `explicit_jit_lowering`: eager, JIT, and one explicit
  `lower().compile()` projection were byte-identical. The clean JAXPR contained
  no `pure_callback`, `io_callback`, or host-callback primitive. StableHLO and
  recursive JAXPR commitments reproduced across isolated workers.
- `boundary_negative_controls`: all four declared record perturbations changed
  exactly one recursive path, equal bytes with changed dtype or shape changed
  typed identity, and the callback scanner rejected a traced
  `pure_callback` sentinel while accepting the clean transition.
- `process_isolation`: two fresh credential-scrubbed, network-disabled CPU
  workers reproduced the complete non-process projection byte-for-byte.

## Boundary interpretation

The current route is partly compiled and partly host-controlled:

1. `dfbench.Objective` invokes a vmapped, jitted value-and-gradient transform.
2. Its public logger makes device-dependent host decisions before returning.
3. The optimizer then performs its explicit ready barrier and stops the batch
   clock.
4. Feasible selection and restart control make further device-scalar crossings
   on the host.
5. Adam arithmetic is expressed with JAX arrays but is not wrapped in an
   explicit whole-transition JIT.
6. Optional telemetry is a host callback and is not timing-equivalent to the
   packaged default.

The positive result isolates item 5: for the frozen static no-restart batch,
the arithmetic, observation pass-through, incumbent update, and telemetry
construction can live inside a pure compiled JAX function with exact current-
runtime identity. It does not eliminate Objective logging, budget properties,
clock reads, callbacks, partial-tail control, conditional restart RNG, or the
outer Python loop. Those boundaries would need separate designs and invariants
before any integration claim.

## Claim boundary

The result is synthetic, deterministic, one-batch, full-vmap, no-restart,
instrumented, local-CPU, and locked-runtime only. It does not establish:

- a wall-clock, throughput, memory, compilation-count, fusion, or kernel gain;
- accelerator or cross-version numerical equivalence;
- restart, partial-tail, chunked, warmup, or default-callback equivalence;
- real UIFO or official-budget behavior;
- a useful supervised, bandit, RL, or meta-RL controller;
- predictive value in any public diagnostic;
- competition-score or leaderboard improvement.

No official problem, dataset, topology, private trajectory, generated panel,
GPU, provider, paid endpoint, or portal was used. The protected submission tree
and patience-600/no-prior random-start defaults are unchanged. The owner-
uploaded ZIP remained SHA-256
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.

## Repository verification

The focused contract tests passed before the controller execution. The single
full repository verification pass then completed with the expected two skips,
`git diff --check` was clean, and a fresh scratch-only package build produced
SHA-256
`4b7384dd6d401918b9b46ace4e65c3f116c34feeeca825ae25745dfb9e7908bd`.
The controller authenticated the frozen registry, five source hashes, exact
result schema, protected source/tree/artifacts, clean revision, CPU worker,
network block, immutable output, and SHA sidecar. It returned to
`awaiting_study` with failure streak zero and released its lease.

## Next research rung

The next admissible checkpoint is the learning contract, not model training or
candidate integration. Before fitting anything, freeze a synthetic task family
and its topology-independent observation, action, reward, trajectory, split,
and leakage rules. The first falsifiable question should ask whether a small
supervised or surrogate policy can recover a deliberately learnable toy signal
on held-out generator regimes and beat frozen constant/random baselines while
failing the corresponding label-shuffle control.

Only after that contract and those controls pass should a contextual bandit be
considered; meta-RL comes later, if at all. Official-data training, private-
trajectory selection, a submission treatment, native rewrite, accelerator
benchmark, or paid training run remains a separate owner decision.
