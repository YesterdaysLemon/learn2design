# Full-surface prefix audit - validated result

Date: 2026-08-27

Study ID: `full-surface-prefix-indistinguishability-v1`

Study revision: `f941ad324d34509b5136fb04c5465bf674b835aa`

Private immutable result SHA-256:
`c1a1ec7cb7b40ea106f42a0511ccbc9a0b70174621db7d7f51c076db1a583ef5`

## Decision

All nine frozen deterministic CPU cases passed. The authenticated action is:

```text
synthetic_full_surface_prefix_twin_confirmed
```

For the frozen synthetic adapter, the two protected optimizer executions had
byte-identical complete allowed snapshots through `B = 8` and differed at the
next feasibility observation. This establishes a finite information-boundary
counterexample only. It does not show that the visible diagnostics lack
distributional value or that a programmed or learned policy improves the
competition objective.

## Frozen-case results

- `normal_path_execution`: both worlds ran the checked-in
  `BatchedRestartAdam.optimize` path for nine complete population-four batches
  and 36 evaluations. Each had one optimizer RNG draw, nine complete public
  telemetry events, nine chained transition commitments, zero scalar calls,
  and zero restart events.
- `adapter_schema`: every snapshot matched the exact eight-field schema:
  candidate, loss, total gradient, complete aux tree, optimizer-transition
  commitment, RNG transcript, budget counters, and incumbent state. Array
  identity included exact dtype, shape, and SHA-256.
- `shared_full_surface_prefix`: both eight-batch prefixes were finite, strictly
  improving member-wise, infeasible, schema-valid, and byte-identical. At batch
  nine, `aux.is_feasible` was the only primary evaluation leaf that differed;
  the complete downstream snapshots then differed as expected.
- `signal_class_negative_controls`: all seven non-auxiliary signal-class
  controls changed exactly their declared recursive path.
- `aux_leaf_negative_controls`: each of the seven auxiliary leaves changed
  exactly its declared recursive path.
- `typed_array_metadata_boundary`: holding bytes fixed while changing dtype or
  shape changed the typed projection identity in both controls.
- `forbidden_extension_rejection`: all seven sentinels for unlogged callables,
  extra evaluations, Hessians, manual logs, private attributes, saved records,
  and structural metadata were rejected by the exact schema. This was a schema
  closure check, not an attempt to exercise those APIs.
- `action_vector_exhaustion`: all `2^8 = 256` abstract binary action
  transcripts were exhausted. The 255 transcripts containing a restart met
  the bounded-restart obligation but did not preserve the late crossing; the
  all-keep transcript preserved it but did not restart. There was no joint
  satisfier.
- `process_isolation`: two fresh credential-scrubbed, network-disabled CPU
  workers reproduced the complete non-process projection byte-for-byte.

The controller authenticated the frozen registry, five source hashes, result
schema, case fields, success action, protected submission source/tree, clean
revision, and CPU worker. It returned to `awaiting_study` with failure streak
zero and released its lease.

## Interpretation

This closes the current universal-certificate question. Even after adding the
actual candidate parameters, total gradient, every public UIFO-shaped aux
leaf, deterministic transition inputs, RNG transcript, budget, timing window,
and incumbent state, a finite shared prefix need not determine the next
feasibility bit. Any useful controller therefore needs assumptions learned or
programmed from a distribution, not a theorem that these finite observations
always certify the next crossing.

The optimizer-state field is deliberately narrow. It is a chained commitment
to the public telemetry and allowed deterministic transition inputs, not a
private dump of Adam moments or an assertion that arbitrary hidden state was
observed. The fixture used existing public callbacks and never monkeypatched a
private optimizer attribute or read a saved Objective history.

## Claim boundary

The result is synthetic, deterministic, finite-bound, and local-CPU only. It
does not establish that a twin occurs on real UIFO trajectories, that any aux
field is useless, that restarts are harmful, or that RL, supervised learning,
native code, kernel fusion, or a different optimizer improves score or
runtime. It used no official problem, topology, private trajectory, generated
panel, candidate comparison, GPU, provider, paid endpoint, or portal action.

The submitted patience-600/no-prior random-start package and its defaults are
unchanged. The result does not reopen Stage A or Stage B and does not authorize
candidate integration, official-data training, accelerator work, or spending.

## Repository verification

The focused full-surface tests passed before the single retained controller
execution. `uv sync --frozen --group dev --group integration` completed, the
one full repository test pass succeeded with the expected skips, and
`git diff --check` was clean. A fresh scratch build produced archive SHA-256
`4b7384dd6d401918b9b46ace4e65c3f116c34feeeca825ae25745dfb9e7908bd`.
The protected owner-uploaded artifact was not overwritten.

## Next research rung

The next admissible checkpoint is systems mechanics, not a treatment. Freeze a
synthetic one-batch execution boundary and inventory compilation, dispatch,
host conversion, host-device synchronization, callback, RNG, budget, and
timing transitions on the current protected path. Then ask whether an
experiment-only pure-JAX state transition can reproduce the same typed public
observations without changing submission code.

Any local CPU timing is diagnostic only. A later learning rung must separately
freeze the topology, observation, action, reward, trajectory, split, and
leakage contract; establish toy controls; and test a supervised or surrogate
baseline before considering contextual bandits or meta-RL. A native rewrite,
official-data training, candidate change, accelerator benchmark, or paid RL
run remains a separate owner decision.
