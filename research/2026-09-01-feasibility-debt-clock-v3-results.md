# Feasibility-debt restart clock v3 - terminal result

Date: 2026-09-01

Study ID: `feasibility-debt-clock-v3`

Frozen plan revision:
`a61ba6003ec7cc5de5f41fc0c4349e62364ebd89`

Invocation revision:
`6d2a36f5a41687dd8e38a2aa15120c84dc9b535b`

Status: passed; terminal; never rerun

## Terminal gate

All three jobs in GitHub Actions run `33577681677` passed against the exact
clean invocation revision. Immediately before execution, the local, origin,
and draft-PR heads matched; the worktree was clean; no V3 process or earlier
V3 result record existed; the protected source, plan, and complete source-root
hashes matched the pre-result record.

The one permitted command was:

```text
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --group dev --group integration python -m experiments.candidates.feasibility_debt_clock_v3_fixture --run
```

It exited zero in 7.7 seconds. Both fresh CPU workers started and exited zero,
emitted empty stderr, stayed beneath the stdout cap, produced valid closed
schemas with the exact study identity, and returned byte-identical canonical
results. No matching process remained after the parent exited.

## Frozen result

All nine cases passed:

- `pre_feasible_penalty_routing`
- `first_feasible_irreversible_handoff`
- `post_handoff_infeasible_reentry`
- `auxiliary_nonfinite_fail_closed`
- `masked_restart_state_rng_alignment`
- `partial_tail_transactionality`
- `chunk_partition_trace_equivalence`
- `protected_composite_trace_identity`
- `source_delta_and_process_seal`

The authenticated roots and source identities are:

| Object | SHA-256 |
| --- | --- |
| Process replay root | `c440ede7eb07afd889f907bc3326df55b01c3af473a729de3933fa3d65590b8d` |
| Source boundary root | `be929a38a6ebe6a66a3a19ef671a6d2d501f198211f2f9c7cb52df4c388d024f` |
| Protected source | `0fefbaaf18d9831895d788df45c92cbaf4522da7c54d8f78646e449ffa9374c9` |
| Candidate source | `ca7abd365c5d1172dab2f47fccdf0afa3df9652e75cc2003385312cec48844d6` |
| Fixture source | `ef5f2353b2324381f2e24a5b790f074396530652e294a41c9340f9783200585f` |
| Worker source | `fa9b64832913a276f7f509d2bc1c050dfdaaa45cc7bcebae4c947ee855038c13` |
| Frozen plan | `1bf96ddd42c95dd9aa4ea516b1813929b6835f3949c4feb516fd2d7db62f57b8` |

The frozen action is:

```text
approve_feasibility_debt_v3_for_fresh_candidate_screen_planning
```

## Interpretation and boundary

This validates the exact experiment-owned five-lane transition and sealed
fresh-process transport. Before a lane first becomes finite and feasible, the
candidate can use the exact public auxiliary penalty as its restart-progress
clock; the first feasible row irreversibly hands that generation to ordinary
total loss; later infeasible rows do not switch the clock back. Masked
restarts, partial tails, nonfinite inputs, chunk boundaries, RNG state, and the
protected `total_loss` route matched their frozen invariants.

This is mechanics evidence only. It does not establish lower UIFO loss, H100
value, topology generalization, an optimal patience, candidate promotion, or
an improvement over the owner-reported Round-1 score `0.444293`. V3 is now
terminal and must never be rerun, repaired, or selected against this result.

No GPU, paid endpoint, official dataset, private outcome panel, submission
package, portal, or merge was used. `submission/` and every protected panel
remain unchanged.

## Next gate

Freeze a fresh, archive- and prior-panel-disjoint, paired H100 candidate-screen
plan before accessing official topology identities or spending money. The plan
must bind the exact Round-1 lifecycle as control, the V3 candidate source and
settings, all challenger bundles, panel generation, run order, score
projection, selection and promotion rules, hardware/runtime identity, and a
hard dollar/time cleanup envelope. The development-panel number must remain
distinct from a genuinely comparable hidden Round-2 score. Official-data
access and paid provisioning require a new explicit owner approval after that
contract is frozen.
