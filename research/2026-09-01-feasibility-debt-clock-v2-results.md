# Feasibility-debt restart clock v2 - terminal result

Date: 2026-09-01

Study ID: `feasibility-debt-clock-v2`

Frozen plan revision:
`47efe9d7a55c6f308291b2faa12f160933dce8a5`

Invocation revision:
`04c0a2e3bad43e79f33e89630a52493a56e04f05`

Status: terminal transport failure; V2 parked

## Terminal condition

All three jobs in GitHub Actions run `33572073066` passed against the exact
clean invocation revision. The worktree was clean and no matching fixture
process existed before the one permitted command:

```text
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --group dev --group integration python -m experiments.candidates.feasibility_debt_clock_v2_fixture --run
```

The parent exited `1` after emitting one valid sanitized projection. Both
children started, exited zero, emitted no stderr, and remained inside the
stdout cap. Neither complete stdout stream parsed as the frozen JSON envelope.
Consequently no child schema, study identity, case result, source hash, or
replay root authenticated. The parent returned the frozen closed failure
projection and action `park_feasibility_debt_v2`.

The exact sanitized projection was:

```json
{"action":"park_feasibility_debt_v2","all_cases_passed":false,"candidate_source_sha256":null,"case_count":10,"case_outcomes":{"auxiliary_schema_rejection":false,"chunk_projection_equivalence":false,"nonfinite_progress_semantics":false,"partial_tail_no_transition":false,"post_feasibility_total_loss_handoff":false,"pre_feasibility_lane_routing":false,"restart_state_isolation":false,"source_delta_boundary":false,"total_loss_mixed_restart_identity":false,"total_loss_no_restart_identity":false},"fixture_source_sha256":null,"invocation_revision":"04c0a2e3bad43e79f33e89630a52493a56e04f05","plan_revision":"47efe9d7a55c6f308291b2faa12f160933dce8a5","plan_sha256":"fac7a4f5a0c6624685f761d95be4f13d8b4f3db1695695fbd808cf8ff8c84df9","process_replay_root_sha256":null,"protected_source_sha256":null,"runs_equal":false,"source_boundary_root_sha256":null,"study_id":"feasibility-debt-clock-v2","transport_outcomes":{"child_1_exit_code_zero":true,"child_1_json_parsed":false,"child_1_process_started":true,"child_1_schema_valid":false,"child_1_stderr_empty":true,"child_1_stdout_within_cap":true,"child_1_study_identity_valid":false,"child_2_exit_code_zero":true,"child_2_json_parsed":false,"child_2_process_started":true,"child_2_schema_valid":false,"child_2_stderr_empty":true,"child_2_stdout_within_cap":true,"child_2_study_identity_valid":false}}
```

A post-invocation process check found zero matching fixture processes. The
terminal command was not repeated, neither child entry point was called
directly, and no raw child stdout, case metric, trajectory, or private outcome
was inspected.

## Interpretation

The `false` case values are the parent fallback projection. They do not say
that any named mechanism case ran and failed. The only authenticated
result-bearing facts are the parent transport booleans above: two clean child
exits produced bounded, stderr-free streams that were not whole-stream JSON.
That supports a fresh transport design which seals the result envelope away
from incidental process stdout; it does not authorize repairing or rerunning
V2.

## Frozen action and claim boundary

The frozen action is:

```text
park_feasibility_debt_v2
```

V2 is terminal and must never be rerun, repaired, registered as a passing
mechanics study, or used as candidate-performance evidence. This result does
not establish whether the feasibility-debt transition helps, harms, or
completed any case. It is not a UIFO result, topology result, H100 benchmark,
candidate score, competition score, or evidence that the owner-reported
Round-1 `0.444293` improved.

No GPU, paid endpoint, official dataset, private outcome panel, submission
package, portal, or merge was used. `submission/` and every protected panel
remain unchanged.

## Next gate

Any successor must use a fresh versioned ID, independently frozen source,
family, cases, and seeds. It may use only this sanitized parent-level transport
fact. It must route or suppress incidental stdout before emitting a
parent-authenticated envelope, test that boundary without executing the new
terminal cases, and earn a new clean pre-result revision and green CI before
one new terminal invocation. Paid-panel planning remains closed until such a
successor validates the mechanism.
