# Feasibility-debt restart clock v1 - terminal result

Date: 2026-09-01

Study ID: `feasibility-debt-clock-v1`

Frozen plan revision:
`197a7433c235ef9cf2e160e8a3bd4a8889d33029`

Invocation revision:
`46e63d47d1856986669e2e6655866a9692143a7c`

Status: terminal invocation failed closed; candidate parked

## Terminal condition

The study was invoked exactly once from a clean worktree after all three jobs
in GitHub Actions run `33566085387` passed. The invocation command was:

```text
uv run --frozen --group dev --group integration python -m experiments.candidates.feasibility_debt_clock_v1_fixture --run
```

The parent process exited `1` after reporting only:

```text
RuntimeError: child exited 1
```

The failure occurred while the parent was handling the first of the two frozen
child projections. The parent emitted no sanitized JSON projection. No second
terminal invocation was made, the child entry point was not called directly,
and no child stdout, stderr, case outcome, raw trajectory, or result-bearing
metric was inspected. A post-invocation process check found zero matching
candidate projection processes.

## Frozen action

The plan assigns any malformed output, case mismatch, nondeterminism, source
mismatch, process effect, or test failure the single action:

```text
park_feasibility_debt_candidate
```

That action now applies. This ID is terminal and must never be rerun, repaired,
or used as an evidence-bearing candidate. Its implementation remains useful
only as an auditable pre-result artifact; it has no validated mechanics result.

## Claim boundary

This terminal condition does not identify which case or boundary failed and
does not establish that the proposed penalty clock helps, harms, or even
executed to completion. It is not a UIFO result, a topology result, a candidate
comparison, an accelerator benchmark, a competition score, or evidence that
the owner-reported Round-1 `0.444293` improved.

The protected `submission/` tree, closed topology panels, private laboratory
controller, official dataset, provider resources, and portal were not touched.
No GPU or paid endpoint was used and no money was spent.

## Next gate

Any successor must use a fresh versioned ID, independently frozen cases and
seeds, and a new pre-result boundary. It may use only the public parent-level
fact that V1 produced no authenticated sanitized projection; it must not
inspect or select against V1 child output, rerun its fixture, reuse its terminal
cases, or claim that the candidate mechanism itself failed.
