# Feasibility-debt candidate screen V1: pre-result wind-down

Date: 2026-09-01

## Disposition

The owner asked the active Round-2 research program to wind down before any
official-topology access, provider action, smoke, scored run, or result
observation. Work stopped at a pre-result implementation checkpoint on branch
`codex/h100-feasibility-debt-screen-v1-impl`, stacked on frozen plan commit
`2a8cd4c4800485ad6564d6702c741a221507af07`.

This checkpoint is **parked and not executable evidence**. It does not change
the retained patience-600/no-prior/random-start submission, authenticate a new
competition score, authorize a candidate, or support a claim that the local
screen would move the owner-reported `0.444293` hidden score toward `0.14`.

No official topology string was read. No private panel or result was generated.
No Runpod credential was accessed, no cloud object was created, no GPU was
invoked, and no money was spent. The scheduled two-hour local laboratory was
already paused and remains paused.

## Preserved implementation state

The checkpoint preserves an experiment-owned package beneath
`experiments/feasibility_debt_candidate_screen/`, its focused tests, and the
private-panel builder entry point. The implemented surface includes:

- exact frozen arm/config/result contracts and independent Stage-1/Stage-2
  analysis paths;
- outcome-blind panel construction, split commitment, sealing, and exact
  topology/config/selection-token replay;
- source/runtime locks, cold-smoke and scored-worker boundaries, bounded packet
  capture, archive and evidence evacuation, and terminal sealing;
- a zero-retry Runpod REST-v2 transport with strict JSON, quote/inventory/create
  receipts, durable create intent, billing-window validation, and partial
  transcript preservation;
- an owner-host finalizer and detached watchdog prototype with one-delete
  ownership and a fail-safe cleanup claim.

The exact affected verification command was:

```powershell
uv run --frozen --group dev --group integration pytest -q tests\test_feasibility_debt_screen_analysis.py tests\test_feasibility_debt_screen_contract.py tests\test_feasibility_debt_screen_host_finalizer.py tests\test_feasibility_debt_screen_orchestrator.py tests\test_feasibility_debt_screen_panel.py tests\test_feasibility_debt_screen_provider.py tests\test_feasibility_debt_screen_runtime.py tests\test_feasibility_debt_screen_watchdog.py tests\test_feasibility_debt_screen_worker.py
```

It passed `60/60` on 2026-09-01. This is focused implementation verification,
not a full-repository pass, provider integration test, accelerator result, or
candidate-performance result.

## Unresolved pre-result blockers

The parked implementation must not be provisioned or used as a result-bearing
screen until all of the following are resolved without relaxing the frozen
scientific contract:

1. Add and independently audit a worker-side NVIDIA receipt that authenticates
   the exact H100 model/count, MIG-disabled state, driver, CUDA/runtime binding,
   and private device-UUID digest before smoke and every scored dispatch. The
   current JAX device-kind check alone is insufficient.
2. Reconcile Runpod REST-v2's provider disk unit (`GB`) with the current
   `ephemeral_disk_gib` receipt field and prove the billed disk quantity and
   all-in bound without unit ambiguity.
3. Close post-delete evidence and account/task billing reconciliation, including
   a fail-closed relation between the deleted pod ID, the final inventory, the
   pod billing window, and any account aggregate used only as a cross-check.
4. Complete the concrete top-level provider/supervisor wiring and hostile tests
   for non-retry status handling, ambiguous create cleanup, partial transcripts,
   owner-host death, and watchdog/host-finalizer races.
5. Obtain fresh independent hostile audits of scientific panel/config replay,
   the provider/cost boundary, and the finalizer/watchdog/terminal-seal path.
6. Build the real source/runtime locks at a clean committed revision, run at
   most one full repository verification pass, push the review surface, and
   require green CI before any later owner gate.

Runpod REST v2 exposes no provider-native pod TTL. The detached owner-host
watchdog can make one bounded deletion attempt, but it cannot guarantee the
`$25` ceiling during a simultaneous owner-host or network failure and provider
control-plane outage. A later paid decision must explicitly accept that
residual best-effort risk or choose a provider with an enforceable native TTL;
the frozen cost cap must not be described as guaranteed otherwise.

## Takeover gate

Do not resume automatically. A future owner-directed continuation should first
re-read this record, the frozen plan, `AGENTS.md`, and
`docs/CURRENT_HANDOFF.md`; confirm this exact checkpoint and a clean worktree;
then resolve the six blockers above without touching `submission/`, terminal
studies, rejected fixtures, or prior panels.

Only after a clean audited implementation exists should the owner be asked for
two fresh, separate decisions:

1. one read of only `entries.topology_string` from the exact official archive
   hash to create and privately back up the disjoint panel while committing only
   sanitized commitments; and
2. conditional paid provisioning for one loss-blind smoke and, only if it
   passes, one no-retry H100 screen under a newly verified cost/risk envelope.

Portal interaction, private outcome access, candidate promotion, packaging,
merge, and submission remain separate gates.
