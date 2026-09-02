# Feasibility-debt candidate screen v1

This is the isolated implementation surface for the frozen plan in
[`research/2026-09-01-feasibility-debt-candidate-screen-v1-plan.md`](../../research/2026-09-01-feasibility-debt-candidate-screen-v1-plan.md).

It is intentionally separate from `experiments/uifo_paired`. Every older paid
profile, panel, runner, analyzer, and result is terminal and byte-frozen.

The namespace supplies:

- compact path-free canonical receipts and SHA-256 sidecars;
- an append-only private panel generator plus a no-project-import independent
  reconstruction;
- exact A/B/C/D arm identities, including isolated loading of the Round-1 ZIP;
- strict raw-history projection, a separate history-first replay, and opaque
  replay-agreement gates;
- Stage-1-only planning and selection-receipt-gated Stage 2;
- deterministic sealed archives, binary packet handling, the seven-hour
  `T0`/`B0`/charge horizon, and scoped cleanup primitives.

No checked-in file in this namespace is a panel, source/runtime lock, result,
candidate, or authorization. Importing it does not read the official archive,
execute UIFO, provision a provider object, or modify `submission/`.

The private panel command is deliberately closed unless an operator supplies
the explicit authorization flag. Even that flag is only a software tripwire;
the frozen plan still requires the owner's exact data-scope decision before it
may be used. Paid execution, portal actions, candidate integration, merge, and
private outcome access remain separate decisions.
