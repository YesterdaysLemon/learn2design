# Feasibility-debt restart clock v3 - pre-result implementation

Date: 2026-09-01

Study ID: `feasibility-debt-clock-v3`

Frozen plan revision:
`a61ba6003ec7cc5de5f41fc0c4349e62364ebd89`

Implementation revision:
`08641baea5c12bd0783103706187815c49d69d40`

Status: exact implementation and hostile pre-result audit complete; no frozen
case, child terminal mode, or parent terminal projection invoked

## Implemented boundary

The V3 candidate is a fresh experiment-owned copy of the protected optimizer.
It is not imported by `submission/`. Its required `progress_mode="total_loss"`
retains the protected loss-driven transition, while
`progress_mode="feasibility_debt"` uses only the public nonnegative penalty
before a lane first becomes finitely feasible. The first finite feasible row
is an unconditional progress event; that generation then irreversibly uses
finite total loss even after infeasible reentry. An authenticated restart
resets the latch, progress incumbent, stall, Adam member state, and generation
together.

The complete auxiliary mapping is authenticated before any candidate-owned
progress or generation state can change, including on a partial tail. The
candidate adds no objective call, random draw, topology input, learned
parameter, private read, submission default, or provider action.

The five-lane fixture implements all nine frozen cases. The chunk case really
assembles each logical population through physical `(2,1,2)` partitions and
separately authenticates the three-versus-nine physical call counts. The
masked-restart case compares the candidate state and RNG root against an
independent frozen-mask replay. The partial-tail case compares every mutable
final state field against a no-tail control. The nonfinite case exercises both
unlatched penalty and latched total-loss selection.

The stdlib-only worker duplicates the inherited result descriptor and redirects
fd 1 before importing result-bearing code. Its probe traverses both Python and
raw-fd noise plus an injected import-time noise module. The terminal parent
will accept a child only when canonical JSON, schema, exact parent revision,
frozen plan identity, and an independently recomputed local source boundary all
match. It uses two fresh CPU children, a credential-scrubbed environment, no
stdin, zero allowed stderr, a 262,144-byte stdout cap, a 180-second timeout,
and exactly fourteen transport Booleans.

## Exact source boundary

The source verifier requires AST identity for every inherited helper except
`optimize`, permits exactly the two new candidate helpers, and pins the full
AST of `optimize`, the auxiliary validator, and the progress transition. The
implementation projection is:

| Field | SHA-256 |
| --- | --- |
| normalized candidate delta | `f2fbe67ac12f80b4ddee05a93ac352dec5d883588f6f7e5aeed68af899c1ad24` |
| complete source boundary | `be929a38a6ebe6a66a3a19ef671a6d2d501f198211f2f9c7cb52df4c388d024f` |

## Committed raw-file hashes

| Path | SHA-256 |
| --- | --- |
| protected `submission/submission.py` | `0fefbaaf18d9831895d788df45c92cbaf4522da7c54d8f78646e449ffa9374c9` |
| `experiments/candidates/feasibility_debt_clock_v3.py` | `ca7abd365c5d1172dab2f47fccdf0afa3df9652e75cc2003385312cec48844d6` |
| `experiments/candidates/feasibility_debt_clock_v3_fixture.py` | `ef5f2353b2324381f2e24a5b790f074396530652e294a41c9340f9783200585f` |
| `experiments/candidates/feasibility_debt_clock_v3_worker.py` | `fa9b64832913a276f7f509d2bc1c050dfdaaa45cc7bcebae4c947ee855038c13` |
| `experiments/candidates/feasibility_debt_clock_v3_source.py` | `a23f0d1a8ce008923ee2baf7e3ccfe7c905b06c9fe77e229a6da3b50d785c084` |
| `tests/test_feasibility_debt_clock_v3.py` | `59944b6c8c86099c5195b3cb31d5f5a01da8cbc57f18ee26c228ec20ba158928` |
| frozen plan | `1bf96ddd42c95dd9aa4ea516b1813929b6835f3949c4feb516fd2d7db62f57b8` |

## Verification receipt

- AST parsing passed for the candidate, fixture, source verifier, worker, and
  focused tests with bytecode writing disabled.
- The exact source projection passed and produced the roots above.
- The focused V3 suite passed `25/25` tests.
- The final affected boundary passed `50/50` tests across
  `tests/test_feasibility_debt_clock_v3.py` and
  `tests/test_candidate_integration.py` in 16.8 seconds.
- The standalone transport probe exited zero, emitted zero stderr, and emitted
  exactly
  `{"probe":"feasibility-debt-clock-v3","stdout_sealed":true}` plus one
  newline despite Python, fd, and import-time noise attempts.
- The focused test source self-audits that it cannot call the case projector,
  child terminal mode, or parent `--run` entry point.
- Frozen case keys, seeds, trace keys, case/table ASTs, candidate method ASTs,
  parent/child schemas, exact source identities, timeout, caps, environment,
  and process boundaries are pinned.
- `git diff --check` was clean before the implementation commit.

No full local repository pass was attempted because the broad suite is known
to enter a retired terminal trace worker. The affected boundary is the retained
local verification surface; draft-PR CI is a separate precondition and is not
a V3 terminal result.

## Contained pre-result corrections

Before any frozen case executed, review corrected only implementation-to-plan
conformance: float32 synthetic arrays avoid forbidden JAX dtype warnings; the
first-feasible assertion checks the frozen first restart rather than forbidding
later generations; chunking, independent mask replay, partial-tail state
comparison, and all four nonfinite paths are exercised rather than asserted;
auxiliary validation covers partial tails without a duplicate per-batch sync;
and the parent binds child identities to the exact local revision and source
root. No plan, case, seed, row generator, patience, threshold, action, or claim
boundary changed.

## Result guard

Neither `_case_projection`, the worker's `--child` mode, nor the parent's
terminal `--run` entry point has been called. There is no V3 case outcome,
child terminal stream, score, generated topology, private outcome, official
data, GPU use, paid endpoint, provider resource, portal action, submission
artifact, or merge.

After this record commits, push the exact review surface and require every
draft-PR job to be green at that revision. If the worktree and revision remain
exact and clean, the next checkpoint is the single frozen local-CPU parent
invocation. A pass opens only a fresh paid candidate-screen plan; a failure
parks V3 permanently.
