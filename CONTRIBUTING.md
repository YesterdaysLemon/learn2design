# Contributing

Start with the [current handoff](docs/CURRENT_HANDOFF.md), then read the README
nearest the code you intend to change. Small, falsifiable contributions are
preferred over broad optimizer rewrites.

## Local setup

```bash
uv sync --frozen --group dev --group integration
uv run --frozen --group dev --group integration pytest -q
python tools/build_submission.py
```

The fast tests do not run the expensive UIFO simulator. Label CPU, smoke, and
accelerator evidence separately.

## Where changes belong

- `submission/`: files shipped in the competition ZIP;
- `experiments/`: bounded comparisons and their executable harnesses;
- `research/`: dated plans, literature notes, results, and decisions;
- `docs/`: concise contributor and operating guides;
- `artifacts/generated/`: ignored machine output, never hand-edited or committed.

## Pull-request checklist

- State the question or failure being addressed.
- Keep the comparison, seeds, budgets, and stopping rule explicit.
- Add focused regression tests and report exact commands run.
- Record hardware and raw-artifact locations for performance claims.
- Confirm `python tools/build_submission.py` still produces a valid bundle.
- Do not promote a method from an offline proxy or a single topology.

When a result changes the project direction, add a dated research record and
make the smallest corresponding update to the current handoff.
