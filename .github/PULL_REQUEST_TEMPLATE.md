## What this changes

<!-- What's different, and why. If it fixes a bug, describe the failing scenario. -->

## How it was verified

<!-- Tests added or run, and anything you checked by hand. -->

---

- [ ] `pytest` passes
- [ ] `ruff check src/ tests/ scripts/` is clean
- [ ] Any new agent action appends to `execution_trace` (or this doesn't touch that path)
- [ ] Confidence factors that can't be computed stay `None` with a reason, never `0.0`
- [ ] Template element ids and the JavaScript that binds them were changed together

<!-- Formatting and mypy carry a known backlog and are advisory — see CONTRIBUTING.md.
     Please don't reformat files this PR isn't otherwise touching. -->
