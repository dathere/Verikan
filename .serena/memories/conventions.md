# Conventions

The behavioural invariants (GraphState, two graphs, `execution_trace`, notebook codegen, confidence `None`, route order, `/query` error handling, UI id/theme coupling) are stated authoritatively in `CONTRIBUTING.md` § "Things that will bite you" and, in more detail, in `.roborev.toml` `review_guidelines`. Read those. What follows is what they do not cover.

## `(str, Enum)` must not become `StrEnum`
`UP042` is disabled in `[tool.ruff.lint] ignore` with a written rationale. With `(str, Enum)`, `str(X.MEMBER)` and f-strings render `"X.MEMBER"`; `StrEnum` renders the *value*. These enums are serialised through Pydantic models and interpolated into prompts, notebooks and logs, so the rewrite silently changes output. Do not "clean up" this lint, and do not re-enable the rule as part of an unrelated change.

## Lint / format / type policy is deliberately asymmetric
- `ruff check` is a CI gate and is currently clean. Keep it clean.
- `ruff format --check` and `mypy src/` are **advisory** (`continue-on-error: true` in the `types` job) — they carry a pre-open-source backlog (~90 mypy errors, mostly missing annotations).
- **Do not reformat files you are not otherwise changing.** A repo-wide `ruff format` would bury real changes in noise; that is precisely why it is not a gate. Match surrounding style instead.
- Don't add *new* type errors. Fixing ones in code you touch is welcome.
- `[tool.mypy]` sets `disallow_untyped_defs = true` — new code should be fully annotated even though the gate is advisory.

## Style
- Imports: ruff isort with `known-first-party = ["data_concierge"]`.
- Logging: `structlog` via `core.logging.get_logger(__name__)`, keyword event fields (`logger.warning("...", error=str(exc))`), not f-strings.
- Secrets are `SecretStr` in `core/config.py`. Never log or serialise one; never add a default that points a fresh install at someone else's infrastructure.
- Commit messages explain *why* in prose. No conventional-commit prefix required.

## Tests
- Prefer real classifiers and state objects over mocks; mock only the outbound boundary (`respx` for httpx). If a change can only be tested with a live key or running service, the test is at the wrong boundary.
- A bug fix should come with a test naming the failing scenario.

## Automated PR review
`.roborev.toml` drives an adversarial reviewer on every PR (`.github/workflows/roborev.yml`). It is a reviewer, not a gate — a finding is a prompt to think. Locally: `roborev review --dirty` or `--branch`.
