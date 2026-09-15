# Gateway (HTTP surface + runtime state)

`gateway/router.py` holds every `/api/v1` endpoint in one 4.4k-line module.

## Route order is load-bearing
Specific paths must be declared **before** wildcards such as `/{query_id}`, or the wildcard captures them. A new specific route appended near the bottom looks correct in isolation and is a defect. Locate the wildcard before inserting.

## `/query` never surfaces a raw error
Every failure path returns a friendly answer plus `suggested_questions` — never a stack trace, a bare exception string, or a generic "something went wrong". Also narrow what is caught: a blanket `except Exception` that swallows the cause and logs nothing makes production failures undiagnosable.

## Module map
- `intent_classifier.py` — regex intent + complexity classification (no LLM)
- `followup.py` — classifies a chat message as new question vs revision of the current notebook
- `notebook_verification.py` — schedules execution + adversarial review, merges the verdict into the confidence score
- `verified_notebooks.py` (2.5k lines) — the verified-notebook library, submission/approval flow
- `evidence.py`, `evidence_router.py`, `evidence_signing.py`, `evidence_attestation.py` — typed Standards evidence packages and their signatures
- `github_publisher.py`, `github_webhook.py` — publish approved notebooks to a repo and reconcile back
- `ckan_sites.py` — admin-managed CKAN portal registry; also feeds graph selection (`mem:agents/core`)
- `onboarding_jobs.py` — runs/monitors portal onboarding as supervised child processes
- `session.py`, `chats.py`, `roles.py`, `approved_members.py`, `runtime_settings.py`, `system_prompt.py`, `landing_page.py`, `query_logs.py`, `feedback.py`

## Runtime state is written into the checkout and gitignored
`users.json`, `roles.json`, `github_settings.json`, `landing_settings.json`, `system_prompt_settings.json`, `runtime_settings.json`, `approved_members.json`, `pending_requests.json`, `notebook_verification/`, `query_logs/`, `feedback/`, `chats/`, `verified_notebooks/` — each has an explicit root-anchored `.gitignore` entry. Never add one back to the index; it holds user data and secrets.

`configs/mcp_servers.json` is the exception: **tracked and rewritten at runtime**. Any write to it that could include a credential is a defect.

Writes go through the `storage` singleton, which does atomic temp-file-then-rename (`LocalStorage._atomic_write`) so a crash mid-write cannot truncate an existing state file. Don't bypass it with plain `open(...)`.

## Other standing rules
- MCP server URLs are validated in `mcp/guards.py` against loopback, private, link-local and cloud-metadata addresses. Any new outbound fetch built from user- or admin-supplied input must go through that check.
- HTTP clients are async httpx, initialised lazily and reused. No blocking I/O on an async path, no client per request, no unclosed client.
- `session.py` storage is an in-memory dict cleaned lazily on access. New unbounded per-user/per-session dicts with no expiry are a leak.

## Web UI coupling
`ui/static/js/{app,admin,library,dictionary,mcp,theme}.js` bind to DOM ids in `ui/templates/*.html`. Renaming or removing an id on one side without the other breaks the control silently — no error, just a dead button. The design system is token-based with light and dark themes; a hardcoded hex colour in a template or in `static/css/style.css` is a defect.
