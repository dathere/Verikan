# Security Policy

## Reporting a vulnerability

**Please report privately, not in a public issue.**

Use GitHub's private vulnerability reporting: go to the
[Security tab](https://github.com/dathere/Verikan/security/advisories) and choose *Report a
vulnerability*. That opens a channel visible only to maintainers.

Please include what an attacker can do, the steps to reproduce it, and the affected version or
commit. We'll acknowledge the report and keep you updated as we work on it. If you'd like
credit in the advisory, say so and tell us how to name you.

Verikan is alpha software with no long-term support branches — fixes land on `main`.

## What this software does that's worth knowing about

Verikan is a self-hosted application that runs AI-generated code and talks to third-party
services. If you deploy it, these are the parts that matter.

### It executes LLM-generated code

Notebook verification runs each generated notebook to check that its output reproduces the
answer. The notebook is written by a model, and dataset descriptions retrieved from open data
portals reach the generator, so **its contents are untrusted input**. Execution is contained:

- a **subprocess**, never in-process, killed on a timeout;
- a **minimal environment allowlist** — read-only data API keys and nothing else. Anthropic,
  GitHub, Auth0 and evidence-signing credentials are never exposed to it;
- a scratch working directory, with `HOME` pointed away from the real one;
- **shell-escape cells skipped** (`!pip install ...` is a Colab convenience, not analysis);
- an **egress guard** installed into the kernel at interpreter startup that blocks loopback,
  private, link-local, reserved and multicast addresses — so a notebook can still reach
  public data APIs but cannot read a cloud metadata endpoint to steal instance credentials,
  or reach services inside your network.

This is **containment, not a sandbox.** It bounds blast radius; it does not stop determined
arbitrary code execution. If you point Verikan at portals you do not control, run it somewhere
you're willing to have code execute — a locked-down container, gVisor, or a separate worker.
You can turn execution off entirely with `NOTEBOOK_VERIFICATION_ENABLED=false`.

### It connects to MCP servers and open data portals

Admin-registered MCP server URLs are validated against loopback, private, link-local and
cloud-metadata addresses, so a server entry cannot turn the service into an SSRF proxy against
your own network. Local development legitimately needs localhost servers, so
`MCP_ALLOW_PRIVATE_URLS=true` opts out — **leave it off anywhere reachable.**

Data retrieved from third-party portals is treated as data, not instruction, but it does reach
model context. Treat any deployment that indexes portals you don't control accordingly.

### Deployment notes

- A fresh install seeds a local account with a **well-known default password**. Set
  `USER_PASSWORD` before exposing the app to anyone, and `ADMIN_USERS` to control who gets
  admin.
- Admin routes gate on role, and evidence-signing keys are read from the environment as
  `SecretStr`. Don't commit a `.env`; `.gitignore` already excludes it and the runtime state
  files alongside it.
