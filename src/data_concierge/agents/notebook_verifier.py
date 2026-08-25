"""Execute a generated notebook and check that it supports the answer.

Issue #131. Every answer ships with a Colab-ready notebook that, until now,
had never been run. Nothing checked that its code executes, or that the
numbers it produces are the numbers the answer claims. That gap is the main
hallucination surface: a plausible answer and a notebook that does not
actually compute it look identical to the user.

This module closes it with two independent signals:

* **Execution** — run the notebook and see whether it completes.
* **Reconciliation** — pull the numeric claims out of the answer and check
  they appear in the notebook's *own* output, not in the tool transcript the
  answer was written from. This is the part that catches a fabricated figure:
  re-deriving the number from the published code is much stronger evidence
  than string-matching it against text the model already had in context.

A third signal, an adversarial static review of the method, is produced by
``agents/notebook_reviewer`` and merged in by the caller.

Security
--------
The notebook is LLM-generated, and dataset descriptions retrieved from open
data portals reach the generator, so its contents are **untrusted input**.
Execution is therefore:

* in a **subprocess**, never in-process, killed on timeout;
* under a **minimal environment allowlist** — the read-only data API keys the
  notebook legitimately needs and nothing else. ``GITHUB_TOKEN``,
  ``ANTHROPIC_API_KEY``, the Auth0 secret and ``EVIDENCE_SIGNING_KEY_SEED``
  are never exposed, so injected code cannot exfiltrate them or forge an
  attestation;
* in a scratch working directory, with shell-escape cells skipped.

This is containment, not a sandbox. It bounds blast radius; it does not stop
determined arbitrary code execution. Running this on untrusted portals wants
a real jail (gVisor, a locked-down container, or a separate Cloud Run job) —
tracked in #135.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any

from pydantic import BaseModel, Field

from data_concierge.core.logging import get_logger

logger = get_logger(__name__)

# Only these reach the notebook subprocess. Read-only, public statistical
# data APIs. Everything else in the parent environment is withheld.
ALLOWED_ENV_KEYS = frozenset(
    {
        "BLS_API_KEY",
        "CENSUS_API_KEY",
        "BEA_API_KEY",
        "FRED_API_KEY",
        "DATA_COMMONS_API_KEY",
        "CKAN_URL",
        "WPRDC_CKAN_URL",
        "PATH",
        "LANG",
        "LC_ALL",
    }
)

# Cells that shell out are Colab conveniences (`!pip install ...`). They are
# not part of the analysis and we will not run a shell for untrusted input.
_SHELL_ESCAPE = re.compile(r"^\s*!", re.MULTILINE)

# Same numeric grammar the grounding factor uses, so the two agree on what
# counts as a claim: integers, decimals, and comma-grouped thousands.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Bare years and small integers match by coincidence far too often to be
# evidence of anything.
_TRIVIAL = frozenset({str(n) for n in range(0, 101)})


class NotebookVerdict(BaseModel):
    """Outcome of executing a generated notebook and reconciling its output."""

    executed: bool = Field(description="Notebook ran to completion without error")
    execution_error: str | None = Field(default=None, description="First error, truncated")
    cells_total: int = 0
    cells_executed: int = 0
    skipped_shell_cells: int = 0
    timed_out: bool = False

    claimed_values: list[str] = Field(default_factory=list)
    reconciled_values: list[str] = Field(default_factory=list)
    reconciliation_ratio: float | None = Field(
        default=None,
        description="Fraction of the answer's numeric claims found in notebook output; "
        "None when the answer makes no checkable claims",
    )

    score: float | None = Field(default=None, description="None when nothing was measurable")
    reason: str | None = Field(default=None, description="Why the score is None, if it is")

    def as_signals(self) -> dict[str, Any]:
        """Compact debug payload for ``ConfidenceScore.signals``."""
        return {
            "notebook_executed": self.executed,
            "notebook_cells_executed": f"{self.cells_executed}/{self.cells_total}",
            "notebook_timed_out": self.timed_out,
            "notebook_claims": len(self.claimed_values),
            "notebook_reconciled": len(self.reconciled_values),
            "notebook_error": (self.execution_error or "")[:200] or None,
        }


def extract_numeric_claims(answer: str) -> list[str]:
    """Numeric claims in the answer that are worth reconciling.

    Bare years and integers under 101 are dropped — they collide with row
    counts, axis ticks and dates often enough that matching them proves
    nothing.
    """
    claims: list[str] = []
    seen: set[str] = set()
    for raw in _NUMBER.findall(answer or ""):
        token = raw.rstrip(".")
        normalised = token.replace(",", "")
        if normalised in _TRIVIAL:
            continue
        # A 4-digit number in a plausible year range is almost always a year.
        if len(normalised) == 4 and normalised.isdigit() and 1900 <= int(normalised) <= 2099:
            continue
        if normalised in seen:
            continue
        seen.add(normalised)
        claims.append(token)
    return claims


def _strip_shell_cells(nb: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Normalise cell sources and blank out cells that shell out.

    nbformat's JSON form stores ``source`` as a *list of lines*, and that is
    what comes back off storage. nbclient calls ``.strip()`` on it, so every
    cell source is joined to a plain string here first — otherwise execution
    dies with "'list' object has no attribute 'strip'" before a single cell
    runs, and every notebook looks broken when none of them are.
    """
    skipped = 0
    for cell in nb.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
            cell["source"] = source
        if cell.get("cell_type") != "code":
            continue
        if _SHELL_ESCAPE.search(source):
            cell["source"] = (
                "# [verifier] setup cell skipped — shell escapes are not executed\n"
            )
            skipped += 1
    return nb, skipped


def _collect_outputs(nb: dict[str, Any]) -> str:
    """All textual output the executed notebook produced."""
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []) or []:
            if "text" in out:
                text = out["text"]
                chunks.append("".join(text) if isinstance(text, list) else str(text))
            data = out.get("data", {})
            for mime in ("text/plain", "text/html", "text/markdown"):
                if mime in data:
                    val = data[mime]
                    chunks.append("".join(val) if isinstance(val, list) else str(val))
    return "\n".join(chunks)


# Runs inside the subprocess. Reads {"nb": ..., "timeout": ...} on stdin and
# writes the executed notebook (or an error) as JSON on stdout.
_RUNNER = r"""
import json, sys
payload = json.load(sys.stdin)
try:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
except Exception as e:
    json.dump({"ok": False, "error": f"verifier deps unavailable: {e}", "kind": "setup"}, sys.stdout)
    sys.exit(0)

nb = nbformat.from_dict(payload["nb"])
client = NotebookClient(
    nb,
    timeout=payload["cell_timeout"],
    kernel_name="python3",
    allow_errors=False,
    record_timing=False,
)
try:
    client.execute()
    json.dump({"ok": True, "nb": nb}, sys.stdout, default=str)
except CellExecutionError as e:
    json.dump({"ok": False, "error": str(e)[:2000], "kind": "cell", "nb": nb}, sys.stdout, default=str)
except Exception as e:
    json.dump({"ok": False, "error": f"{type(e).__name__}: {e}"[:2000], "kind": "other"}, sys.stdout)
"""


# Written into the scratch dir and put on PYTHONPATH so Python auto-imports it
# at interpreter startup — including the Jupyter kernel the notebook runs in,
# before any cell executes and with no way for a cell to skip it. It blocks
# outbound connections to loopback / private / link-local / metadata addresses
# while leaving the public internet reachable, so a notebook can still fetch
# census.gov or datacommons but cannot read the cloud metadata endpoint
# (169.254.169.254) to steal the instance's service-account token, nor reach
# internal VPC services. This is the runtime half of the #135 containment that
# makes it safe to execute LLM-generated notebooks in production.
_SITECUSTOMIZE = r'''
import ipaddress
import socket

_orig_getaddrinfo = socket.getaddrinfo
_orig_connect = socket.socket.connect


def _blocked(host):
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if (ip.is_loopback or ip.is_link_local or ip.is_private
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
        return str(ip)
    return None


def _guarded_getaddrinfo(host, *a, **kw):
    infos = _orig_getaddrinfo(host, *a, **kw)
    for info in infos:
        addr = info[4][0]
        if _blocked(addr):
            raise OSError(
                "notebook-verifier: blocked connection to internal address %s" % addr
            )
    return infos


def _guarded_connect(self, address):
    try:
        host = address[0]
    except (TypeError, IndexError):
        host = None
    if host and _blocked(host):
        raise OSError(
            "notebook-verifier: blocked connection to internal address %s" % host
        )
    return _orig_connect(self, address)


socket.getaddrinfo = _guarded_getaddrinfo
socket.socket.connect = _guarded_connect
'''


def _subprocess_env(workdir: str) -> dict[str, str]:
    """Minimal allowlisted environment for the notebook subprocess.

    ``HOME`` is deliberately *not* inherited. It is pointed at the scratch
    directory instead, which both isolates the notebook from the real home
    and avoids depending on the container having a writable one — the
    Dockerfile creates ``appuser`` with a home directory but never sets
    ``HOME``, so the inherited value may not be writable, and the Jupyter
    kernel needs somewhere to put its runtime and connection files.
    """
    env = {k: v for k, v in os.environ.items() if k in ALLOWED_ENV_KEYS}
    env["HOME"] = workdir
    env["JUPYTER_RUNTIME_DIR"] = os.path.join(workdir, ".jupyter-runtime")
    env["JUPYTER_DATA_DIR"] = os.path.join(workdir, ".jupyter-data")
    # Scratch dir first on the path so our sitecustomize is the one Python
    # auto-imports, in the kernel subprocess as well as the runner.
    env["PYTHONPATH"] = workdir
    # Keep the interpreter quiet and deterministic.
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"  # never try to open a display
    return env


def _install_egress_guard(workdir: str) -> None:
    """Drop the egress guard where the kernel will auto-import it."""
    with open(os.path.join(workdir, "sitecustomize.py"), "w") as fh:
        fh.write(_SITECUSTOMIZE)


def verify_notebook(
    notebook: dict[str, Any],
    answer: str,
    *,
    total_timeout: int = 180,
    cell_timeout: int = 60,
) -> NotebookVerdict:
    """Execute ``notebook`` and check it supports ``answer``.

    Never raises: a verifier that fails must degrade to "could not measure"
    rather than take down the query that triggered it.
    """
    claims = extract_numeric_claims(answer)

    try:
        nb = json.loads(json.dumps(notebook))  # deep copy, and rejects non-JSON input
    except (TypeError, ValueError) as e:
        return NotebookVerdict(
            executed=False,
            execution_error=f"notebook is not JSON-serialisable: {e}",
            claimed_values=claims,
            score=None,
            reason=(
                "Notebook verification could not run because the generated notebook "
                "was not readable"
            ),
        )

    nb, skipped = _strip_shell_cells(nb)
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]

    if not code_cells:
        return NotebookVerdict(
            executed=False,
            cells_total=0,
            skipped_shell_cells=skipped,
            claimed_values=claims,
            score=None,
            reason=(
                "Notebook verification could not run because the generated notebook "
                "contains no executable code"
            ),
        )

    with tempfile.TemporaryDirectory(prefix="nbverify-") as workdir:
        _install_egress_guard(workdir)
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, allowlisted env
                [sys.executable, "-c", _RUNNER],
                input=json.dumps({"nb": nb, "cell_timeout": cell_timeout}),
                capture_output=True,
                text=True,
                timeout=total_timeout,
                cwd=workdir,
                env=_subprocess_env(workdir),
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return NotebookVerdict(
                executed=False,
                timed_out=True,
                cells_total=len(code_cells),
                skipped_shell_cells=skipped,
                claimed_values=claims,
                execution_error=f"exceeded {total_timeout}s",
                # A timeout IS a measurement: the notebook does not complete.
                score=0.0,
                reason=None,
            )
        except Exception as e:  # noqa: BLE001 - verifier must never break the query
            logger.warning("Notebook verifier subprocess failed", error=str(e))
            return NotebookVerdict(
                executed=False,
                cells_total=len(code_cells),
                skipped_shell_cells=skipped,
                claimed_values=claims,
                execution_error=str(e)[:500],
                score=None,
                reason=(
                    "Notebook verification could not run because the verifier itself "
                    "failed to start"
                ),
            )

    try:
        result = json.loads(proc.stdout or "{}")
    except ValueError:
        result = {}

    if not result:
        return NotebookVerdict(
            executed=False,
            cells_total=len(code_cells),
            skipped_shell_cells=skipped,
            claimed_values=claims,
            execution_error=(proc.stderr or "no output from verifier")[:500],
            score=None,
            reason=(
                "Notebook verification could not run because the verifier produced no result"
            ),
        )

    if result.get("kind") in ("setup", "other"):
        # Not the notebook's fault: "setup" is a missing dependency, "other"
        # is the harness itself failing before or around execution. Either way
        # we learned nothing about the notebook, so report unmeasurable rather
        # than scoring it zero — a verifier bug must never be presented to the
        # user as "this answer's notebook is broken".
        return NotebookVerdict(
            executed=False,
            cells_total=len(code_cells),
            skipped_shell_cells=skipped,
            claimed_values=claims,
            execution_error=result.get("error"),
            score=None,
            reason=(
                "Notebook verification could not run because the execution environment "
                "is unavailable"
            ),
        )

    executed = bool(result.get("ok"))
    executed_nb = result.get("nb") or {}
    outputs = _collect_outputs(executed_nb) if executed_nb else ""

    reconciled: list[str] = []
    ratio: float | None = None
    if claims:
        flat = outputs.replace(",", "")
        for claim in claims:
            if claim in outputs or claim.replace(",", "") in flat:
                reconciled.append(claim)
        ratio = len(reconciled) / len(claims)

    verdict = NotebookVerdict(
        executed=executed,
        execution_error=None if executed else (result.get("error") or "")[:2000] or None,
        cells_total=len(code_cells),
        cells_executed=len(code_cells) if executed else 0,
        skipped_shell_cells=skipped,
        claimed_values=claims,
        reconciled_values=reconciled,
        reconciliation_ratio=ratio,
    )
    verdict.score, verdict.reason = _score(verdict)
    return verdict


def _score(v: NotebookVerdict) -> tuple[float | None, str | None]:
    """Combine execution and reconciliation into one factor.

    Execution is worth 40% and reconciliation 60% — a notebook that runs but
    produces different numbers than the answer claims is a worse failure than
    one that simply errors, because it looks convincing.

    When the answer makes no numeric claims there is nothing to reconcile;
    the factor falls back to execution alone rather than inventing a value.
    """
    if not v.executed:
        # Whether or not there were claims: it did not run, and that is a
        # measured failure, not an absence of measurement.
        return 0.0, None

    if v.reconciliation_ratio is None:
        # Ran cleanly, but the answer makes no numeric claim to re-derive, so
        # only the execution half of this factor is evidenced.
        return 0.6, None

    return round(0.40 + 0.60 * v.reconciliation_ratio, 4), None
