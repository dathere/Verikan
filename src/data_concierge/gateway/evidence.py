"""Typed Standards `datHere` evidence-package builder.

Turns a verified notebook + its captured ``agent_log`` into a Typed Standards
evidence package under the **`datHere` content profile** (Open Evidence
Standard / Typed Standards Specification §8.7), the profile defined for exactly
this project — the datHere / WPRDC pilot.

In this model the *notebook itself* is the published, signed evidence artifact:
the cryptographic envelope and commitment view live in the notebook's root
``metadata["org.civicaitools.evidence"]`` namespace (spec §8.8.2), and the
package is verifiable against datHere's own trust registry — independent of the
git host and of civicaitools.org.

The package content is organized as the **A–G envelope** (spec §8.7), a mapping
over the standard's top-level fields:

    A  prompt.text                              the user's question, verbatim
    B  skillMetadata.skillText                  the system prompt(s) in force
    C  cost.model + extensions[...environment]  model card + environment
    D  trace + queries[]                        deliberative trace
    E  extensions[...notebook]                  the answer notebook (nbformat)
    F  output                                   the rendered answer
    G  summary                                  short citation-ready summary

Most of A–D comes straight from the ``agent_log`` captured in
``agents/llm_agent.py`` (``session_start`` carries the system prompt + tools +
model; ``llm_response`` carries verbatim text + token usage incl. cache tokens;
``tool_execution`` carries each verbatim tool call + result + source +
operation type).

**Signing is deliberately pluggable and OFF by default.** Producing a
*verifiable* package additionally requires (a) an Ed25519ph signing key for
datHere, (b) FreeTSA RFC-3161 timestamping, (c) a Sigstore Rekor inclusion
proof, and (d) a published trust registry at datHere's
``/.well-known/typed-publisher.json``. Those are infra/secret decisions; until
they land, :class:`UnsignedSigner` produces a structurally-complete but
unsigned commitment view (clearly marked ``dev-unsigned``). The cryptographic
seam is :class:`EvidenceSigner` — swap in a real signer with no change to the
builder. The single canonicalization swap point is :func:`canonicalize_jcs`.
"""

from __future__ import annotations

import copy
import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Spec constants (Typed Standards Specification, schema 0.1.0)
# ---------------------------------------------------------------------------
# The spec revision this implementation was verified against. Pin the tag
# rather than tracking main: the vocabulary settlement landed across
# v0.1.5-v0.1.7, and Appendix J of this tag carries the prior-era -> settlement
# mapping plus the dual-era rules (J.4 rule 1: an identifier frozen inside an
# already-signed artifact is never rewritten -- which is why the
# ``datHere-evidence:`` package-id literal must never be "fixed").
SPEC_REVISION = "v0.1.7-typed-standards-spec"
SCHEMA_VERSION = "0.1.0"
CONTENT_PROFILE = "datHere"
PRODUCER_PROFILE = "ai-assisted-analysis/datHere"
NODE_TYPE = "content/analysis/v1"
# §8.2: the off-log content (the notebook) is canonicalized under this rule.
CONTENT_CANONICALIZATION = "https://typedstandards.org/canonicalization/dathere-ag-jupyter/v1"
EVIDENCE_NS = "org.civicaitools.evidence"
NOTEBOOK_NS = "org.civicaitools.notebook"
ENVIRONMENT_NS = "org.civicaitools.environment"
# Our content is captured verbatim from the Anthropic response stream inside
# the server agent loop — the analog of the website chat flow's stream capture.
CAPTURE_METHOD = "chat-flow-stream"
# Identifies datHere as the publishing host in the environment metadata and the
# trust-registry URL the commitment view points readers at. Overridable via env.
_DEFAULT_EVIDENCE_HOST = "data-concierge.dathere.com"
# host[:port] only — this value is interpolated into URLs that surface in the
# admin UI, so reject anything that isn't a bare hostname (no quotes, slashes,
# or scheme) and fall back to the default. Defense-in-depth: the host is
# operator config, not user input, but a malformed value must not flow into a
# served URL unescaped.
_HOST_RE = re.compile(r"^[A-Za-z0-9.-]+(:\d+)?$")


def _validated_evidence_host(raw: str | None) -> str:
    value = (raw or "").strip()
    if value and _HOST_RE.match(value):
        return value
    if value:
        logger.warning("Ignoring invalid evidence_host; using default", value=value)
    return _DEFAULT_EVIDENCE_HOST


EVIDENCE_HOST = _validated_evidence_host(getattr(settings, "evidence_host", None))
TRUST_REGISTRY_URL = (
    getattr(settings, "evidence_trust_registry_url", None)
    or f"https://{EVIDENCE_HOST}/.well-known/typed-publisher.json"
)

_SUMMARY_MAX_CHARS = 280


# ---------------------------------------------------------------------------
# Commitment / package serving (spec §8.8 — verifier-facing endpoints)
# ---------------------------------------------------------------------------
# The typedstandards.org verifier resolves a hosted URL to a commitment-view
# JSON (a `/commitment` endpoint or a bare commitment object), then fetches the
# canonical package from the commitment's `packageUrl` — both as application/json
# over CORS. A GitHub-raw URL serves text/plain and is not a commitment, so we
# serve both from our own host and store them at publish time, keyed by the
# package (envelope) hash.
def evidence_package_storage_key(package_hash: str) -> str:
    return f"evidence/packages/{package_hash}.json"


def evidence_commitment_storage_key(package_hash: str) -> str:
    return f"evidence/commitments/{package_hash}.json"


def package_endpoint_url(package_hash: str) -> str:
    """The canonical-package URL the commitment view's ``packageUrl`` points at."""
    return f"https://{EVIDENCE_HOST}/api/evidence/{package_hash}/package"


def commitment_endpoint_url(package_hash: str) -> str:
    """The commitment-view URL a reader verifies (``…/verify?url=`` target)."""
    return f"https://{EVIDENCE_HOST}/api/evidence/{package_hash}/commitment"


# ---------------------------------------------------------------------------
# RFC 8785 JSON Canonicalization Scheme (JCS) — spec §8.2
# ---------------------------------------------------------------------------
# The envelope hash is SHA-256 over the JCS bytes of the unsigned package, and
# the Ed25519ph signature covers that hash. This is the ONE place canonical
# bytes are produced; when real signing lands, this function must be exact JCS
# (object keys sorted by UTF-16 code units, compact separators, UTF-8 output,
# ECMAScript number formatting). The implementation below is faithful for the
# ASCII-keyed, JSON-safe values these packages contain.


def _jcs_number(value: int | float) -> str:
    """Serialize a number per ECMAScript ``Number::toString`` (JCS §3.2.2.3)."""
    if isinstance(value, bool):  # bool is an int subclass — guard first
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("NaN/Infinity are not valid JCS numbers")
    if value.is_integer():
        return str(int(value))
    # Python's repr is shortest-round-trip (since 3.1), matching ES6 for the
    # decimal magnitudes these packages use (temperatures, ratios).
    return repr(value)


def _jcs_string(value: str) -> str:
    """Serialize a string with minimal JSON escaping, UTF-8 (no \\u for non-ASCII)."""
    out = ['"']
    for ch in value:
        code = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _jcs_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return _jcs_number(value)
    if isinstance(value, str):
        return _jcs_string(value)
    if isinstance(value, list | tuple):
        return "[" + ",".join(_jcs_value(v) for v in value) + "]"
    if isinstance(value, dict):
        # JCS sorts members by UTF-16 code unit; ASCII keys make code-point
        # sorting equivalent here.
        items = sorted(value.items(), key=lambda kv: kv[0])
        return "{" + ",".join(_jcs_string(k) + ":" + _jcs_value(v) for k, v in items) + "}"
    raise TypeError(f"Type {type(value).__name__} is not JSON/JCS serializable")


def canonicalize_jcs(obj: Any) -> bytes:
    """Return the RFC 8785 JCS canonical byte serialization of ``obj``."""
    return _jcs_value(obj).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Pluggable signing seam — spec §8.3
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SignatureEnvelope:
    """The §8.3.1 signed envelope persisted alongside the package."""

    signature: str  # base64 Ed25519ph signature ("" when unsigned)
    publicKey: str  # base64 DER SPKI ("" when unsigned)
    algorithm: str  # "Ed25519ph"
    kid: str  # trust-registry key id ("dev-unsigned" when unsigned)

    def as_dict(self) -> dict[str, str]:
        return {
            "signature": self.signature,
            "publicKey": self.publicKey,
            "algorithm": self.algorithm,
            "kid": self.kid,
        }


class EvidenceSigner(Protocol):
    """Signs the envelope-hash hex string (spec §8.3.1, step 4).

    A real implementation Ed25519ph-signs the UTF-8 bytes of ``envelope_hash``
    with datHere's key, obtains an RFC-3161 timestamp + Rekor inclusion proof,
    and reports the key id registered in the trust registry. ``kid`` MUST match
    a ``keys[]`` entry in ``/.well-known/typed-publisher.json``.
    """

    @property
    def kid(self) -> str: ...

    @property
    def signed(self) -> bool: ...

    def sign(self, envelope_hash: str) -> SignatureEnvelope: ...


class UnsignedSigner:
    """Default dev signer: produces no signature, only a clear marker.

    Packages built with this signer are structurally complete and carry the
    full commitment view, but ``keyTrust`` will resolve to unsigned at any
    verifier — honest until datHere's key + trust registry are provisioned.
    """

    kid = "dev-unsigned"
    signed = False

    def sign(self, envelope_hash: str) -> SignatureEnvelope:  # noqa: ARG002
        return SignatureEnvelope(signature="", publicKey="", algorithm="Ed25519ph", kid=self.kid)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
@dataclass
class EvidencePackage:
    """A built datHere evidence package and its derived artifacts."""

    package: dict[str, Any]  # the canonical-JSON A–G package object
    signature: SignatureEnvelope
    envelope_hash: str  # SHA-256 hex of JCS(package) — the packageHash
    content_hash: str  # SHA-256 hex of JCS(notebook) per dathere-ag-jupyter/v1
    commitment_view: dict[str, Any]  # §8.8.1 field set

    @property
    def signed(self) -> bool:
        return bool(self.signature.signature)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _derive_summary(answer: str) -> str:
    """G-section summary: a short, citation-ready slice of the answer (§8.7.1 #6)."""
    text = " ".join((answer or "").split())
    if len(text) <= _SUMMARY_MAX_CHARS:
        return text or "Verified analysis."
    cut = text[:_SUMMARY_MAX_CHARS]
    # Prefer a sentence boundary, else the last word boundary.
    for sep in (". ", "! ", "? "):
        idx = cut.rfind(sep)
        if idx > 120:
            return cut[: idx + 1].strip()
    idx = cut.rfind(" ")
    return (cut[:idx] if idx > 0 else cut).strip() + "…"


def _agent_log_section(agent_log: list[dict[str, Any]], entry_type: str) -> dict[str, Any]:
    for entry in agent_log:
        if entry.get("type") == entry_type:
            return entry
    return {}


def _build_queries_and_sources(
    agent_log: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Section D: one ``queries[]`` entry per tool call, plus distinct sources."""
    queries: list[dict[str, Any]] = []
    sources: dict[str, dict[str, Any]] = {}
    for entry in agent_log:
        if entry.get("type") != "tool_execution":
            continue
        source = entry.get("source", "unknown")
        queries.append(
            {
                "name": entry.get("tool", ""),
                "source": source,
                "operationType": entry.get("operation_type", "query"),
                "arguments": entry.get("input", {}),
                "status": entry.get("status", "success"),
                "timestamp": entry.get("timestamp"),
            }
        )
        sources.setdefault(source, {"sourceId": source})
    return queries, list(sources.values())


def _build_trace(
    agent_log: list[dict[str, Any]], duration_ms: int, trace_id: str
) -> dict[str, Any]:
    """Section D: a minimal OpenTelemetry-shaped trace (spec §8.4 span kinds)."""
    spans: list[dict[str, Any]] = []
    for i, entry in enumerate(agent_log):
        etype = entry.get("type")
        if etype == "llm_response":
            spans.append(
                {
                    "name": "llm_inference",
                    "spanId": f"{i:04x}",
                    "kind": "llm_inference",
                    "attributes": {
                        "llm.model": entry.get("model"),
                        "llm.stop_reason": entry.get("stop_reason"),
                        "llm.tokens.input": entry.get("tokens", {}).get("input", 0),
                        "llm.tokens.output": entry.get("tokens", {}).get("output", 0),
                    },
                    "durationMs": entry.get("duration_ms", 0),
                }
            )
        elif etype == "tool_execution":
            spans.append(
                {
                    "name": "mcp_tool_call",
                    "spanId": f"{i:04x}",
                    "kind": "mcp_tool_call",
                    "attributes": {
                        "mcp.source": entry.get("source"),
                        "tool.name": entry.get("tool"),
                        "tool.operation_type": entry.get("operation_type"),
                        "tool.status": entry.get("status"),
                    },
                    "durationMs": entry.get("duration_ms", 0),
                }
            )
    return {
        "traceId": trace_id,
        "rootSpan": {"name": "analysis", "kind": "analysis", "durationMs": duration_ms},
        "spans": spans,
    }


def _build_environment(session_start: dict[str, Any], model: str) -> dict[str, Any]:
    """Section C: ``extensions[org.civicaitools.environment]`` (spec §8.7.1 #3)."""
    tools = session_start.get("tools_available", [])
    mcp_servers = []
    portal_url = session_start.get("portal_url")
    if portal_url:
        mcp_servers.append({"url": portal_url, "name": session_start.get("data_source")})
    return {
        "modelVersion": model,
        "temperature": float(getattr(settings, "llm_temperature", 0.0)),
        "mcpServers": mcp_servers,
        "toolDefinitions": tools,  # tool names captured at session start
        "host": EVIDENCE_HOST,
        "maxTokens": int(getattr(settings, "llm_max_tokens", 0)),
    }


def build_evidence_package(
    *,
    notebook_json: dict[str, Any],
    answer: str,
    query: str,
    agent_log: list[dict[str, Any]],
    title: str | None = None,
    package_url: str | None = None,
    signer: EvidenceSigner | None = None,
    package_id: str | None = None,
    created_at: str | None = None,
) -> EvidencePackage:
    """Build a ``datHere`` evidence package from a verified notebook + agent log.

    ``package_url`` is recorded in the commitment view's ``packageUrl``.
    ``signer`` defaults to :class:`UnsignedSigner`.

    ``package_id`` and ``created_at`` default to a fresh UUID / the current time.
    Pass **stable** values (e.g. ``uuid5`` of the notebook id and the notebook's
    ``verified_at``) to make the envelope hash a deterministic content address —
    rebuilding the same notebook then yields the same hash, so re-minting is
    idempotent instead of orphaning prior packages under fresh random ids.
    """
    signer = signer or UnsignedSigner()
    session_start = _agent_log_section(agent_log, "session_start")
    summary_entry = _agent_log_section(agent_log, "summary")

    model = str(
        summary_entry.get("model")
        or session_start.get("model")
        or getattr(settings, "llm_model", "unknown")
    )
    duration_ms = int(summary_entry.get("total_elapsed_ms", 0))
    token_totals = summary_entry.get("token_totals", {})
    prompt_tokens = (
        token_totals.get("input", 0)
        + token_totals.get("cache_creation_input", 0)
        + token_totals.get("cache_read_input", 0)
    )
    completion_tokens = token_totals.get("output", 0)

    queries, data_sources = _build_queries_and_sources(agent_log)
    summary = _derive_summary(answer)

    pid = package_id or str(uuid.uuid4())
    # Derive the trace id from the package id so a fixed package_id yields a
    # fully deterministic envelope (no stray uuid4 in the trace).
    trace_id = uuid.uuid5(uuid.NAMESPACE_URL, f"trace:{pid}").hex

    # ── A–G top-level fields (spec §8.1, §8.7) ───────────────────────────
    package: dict[str, Any] = {
        "metadata": {
            "schemaVersion": SCHEMA_VERSION,
            "packageId": pid,
            "createdAt": created_at or _utc_now(),
            "signingKeyId": signer.kid,  # MUST equal the envelope kid (§8.3.1)
            "captureMethod": CAPTURE_METHOD,
            "contentProfile": CONTENT_PROFILE,
        },
        # A — verbatim prompt, full_text (datHere requires readable section A)
        "prompt": {
            "hash": _sha256_hex((query or "").encode("utf-8")),
            "visibility": "full_text",
            "text": query,
        },
        # D — tool calls + trace
        "queries": queries,
        "dataSources": data_sources,
        "trace": _build_trace(agent_log, duration_ms, trace_id),
        # C — cost / model card
        "cost": {
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": prompt_tokens + completion_tokens,
            "model": model,
            "durationMs": duration_ms,
        },
        # B — system prompt(s)
        "skillMetadata": {
            "skillText": session_start.get("system_prompt", ""),
            "skillTextHash": session_start.get("system_prompt_sha256", ""),
            "mcpServerUrl": session_start.get("portal_url"),
        },
        # F — rendered answer
        "output": answer,
        # G — summary (required for datHere)
        "summary": summary,
        "contentProfile": CONTENT_PROFILE,
        "producerProfile": PRODUCER_PROFILE,
        "contentCanonicalization": CONTENT_CANONICALIZATION,
        "type": NODE_TYPE,
        "extensions": {
            # E — the answer notebook (promoted to normatively required)
            NOTEBOOK_NS: {
                "format": "jupyter-v4.5",
                "provenance": "skeleton",  # our notebooks embed a chat-authored answer
                "notebook": notebook_json,
            },
            # C — environment metadata
            ENVIRONMENT_NS: _build_environment(session_start, model),
        },
    }

    # ── Content hash (§8.2): fingerprints the notebook under the datHere rule
    content_hash = _sha256_hex(canonicalize_jcs(package["extensions"][NOTEBOOK_NS]))
    package["contentHash"] = {"sha256": content_hash}

    # signer identity binding (§8.1.1 signer). This is a top-level *signed*
    # field, so it must be in the package BEFORE the envelope hash is computed —
    # the identity claim doesn't depend on the signature value (only on the
    # signer's kid/tier), so it can be set first.
    signer_identity = {
        "bindingTier": "platform" if signer.signed else "unsigned",
        "identifier": f"platform:{EVIDENCE_HOST}",
        "displayName": "datHere Data Concierge",
    }
    package["signer"] = signer_identity

    # ── Envelope hash (§8.2/§8.3): SHA-256 over JCS of the unsigned package.
    # The sig envelope is stored alongside, not inside the package, so the
    # package object as-is (every signed field present) is the unsigned envelope.
    envelope_hash = _sha256_hex(canonicalize_jcs(package))
    signature = signer.sign(envelope_hash)

    commitment_view: dict[str, Any] = {
        "evidenceProtocolVersion": SCHEMA_VERSION,
        "packageHash": envelope_hash,
        "packageUrl": package_url or "",
        "captureMethod": CAPTURE_METHOD,
        "contentProfile": CONTENT_PROFILE,
        "signature": signature.as_dict(),
        # §8.8.1 expects BOTH: ``signer`` is the claim a verifier reads when it
        # follows a lifecycle chain (commitment.signer.identifier), and
        # ``signerIdentity`` is the informational block describing the
        # publisher. Emitting only the latter left the identifier empty, so a
        # withdrawal or supersession chain would not resolve. Same object the
        # package carries; the commitment view is not hashed, so this changes
        # no packageHash.
        "signer": signer_identity,
        "signerIdentity": signer_identity,
        "trustRegistryUrl": TRUST_REGISTRY_URL,
        "subjectTitle": title or (query or "Verified analysis")[:120],
        "subjectSummary": summary,
        "attestations": [],
    }
    if not signer.signed:
        commitment_view["_status"] = "dev-unsigned"

    return EvidencePackage(
        package=package,
        signature=signature,
        envelope_hash=envelope_hash,
        content_hash=content_hash,
        commitment_view=commitment_view,
    )


# ---------------------------------------------------------------------------
# Notebook-embedded serialization — spec §8.8.2 + cell-0 table §8.8.4
# ---------------------------------------------------------------------------
def _cell0_markdown(cv: dict[str, Any]) -> dict[str, Any]:
    """Build the §8.8.4 human-readable metadata table (reader affordance only)."""
    sig = cv.get("signature", {})
    si = cv.get("signerIdentity", {})
    short_hash = cv.get("packageHash", "")[:12]
    status = "⚠ dev-unsigned (not yet verifiable)" if cv.get("_status") else "signed"
    lines = [
        "## 🔏 Typed Standards evidence",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Signer | {si.get('displayName', '?')} ({si.get('bindingTier', '?')}) |",
        f"| Package hash | `{short_hash}…` |",
        f"| Capture / profile | {cv.get('captureMethod')} · {cv.get('contentProfile')} |",
        f"| Signature | {status} · `{sig.get('kid', '?')}` |",
        f"| Attestations | {len(cv.get('attestations', []))} |",
        f"| Trust registry | {cv.get('trustRegistryUrl')} |",
        "",
        # Host-anchored short link: ``/verify/{host}/{hash}`` resolves directly
        # against our origin. A bare ``?hash=`` is origin-less and resolves
        # against the directory's anchor host instead, landing the reader on a
        # host-picker rather than on the verified record.
        f"[Verify with Typed Standards]"
        f"(https://typedstandards.org/verify/{EVIDENCE_HOST}/{cv.get('packageHash', '')})",
        "",
        "_This table is a reader affordance; the authoritative metadata is the "
        "`org.civicaitools.evidence` namespace in this notebook's root metadata._",
    ]
    return {
        "cell_type": "markdown",
        "metadata": {EVIDENCE_NS: {"role": "commitment-table"}},
        "source": [line + "\n" for line in lines],
    }


def embed_commitment_view(notebook_json: dict[str, Any], pkg: EvidencePackage) -> dict[str, Any]:
    """Return a copy of ``notebook_json`` with the §8.8.2 commitment view embedded.

    Writes the commitment view into ``metadata[org.civicaitools.evidence]`` and
    prepends/refreshes the §8.8.4 cell-0 metadata table. Existing publisher
    namespaces (``kernelspec``, ``language_info``) are preserved.
    """
    nb = copy.deepcopy(notebook_json)
    nb.setdefault("metadata", {})[EVIDENCE_NS] = pkg.commitment_view

    cells = nb.setdefault("cells", [])
    table = _cell0_markdown(pkg.commitment_view)
    # Replace an existing commitment table rather than stacking duplicates.
    if (
        cells
        and cells[0].get("metadata", {}).get(EVIDENCE_NS, {}).get("role") == "commitment-table"
    ):
        cells[0] = table
    else:
        cells.insert(0, table)
    return nb
