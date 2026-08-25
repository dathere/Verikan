"""Containment for MCP server endpoints.

Issue #135. An MCP server is third-party code we do not control, reached over
a URL an admin supplies. Two holes made that worse than it needed to be:

1. **No target validation.** ``MCPServerConfig.url`` was a bare ``str`` with no
   validator, and nothing checked scheme or host. Pointing a server at
   ``http://169.254.169.254/`` — the cloud metadata endpoint — would have had
   the service fetch its own instance credentials and hand the response to the
   agent. Same for anything else reachable from inside the VPC.

2. **The server chose where we sent data.** On SSE connect the client adopted
   whatever ``endpoint`` the server advertised, and used it verbatim if it
   started with ``http``. A compromised or hostile server could therefore
   redirect every subsequent JSON-RPC body — including whatever the agent had
   in flight — to a host of its choosing. Exfiltration by design.

Both are closed here: targets are validated before we ever connect, and an
advertised SSE endpoint is pinned to the origin we were configured with.

This does not make an MCP server trustworthy. It bounds where it can send us
and where we can be pointed; the contents it returns are still untrusted input
that reaches the model.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse, urlunparse

from data_concierge.core.logging import get_logger

logger = get_logger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})

# Cloud instance-metadata services. Reachable from inside most VPCs and they
# hand out credentials to anything that asks, so they are the highest-value
# SSRF target by a distance.
METADATA_HOSTS = frozenset(
    {
        "169.254.169.254",  # AWS / GCP / Azure / DigitalOcean
        "metadata.google.internal",
        "metadata",
        "100.100.100.100",  # Alibaba
    }
)


class UnsafeMCPTarget(ValueError):
    """Raised when an MCP URL points somewhere it must not."""


# Under the local-dev opt-out we permit loopback and private addresses, but
# never the cloud-metadata / link-local range — that is dangerous everywhere.
_DEV_ALLOWED_REASONS = frozenset({"loopback address", "private address"})


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Reason this address is off limits, or None if it is fine."""
    if ip.is_loopback:
        return "loopback address"
    if ip.is_link_local:
        # Covers 169.254.0.0/16, i.e. the metadata range.
        return "link-local address (cloud metadata range)"
    if ip.is_private:
        return "private address"
    if ip.is_reserved:
        return "reserved address"
    if ip.is_multicast:
        return "multicast address"
    if ip.is_unspecified:
        return "unspecified address"
    return None


def validate_server_url(
    url: str, *, allow_private: bool = False, resolve: bool = True
) -> str:
    """Check an MCP server URL is safe to connect to. Returns it normalised.

    ``allow_private`` exists for local development, where an MCP server on
    ``localhost`` is the normal case. It is opt-in per call, never the default.

    ``resolve`` controls the DNS check. With ``resolve=True`` (the default, used
    at the API and connect boundaries) the hostname is resolved and every
    address it answers with is checked, so a name pointing at 127.0.0.1 or the
    metadata IP is caught. With ``resolve=False`` only the cheap syntactic
    checks run — scheme, literal-IP, and the metadata-host name — which is what
    the model validator uses so it stays fast and offline-safe on every
    construction (including disk load). The boundary check still does the DNS
    pass, so rebinding is caught where it matters.

    Raises:
        UnsafeMCPTarget: scheme is not http(s), the host is missing, or the
            host is / resolves to a loopback / private / link-local / metadata
            address.
    """
    if not url or not url.strip():
        raise UnsafeMCPTarget("URL is empty")

    parsed = urlparse(url.strip())

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeMCPTarget(
            f"scheme {parsed.scheme or '(none)'!r} is not allowed; use http or https"
        )

    host = parsed.hostname
    if not host:
        raise UnsafeMCPTarget("URL has no host")

    # Normalise a trailing dot (metadata.google.internal.) before the name
    # check, or it slips past the exact-match set.
    host_norm = host.rstrip(".").lower()
    if host_norm in METADATA_HOSTS:
        raise UnsafeMCPTarget(
            f"{host} is a cloud metadata endpoint and would expose instance credentials"
        )

    # Literal IP in the URL: always check it, even under allow_private —
    # loopback/private are fine in dev, but the metadata range never is.
    try:
        literal = ipaddress.ip_address(host_norm)
    except ValueError:
        literal = None

    if literal is not None:
        reason = _is_blocked_ip(literal)
        if reason and not (allow_private and reason in _DEV_ALLOWED_REASONS):
            raise UnsafeMCPTarget(f"{host} is a {reason}")
        return urlunparse(parsed)

    if allow_private or not resolve:
        return urlunparse(parsed)

    # Hostname: resolve and check every address it answers with, so a name
    # that points at 127.0.0.1 or the metadata IP is caught too.
    try:
        infos = socket.getaddrinfo(host, parsed.port or None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise UnsafeMCPTarget(f"{host} does not resolve ({e})") from e

    for info in infos:
        addr = info[4][0]
        try:
            resolved = ipaddress.ip_address(addr)
        except ValueError:
            continue
        reason = _is_blocked_ip(resolved)
        if reason:
            raise UnsafeMCPTarget(f"{host} resolves to {addr}, a {reason}")

    return urlunparse(parsed)


def pin_sse_endpoint(configured_url: str, advertised: str) -> str:
    """Resolve the endpoint an SSE server advertised, pinned to its own origin.

    The server is allowed to tell us *which path* to POST to. It is not
    allowed to tell us which host — that would let it redirect our traffic,
    and everything in it, somewhere else.

    A relative path is joined to the configured origin. An absolute URL is
    accepted only when its origin matches; otherwise the path is taken and
    the origin discarded.
    """
    parsed_cfg = urlparse(configured_url)
    origin = f"{parsed_cfg.scheme}://{parsed_cfg.netloc}"

    advertised = (advertised or "").strip()
    if not advertised:
        raise UnsafeMCPTarget("server advertised an empty SSE endpoint")

    if not advertised.startswith(("http://", "https://")):
        if not advertised.startswith("/"):
            advertised = "/" + advertised
        return origin + advertised

    parsed_adv = urlparse(advertised)
    adv_origin = f"{parsed_adv.scheme}://{parsed_adv.netloc}"
    if adv_origin == origin:
        return advertised

    # Keep the path, drop the origin it tried to substitute.
    path = parsed_adv.path or "/"
    if parsed_adv.query:
        path = f"{path}?{parsed_adv.query}"
    logger.warning(
        "MCP server advertised an SSE endpoint on a different origin; pinning to its own",
        configured_origin=origin,
        advertised_origin=adv_origin,
    )
    return origin + path
