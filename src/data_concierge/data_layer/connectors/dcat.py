"""DCAT catalog connector.

Talks to open-data portals that publish a **DCAT catalog** — the
``/data.json`` document mandated by Project Open Data / DCAT-US 1.1 and served
by nearly every US federal, state, and municipal portal (Socrata, ArcGIS Hub,
data.gov, and CKAN with the DCAT extension).

This is the non-CKAN half of the portal story.  A CKAN portal exposes a live
*action API* (``package_search``, ``datastore_search_sql``) that supports
server-side query.  A DCAT portal exposes a single static catalog document
listing every dataset and its distributions (download URLs).  So the shape of
this client is deliberately different from :mod:`.ckan`:

* the catalog is fetched **once** and cached in memory (see ``_CATALOG_CACHE``),
  because it is one document rather than a queryable endpoint;
* search is performed **locally** over that cached document;
* row data comes from streaming a distribution's ``downloadURL``.

That last point is the one real constraint worth knowing.  A DCAT distribution
is a plain file URL with no row-limit parameter — Socrata's ``export.csv``
silently ignores ``$limit``, for instance — so a naive fetch of a
multi-million-row export would stream gigabytes into memory.  Every read here
goes through :meth:`DCATClient.load_distribution`, which streams and **stops**
at a row and byte ceiling, reporting truncation to the caller.

Every outbound URL (the catalog itself and each distribution) is checked with
``mcp.guards.validate_server_url`` first.  Catalog documents are third-party
content that names arbitrary hosts, so an unchecked ``downloadURL`` is a
server-side request forgery vector straight at the cloud metadata endpoint.
"""

from __future__ import annotations

import asyncio
import csv
import io
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from data_concierge.core.logging import get_logger
from data_concierge.mcp.guards import UnsafeMCPTarget, validate_server_url

logger = get_logger(__name__)

# Standard locations for a DCAT catalog, in probe order.  ``/data.json`` is the
# Project Open Data path and by far the most widely deployed.
CATALOG_PATHS: tuple[str, ...] = ("/data.json", "/api/dcat.json", "/catalog.json")

# A catalog is parsed as one JSON document, so an oversized one cannot be
# truncated — it has to be rejected with a clear message instead.  data.gov's
# full catalog is ~1 GB; a state portal's is ~1-5 MB.
MAX_CATALOG_BYTES = 64 * 1024 * 1024

# Defaults for streaming a distribution.  Both ceilings are enforced.
DEFAULT_MAX_ROWS = 1000
MAX_DISTRIBUTION_BYTES = 32 * 1024 * 1024

CATALOG_TTL_SECONDS = 6 * 60 * 60

# Media types we can parse into rows, mapped to a short format label.
_TABULAR_MEDIA_TYPES = {
    "text/csv": "CSV",
    "application/csv": "CSV",
    "text/tab-separated-values": "TSV",
    "application/vnd.ms-excel": "CSV",
}

_CATALOG_CACHE: dict[str, tuple[float, DCATCatalog]] = {}
_CATALOG_LOCKS: dict[str, asyncio.Lock] = {}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def short_id(identifier: str, title: str = "") -> str:
    """Derive a short, stable dataset ID from a DCAT ``identifier``.

    Identifiers are usually URLs (``https://data.pa.gov/api/views/23n7-cwjw``);
    the trailing segment is the portal's own dataset ID and is what a user or
    the agent will recognise.  Non-URL identifiers pass through unchanged.
    """
    ident = (identifier or "").strip()
    if ident:
        parsed = urlparse(ident)
        if parsed.scheme in ("http", "https"):
            tail = parsed.path.rstrip("/").rsplit("/", 1)[-1]
            if tail:
                return tail
        return ident
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return slug or "unknown"


@dataclass
class DCATDistribution:
    """One downloadable representation of a dataset."""

    title: str = ""
    description: str = ""
    media_type: str = ""
    format: str = ""
    download_url: str = ""
    access_url: str = ""
    described_by: str = ""

    @property
    def is_tabular(self) -> bool:
        if self.media_type in _TABULAR_MEDIA_TYPES:
            return True
        return (self.format or "").upper() in ("CSV", "TSV")

    @property
    def best_url(self) -> str:
        return self.download_url or self.access_url

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "media_type": self.media_type,
            "format": self.format or _TABULAR_MEDIA_TYPES.get(self.media_type, ""),
            "download_url": self.download_url,
            "access_url": self.access_url,
            "is_tabular": self.is_tabular,
        }


@dataclass
class DCATDataset:
    """A dataset entry from a DCAT catalog, normalised across catalog dialects."""

    id: str
    title: str = ""
    description: str = ""
    identifier: str = ""
    keywords: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    publisher: str = ""
    modified: str = ""
    issued: str = ""
    landing_page: str = ""
    license: str = ""
    contact: str = ""
    distributions: list[DCATDistribution] = field(default_factory=list)

    @property
    def tabular_distributions(self) -> list[DCATDistribution]:
        return [d for d in self.distributions if d.is_tabular and d.best_url]

    def to_dict(self, include_distributions: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "identifier": self.identifier,
            "keywords": self.keywords,
            "themes": self.themes,
            "publisher": self.publisher,
            "modified": self.modified,
            "issued": self.issued,
            "landing_page": self.landing_page,
            "license": self.license,
        }
        if include_distributions:
            out["distributions"] = [d.to_dict() for d in self.distributions]
        return out


@dataclass
class DCATCatalog:
    """A parsed catalog: the datasets plus where they came from."""

    catalog_url: str
    datasets: list[DCATDataset] = field(default_factory=list)
    title: str = ""
    fetched_at: float = 0.0

    def get(self, dataset_id: str) -> DCATDataset | None:
        """Look up a dataset by short ID, full identifier, or exact title."""
        target = (dataset_id or "").strip().lower()
        if not target:
            return None
        for ds in self.datasets:
            if ds.id.lower() == target:
                return ds
        for ds in self.datasets:
            if ds.identifier.lower() == target or ds.landing_page.lower() == target:
                return ds
        for ds in self.datasets:
            if ds.title.lower() == target:
                return ds
        return None

    def search(
        self,
        query: str,
        limit: int = 10,
        *,
        tabular_only: bool = False,
    ) -> list[tuple[DCATDataset, float]]:
        """Score datasets against ``query`` by weighted field overlap.

        Titles and keywords carry more weight than the description, and an
        exact phrase hit in the title is boosted so that searching a dataset's
        actual name puts it first.  An empty query returns the head of the
        catalog so the agent can still browse.
        """
        candidates = [
            ds for ds in self.datasets if not tabular_only or ds.tabular_distributions
        ]
        q = (query or "").strip().lower()
        if not q:
            return [(ds, 0.0) for ds in candidates[:limit]]

        terms = set(_tokens(q))
        if not terms:
            return [(ds, 0.0) for ds in candidates[:limit]]

        scored: list[tuple[DCATDataset, float]] = []
        for ds in candidates:
            title_tokens = set(_tokens(ds.title))
            keyword_tokens = set(_tokens(" ".join(ds.keywords)))
            theme_tokens = set(_tokens(" ".join(ds.themes)))
            desc_tokens = set(_tokens(ds.description))

            score = (
                3.0 * len(terms & title_tokens)
                + 2.0 * len(terms & keyword_tokens)
                + 1.5 * len(terms & theme_tokens)
                + 1.0 * len(terms & desc_tokens)
            )
            if q in ds.title.lower():
                score += 6.0
            if q in ds.description.lower():
                score += 1.5
            if score > 0:
                # Normalise by query size so scores are comparable across queries.
                scored.append((ds, round(score / (len(terms) * 3.0), 4)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]


def _as_list(value: Any) -> list[str]:
    """Coerce a DCAT field that may be a string, list, or absent into a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                label = item.get("label") or item.get("name") or item.get("@value")
                if isinstance(label, str) and label.strip():
                    out.append(label.strip())
        return out
    return []


def _text(value: Any) -> str:
    """Pull display text out of a field that may be a string or a JSON-LD node."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("name", "fn", "label", "@value", "title", "@id"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    if isinstance(value, list) and value:
        return _text(value[0])
    return ""


def _parse_distribution(raw: dict[str, Any]) -> DCATDistribution:
    media = _text(raw.get("mediaType") or raw.get("dcat:mediaType"))
    fmt = _text(raw.get("format") or raw.get("dct:format"))
    return DCATDistribution(
        title=_text(raw.get("title")),
        description=_text(raw.get("description")),
        media_type=media,
        format=fmt or _TABULAR_MEDIA_TYPES.get(media, ""),
        download_url=_text(raw.get("downloadURL") or raw.get("dcat:downloadURL")),
        access_url=_text(raw.get("accessURL") or raw.get("dcat:accessURL")),
        described_by=_text(raw.get("describedBy")),
    )


def parse_dataset(raw: dict[str, Any]) -> DCATDataset:
    """Normalise one raw DCAT dataset node."""
    identifier = _text(raw.get("identifier") or raw.get("@id"))
    title = _text(raw.get("title") or raw.get("dct:title"))
    distributions = [
        _parse_distribution(d)
        for d in (raw.get("distribution") or raw.get("dcat:distribution") or [])
        if isinstance(d, dict)
    ]
    return DCATDataset(
        id=short_id(identifier, title),
        title=title,
        description=_text(raw.get("description") or raw.get("dct:description")),
        identifier=identifier,
        keywords=_as_list(raw.get("keyword") or raw.get("dcat:keyword")),
        themes=_as_list(raw.get("theme") or raw.get("dcat:theme")),
        publisher=_text(raw.get("publisher")),
        modified=_text(raw.get("modified")),
        issued=_text(raw.get("issued")),
        landing_page=_text(raw.get("landingPage")),
        license=_text(raw.get("license")),
        contact=_text(raw.get("contactPoint")),
        distributions=distributions,
    )


def parse_catalog(body: Any, catalog_url: str) -> DCATCatalog:
    """Parse a catalog document into :class:`DCATCatalog`.

    Handles the three shapes seen in the wild: DCAT-US 1.1 (``{"dataset": [...]}``),
    a bare JSON-LD array of nodes, and a JSON-LD document with ``@graph``.
    """
    raw_datasets: list[dict[str, Any]] = []
    title = ""

    if isinstance(body, dict):
        title = _text(body.get("title"))
        if isinstance(body.get("dataset"), list):
            raw_datasets = [d for d in body["dataset"] if isinstance(d, dict)]
        elif isinstance(body.get("@graph"), list):
            raw_datasets = [
                node
                for node in body["@graph"]
                if isinstance(node, dict) and "Dataset" in str(node.get("@type", ""))
            ]
    elif isinstance(body, list):
        raw_datasets = [
            node
            for node in body
            if isinstance(node, dict) and "Dataset" in str(node.get("@type", ""))
        ]
        if not raw_datasets:
            # A plain array of dataset objects with no @type annotation.
            raw_datasets = [
                node for node in body if isinstance(node, dict) and node.get("title")
            ]

    datasets = [parse_dataset(node) for node in raw_datasets]
    # Drop entries with neither a title nor an identifier — nothing to cite.
    datasets = [d for d in datasets if d.title or d.identifier]
    return DCATCatalog(
        catalog_url=catalog_url,
        datasets=datasets,
        title=title,
        fetched_at=time.time(),
    )


class DCATClient:
    """Async client for a single DCAT portal."""

    def __init__(
        self,
        portal_url: str,
        catalog_url: str | None = None,
        *,
        timeout: float = 60.0,
        allow_private: bool = False,
    ) -> None:
        self.portal_url = (portal_url or "").strip().rstrip("/")
        self.catalog_url = (catalog_url or "").strip()
        self.timeout = timeout
        self.allow_private = allow_private
        self._client: httpx.AsyncClient | None = None
        self.logger = get_logger("dcat")

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"Accept": "application/json, */*"},
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _safe_url(self, url: str) -> str:
        """Validate an outbound URL, raising ``UnsafeMCPTarget`` if it is not safe."""
        return validate_server_url(url, allow_private=self.allow_private)

    async def resolve_catalog_url(self) -> str:
        """Return the catalog URL, probing standard paths if none was configured.

        A configured ``catalog_url`` is trusted and used as-is.  Otherwise each
        of :data:`CATALOG_PATHS` is tried against the portal root, and the first
        that answers with JSON wins.
        """
        if self.catalog_url:
            return self._safe_url(self.catalog_url)
        if not self.portal_url:
            raise ValueError("DCAT portal has neither a portal URL nor a catalog URL")

        client = await self._http()
        errors: list[str] = []
        for path in CATALOG_PATHS:
            candidate = self._safe_url(urljoin(self.portal_url + "/", path.lstrip("/")))
            try:
                resp = await client.head(candidate)
                # Some portals refuse HEAD; fall through to a ranged GET.
                if resp.status_code >= 400:
                    resp = await client.get(candidate, headers={"Range": "bytes=0-2047"})
                if resp.status_code < 400:
                    self.catalog_url = candidate
                    return candidate
                errors.append(f"{path} -> HTTP {resp.status_code}")
            except httpx.HTTPError as exc:
                errors.append(f"{path} -> {exc}")
        raise ValueError(
            f"No DCAT catalog found at {self.portal_url} "
            f"(tried {', '.join(CATALOG_PATHS)}: {'; '.join(errors)})"
        )

    async def fetch_catalog(self, *, force: bool = False) -> DCATCatalog:
        """Fetch and parse the catalog, cached in memory for :data:`CATALOG_TTL_SECONDS`.

        Concurrent callers for the same catalog share one fetch — an agent run
        can issue several catalog-backed tool calls at once, and a 1-5 MB
        document should not be pulled once per call.
        """
        url = await self.resolve_catalog_url()

        cached = _CATALOG_CACHE.get(url)
        if cached and not force and (time.time() - cached[0]) < CATALOG_TTL_SECONDS:
            return cached[1]

        lock = _CATALOG_LOCKS.setdefault(url, asyncio.Lock())
        async with lock:
            # Re-check: another coroutine may have populated it while we waited.
            cached = _CATALOG_CACHE.get(url)
            if cached and not force and (time.time() - cached[0]) < CATALOG_TTL_SECONDS:
                return cached[1]

            client = await self._http()
            body_bytes = bytearray()
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    body_bytes.extend(chunk)
                    if len(body_bytes) > MAX_CATALOG_BYTES:
                        raise ValueError(
                            f"DCAT catalog at {url} exceeds "
                            f"{MAX_CATALOG_BYTES // (1024 * 1024)} MB; "
                            "point catalog_url at a filtered catalog instead"
                        )

            import json

            catalog = parse_catalog(json.loads(body_bytes.decode("utf-8")), url)
            _CATALOG_CACHE[url] = (time.time(), catalog)
            self.logger.info(
                "DCAT catalog loaded",
                url=url,
                datasets=len(catalog.datasets),
                bytes=len(body_bytes),
            )
            return catalog

    async def search_datasets(
        self,
        query: str,
        limit: int = 10,
        *,
        tabular_only: bool = False,
    ) -> list[tuple[DCATDataset, float]]:
        catalog = await self.fetch_catalog()
        return catalog.search(query, limit=limit, tabular_only=tabular_only)

    async def get_dataset(self, dataset_id: str) -> DCATDataset | None:
        catalog = await self.fetch_catalog()
        return catalog.get(dataset_id)

    async def load_distribution(
        self,
        url: str,
        max_rows: int = DEFAULT_MAX_ROWS,
        *,
        max_bytes: int = MAX_DISTRIBUTION_BYTES,
    ) -> dict[str, Any]:
        """Stream a tabular distribution, stopping at ``max_rows``.

        DCAT distributions are static files with no row-limit parameter (Socrata
        even ignores ``$limit`` on its export URLs), so the ceiling has to be
        enforced on our side: we stop reading and drop the connection once we
        have enough rows.  Returns ``columns``, ``rows``, and a ``truncated``
        flag so the caller never mistakes a partial read for a whole dataset.
        """
        safe = self._safe_url(url)
        client = await self._http()

        rows: list[dict[str, Any]] = []
        columns: list[str] = []
        total_bytes = 0
        truncated = False
        hit_byte_cap = False

        buffer = ""
        reader_header: list[str] | None = None

        async with client.stream("GET", safe) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    hit_byte_cap = True
                    truncated = True
                    break
                buffer += chunk.decode("utf-8", errors="replace")

                # Parse only whole lines; keep the trailing partial in `buffer`.
                # A quoted field may legitimately contain newlines, so the last
                # line is always held back until the next chunk confirms it.
                if "\n" not in buffer:
                    continue
                head, _, buffer = buffer.rpartition("\n")
                parsed = list(csv.reader(io.StringIO(head)))
                if reader_header is None and parsed:
                    reader_header = [str(c).strip() for c in parsed[0]]
                    columns = reader_header
                    parsed = parsed[1:]
                for record in parsed:
                    if len(rows) >= max_rows:
                        truncated = True
                        break
                    rows.append(_zip_row(reader_header or [], record))
                if truncated:
                    break

        # Flush whatever is left in the buffer if we did not hit a ceiling.
        if not truncated and buffer.strip():
            parsed = list(csv.reader(io.StringIO(buffer)))
            if reader_header is None and parsed:
                reader_header = [str(c).strip() for c in parsed[0]]
                columns = reader_header
                parsed = parsed[1:]
            for record in parsed:
                if len(rows) >= max_rows:
                    truncated = True
                    break
                rows.append(_zip_row(reader_header or [], record))

        return {
            "url": safe,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "bytes_read": total_bytes,
            "hit_byte_cap": hit_byte_cap,
        }


def _zip_row(header: list[str], record: list[str]) -> dict[str, Any]:
    """Zip a CSV record against the header, tolerating ragged rows."""
    if not header:
        return {f"col_{i}": v for i, v in enumerate(record)}
    out: dict[str, Any] = {}
    for i, name in enumerate(header):
        out[name] = record[i] if i < len(record) else None
    if len(record) > len(header):
        for i in range(len(header), len(record)):
            out[f"col_{i}"] = record[i]
    return out


def clear_catalog_cache() -> None:
    """Drop every cached catalog (used by tests and after a portal edit)."""
    _CATALOG_CACHE.clear()


__all__ = [
    "CATALOG_PATHS",
    "DCATCatalog",
    "DCATClient",
    "DCATDataset",
    "DCATDistribution",
    "UnsafeMCPTarget",
    "clear_catalog_cache",
    "parse_catalog",
    "parse_dataset",
    "short_id",
]
