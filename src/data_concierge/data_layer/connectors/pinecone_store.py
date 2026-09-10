"""Pinecone Vector Store for CKAN resource semantic search.

This client interfaces with a Pinecone index containing pre-indexed CKAN
resources with AI-generated metadata for semantic search capabilities.
"""

import threading
from typing import Any

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger

logger = get_logger(__name__)

# Try to import Pinecone (new SDK v3+ — exposes `Pinecone` at top level)
try:
    from pinecone import Pinecone  # type: ignore[attr-defined]
    PINECONE_AVAILABLE = True
except ImportError:
    try:
        # Some intermediate v3/v4 versions exposed it from pinecone.pinecone
        from pinecone.pinecone import Pinecone  # type: ignore[no-redef]
        PINECONE_AVAILABLE = True
    except ImportError:
        PINECONE_AVAILABLE = False
        Pinecone = None  # type: ignore[assignment,misc]


class PineconeVectorStore:
    """Vector store using Pinecone for CKAN resource semantic search.

    Optimized for searching pre-indexed CKAN resources with enhanced
    metadata including AI-generated tags, temporal/geographic/demographic
    flags, and embedded descriptions.
    """

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        namespace: str | None = None,
    ) -> None:
        """Initialize the Pinecone vector store.

        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            namespace: Namespace within the index
        """
        self.api_key = api_key or (
            settings.pinecone_api_key.get_secret_value()
            if hasattr(settings, 'pinecone_api_key') and settings.pinecone_api_key
            else None
        )
        self.index_name = index_name or getattr(settings, 'pinecone_index_name', 'ckan-resources')
        self.namespace = namespace or getattr(settings, 'pinecone_namespace', 'default')

        self.logger = get_logger("pinecone_store")

        # Query cache
        self.query_cache: dict[str, list[dict[str, Any]]] = {}
        self.cache_lock = threading.Lock()

        # Initialize Pinecone
        self.pc = None
        self.index = None
        self.use_pinecone = False

        # Debug logging
        self.logger.debug(
            "Pinecone initialization",
            available=PINECONE_AVAILABLE,
            has_api_key=bool(self.api_key),
            index_name=self.index_name,
        )

        if PINECONE_AVAILABLE and self.api_key:
            try:
                self.pc = Pinecone(api_key=self.api_key)
                self.index = self._get_index()
                self.use_pinecone = self.index is not None
                if self.use_pinecone:
                    self.logger.info(
                        "PineconeVectorStore initialized",
                        index=self.index_name,
                        namespace=self.namespace,
                    )
                else:
                    self.logger.warning(
                        "Pinecone index not found",
                        index_name=self.index_name,
                    )
            except Exception as e:
                self.logger.error("Failed to initialize Pinecone", error=str(e))
                self.use_pinecone = False
        else:
            reason = "library not available" if not PINECONE_AVAILABLE else "no API key"
            self.logger.warning(f"Pinecone not available - {reason}")

    def _get_index(self):
        """Get the Pinecone index if it exists."""
        try:
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name in existing_indexes:
                index = self.pc.Index(self.index_name)
                self.logger.info(f"Connected to Pinecone index: {self.index_name}")
                return index
            else:
                self.logger.warning(f"Index {self.index_name} not found")
                return None
        except Exception as e:
            self.logger.error(f"Failed to connect to Pinecone index: {e}")
            return None

    def _safe_get_field(self, obj: Any, field_name: str, default: Any = None) -> Any:
        """Safely get field from object that might be dict or object."""
        if hasattr(obj, field_name):
            return getattr(obj, field_name, default)
        elif isinstance(obj, dict):
            return obj.get(field_name, default)
        return default

    def portal_filter(self, site_id: str | None) -> dict[str, Any] | None:
        """Metadata filter restricting a search to one portal's records.

        One namespace holds every portal's records, so an unscoped search
        returns another portal's resource IDs — which then 404 when the agent
        tries to load them against the portal it is actually querying.

        Records written before per-portal tagging carry no ``site_id`` at all.
        Rather than backfilling them (a one-way write over a live corpus), the
        portal that owns them — ``settings.pinecone_legacy_site_id`` — also
        matches untagged records. Every other portal matches strictly, so a
        newly indexed portal can never pick them up.
        """
        if not site_id:
            return None
        legacy = getattr(settings, "pinecone_legacy_site_id", "ckan")
        if site_id == legacy:
            return {
                "$or": [
                    {"site_id": {"$eq": site_id}},
                    {"site_id": {"$exists": False}},
                ]
            }
        return {"site_id": {"$eq": site_id}}

    def search_resources(
        self,
        query: str,
        n_results: int = 10,
        filters: dict[str, Any] | None = None,
        site_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for portal resources using semantic similarity.

        Args:
            query: Search query
            n_results: Maximum number of results
            filters: Optional metadata filters
            site_id: Restrict results to one portal (see :meth:`portal_filter`).
                Leaving this ``None`` searches every portal in the namespace.

        Returns:
            List of matching resources with metadata
        """
        self.logger.info(f"Vector search: '{query}' (max: {n_results}, site: {site_id})")

        # Check cache — the site scope is part of the key, or a scoped and an
        # unscoped search for the same text would share an entry.
        cache_key = f"{query}_{n_results}_{filters}_{site_id}"
        with self.cache_lock:
            if cache_key in self.query_cache:
                self.logger.info("Using cached search results")
                return self.query_cache[cache_key]

        # Try Pinecone search
        if self.use_pinecone and self.index:
            try:
                results = self._pinecone_search(query, n_results, filters, site_id)
                # Return even an EMPTY result: Pinecone answered, and "this
                # portal has nothing indexed" is a real answer. Falling through
                # to the fallback here logged "Pinecone unavailable" for a
                # perfectly healthy scoped search that simply had no matches.
                with self.cache_lock:
                    self.query_cache[cache_key] = results
                    if len(self.query_cache) > 100:
                        # Prune old entries
                        keys = list(self.query_cache.keys())[:20]
                        for k in keys:
                            del self.query_cache[k]
                return results
            except Exception as e:
                self.logger.error(f"Pinecone search failed: {e}")

        # Fallback search
        results = self._fallback_search(query, n_results, filters)
        with self.cache_lock:
            self.query_cache[cache_key] = results
        return results

    def _pinecone_search(
        self,
        query: str,
        n_results: int,
        filters: dict[str, Any] | None,
        site_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute search using Pinecone's integrated embeddings."""
        # Build metadata filter
        filter_dict: dict[str, Any] = {}
        if filters:
            if filters.get("has_temporal"):
                filter_dict["has_temporal"] = True
            if filters.get("has_geographic"):
                filter_dict["has_geographic"] = True
            if filters.get("has_demographic"):
                filter_dict["has_demographic"] = True
            if filters.get("has_financial"):
                filter_dict["has_financial"] = True
            if filters.get("format"):
                filter_dict["format"] = filters["format"]
            if filters.get("min_records"):
                filter_dict["record_count"] = {"$gte": filters["min_records"]}

        # Build search params using Pinecone's inference API (v5+ SDK flattens
        # the previous nested ``query={...}`` block into top-level kwargs).
        search_params: dict[str, Any] = {
            "namespace": self.namespace,
            "inputs": {"text": query},
            "top_k": min(n_results * 2, 100),
            "fields": [
                "resource_id", "resource_name", "dataset_id", "dataset_title",
                "format", "record_count", "column_count",
                "ai_tags", "has_temporal", "has_geographic", "has_demographic",
                "has_financial", "temporal_min", "temporal_max",
                "text", "description", "site_id",
            ],
        }

        # Portal scoping is applied server-side. It cannot be done after the
        # fact: `fields` below determines what the server returns, so a
        # client-side drop would be filtering on data it never received — and
        # top_k would already have been filled with the wrong portal's records.
        scope = self.portal_filter(site_id)
        if scope and filter_dict:
            search_params["filter"] = {"$and": [scope, filter_dict]}
        elif scope:
            search_params["filter"] = scope
        elif filter_dict:
            search_params["filter"] = filter_dict

        results = self.index.search(**search_params)

        # Parse results
        formatted_results = []
        seen_resources = set()

        hits = None
        if hasattr(results, 'result') and hasattr(results.result, 'hits'):
            hits = results.result.hits
        elif hasattr(results, 'hits'):
            hits = results.hits
        elif isinstance(results, dict):
            hits = results.get('result', {}).get('hits', []) or results.get('hits', [])

        if hits:
            for hit in hits:
                fields = self._safe_get_field(hit, 'fields', {})
                resource_id = fields.get('resource_id') if fields else None

                if resource_id and resource_id not in seen_resources:
                    seen_resources.add(resource_id)

                    score = (
                        self._safe_get_field(hit, '_score') or
                        self._safe_get_field(hit, 'score') or 0.0
                    )

                    result = {
                        'resource_id': resource_id,
                        'resource_name': fields.get('resource_name', 'Unknown'),
                        'dataset_id': fields.get('dataset_id', ''),
                        'dataset_title': fields.get('dataset_title', 'Unknown'),
                        'format': fields.get('format', 'unknown'),
                        'record_count': int(fields.get('record_count', 0) or 0),
                        'column_count': int(fields.get('column_count', 0) or 0),
                        'score': float(score),
                        'has_temporal': bool(fields.get('has_temporal', False)),
                        'has_geographic': bool(fields.get('has_geographic', False)),
                        'has_demographic': bool(fields.get('has_demographic', False)),
                        'has_financial': bool(fields.get('has_financial', False)),
                        'ai_tags': fields.get('ai_tags', ''),
                        'description': fields.get('description', ''),
                        'site_id': fields.get('site_id', ''),
                        'temporal_coverage': self._extract_temporal_coverage(fields),
                    }

                    # Extract description from text field if not present
                    if not result['description'] and fields.get('text'):
                        content = fields['text']
                        if 'Description:' in content:
                            desc_start = content.find('Description:') + len('Description:')
                            desc_end = content.find('|', desc_start)
                            if desc_end == -1:
                                desc_end = desc_start + 200
                            result['description'] = content[desc_start:desc_end].strip()

                    formatted_results.append(result)

        formatted_results.sort(key=lambda x: x['score'], reverse=True)
        return formatted_results[:n_results]

    def _fallback_search(
        self,
        query: str,
        n_results: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Fallback search when Pinecone is not available.

        Returns empty list since we can't do semantic search without Pinecone,
        and mock results would cause 404 errors when trying to load actual data.
        """
        self.logger.warning(
            "Pinecone unavailable - semantic search not possible. "
            "Configure Pinecone API key or use Data Commons source instead."
        )
        return []

    def _extract_temporal_coverage(self, metadata: dict) -> dict | None:
        """Extract temporal coverage from metadata."""
        if metadata.get('has_temporal') and metadata.get('temporal_min'):
            return {
                'min': metadata.get('temporal_min'),
                'max': metadata.get('temporal_max'),
            }
        return None

    def get_resource_profile(self, resource_id: str) -> dict[str, Any] | None:
        """Get detailed profile for a specific resource.

        Args:
            resource_id: Resource ID to look up

        Returns:
            Resource metadata or None if not found
        """
        if not self.use_pinecone or not self.index:
            return {
                'resource_id': resource_id,
                'resource_name': f'Resource {resource_id}',
                'format': 'CSV',
                'record_count': 1000,
                'column_count': 10,
            }

        try:
            # Use dummy vector for metadata-only query
            results = self.index.query(
                namespace=self.namespace,
                vector=[0.0] * 2048,
                filter={"resource_id": {"$eq": resource_id}},
                top_k=1,
                include_metadata=True,
                include_values=False,
            )

            if results and results.matches:
                metadata = results.matches[0].metadata
                return {
                    'resource_id': resource_id,
                    'resource_name': metadata.get('resource_name'),
                    'dataset_id': metadata.get('dataset_id'),
                    'dataset_title': metadata.get('dataset_title'),
                    'format': metadata.get('format'),
                    'record_count': int(metadata.get('record_count', 0) or 0),
                    'column_count': int(metadata.get('column_count', 0) or 0),
                    'has_temporal': bool(metadata.get('has_temporal', False)),
                    'has_geographic': bool(metadata.get('has_geographic', False)),
                    'has_demographic': bool(metadata.get('has_demographic', False)),
                    'ai_tags': metadata.get('ai_tags', ''),
                    'temporal_coverage': self._extract_temporal_coverage(metadata),
                }

            return None

        except Exception as e:
            self.logger.error(f"Error getting resource profile: {e}")
            return None

    def get_stats(self) -> dict[str, Any]:
        """Get vector store statistics."""
        if not self.use_pinecone or not self.index:
            return {
                "status": "fallback mode",
                "using_pinecone": False,
            }

        try:
            stats = self.index.describe_index_stats()
            total_docs = 0

            if hasattr(stats, 'namespaces') and stats.namespaces:
                namespace_stats = stats.namespaces.get(self.namespace, {})
                if hasattr(namespace_stats, 'vector_count'):
                    total_docs = namespace_stats.vector_count
                elif isinstance(namespace_stats, dict):
                    total_docs = namespace_stats.get('vector_count', 0)

            return {
                "status": "healthy",
                "total_documents": total_docs,
                "index_name": self.index_name,
                "namespace": self.namespace,
                "using_pinecone": True,
                "cache_size": len(self.query_cache),
            }

        except Exception as e:
            self.logger.error(f"Stats error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "using_pinecone": self.use_pinecone,
            }

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        """True when an upsert failed because the embedding quota was hit.

        Integrated-embedding indexes embed server-side, so a batch is billed
        against a per-minute token quota for the embedding model
        (llama-text-embed-v2: 250k tokens/minute). Onboarding a whole portal
        sends far more than that, and it comes back as a generic 429 rather
        than anything upsert-specific.
        """
        text = str(exc)
        return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()

    def upsert_records(
        self,
        records: list[dict[str, Any]],
        *,
        namespace: str | None = None,
        batch_size: int = 40,
        max_retries: int = 5,
    ) -> dict[str, Any]:
        """Upsert records into the index, embedding them server-side.

        The indexes this store targets are **integrated-embedding** indexes:
        the embedding model lives in the index (``llama-text-embed-v2`` on a
        semantic field named ``text``), so records are sent as plain
        dictionaries and Pinecone embeds them. That is why this uses
        ``upsert_records`` rather than ``upsert`` — the latter expects vectors
        we would have to produce ourselves, with a model that must match the
        index's exactly.

        Each record needs an ``_id`` and the ``text`` field that gets embedded;
        every other key is stored alongside and is filterable/returnable.

        Returns a summary with ``upserted``, ``failed``, and any ``errors``;
        a failing batch does not abort the rest, so one oversized record
        cannot cost an entire onboarding run.

        Rate limiting is retried rather than reported: the embedding quota is
        per *minute*, so a portal-sized upload hits it partway through, and
        treating that as a permanent failure silently drops most of the corpus
        (a real WPRDC run lost 90 of 110 records this way). A 429 backs off and
        retries the same batch; any other error is recorded and the run moves on.
        """
        import random
        import time as _time

        if not self.use_pinecone or not self.index:
            return {
                "upserted": 0,
                "failed": len(records),
                "errors": ["Pinecone is not configured (no API key, or index not found)"],
            }

        target_ns = namespace or self.namespace
        upserted = 0
        failed = 0
        errors: list[str] = []

        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            missing = [r for r in batch if not r.get("_id") or not r.get("text")]
            if missing:
                failed += len(missing)
                errors.append(
                    f"{len(missing)} record(s) in batch at offset {start} lack _id or text"
                )
                batch = [r for r in batch if r.get("_id") and r.get("text")]
                if not batch:
                    continue
            for attempt in range(max_retries + 1):
                try:
                    # Keyword args, not positional: the SDK made this method
                    # keyword-only in v10, and the parameter names are unchanged
                    # back to v5 — so this call works across the pinned range.
                    self.index.upsert_records(namespace=target_ns, records=batch)
                    upserted += len(batch)
                    self.logger.info(
                        "Upserted records to Pinecone",
                        count=len(batch),
                        namespace=target_ns,
                        offset=start,
                    )
                    break
                except Exception as e:
                    if self._is_rate_limited(e) and attempt < max_retries:
                        # The quota window is a minute, so back off toward it
                        # rather than hammering: 15s, 30s, 60s, 60s... with
                        # jitter so concurrent uploads do not resynchronise.
                        delay = min(15 * (2**attempt), 60) + random.uniform(0, 5)
                        self.logger.warning(
                            "Pinecone embedding quota hit; backing off",
                            offset=start,
                            attempt=attempt + 1,
                            delay_seconds=round(delay, 1),
                        )
                        _time.sleep(delay)
                        continue
                    failed += len(batch)
                    errors.append(f"batch at offset {start}: {e}")
                    self.logger.error(
                        "Pinecone upsert failed", offset=start, error=str(e)
                    )
                    break

        # Newly written records invalidate cached query answers.
        if upserted:
            self.clear_cache()

        return {
            "upserted": upserted,
            "failed": failed,
            "errors": errors,
            "namespace": target_ns,
            "index_name": self.index_name,
        }

    def delete_records(self, ids: list[str], *, namespace: str | None = None) -> int:
        """Delete records by ID.  Returns the number requested for deletion."""
        if not self.use_pinecone or not self.index or not ids:
            return 0
        target_ns = namespace or self.namespace
        try:
            self.index.delete(ids=ids, namespace=target_ns)
            self.clear_cache()
            return len(ids)
        except Exception as e:
            self.logger.error("Pinecone delete failed", error=str(e))
            return 0

    def clear_cache(self) -> None:
        """Clear the query cache."""
        with self.cache_lock:
            self.query_cache.clear()
        self.logger.info("Query cache cleared")
