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

    def search_resources(
        self,
        query: str,
        n_results: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for CKAN resources using semantic similarity.

        Args:
            query: Search query
            n_results: Maximum number of results
            filters: Optional metadata filters

        Returns:
            List of matching resources with metadata
        """
        self.logger.info(f"Vector search: '{query}' (max: {n_results})")

        # Check cache
        cache_key = f"{query}_{n_results}_{filters}"
        with self.cache_lock:
            if cache_key in self.query_cache:
                self.logger.info("Using cached search results")
                return self.query_cache[cache_key]

        # Try Pinecone search
        if self.use_pinecone and self.index:
            try:
                results = self._pinecone_search(query, n_results, filters)
                if results:
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
    ) -> list[dict[str, Any]]:
        """Execute search using Pinecone's integrated embeddings."""
        # Build metadata filter
        filter_dict = {}
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
                "text", "description",
            ],
        }

        if filter_dict:
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

    def clear_cache(self) -> None:
        """Clear the query cache."""
        with self.cache_lock:
            self.query_cache.clear()
        self.logger.info("Query cache cleared")
