"""Redis cache client for query caching."""

import json
from typing import Any

import redis.asyncio as redis

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger

logger = get_logger(__name__)


class CacheClient:
    """Client for Redis-based caching.

    Implements caching strategy from TAD:
    - Cache key: {query_hash}:{data_version}:{model_version}
    - TTL varies by data source update frequency
    - Invalidation on data source update or model version change
    """

    # TTL values by data source (in seconds)
    SOURCE_TTL = {
        "bls": 30 * 24 * 3600,  # 30 days for monthly data
        "census": 90 * 24 * 3600,  # 90 days
        "ncses": 365 * 24 * 3600,  # 365 days for annual data
        "bea": 90 * 24 * 3600,  # 90 days for quarterly
        "data_commons": 7 * 24 * 3600,  # 7 days for aggregated
        "default": settings.cache_ttl_default,
    }

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
    ) -> None:
        """Initialize the cache client.

        Args:
            host: Redis host. Defaults to settings.
            port: Redis port. Defaults to settings.
            db: Redis database number. Defaults to settings.
        """
        self.host = host or settings.redis_host
        self.port = port or settings.redis_port
        self.db = db or settings.redis_db
        self._client: redis.Redis | None = None
        self.logger = get_logger("cache_client")
        self.model_version = settings.app_version
        self.data_version = "v1"  # Updated when data sources refresh

    async def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            password = settings.redis_password.get_secret_value()
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=password if password else None,
                decode_responses=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the Redis client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _build_key(
        self,
        query_hash: str,
        data_source: str = "default",
    ) -> str:
        """Build cache key from components.

        Args:
            query_hash: Hash of the normalized query
            data_source: Primary data source

        Returns:
            Cache key string
        """
        return f"query:{query_hash}:{self.data_version}:{self.model_version}"

    async def get(
        self,
        query_hash: str,
        data_source: str = "default",
    ) -> dict[str, Any] | None:
        """Get cached result.

        Args:
            query_hash: Hash of the normalized query
            data_source: Primary data source

        Returns:
            Cached result or None
        """
        client = await self._get_client()
        key = self._build_key(query_hash, data_source)

        try:
            value = await client.get(key)
            if value:
                self.logger.debug("Cache hit", key=key)
                return json.loads(value)
            self.logger.debug("Cache miss", key=key)
            return None

        except Exception as e:
            self.logger.error("Cache get failed", error=str(e), key=key)
            return None

    async def set(
        self,
        query_hash: str,
        result: dict[str, Any],
        data_source: str = "default",
        ttl: int | None = None,
    ) -> bool:
        """Cache a result.

        Args:
            query_hash: Hash of the normalized query
            result: Result to cache
            data_source: Primary data source for TTL calculation
            ttl: Optional explicit TTL in seconds

        Returns:
            True if successful
        """
        client = await self._get_client()
        key = self._build_key(query_hash, data_source)

        if ttl is None:
            ttl = self.SOURCE_TTL.get(data_source, self.SOURCE_TTL["default"])

        try:
            await client.setex(key, ttl, json.dumps(result))
            self.logger.debug("Cache set", key=key, ttl=ttl)
            return True

        except Exception as e:
            self.logger.error("Cache set failed", error=str(e), key=key)
            return False

    async def delete(
        self,
        query_hash: str,
        data_source: str = "default",
    ) -> bool:
        """Delete a cached result.

        Args:
            query_hash: Hash of the normalized query
            data_source: Primary data source

        Returns:
            True if deleted
        """
        client = await self._get_client()
        key = self._build_key(query_hash, data_source)

        try:
            result = await client.delete(key)
            return result > 0

        except Exception as e:
            self.logger.error("Cache delete failed", error=str(e), key=key)
            return False

    async def invalidate_by_source(
        self,
        data_source: str,
    ) -> int:
        """Invalidate all cached results for a data source.

        Used when a data source is updated.

        Args:
            data_source: Data source identifier

        Returns:
            Number of keys deleted
        """
        client = await self._get_client()

        try:
            # Find all keys for this source pattern
            pattern = f"query:*:{self.data_version}:{self.model_version}"
            keys = []
            async for key in client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                result = await client.delete(*keys)
                self.logger.info(
                    "Cache invalidated by source",
                    source=data_source,
                    count=result,
                )
                return result

            return 0

        except Exception as e:
            self.logger.error(
                "Cache invalidation failed",
                error=str(e),
                source=data_source,
            )
            return 0

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache stats dict
        """
        client = await self._get_client()

        try:
            info = await client.info()
            return {
                "connected": True,
                "used_memory": info.get("used_memory_human", "unknown"),
                "total_keys": info.get("db0", {}).get("keys", 0) if "db0" in info else 0,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
            }

        except Exception as e:
            self.logger.error("Failed to get cache stats", error=str(e))
            return {"connected": False, "error": str(e)}

    async def health_check(self) -> bool:
        """Check if Redis is healthy.

        Returns:
            True if healthy
        """
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception:
            return False
