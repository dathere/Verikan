"""Admin-editable runtime settings (currently: the query-processing timeout).

The analysis pipeline (``POST /api/v1/query``) is wrapped in a hard timeout so
a stuck or very long analysis fails *gracefully* — with a friendly answer and
follow-up suggestions — instead of hanging until the platform (e.g. Cloud Run,
default 300 s) kills the connection and the UI shows a generic error. Admins
tune the limit from the admin panel; changes apply to the next query with no
restart needed.

Persists via the shared storage backend, same pattern as
``system_prompt`` / ``github_settings``.
"""

from typing import Any

from data_concierge.core.logging import get_logger
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

_RUNTIME_SETTINGS_KEY = "runtime_settings.json"

# Default stays below Cloud Run's 300 s request timeout so the app can answer
# gracefully before the platform cuts the connection.
DEFAULT_QUERY_TIMEOUT_SECONDS = 240
MIN_QUERY_TIMEOUT_SECONDS = 30
MAX_QUERY_TIMEOUT_SECONDS = 3600


def load_runtime_settings() -> dict[str, Any]:
    """Return the effective runtime settings for the admin UI.

    A read error degrades gracefully to the defaults.
    """
    try:
        saved = storage.read_json(_RUNTIME_SETTINGS_KEY) or {}
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to read runtime settings; using defaults", error=str(e))
        saved = {}

    custom = saved.get("query_timeout_seconds")
    timeout = DEFAULT_QUERY_TIMEOUT_SECONDS
    is_custom = False
    if isinstance(custom, int | float) and custom > 0:
        timeout = int(custom)
        is_custom = True
    return {
        "query_timeout_seconds": timeout,
        "query_timeout_is_custom": is_custom,
        "default_query_timeout_seconds": DEFAULT_QUERY_TIMEOUT_SECONDS,
        "min_query_timeout_seconds": MIN_QUERY_TIMEOUT_SECONDS,
        "max_query_timeout_seconds": MAX_QUERY_TIMEOUT_SECONDS,
    }


def get_query_timeout_seconds() -> int:
    """Effective analysis timeout in seconds (custom override or default)."""
    return int(load_runtime_settings()["query_timeout_seconds"])


def save_runtime_settings(query_timeout_seconds: int | None = None) -> dict[str, Any]:
    """Validate + persist runtime-setting overrides.

    ``None`` leaves the timeout untouched; ``0`` resets it to the default.
    Out-of-range values raise ``ValueError``. Returns the merged settings for
    echoing back to the UI.
    """
    try:
        existing = storage.read_json(_RUNTIME_SETTINGS_KEY) or {}
    except Exception:  # pragma: no cover - defensive
        existing = {}
    to_store: dict[str, Any] = {
        k: v for k, v in existing.items() if k in ("query_timeout_seconds",)
    }

    if query_timeout_seconds is not None:
        if query_timeout_seconds == 0:
            to_store.pop("query_timeout_seconds", None)  # reset to default
        elif not (MIN_QUERY_TIMEOUT_SECONDS <= query_timeout_seconds <= MAX_QUERY_TIMEOUT_SECONDS):
            raise ValueError(
                f"Query timeout must be between {MIN_QUERY_TIMEOUT_SECONDS} and "
                f"{MAX_QUERY_TIMEOUT_SECONDS} seconds (or 0 to reset to the "
                f"default of {DEFAULT_QUERY_TIMEOUT_SECONDS})."
            )
        else:
            to_store["query_timeout_seconds"] = int(query_timeout_seconds)

    storage.write_json(_RUNTIME_SETTINGS_KEY, to_store)
    logger.info("Runtime settings saved", settings=to_store)
    return load_runtime_settings()
