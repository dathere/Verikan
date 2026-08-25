"""Landing-page configuration store (#109).

Lets admins customize the phrasing shown on the public landing page so the
app can be re-skinned for different data portals (Chris's request):

* **Above the prompt** — title, optional Beta badge, logo, and a
  call-to-action lead paragraph.
* **Below the prompt** — a "Try asking about ..." heading, a "Powered by"
  label + link to the source portal, and an admin-curated list of sample
  questions.

Settings persist via the shared storage backend (same pattern as
``github_settings``). Defaults reproduce the previously-hardcoded
Pittsburgh/WPRDC landing page, so behavior is unchanged until an admin
edits anything.

NOTE: click-tracked "Popular Questions" are intentionally deferred to a
follow-up ticket; only admin-curated sample questions are supported here.
"""

from typing import Any

from data_concierge.core.logging import get_logger
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

# Storage key for landing-page settings (persisted across restarts).
_LANDING_SETTINGS_KEY = "landing_settings.json"

# Defaults reproduce the original hardcoded landing page.
DEFAULT_LANDING_SETTINGS: dict[str, Any] = {
    "title": "Verikan",
    "tagline": "The verified Data Concierge",
    "show_beta_badge": True,
    "logo_url": "/static/images/logo.png",
    "call_to_action": (
        "Turn natural-language questions about government & public data into "
        "citation-backed answers and reproducible notebooks — every answer "
        "independently verifiable."
    ),
    "search_placeholder": "Ask about economic, demographic, or government data...",
    "try_asking_label": "Try asking about Pittsburgh data...",
    "powered_by_label": "WPRDC - City of Pittsburgh",
    "powered_by_url": "https://data.wprdc.org/organization/city-of-pittsburgh",
    "sample_questions": [
        "What are the most common types of 311 requests in Pittsburgh?",
        "Which Pittsburgh neighborhoods have the most police incidents?",
        "How many building permits were issued in Pittsburgh last year?",
        "Show me community center attendance trends in Pittsburgh",
        "What is the distribution of property values across Pittsburgh neighborhoods?",
    ],
}

# Fields that must be plain strings when provided.
_STRING_FIELDS = (
    "title",
    "tagline",
    "logo_url",
    "call_to_action",
    "search_placeholder",
    "try_asking_label",
    "powered_by_label",
    "powered_by_url",
)


def load_landing_settings() -> dict[str, Any]:
    """Load landing-page settings, merged over defaults.

    Unknown/missing keys fall back to ``DEFAULT_LANDING_SETTINGS`` so a
    partially-saved or older settings file still renders a complete page.
    A read error degrades gracefully to defaults (the landing page should
    never fail to render because of a corrupt settings file).
    """
    merged = dict(DEFAULT_LANDING_SETTINGS)
    merged["sample_questions"] = list(DEFAULT_LANDING_SETTINGS["sample_questions"])
    try:
        saved = storage.read_json(_LANDING_SETTINGS_KEY)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to read landing settings; using defaults", error=str(e))
        return merged
    if saved:
        for key, value in saved.items():
            if key in DEFAULT_LANDING_SETTINGS:
                merged[key] = value
    return merged


def save_landing_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Validate, normalize, and persist landing-page settings.

    Only known keys are stored. Strings are stripped; ``sample_questions``
    is coerced to a list of non-empty trimmed strings. Updates are merged
    over any previously-saved values, so a partial save only changes the
    fields it provides. The fully-merged settings (defaults + saved) are
    returned for echoing back to the UI.
    """
    try:
        existing = storage.read_json(_LANDING_SETTINGS_KEY) or {}
    except Exception:  # pragma: no cover - defensive
        existing = {}
    to_store: dict[str, Any] = {k: v for k, v in existing.items() if k in DEFAULT_LANDING_SETTINGS}

    for field in _STRING_FIELDS:
        if field in settings and settings[field] is not None:
            to_store[field] = str(settings[field]).strip()

    if "show_beta_badge" in settings:
        to_store["show_beta_badge"] = bool(settings["show_beta_badge"])

    if "sample_questions" in settings and settings["sample_questions"] is not None:
        raw = settings["sample_questions"]
        if not isinstance(raw, list):
            raise ValueError("sample_questions must be a list of strings")
        cleaned = [str(q).strip() for q in raw if str(q).strip()]
        to_store["sample_questions"] = cleaned

    storage.write_json(_LANDING_SETTINGS_KEY, to_store)
    logger.info("Landing-page settings saved", fields=sorted(to_store.keys()))
    return load_landing_settings()
