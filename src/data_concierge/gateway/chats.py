"""Server-side chat session storage per authenticated user.

Stores each user's conversations in ``chats/{user_id}.json`` via the unified
storage backend so chats are shared across browsers and survive page refreshes.

Notebook JSON is intentionally excluded — only lightweight message metadata
is stored (the full notebook can always be re-fetched via its ``queryId``).
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime
from typing import Any

from data_concierge.core.logging import get_logger
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

_PREFIX = "chats"

# Per-user locks serialize the read-modify-write of a user's chats file so
# concurrent PUT/DELETE requests for the same user can't lose writes (#95).
# Keyed by user_id; created on demand. Access to this dict happens only
# inside synchronous helpers (no await between get-or-create and use), so
# the dict itself needs no extra guarding under a single event loop.
_user_locks: dict[str, asyncio.Lock] = {}


def _lock_for(user_id: str) -> asyncio.Lock:
    lock = _user_locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        _user_locks[user_id] = lock
    return lock

# Fields to strip from messages before persisting (can be very large)
_STRIP_MESSAGE_FIELDS = {"notebook", "quickAnswer"}


def _key(user_id: str) -> str:
    """Return the storage key for a user's chats file."""
    # Sanitise to avoid path traversal or filesystem issues
    safe = (
        user_id.strip()
        .lower()
        .replace("/", "_")
        .replace("\\", "_")
        .replace("..", "_")
        .replace("@", "_at_")
    )
    return f"{_PREFIX}/{safe}.json"


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _strip_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a message with large fields removed."""
    return {k: v for k, v in msg.items() if k not in _STRIP_MESSAGE_FIELDS}


def load_chats(user_id: str) -> dict[str, Any]:
    """Load all chats for a user. Returns ``{chat_id: chat_data}``."""
    data = storage.read_json(_key(user_id))
    if data and isinstance(data.get("chats"), dict):
        return data["chats"]
    return {}


async def save_chat(user_id: str, chat_id: str, chat_data: dict[str, Any]) -> dict[str, Any]:
    """Create or update a chat for a user. Returns the stored entry."""
    entry: dict[str, Any] = dict(chat_data)
    entry["id"] = chat_id
    entry["synced_at"] = _now()

    # Strip heavy fields from each message before persisting
    if isinstance(entry.get("messages"), list):
        entry["messages"] = [_strip_message(m) for m in entry["messages"]]

    async with _lock_for(user_id):
        chats = load_chats(user_id)
        chats[chat_id] = entry
        # Persist de-duplication at WRITE time, under this user's lock (so it is
        # serialized with other PUT/DELETE and can't race). Only EXACT duplicates
        # collapse and the just-saved chat is protected, so nothing unique is lost
        # and storage never accumulates redundant copies. GET stays a pure read.
        chats, removed = dedupe_chats(chats, protected_id=chat_id)
        storage.write_json(_key(user_id), {"chats": chats})
    if removed:
        logger.debug("Pruned duplicate chats on save", user=user_id, removed=removed)
    logger.debug("Chat saved", user=user_id, chat_id=chat_id)
    return entry


async def delete_chat(user_id: str, chat_id: str) -> bool:
    """Delete a chat. Returns ``True`` if it existed and was removed."""
    async with _lock_for(user_id):
        chats = load_chats(user_id)
        if chat_id not in chats:
            return False
        del chats[chat_id]
        storage.write_json(_key(user_id), {"chats": chats})
    logger.debug("Chat deleted", user=user_id, chat_id=chat_id)
    return True


# ---------------------------------------------------------------------------
# De-duplication (server-side, non-destructive view)
# ---------------------------------------------------------------------------
# Duplicates accumulate when a conversation is synced from multiple devices or
# re-created. We collapse only EXACT duplicates — same title AND same full
# message sequence — so two genuinely different conversations that merely open
# with the same question are NEVER merged. De-duplication runs in two places,
# both fully in the backend (no client button):
#   * WRITE time — ``save_chat`` persists the deduped map under the per-user
#     lock, so storage never accumulates redundant copies. The just-saved chat
#     is protected, so a write can't delete its own target.
#   * READ time — ``deduped_chats_view`` returns a deduped VIEW (non-mutating)
#     as a defensive backstop for any legacy duplicates not yet pruned by a
#     subsequent write.
# Because only identical conversations collapse and the survivor choice is
# deterministic, nothing unique is ever lost and the result can't oscillate.


def _chat_signature(chat: dict[str, Any]) -> str:
    """Signature of a conversation: title + its full (role, content) sequence.

    Only conversations that are identical in full collapse together. In-progress
    chats with no user message yet get an empty signature and are never merged.
    Non-dict / malformed entries return "" so a single bad record can't break
    the whole list.
    """
    if not isinstance(chat, dict):
        return ""
    messages = chat.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""

    saw_user = False
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        content = str(msg.get("content") or "")
        if role == "user" and content:
            saw_user = True
        parts.append(role + "\n" + content)
    if not saw_user:
        return ""

    raw = str(chat.get("title") or "") + "\n--\n" + "\n--\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _chat_richness(chat: dict[str, Any]) -> tuple[int, str]:
    """Tie-break key among exact duplicates: more messages, then newer."""
    messages = chat.get("messages") if isinstance(chat, dict) else None
    count = len(messages) if isinstance(messages, list) else 0
    recency = ""
    if isinstance(chat, dict):
        recency = str(chat.get("createdAt") or chat.get("synced_at") or "")
    return (count, recency)


def dedupe_chats(
    chats: dict[str, Any], protected_id: str | None = None
) -> tuple[dict[str, Any], int]:
    """Collapse exact-duplicate conversations, keeping one representative per
    signature. Returns ``(deduped_chats, removed_count)``.

    Chats with no signature (empty / in-progress / malformed) are always kept.
    Because only identical conversations share a signature, the dropped copies
    carry no unique content. ``protected_id`` (the chat currently being saved /
    viewed) is always kept as its group's survivor so an in-flight write can
    never delete its own target.
    """
    groups: dict[str, list[str]] = {}
    for chat_id, chat in chats.items():
        sig = _chat_signature(chat)
        if not sig:
            continue
        groups.setdefault(sig, []).append(chat_id)

    remove: set[str] = set()
    for ids in groups.values():
        if len(ids) < 2:
            continue
        ranked = sorted(ids, key=lambda cid: _chat_richness(chats[cid]), reverse=True)
        # Keep the protected chat if it's in this group; otherwise the richest.
        keep = protected_id if protected_id in ids else ranked[0]
        remove.update(cid for cid in ids if cid != keep)

    if not remove:
        return chats, 0
    deduped = {cid: chat for cid, chat in chats.items() if cid not in remove}
    return deduped, len(remove)


def deduped_chats_view(user_id: str) -> dict[str, Any]:
    """Return the user's chats with exact duplicates collapsed — a read-only
    VIEW. Storage is never mutated, so nothing is permanently deleted."""
    deduped, _ = dedupe_chats(load_chats(user_id))
    return deduped
