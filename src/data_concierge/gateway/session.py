"""Session management for user context tracking."""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.core.models import Session

logger = get_logger(__name__)


class SessionManager:
    """Manages user sessions and context."""

    def __init__(self) -> None:
        """Initialize session manager with in-memory storage."""
        # In production, this would use Redis
        self._sessions: dict[str, Session] = {}
        self._timeout = timedelta(minutes=settings.session_timeout_minutes)
        # Serializes all read-modify-write access to ``_sessions`` so
        # concurrent requests can't lose writes or hit transient KeyErrors
        # (#95). asyncio.Lock is not reentrant, so locked methods call the
        # ``_unlocked`` helpers below rather than each other.
        self._lock = asyncio.Lock()

    def _expired(self, session: Session) -> bool:
        return datetime.now() - session.last_active > self._timeout

    def _get_session_unlocked(self, session_id: str) -> Session | None:
        """Lookup + lazy-expiry without taking the lock (caller holds it)."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if self._expired(session):
            logger.info("Session expired", session_id=session_id)
            self._sessions.pop(session_id, None)
            return None
        return session

    async def create_session(self, user_id: str | None = None) -> Session:
        """Create a new session.

        Args:
            user_id: Optional user identifier

        Returns:
            New Session object
        """
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_active=datetime.now(),
        )
        async with self._lock:
            self._sessions[session_id] = session
        logger.info("Session created", session_id=session_id, user_id=user_id)
        return session

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session if found and not expired, None otherwise
        """
        async with self._lock:
            return self._get_session_unlocked(session_id)

    async def update_session(
        self,
        session_id: str,
        query: str | None = None,
        context_updates: dict[str, Any] | None = None,
    ) -> Session | None:
        """Update a session with new activity.

        Args:
            session_id: Session identifier
            query: Query to add to history
            context_updates: Context values to update

        Returns:
            Updated Session or None if not found
        """
        async with self._lock:
            session = self._get_session_unlocked(session_id)
            if session is None:
                return None

            session.last_active = datetime.now()

            if query:
                session.query_history.append(query)
                # Keep only last 10 queries
                if len(session.query_history) > 10:
                    session.query_history = session.query_history[-10:]

            if context_updates:
                session.context.update(context_updates)

            self._sessions[session_id] = session
            return session

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info("Session deleted", session_id=session_id)
                return True
            return False

    async def cleanup_expired_sessions(self) -> int:
        """Remove all expired sessions.

        Returns:
            Number of sessions removed
        """
        async with self._lock:
            now = datetime.now()
            expired = [
                sid
                for sid, session in self._sessions.items()
                if now - session.last_active > self._timeout
            ]
            for sid in expired:
                del self._sessions[sid]

        if expired:
            logger.info("Expired sessions cleaned up", count=len(expired))

        return len(expired)

    async def get_active_session_count(self) -> int:
        """Get count of active (non-expired) sessions."""
        await self.cleanup_expired_sessions()
        async with self._lock:
            return len(self._sessions)


# Global session manager instance
session_manager = SessionManager()
