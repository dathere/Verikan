"""Tests for the durability/hardening batch (issues #94, #95, #96, #101).

* #94 — atomic JSON/bytes writes in ``LocalStorage``
* #95 — locking around concurrent chat / session state writes
* #96 — registry + ``close_all_clients`` for HTTP connection pools
* #101 — read-only SELECT guard for ``run_sql_query``
"""

from __future__ import annotations

import asyncio
import json

import pytest

from data_concierge.agents.llm_agent import SQLValidationError, _validate_select_sql
from data_concierge.data_layer import connectors
from data_concierge.data_layer.storage import LocalStorage
from data_concierge.gateway import chats as chats_store
from data_concierge.gateway.session import SessionManager


# ---------------------------------------------------------------------------
# #94 — atomic writes
# ---------------------------------------------------------------------------
class TestAtomicWrites:
    def test_write_json_roundtrip(self, tmp_path):
        store = LocalStorage(tmp_path)
        store.write_json("a/b/data.json", {"hello": "world", "n": 1})
        assert store.read_json("a/b/data.json") == {"hello": "world", "n": 1}

    def test_write_bytes_roundtrip(self, tmp_path):
        store = LocalStorage(tmp_path)
        store.write_bytes("blob.bin", b"\x00\x01\x02payload")
        assert store.read_bytes("blob.bin") == b"\x00\x01\x02payload"

    def test_overwrite_leaves_valid_file(self, tmp_path):
        store = LocalStorage(tmp_path)
        store.write_json("state.json", {"v": 1})
        store.write_json("state.json", {"v": 2})
        assert store.read_json("state.json") == {"v": 2}

    def test_no_temp_files_left_behind(self, tmp_path):
        store = LocalStorage(tmp_path)
        store.write_json("dir/state.json", {"v": 1})
        leftovers = [p.name for p in (tmp_path / "dir").iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_written_file_is_valid_json_on_disk(self, tmp_path):
        store = LocalStorage(tmp_path)
        store.write_json("state.json", {"k": [1, 2, 3]})
        # The destination must be a complete, parseable file (never a
        # truncated partial write).
        raw = (tmp_path / "state.json").read_text()
        assert json.loads(raw) == {"k": [1, 2, 3]}


# ---------------------------------------------------------------------------
# #101 — SELECT-only SQL guard
# ---------------------------------------------------------------------------
class TestSQLGuard:
    @pytest.mark.parametrize(
        "sql",
        [
            'SELECT * FROM "t" LIMIT 5',
            'select count(*) from "t"',
            'WITH x AS (SELECT 1) SELECT * FROM x',
            "SELECT 1;",  # single trailing semicolon tolerated
        ],
    )
    def test_accepts_select(self, sql):
        out = _validate_select_sql(sql)
        assert out.lower().startswith(("select", "with"))

    @pytest.mark.parametrize(
        "sql",
        [
            'DELETE FROM "t"',
            'UPDATE "t" SET a=1',
            'DROP TABLE "t"',
            "SELECT 1; DROP TABLE x",  # stacked statement
            'WITH d AS (DELETE FROM "t" RETURNING *) SELECT * FROM d',
            "",
        ],
    )
    def test_rejects_non_select(self, sql):
        with pytest.raises(SQLValidationError):
            _validate_select_sql(sql)

    def test_appends_limit_when_missing(self):
        out = _validate_select_sql('SELECT * FROM "t"', max_rows=500)
        assert out.endswith("LIMIT 500")

    def test_preserves_existing_limit(self):
        out = _validate_select_sql('SELECT * FROM "t" LIMIT 7')
        assert out.count("LIMIT") == 1
        assert out.endswith("LIMIT 7")


# ---------------------------------------------------------------------------
# #96 — close_all_clients registry
# ---------------------------------------------------------------------------
class TestCloseRegistry:
    async def test_close_all_clients_closes_registered(self):
        closed = {"async": False, "sync": False}

        class AsyncCloseable:
            async def close(self):
                closed["async"] = True

        class SyncCloseable:
            def close(self):
                closed["sync"] = True

        a, s = AsyncCloseable(), SyncCloseable()
        connectors.register_closeable(a)
        connectors.register_closeable(s)

        n = await connectors.close_all_clients()
        assert closed["async"] is True
        assert closed["sync"] is True
        assert n >= 2

    async def test_close_all_clients_survives_errors(self):
        ok = {"called": False}

        class Boom:
            async def close(self):
                raise RuntimeError("boom")

        class Fine:
            async def close(self):
                ok["called"] = True

        # Keep strong refs — the registry holds only weak refs, so unbound
        # instances would be GC'd before close_all_clients() runs.
        boom, fine = Boom(), Fine()
        connectors.register_closeable(boom)
        connectors.register_closeable(fine)
        # Must not raise even though one connector's close() fails.
        await connectors.close_all_clients()
        assert ok["called"] is True


# ---------------------------------------------------------------------------
# #95 — locking on concurrent state writes
# ---------------------------------------------------------------------------
class TestChatsLocking:
    async def test_concurrent_saves_do_not_drop_writes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(chats_store, "storage", LocalStorage(tmp_path))

        async def save(i: int):
            await chats_store.save_chat("user@x.com", f"chat{i}", {"messages": []})

        await asyncio.gather(*(save(i) for i in range(25)))

        stored = chats_store.load_chats("user@x.com")
        assert len(stored) == 25
        assert {f"chat{i}" for i in range(25)} == set(stored.keys())

    async def test_delete_after_save(self, tmp_path, monkeypatch):
        monkeypatch.setattr(chats_store, "storage", LocalStorage(tmp_path))
        await chats_store.save_chat("u", "c1", {"messages": []})
        assert await chats_store.delete_chat("u", "c1") is True
        assert await chats_store.delete_chat("u", "c1") is False

    async def test_save_persists_dedup_of_exact_duplicates(self, tmp_path, monkeypatch):
        # Exact-duplicate conversations collapse in storage at WRITE time, and
        # the just-saved chat is the protected survivor.
        monkeypatch.setattr(chats_store, "storage", LocalStorage(tmp_path))
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        await chats_store.save_chat("u", "old", {"title": "T", "messages": msgs})
        await chats_store.save_chat("u", "new", {"title": "T", "messages": msgs})

        stored = chats_store.load_chats("u")
        assert set(stored) == {"new"}  # duplicate "old" pruned, just-saved kept

    async def test_save_keeps_distinct_conversations(self, tmp_path, monkeypatch):
        monkeypatch.setattr(chats_store, "storage", LocalStorage(tmp_path))
        await chats_store.save_chat(
            "u", "a", {"title": "A", "messages": [{"role": "user", "content": "x"}]}
        )
        await chats_store.save_chat(
            "u", "b", {"title": "B", "messages": [{"role": "user", "content": "y"}]}
        )
        assert set(chats_store.load_chats("u")) == {"a", "b"}


class TestSessionLocking:
    async def test_create_get_update_delete(self):
        mgr = SessionManager()
        session = await mgr.create_session(user_id="alice")
        sid = session.session_id

        got = await mgr.get_session(sid)
        assert got is not None and got.user_id == "alice"

        await mgr.update_session(sid, query="hello")
        updated = await mgr.get_session(sid)
        assert updated is not None
        assert updated.query_history[-1] == "hello"

        assert await mgr.delete_session(sid) is True
        assert await mgr.get_session(sid) is None

    async def test_concurrent_creates_all_tracked(self):
        mgr = SessionManager()
        sessions = await asyncio.gather(
            *(mgr.create_session(user_id=f"u{i}") for i in range(50))
        )
        assert len({s.session_id for s in sessions}) == 50
        assert await mgr.get_active_session_count() == 50
