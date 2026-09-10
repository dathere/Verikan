"""Guards on how a dead DataStore SQL endpoint is handled.

WPRDC's `datastore_search_sql` began answering HTTP 403 behind a CDN. The
agent retried it five times per query, each retry a guaranteed failure that
counted against the retrieval-success confidence signal (0.786 -> 0.417 on a
measured run), and the model was handed 500 characters of CloudFront HTML as
its only explanation.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from data_concierge.agents.llm_agent import (
    _SQL_ENDPOINT_DEAD_STATUSES,
    LLMAnalysisAgent,
    _summarize_error_body,
)

BLOCKED_HTML = (
    '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">'
    "<HTML><HEAD><TITLE>ERROR: The request could not be satisfied</TITLE>"
    "</HEAD><BODY><H1>403 ERROR</H1><H2>The request could not be satisfied.</H2>"
    "</BODY></HTML>"
)


def _agent_with(handler) -> tuple[LLMAnalysisAgent, httpx.AsyncClient]:
    agent = LLMAnalysisAgent()
    agent._sql_disabled = {}
    client = httpx.AsyncClient(
        base_url="https://portal.example.org",
        transport=httpx.MockTransport(handler),
    )
    return agent, client


class TestEndpointLevelFailures:
    @pytest.mark.parametrize("status", sorted(_SQL_ENDPOINT_DEAD_STATUSES))
    def test_dead_statuses_trip_the_breaker(self, status):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(status, text=BLOCKED_HTML)

        agent, client = _agent_with(handler)

        async def run():
            first = await agent._tool_sql(client, {"sql": "SELECT 1"})
            second = await agent._tool_sql(client, {"sql": "SELECT 2"})
            third = await agent._tool_sql(client, {"sql": "SELECT 3"})
            await client.aclose()
            return first, second, third

        first, second, third = asyncio.run(run())
        # Only the FIRST attempt reaches the network.
        assert calls["n"] == 1
        for msg in (first, second, third):
            assert "NOT available on this portal" in msg
            assert "Do NOT call run_sql_query" in msg
            assert "load_resource_data" in msg

    def test_the_model_is_not_fed_raw_html(self):
        def handler(request):
            return httpx.Response(403, text=BLOCKED_HTML)

        agent, client = _agent_with(handler)
        msg = asyncio.run(agent._tool_sql(client, {"sql": "SELECT 1"}))
        asyncio.run(client.aclose())
        assert "<HTML>" not in msg and "<H1>" not in msg and "DOCTYPE" not in msg

    def test_breaker_is_scoped_to_one_portal(self):
        """A blocked portal must not disable SQL for a working one."""
        def dead(request):
            return httpx.Response(403, text=BLOCKED_HTML)

        agent, blocked_client = _agent_with(dead)

        def alive(request):
            return httpx.Response(
                200,
                json={"success": True, "result": {"records": [{"n": 1}],
                                                  "fields": [{"id": "n"}]}},
            )

        good_client = httpx.AsyncClient(
            base_url="https://other.example.org",
            transport=httpx.MockTransport(alive),
        )

        async def run():
            bad = await agent._tool_sql(blocked_client, {"sql": "SELECT 1"})
            ok = await agent._tool_sql(good_client, {"sql": "SELECT 1"})
            await blocked_client.aclose()
            await good_client.aclose()
            return bad, ok

        bad, ok = asyncio.run(run())
        assert "NOT available" in bad
        assert "Rows: 1" in ok


class TestQueryLevelFailuresStayRetryable:
    @pytest.mark.parametrize("status", [400, 409, 500])
    def test_query_errors_do_not_disable_the_endpoint(self, status):
        """CKAN reports a bad statement with 400/409 — a different query may
        well work, so these must not trip the breaker."""
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(status, text='{"error": "syntax error"}')

        agent, client = _agent_with(handler)

        async def run():
            await agent._tool_sql(client, {"sql": "SELECT 1"})
            msg = await agent._tool_sql(client, {"sql": "SELECT 2"})
            await client.aclose()
            return msg

        msg = asyncio.run(run())
        assert calls["n"] == 2, "a retryable error must still reach the portal"
        assert "NOT available on this portal" not in msg
        assert agent._sql_disabled == {}

    def test_invalid_sql_is_still_rejected_locally(self):
        def handler(request):  # pragma: no cover - must never be reached
            raise AssertionError("validation should short-circuit")

        agent, client = _agent_with(handler)
        msg = asyncio.run(agent._tool_sql(client, {"sql": "DROP TABLE x"}))
        asyncio.run(client.aclose())
        assert msg.startswith("SQL rejected")


class TestExpiry:
    def test_the_breaker_expires(self):
        """A transient block must not disable SQL for the process lifetime."""
        agent = LLMAnalysisAgent()
        agent._sql_disabled = {"https://p.example": (time.time() - 1, 403)}
        assert agent._sql_unavailable_status("https://p.example") is None
        assert agent._sql_disabled == {}

    def test_an_unexpired_entry_is_reported(self):
        agent = LLMAnalysisAgent()
        agent._sql_disabled = {"https://p.example": (time.time() + 600, 403)}
        assert agent._sql_unavailable_status("https://p.example") == 403

    def test_unknown_portal_is_usable(self):
        agent = LLMAnalysisAgent()
        agent._sql_disabled = {}
        assert agent._sql_unavailable_status("https://never-seen.example") is None


class TestErrorBodySummary:
    def test_html_is_flattened(self):
        out = _summarize_error_body(BLOCKED_HTML)
        assert "<" not in out and ">" not in out
        assert "could not be satisfied" in out

    def test_empty_body_is_safe(self):
        assert _summarize_error_body("") == ""

    def test_length_is_capped(self):
        assert len(_summarize_error_body("x" * 5000)) <= 200


class TestConfidenceAccounting:
    """A capability the portal lacks must not be scored as a retrieval."""

    def test_message_carries_the_unavailable_marker(self):
        msg = LLMAnalysisAgent._sql_unavailable_message(403, "blocked")
        assert msg.startswith(LLMAnalysisAgent.TOOL_UNAVAILABLE_PREFIX)

    def test_marker_is_not_mistaken_for_an_error(self):
        """`is_error` matches ('Error:', 'HTTP ', 'SQL error'); the unavailable
        message must not collide with those, or it would count as a failure."""
        msg = LLMAnalysisAgent._sql_unavailable_message(403)
        assert not msg.startswith(("Error:", "HTTP ", "SQL error"))

    def test_accounting_excludes_rather_than_counting_either_way(self):
        """Guards the intent: counting it a success would inflate a
        user-visible confidence number with a call that fetched nothing."""
        import inspect

        from data_concierge.agents import llm_agent

        src = inspect.getsource(llm_agent.LLMAnalysisAgent.process)
        assert "is_unavailable" in src
        # The unavailable branch must touch neither counter.
        branch = src.split("if is_unavailable:")[1].split("elif is_error:")[0]
        assert "successful_tool_calls" not in branch
        assert "failed_tool_calls" not in branch


class TestBreakerKeyAlignment:
    """The breaker is keyed by portal URL from two different places: _tool_sql
    derives it from httpx's base_url, while the tool-list check reads it from
    the registry config. If those disagree the tool is never withheld and the
    mismatch is silent."""

    def test_httpx_base_url_matches_the_registry_form(self):
        agent = LLMAnalysisAgent()
        agent._sql_disabled = {}

        def handler(request):
            return httpx.Response(403, text=BLOCKED_HTML)

        client = httpx.AsyncClient(
            base_url="https://data.example.org".rstrip("/"),
            transport=httpx.MockTransport(handler),
        )

        async def run():
            await agent._tool_sql(client, {"sql": "SELECT 1"})
            await client.aclose()

        asyncio.run(run())
        # The key the tool-list check would look up, built from a registry URL.
        assert agent._sql_unavailable_status("https://data.example.org") == 403

    def test_trailing_slash_does_not_create_a_second_entry(self):
        agent = LLMAnalysisAgent()
        agent._sql_disabled = {}

        def handler(request):
            return httpx.Response(403, text=BLOCKED_HTML)

        client = httpx.AsyncClient(
            base_url="https://data.example.org/".rstrip("/"),
            transport=httpx.MockTransport(handler),
        )

        async def run():
            await agent._tool_sql(client, {"sql": "SELECT 1"})
            await client.aclose()

        asyncio.run(run())
        assert list(agent._sql_disabled) == ["https://data.example.org"]


class TestMidRunWithdrawal:
    """Telling the model in prose not to retry was not enough. A measured trial
    had it call run_sql_query five times after being told the endpoint was
    blocked, so the tool is withdrawn from the tool list mid-run."""

    def test_process_removes_the_tool_when_it_reports_unavailable(self):
        import inspect

        from data_concierge.agents import llm_agent

        src = inspect.getsource(llm_agent.LLMAnalysisAgent.process)
        branch = src.split("if is_unavailable:")[1].split("elif is_error:")[0]
        # It must rebuild all_tools without the offending tool...
        assert "all_tools = [" in branch
        assert 't.get("name") != tool_name' in branch

    def test_tools_are_re_read_each_iteration(self):
        """Withdrawal only works if the loop passes the current list each time."""
        import inspect

        from data_concierge.agents import llm_agent

        src = inspect.getsource(llm_agent.LLMAnalysisAgent.process)
        assert "tools=all_tools" in src
        # The call site must sit inside the iteration loop, after the point
        # where all_tools can be rebuilt.
        assert src.index("all_tools = list(TOOLS)") < src.index("tools=all_tools")
