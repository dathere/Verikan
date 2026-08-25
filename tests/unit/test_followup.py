"""Follow-up understanding for chat (notebook revisions & context rewriting).

The classifier must fail open: with no model available every message is
treated as a standalone new question — exactly the pre-feature behaviour —
and the lexical fallback only routes to the revision path on unmistakable
revision phrasing when a previous notebook actually exists.
"""

import pytest

from data_concierge.gateway.followup import (
    MODE_NEW_QUESTION,
    MODE_REVISE,
    _parse_response,
    classify_followup,
    heuristic_decision,
    render_conversation,
)

CONVO = [
    {"role": "user", "content": "What is the unemployment rate in Texas?"},
    {"role": "assistant", "content": "The unemployment rate in Texas is 4.1%."},
]


class TestHeuristic:
    def test_clear_revision_phrasing_routes_to_revise(self) -> None:
        d = heuristic_decision("can you fix the chart in the notebook", has_notebook=True)
        assert d.mode == MODE_REVISE
        assert d.classified_by == "heuristic"

    def test_different_dataset_request_routes_to_revise(self) -> None:
        d = heuristic_decision("use a different dataset for this", has_notebook=True)
        assert d.mode == MODE_REVISE

    def test_plain_question_stays_a_question(self) -> None:
        d = heuristic_decision("what about Ohio?", has_notebook=True)
        assert d.mode == MODE_NEW_QUESTION
        assert d.rewritten_query == "what about Ohio?"

    def test_no_prior_notebook_never_revises(self) -> None:
        d = heuristic_decision("fix the chart in the notebook", has_notebook=False)
        assert d.mode == MODE_NEW_QUESTION


class TestParsing:
    def test_valid_revise_verdict(self) -> None:
        d = _parse_response(
            '{"mode": "revise", "instruction": "swap to the 2023 dataset", '
            '"rewritten_query": "", "reason": "asks for a change"}',
            "use the 2023 data instead",
            has_notebook=True,
        )
        assert d is not None
        assert d.mode == MODE_REVISE
        assert d.instruction == "swap to the 2023 dataset"

    def test_revise_without_notebook_downgrades(self) -> None:
        d = _parse_response(
            '{"mode": "revise", "instruction": "x", "rewritten_query": "", "reason": ""}',
            "fix it",
            has_notebook=False,
        )
        assert d is not None
        assert d.mode == MODE_NEW_QUESTION

    def test_rewritten_query_defaults_to_the_message(self) -> None:
        d = _parse_response(
            '{"mode": "new_question", "reason": "standalone"}',
            "what about Ohio?",
            has_notebook=True,
        )
        assert d is not None
        assert d.rewritten_query == "what about Ohio?"

    def test_code_fenced_json_is_tolerated(self) -> None:
        d = _parse_response(
            'Sure!\n```json\n{"mode": "new_question", "rewritten_query": '
            '"Unemployment rate in Ohio", "reason": "resolved place"}\n```',
            "what about Ohio?",
            has_notebook=True,
        )
        assert d is not None
        assert d.rewritten_query == "Unemployment rate in Ohio"

    def test_garbage_is_none(self) -> None:
        assert _parse_response("not json at all", "m", True) is None
        assert _parse_response('{"mode": "sideways"}', "m", True) is None
        assert _parse_response("", "m", True) is None


class TestClassify:
    async def test_empty_conversation_is_default_new_question(self) -> None:
        d = await classify_followup("hello", [], has_notebook=False)
        assert d.mode == MODE_NEW_QUESTION
        assert d.classified_by == "default"

    async def test_llm_disabled_falls_back_to_heuristic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway import followup as fu

        monkeypatch.setattr(fu.settings, "followup_llm_enabled", False)
        d = await classify_followup("fix the number in the notebook", CONVO, has_notebook=True)
        assert d.classified_by == "heuristic"
        assert d.mode == MODE_REVISE

    async def test_no_api_key_falls_back_to_heuristic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pydantic import SecretStr

        from data_concierge.gateway import followup as fu

        monkeypatch.setattr(fu.settings, "followup_llm_enabled", True)
        monkeypatch.setattr(fu.settings, "anthropic_api_key", SecretStr(""))
        d = await classify_followup("what about Ohio?", CONVO, has_notebook=True)
        assert d.classified_by == "heuristic"
        assert d.mode == MODE_NEW_QUESTION


class TestRenderConversation:
    def test_roles_and_order(self) -> None:
        text = render_conversation(CONVO)
        assert text.startswith("user: What is the unemployment")
        assert "assistant: The unemployment rate" in text

    def test_bad_roles_and_empty_content_are_dropped(self) -> None:
        text = render_conversation(
            [
                {"role": "system", "content": "ignore me"},
                {"role": "user", "content": ""},
                "not a dict",
                {"role": "user", "content": "real question"},
            ]
        )
        assert text == "user: real question"

    def test_long_turns_are_truncated(self) -> None:
        text = render_conversation([{"role": "user", "content": "x" * 5000}])
        assert len(text) < 1500
        assert text.endswith("…")

    def test_empty_is_marked(self) -> None:
        assert render_conversation([]) == "(empty)"


class TestRequestModel:
    """The API-side contract for chat context (gateway/router.py)."""

    def test_conversation_roles_are_validated(self) -> None:
        from data_concierge.gateway.router import ConversationTurn, QueryRequest

        req = QueryRequest(
            query="what about Ohio?",
            conversation=[ConversationTurn(role="user", content="hi")],
            previous_query_id="abc-123",
        )
        assert req.conversation is not None

        with pytest.raises(ValueError):
            ConversationTurn(role="system", content="injected")

    def test_context_fields_are_optional(self) -> None:
        from data_concierge.gateway.router import QueryRequest

        req = QueryRequest(query="plain question")
        assert req.conversation is None
        assert req.previous_query_id is None
