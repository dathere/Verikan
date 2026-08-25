"""Notebook editing for chat follow-ups (structural operations).

The LLM decides *what* to change; ``apply_edit`` decides what a change is
allowed to do. These tests pin the structural contract: bounds are enforced,
outputs of replaced code cells are cleared (the notebook is re-executed and
re-reviewed after editing), the notebook can never be emptied, and errors
come back as strings for the model to recover from — never exceptions.
"""

import pytest

from data_concierge.agents.notebook_editor import (
    EditResult,
    apply_edit,
    edit_notebook,
    render_notebook_cells,
)


def make_nb() -> dict:
    return {
        "cells": [
            {"cell_type": "markdown", "source": "# Title", "metadata": {}},
            {
                "cell_type": "code",
                "source": "print(1)",
                "metadata": {},
                "outputs": [{"output_type": "stream", "text": "1\n"}],
                "execution_count": 1,
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


class TestReplace:
    def test_replace_clears_stale_outputs(self) -> None:
        nb = make_nb()
        result = apply_edit(nb, "replace_cell", {"index": 1, "source": "print(2)"})
        assert result.startswith("Replaced cell 1")
        cell = nb["cells"][1]
        assert cell["source"] == "print(2)"
        assert cell["outputs"] == []
        assert cell["execution_count"] is None

    def test_replace_can_switch_cell_type(self) -> None:
        nb = make_nb()
        apply_edit(nb, "replace_cell", {"index": 1, "source": "note", "cell_type": "markdown"})
        assert nb["cells"][1]["cell_type"] == "markdown"
        assert "outputs" not in nb["cells"][1]

    def test_out_of_range_is_an_error_string(self) -> None:
        nb = make_nb()
        result = apply_edit(nb, "replace_cell", {"index": 9, "source": "x"})
        assert result.startswith("Error")
        assert "2 cells" in result

    def test_bad_index_type_is_an_error_string(self) -> None:
        assert apply_edit(make_nb(), "replace_cell", {"source": "x"}).startswith("Error")

    def test_bad_cell_type_is_rejected(self) -> None:
        result = apply_edit(
            make_nb(), "replace_cell", {"index": 1, "source": "x", "cell_type": "raw"}
        )
        assert result.startswith("Error")


class TestInsertDelete:
    def test_insert_at_index(self) -> None:
        nb = make_nb()
        result = apply_edit(nb, "insert_cell", {"index": 1, "cell_type": "code", "source": "x = 1"})
        assert "Inserted code cell at index 1" in result
        assert nb["cells"][1]["source"] == "x = 1"
        assert len(nb["cells"]) == 3

    def test_insert_past_end_appends_and_clamps(self) -> None:
        nb = make_nb()
        apply_edit(nb, "insert_cell", {"index": 99, "cell_type": "markdown", "source": "end"})
        assert nb["cells"][-1]["source"] == "end"

    def test_new_code_cell_has_notebook_shape(self) -> None:
        nb = make_nb()
        apply_edit(nb, "insert_cell", {"index": 0, "cell_type": "code", "source": "s"})
        cell = nb["cells"][0]
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        assert cell["metadata"] == {}

    def test_delete(self) -> None:
        nb = make_nb()
        result = apply_edit(nb, "delete_cell", {"index": 0})
        assert result.startswith("Deleted cell 0")
        assert len(nb["cells"]) == 1

    def test_cannot_empty_the_notebook(self) -> None:
        nb = {"cells": [{"cell_type": "code", "source": "x", "metadata": {}, "outputs": []}]}
        assert apply_edit(nb, "delete_cell", {"index": 0}).startswith("Error")
        assert len(nb["cells"]) == 1

    def test_unknown_tool_is_reported(self) -> None:
        assert "Unknown edit tool" in apply_edit(make_nb(), "rename_cell", {"index": 0})


class TestRendering:
    def test_cells_indexed_and_typed(self) -> None:
        text = render_notebook_cells(make_nb())
        assert "--- cell 0 (markdown) ---" in text
        assert "--- cell 1 (code) ---" in text

    def test_list_form_source_is_joined(self) -> None:
        nb = {"cells": [{"cell_type": "code", "source": ["a = 1\n", "print(a)\n"]}]}
        assert "a = 1\nprint(a)" in render_notebook_cells(nb)


class TestEditNotebookDegradation:
    """The editor fails soft: the caller falls back to a fresh analysis."""

    async def test_unavailable_client_returns_error_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.agents import llm_agent as la
        from data_concierge.agents.llm_agent import get_llm_agent

        monkeypatch.setattr(la, "ANTHROPIC_AVAILABLE", False)
        monkeypatch.setattr(get_llm_agent(), "_anthropic", None)

        nb = make_nb()
        result = await edit_notebook(
            nb,
            instruction="fix the chart",
            query="original question",
            previous_answer="answer",
            data_source="wprdc",
        )
        assert isinstance(result, EditResult)
        assert result.error
        assert result.notebook is None
        # The caller's notebook must never be mutated — assert on the actual
        # object passed in, not a freshly built copy.
        assert nb == make_nb()


class _FakeBlock:
    def __init__(self, **kw: object) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeResponse:
    def __init__(self, stop_reason: str, content: list) -> None:
        self.stop_reason = stop_reason
        self.content = content
        self.id = "msg_fake"
        self.model = "fake-model"
        self.usage = _FakeBlock(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )


class TestEditLoopScripted:
    """Drive the real edit loop with scripted LLM responses (no network).

    Pins the review findings' fixes: a batched second structural edit is
    rejected (stale indices), edit calls never enter the data-retrieval
    signals (#132), the evidence log carries system_prompt_sha256 and
    token_totals, and the caller's notebook stays untouched.
    """

    async def _run_scripted(
        self, monkeypatch: pytest.MonkeyPatch, responses: list, nb: dict
    ) -> EditResult:
        from data_concierge.agents.llm_agent import get_llm_agent

        agent = get_llm_agent()
        monkeypatch.setattr(agent, "_get_anthropic_client", lambda: object())
        monkeypatch.setattr(agent, "_get_mcp_tools", lambda: [])
        queue = list(responses)

        async def fake_call(client: object, **kwargs: object) -> _FakeResponse:
            return queue.pop(0)

        monkeypatch.setattr(agent, "_call_llm_with_retry", fake_call)
        return await edit_notebook(
            nb,
            instruction="delete the broken cells",
            query="original question",
            previous_answer="",
            data_source="wprdc",
        )

    async def test_second_structural_edit_in_a_batch_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nb = {
            "cells": [
                {"cell_type": "code", "source": f"print({i})", "metadata": {}, "outputs": []}
                for i in range(5)
            ]
        }
        original = copy_of(nb)
        responses = [
            _FakeResponse(
                "tool_use",
                [
                    _FakeBlock(type="tool_use", name="delete_cell", input={"index": 1}, id="t1"),
                    # Pre-edit index: after the first delete this would hit
                    # the wrong cell, so it must be rejected, not applied.
                    _FakeBlock(type="tool_use", name="delete_cell", input={"index": 3}, id="t2"),
                ],
            ),
            _FakeResponse("end_turn", [_FakeBlock(type="text", text="Removed the broken cell.")]),
        ]
        result = await self._run_scripted(monkeypatch, responses, nb)

        assert result.error is None
        assert result.edits_applied == 1
        assert result.notebook is not None
        sources = [c["source"] for c in result.notebook["cells"]]
        assert sources == ["print(0)", "print(2)", "print(3)", "print(4)"]
        # The rejected call is logged as an error for the model to recover from.
        tool_entries = [e for e in result.agent_log if e["type"] == "tool_execution"]
        assert tool_entries[1]["status"] == "error"
        assert "indices shifted" in tool_entries[1]["result"]
        # Caller's notebook untouched.
        assert nb == original

    async def test_edit_calls_never_enter_retrieval_signals(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A markdown-only revision must leave data-retrieval unmeasurable —
        counting mechanical cell edits as retrieval successes fabricated a
        ~0.96 'measured' retrieval score (#132)."""
        nb = make_nb()
        responses = [
            _FakeResponse(
                "tool_use",
                [
                    _FakeBlock(
                        type="tool_use",
                        name="replace_cell",
                        input={"index": 0, "source": "# Better title"},
                        id="t1",
                    )
                ],
            ),
            _FakeResponse("end_turn", [_FakeBlock(type="text", text="Fixed the title.")]),
        ]
        result = await self._run_scripted(monkeypatch, responses, nb)

        assert result.edits_applied == 1
        assert result.tool_signals.successful_tool_calls == 0
        assert result.tool_signals.failed_tool_calls == 0

        from data_concierge.core.confidence import confidence_calculator

        confidence = confidence_calculator.calculate_from_signals(
            tool_signals=result.tool_signals,
            final_answer=result.answer,
            tool_results=result.tool_result_texts,
        )
        assert confidence.data_retrieval_quality is None
        assert "data_retrieval_quality" in confidence.unavailable

    async def test_evidence_log_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        nb = make_nb()
        responses = [
            _FakeResponse("end_turn", [_FakeBlock(type="text", text="Nothing to change.")]),
        ]
        result = await self._run_scripted(monkeypatch, responses, nb)

        session_start = result.agent_log[0]
        assert session_start["type"] == "session_start"
        assert len(session_start["system_prompt_sha256"]) == 64
        summary = result.agent_log[-1]
        assert summary["type"] == "summary"
        assert summary["token_totals"]["input"] == 100
        assert summary["token_totals"]["output"] == 20


def copy_of(nb: dict) -> dict:
    import copy

    return copy.deepcopy(nb)
