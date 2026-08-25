"""Generated notebook cells must always be valid Python (#131).

A production MCP notebook failed verification with a SyntaxError because the
reproducible-code template commented only the first line of the arguments
JSON — every line after ``# {`` was bare JSON. These tests pin the fix at
three layers: the MCP codegen emits valid parseable cells for hostile inputs,
the llm_agent fallback does too, and the notebook generator's compile guard
comments out anything that still slips through.
"""

import json

from data_concierge.agents.llm_agent import LLMAnalysisAgent
from data_concierge.agents.notebook_generator import _compile_safe_code_cell
from data_concierge.mcp.connector import MCPDataConnector

NASTY_RESULT = 'He said """quotes""" and \\ backslashes\nand newlines {"a": 1}'


def gen(server_id: str, tool: str, args: dict, result: str) -> str:
    return MCPDataConnector().generate_reproducible_code(server_id, tool, args, result)


class TestMCPCodegen:
    def test_multiline_arguments_compile(self) -> None:
        """The exact production failure: a dict argument rendered as
        multi-line JSON with only its first line commented."""
        code = gen(
            "fbi-crime-data",
            "lookup_agency",
            {"lookup_type": "by_state", "state_abbr": "PA"},
            json.dumps({"agencies": [{"ori": "PAPPD0000"}]}),
        )
        compile(code, "<cell>", "exec")

    def test_hostile_result_content_compiles(self) -> None:
        """Triple quotes, backslashes, and newlines in the result must not
        break the cell — the old template pasted them into a raw triple-quoted
        string."""
        code = gen("fbi-crime-data", "t", {"x": 1}, NASTY_RESULT)
        compile(code, "<cell>", "exec")

    def test_result_is_parsed_and_exposed(self) -> None:
        """The cell parses the embedded JSON into `result` so later cells can
        compute from it, instead of shipping an inert comment block."""
        payload = {"rates": {"2019": 215.79}, "population": 302971}
        code = gen("fbi-crime-data", "summarized", {"offense": "P"}, json.dumps(payload))
        namespace: dict = {}
        exec(compile(code, "<cell>", "exec"), namespace)  # noqa: S102 - our own generated code
        assert namespace["result"] == payload
        assert namespace["arguments"] == {"offense": "P"}

    def test_non_json_result_degrades_to_text(self) -> None:
        code = gen("fbi-crime-data", "t", {}, "plain prose result")
        namespace: dict = {}
        exec(compile(code, "<cell>", "exec"), namespace)  # noqa: S102
        assert namespace["result"] == "plain prose result"

    def test_oversized_result_is_truncated_but_still_compiles(self) -> None:
        code = gen("fbi-crime-data", "t", {}, "x" * 20000)
        compile(code, "<cell>", "exec")
        assert "truncated" in code


class TestLLMAgentMCPFallback:
    def test_fallback_code_compiles_with_nested_arguments(self) -> None:
        code = LLMAnalysisAgent._code_for_mcp_tool(
            LLMAnalysisAgent.__new__(LLMAnalysisAgent),
            "mcp__unknown",  # malformed name -> connector path bails to fallback
            {"lookup_type": "by_state", "nested": {"a": [1, 2]}},
            "result",
        )
        compile(code, "<cell>", "exec")


class TestCompileGuard:
    def test_valid_code_passes_through(self) -> None:
        cell = _compile_safe_code_cell("x = 1\nprint(x)")
        assert cell.source == "x = 1\nprint(x)"

    def test_invalid_code_ships_commented_out(self) -> None:
        broken = '# Arguments:\n# {\n  "lookup_type": "by_state",\n}'
        cell = _compile_safe_code_cell(broken, "Step 1")
        compile(cell.source, "<cell>", "exec")  # the emitted cell always parses
        assert "did not parse" in cell.source
        assert "# # {" in cell.source or '#   "lookup_type"' in cell.source
