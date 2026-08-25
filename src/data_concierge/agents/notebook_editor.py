"""Edit a previously generated notebook in response to a chat follow-up.

When a user's next message asks to *change* the notebook they just received —
"use the 311 dataset instead", "that number looks wrong, fix it", "make it a
line chart" — regenerating from scratch would either reproduce the mistake or
discard work the user liked. This agent instead loads the existing notebook
and lets Claude edit it cell by cell through structured tools, with the same
data-retrieval tools the analysis agent uses so "a different dataset" can be
found, loaded, and validated for real rather than guessed at.

The result is a *new* notebook under a new query id (the original is kept
untouched for audit), a chat-friendly answer describing what changed, and an
agent log in the standard evidence format. The caller then schedules the
normal verification + adversarial review pass on the edited notebook, so an
edit is held to exactly the same standard as a fresh generation (#131).

Reuses ``LLMAnalysisAgent``'s client, retry/fallback, and tool executors via
the module singleton — the editor is a sibling of the analyst inside the same
package, not an external consumer.
"""

from __future__ import annotations

import copy
import hashlib
import time
from typing import Any

from pydantic import BaseModel, Field

from data_concierge.agents.llm_agent import (
    AGENT_LOG_FORMAT_VERSION,
    TOOLS,
    _operation_type_for_tool,
    _source_for_tool,
    _utc_now,
    get_llm_agent,
)
from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.core.models import ToolCallSignals

logger = get_logger(__name__)

_MAX_ITERATIONS = 10
_MAX_CELL_CHARS = 4000
_MAX_NOTEBOOK_CHARS = 60000
_MAX_TOOL_RESULT_CHARS = 10000

EDIT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "replace_cell",
        "description": (
            "Replace the source of an existing notebook cell. Keeps the cell "
            "type unless cell_type is given. Outputs of a replaced code cell "
            "are cleared (the notebook is re-executed later)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "0-based cell index"},
                "source": {"type": "string", "description": "Complete new cell source"},
                "cell_type": {
                    "type": "string",
                    "enum": ["code", "markdown"],
                    "description": "Optionally change the cell type",
                },
            },
            "required": ["index", "source"],
        },
    },
    {
        "name": "insert_cell",
        "description": (
            "Insert a new cell before the given index (an index past the end appends)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "0-based insertion position"},
                "cell_type": {"type": "string", "enum": ["code", "markdown"]},
                "source": {"type": "string", "description": "Complete cell source"},
            },
            "required": ["index", "cell_type", "source"],
        },
    },
    {
        "name": "delete_cell",
        "description": "Delete the cell at the given index.",
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "0-based cell index"},
            },
            "required": ["index"],
        },
    },
]

_SYSTEM_PROMPT = """You are the notebook revision agent of an AI data \
concierge. The user received a generated Jupyter notebook with their answer \
and has now asked for a change. Apply the requested change to the notebook \
using the cell-editing tools, keeping everything else intact.

Ground rules:

- Edit minimally. Change only what the request requires, but keep the \
notebook coherent: if new code changes a number or dataset, update the \
markdown cells (including the results cell near the bottom) that mention it.
- Never fabricate. If the change needs different data (another dataset, \
another time period), use the data tools to actually find and inspect it \
first — real resource IDs, real column names — and write code that computes \
results from that data. Never hardcode a result value into the notebook.
- Cell indices shift after insert/delete. Indices you are shown are only \
valid until your first structural edit; after inserting or deleting, reason \
from the tool results, which restate the current cell count.
- The notebook will be re-executed and adversarially reviewed after your \
edits, so the code must run top to bottom on its own.
- If the request cannot be satisfied (the data does not exist, the request \
is out of scope), make no edits and say so plainly in your final reply.

When you are done editing, reply with a short, friendly chat message \
describing what you changed and what the updated notebook shows. Do not \
mention tools or cell indices in that final message."""


class EditResult(BaseModel):
    """Outcome of a notebook-editing run."""

    notebook: dict[str, Any] | None = None
    answer: str = ""
    agent_log: list[dict[str, Any]] = Field(default_factory=list)
    execution_trace: list[dict[str, Any]] = Field(default_factory=list)
    tool_signals: ToolCallSignals = Field(default_factory=ToolCallSignals)
    tool_result_texts: list[str] = Field(default_factory=list)
    edits_applied: int = 0
    error: str | None = None


def _cell_excerpt(source: Any) -> str:
    text = "".join(source) if isinstance(source, list) else str(source or "")
    if len(text) > _MAX_CELL_CHARS:
        text = text[:_MAX_CELL_CHARS] + f"\n… [truncated, {len(text)} chars total]"
    return text


def render_notebook_cells(notebook: dict[str, Any]) -> str:
    """Indexed, size-capped rendering of the notebook for the editor prompt."""
    parts: list[str] = []
    total = 0
    for i, cell in enumerate(notebook.get("cells", [])):
        kind = cell.get("cell_type", "code")
        block = f"--- cell {i} ({kind}) ---\n{_cell_excerpt(cell.get('source'))}\n"
        total += len(block)
        if total > _MAX_NOTEBOOK_CHARS:
            parts.append("--- remaining cells omitted (size cap) ---")
            break
        parts.append(block)
    return "\n".join(parts)


def _new_cell(cell_type: str, source: str) -> dict[str, Any]:
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def apply_edit(notebook: dict[str, Any], tool_name: str, tool_input: dict[str, Any]) -> str:
    """Apply one edit tool call to ``notebook`` in place.

    Returns a human-readable result string for the model. Structural errors
    (bad index, emptying the notebook) are reported back as ``Error: …`` so
    the model can recover, never raised.
    """
    cells = notebook.setdefault("cells", [])
    try:
        index = int(tool_input["index"])
    except (KeyError, TypeError, ValueError):
        return "Error: 'index' must be an integer."

    if tool_name == "replace_cell":
        if not 0 <= index < len(cells):
            return f"Error: cell index {index} out of range (notebook has {len(cells)} cells)."
        source = str(tool_input.get("source") or "")
        cell = cells[index]
        new_type = tool_input.get("cell_type") or cell.get("cell_type", "code")
        if new_type not in ("code", "markdown"):
            return "Error: cell_type must be 'code' or 'markdown'."
        if new_type != cell.get("cell_type"):
            cells[index] = _new_cell(new_type, source)
        else:
            cell["source"] = source
            if new_type == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        return f"Replaced cell {index} ({new_type}). Notebook now has {len(cells)} cells."

    if tool_name == "insert_cell":
        cell_type = str(tool_input.get("cell_type") or "code")
        if cell_type not in ("code", "markdown"):
            return "Error: cell_type must be 'code' or 'markdown'."
        index = max(0, min(index, len(cells)))
        cells.insert(index, _new_cell(cell_type, str(tool_input.get("source") or "")))
        return f"Inserted {cell_type} cell at index {index}. Notebook now has {len(cells)} cells."

    if tool_name == "delete_cell":
        if not 0 <= index < len(cells):
            return f"Error: cell index {index} out of range (notebook has {len(cells)} cells)."
        if len(cells) == 1:
            return "Error: cannot delete the notebook's only cell."
        removed = cells.pop(index)
        return (
            f"Deleted cell {index} ({removed.get('cell_type', 'code')}). "
            f"Notebook now has {len(cells)} cells."
        )

    return f"Unknown edit tool: {tool_name}"


_EDIT_TOOL_NAMES = {t["name"] for t in EDIT_TOOLS}


async def edit_notebook(  # noqa: C901 - one linear tool loop, mirrors llm_agent.process
    notebook: dict[str, Any],
    instruction: str,
    query: str,
    previous_answer: str,
    data_source: str,
    conversation_text: str = "",
) -> EditResult:
    """Apply ``instruction`` to ``notebook`` via LLM tool calling.

    Operates on a deep copy — the caller's notebook is never mutated. On any
    failure the result carries ``error`` and no notebook; the caller decides
    how to degrade (typically: fall back to a fresh analysis run).
    """
    agent = get_llm_agent()
    result = EditResult()
    working = copy.deepcopy(notebook)

    try:
        client = agent._get_anthropic_client()
    except (ImportError, ValueError) as e:
        result.error = f"editor unavailable: {e}"
        return result

    portal_cfg = agent.get_portal_config(data_source)
    portal_url = portal_cfg.get("url", "")

    mcp_tools = agent._get_mcp_tools()
    from data_concierge.agents.llm_agent import _STATIC_PORTAL_CONFIGS

    if data_source in _STATIC_PORTAL_CONFIGS:
        data_tools: list[dict[str, Any]] = list(mcp_tools)
    else:
        data_tools = list(TOOLS) + list(mcp_tools)
    tools = EDIT_TOOLS + data_tools

    user_message = (
        f"## Original question\n{(query or '').strip()[:2000]}\n\n"
        f"## Answer previously given\n{(previous_answer or '').strip()[:3000]}\n\n"
        + (f"## Recent conversation\n{conversation_text[:3000]}\n\n" if conversation_text else "")
        + f"## Requested change\n{instruction.strip()[:2000]}\n\n"
        f"## Current notebook\n{render_notebook_cells(working)}"
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    model = settings.llm_model
    max_tokens = max(settings.llm_max_tokens, 8192)
    started = time.time()

    agent_log = result.agent_log
    agent_log.append(
        {
            "type": "session_start",
            "log_format_version": AGENT_LOG_FORMAT_VERSION,
            "timestamp": _utc_now(),
            "mode": "notebook_edit",
            "query": query,
            "instruction": instruction,
            "data_source": data_source,
            "portal_url": portal_url,
            "model": model,
            "max_iterations": _MAX_ITERATIONS,
            "tools_available": [t["name"] for t in tools],
            "system_prompt": _SYSTEM_PROMPT,
            # Same integrity field the analyst records — evidence packages
            # read it into skillTextHash (gateway/evidence.py).
            "system_prompt_sha256": hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        }
    )

    # Data-retrieval tool outcomes ONLY. Cell edits are mechanical operations,
    # not retrieval — counting them here would fabricate a measured
    # data_retrieval_quality for revisions that loaded no data (#132).
    successful_calls = 0
    failed_calls = 0
    edit_failures = 0
    total_tokens = {"input": 0, "output": 0, "cache_creation_input": 0, "cache_read_input": 0}
    final_answer = ""
    exhausted = False

    try:
        for iteration in range(_MAX_ITERATIONS):
            call_started = time.time()
            response = await agent._call_llm_with_retry(
                client,
                model=model,
                max_tokens=max_tokens,
                system=_SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
                event_log=agent_log,
                # One tool call per turn: a parallel batch of structural edits
                # would carry indices that are stale after the first
                # insert/delete lands. The in-loop guard below is the hard
                # backstop for models that batch anyway.
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            )
            if getattr(response, "_used_model", None):
                model = response._used_model

            usage = getattr(response, "usage", None)
            total_tokens["input"] += getattr(usage, "input_tokens", 0) or 0
            total_tokens["output"] += getattr(usage, "output_tokens", 0) or 0
            total_tokens["cache_creation_input"] += (
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            )
            total_tokens["cache_read_input"] += getattr(usage, "cache_read_input_tokens", 0) or 0
            texts = [
                getattr(b, "text", "")
                for b in response.content
                if getattr(b, "type", None) == "text"
            ]
            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            agent_log.append(
                {
                    "type": "llm_response",
                    "timestamp": _utc_now(),
                    "iteration": iteration + 1,
                    "message_id": getattr(response, "id", None),
                    "model": str(getattr(response, "model", model)),
                    "stop_reason": getattr(response, "stop_reason", None),
                    "texts": texts,
                    "tool_calls": [
                        {"tool": b.name, "input": b.input, "id": b.id} for b in tool_uses
                    ],
                    "tokens": {
                        "input": getattr(usage, "input_tokens", 0),
                        "output": getattr(usage, "output_tokens", 0),
                        "cache_creation_input": getattr(usage, "cache_creation_input_tokens", 0)
                        or 0,
                        "cache_read_input": getattr(usage, "cache_read_input_tokens", 0) or 0,
                    },
                    "duration_ms": int((time.time() - call_started) * 1000),
                }
            )

            if getattr(response, "stop_reason", None) != "tool_use" or not tool_uses:
                final_answer = "\n".join(t for t in texts if t).strip()
                break

            tool_results: list[dict[str, Any]] = []
            # Indices in a batched response all refer to the notebook the
            # model was last shown. After the first insert/delete lands the
            # remaining batched indices are stale, so reject them and make
            # the model re-issue against the updated notebook.
            structural_edit_applied = False
            for block in tool_uses:
                tool_name = block.name
                tool_input = dict(block.input or {})
                tool_started = time.time()

                if tool_name in _EDIT_TOOL_NAMES:
                    if structural_edit_applied:
                        result_text = (
                            "Error: cell indices shifted after an earlier insert/delete "
                            "in this batch — re-issue this edit against the updated "
                            "notebook shown by the previous tool result."
                        )
                    else:
                        result_text = apply_edit(working, tool_name, tool_input)
                    is_error = result_text.startswith("Error")
                    if is_error:
                        edit_failures += 1
                    else:
                        result.edits_applied += 1
                        if tool_name in ("insert_cell", "delete_cell"):
                            structural_edit_applied = True
                    source = "notebook_editor"
                    operation_type = "edit"
                    code = ""
                elif tool_name.startswith("mcp__"):
                    result_text = await agent._execute_mcp_tool(tool_name, tool_input)
                    code = agent._code_for_mcp_tool(tool_name, tool_input, result_text)
                    source = _source_for_tool(tool_name, data_source)
                    operation_type = _operation_type_for_tool(tool_name)
                    is_error = result_text.startswith(("Error", "HTTP ", "SQL error"))
                    if is_error:
                        failed_calls += 1
                    else:
                        successful_calls += 1
                        result.tool_result_texts.append(result_text[:_MAX_TOOL_RESULT_CHARS])
                else:
                    result_text = await agent._execute_tool(tool_name, tool_input, portal_url)
                    code = agent._code_for_tool(tool_name, tool_input, portal_url)
                    source = _source_for_tool(tool_name, data_source)
                    operation_type = _operation_type_for_tool(tool_name)
                    is_error = result_text.startswith(("Error", "HTTP ", "SQL error"))
                    if is_error:
                        failed_calls += 1
                    else:
                        successful_calls += 1
                        result.tool_result_texts.append(result_text[:_MAX_TOOL_RESULT_CHARS])

                result_for_model = result_text[:_MAX_TOOL_RESULT_CHARS]
                if code:
                    # Hand the model the reproducible snippet for the call so
                    # the cell it writes matches what actually ran.
                    result_for_model += "\n\n[Reproducible code for this call]\n" + code

                agent_log.append(
                    {
                        "type": "tool_execution",
                        "timestamp": _utc_now(),
                        "iteration": iteration + 1,
                        "tool": tool_name,
                        "tool_use_id": block.id,
                        "source": source,
                        "operation_type": operation_type,
                        "status": "error" if is_error else "success",
                        "input": tool_input,
                        # Byte-identical to the tool_result content the model
                        # receives (verbatim-capture rule); result_chars /
                        # result_truncated describe the raw tool output.
                        "result": result_for_model,
                        "result_chars": len(result_text),
                        "result_truncated": len(result_text) > _MAX_TOOL_RESULT_CHARS,
                        "duration_ms": int((time.time() - tool_started) * 1000),
                    }
                )
                result.execution_trace.append(
                    {
                        "agent": "notebook_editor",
                        "action": tool_name,
                        "tool_name": tool_name,
                        "arguments": tool_input,
                        "result_preview": result_text[:800],
                        "code": code,
                    }
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_for_model,
                    }
                )

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # The for-loop ran out of iterations while the model was still
            # calling tools — there was no final summarising turn.
            exhausted = True
    except Exception as e:  # noqa: BLE001 - editing must fail soft, caller degrades
        logger.warning("Notebook edit failed", error=str(e), data_source=data_source)
        agent_log.append(
            {
                "type": "error",
                "timestamp": _utc_now(),
                "error": str(e)[:500],
                "exception_type": type(e).__name__,
            }
        )
        result.error = str(e)[:500]
        return result

    result.tool_signals = ToolCallSignals(
        successful_tool_calls=successful_calls,
        failed_tool_calls=failed_calls,
        iterations_used=len([e for e in agent_log if e.get("type") == "llm_response"]),
    )
    agent_log.append(
        {
            "type": "summary",
            "timestamp": _utc_now(),
            "mode": "notebook_edit",
            "total_iterations": result.tool_signals.iterations_used,
            "total_tool_calls": successful_calls + failed_calls,
            "edits_applied": result.edits_applied,
            "edit_failures": edit_failures,
            "total_elapsed_ms": int((time.time() - started) * 1000),
            "model": model,
            "data_source": data_source,
            # Named token_totals (not tokens) per the v2 evidence convention
            # so aggregators and evidence packages don't double-count.
            "token_totals": dict(total_tokens),
        }
    )

    if result.edits_applied == 0:
        # The model declined (or failed) to change anything. Surface its
        # explanation as the answer but return no notebook so the caller
        # does not present an identical copy as an "update".
        result.answer = final_answer or (
            "I couldn't make that change to the notebook. Could you describe "
            "the change differently, or ask a new question?"
        )
        return result

    result.notebook = working
    if final_answer:
        result.answer = final_answer
    elif exhausted:
        # Edits landed but the model never got a summarising turn, so the
        # notebook may be mid-change (e.g. code updated, results cell not
        # yet). Say so honestly — verification will re-check it anyway.
        result.answer = (
            "I made the requested edits, but ran out of room to double-check "
            "every related cell — please skim the updated notebook. It will "
            "also be re-executed and reviewed automatically."
        )
    else:
        result.answer = "I've updated the notebook as requested."
    return result
