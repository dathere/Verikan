# Agents / LangGraph

## Two graphs, one switch
`agents/supervisor.py` builds both and routes by `state["data_source"]` in `process_query`, via `is_llm_graph_source(data_source)` (same file, ~line 754):

- **LLM-driven graph** — `agents/llm_agent.py`. Claude drives retrieval with tool calling (search / inspect / load rows / run SQL, then answer). Used for CKAN, DCAT and MCP sources.
- **Deterministic graph** — parse entities → route → retrieve (`data_finder.py`) → compute (`stats_computer.py`) → visualise (`viz_builder.py`) → cite (`citation_builder.py`) → notebook. Used for Data Commons.

`is_llm_graph_source` is **not** a static list:
1. `_STATIC_LLM_GRAPH_SOURCES = {"census-data-api", "fbi-crime-data"}`, then
2. admin-added CKAN portals merged dynamically from `gateway.ckan_sites.list_site_ids()` — new portals take effect with no code change or restart, so a grep for source ids will not find them,
3. on a storage failure it falls back to recognising the legacy ids `wprdc` / `ckan`.

Any change to shared state, confidence, or notebook generation has to be reasoned through on **both** graphs. Reviewers here explicitly ask which graph was missed.

## `GraphState` (`agents/state.py`)
`TypedDict`, not a Pydantic model — LangGraph requires it. No defaults, no validators; attribute access, `.model_dump()` and construct-time defaults are all bugs. Read with `.get()`.

Keys, grouped as in the source:
- input — `query`, `session`, `data_source`, `concierge_mode` (legacy, always `"analyze"`)
- classification — `intent`, `tier`
- parsed — `entities`, `normalized_query`, `query_hash`, `parse_confidence`
- retrieval — `retrieved_data`, `retrieval_attempts`
- results — `computed_results`
- generated — `visualization`, `citations`, `notebook`
- evidence — `execution_trace`, `agent_log`, `tool_call_signals` (LLM graph only), `tool_result_texts`
- LLM — `messages` (`Annotated[list, add_messages]`)

`messages` is the **only** `Annotated`/reducer-backed key, and `GraphState` is the only state schema (both `StateGraph(...)` calls in `supervisor.py` use it). So `messages` is the one key that accumulates automatically; every other list/dict key (`execution_trace`, `agent_log`, `citations`, `tool_result_texts`, `computed_results`) is plain last-write-wins — a node that returns a partial list **replaces** it. Append to the existing value and return the whole list, or the earlier entries are lost with no error.
- output — `answer`, `confidence`
- quick-answer path — `quick_answer_mode`, `quick_answer`, `source_links` (skips notebook generation for simple factual lookups; a change to the notebook pipeline may not apply on this path)
- control flow — `current_agent`, `should_escalate`, `needs_clarification`, `clarification_question`, `error`

## The two evidence lists are not interchangeable
- `execution_trace` → becomes the notebook. `notebook_generator.py` turns it into runnable cells; an action that retrieves/computes/visualises/cites and does not append its entry **with its code snippet** yields a silently incomplete, irreproducible notebook.
- `agent_log` → the evidence record behind a published answer. Versioned by `AGENT_LOG_FORMAT_VERSION` (`llm_agent.py:114`, currently `2`), stamped as `log_format_version` by both `llm_agent.py` and `notebook_editor.py`. Changing its shape without bumping the constant breaks downstream evidence consumers. Entries must keep tool output verbatim and keep provenance — paraphrasing, unmarked truncation, or dropping the source reference defeats the point.

## Codegen
Everything interpolated into a generated cell (tool arguments, retrieved portal text) goes through `repr()` or equivalent. One unparseable cell fails verification for the whole notebook. A figure quoted in the answer must be *computed by a cell*, never asserted in markdown — that is the exact defect the adversarial review exists to catch.

## Verification pass
`notebook_verifier.py` executes, `notebook_reviewer.py` adversarially reviews the method, `notebook_editor.py` applies chat follow-up revisions cell-by-cell. Generated notebooks are untrusted (LLM-written, portal text embedded): execution stays in the contained subprocess — env allowlist withholding every credential, hard timeout, shell-escape cells skipped, egress guard against loopback/private/link-local/cloud-metadata. Never widen that boundary or execute generated code in-process.

Scoring lives in `core/confidence.py`: `ComponentScore.unavailable(reason)` vs a measured score is the central distinction; an uncomputable factor is `None` with a reason in the `unavailable` map, never `0.0`, and branches on `final_score` must respect `measured_weight`.
