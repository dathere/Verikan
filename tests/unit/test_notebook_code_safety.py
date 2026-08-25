"""The generated notebook must not be a code-injection sink (review finding).

The LLM agent writes CKAN tool calls into the .ipynb the user later runs.
Model-supplied `organization` and `rows` used to be interpolated raw, so an
org like `x"}\nimport os...` injected standalone statements. The generated
search cell must always parse as a single safe assignment.
"""

import ast

from data_concierge.agents.llm_agent import LLMAnalysisAgent


def _search_cell(org, rows):
    code = LLMAnalysisAgent._code_for_tool(
        "search_datasets",
        {"query": "housing", "organization": org, "rows": rows},
        "https://example.com",
    )
    # Extract the `params = {...}` line the injection targets.
    line = next(ln for ln in code.splitlines() if ln.startswith("params = "))
    return line


def test_malicious_org_cannot_inject_statements():
    line = _search_cell('x"}\nimport os\nos.system("id")\ny={"a":"b', 10)
    tree = ast.parse(line)
    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.Assign)


def test_non_integer_rows_is_coerced():
    line = _search_cell("cityofpittsburgh", "99999; drop table")
    # rows must be a bare int literal, never the raw string
    assert '"rows": 10' in line
    ast.parse(line)  # still valid


def test_normal_search_is_well_formed():
    line = _search_cell("cityofpittsburgh", 5)
    tree = ast.parse(line)
    assert len(tree.body) == 1
    assert '"rows": 5' in line
