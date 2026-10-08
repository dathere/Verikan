"""Tests for data_layer.qsv_profiling."""

from __future__ import annotations

import json
import os
import sys

import pytest

from data_concierge.data_layer.qsv_profiling import run_qsv_describegpt

KEY = "sk-or-v1-0123456789abcdef"


@pytest.mark.skipif(sys.platform == "win32", reason="fake qsv is a POSIX shell script")
async def test_describegpt_gets_the_api_key_from_the_environment_not_argv(tmp_path, monkeypatch):
    # describegpt copies its argv into the descriptions it writes, so the key must not be in it.
    record = tmp_path / "record.json"
    fake = tmp_path / "bin" / "qsv"
    fake.parent.mkdir()
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"json.dump({{'argv': sys.argv[1:], 'key': os.environ.get('QSV_LLM_APIKEY')}},"
        f" open({str(record)!r}, 'w'))\n"
        "print(json.dumps({'description': 'ok'}))\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    monkeypatch.delenv("QSV_LLM_APIKEY", raising=False)

    out = tmp_path / "qsv_dict.json"
    data = await run_qsv_describegpt(tmp_path / "data.csv", KEY, out)

    assert data == {"description": "ok"}
    assert json.loads(out.read_text()) == {"description": "ok"}
    seen = json.loads(record.read_text())
    assert seen["key"] == KEY
    assert KEY not in " ".join(seen["argv"])
    assert "--api-key" not in seen["argv"]
    assert seen["argv"][:2] == ["describegpt", str(tmp_path / "data.csv")]
