"""Tests for data_layer.qsv_profiling."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from data_concierge.data_layer import qsv_profiling
from data_concierge.data_layer.qsv_profiling import (
    run_qsv_count,
    run_qsv_describegpt,
    run_qsv_frequency,
    run_qsv_stats,
)

KEY = "sk-or-v1-0123456789abcdef"

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="fake qsv is a POSIX script")


def _fake_qsv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> Path:
    """Put a Python script named ``qsv`` first on PATH, ahead of any real qsv."""
    fake = tmp_path / "bin" / "qsv"
    fake.parent.mkdir(exist_ok=True)
    fake.write_text(f"#!{sys.executable}\nimport json, os, sys, time\n{body}")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    monkeypatch.delenv("QSV_BIN", raising=False)
    monkeypatch.setattr(qsv_profiling, "_describegpt_checked", set())
    return fake


@posix_only
async def test_describegpt_gets_the_api_key_from_the_environment_not_argv(tmp_path, monkeypatch):
    # describegpt copies its argv into the descriptions it writes, so the key must not be in it.
    record = tmp_path / "record.json"
    _fake_qsv(
        tmp_path,
        monkeypatch,
        f"json.dump({{'argv': sys.argv[1:], 'key': os.environ.get('QSV_LLM_APIKEY')}},"
        f" open({str(record)!r}, 'w'))\n"
        "print(json.dumps({'description': 'ok'}))\n",
    )
    monkeypatch.delenv("QSV_LLM_APIKEY", raising=False)

    out = tmp_path / "qsv_dict.json"
    data = await run_qsv_describegpt(tmp_path / "data.csv", KEY, out)

    assert data == {"description": "ok"}
    # qsv_dict.json is published and hashed: keep the indent=2 re-dump, not qsv's raw bytes
    assert out.read_text() == json.dumps({"description": "ok"}, indent=2)
    seen = json.loads(record.read_text())
    assert seen["key"] == KEY
    assert KEY not in " ".join(seen["argv"])
    assert "--api-key" not in seen["argv"]
    assert seen["argv"][:2] == ["describegpt", str(tmp_path / "data.csv")]


STATS_CSV = "field,type,cardinality\nname,String,3\nage,Integer,2\n"
FREQ_CSV = "field,value,count,percentage\nname,ann,2,50\nname,bob,1,25\nage,30,4,100\n"


@posix_only
async def test_stats_streams_qsv_output_to_disk_unchanged_and_parses_it(tmp_path, monkeypatch):
    _fake_qsv(tmp_path, monkeypatch, f"sys.stdout.write({STATS_CSV!r})\n")
    out = tmp_path / "res" / "qsv_stats.csv"

    stats = await run_qsv_stats(tmp_path / "data.csv", out)

    assert stats == {
        "name": {"type": "String", "cardinality": "3"},
        "age": {"type": "Integer", "cardinality": "2"},
    }
    # mirrored to the Fair Store as-is, so the bytes must be exactly qsv's stdout
    assert out.read_bytes() == STATS_CSV.encode()
    assert not out.with_name(out.name + ".part").exists()


@posix_only
async def test_frequency_parses_counts_as_ints(tmp_path, monkeypatch):
    _fake_qsv(tmp_path, monkeypatch, f"sys.stdout.write({FREQ_CSV!r})\n")
    out = tmp_path / "qsv_frequency.csv"

    freq = await run_qsv_frequency(tmp_path / "data.csv", out)

    assert freq == {
        "name": [{"value": "ann", "count": 2}, {"value": "bob", "count": 1}],
        "age": [{"value": "30", "count": 4}],
    }
    assert out.read_text() == FREQ_CSV


@posix_only
async def test_failed_run_returns_none_and_keeps_the_previous_output(tmp_path, monkeypatch, capsys):
    _fake_qsv(
        tmp_path,
        monkeypatch,
        "sys.stdout.write('field,type\\nhalf')\n"
        "sys.stderr.write('csv error: unequal lengths\\n')\n"
        "sys.exit(1)\n",
    )
    out = tmp_path / "qsv_stats.csv"
    out.write_text("previous good output\n")

    assert await run_qsv_stats(tmp_path / "data.csv", out) is None

    assert out.read_text() == "previous good output\n"
    assert not out.with_name(out.name + ".part").exists()
    assert "qsv stats failed" in capsys.readouterr().out


@posix_only
async def test_hung_run_is_killed_at_the_timeout(tmp_path, monkeypatch, capsys):
    pid_file = tmp_path / "pid"
    _fake_qsv(
        tmp_path,
        monkeypatch,
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n",
    )
    monkeypatch.setenv("VERIKAN_QSV_FREQUENCY_TIMEOUT", "1")
    out = tmp_path / "qsv_frequency.csv"

    started = time.monotonic()
    assert await run_qsv_frequency(tmp_path / "data.csv", out) is None
    assert time.monotonic() - started < 10

    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    assert not out.exists()
    assert not out.with_name(out.name + ".part").exists()
    assert "QsvTimeout, timeout" in capsys.readouterr().out


@posix_only
async def test_count(tmp_path, monkeypatch):
    fake = _fake_qsv(tmp_path, monkeypatch, "print(42)\n")
    assert await run_qsv_count(tmp_path / "data.csv") == 42

    fake.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(1)\n")
    assert await run_qsv_count(tmp_path / "data.csv") is None


async def test_missing_binary_returns_none_instead_of_raising(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QSV_BIN", str(tmp_path / "no-such-qsv"))

    assert await run_qsv_count(tmp_path / "data.csv") is None
    assert await run_qsv_stats(tmp_path / "data.csv", tmp_path / "s.csv") is None
    assert await run_qsv_describegpt(tmp_path / "data.csv", KEY, tmp_path / "d.json") is None
    assert "QsvNotFound" in capsys.readouterr().out


@posix_only
async def test_describegpt_failure_on_a_binary_without_it_says_so(tmp_path, monkeypatch, capsys):
    # qsvlite has no describegpt, and binary discovery falls back to it.
    _fake_qsv(
        tmp_path,
        monkeypatch,
        "arg = sys.argv[1]\n"
        "if arg == '--version':\n"
        "    print('qsvlite 24.0.0-standard-16-16; (aarch64-apple-darwin compiled with Rust 1.99)')\n"
        "elif arg == '--list':\n"
        "    print('Installed commands (2):')\n"
        "    print('    count       Count records')\n"
        "    print('    stats       Infer data types')\n"
        "else:\n"
        "    sys.stderr.write('Invalid arguments.\\n')\n"
        "    sys.exit(1 if arg == '--capabilities' else 2)\n",
    )
    out = tmp_path / "qsv_dict.json"

    assert await run_qsv_describegpt(tmp_path / "data.csv", KEY, out) is None

    printed = capsys.readouterr().out
    assert "has no describegpt" in printed
    assert "qsvlite 24.0.0" in printed
    assert not out.exists()
