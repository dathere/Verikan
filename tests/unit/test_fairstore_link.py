"""Tests for Verikan's link to the Fair Store (admin settings, mirror runs, chat).

Covers ``gateway/fairstore.py`` (settings, token/host binding, the chat-source
registry entry), mirror runs through ``gateway/onboarding_jobs.py``, the admin
routes, and how the agent routes a mirrored resource's rows to its origin.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import HTTPException

from data_concierge.data_layer.storage import storage
from data_concierge.gateway import ckan_sites, fairstore
from data_concierge.gateway import onboarding_jobs as oj

SECRET = "eyJhbGciOiJIUzI1NiJ9.test-token-value-7f3a"


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private storage root, and no FAIRSTORE_* environment seeds."""
    monkeypatch.setattr(storage, "root", tmp_path)
    monkeypatch.setattr(fairstore.app_settings, "fairstore_url", "")
    monkeypatch.setattr(
        fairstore.app_settings, "fairstore_api_key", fairstore.app_settings.ckan_api_key.__class__("")
    )
    monkeypatch.setattr(oj, "_jobs", {})
    monkeypatch.setattr(oj, "_logs", {})
    monkeypatch.setattr(oj, "_processes", {})
    return tmp_path


def _secret(value: str) -> Any:
    return fairstore.app_settings.ckan_api_key.__class__(value)


class TestUrlValidation:
    @pytest.mark.parametrize(
        "url",
        ["https://fairstore.example.org", "https://1-2-3-4.sslip.io/", "http://localhost:5001"],
    )
    def test_accepts_https_and_local_http(self, url: str) -> None:
        assert fairstore.normalize_url(url) == url.rstrip("/")

    @pytest.mark.parametrize(
        ("url", "message"),
        [
            ("http://fairstore.example.org", "https"),
            ("https://admin:pw@fairstore.example.org", "token field"),
            ("https://fairstore.example.org/?x=1", "query"),
            ("ftp://fairstore.example.org", "https://"),
            ("fairstore.example.org", "https://"),
        ],
    )
    def test_rejects_unsafe_urls(self, url: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            fairstore.normalize_url(url)

    def test_blank_means_unset(self) -> None:
        assert fairstore.normalize_url("  ") == ""


class TestSettings:
    def test_defaults_are_unconfigured(self, store: Path) -> None:
        current = fairstore.load_settings()
        assert current["url"] == "" and current["token"] == ""
        assert current["chat_source"] is False and current["enrich"] is True

    def test_token_is_never_public(self, store: Path) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "token": SECRET})
        public = fairstore.public_settings()
        assert public["token_set"] is True and public["token_source"] == "admin"
        assert SECRET not in json.dumps(public)
        assert "token" not in public

    def test_blank_token_keeps_the_saved_one(self, store: Path) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "token": SECRET})
        fairstore.save_settings({"url": "https://fs.example.org", "token": ""})
        assert fairstore.load_settings()["token"] == SECRET

    def test_clear_token_removes_it(self, store: Path) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "token": SECRET})
        fairstore.save_settings({"clear_token": True})
        assert fairstore.load_settings()["token"] == ""

    def test_moving_to_another_host_drops_the_token(self, store: Path) -> None:
        """A URL edit must never send the sysadmin token to a different server."""
        fairstore.save_settings({"url": "https://fs.example.org", "token": SECRET})
        fairstore.save_settings({"url": "https://attacker.example.net"})
        current = fairstore.load_settings()
        assert current["url"] == "https://attacker.example.net"
        assert current["token"] == ""

    def test_same_host_path_change_keeps_the_token(self, store: Path) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "token": SECRET})
        fairstore.save_settings({"url": "https://fs.example.org/catalog"})
        assert fairstore.load_settings()["token"] == SECRET

    def test_environment_token_only_for_the_environment_host(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fairstore.app_settings, "fairstore_url", "https://fs.example.org")
        monkeypatch.setattr(fairstore.app_settings, "fairstore_api_key", _secret(SECRET))
        current = fairstore.load_settings()
        assert current["token"] == SECRET and current["token_source"] == "environment"
        assert current["url_source"] == "environment"

        fairstore.save_settings({"url": "https://elsewhere.example.net"})
        assert fairstore.load_settings()["token"] == ""

    def test_token_needs_a_url(self, store: Path) -> None:
        with pytest.raises(ValueError, match="URL"):
            fairstore.save_settings({"token": SECRET})

    def test_token_with_whitespace_is_rejected(self, store: Path) -> None:
        with pytest.raises(ValueError, match="token"):
            fairstore.save_settings({"url": "https://fs.example.org", "token": "a b"})

    def test_mirror_sites_must_be_registered(self, store: Path) -> None:
        with pytest.raises(ValueError, match="Unknown portal"):
            fairstore.save_settings({"mirror_sites": ["wprdc", "nope"]})
        saved = fairstore.save_settings({"mirror_sites": ["wprdc", "wprdc"]})
        assert saved["mirror_sites"] == ["wprdc"]


class TestChatSource:
    def test_enabling_registers_a_managed_ckan_portal(self, store: Path) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})
        site = ckan_sites.get_site(fairstore.PORTAL_ID)
        assert site is not None
        assert site["url"] == "https://fs.example.org"
        assert site["portal_type"] == "ckan"
        assert site["managed_by"] == "fairstore"

    def test_settings_changes_follow_through(self, store: Path) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})
        fairstore.save_settings({"portal_name": "All PA data", "quality_score": 0.9})
        site = ckan_sites.get_site(fairstore.PORTAL_ID)
        assert site["name"] == "All PA data" and site["quality_score"] == 0.9

    def test_disabling_removes_it(self, store: Path) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})
        fairstore.save_settings({"chat_source": False})
        assert ckan_sites.get_site(fairstore.PORTAL_ID) is None

    def test_a_hand_registered_portal_is_left_alone(self, store: Path) -> None:
        ckan_sites.add_site(url="https://other.example.org", name="Mine", site_id="fairstore")
        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})
        fairstore.save_settings({"chat_source": False})
        site = ckan_sites.get_site("fairstore")
        assert site is not None and site["url"] == "https://other.example.org"

    def test_public_info_has_no_secrets(self, store: Path) -> None:
        fairstore.save_settings(
            {"url": "https://fs.example.org", "token": SECRET, "chat_source": True}
        )
        info = fairstore.public_info()
        assert info == {
            "configured": True,
            "url": "https://fs.example.org",
            "name": fairstore.DEFAULT_PORTAL_NAME,
            "chat_source": True,
            "portal_id": "fairstore",
        }


class TestSiteConfig:
    async def test_only_known_options_are_accepted(self, store: Path) -> None:
        with pytest.raises(ValueError, match="ckan.site_url"):
            await fairstore.update_site_config({"ckan.site_url": "https://evil.example"})

    async def test_updates_go_to_config_option_update(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, dict[str, Any] | None, bool]] = []

        async def fake_call(current: dict, action: str, params=None, *, auth=False):
            calls.append((action, params, auth))
            if action == "config_option_list":
                return list(fairstore.SITE_OPTIONS) + ["ckan.site_url", "ckan.theme"]
            if action == "config_option_show":
                return "value"
            return {}

        monkeypatch.setattr(fairstore, "_call", fake_call)
        config = await fairstore.update_site_config(
            {"ckan.site_title": "Verikan Fair Store"},
            {"url": "https://fs.example.org", "token": SECRET},
        )
        assert ("config_option_update", {"ckan.site_title": "Verikan Fair Store"}, True) in calls
        keys = [option["key"] for option in config["options"]]
        assert "ckan.site_url" not in keys and "ckan.theme" not in keys
        assert keys == list(fairstore.SITE_OPTIONS)


class TestStatus:
    async def test_unconfigured(self, store: Path) -> None:
        report = await fairstore.status()
        assert report["configured"] is False and report["reachable"] is False

    async def test_unreachable_is_reported_not_raised(self, store: Path) -> None:
        report = await fairstore.status({"url": "http://127.0.0.1:9", "token": ""})
        assert report["reachable"] is False and report["error"]

    async def test_counts_per_source_portal(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_call(current: dict, action: str, params=None, *, auth=False):
            if action == "status_show":
                return {"ckan_version": "2.11.6", "extensions": ["hierarchy_display"]}
            if action == "config_option_list":
                return []
            if action == "package_search":
                return {"count": {None: 921, 'mirror_source_portal:"wprdc"': 371}.get(
                    params.get("fq"), 0
                )}
            return ["a", "b"]

        monkeypatch.setattr(fairstore, "_call", fake_call)
        report = await fairstore.status({"url": "https://fs.example.org", "token": SECRET})
        assert report["token"] == "sysadmin"
        assert report["datasets"] == 921 and report["organizations"] == 2
        wprdc = next(p for p in report["by_portal"] if p["site_id"] == "wprdc")
        assert wprdc["datasets"] == 371


class TestMirrorCommand:
    def _build(self, **options: Any) -> list[str]:
        argv, _sites = oj.build_mirror_command(
            options, target_url="https://fs.example.org", summary_path="/tmp/summary.json"
        )
        return argv

    def test_default_is_a_dry_run_of_everything(self, store: Path) -> None:
        argv = self._build()
        assert argv[1].endswith("populate_fairstore.py")
        assert argv[argv.index("--site") + 1] == "all"
        assert argv[argv.index("--target-url") + 1] == "https://fs.example.org"
        assert "--apply" not in argv and "--no-enrich" not in argv

    def test_flags(self, store: Path) -> None:
        argv = self._build(site="wprdc", apply=True, enrich=False, limit=5)
        assert argv[argv.index("--site") + 1] == "wprdc"
        assert "--apply" in argv and "--no-enrich" in argv
        assert argv[argv.index("--limit") + 1] == "5"

    def test_a_portal_list(self, store: Path) -> None:
        argv = self._build(sites=["wprdc", "data-pa-gov"])
        assert argv[argv.index("--site") + 1] == "wprdc,data-pa-gov"

    @pytest.mark.parametrize("site", ["nope", "fairstore", "wprdc;rm -rf /"])
    def test_unknown_or_self_portals_are_refused(self, store: Path, site: str) -> None:
        with pytest.raises(oj.JobError):
            self._build(site=site)

    def test_limit_is_range_checked(self, store: Path) -> None:
        with pytest.raises(oj.JobError, match="between"):
            self._build(limit=100000)

    def test_target_url_is_validated(self, store: Path) -> None:
        with pytest.raises(oj.JobError, match="https"):
            oj.build_mirror_command({}, target_url="http://fs.example.org", summary_path="x")
        with pytest.raises(oj.JobError, match="URL"):
            oj.build_mirror_command({}, target_url="", summary_path="x")


class _FakeProcess:
    def __init__(self, lines: list[bytes], exit_code: int = 0) -> None:
        self.stdout = asyncio.StreamReader()
        for line in lines:
            self.stdout.feed_data(line)
        self.stdout.feed_eof()
        self._exit_code = exit_code

    async def wait(self) -> int:
        return self._exit_code

    def terminate(self) -> None:  # pragma: no cover - not exercised
        pass


class TestMirrorJob:
    async def test_writing_needs_a_token(self, store: Path) -> None:
        with pytest.raises(oj.JobError, match="token"):
            await oj.start_mirror_job(
                {"apply": True}, target_url="https://fs.example.org", token=""
            )

    async def test_token_travels_in_the_environment_and_the_summary_is_kept(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        async def fake_exec(*argv: str, env: dict, **_: Any) -> _FakeProcess:
            seen["argv"], seen["env"] = list(argv), env
            summary = argv[argv.index("--summary-file") + 1]
            Path(summary).write_text(json.dumps([{"site": "wprdc", "created": 3}]))
            return _FakeProcess([b"mirroring wprdc\n"])

        monkeypatch.setattr(oj.asyncio, "create_subprocess_exec", fake_exec)
        job = await oj.start_mirror_job(
            {"site": "wprdc", "apply": True},
            target_url="https://fs.example.org",
            token=SECRET,
            started_by="pytest",
        )
        await oj._jobs[job["id"]]["_task"]

        assert SECRET not in " ".join(seen["argv"])
        assert seen["env"]["FAIRSTORE_API_KEY"] == SECRET
        assert seen["env"]["FAIRSTORE_URL"] == "https://fs.example.org"
        assert SECRET not in json.dumps(job)

        record = oj.get_job(job["id"])
        assert record["kind"] == oj.KIND_FAIRSTORE_MIRROR
        assert record["status"] == oj.STATUS_SUCCEEDED
        assert record["result"] == [{"site": "wprdc", "created": 3}]
        assert record["log"] == ["mirroring wprdc"]
        stored = storage.read_json(f"onboarding_jobs/{job['id']}.json")
        assert stored["result"] == record["result"] and SECRET not in json.dumps(stored)
        summary_path = seen["argv"][seen["argv"].index("--summary-file") + 1]
        assert not Path(summary_path).exists()

    async def test_a_dry_run_without_a_token_drops_an_inherited_one(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAIRSTORE_API_KEY", "inherited")
        seen: dict[str, Any] = {}

        async def fake_exec(*argv: str, env: dict, **_: Any) -> _FakeProcess:
            seen["env"] = env
            return _FakeProcess([])

        monkeypatch.setattr(oj.asyncio, "create_subprocess_exec", fake_exec)
        job = await oj.start_mirror_job({}, target_url="https://fs.example.org", token="")
        await oj._jobs[job["id"]]["_task"]
        assert "FAIRSTORE_API_KEY" not in seen["env"]

    async def test_mirror_and_onboarding_share_one_slot(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(oj._jobs, "busy", {"id": "busy", "status": oj.STATUS_RUNNING})
        with pytest.raises(oj.JobError, match="busy"):
            await oj.start_mirror_job({}, target_url="https://fs.example.org", token="")

    def test_lists_filter_by_kind_and_old_records_are_onboarding(self, store: Path) -> None:
        storage.write_json("onboarding_jobs/old.json", {"id": "old", "status": "succeeded"})
        storage.write_json(
            "onboarding_jobs/new.json",
            {"id": "new", "status": "succeeded", "kind": oj.KIND_FAIRSTORE_MIRROR},
        )
        storage.write_json("onboarding_jobs/index.json", {"job_ids": ["new", "old"]})
        assert [j["id"] for j in oj.list_jobs(kind=oj.KIND_ONBOARDING)] == ["old"]
        assert [j["id"] for j in oj.list_jobs(kind=oj.KIND_FAIRSTORE_MIRROR)] == ["new"]


class TestRoutes:
    ADMIN = {"user": "admin"}

    async def test_get_never_returns_the_token(self, store: Path) -> None:
        from data_concierge.gateway.router import (
            FairStoreSettingsRequest,
            get_fairstore_settings,
            update_fairstore_settings,
        )

        await update_fairstore_settings(
            FairStoreSettingsRequest(url="https://fs.example.org", token=SECRET),
            admin_user=self.ADMIN,
        )
        body = await get_fairstore_settings(_admin=self.ADMIN)
        assert body["settings"]["token_set"] is True
        assert SECRET not in json.dumps(body)

    async def test_invalid_settings_are_a_400(self, store: Path) -> None:
        from data_concierge.gateway.router import (
            FairStoreSettingsRequest,
            update_fairstore_settings,
        )

        with pytest.raises(HTTPException) as exc:
            await update_fairstore_settings(
                FairStoreSettingsRequest(url="http://fs.example.org"), admin_user=self.ADMIN
            )
        assert exc.value.status_code == 400

    async def test_the_managed_portal_cannot_be_edited_elsewhere(self, store: Path) -> None:
        from data_concierge.gateway.router import (
            StartOnboardingJobRequest,
            UpdateCkanSiteRequest,
            delete_ckan_site,
            start_onboarding_job,
            update_ckan_site,
        )

        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})
        with pytest.raises(HTTPException) as exc:
            await delete_ckan_site("fairstore", _admin=self.ADMIN)
        assert exc.value.status_code == 409
        with pytest.raises(HTTPException) as exc:
            await update_ckan_site("fairstore", UpdateCkanSiteRequest(name="x"), _admin=self.ADMIN)
        assert exc.value.status_code == 409
        with pytest.raises(HTTPException) as exc:
            await start_onboarding_job(
                StartOnboardingJobRequest(site_id="fairstore"), admin_user=self.ADMIN
            )
        assert exc.value.status_code == 409

    async def test_all_means_the_chosen_portals(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway.router import (
            StartMirrorJobRequest,
            start_fairstore_mirror_job,
        )

        fairstore.save_settings(
            {"url": "https://fs.example.org", "mirror_sites": ["wprdc"], "enrich": False}
        )
        seen: dict[str, Any] = {}

        async def fake_start(options: dict, **kwargs: Any) -> dict:
            seen.update(options=options, **kwargs)
            return {"id": "x", "site_name": "wprdc"}

        monkeypatch.setattr(oj, "start_mirror_job", fake_start)
        await start_fairstore_mirror_job(StartMirrorJobRequest(), admin_user=self.ADMIN)
        assert seen["options"]["sites"] == ["wprdc"]
        assert seen["options"]["enrich"] is False
        assert seen["target_url"] == "https://fs.example.org"


class TestAgentRouting:
    """A Fair Store resource is a link; its rows live at the portal it mirrors."""

    def _agent(self) -> Any:
        from data_concierge.agents.llm_agent import LLMAnalysisAgent

        return LLMAnalysisAgent.__new__(LLMAnalysisAgent)

    def test_ckan_origin_uses_the_same_resource_id(self, store: Path) -> None:
        agent = self._agent()
        origin = agent._mirror_origin("wprdc")
        route = agent._origin_route(
            {
                "id": "044f2016",
                "datastore_active": False,
                "mirror_source_portal": "wprdc",
                "mirror_source_id": "044f2016",
                "mirror_source_datastore_active": True,
            },
            origin,
        )
        assert "portal_id=`wprdc`" in route and "resource_id=`044f2016`" in route

    def test_dcat_origin_loads_the_file(self, store: Path) -> None:
        agent = self._agent()
        route = agent._origin_route(
            {
                "id": "717971b3",
                "url": "https://data.pa.gov/api/v3/views/ytxr-yigf/export.csv",
                "format": "CSV",
                "mirror_source_portal": "data-pa-gov",
            },
            agent._mirror_origin("data-pa-gov"),
        )
        assert "portal_id=`data-pa-gov`" in route
        assert "resource_id=`https://data.pa.gov/api/v3/views/ytxr-yigf/export.csv`" in route

    @pytest.mark.parametrize(
        ("fmt", "mimetype"),
        [("JSON", "application/json"), ("XML", "application/xml"), ("PDF", ""), ("", "")],
    )
    def test_non_tabular_dcat_files_have_no_rows(
        self, store: Path, fmt: str, mimetype: str
    ) -> None:
        """Parsed as CSV they load garbage that scored as a successful load."""
        agent = self._agent()
        resource = {
            "id": "1d40596d",
            "url": "https://data.pa.gov/api/v3/views/ytxr-yigf/query.json",
            "format": fmt,
            "mimetype": mimetype,
            "mirror_source_portal": "data-pa-gov",
        }
        assert agent._origin_route(resource, agent._mirror_origin("data-pa-gov")) == ""

    def test_a_tabular_mimetype_counts(self, store: Path) -> None:
        agent = self._agent()
        resource = {
            "url": "https://x.example/rows",
            "mimetype": "text/csv; charset=utf-8",
            "mirror_source_portal": "data-pa-gov",
        }
        assert agent._origin_route(resource, agent._mirror_origin("data-pa-gov"))

    def test_no_route_for_local_tables_or_unregistered_origins(self, store: Path) -> None:
        agent = self._agent()
        assert agent._origin_route({"datastore_active": True}, agent._mirror_origin("wprdc")) == ""
        assert agent._mirror_origin("not-registered") is None
        assert agent._origin_route({"mirror_source_portal": "x"}, None) == ""
        # A CKAN resource with no DataStore at the source has no rows to load.
        assert (
            agent._origin_route(
                {"mirror_source_portal": "wprdc", "mirror_source_datastore_active": False},
                agent._mirror_origin("wprdc"),
            )
            == ""
        )

    async def test_a_datastore_miss_points_at_the_origin(self, store: Path) -> None:
        agent = self._agent()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/datastore_search"):
                return httpx.Response(404, json={"success": False, "error": {"message": "nf"}})
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "result": {
                        "id": "044f2016",
                        "mirror_source_portal": "wprdc",
                        "mirror_source_id": "044f2016",
                        "mirror_source_datastore_active": True,
                    },
                },
            )

        async with httpx.AsyncClient(
            base_url="https://fs.example.org", transport=httpx.MockTransport(handler)
        ) as client:
            text = await agent._tool_load(client, {"resource_id": "044f2016"})
        # Not TOOL_UNAVAILABLE_PREFIX: that withdraws load_resource_data for the
        # rest of the run, exactly when the model needs it for the redirect.
        assert text.startswith(agent.TOOL_REDIRECT_PREFIX)
        assert not text.startswith(agent.TOOL_UNAVAILABLE_PREFIX)
        assert "portal_id=`wprdc`" in text

    def test_notebook_code_targets_the_overridden_portal(self, store: Path) -> None:
        agent = self._agent()
        wprdc = ckan_sites.get_site("wprdc")["url"]
        assert agent._tool_portal_url({"portal_id": "wprdc"}, "https://fs.example.org") == wprdc
        assert agent._tool_portal_url({}, "https://fs.example.org") == "https://fs.example.org"


class TestStagingFromStorage:
    """A mirror started on Cloud Run reads qsv output from GCS, not local disk."""

    def _populate(self) -> Any:
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from scripts import populate_fairstore

        return populate_fairstore

    def test_sync_records_headers_and_staging_restores_the_profile(
        self, store: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        from data_concierge.data_layer.qsv_profiling import sync_to_storage

        pf = self._populate()
        onboard = tmp_path_factory.mktemp("work") / "data" / "ckan_onboard"
        dataset = onboard / "wprdc" / "parks"
        dataset.mkdir(parents=True)
        (dataset / "parks.csv").write_text("﻿name,acres\nFrick,644\n")
        (dataset / "qsv_stats.csv").write_text("field,type\nname,String\nacres,Integer\n")
        (dataset / "qsv_frequency.csv").write_text("field,value,count\nname,Frick,1\n")
        (dataset / "qsv_dict.json").write_text(json.dumps({"Dictionary": {}}))
        index = {
            "datasets": [
                {
                    "resources": [
                        {
                            "resource_id": "r1",
                            "status": "ok",
                            "local_path": "data/ckan_onboard/wprdc/parks/parks.csv",
                        }
                    ]
                }
            ]
        }
        (onboard / "wprdc" / "index.json").write_text(json.dumps(index))

        sync_to_storage(onboard / "wprdc", "wprdc")
        assert storage.read_json("ckan_onboard/wprdc/sync_manifest.json") == {
            "version": 1,
            "headers": {"ckan_onboard/wprdc/parks/parks.csv": ["name", "acres"]},
            "dirs": {
                "ckan_onboard/wprdc/parks": ["qsv_dict.json", "qsv_frequency.csv", "qsv_stats.csv"]
            },
        }

        stage = tmp_path_factory.mktemp("stage")
        staged = pf._load_index("wprdc", source="storage", stage_root=stage)
        resource = staged["datasets"][0]["resources"][0]
        assert Path(resource["local_path"]).parent == stage / "ckan_onboard" / "wprdc" / "parks"
        assert resource["profiled_header"] == ["name", "acres"]
        # The CSV itself is not staged; the recorded header stands in for it.
        assert not Path(resource["local_path"]).exists()
        assert pf._profiled_header(resource) == ["name", "acres"]
        vetted = pf._vet_profile(resource)
        assert vetted["_stats_ok"] is True and vetted["_frequency_ok"] is True

    def test_mismatched_stats_are_still_caught_from_storage(
        self, store: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        pf = self._populate()
        storage.write_json(
            "ckan_onboard/wprdc/index.json",
            {
                "datasets": [
                    {
                        "resources": [
                            {"resource_id": "r1", "status": "ok", "local_path": "d/wprdc/x/a.csv"}
                        ]
                    }
                ]
            },
        )
        storage.write_json(
            "ckan_onboard/wprdc/sync_manifest.json",
            {
                "headers": {"ckan_onboard/wprdc/x/a.csv": ["a"]},
                "dirs": {"ckan_onboard/wprdc/x": ["qsv_stats.csv"]},
            },
        )
        storage.write_bytes("ckan_onboard/wprdc/x/qsv_stats.csv", b"field,type\nzzz,String\n")
        staged = pf._load_index(
            "wprdc", source="storage", stage_root=tmp_path_factory.mktemp("stage")
        )
        vetted = pf._vet_profile(staged["datasets"][0]["resources"][0])
        assert vetted["_stats_ok"] is False

    def test_local_source_never_reads_storage(self, store: Path) -> None:
        pf = self._populate()
        storage.write_json("dcat_onboard/nowhere/index.json", {"datasets": [{"resources": []}]})
        storage.write_json("dcat_onboard/nowhere/sync_manifest.json", {"headers": {}, "dirs": {}})
        assert pf._load_index("nowhere", source="local") == {}
        assert pf._load_index("nowhere", source="auto") == {"datasets": [{"resources": []}]}

    def _index_with(self, *dirs: str) -> None:
        storage.write_json(
            "ckan_onboard/wprdc/index.json",
            {
                "datasets": [
                    {
                        "resources": [
                            {"resource_id": f"r-{d}", "status": "ok", "local_path": f"d/wprdc/{d}/a.csv"}
                            for d in dirs
                        ]
                    }
                ]
            },
        )

    def test_an_unlisted_directory_is_not_created(
        self, store: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Locally the directory is absent, so its qsv resources are not removed;
        staging must not create it (which made the storage run remove them)."""
        pf = self._populate()
        self._index_with("gone", "here")
        storage.write_json(
            "ckan_onboard/wprdc/sync_manifest.json",
            {"headers": {}, "dirs": {"ckan_onboard/wprdc/here": []}},
        )
        stage = tmp_path_factory.mktemp("stage")
        resources = pf._load_index("wprdc", source="storage", stage_root=stage)["datasets"][0][
            "resources"
        ]
        by_id = {r["resource_id"]: r for r in resources}
        assert pf._qsv_dir(by_id["r-gone"]) is None
        assert pf._qsv_dir(by_id["r-here"]) == stage / "ckan_onboard" / "wprdc" / "here"

    def test_an_output_deleted_since_is_not_staged(
        self, store: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """sync_to_storage never deletes; the manifest says what exists now."""
        pf = self._populate()
        self._index_with("ds")
        storage.write_bytes("ckan_onboard/wprdc/ds/qsv_dict.json", b'{"stale": true}')
        storage.write_bytes("ckan_onboard/wprdc/ds/qsv_stats.csv", b"field\na\n")
        storage.write_json(
            "ckan_onboard/wprdc/sync_manifest.json",
            {"headers": {}, "dirs": {"ckan_onboard/wprdc/ds": ["qsv_stats.csv"]}},
        )
        stage = tmp_path_factory.mktemp("stage")
        pf._load_index("wprdc", source="storage", stage_root=stage)
        staged = stage / "ckan_onboard" / "wprdc" / "ds"
        assert (staged / "qsv_stats.csv").is_file()
        assert not (staged / "qsv_dict.json").exists()

    def test_summary_file(self, tmp_path: Path) -> None:
        pf = self._populate()
        path = tmp_path / "summary.json"
        pf._write_summary(str(path), [{"site": "wprdc"}])
        assert json.loads(path.read_text()) == [{"site": "wprdc"}]
        pf._write_summary(None, {"ignored": True})


class TestToolArgumentsAndSqlBreaker:
    async def test_string_limits_do_not_crash_a_load(self, store: Path) -> None:
        """The model sent limit='100' and every load died on min(str, int)."""
        from data_concierge.agents.llm_agent import LLMAnalysisAgent

        agent = LLMAnalysisAgent.__new__(LLMAnalysisAgent)
        sent: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            sent.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"success": True, "result": {"records": [], "total": 0, "fields": []}},
            )

        async with httpx.AsyncClient(
            base_url="https://p.example.org", transport=httpx.MockTransport(handler)
        ) as client:
            text = await agent._tool_load(
                client, {"resource_id": "r", "limit": "100", "offset": "20"}
            )
        assert "Total records: 0" in text
        assert sent["limit"] == 100 and sent["offset"] == 20

    def test_a_disabled_sql_action_counts_as_a_dead_endpoint(self) -> None:
        from data_concierge.agents.llm_agent import _sql_action_missing

        disabled = httpx.Response(
            400, text='"Bad request - Action name not known: datastore_search_sql"'
        )
        bad_statement = httpx.Response(400, json={"error": {"message": "syntax error"}})
        assert _sql_action_missing(disabled) is True
        assert _sql_action_missing(bad_statement) is False

    def test_notebook_code_never_embeds_a_raw_limit(self) -> None:
        from data_concierge.agents.llm_agent import LLMAnalysisAgent

        code = LLMAnalysisAgent._code_for_tool(
            "load_resource_data",
            {"resource_id": "r", "limit": "5}; import os; os.system('x') #"},
            "https://p.example.org",
        )
        assert "os.system" not in code and '"limit": 100' in code
        code = LLMAnalysisAgent._code_for_tool(
            "load_resource_data", {"resource_id": "r", "limit": "250"}, "https://p.example.org"
        )
        assert '"limit": 250' in code


class TestLoadArgumentRepair:
    def test_fields_sent_as_a_string_become_a_list(self) -> None:
        from data_concierge.agents.llm_agent import _normalize_load_args

        args: dict[str, Any] = {"fields": '["NAME", "ACRES"]', "sort": "Date Last desc"}
        _normalize_load_args("load_resource_data", args)
        # CKAN parses the sort itself (quotes optional); it is left alone.
        assert args == {"fields": ["NAME", "ACRES"], "sort": "Date Last desc"}
        args = {"fields": "NAME, ACRES"}
        _normalize_load_args("load_resource_data", args)
        assert args["fields"] == ["NAME", "ACRES"]
        other: dict[str, Any] = {"fields": "x"}
        _normalize_load_args("search_datasets", other)
        assert other == {"fields": "x"}


class TestRouteToOrigin:
    def _agent(self, resource: dict[str, Any] | None) -> Any:
        from data_concierge.agents.llm_agent import LLMAnalysisAgent
        from data_concierge.core.logging import get_logger

        agent = LLMAnalysisAgent.__new__(LLMAnalysisAgent)
        agent.logger = get_logger("test")

        def handler(request: httpx.Request) -> httpx.Response:
            if resource is None:
                return httpx.Response(404, json={"success": False})
            return httpx.Response(200, json={"success": True, "result": resource})

        client = httpx.AsyncClient(
            base_url="https://fs.example.org", transport=httpx.MockTransport(handler)
        )

        async def get_client(url: str) -> httpx.AsyncClient:
            return client

        agent._get_http_client = get_client
        return agent

    async def test_a_mirrored_ckan_resource_loads_from_its_origin(self, store: Path) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})
        agent = self._agent(
            {
                "id": "r1",
                "mirror_source_portal": "wprdc",
                "mirror_source_id": "r1",
                "mirror_source_datastore_active": True,
            }
        )
        args: dict[str, Any] = {"resource_id": "r1", "limit": 5}
        note = await agent._route_to_origin("load_resource_data", args, "https://fs.example.org")
        assert args == {"resource_id": "r1", "limit": 5, "portal_id": "wprdc"}
        assert "portal_id `wprdc`" in note
        # The notebook follows the same rewrite.
        wprdc = ckan_sites.get_site("wprdc")["url"]
        assert agent._tool_portal_url(args, "https://fs.example.org") == wprdc

    async def test_a_mirrored_dcat_resource_loads_its_file(self, store: Path) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})
        url = "https://data.pa.gov/api/v3/views/ytxr-yigf/export.csv"
        agent = self._agent(
            {"id": "u1", "url": url, "format": "CSV", "mirror_source_portal": "data-pa-gov"}
        )
        args: dict[str, Any] = {"resource_id": "u1"}
        assert await agent._route_to_origin("load_resource_data", args, "https://fs.example.org")
        assert args == {"resource_id": url, "portal_id": "data-pa-gov"}

    @pytest.mark.parametrize("body", [None, [], "maintenance", {"result": "x"}])
    async def test_an_odd_resource_show_body_never_aborts_the_run(
        self, store: Path, body: Any
    ) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})
        agent = self._agent(None)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body)

        client = httpx.AsyncClient(
            base_url="https://fs.example.org", transport=httpx.MockTransport(handler)
        )

        async def get_client(url: str) -> httpx.AsyncClient:
            return client

        agent._get_http_client = get_client
        args: dict[str, Any] = {"resource_id": "r1"}
        assert await agent._route_to_origin("load_resource_data", args, "https://fs.example.org") == ""
        assert args == {"resource_id": "r1"}

    async def test_other_portals_and_local_tables_are_untouched(self, store: Path) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})
        agent = self._agent({"id": "q1", "datastore_active": True})
        args: dict[str, Any] = {"resource_id": "q1"}
        assert await agent._route_to_origin("load_resource_data", args, "https://fs.example.org") == ""
        assert args == {"resource_id": "q1"}
        wprdc = ckan_sites.get_site("wprdc")["url"]
        args = {"resource_id": "r1"}
        assert await agent._route_to_origin("load_resource_data", args, wprdc) == ""
        assert await agent._route_to_origin("search_datasets", {}, "https://fs.example.org") == ""


class TestNotebookSkipsCallsThatFetchedNothing:
    @pytest.mark.parametrize(
        "preview",
        [
            "Tool unavailable: run_sql_query is NOT available on this portal",
            "Rows elsewhere: resource `r` is a mirror of WPRDC's resource",
        ],
    )
    def test_the_step_is_commented_out(self, preview: str) -> None:
        from data_concierge.agents.notebook_generator import (
            NotebookGeneratorAgent as NotebookGenerator,
        )

        cells = NotebookGenerator.__new__(NotebookGenerator)._create_cells_from_llm_trace(
            [
                {
                    "action": "run_sql_query",
                    "arguments": {"sql": "SELECT 1"},
                    "result_preview": preview,
                    "code": 'resp = requests.post("https://fs.example.org/x")',
                }
            ]
        )
        code = [c for c in cells if c.cell_type == "code"]
        assert code and all(
            line.startswith("#") for line in code[0].source.splitlines() if line.strip()
        )


class TestStickyPortals:
    def test_an_id_stays_on_the_portal_that_served_it(self) -> None:
        from data_concierge.agents.llm_agent import _remember_portal, _stick_to_portal

        seen: dict[str, str] = {}
        _remember_portal({"dataset_id": "ytxr-yigf"}, "data-pa-gov", seen)
        later: dict[str, Any] = {"resource_id": "ytxr-yigf", "limit": 500}
        _stick_to_portal(later, seen)
        assert later["portal_id"] == "data-pa-gov"

        explicit: dict[str, Any] = {"resource_id": "ytxr-yigf", "portal_id": "fairstore"}
        _stick_to_portal(explicit, seen)
        assert explicit["portal_id"] == "fairstore"

        unknown: dict[str, Any] = {"resource_id": "other"}
        _stick_to_portal(unknown, seen)
        assert "portal_id" not in unknown


class TestRunsAcrossInstances:
    """Cloud Run sends each poll to any instance; only one holds the process."""

    def _stored_running(self, job_id: str, beat_seconds_ago: float) -> None:
        from datetime import UTC, datetime, timedelta

        beat = (datetime.now(UTC) - timedelta(seconds=beat_seconds_ago)).isoformat()
        storage.write_json(
            f"onboarding_jobs/{job_id}.json",
            {"id": job_id, "status": "running", "heartbeat_at": beat, "log": ["line 1"]},
        )
        storage.write_json("onboarding_jobs/index.json", {"job_ids": [job_id]})

    def test_a_fresh_heartbeat_reads_as_running(self, store: Path) -> None:
        self._stored_running("other", 5)
        record = oj.get_job("other")
        assert record["status"] == oj.STATUS_RUNNING and record["running_elsewhere"]
        assert record["log"] == ["line 1"]

    def test_a_stale_heartbeat_reads_as_interrupted(self, store: Path) -> None:
        self._stored_running("gone", oj.STALE_AFTER_SECONDS + 5)
        record = oj.get_job("gone")
        assert record["status"] == oj.STATUS_FAILED and "interrupted" in record["error"]

    async def test_a_run_elsewhere_blocks_a_new_one(self, store: Path) -> None:
        self._stored_running("other", 5)
        with pytest.raises(oj.JobError, match="other"):
            await oj.start_mirror_job({}, target_url="https://fs.example.org", token="")

    async def test_cancel_elsewhere_is_recorded_for_the_owner(self, store: Path) -> None:
        self._stored_running("other", 5)
        assert await oj.cancel_job("other") is True
        # Its own key: the owner's heartbeat keeps rewriting the job record.
        assert storage.exists("onboarding_jobs/other.cancel.json")
        assert "cancel_requested" not in storage.read_json("onboarding_jobs/other.json")

    async def test_the_owner_honours_a_recorded_cancel(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(oj, "HEARTBEAT_SECONDS", 0.01)
        cancelled: list[str] = []

        async def fake_cancel(job_id: str) -> bool:
            cancelled.append(job_id)
            return True

        monkeypatch.setattr(oj, "cancel_job", fake_cancel)
        oj._jobs["mine"] = {"id": "mine", "status": oj.STATUS_RUNNING}
        storage.write_json("onboarding_jobs/mine.cancel.json", {"requested_at": "now"})
        await asyncio.wait_for(oj._heartbeat("mine"), timeout=2)
        assert cancelled == ["mine"]

    async def test_the_heartbeat_persists_the_log_tail(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(oj, "HEARTBEAT_SECONDS", 0.01)
        from collections import deque

        oj._jobs["mine"] = {"id": "mine", "status": oj.STATUS_RUNNING}
        oj._logs["mine"] = deque(["a", "b"])
        task = asyncio.create_task(oj._heartbeat("mine"))
        await asyncio.sleep(0.1)
        oj._jobs["mine"]["status"] = oj.STATUS_SUCCEEDED
        await asyncio.wait_for(task, timeout=2)
        stored = storage.read_json("onboarding_jobs/mine.json")
        assert stored["log"] == ["a", "b"] and stored["heartbeat_at"]


class TestReviewRegressions:
    async def test_a_failed_routed_load_stays_a_failure(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The success note must not mask an origin error from the accounting."""
        from data_concierge.agents import llm_agent

        assert "HTTP 403: Forbidden".startswith(llm_agent._NOT_RETRIEVED_PREFIXES)
        assert not "Resource: r\nTotal records: 5".startswith(
            llm_agent._NOT_RETRIEVED_PREFIXES
        )

    def test_the_final_record_beats_an_in_flight_heartbeat(self, store: Path) -> None:
        oj._finished.add("done")
        storage.write_json("onboarding_jobs/done.json", {"id": "done", "status": "succeeded"})
        oj._heartbeat_write({"id": "done", "status": "running"})
        assert storage.read_json("onboarding_jobs/done.json")["status"] == "succeeded"

    def test_storage_without_headers_stages_nothing(
        self, store: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        pf = TestStagingFromStorage()._populate()
        storage.write_json(
            "ckan_onboard/wprdc/index.json",
            {"datasets": [{"resources": [{"resource_id": "r1", "local_path": "d/wprdc/x/a.csv"}]}]},
        )
        stage = tmp_path_factory.mktemp("stage")
        assert pf._load_index("wprdc", source="storage", stage_root=stage) == {}
        assert not any(stage.iterdir())

    def test_an_unparseable_csv_does_not_stop_the_sync(
        self, store: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import csv as csv_module

        from data_concierge.data_layer import qsv_profiling

        base = tmp_path_factory.mktemp("w") / "ckan_onboard" / "wprdc"
        (base / "ds").mkdir(parents=True)
        (base / "ds" / "bad.csv").write_text("a,b\n")
        (base / "index.json").write_text("{}")

        def boom(path: Path) -> list[str]:
            raise csv_module.Error("field larger than field limit")

        monkeypatch.setattr(qsv_profiling, "csv_header", boom)
        qsv_profiling.sync_to_storage(base, "wprdc")
        assert storage.read_json("ckan_onboard/wprdc/index.json") == {}

    async def test_a_removed_portal_is_dropped_from_all(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway.router import (
            StartMirrorJobRequest,
            start_fairstore_mirror_job,
        )

        fairstore.save_settings(
            {"url": "https://fs.example.org", "mirror_sites": ["wprdc", "data-pa-gov"]}
        )
        ckan_sites.remove_site("data-pa-gov")
        seen: dict[str, Any] = {}

        async def fake_start(options: dict, **kwargs: Any) -> dict:
            seen.update(options)
            return {"id": "x", "site_name": "wprdc"}

        monkeypatch.setattr(oj, "start_mirror_job", fake_start)
        await start_fairstore_mirror_job(StartMirrorJobRequest(), admin_user={"user": "a"})
        assert seen["sites"] == ["wprdc"]


class TestSyncScrubsKeys:
    def test_describegpt_keys_never_reach_storage(
        self, store: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        from data_concierge.data_layer.qsv_profiling import sync_to_storage

        base = tmp_path_factory.mktemp("w") / "ckan_onboard" / "wprdc"
        (base / "ds").mkdir(parents=True)
        key = "sk-or-v1-" + "a1b2c3d4" * 8
        (base / "ds" / "qsv_dict.json").write_text(
            json.dumps({"attribution": f"qsv describegpt --api-key {key} --model m", "n": 1})
        )
        sync_to_storage(base, "wprdc")
        stored = storage.read_json("ckan_onboard/wprdc/ds/qsv_dict.json")
        assert key not in json.dumps(stored)
        assert stored["attribution"] == "qsv describegpt [redacted] --model m"
        assert stored["n"] == 1

    def test_stored_text_is_what_the_mirror_publishes(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """describegpt's raw text puts an escaped newline after the key; the
        mirror's raw scrub removes the word after it, so the stored copy must be
        raw-scrubbed too or storage and local runs publish different text."""
        from data_concierge.data_layer.onboard_index import _scrub_secrets
        from data_concierge.data_layer.qsv_profiling import scrubbed_json_bytes

        key = "sk-or-v1-" + "f0e1d2c3" * 8
        path = tmp_path_factory.mktemp("d") / "qsv_dict.json"
        path.write_text(
            json.dumps({"a": f"qsv describegpt --api-key {key}\nPrompt: x", "b": [1]}, indent=2)
        )
        stored = scrubbed_json_bytes(path).decode()
        assert key not in stored
        assert stored == _scrub_secrets(path.read_text())
        assert _scrub_secrets(stored) == stored
        json.loads(stored)

    def test_falls_back_to_value_scrub_when_raw_scrub_breaks_json(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        from data_concierge.data_layer.qsv_profiling import scrubbed_json_bytes

        key = "sk-or-v1-" + "9a8b7c6d" * 8
        path = tmp_path_factory.mktemp("d") / "meta.json"
        path.write_text(json.dumps({"cmd": f"--api-key {key}", "next": 2}))
        stored = json.loads(scrubbed_json_bytes(path))
        assert stored == {"cmd": "[redacted]", "next": 2}


class TestSecondReviewRegressions:
    def test_citations_credit_the_portal_that_served_the_rows(self, store: Path) -> None:
        from data_concierge.agents.llm_agent import _sources_from_trace

        sources = _sources_from_trace(
            [
                {"tool_name": "search_datasets", "portal_id": "fairstore"},
                {"tool_name": "load_resource_data", "portal_id": "wprdc"},
                {"tool_name": "load_resource_data", "portal_id": "data-pa-gov"},
            ],
            "fairstore",
            {"name": "Verikan Fair Store", "quality_score": 0.85},
            "https://fs.example.org",
        )
        assert [s.id for s in sources] == ["fairstore", "wprdc", "data-pa-gov"]
        assert sources[1].url == ckan_sites.get_site("wprdc")["url"]

    def test_an_unregistered_served_portal_falls_back_to_the_primary(self, store: Path) -> None:
        from data_concierge.agents.llm_agent import _sources_from_trace

        sources = _sources_from_trace(
            [{"tool_name": "load_resource_data", "portal_id": "ghost"}],
            "wprdc",
            {"name": "WPRDC"},
            "https://data.wprdc.org",
        )
        assert [s.id for s in sources] == ["wprdc"]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("100", 100), (0, 100), ("0", 100), (-5, 100), (float("inf"), 100), ("x", 100), (9999, 500)],
    )
    def test_int_param_defaults_below_the_floor(self, raw: Any, expected: int) -> None:
        from data_concierge.agents.llm_agent import _int_param

        assert _int_param({"limit": raw}, "limit", 100, cap=500) == expected

    def test_a_ckan_count_only_load_keeps_limit_zero(self) -> None:
        from data_concierge.agents.llm_agent import _int_param

        assert _int_param({"limit": 0}, "limit", 100, cap=500, floor=0) == 0

    async def test_sql_on_the_fair_store_redirects_without_tripping_the_breaker(
        self, store: Path
    ) -> None:
        from data_concierge.agents.llm_agent import LLMAnalysisAgent

        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})
        agent = LLMAnalysisAgent.__new__(LLMAnalysisAgent)
        agent._sql_disabled = {}
        async with httpx.AsyncClient(base_url="https://fs.example.org") as client:
            text = await agent._tool_sql(client, {"sql": 'SELECT 1 FROM "x"'})
        assert text.startswith(agent.TOOL_REDIRECT_PREFIX)
        assert agent._sql_disabled == {}

    def test_log_offsets_are_absolute(self, store: Path) -> None:
        """A tail past MAX_LOG_LINES kept returning the same offset (stalled), and
        a lagging instance's snapshot sent lines already shown (duplicated)."""
        storage.write_json(
            "onboarding_jobs/j.json",
            {"id": "j", "status": "succeeded", "log_line_count": 5000, "log": ["l4998", "l4999"]},
        )
        first = oj.get_job("j", log_offset=0)
        assert first["log"] == ["l4998", "l4999"] and first["log_next_offset"] == 5000
        assert first["log_truncated"] is True
        assert oj.get_job("j", log_offset=4999)["log"] == ["l4999"]
        again = oj.get_job("j", log_offset=5000)
        assert again["log"] == [] and again["log_next_offset"] == 5000
        # A stale snapshot behind the client's offset returns nothing, never rewinds.
        lagging = oj.get_job("j", log_offset=6000)
        assert lagging["log"] == [] and lagging["log_next_offset"] == 6000

    def test_the_final_write_is_retried(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts: list[int] = []

        def flaky(record: dict) -> bool:
            attempts.append(1)
            return len(attempts) >= 3

        monkeypatch.setattr(oj, "_write_record", flaky)
        monkeypatch.setattr(oj.time, "sleep", lambda s: None)
        oj._final_write({"id": "retry-me"})
        assert len(attempts) == 3 and "retry-me" in oj._finished

    async def test_cancel_after_exit_changes_nothing(self, store: Path) -> None:
        class Exited:
            returncode = 0

        oj._jobs["done"] = {"id": "done", "status": oj.STATUS_RUNNING}
        oj._processes["done"] = Exited()  # type: ignore[assignment]
        assert await oj.cancel_job("done") is False
        assert oj._jobs["done"]["status"] == oj.STATUS_RUNNING

    async def test_remote_cancel_of_a_job_that_just_finished_is_refused(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import UTC, datetime

        running = {"id": "x", "status": "running", "heartbeat_at": datetime.now(UTC).isoformat()}
        reads = iter([running, {"id": "x", "status": "succeeded"}])
        with monkeypatch.context() as patch:
            patch.setattr(oj.storage, "read_json", lambda key: dict(next(reads)))
            assert await oj.cancel_job("x") is False
        # Written, then withdrawn once the re-read showed the job had finished.
        assert not (store / "onboarding_jobs" / "x.cancel.json").exists()

    @pytest.mark.parametrize("site", ["FairStore", " FAIRSTORE "])
    def test_the_fair_store_cannot_be_mirrored_into_itself_by_case(
        self, store: Path, site: str
    ) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})
        with pytest.raises(oj.JobError):
            oj.build_mirror_command(
                {"site": site}, target_url="https://fs.example.org", summary_path="x"
            )

    @pytest.mark.parametrize(
        "url",
        [
            "https://fs.example.org:abc",
            "https://fs.exa\tmple.org",
            "https://fs.example.org/a\nb",
            "https://ｆｓ.example.org",
        ],
    )
    def test_urls_httpx_cannot_use_are_refused(self, url: str) -> None:
        with pytest.raises(ValueError):
            fairstore.normalize_url(url)

    def test_the_token_does_not_follow_a_port_change(self, store: Path) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "token": SECRET})
        fairstore.save_settings({"url": "https://fs.example.org:8443"})
        assert fairstore.load_settings()["token"] == ""

    def test_a_source_portal_cannot_be_the_fair_store(self, store: Path) -> None:
        wprdc = ckan_sites.get_site("wprdc")["url"]
        with pytest.raises(ValueError, match="wprdc"):
            fairstore.save_settings({"url": wprdc})

    def test_an_environment_url_on_a_source_portal_is_not_offered(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            fairstore.app_settings, "fairstore_url", ckan_sites.get_site("ckan")["url"]
        )
        fairstore.save_settings({"chat_source": True})
        assert ckan_sites.get_site(fairstore.PORTAL_ID) is None
        assert fairstore.public_settings()["conflicting_portal"] == "ckan"

    async def test_a_saved_subset_that_is_all_gone_is_refused(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway.router import (
            StartMirrorJobRequest,
            start_fairstore_mirror_job,
        )

        fairstore.save_settings({"url": "https://fs.example.org", "mirror_sites": ["data-pa-gov"]})
        ckan_sites.remove_site("data-pa-gov")
        assert fairstore.public_settings()["mirror_sites_missing"] == ["data-pa-gov"]

        async def never(*a: Any, **k: Any) -> dict:
            raise AssertionError("must not start")

        monkeypatch.setattr(oj, "start_mirror_job", never)
        with pytest.raises(HTTPException) as exc:
            await start_fairstore_mirror_job(StartMirrorJobRequest(), admin_user={"user": "a"})
        assert exc.value.status_code == 409


class TestThirdReviewRegressions:
    WPRDC = "https://data.wprdc.org"
    RID = "044f2016-1dfd-4ab0-bc1e-065da05fca2e"

    def _transport(self) -> httpx.MockTransport:
        """The Fair Store knows the resource only as a WPRDC mirror; WPRDC has rows."""

        def handler(request: httpx.Request) -> httpx.Response:
            host, path = request.url.host, request.url.path
            if host == "fs.example.org" and path.endswith("/resource_show"):
                return httpx.Response(
                    200,
                    json={
                        "success": True,
                        "result": {
                            "id": self.RID,
                            "mirror_source_portal": "wprdc",
                            "mirror_source_id": self.RID,
                            "mirror_source_datastore_active": True,
                        },
                    },
                )
            if host == "data.wprdc.org" and path.endswith("/datastore_search"):
                return httpx.Response(
                    200,
                    json={
                        "success": True,
                        "result": {
                            "records": [{"_id": 1, "name": "Frick", "acres": 644}],
                            "total": 1,
                            "fields": [{"id": "name"}, {"id": "acres"}],
                        },
                    },
                )
            return httpx.Response(404, json={"success": False, "error": {"message": "nf"}})

        return httpx.MockTransport(handler)

    async def test_editor_revisions_route_and_attribute_like_the_analysis_loop(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.agents.llm_agent import get_llm_agent
        from data_concierge.agents.notebook_editor import edit_notebook

        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})
        agent = get_llm_agent()
        transport = self._transport()

        async def client_for(url: str) -> httpx.AsyncClient:
            return httpx.AsyncClient(base_url=url, transport=transport)

        monkeypatch.setattr(agent, "_get_http_client", client_for)
        monkeypatch.setattr(agent, "_get_anthropic_client", lambda: object())
        monkeypatch.setattr(agent, "_get_mcp_tools", lambda: [])

        class Block:
            def __init__(self, **kw: Any) -> None:
                self.__dict__.update(kw)

        def response(stop: str, content: list) -> Any:
            usage = Block(
                input_tokens=1, output_tokens=1, cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )
            return Block(stop_reason=stop, content=content, id="m", model="fake", usage=usage)

        queue = [
            response(
                "tool_use",
                [Block(type="tool_use", id="t1", name="load_resource_data",
                       input={"resource_id": self.RID, "limit": 5})],
            ),
            response("end_turn", [Block(type="text", text="Done.")]),
        ]

        async def fake_call(client: object, **kwargs: object) -> Any:
            return queue.pop(0)

        monkeypatch.setattr(agent, "_call_llm_with_retry", fake_call)
        result = await edit_notebook(
            {"cells": [{"cell_type": "markdown", "source": "x", "metadata": {}}],
             "metadata": {}, "nbformat": 4, "nbformat_minor": 5},
            instruction="use the parks rows",
            query="parks",
            previous_answer="",
            data_source="fairstore",
        )
        call = next(e for e in result.agent_log if e.get("type") == "tool_execution")
        assert call["source"] == "ckan:wprdc" and call["status"] == "success"
        assert result.tool_signals.successful_tool_calls == 1
        code = result.execution_trace[0]["code"]
        assert self.WPRDC in code and "fs.example.org" not in code

    async def test_a_redirect_in_the_editor_is_neither_success_nor_failure(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.agents.llm_agent import classify_tool_result

        assert classify_tool_result("Rows elsewhere: x") == "redirected"
        assert classify_tool_result("Tool unavailable: x") == "unavailable"
        assert classify_tool_result("Error calling tool") == "error"
        assert classify_tool_result("Error: dataset 'x' not found in the catalog") == "error"
        assert classify_tool_result("Resource: r\nTotal records: 1") == "retrieved"

    def test_unknown_portal_ids_are_attributed_to_the_portal_actually_called(
        self, store: Path
    ) -> None:
        from data_concierge.agents.llm_agent import LLMAnalysisAgent, _site_id_for_url

        agent = LLMAnalysisAgent.__new__(LLMAnalysisAgent)
        # get_portal_config falls back to a registered portal for unknown IDs;
        # the canonical ID of the URL it resolves to is what served the call.
        url = agent._tool_portal_url({"portal_id": "WPRDC"}, "https://fs.example.org")
        assert _site_id_for_url(url) == "wprdc"

    def test_empty_headers_are_recorded(self, tmp_path: Path) -> None:
        from data_concierge.data_layer.qsv_profiling import recorded_headers

        base = tmp_path / "ckan_onboard" / "wprdc"
        (base / "ds").mkdir(parents=True)
        (base / "ds" / "empty.csv").write_text("")
        assert recorded_headers(base, "ckan_onboard") == {"ckan_onboard/wprdc/ds/empty.csv": []}

    def test_the_manifest_is_written_last(
        self, store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.data_layer import qsv_profiling

        base = tmp_path / "ckan_onboard" / "wprdc"
        (base / "ds").mkdir(parents=True)
        (base / "ds" / "qsv_stats.csv").write_text("field\na\n")
        (base / "index.json").write_text("{}")
        order: list[str] = []
        real_json, real_bytes = storage.write_json, storage.write_bytes
        monkeypatch.setattr(storage, "write_json", lambda k, d: (order.append(k), real_json(k, d)))
        monkeypatch.setattr(storage, "write_bytes", lambda k, d: (order.append(k), real_bytes(k, d)))
        qsv_profiling.sync_to_storage(base, "wprdc")
        assert order[-1] == "ckan_onboard/wprdc/sync_manifest.json"

    def test_a_listed_object_missing_from_storage_fails_closed(
        self, store: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        pf = TestStagingFromStorage()._populate()
        TestStagingFromStorage()._index_with("ds")
        storage.write_json(
            "ckan_onboard/wprdc/sync_manifest.json",
            {"headers": {}, "dirs": {"ckan_onboard/wprdc/ds": ["qsv_stats.csv"]}},
        )
        assert pf._load_index(
            "wprdc", source="storage", stage_root=tmp_path_factory.mktemp("s")
        ) == {}

    @pytest.mark.parametrize("name", ["_private", "-x", "a.b", ".hidden"])
    def test_ckan_style_directory_names_are_staged(self, name: str) -> None:
        pf = TestStagingFromStorage()._populate()
        assert pf._SAFE_DIR_RE.match(name)

    @pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\nb"])
    def test_unsafe_directory_names_are_not(self, name: str) -> None:
        pf = TestStagingFromStorage()._populate()
        assert not pf._SAFE_DIR_RE.match(name)

    async def test_site_config_refuses_a_source_portal(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def never(*a: Any, **k: Any) -> Any:
            raise AssertionError("must not call the portal")

        monkeypatch.setattr(fairstore, "_call", never)
        current = {"url": ckan_sites.get_site("ckan")["url"], "token": SECRET}
        with pytest.raises(fairstore.FairStoreError, match="ckan"):
            await fairstore.update_site_config({"ckan.site_title": "x"}, current)
        with pytest.raises(fairstore.FairStoreError):
            await fairstore.get_site_config(current)

    def test_an_unreadable_store_neither_saves_nor_unregisters(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})

        def boom(key: str) -> Any:
            if key == "fairstore_settings.json":
                raise OSError("GCS 503")
            return real(key)

        real = storage.read_json
        monkeypatch.setattr(storage, "read_json", boom)
        fairstore.sync_chat_source()
        with pytest.raises(fairstore.SettingsUnreadable):
            fairstore.save_settings({"chat_source": False})
        monkeypatch.setattr(storage, "read_json", real)
        assert ckan_sites.get_site(fairstore.PORTAL_ID) is not None

    def test_an_unchanged_chat_source_is_not_rewritten(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})
        writes: list[str] = []
        monkeypatch.setattr(ckan_sites, "update_site", lambda *a, **k: writes.append("x"))
        fairstore.sync_chat_source()
        assert writes == []


class TestFinalCheckRegressions:
    def test_a_duplicate_managed_entry_is_cleaned_up(self, store: Path) -> None:
        fairstore.save_settings({"url": "https://fs.example.org", "chat_source": True})
        ckan_sites.add_site(
            url="https://fs.example.org", name="dup", site_id="fairstore", managed_by="fairstore"
        )
        assert ckan_sites.get_site("fairstore-2") is not None
        fairstore.sync_chat_source()
        assert ckan_sites.get_site("fairstore-2") is None
        assert ckan_sites.get_site("fairstore") is not None
