"""Tests for the Fair Store ingestion helpers (issue #133)."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "populate_fairstore",
    Path(__file__).resolve().parent.parent.parent / "scripts" / "populate_fairstore.py",
)
populate_fairstore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(populate_fairstore)

CKANActionError = populate_fairstore.CKANActionError
_slug = populate_fairstore._slug
_column_dictionary = populate_fairstore._column_dictionary
MirrorError = populate_fairstore.MirrorError


class TestSlug:
    def test_lowercases_and_dashes(self) -> None:
        assert _slug("2010 Census Tracts", "x") == "2010-census-tracts"

    def test_strips_punctuation_and_collapses_dashes(self) -> None:
        assert _slug("A/B  --  C!!!", "x") == "a-b-c"

    def test_falls_back_when_too_short(self) -> None:
        assert _slug("!", "the-id") == "the-id"
        assert _slug("", "the-id") == "the-id"

    def test_caps_length_at_100(self) -> None:
        assert len(_slug("a" * 200, "x")) == 100


class TestColumnDictionary:
    def test_maps_qsv_median_key(self) -> None:
        """The dictionary must carry the median under the key qsv writes."""
        res = {
            "columns": [
                {
                    "name": "amount",
                    "qsv_label": "Amount",
                    "qsv_description": "dollars",
                    "qsv_type": "Float",
                    "stats": {"q2_median": "42.5", "mean": "40", "cardinality": "9"},
                    "top_values": [{"value": "1", "count": 5}] * 20,
                }
            ]
        }
        out = _column_dictionary(res)
        assert len(out) == 1
        col = out[0]
        assert col["label"] == "Amount"
        assert col["type"] == "Float"
        assert col["stats"]["q2_median"] == "42.5"
        assert col["stats"]["mean"] == "40"
        # top_values capped at 10
        assert len(col["top_values"]) == 10

    def test_empty_stats_and_columns(self) -> None:
        assert _column_dictionary({}) == []
        out = _column_dictionary({"columns": [{"name": "x"}]})
        assert out[0]["name"] == "x"
        assert out[0]["stats"] == {}

    def test_falls_back_to_ckan_label(self) -> None:
        res = {"columns": [{"name": "y", "ckan_info": {"label": "Y label"}, "ckan_type": "text"}]}
        col = _column_dictionary(res)[0]
        assert col["label"] == "Y label"
        assert col["type"] == "text"


class FakeCKAN:
    """Programmed CKAN responses; a response that is (or returns) an exception is raised."""

    ckan_url = "https://fake.example"

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def _respond(self, name, payload):
        response = self.responses.get(name, {})
        if callable(response):
            response = response(payload)
        if isinstance(response, Exception):
            raise response
        return response

    async def call(self, name, payload=None):
        self.calls.append((name, payload or {}))
        return self._respond(name, payload or {})

    async def action(self, name, payload=None):
        try:
            return await self.call(name, payload)
        except CKANActionError:
            return {}

    async def upload_call(self, name, payload, *, filename, content, content_type):
        self.calls.append((name, {**payload, "_upload": (filename, content, content_type)}))
        return self._respond(name, payload)


def _http_error(action, status=502):
    return CKANActionError(action, f"HTTP {status}", status_code=status)


def _datastore():
    """Handlers for a fake DataStore that counts the rows each table holds."""
    rows = {}

    def create(payload):
        table = payload["resource_id"]
        rows[table] = rows.get(table, 0) + len(payload.get("records") or [])
        return {"resource_id": table}

    def upsert(payload):
        rows[payload["resource_id"]] += len(payload["records"])
        return {"resource_id": payload["resource_id"]}

    def delete(payload):
        if payload["resource_id"] not in rows:
            return CKANActionError("datastore_delete", "Not found", status_code=404)
        del rows[payload["resource_id"]]
        return {}

    def search(payload):
        return {"total": rows.get(payload["resource_id"], 0), "records": []}

    return {
        "datastore_create": create,
        "datastore_upsert": upsert,
        "datastore_delete": delete,
        "datastore_search": search,
        "_rows": rows,
    }


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Retries back off with asyncio.sleep; the tests need not wait."""

    async def instant(_seconds):
        return None

    monkeypatch.setattr(populate_fairstore.asyncio, "sleep", instant)


@pytest.mark.asyncio
async def test_group_rows_follow_portal_page_caps():
    rows = [{"id": str(index), "name": f"publisher-{index}"} for index in range(5)]

    def capped_page(payload):
        offset = payload.get("offset", 0)
        return rows[offset : offset + 2]

    client = FakeCKAN({"organization_list": capped_page})

    result = await populate_fairstore._all_group_rows(client, "organization_list")

    assert result == rows
    assert [payload["offset"] for _, payload in client.calls] == [0, 2, 4, 5]


@pytest.mark.asyncio
async def test_write_action_recovers_when_gateway_times_out_after_create():
    resource = {"id": "resource-id", "name": "Created resource"}
    client = FakeCKAN(
        {"resource_create": _http_error("resource_create"), "resource_show": resource}
    )

    result = await populate_fairstore._write_action(
        client,
        "resource_create",
        resource,
        label="resource-id",
    )

    assert result == resource
    assert [action for action, _ in client.calls] == ["resource_create", "resource_show"]


def _source_client():
    organization = {
        "id": "11111111-1111-4111-8111-111111111111",
        "name": "source-publisher",
        "title": "Source Publisher",
        "description": "Original publisher description",
        "extras": [{"key": "original", "value": "yes"}],
        "groups": [],
    }
    package = {
        "id": "22222222-2222-4222-8222-222222222222",
        "name": "source-dataset",
        "title": "Source Dataset",
        "notes": "Original notes",
        "owner_org": organization["id"],
        "tags": [{"name": "source tag"}],
        "groups": [],
        "extras": [{"key": "frequency", "value": "monthly"}],
        "resources": [
            {
                "id": "33333333-3333-4333-8333-333333333333",
                "name": "Source CSV",
                "url": "https://source.example/data.csv",
                "description": "Original resource description",
                "format": "CSV",
                "datastore_active": True,
            }
        ],
    }
    return (
        FakeCKAN(
            {
                "organization_list": [organization],
                "organization_show": organization,
                "group_list": [],
                "package_search": {"count": 1, "results": [package]},
            }
        ),
        organization,
        package,
    )


@pytest.mark.asyncio
async def test_mirror_preserves_source_identity_and_adds_hierarchy_and_qsv():
    source, organization, package = _source_client()
    target = FakeCKAN(
        {
            "organization_list": [],
            "group_list": [],
            "package_search": {"count": 0, "results": []},
            "organization_create": lambda payload: payload,
            "organization_patch": lambda payload: payload,
            "package_create": lambda payload: payload,
            "resource_create": lambda payload: payload,
            "resource_view_create": lambda payload: payload,
            **_datastore(),
        }
    )
    resource_id = package["resources"][0]["id"]
    qsv_index = {
        "datasets": [
            {
                "resources": [
                    {
                        "resource_id": resource_id,
                        "qsv_description": "Generated profile",
                        "qsv_tags": ["finance"],
                        "row_count": 9,
                        "columns": [{"name": "amount", "qsv_type": "Float"}],
                    }
                ]
            }
        ]
    }

    summary = await populate_fairstore.mirror_catalog(
        source=source,
        target=target,
        site_id="source-store",
        site_title="Source Store",
        source_url="https://source.example",
        apply=True,
        qsv_index=qsv_index,
    )

    assert summary["datasets"] == 1
    assert summary["qsv_resources"] == 1
    # No output files on disk for this profile, so only its dictionary table.
    assert summary["qsv_artifacts"] == 1
    child = next(
        payload
        for action, payload in target.calls
        if action == "organization_create" and payload.get("id") == organization["id"]
    )
    assert child["name"] == organization["name"]
    hierarchy = next(
        payload
        for action, payload in target.calls
        if action == "organization_patch"
        and payload.get("id") == organization["id"]
        and "groups" in payload
    )
    assert hierarchy["groups"] == [{"name": "source-store", "capacity": "parent"}]

    dataset = next(payload for action, payload in target.calls if action == "package_create")
    assert dataset["id"] == package["id"]
    assert dataset["name"] == package["name"]
    assert dataset["notes"] == "Original notes"
    assert dataset["tags"] == [{"name": "source tag"}]

    # A new dataset is created with its resources in the same call; only the
    # qsv dictionary table is created on its own.
    created = [payload for action, payload in target.calls if action == "resource_create"]
    assert [payload.get("qsv_artifact") for payload in created] == ["dictionary"]
    [resource] = dataset["resources"]
    assert resource["id"] == resource_id
    assert resource["url"] == "https://source.example/data.csv"
    assert resource["description"] == "Original resource description"
    assert resource["qsv_description"] == "Generated profile"
    assert resource["row_count"] == 9
    assert "datastore_active" not in resource


@pytest.mark.asyncio
async def test_mirror_supplies_description_when_source_notes_are_blank():
    source, _organization, package = _source_client()
    package["notes"] = ""
    target = FakeCKAN(
        {
            "organization_list": [],
            "group_list": [],
            "package_search": {"count": 0, "results": []},
            "organization_create": lambda payload: payload,
            "organization_patch": lambda payload: payload,
            "package_create": lambda payload: payload,
            "resource_create": lambda payload: payload,
        }
    )

    await populate_fairstore.mirror_catalog(
        source=source,
        target=target,
        site_id="source-store",
        site_title="Source Store",
        source_url="https://source.example",
        apply=True,
    )

    dataset = next(payload for action, payload in target.calls if action == "package_create")
    assert dataset["notes"] == "No description was provided by the source catalog."


@pytest.mark.asyncio
async def test_source_store_slug_does_not_replace_same_named_source_organization():
    source, _organization, _package = _source_client()
    source_url = "https://source.example"
    root_id = str(populate_fairstore.uuid.uuid5(populate_fairstore.uuid.NAMESPACE_URL, source_url))
    target = FakeCKAN(
        {
            "organization_list": [{"id": root_id, "name": "source-publisher"}],
            "group_list": [],
            "package_search": {"count": 0, "results": []},
        }
    )

    summary = await populate_fairstore.mirror_catalog(
        source=source,
        target=target,
        site_id="source-publisher",
        site_title="Source Publisher",
        source_url=source_url,
    )

    assert summary["store_organization"] == "source-publisher-source"


def test_preflight_rejects_dataset_name_collision_but_reuses_group_category():
    source = {
        "organizations": [],
        "groups": [{"id": "source-group", "name": "education"}],
        "packages": [{"id": "source-package", "name": "shared-name", "resources": []}],
    }
    target = {
        "organizations": [],
        "groups": [{"id": "target-group", "name": "education"}],
        "packages": [{"id": "target-package", "name": "shared-name", "resources": []}],
    }
    with pytest.raises(MirrorError, match="dataset name 'shared-name'"):
        populate_fairstore._assert_no_collisions(
            source,
            target,
            store_name="source-store",
            store_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        )


@pytest.mark.asyncio
async def test_existing_source_ids_are_patched_instead_of_duplicated():
    source, organization, package = _source_client()
    root_id = str(
        populate_fairstore.uuid.uuid5(
            populate_fairstore.uuid.NAMESPACE_URL, "https://source.example"
        )
    )
    target = FakeCKAN(
        {
            "organization_list": [
                {"id": root_id, "name": "source-store"},
                {"id": organization["id"], "name": organization["name"]},
            ],
            "group_list": [],
            "package_search": {
                "count": 1,
                "results": [
                    {
                        "id": package["id"],
                        "name": package["name"],
                        "resources": [
                            {**resource, "package_id": package["id"]}
                            for resource in package["resources"]
                        ],
                    }
                ],
            },
            "organization_patch": lambda payload: payload,
            "package_patch": lambda payload: payload,
            "resource_update": lambda payload: payload,
        }
    )

    await populate_fairstore.mirror_catalog(
        source=source,
        target=target,
        site_id="source-store",
        site_title="Source Store",
        source_url="https://source.example",
        apply=True,
    )

    actions = [action for action, _ in target.calls]
    assert "organization_create" not in actions
    assert "package_create" not in actions
    assert "resource_create" not in actions
    assert actions.count("organization_patch") == 3
    assert actions.count("package_patch") == 1
    # A source resource is replaced whole, so a field the source dropped goes.
    assert actions.count("resource_update") == 1


def _mirrored_extras(portal):
    return [
        {"key": "mirror_source_portal", "value": portal},
        {"key": "mirror_source_url", "value": f"https://{portal}.example"},
    ]


@pytest.mark.asyncio
async def test_source_snapshot_skips_records_the_portal_mirrored_from_elsewhere():
    """A Fair Store that already mirrors WPRDC must not re-publish WPRDC as its own."""
    native_org = {"id": "org-native", "name": "dathere", "extras": []}
    copied_org = {
        "id": "org-copy",
        "name": "city-of-pittsburgh",
        "extras": _mirrored_extras("wprdc"),
    }
    native = {"id": "pkg-native", "name": "native", "owner_org": "org-native", "resources": []}
    copied = {
        "id": "pkg-copy",
        "name": "copied",
        "owner_org": "org-copy",
        "extras": _mirrored_extras("wprdc"),
        "resources": [],
    }
    orgs = {row["id"]: row for row in (native_org, copied_org)}
    client = FakeCKAN(
        {
            "organization_list": [native_org, copied_org],
            "organization_show": lambda payload: orgs[payload["id"]],
            "group_list": [],
            "package_search": {"count": 2, "results": [native, copied]},
        }
    )

    snapshot = await populate_fairstore._source_snapshot(client, site_id="ckan")

    assert [row["name"] for row in snapshot["organizations"]] == ["dathere"]
    assert [row["name"] for row in snapshot["packages"]] == ["native"]
    assert snapshot["copies"] == {
        "organizations": 1,
        "groups": 0,
        "datasets": 1,
        "origins": ["wprdc"],
    }

    # Mirroring the origin portal itself keeps its own records.
    own = await populate_fairstore._source_snapshot(client, site_id="wprdc")
    assert len(own["packages"]) == 2


def test_package_payload_carries_schema_fields_the_target_would_drop():
    package = {
        "id": "pkg",
        "name": "pkg",
        "title": "Package",
        "notes": "Notes",
        "owner_org": "org",
        "data_steward_name": "Jane Steward",
        "temporal_coverage": "2015-2024",
        "related_documents": ["https://example.org/a.pdf"],
        "num_resources": 3,
        "tracking_summary": {"total": 9},
        "organization": {"name": "org"},
        "metadata_modified": "2026-09-01T00:00:00",
        "extras": [{"key": "frequency", "value": "monthly"}],
    }

    payload = populate_fairstore._package_payload(
        package,
        site_id="wprdc",
        source_url="https://source.example",
        store_id="root",
        known_org_ids={"org"},
    )

    extras = {item["key"]: item["value"] for item in payload["extras"]}
    assert extras["data_steward_name"] == "Jane Steward"
    assert extras["temporal_coverage"] == "2015-2024"
    assert extras["related_documents"] == '["https://example.org/a.pdf"]'
    assert extras["frequency"] == "monthly"
    assert extras["mirror_source_metadata_modified"] == "2026-09-01T00:00:00"
    assert extras["mirror_source_id"] == "pkg"
    # Computed by CKAN, so neither copied as a field nor carried as an extra.
    for computed in ("num_resources", "tracking_summary", "organization"):
        assert computed not in extras
        assert computed not in payload
    assert payload["owner_org"] == "org"


def test_resource_payload_keeps_uploads_pointing_at_the_source_copy():
    """CKAN rewrites an upload's URL to the serving site, where no file exists."""
    resource = {
        "id": "res",
        "url": "https://source.example/dataset/pkg/resource/res/download/data.csv",
        "url_type": "upload",
        "name": "Data",
        "format": "CSV",
        "position": 0,
        "tracking_summary": {"total": 1},
        "datastore_active": True,
        "datastore_contains_all_records_of_source_file": True,
        "dataspatial_wkt_field": "geom",
        "preview_rows": [1, 2],
    }

    payload = populate_fairstore._resource_payload(
        resource, package_id="pkg", site_id="wprdc", source_url="https://source.example", qsv=None
    )

    assert payload["url"] == resource["url"]
    assert "url_type" not in payload
    assert payload["mirror_source_url_type"] == "upload"
    assert payload["mirror_source_datastore_active"] is True
    assert not any(key.startswith("datastore_") for key in payload)
    assert payload["dataspatial_wkt_field"] == "geom"
    assert payload["preview_rows"] == [1, 2]
    for bookkeeping in ("position", "tracking_summary"):
        assert bookkeeping not in payload


def test_resource_payload_leaves_plain_links_alone():
    payload = populate_fairstore._resource_payload(
        {"id": "res", "url": "https://elsewhere.example/a.csv", "url_type": ""},
        package_id="pkg",
        site_id="wprdc",
        source_url="https://source.example",
        qsv=None,
    )
    assert payload["url_type"] == ""
    assert "mirror_source_url_type" not in payload


def test_uploaded_images_use_the_source_absolute_url():
    uploaded = {
        "image_url": "2015-10-09-seal.jpg",
        "image_display_url": "https://source.example/uploads/group/2015-10-09-seal.jpg",
    }
    assert populate_fairstore._with_image({"image_url": "x"}, uploaded)["image_url"] == (
        "https://source.example/uploads/group/2015-10-09-seal.jpg"
    )
    linked = {"image_url": "https://cdn.example/logo.png"}
    assert populate_fairstore._with_image({}, linked)["image_url"] == "https://cdn.example/logo.png"
    assert "image_url" not in populate_fairstore._with_image({"image_url": ""}, {})


class TestCleanTag:
    def test_valid_tags_pass_through_unchanged(self) -> None:
        for tag in ("covid-19", "K-12 Education", "u.s.", "snake_case"):
            assert populate_fairstore._clean_tag(tag) == tag

    def test_invalid_characters_become_hyphens(self) -> None:
        assert populate_fairstore._clean_tag("l&i") == "l-i"
        assert populate_fairstore._clean_tag("fw&gs") == "fw-gs"

    def test_unusable_tags_are_dropped(self) -> None:
        assert populate_fairstore._clean_tag("&") == ""
        assert populate_fairstore._clean_tag("a") == ""

    def test_duplicates_after_cleaning_collapse(self) -> None:
        refs = populate_fairstore._tag_refs([{"name": "l&i"}, {"name": "l-i"}])
        assert refs == [{"name": "l-i"}]


_PORTAL = "https://data.example.gov"


def _dcat_dataset(view_id, title, **overrides):
    dataset = {
        "@type": "dcat:Dataset",
        "identifier": f"{_PORTAL}/api/views/{view_id}",
        "title": title,
        "description": f"About {title}",
        "landingPage": f"{_PORTAL}/d/{view_id}",
        "modified": "2026-02-28",
        "accessLevel": "public",
        "keyword": ["health", "l&i"],
        "theme": ["Health"],
        "license": "https://www.usa.gov/government-works",
        "contactPoint": {"fn": "Data Team", "hasEmail": "mailto:data@example.gov"},
        "publisher": {"@type": "org:Organization", "name": "data.example.gov"},
        "distribution": [
            {
                "@type": "dcat:Distribution",
                "downloadURL": f"{_PORTAL}/api/views/{view_id}/rows.csv",
                "mediaType": "text/csv",
            },
            {
                "@type": "dcat:Distribution",
                "downloadURL": f"{_PORTAL}/api/views/{view_id}/rows.json",
                "mediaType": "application/json",
                "describedBy": f"{_PORTAL}/api/views/{view_id}/columns.json",
            },
        ],
    }
    dataset.update(overrides)
    return dataset


def _socrata_row(view_id, owner):
    return {
        "resource": {
            "id": view_id,
            "type": "dataset",
            "attribution": "Bureau of Records",
            "columns_field_name": ["county", "deaths"],
            "columns_name": ["County", "Deaths"],
            "columns_datatype": ["Text", "Number"],
            "columns_description": ["County name", "Count of deaths"],
            "page_views": {"page_views_total": 10},
        },
        "classification": {
            "domain_category": "Health",
            "domain_metadata": [
                {"key": "Data-Management_Business-Owner", "value": owner},
                {"key": "Data-Management_Update-Frequency", "value": "Monthly"},
            ],
        },
        "owner": {"display_name": "Someone"},
    }


def _snapshot(datasets, socrata=None):
    return populate_fairstore._dcat_snapshot(
        {"dataset": datasets},
        site_title="Example Open Data",
        source_url=_PORTAL,
        socrata=socrata,
    )


def test_dcat_snapshot_derives_stable_identity_and_keeps_every_field():
    snapshot = _snapshot([_dcat_dataset("abcd-1234", "Overdose Deaths")])
    [package] = snapshot["packages"]

    expected_id = str(
        populate_fairstore.uuid.uuid5(
            populate_fairstore.uuid.NAMESPACE_URL, f"{_PORTAL}/api/views/abcd-1234"
        )
    )
    assert package["id"] == expected_id
    assert package["name"] == "overdose-deaths-abcd-1234"
    assert package["url"] == f"{_PORTAL}/d/abcd-1234"
    assert package["license_id"] == "other-pd"
    assert package["maintainer_email"] == "data@example.gov"
    assert package["groups"] == [{"name": "health"}]
    extras = {item["key"]: item["value"] for item in package["extras"]}
    assert extras["dcat_identifier"] == f"{_PORTAL}/api/views/abcd-1234"
    assert extras["dcat_accessLevel"] == "public"
    assert extras["dcat_keyword"] == '["health", "l&i"]'
    assert extras["dcat_license"] == "https://www.usa.gov/government-works"
    assert "dcat_distribution" not in extras

    csv, json_dist = package["resources"]
    assert csv["format"] == "CSV" and json_dist["format"] == "JSON"
    assert json_dist["dcat_describedBy"] == f"{_PORTAL}/api/views/abcd-1234/columns.json"
    assert csv["_source_id"] == f"{_PORTAL}/api/views/abcd-1234/rows.csv"
    # Deterministic, so a rerun patches instead of duplicating.
    assert (
        _snapshot([_dcat_dataset("abcd-1234", "Overdose Deaths")])["packages"][0]["resources"][0][
            "id"
        ]
        == csv["id"]
    )

    # The portal publishing under its own name is the root, not a child org.
    assert snapshot["organizations"] == []
    assert package["owner_org"] == ""
    assert snapshot["groups"][0]["title"] == "Health"


def test_dcat_snapshot_uses_socrata_owner_for_the_hierarchy_and_columns():
    socrata = {
        "abcd-1234": _socrata_row("abcd-1234", "Department of Health (DOH)"),
        "wxyz-9876": _socrata_row("wxyz-9876", "Department of Health (DOH)"),
    }
    snapshot = _snapshot(
        [_dcat_dataset("abcd-1234", "Overdose Deaths"), _dcat_dataset("wxyz-9876", "Births")],
        socrata=socrata,
    )

    [org] = snapshot["organizations"]
    assert org["name"] == "department-of-health-doh"
    assert org["title"] == "Department of Health (DOH)"
    assert org["groups"] == []  # hangs off the portal root
    assert {"key": "publisher_source", "value": "socrata:Data-Management_Business-Owner"} in org[
        "extras"
    ]
    assert all(package["owner_org"] == org["id"] for package in snapshot["packages"])

    extras = {item["key"]: item["value"] for item in snapshot["packages"][0]["extras"]}
    assert extras["socrata_id"] == "abcd-1234"
    assert extras["socrata_attribution"] == "Bureau of Records"
    assert extras["socrata_Data-Management_Update-Frequency"] == "Monthly"
    # Volatile analytics and account names stay out.
    assert not any("page_views" in key or "owner" == key for key in extras)

    # The column list sits on the CSV it describes, not on the dataset, where
    # Solr would index it as a single (size-limited) term.
    assert "source_data_dictionary" not in extras
    csv, json_dist = snapshot["packages"][0]["resources"]
    assert "source_data_dictionary" not in json_dist
    assert csv["source_column_count"] == "2"
    columns = populate_fairstore.json.loads(csv["source_data_dictionary"])
    assert columns[1] == {
        "name": "deaths",
        "label": "Deaths",
        "type": "Number",
        "description": "Count of deaths",
    }


def test_dataset_extras_over_the_solr_term_limit_are_named_not_stored():
    """One oversized extra must not make CKAN reject the whole dataset."""
    package = {
        "id": "pkg",
        "name": "pkg",
        "notes": "Notes",
        "owner_org": "org",
        "spatial": "x" * 40_000,
        "extras": [{"key": "frequency", "value": "monthly"}],
    }

    payload = populate_fairstore._package_payload(
        package,
        site_id="wprdc",
        source_url="https://source.example",
        store_id="root",
        known_org_ids={"org"},
    )

    extras = {item["key"]: item["value"] for item in payload["extras"]}
    assert "spatial" not in extras
    assert extras["mirror_oversized_extras"] == '["spatial"]'
    assert extras["frequency"] == "monthly"
    assert extras["mirror_source_id"] == "pkg"


def test_dcat_publisher_chain_becomes_nested_organizations():
    chained = _dcat_dataset(
        "abcd-1234",
        "Permits",
        publisher={
            "name": "Office of Permits",
            "subOrganizationOf": {
                "name": "Department of State",
                "subOrganizationOf": {"name": "data.example.gov"},
            },
        },
    )
    snapshot = _snapshot([chained])

    orgs = {org["title"]: org for org in snapshot["organizations"]}
    assert set(orgs) == {"Office of Permits", "Department of State"}
    assert orgs["Office of Permits"]["groups"] == [
        {"name": "department-of-state", "capacity": "parent"}
    ]
    assert orgs["Department of State"]["groups"] == []
    assert snapshot["packages"][0]["owner_org"] == orgs["Office of Permits"]["id"]


def test_dcat_hierarchy_ignores_a_contradiction_that_would_loop():
    a_under_b = _dcat_dataset(
        "abcd-1234",
        "One",
        publisher={"name": "Agency A", "subOrganizationOf": {"name": "Agency B"}},
    )
    b_under_a = _dcat_dataset(
        "wxyz-9876",
        "Two",
        publisher={"name": "Agency B", "subOrganizationOf": {"name": "Agency A"}},
    )

    orgs = {org["title"]: org for org in _snapshot([a_under_b, b_under_a])["organizations"]}

    assert orgs["Agency A"]["groups"] == [{"name": "agency-b", "capacity": "parent"}]
    assert orgs["Agency B"]["groups"] == []


def test_dcat_uuid_keeps_a_source_uuid_identifier():
    """DKAN catalogs identify datasets by UUID; the mirror keeps it, as for CKAN."""
    source_uuid = "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"
    assert populate_fairstore._dcat_uuid(source_uuid, _PORTAL, "dataset") == source_uuid
    name = populate_fairstore._dcat_dataset_name("Water Quality", source_uuid, source_uuid)
    assert name == "water-quality-0a1b2c3d"


def test_dcat_qsv_index_is_rekeyed_to_the_first_tabular_resource():
    snapshot = _snapshot([_dcat_dataset("abcd-1234", "Overdose Deaths")])
    index = {"datasets": [{"resources": [{"resource_id": "abcd-1234", "row_count": 5}]}]}

    rekeyed = populate_fairstore._dcat_qsv_index(index, snapshot)

    [resource] = rekeyed["datasets"][0]["resources"]
    assert resource["resource_id"] == snapshot["packages"][0]["resources"][0]["id"]
    assert resource["row_count"] == 5


@pytest.mark.asyncio
async def test_dcat_mirror_creates_datasets_with_resources_under_the_portal_root():
    socrata = {"abcd-1234": _socrata_row("abcd-1234", "Department of Health (DOH)")}
    snapshot = _snapshot([_dcat_dataset("abcd-1234", "Overdose Deaths")], socrata=socrata)
    target = FakeCKAN(
        {
            "organization_list": [],
            "group_list": [],
            "package_search": {"count": 0, "results": []},
            "organization_create": lambda payload: payload,
            "organization_patch": lambda payload: payload,
            "group_create": lambda payload: payload,
            "package_create": lambda payload: payload,
        }
    )

    summary = await populate_fairstore.mirror_catalog(
        source=None,
        snapshot=snapshot,
        target=target,
        site_id="example",
        site_title="Example Open Data",
        source_url=_PORTAL,
        apply=True,
    )

    assert summary["datasets"] == 1 and summary["resources"] == 2
    hierarchy = next(
        payload
        for action, payload in target.calls
        if action == "organization_patch" and "groups" in payload
    )
    assert hierarchy["groups"] == [{"name": "example", "capacity": "parent"}]
    dataset = next(payload for action, payload in target.calls if action == "package_create")
    assert dataset["tags"] == [{"name": "health"}, {"name": "l-i"}]
    assert len(dataset["resources"]) == 2
    extras = {item["key"]: item["value"] for item in dataset["extras"]}
    assert extras["mirror_source_id"] == f"{_PORTAL}/api/views/abcd-1234"
    assert not any(key.startswith("_") for key in dataset)
    assert not any(key.startswith("_") for resource in dataset["resources"] for key in resource)


_FAKE_KEY = "sk-or-v1-" + "0" * 64
_ATTRIBUTION = (
    "\n\nAttribution:\nGenerated by qsv v20.0.0 describegpt\n"
    f"Command line: qsv describegpt data.csv --all --api-key {_FAKE_KEY}\n"
    "Model: google/gemini-2.5-flash-lite\n"
    "WARNING: Description generated by an LLM and may contain inaccuracies."
)


def _qsv_profile(tmp_path, *, frequency_rows=2):
    """An onboarded profile on disk, shaped like onboard_ckan.py's output."""
    directory = tmp_path / "data" / "ckan_onboard" / "wprdc" / "parking"
    directory.mkdir(parents=True)
    (directory / "qsv_stats.csv").write_text(
        "field,type,min,max,mean,nullcount,cardinality\n"
        "zone,String,A,Z,,0,26\n"
        "spaces,Integer,1,400,52.5,3,120\n"
    )
    rows = "".join(
        f"zone,Z{index},{index + 1},1.5,{index + 1}\n" for index in range(frequency_rows)
    )
    (directory / "qsv_frequency.csv").write_text("field,value,count,percentage,rank\n" + rows)
    (directory / "qsv_dict.json").write_text(
        populate_fairstore.json.dumps({"Description": {"response": "Parking." + _ATTRIBUTION}})
    )
    return {
        "resource_id": "44444444-4444-4444-8444-444444444444",
        "resource_name": "Parking CSV",
        "local_path": str(directory / "data.csv"),
        "onboarded_at": "2026-05-08T07:03:37+00:00",
        "row_count": 138,
        "status": "ok",
        "qsv_description": "Parking spaces by zone." + _ATTRIBUTION,
        "qsv_tags": ["parking", "zoning"],
        "columns": [
            {
                "name": "zone",
                "qsv_label": "Zone",
                "qsv_description": "Parking zone letter",
                "qsv_type": "String",
                "stats": {"nullcount": "0", "cardinality": "26", "min": "A", "max": "Z"},
                "top_values": [{"value": "A", "count": 9}],
            },
            {
                "name": "spaces",
                "qsv_label": "Spaces",
                "qsv_type": "Integer",
                "stats": {"mean": "52.5", "q2_median": "40", "nullcount": "3"},
            },
        ],
    }


def test_qsv_dataset_extras_summarise_the_profile_without_the_api_key(tmp_path):
    extras = populate_fairstore._qsv_dataset_extras(_qsv_profile(tmp_path))

    # Only the prose: the provenance block, key and command line included,
    # stays in the describegpt resource.
    assert extras["qsv_description"] == "Parking spaces by zone."
    assert extras["qsv_tags"] == '["parking", "zoning"]'
    assert extras["qsv_row_count"] == "138"
    assert extras["qsv_column_count"] == "2"
    assert extras["qsv_columns"] == '["zone", "spaces"]'
    assert extras["qsv_model"] == "google/gemini-2.5-flash-lite"
    assert extras["qsv_version"] == "20.0.0"
    assert extras["qsv_profiled_resource_id"] == "44444444-4444-4444-8444-444444444444"


def test_qsv_artifacts_cover_dictionary_stats_frequency_and_describegpt(tmp_path):
    artifacts = {a["kind"]: a for a in populate_fairstore._qsv_artifacts(_qsv_profile(tmp_path))}

    assert list(artifacts) == ["dictionary", "stats", "frequency", "describegpt"]

    fields, records = artifacts["dictionary"]["table"]
    assert [f["id"] for f in fields][:4] == ["column", "label", "type", "description"]
    assert records[0]["label"] == "Zone" and records[0]["cardinality"] == 26
    assert records[1]["median"] == 40 and records[1]["null_count"] == 3
    assert records[1]["description"] == ""

    fields, records = artifacts["frequency"]["table"]
    types = {f["id"]: f["type"] for f in fields}
    assert types == {
        "field": "text",
        "value": "text",
        "count": "numeric",
        "percentage": "numeric",
        "rank": "numeric",
    }
    assert records[1] == {"field": "zone", "value": "Z1", "count": 2, "percentage": 1.5, "rank": 2}

    fields, records = artifacts["stats"]["table"]
    assert records[1]["mean"] == "52.5" and records[0]["mean"] == ""

    filename, content, content_type = artifacts["describegpt"]["file"]
    assert (filename, content_type) == ("describegpt.json", "application/json")
    assert _FAKE_KEY.encode() not in content


def test_qsv_artifacts_without_output_files_still_publish_the_dictionary(tmp_path):
    profile = _qsv_profile(tmp_path)
    profile["local_path"] = str(tmp_path / "missing" / "data.csv")
    assert [a["kind"] for a in populate_fairstore._qsv_artifacts(profile)] == ["dictionary"]


@pytest.mark.asyncio
async def test_publish_qsv_creates_datastore_tables_and_uploads_describegpt(tmp_path):
    profile = _qsv_profile(tmp_path, frequency_rows=2500)
    datastore = _datastore()
    target = FakeCKAN(
        {
            "resource_create": lambda payload: payload,
            "resource_view_create": lambda payload: payload,
            **datastore,
        }
    )

    counts = await populate_fairstore._publish_qsv(
        target,
        package_id="pkg",
        qsv=profile,
        site_id="wprdc",
        existing_resources=set(),
    )

    assert counts == {"created": 4, "refreshed": 0, "unchanged": 0, "removed": 0}
    expected_id = populate_fairstore._qsv_resource_id(profile["resource_id"], "frequency")
    tables = [
        payload
        for action, payload in target.calls
        if action == "resource_create" and "_upload" not in payload
    ]
    assert [t["qsv_artifact"] for t in tables] == ["dictionary", "stats", "frequency"]
    assert tables[2]["id"] == expected_id and tables[2]["package_id"] == "pkg"
    # What datastore_create would set for an inline resource.
    assert tables[2]["url_type"] == "datastore"
    assert tables[2]["url"] == "_datastore_only_resource"
    creates = [payload for action, payload in target.calls if action == "datastore_create"]
    frequency = creates[2]
    assert frequency["resource_id"] == expected_id
    assert len(frequency["records"]) == 1000
    assert datastore["_rows"][expected_id] == 2500
    # The rest of the 2,500 rows follow in insert batches.
    upserts = [payload for action, payload in target.calls if action == "datastore_upsert"]
    assert [len(u["records"]) for u in upserts] == [1000, 500]
    assert all(u["method"] == "insert" and u["resource_id"] == expected_id for u in upserts)

    [upload] = [
        payload
        for action, payload in target.calls
        if action == "resource_create" and "_upload" in payload
    ]
    assert upload["qsv_artifact"] == "describegpt"
    assert upload["_upload"][0] == "describegpt.json"

    # Each table gets its sortable table view once its rows are loaded.
    views = [payload for action, payload in target.calls if action == "resource_view_create"]
    assert [v["view_type"] for v in views] == ["datatables_view"] * 3
    assert views[2]["resource_id"] == expected_id


@pytest.mark.asyncio
async def test_publish_qsv_rerun_replaces_tables_instead_of_appending(tmp_path):
    profile = _qsv_profile(tmp_path)
    existing = {
        populate_fairstore._qsv_resource_id(profile["resource_id"], kind)
        for kind in ("dictionary", "stats", "frequency", "describegpt")
    }
    datastore = _datastore()
    # The tables hold the first run's rows.
    for kind, count in (("dictionary", 2), ("stats", 2), ("frequency", 2)):
        datastore["_rows"][populate_fairstore._qsv_resource_id(profile["resource_id"], kind)] = (
            count
        )
    target = FakeCKAN(
        {
            "resource_patch": lambda payload: payload,
            # The tables already have their view from the first run.
            "resource_view_list": [{"view_type": "datatables_view"}],
            **datastore,
        }
    )

    await populate_fairstore._publish_qsv(
        target,
        package_id="pkg",
        qsv=profile,
        site_id="wprdc",
        existing_resources=existing,
    )

    actions = [action for action, _ in target.calls]
    assert "resource_create" not in actions
    assert "resource_view_create" not in actions
    assert actions.count("datastore_delete") == 3
    for index, action in enumerate(actions):
        if action == "datastore_create":
            assert actions[index - 1] == "datastore_delete"
    creates = [payload for action, payload in target.calls if action == "datastore_create"]
    assert all("resource" not in c and c["resource_id"] in existing for c in creates)
    assert sorted(datastore["_rows"].values()) == [2, 2, 2]


@pytest.mark.asyncio
async def test_mirror_puts_the_qsv_profile_on_the_dataset(tmp_path):
    source, _organization, package = _source_client()
    profile = _qsv_profile(tmp_path)
    profile["resource_id"] = package["resources"][0]["id"]
    target = FakeCKAN(
        {
            "organization_list": [],
            "group_list": [],
            "package_search": {"count": 0, "results": []},
            "organization_create": lambda payload: payload,
            "organization_patch": lambda payload: payload,
            "package_create": lambda payload: payload,
            "resource_create": lambda payload: payload,
            "resource_view_create": lambda payload: payload,
            **_datastore(),
        }
    )

    summary = await populate_fairstore.mirror_catalog(
        source=source,
        target=target,
        site_id="source-store",
        site_title="Source Store",
        source_url="https://source.example",
        apply=True,
        qsv_index={"datasets": [{"resources": [profile]}]},
    )

    assert summary["qsv_artifacts"] == 4
    dataset = next(payload for action, payload in target.calls if action == "package_create")
    extras = {item["key"]: item["value"] for item in dataset["extras"]}
    assert extras["qsv_tags"] == '["parking", "zoning"]'
    assert extras["qsv_columns"] == '["zone", "spaces"]'
    assert extras["mirror_source_id"] == package["id"]


def test_source_clients_never_fall_back_to_the_configured_key(monkeypatch):
    """The Fair Store token in CKAN_API_KEY must not be sent to a source portal."""
    from pydantic import SecretStr

    from data_concierge.core.config import settings
    from data_concierge.data_layer.connectors.ckan import CKANClient

    monkeypatch.setattr(settings, "ckan_api_key", SecretStr("fairstore-sysadmin-token"))

    assert CKANClient("https://source.example", use_default_key=False)._api_key is None
    assert CKANClient("https://target.example")._api_key == "fairstore-sysadmin-token"


def test_summary_tables_read_and_cap_geometry_sized_cells(tmp_path):
    """qsv reports whole WKT polygons as a column's min/max/mode."""
    path = tmp_path / "qsv_stats.csv"
    wkt = "POLYGON((" + "1 2," * 60_000 + "1 2))"
    path.write_text(f'field,mode\ngeom,"{wkt}"\n')

    _fields, [record] = populate_fairstore._csv_table(path)

    assert len(record["mode"]) < 10_100
    assert record["mode"].endswith(f"[truncated: {len(wkt):,} characters]")


# -- reads fail closed (review findings: partial snapshots) -------------------


@pytest.mark.asyncio
async def test_a_failed_page_aborts_instead_of_returning_part_of_the_organizations():
    rows = [{"id": str(index), "name": f"publisher-{index}"} for index in range(5)]

    def page(payload):
        offset = payload.get("offset", 0)
        return rows[offset : offset + 2] if offset == 0 else _http_error("organization_list")

    client = FakeCKAN({"organization_list": page})

    with pytest.raises(MirrorError, match="organization_list failed"):
        await populate_fairstore._all_group_rows(client, "organization_list")
    # The second page was retried before giving up.
    assert [payload["offset"] for _, payload in client.calls] == [0, 2, 2, 2]


@pytest.mark.asyncio
async def test_a_transient_failure_is_retried_and_the_read_completes():
    rows = [{"id": str(index), "name": f"publisher-{index}"} for index in range(3)]
    failures = iter([True])

    def page(payload):
        offset = payload.get("offset", 0)
        if offset == 2 and next(failures, False):
            return _http_error("organization_list", 503)
        return rows[offset : offset + 2]

    client = FakeCKAN({"organization_list": page})

    assert await populate_fairstore._all_group_rows(client, "organization_list") == rows


@pytest.mark.asyncio
async def test_package_paging_that_stops_short_of_the_count_aborts():
    client = FakeCKAN(
        {
            "package_search": lambda payload: (
                {"count": 3, "results": [{"id": "a"}, {"id": "b"}]}
                if payload["start"] == 0
                else {"count": 3, "results": []}
            )
        }
    )

    with pytest.raises(MirrorError, match="stopped at 2 of 3"):
        await populate_fairstore._all_packages(client)


@pytest.mark.asyncio
async def test_organization_show_failure_is_not_replaced_by_the_list_row():
    source, _organization, _package = _source_client()
    source.responses["organization_show"] = _http_error("organization_show", 500)
    target = FakeCKAN(
        {"organization_list": [], "group_list": [], "package_search": {"count": 0, "results": []}}
    )

    with pytest.raises(MirrorError, match="organization_show failed"):
        await populate_fairstore.mirror_catalog(
            source=source,
            target=target,
            site_id="source-store",
            site_title="Source Store",
            source_url="https://source.example",
            apply=True,
        )
    assert not [action for action, _ in target.calls if action.endswith(("_create", "_patch"))]


@pytest.mark.asyncio
async def test_socrata_enrichment_failure_aborts_a_run_that_writes(monkeypatch):
    catalog = {"dataset": [{"identifier": "https://data.example.gov/api/views/abcd-1234"}]}

    async def unavailable(_client, _domain, _offset):
        raise populate_fairstore.httpx.ConnectError("down")

    monkeypatch.setattr(populate_fairstore, "_socrata_page", unavailable)

    with pytest.raises(MirrorError, match="--no-enrich"):
        await populate_fairstore._socrata_metadata(catalog, required=True)
    # A dry run only warns.
    assert await populate_fairstore._socrata_metadata(catalog, required=False) == {}


@pytest.mark.asyncio
async def test_socrata_pages_are_retried_on_rate_limits():
    import httpx

    responses = iter(
        [
            httpx.Response(429),
            httpx.Response(200, json={"results": [{"resource": {"id": "abcd-1234"}}]}),
        ]
    )
    transport = httpx.MockTransport(lambda request: next(responses))
    async with httpx.AsyncClient(transport=transport) as client:
        body = await populate_fairstore._socrata_page(client, "data.example.gov", 0)

    assert body["results"][0]["resource"]["id"] == "abcd-1234"


# -- renames and collisions ----------------------------------------------------


def _target_row(row_id, name, portal):
    return {
        "id": row_id,
        "name": name,
        "extras": [{"key": "mirror_source_portal", "value": portal}],
    }


def test_a_source_rename_of_this_portals_record_is_a_rename_not_a_collision():
    source = {
        "organizations": [],
        "groups": [],
        "packages": [{"id": "pkg-1", "name": "budget-2024", "resources": []}],
    }
    target = {
        "organizations": [],
        "groups": [],
        "packages": [_target_row("pkg-1", "budget-2023", "wprdc")],
    }

    renames = populate_fairstore._assert_no_collisions(
        source, target, store_name="wprdc", store_id="root", site_id="wprdc"
    )

    assert renames == ["dataset 'budget-2023' -> 'budget-2024'"]


def test_the_same_uuid_under_another_portals_provenance_is_still_a_collision():
    source = {
        "organizations": [],
        "groups": [],
        "packages": [{"id": "pkg-1", "name": "budget-2024", "resources": []}],
    }
    target = {
        "organizations": [],
        "groups": [],
        "packages": [_target_row("pkg-1", "budget-2023", "data-pa-gov")],
    }

    with pytest.raises(MirrorError, match="already has a different name"):
        populate_fairstore._assert_no_collisions(
            source, target, store_name="wprdc", store_id="root", site_id="wprdc"
        )


def test_duplicate_uuids_within_one_snapshot_are_collisions():
    source = {
        "organizations": [],
        "groups": [],
        "packages": [
            {"id": "same", "name": "first", "resources": []},
            {"id": "same", "name": "second", "resources": []},
        ],
    }
    empty = {"organizations": [], "groups": [], "packages": []}

    with pytest.raises(MirrorError, match="appears twice"):
        populate_fairstore._assert_no_collisions(
            source, empty, store_name="x", store_id="root", site_id="x"
        )


def test_dcat_names_already_published_are_kept_and_new_ones_avoid_them():
    source = {
        "organizations": [
            # A new publisher now sorts first and took the bare slug.
            {"id": "org-new", "name": "health", "groups": []},
            {"id": "org-old", "name": "health-0badc0de", "groups": []},
            {"id": "org-child", "name": "clinics", "groups": [{"name": "health-0badc0de"}]},
        ],
        "groups": [],
        "packages": [{"id": "pkg-1", "name": "retitled-abcd-1234", "resources": []}],
    }
    target = {
        "organizations": [
            _target_row("org-old", "health", "data-pa-gov"),
            _target_row("org-child", "clinics", "data-pa-gov"),
        ],
        "groups": [],
        "packages": [_target_row("pkg-1", "original-title-abcd-1234", "data-pa-gov")],
    }

    populate_fairstore._pin_target_names(source, target, site_id="data-pa-gov")

    names = {row["id"]: row["name"] for row in source["organizations"]}
    assert names["org-old"] == "health"
    assert names["org-new"] == "health-org-new"
    assert source["organizations"][2]["groups"] == [{"name": "health"}]
    assert source["packages"][0]["name"] == "original-title-abcd-1234"
    # Pinned names pass the preflight as unchanged records.
    assert (
        populate_fairstore._assert_no_collisions(
            source, target, store_name="data-pa-gov", store_id="root", site_id="data-pa-gov"
        )
        == []
    )


# -- retried writes cannot duplicate or fake success ---------------------------


@pytest.mark.asyncio
async def test_an_insert_that_committed_before_its_502_is_rebuilt_not_doubled():
    datastore = _datastore()
    lost = iter([True])

    def upsert_then_lose_the_response(payload):
        datastore["datastore_upsert"](payload)
        return _http_error("datastore_upsert") if next(lost, False) else {}

    target = FakeCKAN({**datastore, "datastore_upsert": upsert_then_lose_the_response})
    records = [{"n": index} for index in range(2500)]

    await populate_fairstore._load_table(target, "table", [{"id": "n"}], records, label="t")

    assert datastore["_rows"]["table"] == 2500
    assert [action for action, _ in target.calls].count("datastore_delete") == 2


@pytest.mark.asyncio
async def test_a_failed_view_listing_never_adds_a_second_table_view():
    target = FakeCKAN({"resource_view_list": _http_error("resource_view_list")})

    with pytest.raises(MirrorError):
        await populate_fairstore._ensure_table_view(target, "table", label="t")
    assert "resource_view_create" not in [action for action, _ in target.calls]


@pytest.mark.asyncio
async def test_a_view_create_whose_response_was_lost_is_not_repeated():
    views = []

    def create(payload):
        views.append({"view_type": payload["view_type"]})
        return _http_error("resource_view_create")

    target = FakeCKAN(
        {"resource_view_list": lambda _payload: list(views), "resource_view_create": create}
    )

    await populate_fairstore._ensure_table_view(target, "table", label="t")

    assert len(views) == 1


@pytest.mark.asyncio
async def test_a_create_that_finds_an_admin_deleted_object_is_not_counted_as_created():
    client = FakeCKAN(
        {
            "package_create": _http_error("package_create", 409),
            "package_show": {"id": "pkg", "name": "pkg", "state": "deleted"},
        }
    )

    with pytest.raises(populate_fairstore.DeletedInTarget):
        await populate_fairstore._write_action(
            client, "package_create", {"id": "pkg", "name": "pkg"}, label="pkg"
        )


@pytest.mark.asyncio
async def test_a_rejected_write_is_not_retried():
    client = FakeCKAN({"package_patch": _http_error("package_patch", 409)})

    with pytest.raises(MirrorError, match="package_patch failed"):
        await populate_fairstore._write_action(client, "package_patch", {"id": "pkg"}, label="pkg")
    assert len(client.calls) == 1


# -- reporting and run control ------------------------------------------------


def test_records_withdrawn_at_the_source_are_reported():
    source = {
        "packages": [{"id": "kept", "resources": [{"id": "r-kept"}]}],
    }
    kept = _target_row("kept", "kept", "wprdc")
    kept["resources"] = [
        {"id": "r-kept", "mirror_source_portal": "wprdc"},
        {"id": "r-gone", "mirror_source_portal": "wprdc"},
        {"id": "qsv", "qsv_artifact": "stats"},
    ]
    target = {
        "packages": [
            kept,
            _target_row("gone", "gone-dataset", "wprdc"),
            _target_row("other", "other-portal", "data-pa-gov"),
        ]
    }

    report = populate_fairstore._withdrawn(source, target, site_id="wprdc")

    assert report == {"datasets": 1, "dataset_names": ["gone-dataset"], "resources": 1}


@pytest.mark.asyncio
async def test_all_sites_continue_past_a_failing_portal_and_skip_the_target(monkeypatch):
    sites = [
        {"id": "broken", "url": "https://broken.example"},
        {"id": "fine", "url": "https://fine.example"},
        {"id": "store", "url": "https://www.store.example/"},
    ]
    monkeypatch.setattr(populate_fairstore, "list_sites", lambda: sites)

    async def mirror_site(site_id, _site, _args, _api_key):
        if site_id == "broken":
            raise MirrorError("source down")
        return {"site": site_id}

    monkeypatch.setattr(populate_fairstore, "_mirror_site", mirror_site)
    args = populate_fairstore.argparse.Namespace(
        site="all",
        organization=None,
        index_path=None,
        api_key_env="FAIRSTORE_API_KEY",
        api_key_file=None,
        apply=False,
        target_url="https://store.example",
    )

    summaries = await populate_fairstore._main(args)

    assert summaries == [
        {"site": "broken", "error": "MirrorError: source down"},
        {"site": "fine"},
        {"site": "store", "skipped": "this portal is the target"},
    ]


@pytest.mark.parametrize(
    "footer",
    [
        "\n\nAttribution: Generated by qsv v20.0.0 describegpt\nCommand line: qsv describegpt x",
        "\n\nGenerated by qsv v20.0.0 describegpt\nCommand line: qsv describegpt x",
        "\n\n## Attribution\nGenerated by qsv v20.0.0 describegpt\nModel: m",
        "\n\n---\nGenerated by qsv v20.0.0 describegpt\nModel: m",
        "\n\n<sub>Generated by qsv v20.0.0 describegpt\nModel: m</sub>",
        "\n\n@attribution Generated by qsv v20.0.0 describegpt\nModel: m",
        "\n\n**Attribution**\n\nCommand line: qsv describegpt data.csv --all",
    ],
)
def test_describegpt_prose_drops_every_provenance_style(footer):
    prose = "Parking by zone.\n\n## Notable Characteristics\n\n* **Zones:** 26 of them."
    assert populate_fairstore._describegpt_prose(prose + footer) == prose


def test_describegpt_prose_leaves_a_description_without_provenance_alone():
    assert populate_fairstore._describegpt_prose("Just prose.\n") == "Just prose."


# -- the client tells failures apart from empty results -----------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "transient", "not_found"),
    [
        (502, "<html>Bad Gateway</html>", True, False),
        (
            404,
            {"success": False, "error": {"__type": "Not Found Error", "message": "x"}},
            False,
            True,
        ),
        (
            409,
            {"success": False, "error": {"__type": "Validation Error", "name": ["taken"]}},
            False,
            False,
        ),
    ],
)
async def test_ckan_client_call_raises_classified_errors(status, body, transient, not_found):
    import httpx

    from data_concierge.data_layer.connectors.ckan import CKANClient

    def respond(request):
        if isinstance(body, dict):
            return httpx.Response(status, json=body)
        return httpx.Response(status, text=body)

    client = CKANClient("https://ckan.example", use_default_key=False)
    client._client = httpx.AsyncClient(
        base_url="https://ckan.example", transport=httpx.MockTransport(respond)
    )
    try:
        with pytest.raises(CKANActionError) as caught:
            await client.call("package_show", {"id": "x"})
        assert caught.value.transient is transient
        assert caught.value.not_found is not_found
        # The lenient wrapper keeps its contract for the app's callers.
        assert await client.action("package_show", {"id": "x"}) == {}
    finally:
        await client.close()


# -- qsv output is published only for the file it describes -------------------


def _profile_on_disk(
    tmp_path, *, header, stats_fields, frequency_fields, status="qsv_failed", described=None
):
    directory = tmp_path / "data" / "ckan_onboard" / "wprdc" / "blotter"
    directory.mkdir(parents=True)
    (directory / "historical.csv").write_text(",".join(header) + "\n")
    (directory / "qsv_stats.csv").write_text(
        "field,type\n" + "".join(f"{name},String\n" for name in stats_fields)
    )
    (directory / "qsv_frequency.csv").write_text(
        "field,value,count,percentage,rank\n"
        + "".join(f"{name},x,1,100,1\n" for name in frequency_fields)
    )
    fields = [{"name": name} for name in (described if described is not None else header)]
    (directory / "qsv_dict.json").write_text(
        populate_fairstore.json.dumps(
            {
                "Dictionary": {"response": {"fields": [{"name": "_id"}, *fields]}},
                "Tags": {"response": {"Attribution": "x", "Tags": ["crime"]}},
            }
        )
    )

    return {
        "resource_id": "55555555-5555-4555-8555-555555555555",
        "resource_name": "Historical Blotter Data",
        "local_path": str(directory / "historical.csv"),
        "status": status,
        "qsv_tags": [],
        "columns": [
            {"name": "ccr", "stats": {"cardinality": "9"}, "top_values": [{"value": "1"}]},
            # Contributed only by the other file's stats.
            {"name": "COUNCIL_DISTRICT", "stats": {"cardinality": "9"}},
            # The portal's own DataStore field, absent from the download.
            {"name": "_geom", "ckan_type": "text"},
        ],
    }


def test_another_files_stats_and_frequency_are_not_published_under_this_files_name(tmp_path):
    profile = _profile_on_disk(
        tmp_path,
        header=["ccr", "offense"],
        stats_fields=["ccr", "offense", "COUNCIL_DISTRICT"],
        frequency_fields=["ccr", "COUNCIL_DISTRICT"],
    )

    kinds = [artifact["kind"] for artifact in populate_fairstore._qsv_artifacts(profile)]
    assert kinds == ["dictionary", "describegpt"]

    vetted = populate_fairstore._vet_profile(profile)
    assert [column["name"] for column in vetted["columns"]] == ["ccr", "_geom"]
    assert all(not column["stats"] for column in vetted["columns"])
    # The profiled file's columns, and describegpt's capitalised Tags key.
    extras = populate_fairstore._qsv_dataset_extras(profile)
    assert extras["qsv_columns"] == '["ccr", "offense"]'
    assert extras["qsv_column_count"] == "2"
    assert extras["qsv_tags"] == '["crime"]'


def test_matching_output_is_published_even_for_a_failed_describegpt_run(tmp_path):
    profile = _profile_on_disk(
        tmp_path,
        header=["ccr", "offense"],
        stats_fields=["ccr", "offense"],
        frequency_fields=["offense"],
    )

    kinds = [artifact["kind"] for artifact in populate_fairstore._qsv_artifacts(profile)]
    assert kinds == ["dictionary", "stats", "frequency", "describegpt"]


def test_extract_qsv_tags_reads_a_capitalised_tags_key():
    from data_concierge.data_layer.qsv_profiling import _extract_qsv_tags

    data = {"Tags": {"response": {"Attribution": "Generated by qsv", "Tags": ["a", "b"]}}}
    assert _extract_qsv_tags(data) == ["a", "b"]
    assert _extract_qsv_tags({"tags": {"response": {"tags": ["c"]}}}) == ["c"]


# -- reruns skip what has not changed -----------------------------------------


@pytest.mark.asyncio
async def test_a_rerun_leaves_unchanged_qsv_resources_alone_and_removes_stale_ones(tmp_path):
    profile = _qsv_profile(tmp_path)
    first = FakeCKAN(
        {
            "resource_create": lambda payload: payload,
            "resource_view_create": lambda payload: payload,
            **_datastore(),
        }
    )
    await populate_fairstore._publish_qsv(
        first, package_id="pkg", qsv=profile, site_id="wprdc", existing_resources=set()
    )
    # What the Fair Store holds after the first run, as a snapshot reports it:
    # each create, then the digest patched in once a table's rows are loaded.
    held = {}
    for action, payload in first.calls:
        if action == "resource_create":
            held[payload["id"]] = {
                **{k: v for k, v in payload.items() if k != "_upload"},
                "datastore_active": "_upload" not in payload,
                "url_type": "upload" if "_upload" in payload else "datastore",
            }
        elif action == "resource_patch":
            held[payload["id"]].update(payload)
    assert all(resource["qsv_digest"] for resource in held.values())
    # A kind this profile no longer produces, left by an earlier run.
    stale = populate_fairstore._qsv_resource_id(profile["resource_id"], "frequency")
    (Path(profile["local_path"]).parent / "qsv_frequency.csv").unlink()

    rerun = FakeCKAN({"resource_view_list": [{"view_type": "datatables_view"}], **_datastore()})
    counts = await populate_fairstore._publish_qsv(
        rerun, package_id="pkg", qsv=profile, site_id="wprdc", existing_resources=held
    )

    assert counts == {"created": 0, "refreshed": 0, "unchanged": 3, "removed": 1}
    writes = [
        (action, payload) for action, payload in rerun.calls if action != "resource_view_list"
    ]
    assert writes == [("resource_delete", {"id": stale})]


@pytest.mark.asyncio
async def test_a_changed_qsv_table_is_refreshed(tmp_path):
    profile = _qsv_profile(tmp_path)
    dictionary = populate_fairstore._qsv_resource_id(profile["resource_id"], "dictionary")
    held = {dictionary: {"id": dictionary, "qsv_digest": "old", "datastore_active": True}}
    target = FakeCKAN(
        {
            "resource_patch": lambda payload: payload,
            "resource_create": lambda payload: payload,
            "resource_view_create": lambda payload: payload,
            **_datastore(),
        }
    )

    counts = await populate_fairstore._publish_qsv(
        target, package_id="pkg", qsv=profile, site_id="wprdc", existing_resources=held
    )

    assert counts["refreshed"] == 1 and counts["created"] == 3
    assert ("resource_patch", dictionary) in [(a, p.get("id")) for a, p in target.calls]


@pytest.mark.asyncio
async def test_an_unchanged_dataset_is_not_patched_and_a_dry_run_plans_it(tmp_path):
    source, _organization, package = _source_client()

    def target_with(held_package):
        return FakeCKAN(
            {
                "organization_list": [],
                "group_list": [],
                "package_search": {"count": 1, "results": [held_package]},
                "organization_create": lambda payload: payload,
                "organization_patch": lambda payload: payload,
                "package_patch": lambda payload: payload,
                "resource_patch": lambda payload: payload,
            }
        )

    run = {
        "source": source,
        "site_id": "source-store",
        "site_title": "Source Store",
        "source_url": "https://source.example",
    }
    # A first dry run against an empty-ish target computes the digests.
    probe = target_with({"id": "other", "name": "other", "resources": []})
    plan = await populate_fairstore.mirror_catalog(target=probe, apply=False, **run)
    assert plan["created"] >= 2 and not [a for a, _ in probe.calls if a.endswith("_create")]

    # Hold the dataset exactly as the mirror would write it.
    payload = populate_fairstore._package_payload(
        package,
        site_id="source-store",
        source_url="https://source.example",
        store_id=str(
            populate_fairstore.uuid.uuid5(
                populate_fairstore.uuid.NAMESPACE_URL, "https://source.example"
            )
        ),
        known_org_ids={package["owner_org"]},
    )
    populate_fairstore._stamp_package(payload)
    resource = populate_fairstore._resource_payload(
        package["resources"][0],
        package_id=package["id"],
        site_id="source-store",
        source_url="https://source.example",
        qsv=None,
    )
    populate_fairstore._stamp_resource(resource)
    target = target_with({**payload, "resources": [resource]})

    summary = await populate_fairstore.mirror_catalog(target=target, apply=True, **run)

    actions = [action for action, _ in target.calls]
    assert "package_patch" not in actions and "resource_patch" not in actions
    assert summary["unchanged"] == 2


def test_a_failed_profile_without_a_description_still_builds_its_payloads():
    """index.json stores qsv_description as null for a failed describegpt run."""
    profile = {"resource_id": "r", "status": "qsv_failed", "qsv_description": None, "columns": []}
    resource = populate_fairstore._resource_payload(
        {"id": "r", "url": "https://source.example/r.csv"},
        package_id="p",
        site_id="wprdc",
        source_url="https://source.example",
        qsv=profile,
    )
    assert resource["qsv_description"] == ""
    assert "qsv_description" not in populate_fairstore._qsv_dataset_extras(profile)


@pytest.mark.asyncio
async def test_a_table_whose_load_was_cut_short_is_reloaded_on_the_next_run(tmp_path):
    profile = _qsv_profile(tmp_path, frequency_rows=2500)
    frequency = populate_fairstore._qsv_resource_id(profile["resource_id"], "frequency")
    held = {}

    def create(payload):
        held[payload["id"]] = {**payload, "datastore_active": True}
        return payload

    def patch(payload):
        held[payload["id"]].update(payload)
        return held[payload["id"]]

    datastore = _datastore()
    target = FakeCKAN(
        {
            "resource_create": create,
            "resource_patch": patch,
            "resource_view_create": lambda payload: payload,
            **datastore,
            # Every insert batch fails: the frequency table stops at 1,000 rows.
            "datastore_upsert": _http_error("datastore_upsert"),
        }
    )
    with pytest.raises(MirrorError):
        await populate_fairstore._publish_qsv(
            target, package_id="pkg", qsv=profile, site_id="wprdc", existing_resources=set()
        )
    assert held[frequency]["qsv_digest"] == ""

    target.responses["datastore_upsert"] = datastore["datastore_upsert"]
    counts = await populate_fairstore._publish_qsv(
        target, package_id="pkg", qsv=profile, site_id="wprdc", existing_resources=held
    )

    assert counts["refreshed"] >= 1
    assert datastore["_rows"][frequency] == 2500
    assert held[frequency]["qsv_digest"]


@pytest.mark.asyncio
async def test_nothing_is_removed_when_the_onboarding_directory_is_missing(tmp_path):
    profile = _qsv_profile(tmp_path)
    profile["local_path"] = str(tmp_path / "elsewhere" / "data.csv")
    held = {
        populate_fairstore._qsv_resource_id(profile["resource_id"], kind): {}
        for kind in ("dictionary", "stats", "frequency", "describegpt")
    }
    target = FakeCKAN(
        {
            "resource_patch": lambda payload: payload,
            "resource_view_create": lambda p: p,
            **_datastore(),
        }
    )

    counts = await populate_fairstore._publish_qsv(
        target, package_id="pkg", qsv=profile, site_id="wprdc", existing_resources=held
    )

    assert counts["removed"] == 0
    assert "resource_delete" not in [action for action, _ in target.calls]


# -- the second review's findings ----------------------------------------------


def test_a_name_reused_by_the_source_evicts_the_withdrawn_record():
    source = {
        "organizations": [],
        "groups": [],
        "packages": [{"id": "new-id", "name": "parking", "resources": []}],
    }
    target = {
        "organizations": [],
        "groups": [],
        "packages": [_target_row("old-id-00000000", "parking", "wprdc")],
    }
    evictions = []

    renames = populate_fairstore._assert_no_collisions(
        source, target, store_name="wprdc", store_id="root", site_id="wprdc", evictions=evictions
    )

    assert evictions == [("dataset", "old-id-00000000", "parking-withdrawn-old-id-0")]
    assert renames == ["dataset 'parking' (withdrawn at source) -> 'parking-withdrawn-old-id-0'"]


def test_a_name_held_by_another_portal_is_still_a_collision():
    source = {"organizations": [], "groups": [], "packages": [{"id": "new", "name": "parking"}]}
    target = {
        "organizations": [],
        "groups": [],
        "packages": [_target_row("old", "parking", "data-pa-gov")],
    }

    with pytest.raises(MirrorError, match="already has a different UUID"):
        populate_fairstore._assert_no_collisions(
            source, target, store_name="wprdc", store_id="root", site_id="wprdc", evictions=[]
        )


def test_row_counts_are_published_only_when_they_are_the_files_own():
    count = populate_fairstore._profile_row_count
    assert count({"row_count": 138, "status": "ok"}) == 138
    assert count({"row_count": 495251, "status": "qsv_failed"}) == 495251
    assert count({"row_count": 0, "status": "qsv_failed"}) is None
    assert count({"row_count": 5000, "status": "ok", "truncated_download": True}) is None


def test_a_field_the_source_cleared_is_sent_as_empty():
    payload = populate_fairstore._package_payload(
        {"id": "p", "name": "p", "title": "P", "notes": "n"},
        site_id="s",
        source_url="https://s.example",
        store_id="root",
        known_org_ids=set(),
    )
    assert payload["license_id"] == "" and payload["maintainer_email"] == ""


@pytest.mark.asyncio
async def test_an_empty_qsv_index_does_not_strip_published_profiles():
    source, _organization, package = _source_client()
    held = {
        **package,
        "resources": [
            *({**resource, "package_id": package["id"]} for resource in package["resources"]),
            {
                "id": "q",
                "package_id": package["id"],
                "qsv_profile_site": "source-store",
                "qsv_profiled_resource_id": "r",
            },
        ],
    }
    target = FakeCKAN(
        {
            "organization_list": [],
            "group_list": [],
            "package_search": {"count": 1, "results": [held]},
        }
    )

    with pytest.raises(MirrorError, match="onboarding index missing"):
        await populate_fairstore.mirror_catalog(
            source=source,
            target=target,
            site_id="source-store",
            site_title="Source Store",
            source_url="https://source.example",
            apply=True,
            qsv_index={},
        )
    assert not [a for a, _ in target.calls if a.endswith(("_create", "_patch", "_update"))]


@pytest.mark.asyncio
async def test_an_organization_deleted_in_the_fair_store_is_skipped_not_fatal():
    source, organization, _package = _source_client()
    target = FakeCKAN(
        {
            "organization_list": [],
            "group_list": [],
            "package_search": {"count": 0, "results": []},
            "organization_create": lambda payload: (
                _http_error("organization_create", 409)
                if payload["id"] == organization["id"]
                else payload
            ),
            "organization_show": {"id": organization["id"], "state": "deleted"},
            "organization_patch": lambda payload: payload,
            "package_create": lambda payload: payload,
        }
    )

    summary = await populate_fairstore.mirror_catalog(
        source=source,
        target=target,
        site_id="source-store",
        site_title="Source Store",
        source_url="https://source.example",
        apply=True,
    )

    assert summary["skipped_deleted_in_target"] == [organization["name"]]
    dataset = next(payload for action, payload in target.calls if action == "package_create")
    # Its dataset hangs off the portal root instead.
    assert dataset["owner_org"] != organization["id"]


@pytest.mark.asyncio
async def test_unchanged_organizations_and_groups_are_not_patched_again():
    source, organization, package = _source_client()
    first = FakeCKAN(
        {
            "organization_list": [],
            "group_list": [],
            "package_search": {"count": 0, "results": []},
            "organization_create": lambda payload: payload,
            "organization_patch": lambda payload: payload,
            "package_create": lambda payload: payload,
        }
    )
    run = {
        "source": source,
        "site_id": "source-store",
        "site_title": "Source Store",
        "source_url": "https://source.example",
        "apply": True,
    }
    await populate_fairstore.mirror_catalog(target=first, **run)
    created = [p for a, p in first.calls if a == "organization_create"]

    rerun = FakeCKAN(
        {
            "organization_list": created,
            "group_list": [],
            "package_search": {"count": 0, "results": []},
            "package_create": lambda payload: payload,
        }
    )
    summary = await populate_fairstore.mirror_catalog(target=rerun, **run)

    assert not [
        a
        for a, _ in rerun.calls
        if a.startswith("organization_") and a not in ("organization_list",)
    ]
    assert summary["unchanged"] >= 2


@pytest.mark.asyncio
async def test_an_organization_the_paged_rows_missed_aborts_the_snapshot():
    client = FakeCKAN({"organization_list": ["a", "b", "c"]})

    with pytest.raises(MirrorError, match="paged 2 of 3"):
        await populate_fairstore._check_complete(
            client, "organization_list", [{"name": "a"}, {"name": "b"}], label="portal"
        )
    # One unpaged call: WPRDC's CDN rejects a names-only listing with limit/offset.
    assert client.calls == [("organization_list", {"sort": "name asc"})]


def test_another_files_describegpt_output_is_not_published(tmp_path):
    """uniform-crime-reporting-data: describegpt described blotter-data-ucr-coded.csv."""
    profile = _profile_on_disk(
        tmp_path,
        header=["ccr", "offense"],
        stats_fields=["ccr", "offense"],
        frequency_fields=["offense"],
        described=["ccr", "offense", "COUNCIL_DISTRICT"],
    )
    profile["qsv_description"] = "Crimes by council district."
    profile["columns"][0]["qsv_label"] = "CCR number"

    kinds = [artifact["kind"] for artifact in populate_fairstore._qsv_artifacts(profile)]
    extras = populate_fairstore._qsv_dataset_extras(profile)
    vetted = populate_fairstore._vet_profile(profile)

    assert kinds == ["dictionary", "stats", "frequency"]
    assert "qsv_description" not in extras and "qsv_tags" not in extras
    assert "qsv_label" not in vetted["columns"][0]
