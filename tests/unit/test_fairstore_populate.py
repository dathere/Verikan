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
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def action(self, name, payload=None):
        self.calls.append((name, payload or {}))
        response = self.responses.get(name, {})
        return response(payload or {}) if callable(response) else response


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

    resource = next(payload for action, payload in target.calls if action == "resource_create")
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
            "resource_patch": lambda payload: payload,
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
    assert actions.count("resource_patch") == 1
