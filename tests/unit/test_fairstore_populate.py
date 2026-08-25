"""Tests for the Fair Store ingestion helpers (issue #133)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "populate_fairstore",
    Path(__file__).resolve().parent.parent.parent / "scripts" / "populate_fairstore.py",
)
populate_fairstore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(populate_fairstore)

_slug = populate_fairstore._slug
_column_dictionary = populate_fairstore._column_dictionary


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
