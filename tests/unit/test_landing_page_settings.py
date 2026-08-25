"""Tests for the configurable landing-page settings store (#109)."""

import pytest

from data_concierge.gateway import landing_page as lp


@pytest.fixture
def isolated_store(monkeypatch):
    """Back the storage with an in-memory dict so tests don't touch disk."""
    store: dict[str, dict] = {}

    def fake_read(key):
        return store.get(key)

    def fake_write(key, data):
        store[key] = data

    monkeypatch.setattr(lp.storage, "read_json", fake_read)
    monkeypatch.setattr(lp.storage, "write_json", fake_write)
    return store


class TestLoadDefaults:
    def test_returns_defaults_when_empty(self, isolated_store):
        settings = lp.load_landing_settings()
        assert settings["title"] == "Verikan"
        assert settings["tagline"] == "The verified Data Concierge"
        assert settings["show_beta_badge"] is True
        assert len(settings["sample_questions"]) == 5

    def test_defaults_not_mutated_by_caller(self, isolated_store):
        settings = lp.load_landing_settings()
        settings["sample_questions"].append("injected")
        # A fresh load must not see the mutation.
        assert "injected" not in lp.load_landing_settings()["sample_questions"]

    def test_read_error_degrades_to_defaults(self, monkeypatch):
        def boom(key):
            raise RuntimeError("storage down")

        monkeypatch.setattr(lp.storage, "read_json", boom)
        settings = lp.load_landing_settings()
        assert settings["title"] == "Verikan"


class TestSave:
    def test_save_and_reload_roundtrip(self, isolated_store):
        merged = lp.save_landing_settings(
            {
                "title": "City Data Portal",
                "show_beta_badge": False,
                "powered_by_label": "Open Data City",
                "sample_questions": ["Q1?", "Q2?"],
            }
        )
        assert merged["title"] == "City Data Portal"
        assert merged["show_beta_badge"] is False
        # Unspecified fields fall back to defaults.
        assert merged["logo_url"] == lp.DEFAULT_LANDING_SETTINGS["logo_url"]

        reloaded = lp.load_landing_settings()
        assert reloaded["title"] == "City Data Portal"
        assert reloaded["sample_questions"] == ["Q1?", "Q2?"]

    def test_strings_are_trimmed(self, isolated_store):
        merged = lp.save_landing_settings({"title": "  Spaced  "})
        assert merged["title"] == "Spaced"

    def test_sample_questions_drops_blanks(self, isolated_store):
        merged = lp.save_landing_settings(
            {"sample_questions": ["  keep me  ", "", "   ", "second"]}
        )
        assert merged["sample_questions"] == ["keep me", "second"]

    def test_only_known_fields_persisted(self, isolated_store):
        lp.save_landing_settings({"title": "X", "evil_field": "nope"})
        assert "evil_field" not in isolated_store[lp._LANDING_SETTINGS_KEY]

    def test_invalid_sample_questions_type_raises(self, isolated_store):
        with pytest.raises(ValueError):
            lp.save_landing_settings({"sample_questions": "not a list"})

    def test_partial_save_merges_with_prior_values(self, isolated_store):
        lp.save_landing_settings({"title": "First", "powered_by_label": "Portal A"})
        lp.save_landing_settings({"title": "Second"})
        reloaded = lp.load_landing_settings()
        assert reloaded["title"] == "Second"
        # powered_by_label from the first save is preserved across the partial
        # second save (merge-on-save behavior).
        assert reloaded["powered_by_label"] == "Portal A"
