"""Tests for GitHub artifact links surfaced in the admin UI (#111).

Covers:
- ``build_blob_url`` URL construction and graceful fallbacks
- ``get_verified_notebook_by_submission`` lookup helper
"""

from data_concierge.gateway.github_publisher import build_blob_url


class TestBuildBlobUrl:
    def test_builds_url_from_explicit_settings(self):
        url = build_blob_url(
            "drafts/verified/foo.ipynb",
            {"repo": "example-org/example-notebooks", "branch": "main"},
        )
        assert url == (
            "https://github.com/example-org/example-notebooks/blob/main/drafts/verified/foo.ipynb"
        )

    def test_defaults_branch_to_main(self):
        url = build_blob_url("verified/foo.ipynb", {"repo": "owner/repo"})
        assert url == "https://github.com/owner/repo/blob/main/verified/foo.ipynb"

    def test_honors_custom_branch(self):
        url = build_blob_url("verified/foo.ipynb", {"repo": "owner/repo", "branch": "demo"})
        assert url == "https://github.com/owner/repo/blob/demo/verified/foo.ipynb"

    def test_returns_none_when_path_missing(self):
        assert build_blob_url(None, {"repo": "owner/repo"}) is None
        assert build_blob_url("", {"repo": "owner/repo"}) is None

    def test_returns_none_when_repo_missing(self):
        assert build_blob_url("verified/foo.ipynb", {"repo": ""}) is None
        assert build_blob_url("verified/foo.ipynb", {}) is None


class TestGetVerifiedNotebookBySubmission:
    def test_finds_notebook_by_submission_id(self, monkeypatch):
        from data_concierge.gateway import verified_notebooks as vn

        index = {
            "verified": {
                "nb-1": {
                    "notebook_id": "nb-1",
                    "submission_id": "sub-123",
                    "query": "q",
                    "answer": "a",
                    "notebook_json": {"cells": []},
                    "verified_by": "admin",
                    "github_path": "verified/nb-1.ipynb",
                },
            }
        }
        monkeypatch.setattr(vn, "_load_index", lambda: index)

        found = vn.get_verified_notebook_by_submission("sub-123")
        assert found is not None
        assert found.notebook_id == "nb-1"
        assert found.github_path == "verified/nb-1.ipynb"

    def test_returns_none_when_no_match(self, monkeypatch):
        from data_concierge.gateway import verified_notebooks as vn

        monkeypatch.setattr(vn, "_load_index", lambda: {"verified": {}})
        assert vn.get_verified_notebook_by_submission("missing") is None
