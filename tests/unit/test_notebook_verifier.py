"""Tests for notebook verification (issue #131).

The point of this factor is to catch an answer whose notebook does not
actually compute it, so the hallucination case is the one that matters most.
"""

import pytest

from data_concierge.agents.notebook_verifier import (
    ALLOWED_ENV_KEYS,
    extract_numeric_claims,
    verify_notebook,
)
from data_concierge.core.models import ConfidenceScore


def make_nb(*sources: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "source": s,
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            }
            for s in sources
        ],
        "metadata": {
            "kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


class TestNumericClaims:
    def test_extracts_real_figures(self) -> None:
        claims = extract_numeric_claims("The rate was 4.1% across 15,234,567 workers.")
        assert "4.1" in claims
        assert "15,234,567" in claims

    def test_drops_years_and_small_integers(self) -> None:
        """Years and small ints coincide with row counts and axis ticks."""
        claims = extract_numeric_claims("In 2024 there were 3 categories and 12 regions.")
        assert claims == []

    def test_deduplicates(self) -> None:
        assert extract_numeric_claims("4.1 and 4.1 again") == ["4.1"]

    def test_empty_answer(self) -> None:
        assert extract_numeric_claims("") == []


class TestVerification:
    def test_honest_notebook_scores_high(self) -> None:
        v = verify_notebook(
            make_nb("rate = 4.1\nprint(f'Texas unemployment: {rate}%')"),
            "The unemployment rate in Texas was 4.1% in 2024.",
        )
        assert v.executed
        assert v.reconciliation_ratio == 1.0
        assert v.score == 1.0

    def test_hallucinated_number_is_caught(self) -> None:
        """The notebook runs perfectly but computes a different number.

        This is the case the whole factor exists for: nothing else in the
        pipeline notices, because the answer is fluent and the code is valid.
        """
        v = verify_notebook(
            make_nb("rate = 7.9\nprint(f'Texas unemployment: {rate}%')"),
            "The unemployment rate in Texas was 4.1% in 2024.",
        )
        assert v.executed, "the notebook itself is valid Python"
        assert v.reconciliation_ratio == 0.0
        assert v.reconciled_values == []
        assert v.score == pytest.approx(0.40)

    def test_broken_notebook_scores_zero_not_none(self) -> None:
        """Failing to run is a measurement, not an absence of one."""
        v = verify_notebook(make_nb("import nonexistent_module_xyz"), "The rate was 4.1%.")
        assert not v.executed
        assert v.score == 0.0
        assert v.reason is None
        assert v.execution_error

    def test_no_numeric_claims_falls_back_to_execution(self) -> None:
        v = verify_notebook(
            make_nb("print('methodology summary')"),
            "Unemployment is measured by the CPS household survey.",
        )
        assert v.executed
        assert v.reconciliation_ratio is None
        assert v.score == 0.6

    def test_shell_cells_are_skipped_not_run(self) -> None:
        """`!pip install` is a Colab convenience; we do not run a shell."""
        v = verify_notebook(make_nb("!pip install -q pandas", "print(4.1)"), "The rate was 4.1%.")
        assert v.skipped_shell_cells == 1
        assert v.executed

    def test_empty_notebook_is_unmeasurable(self) -> None:
        v = verify_notebook({"cells": [], "metadata": {}, "nbformat": 4}, "The rate was 4.1%.")
        assert v.score is None
        assert "no executable code" in v.reason

    def test_partial_reconciliation(self) -> None:
        v = verify_notebook(make_nb("print(4.1)"), "The rate was 4.1% across 15,234,567 workers.")
        assert v.reconciliation_ratio == pytest.approx(0.5)
        assert v.score == pytest.approx(0.70)


class TestContainment:
    def test_secrets_are_not_in_the_allowlist(self) -> None:
        """Injected notebook code must not be able to read these."""
        for secret in (
            "GITHUB_TOKEN",
            "ANTHROPIC_API_KEY",
            "AUTH0_CLIENT_SECRET",
            "EVIDENCE_SIGNING_KEY_SEED",
            "PINECONE_API_KEY",
        ):
            assert secret not in ALLOWED_ENV_KEYS

    def test_subprocess_env_is_minimal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from data_concierge.agents import notebook_verifier as nv

        monkeypatch.setenv("GITHUB_TOKEN", "ghp_should_not_leak")
        monkeypatch.setenv("CENSUS_API_KEY", "census-ok")
        env = nv._subprocess_env("/tmp")
        assert "GITHUB_TOKEN" not in env
        assert env["CENSUS_API_KEY"] == "census-ok"


class TestConfidenceIntegration:
    def _pending(self) -> ConfidenceScore:
        return ConfidenceScore.compute_from_signals(
            answer_grounding=0.8,
            data_retrieval_quality=0.7,
            source_metadata_quality=0.9,
            query_answer_alignment=None,
            computation_complexity=0.6,
            unavailable={
                "query_answer_alignment": "not scored yet",
                "notebook_verification": ConfidenceScore.NOTEBOOK_PENDING_REASON,
            },
        )

    def test_pending_is_renormalised_out(self) -> None:
        s = self._pending()
        assert s.notebook_verification is None
        assert s.measured_weight == pytest.approx(0.60)
        assert "still running" in s.unavailable["notebook_verification"]

    def test_verified_notebook_raises_the_score(self) -> None:
        before = self._pending()
        after = before.with_notebook_verification(0.95)
        assert after.notebook_verification == 0.95
        assert after.final_score > before.final_score
        assert after.measured_weight == pytest.approx(0.90)
        assert "notebook_verification" not in after.unavailable

    def test_failed_notebook_drags_the_score_down_a_level(self) -> None:
        before = self._pending()
        after = before.with_notebook_verification(0.0)
        assert after.final_score < before.final_score
        assert before.level.value == "medium"
        assert after.level.value == "low"

    def test_verifier_failure_leaves_the_score_unchanged(self) -> None:
        """An unavailable verdict must not be folded in as a zero."""
        before = self._pending()
        after = before.with_notebook_verification(None, "verifier could not start")
        assert after.notebook_verification is None
        assert after.final_score == pytest.approx(before.final_score)
        assert "could not start" in after.unavailable["notebook_verification"]

    def test_notebook_weight_is_thirty_percent(self) -> None:
        only_nb = ConfidenceScore.compute_from_signals(
            answer_grounding=None,
            data_retrieval_quality=None,
            source_metadata_quality=None,
            query_answer_alignment=None,
            computation_complexity=None,
            notebook_verification=0.5,
            unavailable={"answer_grounding": "x"},
        )
        assert only_nb.measured_weight == pytest.approx(0.30)
        assert only_nb.final_score == pytest.approx(0.5)


class TestFeatureFlag:
    """The flag gates whether the factor appears at all."""

    @staticmethod
    def _score():
        from data_concierge.core.confidence import ConfidenceCalculator
        from data_concierge.core.models import ToolCallSignals

        return ConfidenceCalculator().calculate_from_signals(
            tool_signals=ToolCallSignals(successful_tool_calls=2, total_rows_loaded=100),
            final_answer="The rate was 4.1 percent.",
            tool_results=["rate 4.1"],
        )

    def test_ships_enabled(self) -> None:
        """Verification is ON (#131) — the egress guard makes it safe."""
        from data_concierge.core.config import settings

        assert settings.notebook_verification_enabled is True

    def test_disabled_omits_the_factor_entirely(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When off, no perpetual 'still running' for a check that never runs.

        Both checks must be off — the adversarial review (#131 third signal)
        alone still schedules the pass and fills the factor in later.
        """
        from data_concierge.core import confidence as conf

        monkeypatch.setattr(conf.settings, "notebook_verification_enabled", False)
        monkeypatch.setattr(conf.settings, "notebook_review_enabled", False)
        result = self._score()
        assert "notebook_verification" not in result.unavailable
        assert result.notebook_verification is None

    def test_enabled_marks_it_pending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from data_concierge.core import confidence as conf

        monkeypatch.setattr(conf.settings, "notebook_verification_enabled", True)
        result = self._score()
        assert "still running" in result.unavailable["notebook_verification"]


class TestEgressGuard:
    """The runtime containment that makes executing notebooks safe (#135).

    A generated notebook is LLM-written and portal text reaches the
    generator, so it is untrusted. The guard must stop it reaching the cloud
    metadata endpoint (service-account token theft) or internal services,
    while leaving the public data APIs usable.
    """

    @staticmethod
    def _run(code: str) -> str:
        import subprocess
        import sys
        import tempfile

        from data_concierge.agents.notebook_verifier import (
            _install_egress_guard,
            _subprocess_env,
        )

        with tempfile.TemporaryDirectory() as d:
            _install_egress_guard(d)
            r = subprocess.run(
                [sys.executable, "-c", code],
                env=_subprocess_env(d),
                cwd=d,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return r.stdout

    def test_metadata_endpoint_is_blocked(self) -> None:
        out = self._run(
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('169.254.169.254',80),timeout=3)\n"
            "    print('NOT BLOCKED')\n"
            "except OSError as e:\n"
            "    print('BLOCKED', e)\n"
        )
        assert "NOT BLOCKED" not in out
        assert "blocked connection to internal address 169.254.169.254" in out

    def test_loopback_is_blocked(self) -> None:
        out = self._run(
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('127.0.0.1',5001),timeout=3)\n"
            "    print('NOT BLOCKED')\n"
            "except OSError as e:\n"
            "    print('BLOCKED', e)\n"
        )
        assert "NOT BLOCKED" not in out
        assert "blocked connection to internal address" in out

    def test_private_range_is_blocked(self) -> None:
        out = self._run(
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('10.0.0.5',80),timeout=3)\n"
            "    print('NOT BLOCKED')\n"
            "except OSError as e:\n"
            "    print('BLOCKED', e)\n"
        )
        assert "NOT BLOCKED" not in out

    def test_guard_does_not_break_pure_computation(self) -> None:
        out = self._run("print(sum(range(10)))")
        assert "45" in out


class TestScheduling:
    """The async trigger (#131): when it fires, and what it stores."""

    @staticmethod
    def _confidence() -> ConfidenceScore:
        return ConfidenceScore.compute_from_signals(
            answer_grounding=0.8,
            data_retrieval_quality=0.7,
            source_metadata_quality=0.9,
            query_answer_alignment=None,
            computation_complexity=0.6,
            unavailable={
                "query_answer_alignment": "not scored yet",
                "notebook_verification": ConfidenceScore.NOTEBOOK_PENDING_REASON,
            },
        )

    async def test_disabled_schedules_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from data_concierge.gateway import notebook_verification as nv

        monkeypatch.setattr(nv.settings, "notebook_verification_enabled", False)
        monkeypatch.setattr(nv.settings, "notebook_review_enabled", False)
        assert (
            nv.schedule_verification("q1", make_nb("print(1)"), "answer", self._confidence())
            is False
        )

    async def test_no_notebook_schedules_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from data_concierge.gateway import notebook_verification as nv

        monkeypatch.setattr(nv.settings, "notebook_verification_enabled", True)
        assert nv.schedule_verification("q1", None, "answer", self._confidence()) is False
        assert nv.schedule_verification("q2", {"cells": []}, "a", self._confidence()) is False

    async def test_end_to_end_updates_the_stored_confidence(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Schedule, wait for the task, and read back the improved score."""
        import asyncio

        from data_concierge.data_layer import storage as storage_mod
        from data_concierge.gateway import notebook_verification as nv

        monkeypatch.setattr(nv.settings, "notebook_verification_enabled", True)
        # Keep the adversarial review (#131 third signal) out of unit tests —
        # it would call the live API with any configured key.
        monkeypatch.setattr(nv.settings, "notebook_review_enabled", False)
        monkeypatch.setattr(storage_mod.storage, "root", tmp_path, raising=False)
        monkeypatch.setattr(nv, "storage", storage_mod.LocalStorage(tmp_path))

        before = self._confidence()
        scheduled = nv.schedule_verification(
            "q-e2e",
            make_nb("print(4.1)"),
            "The rate was 4.1 percent.",
            before,
        )
        assert scheduled is True

        # Let the background task finish.
        for _ in range(200):
            await asyncio.sleep(0.1)
            if not nv._in_flight:
                break

        stored = nv.get_verification("q-e2e")
        assert stored is not None
        assert stored["status"] == "complete"
        assert stored["verdict"]["executed"] is True
        assert stored["verdict"]["reconciliation_ratio"] == 1.0
        # The notebook re-derived the claim, so confidence rose and the
        # pending reason is gone.
        assert stored["confidence"] > before.final_score
        assert "notebook_verification" not in stored["confidence_unavailable"]
        assert stored["confidence_breakdown"]["notebook_verification"] == 1.0

    async def test_failed_notebook_lowers_the_stored_confidence(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import asyncio

        from data_concierge.data_layer import storage as storage_mod
        from data_concierge.gateway import notebook_verification as nv

        monkeypatch.setattr(nv.settings, "notebook_verification_enabled", True)
        monkeypatch.setattr(nv.settings, "notebook_review_enabled", False)
        monkeypatch.setattr(nv, "storage", storage_mod.LocalStorage(tmp_path))

        before = self._confidence()
        nv.schedule_verification(
            "q-bad", make_nb("import nonexistent_module_xyz"), "The rate was 4.1 percent.", before
        )
        for _ in range(200):
            await asyncio.sleep(0.1)
            if not nv._in_flight:
                break

        stored = nv.get_verification("q-bad")
        assert stored["status"] == "complete"
        assert stored["verdict"]["executed"] is False
        assert stored["confidence"] < before.final_score


class TestListFormSource:
    """nbformat JSON stores cell source as a LIST of lines (issue found live).

    Every production verification failed with "'list' object has no attribute
    'strip'" before a single cell ran, so correct answers were being scored
    0.0 and their confidence dragged down. The real-world shape must work.
    """

    @staticmethod
    def _nb_listform(*sources: str) -> dict:
        return {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [ln + "\n" for ln in s.split("\n")],
                    "metadata": {},
                    "outputs": [],
                    "execution_count": None,
                }
                for s in sources
            ],
            "metadata": {
                "kernelspec": {
                    "name": "python3",
                    "language": "python",
                    "display_name": "Python 3",
                },
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    def test_list_source_executes_and_reconciles(self) -> None:
        v = verify_notebook(
            self._nb_listform("rate = 4.1\nprint(f'rate: {rate}%')"),
            "The rate was 4.1 percent.",
        )
        assert v.executed, f"list-form notebook failed: {v.execution_error}"
        assert v.reconciliation_ratio == 1.0
        assert v.score == 1.0

    def test_list_source_shell_cell_still_skipped(self) -> None:
        v = verify_notebook(
            self._nb_listform("!pip install -q pandas", "print(4.1)"),
            "The rate was 4.1 percent.",
        )
        assert v.skipped_shell_cells == 1
        assert v.executed

    def test_markdown_cells_with_list_source_are_normalised(self) -> None:
        nb = self._nb_listform("print(7.5)")
        nb["cells"].insert(
            0, {"cell_type": "markdown", "source": ["# Title\n", "text\n"], "metadata": {}}
        )
        v = verify_notebook(nb, "The value was 7.5.")
        assert v.executed, f"markdown list-source broke execution: {v.execution_error}"


class TestHarnessFaultIsUnmeasurable:
    """A verifier fault must not be reported as 'the notebook is broken'."""

    def test_unparseable_notebook_is_unmeasurable_not_zero(self) -> None:
        v = verify_notebook({"cells": [], "metadata": {}, "nbformat": 4}, "The rate was 4.1%.")
        assert v.score is None, "no executable code is unmeasurable, not a zero"
        assert v.reason
