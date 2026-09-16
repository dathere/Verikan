"""Shared pytest configuration.

Issue #140. The storage backend defaults to ``LocalStorage(_PROJECT_ROOT)``,
so a test run wrote through the real storage layer into the repository —
integration tests that submit a notebook or an answer appended their fixtures
to the tracked ``verified_notebooks/index.json``. A run added, for example, a
"The unemployment rate in Texas is 4.1% (2024)." entry to the real verified
library, and anyone committing with ``git add -A`` would have published it.

The root is repointed at a temp directory for the session. It is mutated on
the existing singleton rather than swapped out, because modules bind the
object directly (``from ...storage import storage``) at import time, so
rebinding the module attribute would leave those references pointing at the
original.
"""

import os

import pytest

# Set before anything imports `data_concierge.core.config`, which builds its
# `settings` singleton at import time. `.env` feeds every field, so without this
# the suite inherits whatever a developer happens to have configured locally —
# which is how a documented `NOTEBOOK_VERIFICATION_ENABLED=false` turned the
# suite red. Empty means "no env file"; export it to a path to opt back in.
os.environ.setdefault("DATA_CONCIERGE_ENV_FILE", "")


@pytest.fixture(autouse=True, scope="session")
def _isolate_storage_from_the_repo(tmp_path_factory):
    """Redirect all storage writes into a throwaway directory."""
    from data_concierge.data_layer.storage import storage

    root = getattr(storage, "root", None)
    if root is None:
        # GCSStorage has no local root; nothing to isolate.
        yield
        return

    storage.root = tmp_path_factory.mktemp("dc-storage")
    try:
        yield
    finally:
        storage.root = root
