# Task Completion

Run from the repo root (see the cwd trap in `mem:suggested_commands`):

```bash
ruff check src/ tests/ scripts/    # must pass — CI gate
pytest -q --no-cov                 # must pass on 3.12 and 3.14 — CI gate
```

Advisory, run on code you touched but do not chase repo-wide:

```bash
mypy src/                          # ~90 pre-existing errors; just don't add new ones
ruff format --check src/ tests/ scripts/
```

Do **not** run `ruff format` across files the change doesn't already touch — see `mem:conventions`.

## Storage isolation — why a test run is safe, and how to keep it that way
`LocalStorage`'s root is the repo checkout (`mem:core`), so without isolation the suite would write `users.json`, `chats/`, `verified_notebooks/` etc. into the working tree.

`tests/conftest.py::_isolate_storage_from_the_repo` is `autouse=True, scope="session"`: it swaps `storage.root` for a `tmp_path_factory` dir and restores it afterwards. It mutates the already-imported singleton — it cannot re-create it, and it no-ops for `GCSStorage`, which has no `root`.

Consequences for new code:
- Code that resolves storage paths from anything other than `storage.root` (a captured module-level `Path`, `_PROJECT_ROOT`, `Path.cwd()`) escapes the fixture and will write into the checkout during tests. Route file access through the `storage` singleton.
- To check for escaped writes, plain `git status` is **useless**: every runtime-state path is root-anchored in `.gitignore`, so an escaped write lands as *ignored* and the tree still looks clean. Use `git status --ignored --short` and look for the `mem:gateway/core` state paths appearing after a suite run.

## Notebook-verification tests really execute notebooks
They need the venv's own Jupyter kernelspec, which comes from `ipykernel` (a hard dependency). No env vars, no secrets, no network. If they start failing after a dependency change, check the kernelspec before suspecting the code.
