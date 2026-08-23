"""Keep pytest from importing this repo's root ``__init__.py`` as a package.

The repo root IS the plugin package Hermes loads, so it carries an
``__init__.py`` whose relative imports (``from ._client import …``) only
resolve under a real parent package. pytest's default
``pytest_collect_directory`` turns any directory holding an ``__init__.py``
into a :class:`pytest.Package`, and ``Package.setup()`` imports that file under
the top-level module name ``__init__`` — where the relative imports have no
parent, so every test in the repo errors at setup. It would also pull the
Hermes agent runtime into every session, which ``tests/conftest.py`` exists to
avoid. Returning a plain :class:`pytest.Dir` for this one directory keeps
collection working exactly as it did before the plugin gained an
``__init__.py``.

The hook cannot be a plain conftest hook: ``Session._collect_path`` resolves it
with ``gethookproxy(path.parent)`` (``_pytest/main.py:762``), so a directory's
own conftest is never consulted about collecting *itself* — only its parent's
is, and the repo root's parent is outside the project. Registering the hook as
a session plugin instead sidesteps that: ``gethookproxy`` filters out
inapplicable *conftest* plugins only, so a plugin registered here participates
for every path.

The real module is executed on demand, under its real name, by the
``recall_package`` fixture in ``tests/conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent


class _CollectRepoRootAsPlainDir:
    """Session plugin holding the one hook above; see the module docstring."""

    @staticmethod
    def pytest_collect_directory(path: Path, parent) -> pytest.Dir | None:
        if path == REPO_ROOT:
            return pytest.Dir.from_parent(parent, path=path)
        return None


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(
        _CollectRepoRootAsPlainDir(), "hermes-recall-root-is-not-a-test-package"
    )
