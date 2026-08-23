"""Test harness for the hermes-recall plugin.

The repo root IS the plugin directory that Hermes clones into
``$HERMES_HOME/plugins/recall/``, and Hermes loads it as a package named
``recall``. Tests must import it the same way, so this conftest registers a
synthetic ``recall`` package whose ``__path__`` is the repo root. That makes
``import recall._client`` work through normal submodule machinery WITHOUT
executing the root ``__init__.py`` (which pulls in the Hermes agent runtime).

Tests that need the real ``__init__.py`` body — ``register(ctx)`` and the
re-exports — ask for the ``recall_package`` fixture, which executes it.

``HERMES_AGENT_SRC`` overrides where the Hermes agent source lives.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HERMES_AGENT_SRC = Path(
    os.environ.get("HERMES_AGENT_SRC", str(Path.home() / ".hermes" / "hermes-agent"))
)

for _path in (str(REPO_ROOT), str(HERMES_AGENT_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _register_recall_namespace() -> types.ModuleType:
    """Expose the repo root as package ``recall`` without running __init__.py."""
    existing = sys.modules.get("recall")
    if existing is not None:
        return existing
    package = types.ModuleType("recall")
    package.__path__ = [str(REPO_ROOT)]
    package.__package__ = "recall"
    sys.modules["recall"] = package
    return package


def _neutralize_root_package_collection() -> None:
    """Stop pytest from executing the repo-root ``__init__.py`` as a package.

    The repo root IS the plugin package, so once ``__init__.py`` exists pytest
    collects the rootdir as a ``Package`` node whose ``setup()`` imports that
    file under the top-level module name ``__init__`` — where the relative
    imports (``from ._client import …``) have no parent package, so every test
    errors at setup. It would also drag the Hermes agent runtime into every
    session, which is exactly what this harness avoids.

    Seeding ``sys.modules["__init__"]`` with a stub carrying the right
    ``__file__`` makes pytest's ``import_path`` return the cached module and
    skip execution (and satisfies its ImportPathMismatch check). The real
    ``__init__.py`` is executed on demand, under its real name, by the
    ``recall_package`` fixture below.
    """
    if "__init__" in sys.modules:
        return
    stub = types.ModuleType("__init__")
    stub.__file__ = str(REPO_ROOT / "__init__.py")
    stub.__path__ = [str(REPO_ROOT)]
    sys.modules["__init__"] = stub


_register_recall_namespace()
_neutralize_root_package_collection()


@pytest.fixture(autouse=True)
def _clear_recall_env(monkeypatch):
    """Every test starts with no ambient Recall env vars.

    Without this, a shell exporting RECALL_API_KEY / RECALL_BASE_URL (as this
    dev machine's global RAG MCP workflow does) leaks into RecallClient's
    env-fallback constructor and makes env-fallback tests nondeterministic.
    Tests that want to exercise the fallback set the vars explicitly via
    monkeypatch.setenv.
    """
    monkeypatch.delenv("RECALL_API_KEY", raising=False)
    monkeypatch.delenv("RECALL_BASE_URL", raising=False)


@pytest.fixture()
def recall_package():
    """Execute the real root ``__init__.py`` as the package ``recall``."""
    spec = importlib.util.spec_from_file_location(
        "recall",
        str(REPO_ROOT / "__init__.py"),
        submodule_search_locations=[str(REPO_ROOT)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["recall"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("recall", None)
        _register_recall_namespace()
