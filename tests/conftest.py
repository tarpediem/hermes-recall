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


_register_recall_namespace()


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


@pytest.fixture()
def write_recall_config():
    """Write a config file where the provider (and the dashboard) expects it.

    Centralised so the ``$HERMES_HOME/recall/config.json`` path — which is a
    contract with ``hermes_cli/web_server.py::_flat_json_path``, not a free
    choice — is spelled out in exactly one place across the suite.
    """
    import json

    from recall._provider import recall_config_path

    def _write(hermes_home, payload):
        path = recall_config_path(str(hermes_home))
        path.parent.mkdir(parents=True, exist_ok=True)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        path.write_text(text, encoding="utf-8")
        return path

    return _write
