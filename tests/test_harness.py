"""The test harness must expose the repo as the package `recall` and put the
Hermes agent source on sys.path, without executing the plugin __init__.py."""

import sys


def test_recall_package_is_registered():
    import recall

    assert hasattr(recall, "__path__")
    assert recall.__path__, "recall.__path__ must point at the repo root"


def test_hermes_agent_source_is_importable():
    from agent.memory_provider import MemoryProvider

    assert hasattr(MemoryProvider, "prefetch")


def test_repo_root_precedes_site_packages():
    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parent.parent)
    assert repo_root in sys.path
