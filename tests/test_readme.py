"""The README must stay in sync with the code's actual defaults and names."""

import importlib.util
from pathlib import Path

from recall._provider import DEFAULTS

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"


def _load_module_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_exists():
    assert README.is_file()


def test_readme_documents_the_dashboard_install():
    text = README.read_text()

    assert "Install from GitHub" in text
    assert "tarpediem/hermes-recall" in text
    assert "RECALL_API_KEY" in text


def test_readme_documents_every_config_key_and_default():
    text = README.read_text()

    for key, default in DEFAULTS.items():
        assert f"`{key}`" in text, key
        assert f"`{default}`" in text, f"{key}={default}"


def test_readme_documents_both_tools():
    text = README.read_text()

    assert "recall_search" in text
    assert "recall_store" in text


def test_readme_states_what_leaves_the_device():
    text = README.read_text()

    assert "leaves the device" in text.lower()
    assert "tool call" in text.lower()


def test_readme_documents_the_version_matrix():
    text = README.read_text()

    assert "0.20.4" in text
    assert "0.19.1" in text


def test_readme_documents_the_real_config_file_path():
    text = README.read_text()

    assert "recall/config.json" in text
    # The old, wrong filename must never resurface as a bare token.
    assert "recall.json" not in text


def test_readme_documents_the_manual_install_directory_name():
    text = README.read_text()

    assert "$HERMES_HOME/plugins/recall" in text


def test_readme_documents_how_the_live_integration_test_is_enabled():
    """The env vars named here are the ones the live test actually reads."""
    # Loaded by path: a bare ``tests`` import resolves to the Hermes agent's
    # own tests package, not this repo's.
    live = _load_module_by_path(
        "live_test_module", Path(__file__).with_name("test_integration_live.py")
    )

    text = README.read_text()

    assert "RECALL_TEST_API_KEY" in text
    assert "RECALL_TEST_BASE_URL" in text
    assert live.PURGE_TAG in text
    assert live.PURGE_TAG == "hermes-recall-live-test"
    live_source = Path(live.__file__).read_text()
    for var in ("RECALL_TEST_API_KEY", "RECALL_TEST_BASE_URL"):
        assert var in live_source


def test_readme_documents_both_read_budgets():
    from recall._client import READ_TIMEOUT, SLOW_READ_TIMEOUT

    text = README.read_text()

    assert f"{READ_TIMEOUT:.0f} s budget" in text
    assert f"{SLOW_READ_TIMEOUT:.0f} s budget" in text


def test_readme_has_no_python_dependencies():
    text = README.read_text()

    assert "python_dependencies" not in text
