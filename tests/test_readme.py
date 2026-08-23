"""The README must stay in sync with the code's actual defaults and names."""

from pathlib import Path

from recall._provider import DEFAULTS

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"


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


def test_readme_has_no_python_dependencies():
    text = README.read_text()

    assert "python_dependencies" not in text
