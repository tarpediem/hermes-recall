"""Provider identity, availability, config resolution, and initialize()."""

import json

import pytest
from recall._client import RecallAuthError, RecallClient, RecallError
from recall._provider import (
    DEFAULTS,
    RecallMemoryProvider,
    load_recall_config,
    recall_config_path,
)


@pytest.fixture()
def provider():
    return RecallMemoryProvider(RecallClient(api_key="rag_k", base_url="https://recall.example"))


def test_provider_name_is_recall(provider):
    assert provider.name == "recall"


def test_is_available_requires_an_api_key_and_makes_no_network_call(monkeypatch):
    from recall import _client

    class Exploding:
        def get(self, *a, **k):
            raise AssertionError("is_available must not touch the network")

        def post(self, *a, **k):
            raise AssertionError("is_available must not touch the network")

    monkeypatch.setattr(_client, "requests", Exploding())

    assert RecallMemoryProvider(RecallClient(api_key="rag_k")).is_available() is True
    assert RecallMemoryProvider(RecallClient(api_key="")).is_available() is False


def test_unavailable_reason_is_actionable(provider):
    assert "RECALL_API_KEY" in provider.unavailable_reason()


def test_backup_paths_is_empty(provider):
    assert provider.backup_paths() == []


def test_config_schema_declares_exactly_one_required_secret(provider):
    schema = provider.get_config_schema()

    assert len(schema) == 1
    field = schema[0]
    assert field["key"] == "api_key"
    assert field["secret"] is True
    assert field["required"] is True
    assert field["env_var"] == "RECALL_API_KEY"
    assert field["url"] == "https://recall.carnival-devops.com"


def test_load_recall_config_returns_defaults_when_no_file(tmp_path):
    config = load_recall_config(str(tmp_path))

    assert config["limit"] == 5
    assert config["rerank"] is True
    assert config["graph_boost"] is False
    assert config["sync_turns"] is True
    assert config["session_summary"] is True
    assert config["max_chars"] == 4000
    assert config["min_chars"] == 40
    assert config["base_url"] == DEFAULTS["base_url"]


def test_load_recall_config_merges_the_json_file(tmp_path, write_recall_config):
    write_recall_config(tmp_path, {"limit": 9, "sync_turns": False})

    config = load_recall_config(str(tmp_path))

    assert config["limit"] == 9
    assert config["sync_turns"] is False
    assert config["max_chars"] == 4000  # untouched default


def test_load_recall_config_survives_a_corrupt_file(tmp_path, write_recall_config):
    write_recall_config(tmp_path, "{ not json")

    assert load_recall_config(str(tmp_path))["limit"] == 5


def test_env_base_url_wins_over_the_file(tmp_path, monkeypatch, write_recall_config):
    write_recall_config(tmp_path, {"base_url": "https://from-file"})
    monkeypatch.setenv("RECALL_BASE_URL", "https://from-env")

    assert load_recall_config(str(tmp_path))["base_url"] == "https://from-env"


def test_initialize_captures_the_runtime_context(provider, tmp_path):
    provider.initialize(
        "sess-1",
        hermes_home=str(tmp_path),
        platform="telegram",
        agent_identity="charlotte",
        agent_context="primary",
    )

    assert provider._session_id == "sess-1"
    assert provider._hermes_home == str(tmp_path)
    assert provider._platform == "telegram"
    assert provider._agent_identity == "charlotte"
    assert provider._agent_context == "primary"
    assert provider._writes_enabled is True


def test_initialize_disables_writes_for_non_primary_contexts(provider, tmp_path):
    for context in ("subagent", "cron", "flush"):
        provider.initialize("s", hermes_home=str(tmp_path), agent_context=context)
        assert provider._writes_enabled is False, context


def test_initialize_applies_the_configured_base_url_to_the_client(
    provider, tmp_path, write_recall_config
):
    write_recall_config(tmp_path, {"base_url": "https://lan.example/"})

    provider.initialize("s", hermes_home=str(tmp_path))

    assert provider._client.base_url == "https://lan.example"


def test_initialize_never_raises_on_a_broken_hermes_home(provider):
    provider.initialize("s", hermes_home="/nonexistent/path/that/does/not/exist")
    assert provider._config["limit"] == 5


def test_tags_include_session_platform_and_agent(provider, tmp_path):
    provider.initialize(
        "sess-1", hermes_home=str(tmp_path), platform="cli", agent_identity="coder"
    )

    assert provider._tags("delegation") == [
        "hermes",
        "session:sess-1",
        "platform:cli",
        "agent:coder",
        "delegation",
    ]


def test_save_config_writes_only_non_secret_values(provider, tmp_path):
    provider.save_config({"limit": 3, "api_key": "rag_leak", "sync_turns": False}, str(tmp_path))

    path = recall_config_path(str(tmp_path))
    assert path == tmp_path / "recall" / "config.json"
    written = json.loads(path.read_text())
    assert written == {"limit": 3, "sync_turns": False}


def test_auth_failure_is_logged_once_per_session(provider, tmp_path, caplog):
    provider.initialize("s", hermes_home=str(tmp_path))

    with caplog.at_level("WARNING"):
        provider._log_failure("search", RecallAuthError("rejected"))
        provider._log_failure("store", RecallAuthError("rejected"))

    assert sum("API key rejected" in r.message for r in caplog.records) == 1


def test_other_failures_are_logged_every_time(provider, tmp_path, caplog):
    provider.initialize("s", hermes_home=str(tmp_path))

    with caplog.at_level("WARNING"):
        provider._log_failure("search", RecallError("boom"))
        provider._log_failure("search", RecallError("boom"))

    assert sum("Recall search failed" in r.message for r in caplog.records) == 2


def test_system_prompt_block_mentions_recall(provider, tmp_path):
    provider.initialize("s", hermes_home=str(tmp_path))
    assert "Recall" in provider.system_prompt_block()
