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


def test_auth_failure_is_logged_once_per_session_at_error_level(provider, tmp_path, caplog):
    """A rejected key means memory is silently off for the whole session —
    the one failure a user must be able to find by grepping the agent log."""
    provider.initialize("s", hermes_home=str(tmp_path))

    with caplog.at_level("WARNING"):
        provider._log_failure("search", RecallAuthError("rejected"))
        provider._log_failure("store", RecallAuthError("rejected"))

    rejected = [r for r in caplog.records if "API key rejected" in r.message]
    assert len(rejected) == 1
    assert rejected[0].levelname == "ERROR"


def test_other_failures_are_logged_every_time(provider, tmp_path, caplog):
    provider.initialize("s", hermes_home=str(tmp_path))

    with caplog.at_level("WARNING"):
        provider._log_failure("search", RecallError("boom"))
        provider._log_failure("search", RecallError("boom"))

    assert sum("Recall search failed" in r.message for r in caplog.records) == 2


def test_system_prompt_block_mentions_recall(provider, tmp_path):
    provider.initialize("s", hermes_home=str(tmp_path))
    assert "Recall" in provider.system_prompt_block()


# -- save_config merges, never truncates -------------------------------------


def test_save_config_merges_into_an_existing_file(provider, tmp_path, write_recall_config):
    """Hermes' own writer reads-modifies-writes; a partial save must not wipe
    the keys it did not carry (dashboard setup submits a subset)."""
    write_recall_config(tmp_path, {"limit": 9, "min_chars": 120, "graph_boost": True})

    provider.save_config({"limit": 3}, str(tmp_path))

    written = json.loads(recall_config_path(str(tmp_path)).read_text())
    assert written == {"limit": 3, "min_chars": 120, "graph_boost": True}


def test_save_config_replaces_a_corrupt_file_with_a_valid_one(
    provider, tmp_path, write_recall_config
):
    write_recall_config(tmp_path, "{ not json")

    provider.save_config({"limit": 7}, str(tmp_path))

    assert json.loads(recall_config_path(str(tmp_path)).read_text()) == {"limit": 7}


def test_save_config_with_no_known_values_leaves_the_file_byte_identical(
    provider, tmp_path, write_recall_config
):
    path = write_recall_config(tmp_path, {"limit": 9, "sync_turns": False})
    before = path.read_bytes()

    provider.save_config({}, str(tmp_path))
    provider.save_config({"api_key": "rag_leak", "unknown": 1}, str(tmp_path))

    assert path.read_bytes() == before


def test_save_config_with_no_values_and_no_file_creates_nothing(provider, tmp_path):
    provider.save_config({}, str(tmp_path))

    assert not recall_config_path(str(tmp_path)).exists()


def test_save_config_keeps_keys_it_does_not_own(provider, tmp_path, write_recall_config):
    """Same rule as _write_provider_flat: unknown keys in the file survive."""
    write_recall_config(tmp_path, {"limit": 9, "written_by_something_else": "keep me"})

    provider.save_config({"limit": 4}, str(tmp_path))

    written = json.loads(recall_config_path(str(tmp_path)).read_text())
    assert written["written_by_something_else"] == "keep me"


# -- the two config declarations may never drift apart -----------------------


def _config_schema_module():
    """``config_schema.py`` imports the Hermes plugin base; load it by path."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "config_schema.py"
    spec = importlib.util.spec_from_file_location("recall_config_schema", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_config_declarations_expose_the_same_keys(provider):
    """One surface is the dashboard panel, the other is ``hermes memory
    setup``. A key present in only one of them is invisible on that surface."""
    schema = _config_schema_module().CONFIG_SCHEMA

    wizard_keys = {f["key"] for f in provider.get_config_schema()}
    panel_keys = {f.key for f in schema.fields}

    assert wizard_keys == panel_keys


def test_every_non_secret_field_is_a_defaults_key(provider):
    schema = _config_schema_module().CONFIG_SCHEMA

    wizard = {f["key"] for f in provider.get_config_schema() if not f.get("secret")}
    panel = {f.key for f in schema.fields if f.kind != "secret"}

    assert wizard == set(DEFAULTS)
    assert panel == set(DEFAULTS)


def test_config_schema_declares_exactly_one_secret_and_it_is_the_api_key(provider):
    schema = provider.get_config_schema()

    secrets = [f for f in schema if f.get("secret")]
    assert len(secrets) == 1
    field = secrets[0]
    assert field["key"] == "api_key"
    assert field["required"] is True
    assert field["env_var"] == "RECALL_API_KEY"
    assert field["url"] == "https://recall.carnival-devops.com"


def test_the_tunables_are_optional_so_the_wizard_can_be_skipped(provider):
    """Only the key is required — every other prompt may be pressed past."""
    for field in provider.get_config_schema():
        if field["key"] == "api_key":
            continue
        assert field.get("required", False) is False, field["key"]


def test_every_tunable_declares_a_default_and_a_type(provider):
    """``_normalize_memory_provider_schema`` keys the widget off ``type``."""
    kinds = {
        "base_url": "text",
        "limit": "integer",
        "rerank": "boolean",
        "graph_boost": "boolean",
        "writes_enabled": "boolean",
        "sync_turns": "boolean",
        "session_summary": "boolean",
        "max_chars": "integer",
        "min_chars": "integer",
    }
    by_key = {f["key"]: f for f in provider.get_config_schema()}

    for key, kind in kinds.items():
        assert by_key[key]["type"] == kind, key
        assert by_key[key]["default"] == DEFAULTS[key], key
        assert by_key[key].get("description"), key


def test_get_config_schema_returns_copies(provider):
    first = provider.get_config_schema()
    first[0]["description"] = "mutated"
    first.append({"key": "injected"})

    second = provider.get_config_schema()

    assert second[0]["description"] != "mutated"
    assert "injected" not in {f["key"] for f in second}


# -- the wizard writes strings; the config must still mean what it says ------


def test_config_values_are_coerced_to_the_type_of_their_default(
    tmp_path, write_recall_config
):
    """``hermes memory setup`` prompts return strings for every field, and
    ``bool("false")`` is True — so a saved "false" would silently mean on."""
    write_recall_config(
        tmp_path,
        {"rerank": "false", "sync_turns": "0", "limit": "9", "max_chars": "1500"},
    )

    config = load_recall_config(str(tmp_path))

    assert config["rerank"] is False
    assert config["sync_turns"] is False
    assert config["limit"] == 9
    assert config["max_chars"] == 1500


def test_an_uncoercible_value_falls_back_to_the_default(tmp_path, write_recall_config):
    write_recall_config(tmp_path, {"limit": "not-a-number", "rerank": "maybe"})

    config = load_recall_config(str(tmp_path))

    assert config["limit"] == DEFAULTS["limit"]
    assert config["rerank"] is DEFAULTS["rerank"]
