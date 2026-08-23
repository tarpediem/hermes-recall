"""The three extra tools: shipped ENABLED, per-agent, read-only.

They cost their schema on every turn, but the owner's call is that an end
user should get them without touching a config file. ``extra_tools`` stays
the authorization boundary in both directions: it is what turns them off —
an explicit empty list (or an empty string from the dashboard's text field)
leaves exactly the two base tools, and the three names stop being callable.
"""

import json
import logging

import pytest
from recall._client import SLOW_READ_TIMEOUT, RecallAuthError, RecallClient, RecallError
from recall._provider import (
    DEFAULTS,
    EXTRA_TOOL_SCHEMAS,
    SNIPPET_CHARS,
    RecallMemoryProvider,
    load_recall_config,
)

GRAPH_PAYLOAD = {
    "entities": [{"name": "recall", "type": "service"}],
    "relations": [{"from": "recall", "to": "neo4j", "type": "USES"}],
}
WHO_KNOWS_PAYLOAD = {"topic": "neo4j", "people": [{"name": "olivier", "score": 0.9}]}
STATS_PAYLOAD = {"total_memories": 278, "graph": {"entities": 500}}


class RecordingClient(RecallClient):
    """Stubbed transport: records every extra call and its parameters."""

    def __init__(self, raiser=None, graph=None, who=None, stats=None):
        super().__init__(api_key="rag_k", base_url="https://recall.example")
        self.raiser = raiser
        self._graph = GRAPH_PAYLOAD if graph is None else graph
        self._who = WHO_KNOWS_PAYLOAD if who is None else who
        self._stats = STATS_PAYLOAD if stats is None else stats
        self.calls = []

    def graph_recall(self, query, *, depth=2, limit=10, timeout=None):
        self.calls.append(
            ("graph_recall", {"query": query, "depth": depth, "limit": limit, "timeout": timeout})
        )
        if self.raiser is not None:
            raise self.raiser
        return self._graph

    def who_knows(self, topic, *, n_results=5, timeout=None):
        self.calls.append(
            ("who_knows", {"topic": topic, "n_results": n_results, "timeout": timeout})
        )
        if self.raiser is not None:
            raise self.raiser
        return self._who

    def stats(self, timeout=None):
        self.calls.append(("stats", {"timeout": timeout}))
        if self.raiser is not None:
            raise self.raiser
        return self._stats


def _provider(tmp_path, client=None, config=None, write_config=None, **init):
    if config is not None:
        assert write_config is not None
        write_config(tmp_path, config)
    provider = RecallMemoryProvider(client or RecordingClient())
    provider.initialize(
        "s1", hermes_home=str(tmp_path), platform="cli", agent_identity="charlotte", **init
    )
    return provider


ALL_EXTRAS = ["recall_graph", "who_knows", "recall_stats"]


# -- default: the three extras are on ----------------------------------------


def test_extra_tools_defaults_to_the_three_extras(tmp_path):
    assert DEFAULTS["extra_tools"] == ALL_EXTRAS
    assert load_recall_config(str(tmp_path))["extra_tools"] == ALL_EXTRAS


def test_every_schema_is_exposed_by_default_in_a_stable_order(tmp_path):
    schemas = _provider(tmp_path).get_tool_schemas()

    assert [s["name"] for s in schemas] == ["recall_search", "recall_store", *ALL_EXTRAS]


def test_the_default_schemas_are_copies_not_module_references(tmp_path):
    """The default path hands out the module constants too — deep-copied."""
    provider = _provider(tmp_path)

    first = provider.get_tool_schemas()
    first[2]["description"] = "mutated"

    assert provider.get_tool_schemas()[2]["description"] != "mutated"
    assert EXTRA_TOOL_SCHEMAS["recall_graph"]["description"] != "mutated"


@pytest.mark.parametrize("name", ALL_EXTRAS)
def test_every_extra_is_callable_by_default(name, tmp_path):
    client = RecordingClient()

    payload = json.loads(
        _provider(tmp_path, client).handle_tool_call(name, {"query": "q", "topic": "t"})
    )

    assert "error" not in payload
    assert len(client.calls) == 1


# -- turning them off --------------------------------------------------------


@pytest.mark.parametrize("value", [[], ""])
def test_an_explicit_empty_value_disables_every_extra(value, tmp_path, write_recall_config):
    """``[]`` in config.json and "" from the dashboard's text field are the
    disable path — neither may fall back to the (now non-empty) default."""
    write_recall_config(tmp_path, {"extra_tools": value})

    assert load_recall_config(str(tmp_path))["extra_tools"] == []

    client = RecordingClient()
    provider = _provider(tmp_path, client)
    assert [s["name"] for s in provider.get_tool_schemas()] == ["recall_search", "recall_store"]

    for name in ALL_EXTRAS:
        payload = json.loads(provider.handle_tool_call(name, {"query": "q", "topic": "t"}))
        assert payload["error"] == f"Unknown tool: {name}"
    assert client.calls == []


def test_the_three_extras_are_declared(tmp_path):
    assert set(EXTRA_TOOL_SCHEMAS) == set(ALL_EXTRAS)
    for name, schema in EXTRA_TOOL_SCHEMAS.items():
        assert schema["name"] == name
        assert schema["description"]
        assert schema["parameters"]["type"] == "object"


# -- opting in ---------------------------------------------------------------


def test_configured_extras_are_appended_after_the_base_tools(tmp_path, write_recall_config):
    provider = _provider(
        tmp_path, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    assert [s["name"] for s in provider.get_tool_schemas()] == [
        "recall_search",
        "recall_store",
        *ALL_EXTRAS,
    ]


def test_only_the_configured_extras_are_exposed(tmp_path, write_recall_config):
    provider = _provider(
        tmp_path, config={"extra_tools": ["who_knows"]}, write_config=write_recall_config
    )

    assert [s["name"] for s in provider.get_tool_schemas()] == [
        "recall_search",
        "recall_store",
        "who_knows",
    ]


def test_the_configured_order_is_preserved_and_deduplicated(tmp_path, write_recall_config):
    provider = _provider(
        tmp_path,
        config={"extra_tools": ["recall_stats", "who_knows", "recall_stats"]},
        write_config=write_recall_config,
    )

    assert [s["name"] for s in provider.get_tool_schemas()][2:] == ["recall_stats", "who_knows"]


def test_extra_schemas_are_copies_not_module_references(tmp_path, write_recall_config):
    provider = _provider(
        tmp_path, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    first = provider.get_tool_schemas()
    first[2]["description"] = "mutated"
    first[2]["parameters"]["properties"]["injected"] = {"type": "string"}

    second = provider.get_tool_schemas()
    assert second[2]["description"] != "mutated"
    assert "injected" not in second[2]["parameters"]["properties"]
    assert EXTRA_TOOL_SCHEMAS["recall_graph"]["description"] != "mutated"


def test_an_unknown_extra_is_ignored_with_a_single_warning(
    tmp_path, write_recall_config, caplog
):
    provider = _provider(
        tmp_path,
        config={"extra_tools": ["who_knows", "recall_delete_everything"]},
        write_config=write_recall_config,
    )

    with caplog.at_level(logging.WARNING, logger="recall._provider"):
        names = [s["name"] for s in provider.get_tool_schemas()]
        provider.get_tool_schemas()

    assert names == ["recall_search", "recall_store", "who_knows"]
    warnings = [r for r in caplog.records if "recall_delete_everything" in r.getMessage()]
    assert len(warnings) == 1


def test_a_csv_string_from_the_wizard_is_coerced_to_a_list(tmp_path, write_recall_config):
    write_recall_config(tmp_path, {"extra_tools": " who_knows , recall_stats ,, "})

    assert load_recall_config(str(tmp_path))["extra_tools"] == ["who_knows", "recall_stats"]


@pytest.mark.parametrize("value", [42, None, True, {"a": 1}])
def test_a_nonsense_extra_tools_value_falls_back_to_the_default(
    value, tmp_path, write_recall_config
):
    """Junk means "I did not say" — the default. Note "" is NOT junk: it is
    the dashboard's disable path, covered above."""
    write_recall_config(tmp_path, {"extra_tools": value})

    assert load_recall_config(str(tmp_path))["extra_tools"] == ALL_EXTRAS


# -- dispatch ----------------------------------------------------------------


def test_recall_graph_forwards_its_parameters_off_the_turn_path(
    tmp_path, write_recall_config
):
    client = RecordingClient()
    provider = _provider(
        tmp_path, client, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    payload = json.loads(
        provider.handle_tool_call("recall_graph", {"query": "recall", "depth": 3, "limit": 4})
    )

    assert client.calls == [
        (
            "graph_recall",
            {"query": "recall", "depth": 3, "limit": 4, "timeout": SLOW_READ_TIMEOUT},
        )
    ]
    assert payload["entities"][0]["name"] == "recall"


def test_recall_graph_uses_the_documented_defaults(tmp_path, write_recall_config):
    client = RecordingClient()
    provider = _provider(
        tmp_path, client, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    provider.handle_tool_call("recall_graph", {"query": "recall"})

    assert client.calls[0][1]["depth"] == 2
    assert client.calls[0][1]["limit"] == 10


def test_recall_graph_without_a_query_is_a_tool_error(tmp_path, write_recall_config):
    client = RecordingClient()
    provider = _provider(
        tmp_path, client, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    assert "error" in json.loads(provider.handle_tool_call("recall_graph", {}))
    assert client.calls == []


def test_recall_graph_wraps_a_list_payload(tmp_path, write_recall_config):
    client = RecordingClient(graph=[{"name": f"e{i}"} for i in range(50)])
    provider = _provider(
        tmp_path, client, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    payload = json.loads(provider.handle_tool_call("recall_graph", {"query": "q"}))

    assert payload["count"] == 50
    assert len(payload["results"]) < 50  # capped for the model


def test_who_knows_forwards_its_parameters_off_the_turn_path(tmp_path, write_recall_config):
    client = RecordingClient()
    provider = _provider(
        tmp_path, client, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    payload = json.loads(
        provider.handle_tool_call("who_knows", {"topic": "neo4j", "n_results": 3})
    )

    assert client.calls == [
        ("who_knows", {"topic": "neo4j", "n_results": 3, "timeout": SLOW_READ_TIMEOUT})
    ]
    assert payload["people"][0]["name"] == "olivier"


def test_who_knows_defaults_to_five_results(tmp_path, write_recall_config):
    client = RecordingClient()
    provider = _provider(
        tmp_path, client, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    provider.handle_tool_call("who_knows", {"topic": "neo4j"})

    assert client.calls[0][1]["n_results"] == 5


def test_who_knows_without_a_topic_is_a_tool_error(tmp_path, write_recall_config):
    client = RecordingClient()
    provider = _provider(
        tmp_path, client, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    assert "error" in json.loads(provider.handle_tool_call("who_knows", {"topic": "  "}))
    assert client.calls == []


def test_recall_stats_forwards_the_off_turn_timeout(tmp_path, write_recall_config):
    client = RecordingClient()
    provider = _provider(
        tmp_path, client, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    payload = json.loads(provider.handle_tool_call("recall_stats", {}))

    assert client.calls == [("stats", {"timeout": SLOW_READ_TIMEOUT})]
    assert payload["total_memories"] == 278
    assert payload["graph"] == {"entities": 500}


def test_recall_stats_degrades_without_the_graph_numbers(tmp_path, write_recall_config):
    client = RecordingClient(stats={"total_memories": 12})
    provider = _provider(
        tmp_path, client, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    payload = json.loads(provider.handle_tool_call("recall_stats", {}))

    assert payload == {"total_memories": 12}


# -- extra_tools gates the CAPABILITY, not just the schema -------------------


@pytest.mark.parametrize(
    "name,args",
    [
        ("recall_graph", {"query": "recall"}),
        ("who_knows", {"topic": "neo4j"}),
        ("recall_stats", {}),
    ],
)
def test_a_disabled_extra_is_an_unknown_tool_and_makes_no_call(
    name, args, tmp_path, write_recall_config
):
    """The names are public (README); a call to one an agent switched OFF
    must not reach Recall."""
    client = RecordingClient()
    provider = _provider(
        tmp_path, client, config={"extra_tools": []}, write_config=write_recall_config
    )

    payload = json.loads(provider.handle_tool_call(name, args))

    assert payload["error"] == f"Unknown tool: {name}"
    assert client.calls == []


@pytest.mark.parametrize(
    "name,args",
    [
        ("recall_graph", {"query": "recall"}),
        ("who_knows", {"topic": "neo4j"}),
        ("recall_stats", {}),
    ],
)
def test_only_the_enabled_extra_is_callable(name, args, tmp_path, write_recall_config):
    """Enabling one extra must not enable the other two."""
    client = RecordingClient()
    provider = _provider(
        tmp_path, client, config={"extra_tools": [name]}, write_config=write_recall_config
    )

    assert "error" not in json.loads(provider.handle_tool_call(name, args))
    others = [n for n in ALL_EXTRAS if n != name]
    for other in others:
        payload = json.loads(
            provider.handle_tool_call(other, {"query": "q", "topic": "t"})
        )
        assert payload["error"] == f"Unknown tool: {other}"
    assert [call[0] for call in client.calls] == [
        {"recall_graph": "graph_recall", "who_knows": "who_knows", "recall_stats": "stats"}[name]
    ]


def test_an_unknown_extra_in_the_config_does_not_become_callable(tmp_path, write_recall_config):
    client = RecordingClient()
    provider = _provider(
        tmp_path,
        client,
        config={"extra_tools": ["recall_delete_everything"]},
        write_config=write_recall_config,
    )

    payload = json.loads(provider.handle_tool_call("recall_delete_everything", {}))

    assert "Unknown tool" in payload["error"]
    assert client.calls == []


# -- read-only: never gated by the write switches ----------------------------


@pytest.mark.parametrize(
    "name,args",
    [
        ("recall_graph", {"query": "recall"}),
        ("who_knows", {"topic": "neo4j"}),
        ("recall_stats", {}),
    ],
)
def test_the_extras_work_with_writes_disabled(name, args, tmp_path, write_recall_config):
    client = RecordingClient()
    provider = _provider(
        tmp_path,
        client,
        config={"extra_tools": ALL_EXTRAS, "writes_enabled": False},
        write_config=write_recall_config,
    )

    payload = json.loads(provider.handle_tool_call(name, args))

    assert "error" not in payload
    assert client.calls


@pytest.mark.parametrize(
    "name,args",
    [
        ("recall_graph", {"query": "recall"}),
        ("who_knows", {"topic": "neo4j"}),
        ("recall_stats", {}),
    ],
)
def test_the_extras_work_in_a_non_primary_agent_context(name, args, tmp_path, write_recall_config):
    client = RecordingClient()
    provider = _provider(
        tmp_path,
        client,
        config={"extra_tools": ALL_EXTRAS},
        write_config=write_recall_config,
        agent_context="subagent",
    )

    assert "error" not in json.loads(provider.handle_tool_call(name, args))
    assert client.calls


# -- failure is a tool_error, never an exception -----------------------------


@pytest.mark.parametrize(
    "raiser", [RecallError("down"), RecallAuthError("rejected"), RuntimeError("kaboom")]
)
@pytest.mark.parametrize(
    "name,args",
    [
        ("recall_graph", {"query": "recall"}),
        ("who_knows", {"topic": "neo4j"}),
        ("recall_stats", {}),
    ],
)
def test_a_failing_extra_returns_a_tool_error(raiser, name, args, tmp_path, write_recall_config):
    provider = _provider(
        tmp_path,
        RecordingClient(raiser=raiser),
        config={"extra_tools": ALL_EXTRAS},
        write_config=write_recall_config,
    )

    payload = json.loads(provider.handle_tool_call(name, args))

    assert "error" in payload


def test_a_failing_extra_never_names_the_api_key(tmp_path, write_recall_config, caplog):
    secret = "rag_" + "extra-tool-5ecret-" + "0123456789"
    client = RecordingClient(raiser=RecallAuthError("rejected"))
    client.api_key = secret
    provider = _provider(
        tmp_path, client, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    with caplog.at_level(logging.DEBUG, logger="recall._provider"):
        blob = "".join(
            provider.handle_tool_call(name, args)
            for name, args in (
                ("recall_graph", {"query": "q"}),
                ("who_knows", {"topic": "t"}),
                ("recall_stats", {}),
            )
        )

    assert secret not in blob
    assert secret not in caplog.text


# -- the payload the model sees is trimmed -----------------------------------


def test_long_strings_are_snippet_truncated(tmp_path, write_recall_config):
    client = RecordingClient(who={"people": [{"note": "x" * 5000}]})
    provider = _provider(
        tmp_path, client, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    payload = json.loads(provider.handle_tool_call("who_knows", {"topic": "t"}))

    assert len(payload["people"][0]["note"]) == SNIPPET_CHARS


def test_long_lists_are_capped(tmp_path, write_recall_config):
    client = RecordingClient(graph={"entities": [{"name": f"e{i}"} for i in range(200)]})
    provider = _provider(
        tmp_path, client, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    payload = json.loads(provider.handle_tool_call("recall_graph", {"query": "q"}))

    assert 0 < len(payload["entities"]) < 200


def test_a_deeply_nested_payload_does_not_explode(tmp_path, write_recall_config):
    nested = {"level": 0}
    node = nested
    for i in range(1, 40):
        node["child"] = {"level": i}
        node = node["child"]
    client = RecordingClient(who=nested)
    provider = _provider(
        tmp_path, client, config={"extra_tools": ALL_EXTRAS}, write_config=write_recall_config
    )

    raw = provider.handle_tool_call("who_knows", {"topic": "t"})

    assert len(raw) < 2000
    json.loads(raw)
