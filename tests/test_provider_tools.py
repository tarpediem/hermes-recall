"""The two tools the model may call explicitly."""

import json

from recall._client import RecallAuthError, RecallClient, RecallError
from recall._provider import RecallMemoryProvider

ITEMS = [
    {"id": "m1", "type": "decision", "timestamp": "2026-07-21T03:11:00Z",
     "snippet": "Marker extraction moved back to GPU", "score": 0.91},
]


class RecordingClient(RecallClient):
    def __init__(self, results=None, search_raiser=None, store_raiser=None):
        super().__init__(api_key="rag_k", base_url="https://recall.example")
        self.results = results if results is not None else ITEMS
        self.search_raiser = search_raiser
        self.store_raiser = store_raiser
        self.searches = []
        self.stored = []

    def search(self, query, *, limit=5, rerank=True, graph_boost=False, tags=None, timeout=None):
        self.searches.append({"query": query, "limit": limit, "rerank": rerank,
                              "graph_boost": graph_boost, "tags": tags})
        if self.search_raiser is not None:
            raise self.search_raiser
        return self.results

    def store(self, content, *, memory_type="context", scope="project", tags=None):
        self.stored.append({"content": content, "memory_type": memory_type,
                            "scope": scope, "tags": list(tags or [])})
        if self.store_raiser is not None:
            raise self.store_raiser
        return "mem-42"


def _provider(client, tmp_path, **init):
    provider = RecallMemoryProvider(client)
    provider.initialize(
        "s1", hermes_home=str(tmp_path), platform="cli",
        agent_identity="charlotte", **init
    )
    return provider


def test_the_five_default_tools_are_exposed(tmp_path):
    """Two base tools, then the three extras that ship enabled."""
    schemas = _provider(RecordingClient(), tmp_path).get_tool_schemas()

    assert [s["name"] for s in schemas] == [
        "recall_search",
        "recall_store",
        "recall_graph",
        "who_knows",
        "recall_stats",
    ]


def test_search_schema_shape(tmp_path):
    schema = _provider(RecordingClient(), tmp_path).get_tool_schemas()[0]

    props = schema["parameters"]["properties"]
    assert set(props) == {"query", "limit"}
    assert schema["parameters"]["required"] == ["query"]
    assert props["limit"]["default"] == 5


def test_store_schema_shape(tmp_path):
    schema = _provider(RecordingClient(), tmp_path).get_tool_schemas()[1]

    props = schema["parameters"]["properties"]
    assert set(props) == {"content", "memory_type", "tags"}
    assert schema["parameters"]["required"] == ["content"]
    assert props["memory_type"]["default"] == "context"
    assert "decision" in props["memory_type"]["enum"]


def test_recall_search_returns_json_results(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    payload = json.loads(provider.handle_tool_call("recall_search", {"query": "marker"}))

    assert payload["count"] == 1
    assert payload["results"][0]["type"] == "decision"
    assert payload["results"][0]["snippet"] == "Marker extraction moved back to GPU"
    assert client.searches[0]["limit"] == 5


def test_recall_search_honours_an_explicit_limit(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.handle_tool_call("recall_search", {"query": "marker", "limit": 12})

    assert client.searches[0]["limit"] == 12


def test_recall_search_reports_no_results_cleanly(tmp_path):
    provider = _provider(RecordingClient(results=[]), tmp_path)

    payload = json.loads(provider.handle_tool_call("recall_search", {"query": "nothing"}))

    assert payload["count"] == 0
    assert payload["results"] == []


def test_recall_search_missing_query_is_a_tool_error(tmp_path):
    provider = _provider(RecordingClient(), tmp_path)

    payload = json.loads(provider.handle_tool_call("recall_search", {}))

    assert "error" in payload


def test_recall_search_failure_is_a_tool_error(tmp_path):
    provider = _provider(RecordingClient(search_raiser=RecallError("down")), tmp_path)

    payload = json.loads(provider.handle_tool_call("recall_search", {"query": "x"}))

    assert "error" in payload


def test_recall_store_stores_and_confirms(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    payload = json.loads(
        provider.handle_tool_call(
            "recall_store",
            {"content": "Charlotte runs on merdouille", "memory_type": "architecture",
             "tags": ["infra"]},
        )
    )

    assert payload["stored"] is True
    assert payload["memory_id"] == "mem-42"
    record = client.stored[0]
    assert record["memory_type"] == "architecture"
    assert record["scope"] == "project"
    assert "infra" in record["tags"]
    assert "hermes" in record["tags"]
    assert "model-tool" in record["tags"]


def test_recall_store_defaults_to_context(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.handle_tool_call("recall_store", {"content": "A fact worth pinning"})

    assert client.stored[0]["memory_type"] == "context"


def test_recall_store_rejects_a_non_canonical_memory_type(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    payload = json.loads(
        provider.handle_tool_call("recall_store", {"content": "x", "memory_type": "nope"})
    )

    assert "error" in payload
    assert client.stored == []


def test_recall_store_missing_content_is_a_tool_error(tmp_path):
    provider = _provider(RecordingClient(), tmp_path)

    payload = json.loads(provider.handle_tool_call("recall_store", {}))

    assert "error" in payload


def test_recall_store_truncates_at_max_chars(tmp_path, write_recall_config):
    write_recall_config(tmp_path, {"max_chars": 100})
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.handle_tool_call("recall_store", {"content": "x" * 5000})

    assert len(client.stored[0]["content"]) == 100


def test_recall_store_failure_is_a_tool_error(tmp_path):
    provider = _provider(RecordingClient(store_raiser=RecallAuthError("bad key")), tmp_path)

    payload = json.loads(provider.handle_tool_call("recall_store", {"content": "x"}))

    assert "error" in payload


def test_recall_store_is_refused_in_a_non_primary_context(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path, agent_context="cron")

    payload = json.loads(provider.handle_tool_call("recall_store", {"content": "x" * 50}))

    assert "error" in payload
    assert client.stored == []


def test_unknown_tool_name_is_a_tool_error(tmp_path):
    provider = _provider(RecordingClient(), tmp_path)

    payload = json.loads(provider.handle_tool_call("recall_nonsense", {}))

    assert "error" in payload


def test_handle_tool_call_always_returns_valid_json(tmp_path):
    provider = _provider(RecordingClient(search_raiser=RuntimeError("kaboom")), tmp_path)

    for name, args in (
        ("recall_search", {"query": "x"}),
        ("recall_store", {"content": "x"}),
        ("recall_nonsense", {}),
    ):
        json.loads(provider.handle_tool_call(name, args))
