"""Transport-level fail-open sweep.

Unlike the per-task tests — which stub ``RecallClient`` — this module breaks
``requests`` itself, so the *real* client code path runs and the provider is
the only thing standing between a network fault and the agent loop.

Every public provider method is exercised under every fault and must:

* return normally — no exception escapes into the agent loop,
* return the documented neutral value, with the documented type,
* never let the API key reach a log record or a returned string.

Nothing here sleeps and nothing here touches the network: an autouse guard
fails the test if a fault fixture forgot to replace ``recall._client.requests``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
import requests
from recall import _client
from recall._client import (
    CONNECT_TIMEOUT,
    READ_TIMEOUT,
    SLOW_READ_TIMEOUT,
    WRITE_TIMEOUT,
    RecallClient,
)
from recall._provider import RecallMemoryProvider

# A key shaped like a real one. It must never appear in a log record or in
# anything a public method returns, under any fault.
# Built by concatenation so Hermes' plugin security scanner does not
# mistake this FAKE fixture for a real exposed credential.
API_KEY = "rag_" + "sweep-5ecret-" + "never-logged-0123456789"

MESSAGES = [
    {"role": "user", "content": "Should we move marker extraction back to the GPU?"},
    {
        "role": "assistant",
        "content": "We decided to move it back with page-by-page chunking.",
    },
    {"role": "user", "content": "And what timeout should the extractor use from now on?"},
    {
        "role": "assistant",
        "content": "EXTRACT_TIMEOUT_SECONDS is now 1800 instead of 300.",
    },
]

QUERY = "why did the graph indexer stop extracting entities last week"


# -- fake transports ---------------------------------------------------------


class FakeResponse:
    """Just enough of ``requests.Response`` for ``RecallClient``."""

    def __init__(self, status_code: int, payload: Any = None, json_exc: Exception | None = None):
        self.status_code = status_code
        self._payload = payload
        self._json_exc = json_exc
        self.text = "<body withheld>"

    def json(self) -> Any:
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload


class RecordingRequests:
    """Base fake: records every call so timeouts/headers can be asserted."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def _record(self, verb: str, url: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((verb, url, kwargs))

    def get(self, url: str, **kwargs: Any) -> Any:
        self._record("get", url, kwargs)
        return self._respond("get")

    def post(self, url: str, **kwargs: Any) -> Any:
        self._record("post", url, kwargs)
        return self._respond("post")

    def _respond(self, verb: str) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError


class ExplodingRequests(RecordingRequests):
    """Every HTTP verb raises, like a fully unreachable backend."""

    def __init__(self, exc_factory: Callable[[], BaseException]) -> None:
        super().__init__()
        self._exc_factory = exc_factory

    def _respond(self, verb: str) -> Any:
        raise self._exc_factory()


class StaticRequests(RecordingRequests):
    """Every HTTP verb answers with the same canned response."""

    def __init__(self, response_factory: Callable[[], FakeResponse]) -> None:
        super().__init__()
        self._response_factory = response_factory

    def _respond(self, verb: str) -> Any:
        return self._response_factory()


class GatedRequests(RecordingRequests):
    """Blocks until an Event is released, then times out.

    Used to prove a caller does not wait on the transport. The test releases
    the gate itself, so nothing ever actually sleeps.
    """

    def __init__(self) -> None:
        super().__init__()
        self.released = threading.Event()
        self.entered = threading.Event()

    def _respond(self, verb: str) -> Any:
        self.entered.set()
        # Bounded so a bug in the test can never hang the suite.
        self.released.wait(timeout=10.0)
        raise requests.exceptions.Timeout("read timed out")


class NoNetwork:
    """Autouse guard: any unpatched HTTP call is a test bug, not a pass."""

    def get(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("test performed a real HTTP GET")

    def post(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("test performed a real HTTP POST")


# -- the fault matrix --------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Fault:
    """One way the transport can misbehave.

    ``hard`` means the client must raise (so the tools return ``error``);
    a soft fault is a well-formed HTTP exchange whose *payload* is wrong, and
    the client is expected to absorb it and return an empty/filtered result.
    """

    id: str
    factory: Callable[[], Any]
    hard: bool


FAULTS: list[Fault] = [
    # -- transport is down
    Fault(
        "connection_error",
        lambda: ExplodingRequests(lambda: requests.exceptions.ConnectionError("refused")),
        True,
    ),
    Fault(
        "read_timeout",
        lambda: ExplodingRequests(lambda: requests.exceptions.Timeout("read timed out")),
        True,
    ),
    Fault("os_error", lambda: ExplodingRequests(lambda: OSError("network unreachable")), True),
    Fault("runtime_error", lambda: ExplodingRequests(lambda: RuntimeError("kaboom")), True),
    Fault("value_error", lambda: ExplodingRequests(lambda: ValueError("nonsense")), True),
    # -- the server answers, badly
    Fault("http_401", lambda: StaticRequests(lambda: FakeResponse(401, {"detail": "nope"})), True),
    Fault("http_403", lambda: StaticRequests(lambda: FakeResponse(403, {"detail": "nope"})), True),
    Fault("http_500", lambda: StaticRequests(lambda: FakeResponse(500, {"detail": "boom"})), True),
    Fault("http_502", lambda: StaticRequests(lambda: FakeResponse(502, {"detail": "gw"})), True),
    Fault(
        "html_body",
        lambda: StaticRequests(
            lambda: FakeResponse(200, json_exc=ValueError("Expecting value: line 1 column 1"))
        ),
        True,
    ),
    Fault("json_null_body", lambda: StaticRequests(lambda: FakeResponse(200, None)), True),
    Fault(
        "payload_is_a_list",
        lambda: StaticRequests(lambda: FakeResponse(200, [{"snippet": "x"}])),
        True,
    ),
    # -- the server answers 200 with the wrong shape
    Fault("items_missing", lambda: StaticRequests(lambda: FakeResponse(200, {"ok": True})), False),
    Fault(
        "items_is_a_dict",
        lambda: StaticRequests(lambda: FakeResponse(200, {"items": {"snippet": "x"}})),
        False,
    ),
    Fault(
        "items_are_junk",
        lambda: StaticRequests(
            lambda: FakeResponse(200, {"items": [None, 3, "text", [], {"id": "no-snippet"}]})
        ),
        False,
    ),
    Fault(
        "items_have_null_fields",
        lambda: StaticRequests(
            lambda: FakeResponse(200, {"items": [{"snippet": None, "type": None, "score": None}]})
        ),
        False,
    ),
]


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Applied before any fault fixture, so an unpatched call is caught."""
    monkeypatch.setattr(_client, "requests", NoNetwork())


@pytest.fixture(params=FAULTS, ids=[f.id for f in FAULTS])
def fault(request, monkeypatch) -> Fault:
    monkeypatch.setattr(_client, "requests", request.param.factory())
    return request.param


@pytest.fixture(autouse=True)
def _capture_warnings(caplog):
    caplog.set_level(logging.DEBUG, logger="recall._provider")
    caplog.set_level(logging.DEBUG, logger="recall._client")
    return caplog


def make_provider(tmp_path) -> RecallMemoryProvider:
    provider = RecallMemoryProvider(
        RecallClient(api_key=API_KEY, base_url="https://recall.invalid")
    )
    provider.initialize(
        "s1",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_identity="charlotte",
        agent_context="primary",
    )
    return provider


@pytest.fixture()
def provider(tmp_path):
    p = make_provider(tmp_path)
    yield p
    p.shutdown()


def assert_no_key(caplog, *returned: Any) -> None:
    """The key must not be in any log record nor in anything returned."""
    blob = caplog.text
    for record in caplog.records:
        blob += record.getMessage() + repr(record.args)
    for value in returned:
        blob += repr(value)
    assert API_KEY not in blob
    # Also catch a truncated/partial leak.
    assert "rag_sweep" not in blob


# -- the sweep ---------------------------------------------------------------


def test_prefetch_returns_empty_string(provider, fault, caplog):
    block = provider.prefetch(QUERY, session_id="s1")

    assert block == ""
    assert_no_key(caplog, block)


def test_recall_status_is_none_after_a_failed_prefetch(provider, fault, caplog):
    provider.prefetch(QUERY, session_id="s1")

    assert provider.recall_status() is None
    assert_no_key(caplog)


def test_queue_prefetch_returns_none_and_shutdown_drains(provider, fault, caplog):
    assert provider.queue_prefetch(QUERY, session_id="s1") is None
    provider.shutdown()

    assert provider.prefetch(QUERY, session_id="s1") == ""
    assert_no_key(caplog)


def test_sync_turn_returns_none(provider, fault, caplog):
    result = provider.sync_turn(
        "Why did the pgvector delete wipe the entire collection last night?",
        "Because an empty ids list is falsy, so the filter was never appended.",
        session_id="s1",
        messages=MESSAGES,
    )
    provider.shutdown()

    assert result is None
    assert_no_key(caplog)


def test_on_memory_write_three_positional_args(provider, fault, caplog):
    result = provider.on_memory_write("add", "user", "Olivier prefers fish, not bash or zsh")
    provider.shutdown()

    assert result is None
    assert_no_key(caplog)


def test_on_memory_write_four_args_with_metadata(provider, fault, caplog):
    result = provider.on_memory_write(
        "replace",
        "agent",
        "The extractor timeout is 1800 seconds since the page-chunking change",
        {"source": "builtin", "nested": {"weird": object()}},
    )
    provider.shutdown()

    assert result is None
    assert_no_key(caplog)


def test_on_delegation_returns_none(provider, fault, caplog):
    result = provider.on_delegation(
        "Audit the pgvector adapter for falsy-list guards",
        "Found one in delete(); an empty ids list skipped the filter entirely.",
        child_session_id="child-1",
    )
    provider.shutdown()

    assert result is None
    assert_no_key(caplog)


def test_on_session_end_returns_none(provider, fault, caplog):
    result = provider.on_session_end(MESSAGES)
    provider.shutdown()

    assert result is None
    assert_no_key(caplog)


def test_on_session_switch_plain_reset_and_rewound(provider, fault, caplog):
    provider.queue_prefetch(QUERY, session_id="s1")

    assert provider.on_session_switch("s2", parent_session_id="s1") is None
    assert provider.on_session_switch("s3", parent_session_id="s2", rewound=True) is None
    assert provider.on_session_switch("s4", parent_session_id="s3", reset=True) is None
    assert_no_key(caplog)


def test_on_pre_compress_still_returns_its_insight_block(provider, fault, caplog):
    text = provider.on_pre_compress(MESSAGES)
    provider.shutdown()

    assert isinstance(text, str)
    assert text.startswith("Insights to preserve")
    assert_no_key(caplog, text)


def test_handle_tool_call_search_returns_json(provider, fault, caplog):
    raw = provider.handle_tool_call("recall_search", {"query": "marker extraction"})
    payload = json.loads(raw)

    if fault.hard:
        assert payload["error"] == "Recall search failed"
    else:
        assert payload["count"] == 0
        assert payload["results"] == []
    assert_no_key(caplog, raw)


def test_handle_tool_call_store_returns_json(provider, fault, caplog):
    raw = provider.handle_tool_call(
        "recall_store",
        {"content": "A fact worth pinning about the extractor", "memory_type": "decision"},
    )
    payload = json.loads(raw)

    if fault.hard:
        assert payload["error"] == "Recall store failed"
    else:
        assert payload["stored"] is True
    assert_no_key(caplog, raw)


# -- the opt-in extras, over the real client, under every fault --------------


EXTRA_CALLS = [
    ("recall_graph", {"query": QUERY}),
    ("who_knows", {"topic": "neo4j"}),
    ("recall_stats", {}),
]


@pytest.fixture()
def provider_with_extras(tmp_path):
    """A provider that opted into all three extras."""
    import json as _json

    from recall._provider import recall_config_path

    path = recall_config_path(str(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _json.dumps({"extra_tools": ["recall_graph", "who_knows", "recall_stats"]}),
        encoding="utf-8",
    )
    p = make_provider(tmp_path)
    yield p
    p.shutdown()


@pytest.mark.parametrize("name,args", EXTRA_CALLS, ids=[c[0] for c in EXTRA_CALLS])
def test_the_extras_return_json_under_every_fault(
    name, args, provider_with_extras, fault, caplog
):
    raw = provider_with_extras.handle_tool_call(name, args)
    payload = json.loads(raw)

    assert isinstance(payload, dict)
    # ``payload_is_a_list`` is hard for every other call, but a top-level
    # array IS the documented shape of /graph/recall — the tool absorbs it.
    absorbed = name == "recall_graph" and fault.id == "payload_is_a_list"
    if fault.hard and not absorbed:
        assert "error" in payload
    if absorbed:
        assert payload["count"] == 1
    assert_no_key(caplog, raw)


def test_the_extras_are_exposed_only_once_opted_in(provider, provider_with_extras, caplog):
    assert [s["name"] for s in provider.get_tool_schemas()] == ["recall_search", "recall_store"]
    assert [s["name"] for s in provider_with_extras.get_tool_schemas()] == [
        "recall_search",
        "recall_store",
        "recall_graph",
        "who_knows",
        "recall_stats",
    ]
    assert_no_key(caplog)


def test_the_extras_reach_the_transport_with_the_off_turn_budget(tmp_path, monkeypatch, caplog):
    fake = ExplodingRequests(lambda: requests.exceptions.Timeout("read timed out"))
    monkeypatch.setattr(_client, "requests", fake)
    provider = make_provider(tmp_path)

    for name, args in EXTRA_CALLS:
        provider.handle_tool_call(name, args)
    provider.shutdown()

    off_turn = (CONNECT_TIMEOUT, SLOW_READ_TIMEOUT)
    assert [c[0] for c in fake.calls] == ["get", "get", "get"], "reads only"
    assert [kwargs["timeout"] for _v, _u, kwargs in fake.calls] == [off_turn] * 3
    assert_no_key(caplog)


def test_handle_tool_call_unknown_tool_returns_json_error(provider, fault, caplog):
    raw = provider.handle_tool_call("recall_delete_everything", {"query": "x"})
    payload = json.loads(raw)

    assert "Unknown tool" in payload["error"]
    assert_no_key(caplog, raw)


def test_handle_tool_call_survives_non_dict_args(provider, fault, caplog):
    for args in (None, [], "query", 7):
        raw = provider.handle_tool_call("recall_search", args)
        assert "error" in json.loads(raw)
    assert_no_key(caplog)


def test_shutdown_is_idempotent(provider, fault, caplog):
    assert provider.shutdown() is None
    assert provider.shutdown() is None
    assert_no_key(caplog)


def test_system_prompt_block_is_a_clean_string(provider, fault, caplog):
    block = provider.system_prompt_block()

    assert isinstance(block, str)
    assert block
    assert_no_key(caplog, block)


def test_get_config_schema_never_carries_the_key(provider, fault, caplog):
    schema = provider.get_config_schema()

    assert isinstance(schema, list)
    assert schema[0]["secret"] is True
    assert_no_key(caplog, schema)


def test_get_tool_schemas_is_a_list_of_two_by_default(provider, fault, caplog):
    """No extra_tools configured: the extras cost nothing, not even a schema."""
    schemas = provider.get_tool_schemas()

    assert [s["name"] for s in schemas] == ["recall_search", "recall_store"]
    assert_no_key(caplog, schemas)


def test_backup_paths_is_empty(provider, fault, caplog):
    assert provider.backup_paths() == []
    assert_no_key(caplog)


def test_availability_never_leaks_the_key(provider, fault, caplog):
    assert provider.is_available() is True
    reason = provider.unavailable_reason()

    assert isinstance(reason, str)
    assert_no_key(caplog, reason, repr(provider._client), provider.name)


def test_save_config_returns_none_even_when_the_path_is_unusable(provider, fault, tmp_path, caplog):
    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a directory", encoding="utf-8")

    assert provider.save_config({"limit": 9, "nope": 1}, str(blocked)) is None
    assert provider.save_config(None, str(tmp_path / "fresh")) is None
    assert_no_key(caplog)


# -- initialize with a broken home ------------------------------------------


@pytest.mark.parametrize(
    "broken",
    ["missing", "invalid_json", "json_list", "json_scalar", "config_is_a_dir", "nul_byte", "empty"],
)
def test_initialize_with_a_broken_home_falls_back_to_defaults(broken, tmp_path, caplog):
    from recall._provider import DEFAULTS, recall_config_path

    home = tmp_path / broken
    if broken == "invalid_json":
        path = recall_config_path(str(home))
        path.parent.mkdir(parents=True)
        path.write_text("{not json at all", encoding="utf-8")
    elif broken == "json_list":
        path = recall_config_path(str(home))
        path.parent.mkdir(parents=True)
        path.write_text('["limit", 9]', encoding="utf-8")
    elif broken == "json_scalar":
        path = recall_config_path(str(home))
        path.parent.mkdir(parents=True)
        path.write_text("42", encoding="utf-8")
    elif broken == "config_is_a_dir":
        recall_config_path(str(home)).mkdir(parents=True)

    home_arg = "a\0b" if broken == "nul_byte" else ("" if broken == "empty" else str(home))

    provider = RecallMemoryProvider(RecallClient(api_key=API_KEY, base_url="https://x.invalid"))
    assert provider.initialize("s1", hermes_home=home_arg, platform="cli") is None

    assert provider._config["limit"] == DEFAULTS["limit"]
    assert provider.prefetch("", session_id="s1") == ""  # no HTTP: trivial prompt
    assert_no_key(caplog)


def test_initialize_survives_junk_kwargs(tmp_path, caplog):
    provider = RecallMemoryProvider(RecallClient(api_key=API_KEY, base_url="https://x.invalid"))

    assert (
        provider.initialize(
            "s1", hermes_home=object(), platform=None, agent_identity=1, agent_context=b"primary"
        )
        is None
    )
    assert provider.backup_paths() == []
    assert_no_key(caplog)


# -- the 401 contract --------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_warned_exactly_once_per_session(status, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(_client, "requests", StaticRequests(lambda: FakeResponse(status, {})))
    provider = make_provider(tmp_path)

    provider.prefetch(QUERY, session_id="s1")
    provider.handle_tool_call("recall_search", {"query": QUERY})
    provider.handle_tool_call("recall_store", {"content": "something durable enough to keep"})
    provider.sync_turn(
        "Why did the pgvector delete wipe the entire collection last night?",
        "Because an empty ids list is falsy, so the filter was never appended.",
        session_id="s1",
    )
    provider.on_session_end(MESSAGES)
    provider.on_delegation("A delegated task", "A delegated result worth remembering")
    provider.queue_prefetch(QUERY, session_id="s1")
    provider.shutdown()

    rejected = [r for r in caplog.records if "API key rejected" in r.getMessage()]
    assert len(rejected) == 1, [r.getMessage() for r in caplog.records]
    assert_no_key(caplog)


def test_the_auth_warning_re_arms_after_a_session_reset(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(_client, "requests", StaticRequests(lambda: FakeResponse(401, {})))
    provider = make_provider(tmp_path)

    provider.prefetch(QUERY, session_id="s1")
    provider.on_session_switch("s2", parent_session_id="s1", reset=True)
    provider.prefetch(QUERY, session_id="s2")
    provider.shutdown()

    rejected = [r for r in caplog.records if "API key rejected" in r.getMessage()]
    assert len(rejected) == 2
    assert_no_key(caplog)


# -- timeouts ----------------------------------------------------------------


def test_the_cold_turn_path_sends_no_rerank_and_the_turn_budget(tmp_path, monkeypatch, caplog):
    """Asserted at the wire, not at the client: this is what prod receives."""
    fake = ExplodingRequests(lambda: requests.exceptions.Timeout("read timed out"))
    monkeypatch.setattr(_client, "requests", fake)
    provider = make_provider(tmp_path)

    provider.prefetch(QUERY, session_id="s1")
    provider.shutdown()

    assert fake.calls, "prefetch never reached the transport"
    for verb, _url, kwargs in fake.calls:
        assert verb == "get"
        assert kwargs["timeout"] == (CONNECT_TIMEOUT, READ_TIMEOUT)
        assert kwargs["params"]["rerank"] == "false"
    assert_no_key(caplog)


def test_the_background_warm_up_gets_the_off_turn_timeout(tmp_path, monkeypatch, caplog):
    """A reranked search really takes ~4.5 s: warming with the 3 s turn budget
    filled the cache never, so the plugin injected nothing in production."""
    fake = ExplodingRequests(lambda: requests.exceptions.Timeout("read timed out"))
    monkeypatch.setattr(_client, "requests", fake)
    provider = make_provider(tmp_path)

    provider.queue_prefetch(QUERY, session_id="s1")
    provider.shutdown()

    assert fake.calls, "the warm-up never reached the transport"
    for _verb, _url, kwargs in fake.calls:
        assert kwargs["timeout"] == (CONNECT_TIMEOUT, SLOW_READ_TIMEOUT)
        # The configured value survives here — only the cold path overrides it.
        assert kwargs["params"]["rerank"] == "true"
    assert SLOW_READ_TIMEOUT > READ_TIMEOUT
    assert_no_key(caplog)


def test_the_search_tool_gets_the_off_turn_timeout(tmp_path, monkeypatch, caplog):
    fake = ExplodingRequests(lambda: requests.exceptions.Timeout("read timed out"))
    monkeypatch.setattr(_client, "requests", fake)
    provider = make_provider(tmp_path)

    provider.handle_tool_call("recall_search", {"query": QUERY})
    provider.shutdown()

    off_turn = (CONNECT_TIMEOUT, SLOW_READ_TIMEOUT)
    assert [kwargs["timeout"] for _v, _u, kwargs in fake.calls] == [off_turn]
    assert [kwargs["params"]["rerank"] for _v, _u, kwargs in fake.calls] == ["true"]
    assert_no_key(caplog)


def test_the_write_path_passes_the_write_timeout(tmp_path, monkeypatch, caplog):
    fake = ExplodingRequests(lambda: requests.exceptions.ConnectionError("refused"))
    monkeypatch.setattr(_client, "requests", fake)
    provider = make_provider(tmp_path)

    provider.handle_tool_call("recall_store", {"content": "a durable fact about the extractor"})
    provider.shutdown()

    posts = [c for c in fake.calls if c[0] == "post"]
    assert len(posts) == 2, "store retries exactly once on a transport failure"
    for _verb, _url, kwargs in posts:
        assert kwargs["timeout"] == (CONNECT_TIMEOUT, WRITE_TIMEOUT)
    assert_no_key(caplog)


def test_the_key_travels_in_the_header_and_nowhere_else(tmp_path, monkeypatch, caplog):
    fake = StaticRequests(lambda: FakeResponse(200, {"items": []}))
    monkeypatch.setattr(_client, "requests", fake)
    provider = make_provider(tmp_path)

    provider.prefetch(QUERY, session_id="s1")
    provider.shutdown()

    verb, url, kwargs = fake.calls[0]
    assert kwargs["headers"]["Authorization"] == f"ApiKey {API_KEY}"
    assert API_KEY not in url
    assert API_KEY not in json.dumps(kwargs.get("params", {}))
    assert_no_key(caplog)


def test_prefetch_does_not_wait_past_a_released_gate(tmp_path, monkeypatch, caplog):
    gate = GatedRequests()
    gate.released.set()  # released up front: nothing ever sleeps
    monkeypatch.setattr(_client, "requests", gate)
    provider = make_provider(tmp_path)

    started = time.monotonic()
    block = provider.prefetch(QUERY, session_id="s1")
    elapsed = time.monotonic() - started
    provider.shutdown()

    assert block == ""
    assert elapsed < 1.0
    assert_no_key(caplog, block)


def test_queue_prefetch_returns_while_the_transport_is_still_gated(tmp_path, monkeypatch, caplog):
    gate = GatedRequests()
    monkeypatch.setattr(_client, "requests", gate)
    provider = make_provider(tmp_path)

    started = time.monotonic()
    provider.queue_prefetch(QUERY, session_id="s1")
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, "queue_prefetch blocked on a wedged transport"
    assert gate.entered.wait(timeout=5.0), "the background search never started"
    # A turn that runs while the warm-up is still in flight must not wait either.
    assert provider.recall_status() is None

    gate.released.set()
    provider.shutdown()
    assert_no_key(caplog)


# -- wrong shapes ------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"items": {"snippet": "a dict, not a list"}},
        {"items": "a string"},
        {"items": [None, 3, [], "x"]},
        {"items": [{"snippet": None}, {"nothing": "useful"}]},
        {"items": None},
        {},
        {"items": [{"snippet": "   "}]},
    ],
)
def test_a_wrong_shape_search_response_yields_an_empty_block(
    payload, tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(_client, "requests", StaticRequests(lambda: FakeResponse(200, payload)))
    provider = make_provider(tmp_path)

    block = provider.prefetch(QUERY, session_id="s1")
    status = provider.recall_status()
    tool = json.loads(provider.handle_tool_call("recall_search", {"query": QUERY}))
    provider.shutdown()

    assert block == ""
    assert status is None
    assert tool["count"] == 0
    assert_no_key(caplog, block, tool)


@pytest.mark.parametrize(
    "payload",
    [{"memory_id": None}, {"id": 12345}, {}, {"memory_id": {"nested": True}}],
)
def test_a_wrong_shape_store_response_still_reports_success(payload, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(_client, "requests", StaticRequests(lambda: FakeResponse(201, payload)))
    provider = make_provider(tmp_path)

    raw = provider.handle_tool_call("recall_store", {"content": "a durable fact about extraction"})
    provider.shutdown()

    result = json.loads(raw)
    assert result["stored"] is True
    assert isinstance(result["memory_id"], str)
    assert_no_key(caplog, raw)


def test_the_two_read_paths_differ_only_where_they_must(tmp_path, monkeypatch, caplog):
    """Cold = unreranked inside the turn budget; warm = configured, off-turn."""
    fake = StaticRequests(lambda: FakeResponse(200, {"items": []}))
    monkeypatch.setattr(_client, "requests", fake)
    provider = make_provider(tmp_path)

    provider.prefetch(QUERY, session_id="s1")
    provider.queue_prefetch(QUERY, session_id="s1")
    provider.shutdown()

    cold, warm = fake.calls[0][2], fake.calls[1][2]
    assert (cold["params"]["rerank"], cold["timeout"]) == ("false", (CONNECT_TIMEOUT, READ_TIMEOUT))
    off_turn = (CONNECT_TIMEOUT, SLOW_READ_TIMEOUT)
    assert (warm["params"]["rerank"], warm["timeout"]) == ("true", off_turn)
    # Everything else is identical: only the two knobs move.
    assert cold["params"]["query"] == warm["params"]["query"]
    assert cold["params"]["limit"] == warm["params"]["limit"]
    assert cold["params"]["graph_boost"] == warm["params"]["graph_boost"]
    assert_no_key(caplog)


def test_shutdown_bumps_the_cache_generation(tmp_path, monkeypatch, caplog):
    """Work abandoned at the 5 s join must not write back after the fact."""
    fake = StaticRequests(lambda: FakeResponse(200, {"items": []}))
    monkeypatch.setattr(_client, "requests", fake)
    provider = make_provider(tmp_path)
    generation = provider._cache_generation

    provider.shutdown()

    assert provider._cache_generation != generation
    assert_no_key(caplog)
