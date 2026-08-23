"""RecallClient's three read-only extras: graph_recall, who_knows, stats.

Same conventions as ``test_client_search.py``: the transport seam is
``recall._client.requests``, nothing here touches the network, and no failure
mode may ever put the API key in a message.
"""

import pytest
from recall import _client
from recall._client import RecallAuthError, RecallClient, RecallError


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = "<body withheld>"

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeRequests:
    """Stand-in for the ``requests`` module inside ``recall._client``."""

    def __init__(self, response=None, raiser=None):
        self.response = response
        self.raiser = raiser
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if self.raiser is not None:
            raise self.raiser
        return self.response

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if self.raiser is not None:
            raise self.raiser
        return self.response



class RoutedRequests:
    """Answers per-path, so the two-call ``stats()`` can be steered."""

    def __init__(self, routes, raisers=None):
        self.routes = routes
        self.raisers = raisers or {}
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        for path, exc in self.raisers.items():
            if url.endswith(path):
                raise exc
        for path, response in self.routes.items():
            if url.endswith(path):
                return response
        raise AssertionError(f"unrouted GET {url}")

    def post(self, url, **kwargs):  # pragma: no cover - reads only
        raise AssertionError("the extras are read-only")


@pytest.fixture()
def client():
    return RecallClient(api_key="rag_testkey", base_url="https://recall.example")


# -- graph_recall ------------------------------------------------------------


def test_graph_recall_sends_the_documented_parameters(monkeypatch, client):
    fake = FakeRequests(FakeResponse(200, {"entities": []}))
    monkeypatch.setattr(_client, "requests", fake)

    client.graph_recall("marker extraction", depth=3, limit=7)

    method, url, kwargs = fake.calls[0]
    assert method == "GET"
    assert url == "https://recall.example/api/v1/graph/recall"
    assert kwargs["params"] == {
        "q": "marker extraction",
        "depth": 3,
        "include_relations": "true",
        "limit": 7,
    }
    assert kwargs["timeout"] == (_client.CONNECT_TIMEOUT, _client.READ_TIMEOUT)
    assert kwargs["headers"]["Authorization"] == "ApiKey rag_testkey"


def test_graph_recall_defaults_are_depth_2_limit_10(monkeypatch, client):
    fake = FakeRequests(FakeResponse(200, {}))
    monkeypatch.setattr(_client, "requests", fake)

    client.graph_recall("q")

    assert fake.calls[0][2]["params"]["depth"] == 2
    assert fake.calls[0][2]["params"]["limit"] == 10


def test_graph_recall_clamps_depth_and_limit_to_the_server_range(monkeypatch, client):
    fake = FakeRequests(FakeResponse(200, {}))
    monkeypatch.setattr(_client, "requests", fake)

    client.graph_recall("q", depth=99, limit=999)
    client.graph_recall("q", depth=0, limit=0)

    assert (fake.calls[0][2]["params"]["depth"], fake.calls[0][2]["params"]["limit"]) == (5, 50)
    assert (fake.calls[1][2]["params"]["depth"], fake.calls[1][2]["params"]["limit"]) == (1, 1)


def test_graph_recall_truncates_a_very_long_query(monkeypatch, client):
    fake = FakeRequests(FakeResponse(200, {}))
    monkeypatch.setattr(_client, "requests", fake)

    client.graph_recall("x" * 9000)

    assert len(fake.calls[0][2]["params"]["q"]) == _client.MAX_QUERY_CHARS


def test_graph_recall_honours_an_explicit_timeout(monkeypatch, client):
    fake = FakeRequests(FakeResponse(200, {}))
    monkeypatch.setattr(_client, "requests", fake)

    client.graph_recall("q", timeout=_client.SLOW_READ_TIMEOUT)

    assert fake.calls[0][2]["timeout"] == (
        _client.CONNECT_TIMEOUT,
        _client.SLOW_READ_TIMEOUT,
    )


def test_graph_recall_accepts_a_list_payload(monkeypatch, client):
    monkeypatch.setattr(
        _client, "requests", FakeRequests(FakeResponse(200, [{"entity": "recall"}]))
    )

    assert client.graph_recall("q") == [{"entity": "recall"}]


def test_graph_recall_rejects_a_scalar_payload(monkeypatch, client):
    monkeypatch.setattr(_client, "requests", FakeRequests(FakeResponse(200, 42)))

    with pytest.raises(RecallError):
        client.graph_recall("q")


def test_graph_recall_without_api_key_or_query_makes_no_call(monkeypatch):
    fake = FakeRequests(FakeResponse(200, {"entities": [1]}))
    monkeypatch.setattr(_client, "requests", fake)

    assert RecallClient(api_key="", base_url="https://x").graph_recall("q") == {}
    assert RecallClient(api_key="k", base_url="https://x").graph_recall("   ") == {}
    assert fake.calls == []


def test_graph_recall_maps_status_codes_like_search(monkeypatch, client):
    monkeypatch.setattr(_client, "requests", FakeRequests(FakeResponse(401, {})))
    with pytest.raises(RecallAuthError):
        client.graph_recall("q")

    monkeypatch.setattr(_client, "requests", FakeRequests(FakeResponse(500, {})))
    with pytest.raises(RecallError):
        client.graph_recall("q")

    monkeypatch.setattr(_client, "requests", FakeRequests(raiser=OSError("down")))
    with pytest.raises(RecallError):
        client.graph_recall("q")


def test_graph_recall_makes_exactly_one_call_no_retry(monkeypatch, client):
    fake = FakeRequests(raiser=OSError("down"))
    monkeypatch.setattr(_client, "requests", fake)

    with pytest.raises(RecallError):
        client.graph_recall("q")

    assert len(fake.calls) == 1


# -- who_knows ---------------------------------------------------------------


def test_who_knows_sends_the_documented_parameters(monkeypatch, client):
    fake = FakeRequests(FakeResponse(200, {"people": []}))
    monkeypatch.setattr(_client, "requests", fake)

    client.who_knows("neo4j", n_results=9)

    method, url, kwargs = fake.calls[0]
    assert method == "GET"
    assert url == "https://recall.example/api/v1/graph/who-knows"
    assert kwargs["params"] == {"topic": "neo4j", "n_results": 9}
    assert kwargs["timeout"] == (_client.CONNECT_TIMEOUT, _client.READ_TIMEOUT)


def test_who_knows_defaults_to_five_and_clamps(monkeypatch, client):
    fake = FakeRequests(FakeResponse(200, {}))
    monkeypatch.setattr(_client, "requests", fake)

    client.who_knows("t")
    client.who_knows("t", n_results=99)
    client.who_knows("t", n_results=0)

    assert [c[2]["params"]["n_results"] for c in fake.calls] == [5, 20, 1]


def test_who_knows_truncates_a_very_long_topic(monkeypatch, client):
    fake = FakeRequests(FakeResponse(200, {}))
    monkeypatch.setattr(_client, "requests", fake)

    client.who_knows("t" * 9000)

    assert len(fake.calls[0][2]["params"]["topic"]) == _client.MAX_QUERY_CHARS


def test_who_knows_without_api_key_or_topic_makes_no_call(monkeypatch):
    fake = FakeRequests(FakeResponse(200, {"people": [1]}))
    monkeypatch.setattr(_client, "requests", fake)

    assert RecallClient(api_key="", base_url="https://x").who_knows("t") == {}
    assert RecallClient(api_key="k", base_url="https://x").who_knows(" ") == {}
    assert fake.calls == []


def test_who_knows_maps_errors(monkeypatch, client):
    monkeypatch.setattr(_client, "requests", FakeRequests(FakeResponse(403, {})))
    with pytest.raises(RecallAuthError):
        client.who_knows("t")

    monkeypatch.setattr(_client, "requests", FakeRequests(FakeResponse(200, ["a list"])))
    with pytest.raises(RecallError):
        client.who_knows("t")


def test_who_knows_honours_an_explicit_timeout(monkeypatch, client):
    fake = FakeRequests(FakeResponse(200, {}))
    monkeypatch.setattr(_client, "requests", fake)

    client.who_knows("t", timeout=_client.SLOW_READ_TIMEOUT)

    assert fake.calls[0][2]["timeout"] == (
        _client.CONNECT_TIMEOUT,
        _client.SLOW_READ_TIMEOUT,
    )


# -- stats -------------------------------------------------------------------


def test_stats_merges_the_graph_numbers_under_a_graph_key(monkeypatch, client):
    fake = RoutedRequests(
        {
            "/api/v1/search/stats": FakeResponse(200, {"total_memories": 278}),
            "/api/v1/graph/stats": FakeResponse(200, {"entities": 500, "relations": 900}),
        }
    )
    monkeypatch.setattr(_client, "requests", fake)

    payload = client.stats()

    assert payload["total_memories"] == 278
    assert payload["graph"] == {"entities": 500, "relations": 900}
    search_call = [c for c in fake.calls if c[1].endswith("/api/v1/search/stats")][0]
    assert search_call[2]["params"] == {"scope": "all"}


def test_stats_degrades_when_the_graph_call_fails(monkeypatch, client):
    fake = RoutedRequests(
        {"/api/v1/search/stats": FakeResponse(200, {"total_memories": 12})},
        raisers={"/api/v1/graph/stats": OSError("neo4j down")},
    )
    monkeypatch.setattr(_client, "requests", fake)

    payload = client.stats()

    assert payload == {"total_memories": 12}
    assert "graph" not in payload


def test_stats_degrades_when_the_graph_call_answers_5xx(monkeypatch, client):
    fake = RoutedRequests(
        {
            "/api/v1/search/stats": FakeResponse(200, {"total_memories": 12}),
            "/api/v1/graph/stats": FakeResponse(503, {"detail": "down"}),
        }
    )
    monkeypatch.setattr(_client, "requests", fake)

    assert client.stats() == {"total_memories": 12}


def test_stats_raises_when_the_search_stats_call_fails(monkeypatch, client):
    monkeypatch.setattr(_client, "requests", FakeRequests(FakeResponse(500, {})))

    with pytest.raises(RecallError):
        client.stats()


def test_stats_without_api_key_makes_no_call(monkeypatch):
    fake = FakeRequests(FakeResponse(200, {"total_memories": 1}))
    monkeypatch.setattr(_client, "requests", fake)

    assert RecallClient(api_key="", base_url="https://x").stats() == {}
    assert fake.calls == []


def test_stats_honours_an_explicit_timeout_on_both_calls(monkeypatch, client):
    fake = RoutedRequests(
        {
            "/api/v1/search/stats": FakeResponse(200, {}),
            "/api/v1/graph/stats": FakeResponse(200, {}),
        }
    )
    monkeypatch.setattr(_client, "requests", fake)

    client.stats(timeout=_client.SLOW_READ_TIMEOUT)

    off_turn = (_client.CONNECT_TIMEOUT, _client.SLOW_READ_TIMEOUT)
    assert [c[2]["timeout"] for c in fake.calls] == [off_turn, off_turn]


def test_the_extras_never_put_the_api_key_in_an_error(monkeypatch):
    secret = "rag_" + "extra-5ecret-" + "0123456789"
    client = RecallClient(api_key=secret, base_url="https://recall.example")
    monkeypatch.setattr(_client, "requests", FakeRequests(FakeResponse(401, {})))

    for call in (
        lambda: client.graph_recall("q"),
        lambda: client.who_knows("t"),
        lambda: client.stats(),
    ):
        with pytest.raises(RecallError) as excinfo:
            call()
        assert secret not in str(excinfo.value)
