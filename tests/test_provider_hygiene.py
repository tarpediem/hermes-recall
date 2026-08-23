"""Small guarantees the final review asked for before rollout.

* the drop warning says WHAT was dropped, and cannot flood the log
* a rewound session cannot be resurrected by an in-flight warm-up
* the vestigial ``_prefetch_counts`` registry is gone
* tool schemas handed out are copies, not the module constants
"""

import pytest
from recall import _provider as provider_module
from recall._client import RecallClient
from recall._provider import (
    DROP_WARNING_INTERVAL_SECONDS,
    SEARCH_TOOL_SCHEMA,
    STORE_TOOL_SCHEMA,
    RecallMemoryProvider,
)


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture()
def provider(tmp_path):
    p = RecallMemoryProvider(RecallClient(api_key="rag_k", base_url="https://recall.example"))
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    return p


@pytest.fixture()
def full_pool(monkeypatch):
    """Every _spawn drops: the pool is 'full' at zero live threads."""
    monkeypatch.setattr(provider_module, "MAX_LIVE_THREADS", 0)


# -- (a) the drop warning ----------------------------------------------------


def test_the_drop_warning_names_what_was_dropped(provider, full_pool, monkeypatch, caplog):
    monkeypatch.setattr(provider_module, "time", FakeClock())

    with caplog.at_level("WARNING"):
        provider._spawn("prefetch", lambda: None)
        provider._spawn("sync", lambda: None)

    messages = [r.getMessage() for r in caplog.records if "dropped" in r.getMessage()]
    assert len(messages) == 2
    assert any("prefetch" in m for m in messages)
    assert any("write" in m for m in messages)


def test_the_drop_warning_is_rate_limited_per_kind(provider, full_pool, monkeypatch, caplog):
    """A wedged Recall drops on every turn; one line per 30 s is a signal,
    one line per drop is a flood that buries everything else."""
    clock = FakeClock()
    monkeypatch.setattr(provider_module, "time", clock)

    with caplog.at_level("WARNING"):
        for _ in range(20):
            provider._spawn("sync", lambda: None)
            clock.advance(1.0)

    dropped = [r for r in caplog.records if "dropped" in r.getMessage()]
    assert len(dropped) == 1


def test_the_drop_warning_returns_after_the_interval(provider, full_pool, monkeypatch, caplog):
    clock = FakeClock()
    monkeypatch.setattr(provider_module, "time", clock)

    with caplog.at_level("WARNING"):
        provider._spawn("sync", lambda: None)
        clock.advance(DROP_WARNING_INTERVAL_SECONDS + 0.1)
        provider._spawn("sync", lambda: None)

    assert len([r for r in caplog.records if "dropped" in r.getMessage()]) == 2


def test_the_two_kinds_are_rate_limited_independently(provider, full_pool, monkeypatch, caplog):
    monkeypatch.setattr(provider_module, "time", FakeClock())

    with caplog.at_level("WARNING"):
        provider._spawn("sync", lambda: None)
        provider._spawn("memwrite", lambda: None)  # same kind: suppressed
        provider._spawn("prefetch", lambda: None)  # other kind: logged

    messages = [r.getMessage() for r in caplog.records if "dropped" in r.getMessage()]
    assert len(messages) == 2


# -- (b) a rewind invalidates in-flight warm-ups -----------------------------


def test_a_rewind_bumps_the_cache_generation(provider):
    """The turn being rewound away may already have a search in flight; when
    it lands it must discard its block instead of writing it back."""
    generation = provider._cache_generation

    provider.on_session_switch("s1", rewound=True)

    assert provider._cache_generation != generation


def test_a_warm_up_in_flight_across_a_rewind_does_not_write_back(provider):
    generation = provider._cache_generation
    provider.on_session_switch("s1", rewound=True)

    # The warm-up finishes now, carrying the generation it was spawned under.
    provider._client.search = lambda *a, **k: [
        {"id": "m1", "type": "context", "timestamp": "2026-07-10", "snippet": "rewound away"}
    ]
    provider._search_and_cache("q", "s1", generation, None)

    assert "s1" not in provider._prefetch_cache


def test_a_plain_switch_does_not_bump_the_generation(provider):
    """A subagent handoff is not a boundary — it must stay O(1)."""
    generation = provider._cache_generation

    provider.on_session_switch("s2")

    assert provider._cache_generation == generation


# -- (c) no vestigial registry ----------------------------------------------


def test_the_vestigial_prefetch_counts_registry_is_gone(provider):
    assert not hasattr(provider, "_prefetch_counts")


# -- (d) tool schemas are copies --------------------------------------------


def test_get_tool_schemas_returns_copies(provider):
    first = provider.get_tool_schemas()
    first[0]["description"] = "mutated"
    first[1]["parameters"]["properties"]["content"]["description"] = "mutated too"

    second = provider.get_tool_schemas()

    assert second[0]["description"] == SEARCH_TOOL_SCHEMA["description"]
    assert "mutated" not in SEARCH_TOOL_SCHEMA["description"]
    assert (
        second[1]["parameters"]["properties"]["content"]["description"]
        == STORE_TOOL_SCHEMA["parameters"]["properties"]["content"]["description"]
    )
