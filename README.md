# hermes-recall

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) **memory provider
plugin** that makes a [Recall](https://recall.carnival-devops.com) tenant your
agent's persistent cross-session memory.

Relevant memories are injected before each turn, substantive turns are written
back, a synthesis is stored when the session ends, and the model gets two
explicit tools. Direct REST — no MCP hop, no local backend process, and **no
Python dependencies** beyond `requests` (already shipped by Hermes), so the
whole thing installs from the dashboard.

## Install

### From the dashboard (recommended, zero terminal)

1. Hermes dashboard → **Plugins** → **Install from GitHub / Git URL**
2. Paste `tarpediem/hermes-recall` → **Install**
3. **Memory provider** dropdown → select **`recall`**
4. In the config panel, paste your Recall API key (`rag_…`) → **Save**

Hermes names the plugin's install directory after `plugin.yaml`'s `name` field
(`recall`), so a dashboard install always lands at
`$HERMES_HOME/plugins/recall/` and registers correctly. No pip, no file
editing. The key is written to the profile's `.env` as `RECALL_API_KEY`;
optional settings are saved to `$HERMES_HOME/recall/config.json` by the same
panel.

To get a key: log in at https://recall.carnival-devops.com → **Settings** →
**API keys** → create one.

### Manual install (git clone / copy)

Hermes keys the provider name, the `memory.provider` value, the skill
namespace (`recall:memory`) and the dashboard config panel on the **plugin
directory's name** — not on anything inside `plugin.yaml`. Clone (or copy)
this repo directly into `$HERMES_HOME/plugins/` **as `recall`**:

```bash
git clone https://github.com/tarpediem/hermes-recall "$HERMES_HOME/plugins/recall"
```

A clone left named `hermes-recall` (e.g. `git clone …` without the trailing
path argument) registers under the provider name `hermes-recall` instead, and
`memory.provider: recall` in Hermes' config will not find it. Then set
`RECALL_API_KEY` in the environment or profile `.env`, and `memory.provider:
recall` in Hermes' config.

## Configuration

Only the API key is required. Everything else is read from
`$HERMES_HOME/recall/config.json` (the file the dashboard's config panel
writes) merged over these defaults:

| key | default | meaning |
|---|---|---|
| `base_url` | `https://recall.carnival-devops.com` | API root. `RECALL_BASE_URL` env overrides it. |
| `limit` | `5` | memories injected per turn |
| `rerank` | `True` | cross-encoder rerank on the ML API GPU (background warm-up only; the first turn of a session always searches unreranked) |
| `graph_boost` | `False` | graph entity boost (adds a Neo4j round-trip) |
| `writes_enabled` | `True` | **master switch.** `false` makes the plugin read-only: no turn, no synthesis, no pre-compress archive, no delegation, no built-in mirror, and `recall_store` returns an error |
| `sync_turns` | `True` | write completed turns |
| `session_summary` | `True` | write an end-of-session synthesis |
| `max_chars` | `4000` | truncation cap on any stored content |
| `min_chars` | `40` | minimum combined turn length worth storing |

Example `$HERMES_HOME/recall/config.json`:

```json
{
  "limit": 8,
  "sync_turns": true,
  "session_summary": true,
  "max_chars": 4000,
  "min_chars": 60
}
```

## Check it's working

You do not need a terminal for this.

1. Tell the agent something worth remembering ("I always want commit messages
   in English"), then let the session end.
2. Start a **new** session and ask about it — "what did I say about commit
   messages?"
3. On a turn where memories were found you will see the **🧠** indicator, and
   the agent answers from what you told it earlier. Memories reach the model
   in a block that starts with `Relevant memories (Recall):`.

**Restart the agent (or the gateway) after saving the key.** It is read from
the environment at start-up, so a key saved into a running agent is not picked
up until it restarts.

If the 🧠 never appears at all, look in the agent log for:

```
Recall API key rejected
```

That one line means the key is wrong or expired — everything else keeps
working, you just get no memory.

## Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| 🧠 never appears; `Recall API key rejected` in the log | wrong, expired or revoked API key | create a new key (Recall → **Settings** → **API keys**), save it in the dashboard panel, **restart the agent** |
| 🧠 never appears; no `rejected` line, but `Recall search failed` in the log | Recall is unreachable (instance down, wrong `base_url`, no network) | the agent keeps working without memory; **writes made while it is unreachable are lost, not queued**. Check `base_url` and that the instance answers |
| Nothing on the first turn of a session, memories from the second turn on | Recall answered slower than the turn budget | expected — the first turn searches synchronously, later turns are served from a background warm-up. Persistent? lower `limit`, or turn `rerank` off |
| Memories appear but nothing new is ever stored | `writes_enabled` is off, or the agent is running in a non-primary context (`subagent`, `cron`, `flush`) | set `writes_enabled: true`; non-primary contexts are read-only by design |

## What you get

**Before each turn** — a short block, one line per memory:

```
Relevant memories (Recall):
- [decision, 2026-07-21] Marker extraction moved back to GPU with page-by-page chunking…
- [bugfix, 2026-07-10] pgvector delete() with an empty ids list wiped the collection…
```

Hermes shows a 🧠 indicator on turns where memories were actually injected.

**The first turn of a session uses an unreranked search** so it fits the turn
budget; every turn after it is served from a reranked background warm-up
started during the previous turn. A reranked query costs ~4.5 s against a real
instance, which no turn-path budget can absorb — and the first turn of a
session is exactly where cross-session recall matters most, so it gets a
slightly worse-ranked block instead of none at all. Same rule whenever the
warm-up has not landed yet.

**After each turn** — the user/assistant pair is stored as a `context` memory,
tagged `hermes`, `session:<id>`, `platform:<platform>`, `agent:<profile>` —
but only when `sync_turns` is on and the turn clears `min_chars`.

**At session end** — one condensed synthesis (topics, decisions, facts), tagged
`session-summary` — only when `session_summary` is on.

**Before context compression** — the discarded messages are archived (tag
`pre-compress`, under `session_summary`, since it is the same kind of
synthesis) *and* the extracted insights are handed to the compressor, so what
Recall judged worth keeping survives compression. The insight block is built
from messages already in context and stores nothing, so it is still returned
when writes are off.

**A built-in memory write also mirrors to Recall** — `add`/`replace` on
Hermes' own memory tool is stored as a `preference` (user profile) or
`context` (agent memory) entry tagged `builtin-mirror`; `remove` is a
documented no-op, nothing is ever deleted from Recall.

**A delegation result is stored too** — the task text and the returned result
are combined into one `context` memory tagged `delegation`.

**Two tools** the model can call explicitly:

| tool | parameters |
|---|---|
| `recall_search` | `query` (required), `limit` (default `5`) |
| `recall_store` | `content` (required), `memory_type` (default `context`), `tags` (default `[]`) |

The plugin also registers a `recall:memory` skill telling the model when to
search and what is worth storing. It loads only while `recall` is the active
memory provider.

## What leaves the device

Only the **user message and the assistant reply** for each turn, plus a
condensed synthesis built from those same pairs, plus whatever content a
`recall_store` tool call explicitly contains.

**Tool calls and tool results are never transmitted.** Hermes offers the full
message list to `sync_turn`; this provider accepts it and deliberately
ignores it, so workspace paths, command output and file dumps cannot reach
Recall through the turn path. The one thing that can reach Recall is what you
or the model type.

Non-primary agent contexts (`subagent`, `cron`, `flush`) **read but never
write** — only `initialize(agent_context="primary")` enables writes, so a
scheduled or delegated run can search memory but cannot add to it. Setting
`writes_enabled: false` makes every context read-only, including the primary
one.

Nothing is stored outside `$HERMES_HOME` (`backup_paths()` is empty), and the
API key is never logged.

## Compatibility

| Hermes | status |
|---|---|
| ≥ **0.20.4** | fully supported — every hook fires |
| **0.19.1** | works, degraded — `on_session_switch`, `recall_status` (the 🧠 indicator) and `backup_paths` are never called by that version's `MemoryManager`; `on_memory_write` is still called (3-positional-arg form), because that call site predates 0.20.4 |
| < 0.19 | not supported |

## Limitations

- **No offline queue.** A write that fails while Recall is unreachable is
  logged and lost — it is not retried later.
- **Reads pause when Recall is unreachable.** After 3 consecutive read
  failures the synchronous turn-path search is skipped for 60 s, so a dead
  host does not tax every turn. Background warm-ups keep probing and the first
  success clears the pause at once.
- **No deletion.** `remove` on the built-in memory tool is a no-op here;
  mirrored entries are never deleted from Recall.
- **Fail open, always.** Every failure degrades to "no memory this turn",
  never to a broken turn. A rejected API key is logged **once per session**,
  not once per turn.

## Operator notes

- `RECALL_BASE_URL` (env) or the `base_url` field points the plugin at a
  Tailscale FQDN or a LAN instance, to validate against preprod before
  rolling out to every agent — change nothing else.
- **Validate on one agent first.** Point a single profile at Recall, watch it
  for a session or two, then flip `memory.provider` on the rest.
- **Rollback is instant**: set `memory.provider: ''` (or pick the built-in
  provider from the dashboard dropdown). No data migration, nothing to undo —
  the plugin never touched anything outside `$HERMES_HOME/recall/`.
- **Write volume is the thing to watch**: each stored memory costs an
  embedding, an entity extraction and a conflict-detection pass server-side.
  Each key governs exactly one thing:

  | key | what it controls |
  |---|---|
  | `writes_enabled` | **everything.** `false` = read-only: no turn, no synthesis, no pre-compress archive, no delegation memory, no built-in mirror, and `recall_store` returns an error |
  | `sync_turns` | the per-turn write only |
  | `session_summary` | the end-of-session synthesis **and** the pre-compress archive (same kind of memory, same switch) |
  | `min_chars` | the floor under which a turn is not written |
  | `max_chars` | truncation of whatever *is* written — it never suppresses a write |

  So the four throttles reduce write volume; only `writes_enabled` turns
  writes off. The delegation memory and the built-in-memory mirror have no
  throttle of their own — `writes_enabled` is the only key that stops them.
- **`rerank` costs latency, and where it is paid matters.** A reranked search
  measures ~4.5 s against the public instance (cross-encoder round-trip on the
  ML API GPU) versus ~0.3 s without. The background warm-up and the
  `recall_search` tool are given a 10 s budget and absorb it; the synchronous
  turn-path fallback keeps a 3 s read budget plus a 1.5 s connect budget
  (`requests` bills connect and read separately, so it is not a single hard
  3 s), and Hermes 0.20.4 additionally caps the whole prefetch at 8 s on its
  side. That is why the cold path drops rerank rather than inject nothing (see
  *What you get*). Turning `rerank` off makes every turn behave like the first
  one: faster, ranked slightly worse.
- **When Recall is unreachable the plugin stops trying for a minute.** After 3
  consecutive read failures the synchronous turn-path search is skipped for 60
  seconds — an unreachable host would otherwise cost every turn its connect
  budget. Background warm-ups keep trying (they are off the turn path, so
  their latency is nobody's wait) and the first one that succeeds clears the
  pause immediately. It is logged once, when it starts, not once per turn.
- Background work runs on daemon threads, capped at 16 concurrently in
  flight; beyond that a wedged Recall causes work to be dropped, never queued
  or awaited on the turn path. The log line names what was dropped — a
  `prefetch warm-up` (costs one turn its memory block) or a `write` (loses a
  memory for good) — at most once per 30 s per kind, so a wedged instance
  cannot flood the log.

## Development

```bash
cd hermes-recall
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

The unit suite never touches the network — `requests` is monkeypatched at
`recall._client.requests`. `HERMES_AGENT_SRC` overrides where the Hermes agent
source lives (default `~/.hermes/hermes-agent`), which the test harness needs
on `sys.path` to import `agent.memory_provider` and `plugins.memory.config_schema`.

`tests/test_fail_open.py` is the transport-level sweep: it breaks `requests`
itself — connection errors, timeouts, 401/403/500/502, non-JSON bodies,
wrong-shaped payloads — and asserts that every public hook still returns its
neutral value and that the API key never reaches a log record.

The live integration test (`tests/test_integration_live.py`) is the one
exception to "no network", and it runs only when `RECALL_TEST_API_KEY` is set:

```bash
RECALL_TEST_API_KEY=rag_… .venv/bin/python -m pytest tests/test_integration_live.py -v
```

`RECALL_TEST_BASE_URL` points it at another instance (default the public one).
It writes one memory per run, tagged **`hermes-recall-live-test`** plus
`marker:<id>`; nothing is ever deleted (the plugin has no delete path), so
purge that tag when the test memories pile up.

## License

MIT
