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
| `rerank` | `True` | cross-encoder rerank on the ML API GPU |
| `graph_boost` | `False` | graph entity boost (adds a Neo4j round-trip) |
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

## What you get

**Before each turn** — a short block, one line per memory:

```
Relevant memories (Recall):
- [decision, 2026-07-21] Marker extraction moved back to GPU with page-by-page chunking…
- [bugfix, 2026-07-10] pgvector delete() with an empty ids list wiped the collection…
```

Hermes shows a 🧠 indicator on turns where memories were actually injected.

**After each turn** — the user/assistant pair is stored as a `context` memory,
tagged `hermes`, `session:<id>`, `platform:<platform>`, `agent:<profile>` —
but only when `sync_turns` is on and the turn clears `min_chars`.

**At session end** — one condensed synthesis (topics, decisions, facts), tagged
`session-summary` — only when `session_summary` is on.

**Before context compression** — the discarded messages are archived (tag
`pre-compress`) *and* the extracted insights are handed to the compressor, so
what Recall judged worth keeping survives compression.

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
scheduled or delegated run can search memory but cannot add to it.

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
  `min_chars`, `max_chars`, `sync_turns` and `session_summary` are the
  throttles — turn them down before turning writes off entirely.
- Background writes run on daemon threads, capped at 16 concurrently in
  flight; beyond that a wedged Recall causes writes to be dropped (logged),
  never queued or awaited on the turn path.

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

## License

MIT
