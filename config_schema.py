"""Recall's declared config surface — rendered by the generic dashboard panel."""

from plugins.memory.config_schema import (
    KIND_BOOL,
    KIND_NUMBER,
    KIND_SECRET,
    KIND_TEXT,
    ProviderConfigSchema,
    ProviderField,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="recall",
    label="Recall",
    docs_url="https://github.com/tarpediem/hermes-recall#configuration",
    fields=(
        ProviderField(
            key="api_key",
            label="API key",
            kind=KIND_SECRET,
            env_key="RECALL_API_KEY",
            description="Authenticates every call to the Recall API.",
            placeholder="rag_…",
            inline=True,
        ),
        ProviderField(
            key="base_url",
            label="API URL",
            kind=KIND_TEXT,
            default="https://recall.carnival-devops.com",
            env_fallbacks=("RECALL_BASE_URL",),
            description="Point this at a LAN or Tailscale endpoint to use preprod.",
            inline=True,
        ),
        ProviderField(
            key="limit",
            label="Memories per turn",
            kind=KIND_NUMBER,
            default="5",
            description="How many memories are injected before each turn.",
            inline=True,
        ),
        ProviderField(
            key="rerank",
            label="Cross-encoder rerank",
            kind=KIND_BOOL,
            default="true",
            description="Higher retrieval quality at the cost of ~1 s on a cold cache.",
            group="Retrieval",
        ),
        ProviderField(
            key="graph_boost",
            label="Graph entity boost",
            kind=KIND_BOOL,
            default="false",
            description="Adds a Neo4j round-trip; off by default on the 3 s turn budget.",
            group="Retrieval",
        ),
        ProviderField(
            key="writes_enabled",
            label="Write to Recall",
            kind=KIND_BOOL,
            default="true",
            description="Master switch. Off = read-only: nothing is ever stored.",
            group="Writes",
        ),
        ProviderField(
            key="sync_turns",
            label="Write completed turns",
            kind=KIND_BOOL,
            default="true",
            description="Store each substantive user/assistant turn.",
            group="Writes",
        ),
        ProviderField(
            key="session_summary",
            label="Write a session synthesis",
            kind=KIND_BOOL,
            default="true",
            description="Store one condensed summary when the session ends.",
            group="Writes",
        ),
        ProviderField(
            key="extra_tools",
            label="Extra tools",
            kind=KIND_TEXT,
            default="recall_graph, who_knows, recall_stats",
            description=(
                "Comma-separated extra tools, all three enabled by default — "
                "clear this field to disable them"
            ),
            placeholder="recall_graph, who_knows, recall_stats",
            group="Retrieval",
        ),
        ProviderField(
            key="max_chars",
            label="Max characters per memory",
            kind=KIND_NUMBER,
            default="4000",
            description="Everything stored is truncated to this length.",
            group="Writes",
        ),
        ProviderField(
            key="min_chars",
            label="Min characters to store a turn",
            kind=KIND_NUMBER,
            default="40",
            description="Turns shorter than this are not worth an embedding pass.",
            group="Writes",
        ),
    ),
)
