---
name: memory
description: Use Recall as persistent memory — when to search it, what is worth storing, and how to read the memories injected into your context.
---

# Recall as your memory

Recall is your persistent memory across sessions. It is already working in the
background: substantive turns and a session synthesis are written for you. Your
job is the two things automation cannot decide — when to go looking, and what is
worth pinning on purpose.

## Read the injected block first

Before each turn, a block like this may appear in your context:

```
Relevant memories (Recall):
- [decision, 2026-07-21] Marker extraction moved back to GPU with page-by-page chunking…
- [bugfix, 2026-07-10] pgvector delete() with an empty ids list wiped the collection…
```

These are **already stored**. Use them, cite them, build on them — but never
call `recall_store` to save them again.

## When to call `recall_search`

Call it — do not guess, and do not ask the user — when the turn touches:

- something from the past ("what did we decide about…", "how did we fix…")
- a person, a machine, a container, a service, an IP, a path
- a project's conventions, architecture, or deployment procedure
- an error message or symptom that may have been solved before

If the injected block already answers the question, that is enough — no search
needed.

## What to `recall_store`

Store durable knowledge, in your own words, compact:

- a **decision** and the reasoning behind it (`memory_type: decision`)
- a **resolved bug**: symptom, root cause, fix (`memory_type: bugfix`)
- a **lasting preference** the user stated (`memory_type: preference`)
- an **architecture** fact or a **procedure** worth repeating
  (`memory_type: architecture` / `snippet`)

Do **not** store:

- small talk, acknowledgements, or a restatement of the current turn
- anything from the injected memory block
- secrets, API keys, tokens, or passwords
- raw command output or file dumps — write the conclusion instead

## Memory is automatic

Turns and end-of-session summaries are written without you doing anything. So
`recall_store` is for things worth *pinning* — not for bookkeeping. One good
memory beats ten mechanical ones: every write costs an embedding, an entity
extraction and a conflict-detection pass on the server.
