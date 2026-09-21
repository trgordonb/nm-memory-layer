# HISTORY.md — nm-memory-layer

Changelog of notable changes. Dates are implementation dates.

## 2026-09-21 — Phase 2: Prompt memory (always-on MEMORY.md / USER.md)

- `PromptMemory` (`prompt_memory.py`): manages the MEMORY.md / USER.md pair in one directory (`$MEMORY_DIR` or `./memories`).
  - `load()` renders the `<agent_memory>` block for the system prompt (`""` when both files are empty); consumers load it once per session so edits apply from the next session and the prompt prefix stays cache-stable.
  - Combined 3,575-char budget enforced on every write across both files; over-budget writes return a non-fatal "Rejected: … consolidate first" message.
  - Ops: `add` (appends a bullet line), `replace` (patch-style, first occurrence), `remove` (deletes and collapses blank lines); missing text and empty adds are rejected without raising.
- `create_memory_manage_tool`: LangChain `@tool` factory exposing `memory_manage` with `Literal`-typed `operation`/`target` (invalid values are blocked at schema level). The docstring teaches the layer boundary: permanent → MEMORY.md/USER.md, topic-specific → session archive.
- Exports now include `MEMORY_CHAR_LIMIT`, `PromptMemory`, `create_memory_manage_tool`.

## 2026-09-21 — v0.1.0: Episodic session store (Phase 1)

Initial release. Code was extracted from the `langgraph-demo` agent repo (branch `memory-layer`, originally `state_store.py`) into this standalone package so the memory layer is agent-agnostic.

- `SessionStore`: SQLite-backed transcript archive in WAL mode.
  - `sessions` table stores every turn's messages with `tool_calls` / `tool_call_id` columns; resume reconstructs valid LangChain message pairs.
  - `sessions_fts` FTS5 virtual table (porter unicode61 tokenizer) kept in sync on insert.
  - `session_meta` tracks per-session created_at and last_turn for automatic turn numbering.
- `SessionStore.search`: FTS5 MATCH with AND → OR → LIKE fallback chain, newest-first.
- `create_session_search_tool`: LangChain `@tool` factory exposing `session_search` for agent-initiated retrieval of past-session excerpts (session id, turn, role, snippet — never full transcripts).
- Build backend: hatchling; dependency: `langchain-core`.

Known tradeoff: keyword search only, no semantic similarity (see CLAUDE.md design decisions).
