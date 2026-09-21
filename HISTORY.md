# HISTORY.md — nm-memory-layer

Changelog of notable changes. Dates are implementation dates.

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
