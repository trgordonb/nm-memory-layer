# HISTORY.md — nm-memory-layer

Changelog of notable changes. Dates are implementation dates.

## 2026-09-21 — Phase 3: Periodic nudge (agent-curated memory)

- `NudgePolicy` (`nudge.py`): per-session turn counter; `should_nudge` fires every N completed turns (default 5), `mark_nudged` resets. Interval floors at 1.
- `build_nudge_prompt`: system prompt for the internal curation LLM call — teaches the layer boundary (memory_manage for every-session knowledge, nothing for episodic detail already in the archive), shows current budget usage, and biases hard toward silence ("most turns produce NO writes").
- `flatten_transcript`: renders recent messages (including tool calls/results) as plain text so the nudge call is provider-safe with a different toolset bound.
- Consumer contract: call `record_turn` after every completed turn; when `should_nudge`, run the internal LLM call with only the memory tool bound (short loop, max ~3 iterations), then `mark_nudged`. Nudge writes go to prompt memory files only — never to the session archive, and take effect next session per the Phase 2 rule.

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
