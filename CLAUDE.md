# CLAUDE.md — nm-memory-layer

Standalone memory layer for AI agents, extracted from the `langgraph-demo` agent so any agent can import it. Implements a Hermes-Agent-style memory architecture on plain SQLite — no external server, no vector DB.

## Current status

**Phase 1 complete: Episodic session store (SQLite + FTS5).** Later phases (prompt memory, periodic nudge, skills, compression) land here first, then get consumed by agents.

## Layout

```
nm_memory_layer/
├── __init__.py    # Public API
└── store.py       # SessionStore + session_search tool factory
```

## Public API

```python
from nm_memory_layer import SessionStore, SessionSearchHit, create_session_search_tool

store = SessionStore()                  # DB path: $SESSION_DB_PATH or ./sessions.db (CWD-relative)
store.record_turn(session_id, messages) # Persist one completed agent turn; returns turn number
store.load_session(session_id)          # Rebuild list[BaseMessage] for resume
store.search(query, limit, session_id)  # FTS5 search -> list[SessionSearchHit]
store.list_sessions()                   # [{session_id, created_at, turns}]

session_search_tool = create_session_search_tool(store)  # LangChain @tool for the agent
```

## Design decisions

- **WAL mode** — concurrent readers, single writer; safe for parallel sessions.
- **Full turn serialization** — assistant `tool_calls` and `tool_call_id` are persisted so `load_session()` reconstructs valid AIMessage→ToolMessage pairs; a dangling ToolMessage would break provider APIs on resume.
- **Search fallback chain** — strict FTS5 `AND` match → `OR` of tokens (natural-language queries rarely satisfy implicit AND) → `LIKE` (invalid FTS5 syntax). Newest-first ordering.
- **CWD-relative DB path** — the database belongs to the consuming agent's working directory, not this package. Override with `SESSION_DB_PATH`.
- **FTS5 over vectors** — deliberate tradeoff: exact/keyword recall is cheap, local, and fast. Semantic compensation comes later from the curation layer (nudge), matching the Hermes architecture.

## Roadmap (from the Hermes implementation plan)

- Phase 2: Prompt memory (`MEMORY.md` / `USER.md`, 3,575-char shared budget, add/replace/remove ops)
- Phase 3: Periodic nudge — agent-curated memory classification (prompt memory vs. session archive vs. nothing)
- Phase 4: Skills layer — agentskills.io-style SKILL.md files, progressive disclosure, `skill_manage` with patch preference
- Phase 5: Context compression with lineage preserved in SQLite

## Dev

```bash
uv sync
uv run python   # package is importable; no test suite yet — add one with Phase 2
```
