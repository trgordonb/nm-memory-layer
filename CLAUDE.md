# CLAUDE.md — nm-memory-layer

Standalone memory layer for AI agents, extracted from the `langgraph-demo` agent so any agent can import it. Implements a Hermes-Agent-style memory architecture on plain SQLite — no external server, no vector DB.

## Current status

**Phases 1–2 complete:**
- **Phase 1: Episodic session store (SQLite + FTS5)** — `store.py`
- **Phase 2: Prompt memory (always-on MEMORY.md / USER.md)** — `prompt_memory.py`

Later phases (periodic nudge, skills, compression) land here first, then get consumed by agents.

## Layout

```
nm_memory_layer/
├── __init__.py       # Public API
├── store.py          # SessionStore + session_search tool factory (episodic)
└── prompt_memory.py  # PromptMemory + memory_manage tool factory (always-on)
```

## Public API

```python
from nm_memory_layer import (
    MEMORY_CHAR_LIMIT,       # 3575 — combined MEMORY.md + USER.md budget
    PromptMemory,            # memory_dir: $MEMORY_DIR or ./memories
    SessionStore,            # db path: $SESSION_DB_PATH or ./sessions.db (CWD-relative)
    SessionSearchHit,
    create_memory_manage_tool,   # LangChain @tool: memory_manage (add/replace/remove)
    create_session_search_tool,  # LangChain @tool: session_search
)

# Episodic
store = SessionStore()
store.record_turn(session_id, messages) # Persist one completed agent turn; returns turn number
store.load_session(session_id)          # Rebuild list[BaseMessage] for resume
store.search(query, limit, session_id)  # FTS5 search -> list[SessionSearchHit]

# Always-on
memory = PromptMemory()
memory.load()                           # Rendered <agent_memory> block ("" when empty) — load ONCE per session
memory.total_chars()                    # Combined budget usage
memory.add / replace / remove           # target: "memory" | "user"
```

### Consumer responsibilities (Phase 2 contract)

- Load `memory.load()` **once per session** and inject it into the system prompt — stable prefix (prompt-cache friendly) and edits take effect from the next session, per the Hermes rule.
- Bind both tools so the agent can choose the right layer: permanent → `memory_manage`; topic-specific → `session_search`.

## Design decisions

- **WAL mode** — concurrent readers, single writer; safe for parallel sessions.
- **Full turn serialization** — assistant `tool_calls` and `tool_call_id` are persisted so `load_session()` reconstructs valid AIMessage→ToolMessage pairs; a dangling ToolMessage would break provider APIs on resume.
- **Search fallback chain** — strict FTS5 `AND` match → `OR` of tokens (natural-language queries rarely satisfy implicit AND) → `LIKE` (invalid FTS5 syntax). Newest-first ordering.
- **CWD-relative DB path** — the database belongs to the consuming agent's working directory, not this package. Override with `SESSION_DB_PATH`.
- **FTS5 over vectors** — deliberate tradeoff: exact/keyword recall is cheap, local, and fast. Semantic compensation comes later from the curation layer (nudge), matching the Hermes architecture.
- **3,575-char combined budget** — enforced on every `memory_manage` write across BOTH files; rejections return a non-fatal "Rejected:" message (consumer agent loops decide whether to retry — in langgraph-demo, tool results starting with "Error:" end the turn, so recoverable memory rejections deliberately avoid that prefix).
- **Two-layer boundary is the agent's judgment call** — the `memory_manage` docstring teaches it: MEMORY.md/USER.md only for knowledge needed every session; everything topic-specific stays in the session archive.

## Roadmap (from the Hermes implementation plan)

- ~~Phase 2: Prompt memory (`MEMORY.md` / `USER.md`, 3,575-char shared budget, add/replace/remove ops)~~ ✓ 2026-09-21
- Phase 3: Periodic nudge — agent-curated memory classification (prompt memory vs. session archive vs. nothing)
- Phase 4: Skills layer — agentskills.io-style SKILL.md files, progressive disclosure, `skill_manage` with patch preference
- Phase 5: Context compression with lineage preserved in SQLite

## Dev

```bash
uv sync
uv run pytest tests/ -q   # 25 tests: store round-trips, FTS fallbacks, budget, tool ops
```

Add tests for every new memory-layer capability; the suite is the contract consumers rely on.
