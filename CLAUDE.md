# CLAUDE.md — nm-memory-layer

Standalone memory layer for AI agents, extracted from the `langgraph-demo` agent so any agent can import it. Implements a Hermes-Agent-style memory architecture on plain SQLite — no external server, no vector DB.

## Current status

**Phases 1–4 + search summarization complete:**
- **Phase 1: Episodic session store (SQLite + FTS5)** — `store.py`
- **Phase 2: Prompt memory (always-on MEMORY.md / USER.md)** — `prompt_memory.py`
- **Phase 3: Periodic nudge (agent-curated memory)** — `nudge.py`
- **Phase 4: Skills (procedural memory, progressive disclosure)** — `skills.py`
- **Search summarization (secondary-LLM condensation of FTS5 excerpts)** — `summarizer.py`

Phase 5 (context compression) remains.

## Layout

```
nm_memory_layer/
├── __init__.py       # Public API
├── store.py          # SessionStore + session_search tool factory (episodic)
├── prompt_memory.py  # PromptMemory + memory_manage tool factory (always-on)
├── nudge.py          # NudgePolicy + nudge prompt + transcript flattener (curation)
├── skills.py         # SkillLibrary + skill_manage/load_skill tools (procedural)
└── summarizer.py     # Secondary-LLM condensation of session_search excerpts
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

# Curation (Phase 3)
policy = NudgePolicy(interval=5)        # fires every N completed turns
policy.record_turn(session_id)          # call after each completed turn
policy.should_nudge(session_id)         # -> bool; then policy.mark_nudged(session_id)
build_nudge_prompt(chars_used, char_budget)  # system prompt for the internal LLM call
flatten_transcript(messages)            # plain-text transcript (no tool-pairing constraints)

# Procedural (Phase 4)
library = SkillLibrary()                # dir: $SKILLS_DIR or ./skills (agentskills.io layout)
library.render_index()                  # names + descriptions ONLY — load once per session
library.load_skill(name)                # full SKILL.md — the on-demand second step
library.create_skill / patch_skill / edit_skill / delete_skill /
    write_skill_file / remove_skill_file
skill_manage_tool = create_skill_manage_tool(library)
load_skill_tool = create_load_skill_tool(library)

# Search summarization (secondary LLM, env-toggled)
summarizer = create_openrouter_summarizer()   # None unless enabled+configured
tool = create_session_search_tool(store, summarizer=summarizer)
```

### Search summarizer (env-toggled)

- `SEARCH_SUMMARIZER_ENABLED` (`true`/`1`/`yes`/`on`; default off) — flips the feature.
- `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`) — secondary provider config.
- When enabled, `session_search` fetches extra excerpts (≥8 vs the agent's requested limit), condenses them via ONE secondary-LLM call, and returns a summary with a `[session_search: condensed by <label> in <ms>ms from <n> raw excerpts]` header — built-in observability for A/B comparison.
- Failure semantics: missing config, API error, or timeout → silent fallback to raw excerpts; empty summary → explicit "No past session matches relevant to this query." Search never breaks because the summarizer did.

### Consumer responsibilities (Phase 2 contract)

- Load `memory.load()` **once per session** and inject it into the system prompt — stable prefix (prompt-cache friendly) and edits take effect from the next session, per the Hermes rule.
- Bind both tools so the agent can choose the right layer: permanent → `memory_manage`; topic-specific → `session_search`.

## Design decisions

- **WAL mode** — concurrent readers, single writer; safe for parallel sessions.
- **`check_same_thread=False`** — the consumer may first touch the store from a LangGraph tool-executor thread (sync tools like `session_search` run in a worker pool) and later from the event-loop thread (`record_turn`, `close`). The connection is shared across threads deliberately; SQLite's C layer serializes access and WAL + `busy_timeout` handle contention.
- **Full turn serialization** — assistant `tool_calls` and `tool_call_id` are persisted so `load_session()` reconstructs valid AIMessage→ToolMessage pairs; a dangling ToolMessage would break provider APIs on resume.
- **Search fallback chain** — strict FTS5 `AND` match → `OR` of tokens (natural-language queries rarely satisfy implicit AND) → `LIKE` (invalid FTS5 syntax). Newest-first ordering.
- **CWD-relative DB path** — the database belongs to the consuming agent's working directory, not this package. Override with `SESSION_DB_PATH`.
- **FTS5 over vectors** — deliberate tradeoff: exact/keyword recall is cheap, local, and fast. Semantic compensation comes later from the curation layer (nudge), matching the Hermes architecture.
- **3,575-char combined budget** — enforced on every `memory_manage` write across BOTH files; rejections return a non-fatal "Rejected:" message (consumer agent loops decide whether to retry — in langgraph-demo, tool results starting with "Error:" end the turn, so recoverable memory rejections deliberately avoid that prefix).
- **Two-layer boundary is the agent's judgment call** — the `memory_manage` docstring teaches it: MEMORY.md/USER.md only for knowledge needed every session; everything topic-specific stays in the session archive.
- **Nudge bias toward silence** — the nudge prompt states that most turns produce no writes and silence is valid; curation over accumulation. Nudge activity itself is never written to the session archive.
- **Transcript flattening for the nudge** — recent turns are rendered as plain text (`flatten_transcript`) because the nudge LLM binds only the memory tool; sending original AIMessage tool_calls referencing other tools would be provider-invalid.
- **Progressive disclosure keeps skill tokens flat** — the index carries only name + description; full SKILL.md enters context solely via `load_skill`. An agent with 200 skills pays roughly the same index cost as one with 40.
- **patch over edit** — `skill_manage` offers both, but patch (exact-string replacement) is the preferred update path: safer and more token-efficient than full rewrites. The tool docstring teaches this.
- **Filesystem guards** — skill names are slugs (`[a-z0-9][a-z0-9_-]*`); `write_skill_file`/`remove_skill_file` reject absolute paths, `..` traversal, and empty paths; deletes remove the whole skill directory.

## Roadmap (from the Hermes implementation plan)

- ~~Phase 2: Prompt memory (`MEMORY.md` / `USER.md`, 3,575-char shared budget, add/replace/remove ops)~~ ✓ 2026-09-21
- ~~Phase 3: Periodic nudge — agent-curated memory classification (prompt memory vs. session archive vs. nothing)~~ ✓ 2026-09-21
- ~~Phase 4: Skills layer — agentskills.io-style SKILL.md files, progressive disclosure, `skill_manage` with patch preference~~ ✓ 2026-09-21
- Phase 5: Context compression with lineage preserved in SQLite

## Dev

```bash
uv sync
uv run pytest tests/ -q   # 71 tests: store round-trips, FTS fallbacks, budget, tool ops, summarizer
```

Add tests for every new memory-layer capability; the suite is the contract consumers rely on.
