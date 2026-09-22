# nm-memory-layer

A standalone memory layer for AI agents, built the Hermes-Agent way: every conversation turn is archived locally in SQLite + FTS5, the agent curates what is worth keeping across turns, and procedures that worked are captured as skills. Plain SQLite, no server, no vector DB.

Initially extracted from the [langgraph-demo](../langgraph-demo) agent (branch `memory-layer`); any agent can install and import it.

## What it provides

| Layer | Mechanism | Entrypoint |
|---|---|---|
| **Episodic memory** | SQLite/WAL turn archive + FTS5 search (AND → OR → LIKE fallback chain) | `SessionStore`, `create_session_search_tool` |
| **Always-on prompt memory** | `MEMORY.md` + `USER.md` (3,575-char combined budget), injected once per session | `PromptMemory`, `create_memory_manage_tool` |
| **Agent-curated memory** | Periodic nudge: every N turns an internal LLM call decides between prompt memory / session archive / nothing | `NudgePolicy`, `build_nudge_prompt` |
| **Procedural memory** | agentskills.io-style `SKILL.md` library with progressive disclosure (names+descriptions in context, full file on demand) | `SkillLibrary`, `create_skill_manage_tool`, `create_load_skill_tool` |
| **Search summarization** | Secondary-LLM (OpenRouter) condensation of FTS5 excerpts before they enter context — env-toggled | `create_openrouter_summarizer` |
| **Context compression** | Pre-flight token check; middle turns summarized with lineage stored, first + recent turns verbatim | `create_openrouter_compressor` |
| **Wiki recall** | Local `llm-wiki` (OKF) knowledge base: pre-flight `<wiki_context>` injection + on-demand `wiki_search` | `WikiStore`, `create_wiki_search_tool` |
| **Trace export** | One JSON line per session trajectory (tool calls, conversation projection, compression lineage) for offline mining | `SessionStore.export_to_jsonl` |

## Quick start

```python
from nm_memory_layer import (
    SessionStore, PromptMemory, SkillLibrary, NudgePolicy,
    create_session_search_tool, create_memory_manage_tool, create_skill_manage_tool,
    create_load_skill_tool,
)

store = SessionStore()          # ./sessions.db (override: SESSION_DB_PATH)
memory = PromptMemory()         # ./memories (override: MEMORY_DIR)
skills = SkillLibrary()         # ./skills (override: SKILLS_DIR)
nudge_policy = NudgePolicy(interval=5)

session_search = create_session_search_tool(store, summarizer=None)
memory_manage  = create_memory_manage_tool(memory)
skill_manage   = create_skill_manage_tool(skills)
load_skill     = create_load_skill_tool(skills)

# per turn end:
store.record_turn(session_id, new_messages)
nudge_policy.record_turn(session_id)   # when policy.should_nudge(): run the nudge LLM call

# per session start (load once per session):
memory.load()                  # -> <agent_memory> block for the system prompt
skills.render_index()          # -> <skills_index> names + descriptions only

# on shutdown / anytime:
store.close()
```

Wiki layer reads `WIKI_DIR` (default `./llm-wiki`) when present. Optional features are env-toggled (default off): `SEARCH_SUMMARIZER_ENABLED=true` +
`OPENROUTER_MODEL` enables search-result condensation; `COMPRESSION_ENABLED=true` +
`COMPRESSION_TOKEN_THRESHOLD` enables pre-flight context compression. Both use
`OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` / `OPENROUTER_MODEL` and degrade
gracefully (raw excerpts / untouched history) on any failure.

## Consumer contract

1. Load `memory.load()` and `skills.render_index()` **once per session** — stable
   system-prompt prefix (prompt-cache friendly), edits and new skills apply next session.
2. Call `store.record_turn()` after every turn; run the nudge via `NudgePolicy`.
3. Run compression pre-turn (in a thread from async code) and persist lineage with
   `store.record_compression`.
4. Bind all four tools; nudge loops may call `memory_manage` and `skill_manage`.

Reference consumer: [langgraph-demo's `alt-main.py`](../langgraph-demo/blob/memory-layer/alt-main.py).

## Design decisions

- **Local-first**: one SQLite file (WAL mode) per consumer; no network dependency for storage.
- **Verbatim durability**: message content stored byte-faithfully (incl. trailing whitespace), full `tool_calls` serialization, so resumes reconstruct provider-valid message pairs.
- **Curation over accumulation**: 3,575-char prompt-memory budget, nudge biased toward silence, patch-over-edit for skills.
- **FTS5 over vectors**: exact/keyword recall is cheap, local, and fast; LLM summarization (secondary model) compensates for noise.
- **Compress with lineage**: summarized turns remain fully archived and searchable; `compressions` table records the turn range, summary, and model.
- **Byte-faithful export**: `export_to_jsonl` produces the offline second loop's mining substrate (verified against the live archive).

## Development

```bash
uv sync
uv run pytest tests/ -q   # 107 tests
```

- `HISTORY.md` — changelog of notable changes and the reasoning behind them
- `CLAUDE.md` — full API reference, env variables, consumer contract, and architecture notes for agents working in this repo

License: see `pyproject.toml`.
