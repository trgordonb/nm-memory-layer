# CLAUDE.md — nm-memory-layer

Standalone memory layer for AI agents, extracted from the `langgraph-demo` agent so any agent can import it. Implements a Hermes-Agent-style memory architecture on plain SQLite — no external server, no vector DB.

## Current status

**All five phases complete — full Hermes-style memory stack:**
- **Phase 1: Episodic session store (SQLite + FTS5)** — `store.py`
- **Phase 2: Prompt memory (always-on MEMORY.md / USER.md)** — `prompt_memory.py`
- **Phase 3: Periodic nudge (agent-curated memory)** — `nudge.py`
- **Phase 4: Skills (procedural memory, progressive disclosure)** — `skills.py` — since 2026-10-04, `SkillLibrary` is a facade over `nm-skills-registry` (sibling repo): local dir by default, S3-compatible registry + per-user enable/disable when `SKILLS_REGISTRY` is set (Hermes-style toggles, `skill_manage enable|disable`). Local `skills/` dir stays the materialized working copy either way; behavior pinned by parity tests in both repos.
- **Search summarization (secondary-LLM condensation of FTS5 excerpts)** — `summarizer.py`
- **Phase 5: Context compression with lineage** — `compression.py`

## Layout

```
nm_memory_layer/
├── __init__.py       # Public API
├── store.py          # SessionStore + session_search tool factory (episodic)
├── prompt_memory.py  # PromptMemory + memory_manage tool factory (always-on)
├── nudge.py          # NudgePolicy + nudge prompt + transcript flattener (curation)
├── skills.py         # SkillLibrary + skill_manage/load_skill tools (procedural)
├── summarizer.py     # Secondary-LLM condensation of session_search excerpts
├── wiki.py           # WikiStore + wiki_search tool (llm-wiki OKF knowledge base)
└── compression.py    # Pre-flight context compression with lineage
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

# Context compression (Phase 5, env-toggled; uses the OpenRouter model)
compressor = create_openrouter_compressor()   # None unless enabled+configured
res = compressor.compress(messages)           # sync; run in a thread from async code
if res.compressed:
    messages = res.compressed_messages        # first turn + summary + recent turns
    store.record_compression(session_id, res.summary, res.summarized_first_turn,
                             res.summarized_last_turn, res.original_count, res.model_label)
store.list_compressions(session_id)           # lineage chain

# Export (offline self-evolution / datagen substrate)
out = store.export_session(session_id)        # full trajectory: messages (flat, with
                                              # turn/seq/tool_calls), OpenAI-style
                                              # conversation projection, compression lineage
store.export_to_jsonl(path, session_ids=None) # one trajectory per line (newest first);
                                              # this is what a self-evolution loop mines
```

### Search summarizer (env-toggled)

- `SEARCH_SUMMARIZER_ENABLED` (`true`/`1`/`yes`/`on`; default off) — flips the feature.
- `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`) — secondary provider config.
- When enabled, `session_search` fetches extra excerpts (≥8 vs the agent's requested limit), condenses them via ONE secondary-LLM call, and returns a summary with a `[session_search: condensed by <label> in <ms>ms from <n> raw excerpts]` header — built-in observability for A/B comparison.
- Failure semantics: missing config, API error, or timeout → silent fallback to raw excerpts; empty summary → explicit "No past session matches relevant to this query." Search never breaks because the summarizer did.

### Context compression (env-toggled)

- `COMPRESSION_ENABLED` — flips the feature (default off).
- `COMPRESSION_TOKEN_THRESHOLD` (default 24000, ~4 chars/token) — pre-flight trigger.
- `COMPRESSION_KEEP_RECENT_TURNS` (default 2) — recent turns kept verbatim.
- Model: the OpenRouter config (`OPENROUTER_API_KEY` / `OPENROUTER_MODEL` / `OPENROUTER_BASE_URL`), shared with the summarizer.
- Semantics: middle turns are summarized (first turn + recent turns stay verbatim); the summary is injected as a `<conversation_summary>` SystemMessage that points back to the archive; lineage (turn range + summary + model) is persisted to the `compressions` table via `store.record_compression`.
- **Usage measure**: the trigger uses the provider-reported `prompt_tokens` of the last AIMessage's `response_metadata` (includes system prompt + tool schemas — matching what the provider actually processed), falling back to a chars/4 estimate when absent. Structurally, compression needs ≥ `COMPRESSION_KEEP_RECENT_TURNS + 2` turns (there must be a middle to summarize).
- Failure semantics: model error, empty/pseudo-tool output, or too-few-turns → original history returned untouched.
- Note: compression is in-memory per run. The archive always holds the full verbatim transcript, so after a restart/resume the history is re-evaluated and may be re-compressed (idempotent outcome, one extra LLM call).

#### Graph hop walking

`graph_hops(wiki_dir, seed_paths, max_hops, max_nodes)` reads the skill's
`wiki/graph/graph.sqlite` (nodes / aliases / edges - typed predicates
mentions, sourced_from, authored, summarizes_raw, works_on, depends_on)
read-only and walks 1-2 hops outward from the hybrid hits' page nodes.
Reached pages merge into `WikiStore.search` results with `retriever:
"graph"` identification (via-chain + anchor page recorded); graph-only
nodes (e.g. authors with no page) stay out of `search()` output but remain
in raw `graph_hops` output for provenance tracing. No new deps (plain
sqlite3, read-only). Absent graph -> no hop tags, silent skip.

## Consumer responsibilities (Phase 2 contract)

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
- ~~Phase 5: Context compression with lineage preserved in SQLite~~ ✓ 2026-09-21

## Next-increment roadmap (Hindsight-inspired; owned by the consumer, priorities are binding order)

Comparison study: vectorize-io/hindsight (Postgres+pgvector **server** product; Retain/Recall/Reflect over World facts / Experiences / Observations; Mental Models; entity graph; TEMPR = semantic+keyword+graph+temporal with RRF; disposition traits; LongMemEval SOTA). Our stack deliberately stays local-first SQLite — the plan adopts its ideas, not its infra.

**Priority 2 — Hybrid session vector recall** (`store.py` upgrade) — build first
   Embed turn sections (skill-parity BAAI/bge-small-en-v1.5, 384d, sqlite-vec `vec0`) alongside the FTS5 index; fuse keyword + semantic via the existing RRF (k=60) path from `wiki.py`, with a `semantic` retrievers-values tag on the hit objects so consumers can A/B. Zero new deps — fastembed + sqlite-vec are already required. Scale is trivial: 1,694 messages × 384 floats ≈ 2.6 MB. Must not regress byte-fidelity/regression tests (empty ToolMessages, pairing).

**Priority 3 — Temporal recall on the session archive** (`store.search`) — second
   `timestamp` already exists on every row. Add date-range predicates + a natural-language "last week / since 2026-09-01" parser that maps human time phrases to epoch bounds and filters recall accordingly — direct payoff for finance queries. Extend the created-at UI so consumers can pass `since=` / `until=` to `session_search`.

**Priority 1 — Offline consolidation loop (the reflect/mental-model gap) — largest structural addition, executed last in the user's binding order. An idle-time (cron or next-session-stale-flag) reflect pass runs over the last N turns mined from `sessions.db` (`export_to_jsonl`) and produces:
   - distilled `synthesis/` concept pages + typed `graph.relationships` edges (topics/formats/tools patterned from the nudge's memory-classifier), and
   - a returnable operation log entry `wiki/log.md` `## [...] reflect` for provenance.
   Emulates Hindsight's "Mental Models" — but rather than shipping a parallel store, ours writes directly into the wiki skill's pages and the typed graph, reachable via `wiki_search` and `wiki_graph_query.py` without consumer changes.

Deferred-by-design (do not build): serving surface (HTTP/MCP/UI), Postgres/pgvector, disposition traits, Hindsight's coding-agent installer — all contradict the local-first, in-process single-consumer philosophy.

## Dev

```bash
uv sync
uv run pytest tests/ -q   # 71 tests: store round-trips, FTS fallbacks, budget, tool ops, summarizer
```

Add tests for every new memory-layer capability; the suite is the contract consumers rely on.
