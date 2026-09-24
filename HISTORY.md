# HISTORY.md — nm-memory-layer

Changelog of notable changes. Dates are implementation dates.

## 2026-09-21 — Session trajectory exporter (offline second loop substrate)

Investigation into `hermes-agent-self-evolution` (DSPy + GEPA, Phase 1: SKILL.md evolution) showed its `--eval-source sessiondb` mining expects exactly what `sessions.db` already archives: per-turn role/content transcripts with serialized tool calls, skill-usage markers, and compression lineage — while LangSmith traces are neither durable (plan-limited retention) nor turn-shaped.

- `SessionStore.export_session(session_id)`: full trajectory as a dict — flat `messages` (turn/seq/role/content/tool_name/tool_call_id/parsed tool_calls/timestamp), an OpenAI-style `conversation` projection (tool_calls in `function/arguments` wire format), and the session's compression lineage.
- `SessionStore.export_to_jsonl(path, session_ids=None)`: one trajectory per line (newest first) — the JSONL substrate datagen / the self-evolution loop consumes. Returns record count; an empty archive writes an empty file.
- Verified against the live 11-session archive (1.9 MB JSONL; tool calls, skill loads, and lineage all present).
- Role in the stack: LangSmith remains the observability layer; `sessions.db` + exporter is the retention substrate for offline mining.

## 2026-09-24 — Graph retrieval layer (typed-edge walking)

- `graph_hops(wiki_dir, seed_paths, max_hops, max_nodes)` reads the skill's `wiki/graph/graph.sqlite` read-only and expands pages matched by hybrid search into their typed neighbors (`authored`, `works_on`, `depends_on`, `mentions`, `summarizes_raw`, `sourced_from`) over 1-2 hops, returning `retriever: "graph"` rows with via-chains.
- `WikiStore.search` fuses three retrievers when the wiki's graph exists: lexical + embedding via the vector index, then graph hop expansion (path-deduped, excerpts filled from disk). Graph nodes without pages are excluded from `search()` but kept in raw `graph_hops` for provenance-aware consumers.
- Essential for vocabulary-diverse queries: an "Artur Sepp authored works-on vol-carry" lookup reaches things section text never mentions.

## 2026-09-22 — Wiki recall layer (llm-wiki OKF knowledge base)

The agent's local llm-wiki (OKF layout: concepts/, entities/, notes/, raw/ plus index.md files) becomes another memory layer.

- `WikiStore` (`wiki.py`): token-overlap ranking (title hits x3) over all pages, frontmatter-stripped scoring/excerpts, `title:` frontmatter with filename fallback, index.md excluded; missing/empty wiki fully graceful.
- `build_context(query)`: renders the `<wiki_context query=...>` block — the consumer injects it per-turn pre-flight (never archived), so recorded knowledge gets consulted even when the user doesn't mention the wiki.
- `create_wiki_search_tool`: agent-initiated `wiki_search` (excerpts + read_file pointer + llm-wiki-okf update pointer).
- Verified live: 3 pages recalled for a mean-reversion question; agent cited distilled rules and pulled raw/ sources.
- Env: `WIKI_DIR` (default `./llm-wiki`).

## 2026-09-21 — Phase 5: Context compression with lineage

Hermes-style pre-flight compression: when the conversation's estimated token size (~4 chars/token, no tokenizer dependency) crosses `COMPRESSION_TOKEN_THRESHOLD`, the middle turns are summarized by the secondary LLM (OpenRouter model, shared config with the summarizer) while the first turn (original task) and the most recent `COMPRESSION_KEEP_RECENT_TURNS` turns stay verbatim.

- `split_into_turns`: segments the message list on HumanMessage boundaries; non-human preamble (e.g. injected SystemMessages) stays attached to the first turn.
- `ConversationCompressor.compress`: sync LLM call over ONLY the middle turns' flattened transcript; summary is truncated to the char budget and injected as a `<conversation_summary turns="a-b">` SystemMessage that points back to the session archive. Tool-call pairs are removed as whole segments, so pairing can never dangle.
- Lineage: `SessionStore.record_compression` / `list_compressions` persist the summarized turn range, summary text, message count, and model to the new `compressions` table. The archive itself always keeps every turn verbatim and searchable.
- Failure semantics: model errors, empty/pseudo-tool output, or edge cases (too few turns) return the original history untouched. Compression state is in-memory per run; resumed sessions are re-evaluated.
- Env: `COMPRESSION_ENABLED` (default off), `COMPRESSION_TOKEN_THRESHOLD` (24000), `COMPRESSION_KEEP_RECENT_TURNS` (2).

## 2026-09-21 — Search summarizer: weak-model robustness

Live run with a tiny free model (`liquid/lfm-2.5-2.6b:free`) showed two failure modes: message content arriving as a block list (previously `str()`-ed into Python-repr garbage) and the model leaking its chat template's special tokens while hallucinating tool-call syntax (`<|tool_call_start|>[summarize(...)]<|tool_call_end|>`) instead of answering.

- `_content_to_text`: block-list content is properly joined to text.
- Special tokens (`<|...|>`) are stripped from the response.
- Pseudo tool-call payloads (`[summarize(`/`[answer(`/`[condense(`/`[search(`) are detected and treated as failure → session_search falls back to raw excerpts. No hallucinated tool-call garbage can reach the agent's context.
- Model guidance: prefer a mid-size instruct model for the summarizer; the sanitization is a safety net, not a quality fix.

## 2026-09-21 — Search summarizer: OpenRouter reasoning-budget fix

First live run surfaced a real failure mode: `max_completion_tokens` on OpenRouter reasoning models (e.g. liquid/lfm-2.5-2.6b) counts reasoning tokens toward the budget — with large excerpt prompts the model returned EMPTY content with `finish_reason=length`, which the tool then misread as "nothing relevant".

- `create_openrouter_summarizer` now uses `max_tokens` (verified: `finish_reason=stop` under identical prompts).
- Empty summarizer content is no longer treated as a relevance judgment: the tool logs a warning and falls back to raw excerpts. Only the explicit "none relevant" reply produces the "no relevance" result.

## 2026-09-21 — Search summarization (secondary-LLM, env-toggled)

Hermes-style condensation of session_search results before they enter the agent's context, using a secondary provider (OpenRouter) separate from the agent's main model.

- `SearchSummarizer` (`summarizer.py`): wraps any sync LangChain chat model; one call condenses numbered excerpts (with session/turn/role/timestamp references) to what is relevant for the query; latency and sizes logged at INFO.
- `create_openrouter_summarizer()`: env-driven factory — `SEARCH_SUMMARIZER_ENABLED` toggle plus `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` / `OPENROUTER_BASE_URL`. Returns None (feature off) when disabled or misconfigured; never raises.
- `create_session_search_tool(store, summarizer=None)`: optional summarizer param. Enabled mode fetches extra excerpts (≥8), returns the summary with a `[session_search: condensed by <label> in <ms>ms from <n> raw excerpts]` header for A/B observability; empty summary → explicit "no relevance" result.
- Failure semantics: summarizer crash/timeout → silent fallback to raw excerpts. `langchain-openai` added as dependency (lazy import, only when enabled).

## 2026-09-21 — Fix: cross-thread SQLite access

- `SessionStore._connect` now opens the connection with `check_same_thread=False`. In the consumer agent, the first store touch can happen inside a LangGraph tool-executor thread (sync `session_search` runs in a worker pool) while `record_turn`/`close` run on the event-loop thread — the strict same-thread check raised `sqlite3.ProgrammingError` at CLI exit (and silently broke turn recording after any `session_search` call). Regression test added (worker-thread write → main-thread read/close).

## 2026-09-21 — Phase 4: Skills layer (procedural memory)

- `SkillLibrary` (`skills.py`): manages an agentskills.io-compatible skills directory (`$SKILLS_DIR` or `./skills`), scanning `**/SKILL.md` with YAML frontmatter (pyyaml added as dependency). Supports category nesting (`skills/<category>/<name>/`).
- Progressive disclosure: `render_index()` emits a `<skills_index>` block with names + descriptions only (consumer injects once per session); `load_skill(name)` returns the full SKILL.md on demand. Body content never leaks into the index.
- `skill_manage` tool with six actions — create / patch / edit / delete / write_file / remove_file — mirroring the Hermes toolset; docstring teaches the patch-over-edit preference and the creation triggers (5+ tool calls, error recovery, user correction, non-obvious workflow).
- `load_skill` tool: the agent-facing progressive-disclosure second step.
- Filesystem safety: slug-validated names, category slugs, duplicate-create rejection, path-traversal guards, whole-directory delete.
- Nudge integration: `build_nudge_prompt` now also teaches skill curation (create/patch on trigger; skills are procedures, prompt memory is facts — no duplication).

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
