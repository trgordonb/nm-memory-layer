"""SQLite + FTS5 session store (Hermes-style episodic memory).

One portable file-backed database replaces the OpenViking recorder:
every turn's messages are persisted after the agent loop completes,
and the agent can deliberately search past sessions via FTS5 instead
of loading whole transcripts into context.

WAL mode allows concurrent readers with a single writer.
"""

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool


def _default_db_path() -> str:
    """DB lives in the consumer's working directory unless overridden."""
    return os.getenv("SESSION_DB_PATH", "sessions.db")


@dataclass
class SessionSearchHit:
    session_id: str
    turn_seq: int
    role: str
    snippet: str
    timestamp: float


class SessionStore:
    """SQLite-backed transcript archive with FTS5 full-text search."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or _default_db_path()
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False: the consumer may first touch the store
            # from a LangGraph tool-executor thread (sync tools run in a worker
            # pool) and later from the event-loop/main thread. SQLite's C layer
            # serializes access to a shared connection, and WAL mode plus
            # busy_timeout handle lock contention; ops here are short.
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._conn = conn
            self._migrate(conn)
        return self._conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT NOT NULL,
                turn_seq INTEGER NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                tool_name TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                PRIMARY KEY (session_id, turn_seq, seq)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
                content,
                session_id UNINDEXED,
                turn_seq UNINDEXED,
                seq UNINDEXED,
                role UNINDEXED,
                tokenize='porter unicode61'
            );

            CREATE TABLE IF NOT EXISTS session_meta (
                session_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                last_turn INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS compressions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                summarized_first_turn INTEGER NOT NULL,
                summarized_last_turn INTEGER NOT NULL,
                message_count INTEGER NOT NULL,
                summary TEXT NOT NULL,
                model TEXT,
                created_at REAL NOT NULL
            );
            """
        )
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        parts = []
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)

    @staticmethod
    def _role_of(message: BaseMessage) -> str:
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, AIMessage):
            return "assistant"
        if isinstance(message, ToolMessage):
            return "tool"
        return message.__class__.__name__.lower()

    def record_turn(self, session_id: str, messages: list[BaseMessage], turn_seq: int | None = None) -> int:
        """Append one completed turn's messages; returns the turn number used."""
        if not messages:
            return 0
        conn = self._connect()
        now = time.time()
        with conn:
            if turn_seq is None:
                row = conn.execute(
                    "SELECT last_turn FROM session_meta WHERE session_id = ?", (session_id,)
                ).fetchone()
                turn_seq = (row[0] + 1) if row else 1
            seq = 0
            for message in messages:
                raw_content = self._message_text(message)
                role = self._role_of(message)
                tool_name = None
                tool_call_id = None
                tool_calls_json = None
                if isinstance(message, AIMessage) and message.tool_calls:
                    calls = [
                        {"name": call["name"], "args": call.get("args", {}), "id": call.get("id") or ""}
                        for call in message.tool_calls
                    ]
                    tool_calls_json = json.dumps(calls)
                    if not raw_content.strip():
                        raw_content = "\n".join(
                            f"[tool call: {call['name']}({json.dumps(call['args'], default=str)})]"
                            for call in calls
                        )
                if isinstance(message, ToolMessage):
                    tool_name = message.name
                    tool_call_id = message.tool_call_id
                # Store content verbatim (no strip) so the archive is byte-faithful.
                # Empty-content TOOL messages must still be recorded: parallel
                # tool_call batches reference EVERY call id, and a dropped
                # ToolMessage (e.g. grep with zero matches) would make the
                # resumed history provider-invalid.
                is_tool = role == "tool"
                if not is_tool and not raw_content.strip():
                    continue
                content = raw_content
                conn.execute(
                    "INSERT INTO sessions (session_id, turn_seq, seq, role, tool_name, tool_call_id, tool_calls, content, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (session_id, turn_seq, seq, role, tool_name, tool_call_id, tool_calls_json, content, now),
                )
                conn.execute(
                    "INSERT INTO sessions_fts (content, session_id, turn_seq, seq, role) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (content, session_id, turn_seq, seq, role),
                )
                seq += 1
            if seq == 0:
                return turn_seq
            conn.execute(
                "INSERT INTO session_meta (session_id, created_at, last_turn) VALUES (?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET last_turn = MAX(last_turn, excluded.last_turn)",
                (session_id, now, turn_seq),
            )
        return turn_seq

    def load_session(self, session_id: str) -> list[BaseMessage]:
        """Rebuild a chronological message list for resuming a session.

        Assistant messages with recorded tool calls are reconstructed with
        their original tool_calls so the following ToolMessages remain valid
        message pairs for the provider API.
        """
        conn = self._connect()
        rows = conn.execute(
            "SELECT role, tool_name, tool_call_id, tool_calls, content FROM sessions "
            "WHERE session_id = ? ORDER BY turn_seq, seq",
            (session_id,),
        ).fetchall()
        messages: list[BaseMessage] = []
        for role, tool_name, tool_call_id, tool_calls_json, content in rows:
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                if tool_calls_json:
                    try:
                        calls = json.loads(tool_calls_json)
                    except json.JSONDecodeError:
                        calls = []
                    messages.append(AIMessage(content=content, tool_calls=calls))
                else:
                    messages.append(AIMessage(content=content))
            elif role == "tool":
                messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id or f"{tool_name or 'tool'}",
                        name=tool_name,
                    )
                )
        return messages

    def new_session_id(self) -> str:
        return str(uuid.uuid4())

    def search(self, query: str, limit: int = 5, session_id: str | None = None) -> list[SessionSearchHit]:
        """FTS5 search across all archived sessions, newest first.

        Falls back to LIKE matching when the query is not valid FTS5 syntax
        (e.g. bare quotes or dangling operators).
        """
        conn = self._connect()
        if session_id:
            scope = "AND s.session_id = ?"
            scope_params: list = [session_id]
        else:
            scope = ""
            scope_params = []

        def _rows(match_sql: str, match_param: str) -> list[tuple]:
            return conn.execute(
                "SELECT s.session_id, s.turn_seq, s.role, s.content, s.timestamp "
                "FROM sessions_fts f JOIN sessions s ON "
                "s.session_id = f.session_id AND s.turn_seq = f.turn_seq AND s.seq = f.seq "
                f"WHERE ({match_sql}) {scope} "
                "ORDER BY s.timestamp DESC, s.turn_seq DESC, s.seq "
                "LIMIT ?",
                [match_param, *scope_params, limit],
            ).fetchall()

        try:
            rows = _rows("sessions_fts MATCH ?", query)
            if not rows and len(query.split()) > 1:
                # Natural-language queries rarely satisfy the implicit AND;
                # widen to OR, best matches first.
                rows = _rows("sessions_fts MATCH ?", " OR ".join(query.split()))
        except sqlite3.OperationalError:
            rows = _rows("f.content LIKE ?", f"%{query}%")
        return [
            SessionSearchHit(
                session_id=row[0],
                turn_seq=row[1],
                role=row[2],
                snippet=row[3][:600],
                timestamp=row[4],
            )
            for row in rows
        ]

    def record_compression(
        self,
        session_id: str,
        summary: str,
        summarized_first_turn: int,
        summarized_last_turn: int,
        message_count: int,
        model: str | None = None,
    ) -> int:
        """Persist a compression event (lineage: summarized turn range + summary)."""
        conn = self._connect()
        with conn:
            cur = conn.execute(
                "INSERT INTO compressions (session_id, summarized_first_turn, summarized_last_turn, "
                "message_count, summary, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, summarized_first_turn, summarized_last_turn, message_count, summary, model, time.time()),
            )
        return cur.lastrowid

    def list_compressions(self, session_id: str) -> list[dict]:
        """Lineage chain: every compression event for a session, oldest first."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, summarized_first_turn, summarized_last_turn, message_count, summary, model, created_at "
            "FROM compressions WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [
            {
                "id": row[0],
                "summarized_first_turn": row[1],
                "summarized_last_turn": row[2],
                "message_count": row[3],
                "summary": row[4],
                "model": row[5],
                "created_at": row[6],
            }
            for row in rows
        ]

    def list_sessions(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT session_id, created_at, last_turn FROM session_meta ORDER BY created_at DESC"
        ).fetchall()
        return [
            {"session_id": row[0], "created_at": row[1], "turns": row[2]}
            for row in rows
        ]

    # -- Export (offline self-evolution / datagen substrate) -------------------

    def export_session(self, session_id: str) -> dict | None:
        """Export one full session trajectory (Hermes-SessionDB-compatible shape).

        The offline second loop (e.g. hermes-agent-self-evolution's
        ``--eval-source sessiondb``) mines exactly this: per-turn role/content
        transcripts with serialized tool calls. Includes the compression
        lineage when present. Returns None for unknown sessions.
        """
        conn = self._connect()
        meta = conn.execute(
            "SELECT session_id, created_at, last_turn FROM session_meta WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if meta is None:
            return None
        rows = conn.execute(
            "SELECT turn_seq, seq, role, tool_name, tool_call_id, tool_calls, content, timestamp "
            "FROM sessions WHERE session_id = ? ORDER BY turn_seq, seq",
            (session_id,),
        ).fetchall()
        messages = []
        for turn_seq, seq, role, tool_name, tool_call_id, tool_calls, content, timestamp in rows:
            entry = {"turn": turn_seq, "seq": seq, "role": role, "timestamp": timestamp}
            if tool_name:
                entry["tool_name"] = tool_name
            if tool_call_id:
                entry["tool_call_id"] = tool_call_id
            if tool_calls:
                entry["tool_calls"] = json.loads(tool_calls)
            entry["content"] = content
            messages.append(entry)
        conversation: list[dict] = []
        for entry in messages:
            role = "assistant" if entry["role"] == "assistant" else ("tool" if entry["role"] == "tool" else "user")
            item = {"role": role, "content": entry["content"]}
            if entry.get("tool_calls"):
                item["tool_calls"] = [
                    {"type": "function", "id": c["id"], "function": {"name": c["name"], "arguments": json.dumps(c.get("args", {}))}}
                    for c in entry["tool_calls"]
                ]
            elif role == "tool":
                item["tool_call_id"] = entry.get("tool_call_id") or ""
                item["name"] = entry.get("tool_name") or "tool"
            conversation.append(item)
        return {
            "session_id": session_id,
            "created_at": meta[1],
            "turns": meta[2],
            "message_count": len(messages),
            "messages": messages,
            "conversation": conversation,
            "compressions": self.list_compressions(session_id),
        }

    def export_to_jsonl(self, path: str, session_ids: list[str] | None = None) -> int:
        """Write one JSON line per session to ``path``; returns the record count.

        Pass ``session_ids`` to export specific sessions; omit to export the
        whole archive. The JSONL substrate is what offline datagen / the
        self-evolution loop consumes (one full trajectory per line).
        """
        ids = session_ids if session_ids is not None else [s["session_id"] for s in self.list_sessions()]
        count = 0
        with open(path, "w") as fh:
            for session_id in ids:
                record = self.export_session(session_id)
                if record is not None:
                    fh.write(json.dumps(record, default=str) + "\n")
                    count += 1
        return count


def create_session_search_tool(store: SessionStore, summarizer: "SearchSummarizer | Callable[..., object] | None" = None):
    """Factory returning an agent-callable session_search tool bound to the store.

    If a ``summarizer`` is provided, FTS5 excerpts are first condensed by a
    secondary LLM (Hermes-style) before entering the agent's context; on any
    summarizer failure the tool falls back to the raw excerpt list. Search
    must never break because the summarizer did.
    """

    @tool
    def session_search(query: str, limit: int = 5) -> str:
        """Search past session transcripts (episodic memory) for context relevant
        to the current task. Use when past conversations may contain decisions,
        findings, or procedures that help now. Returns matching excerpts with
        session id, turn number, and role — NOT full transcripts.

        Args:
            query: keywords or phrases to search for (supports FTS5 query syntax).
            limit: maximum number of excerpts to return (default 5).
        """
        if not query.strip():
            return "Error: empty query"
        # With a summarizer, fetch extra material for it to condense.
        search_limit = max(limit, 8) if summarizer is not None else max(1, min(limit, 20))
        hits = store.search(query, limit=search_limit)
        if not hits:
            return "No past session matches found."
        if summarizer is not None:
            try:
                result = summarizer.summarize(query, hits)
            except Exception as exc:
                logging.warning(
                    "session_search summarizer failed (%s: %s) — falling back to raw excerpts",
                    type(exc).__name__, str(exc)[:150],
                )
                result = None
            if result is not None:
                text = (result.text or "").strip()
                if text.lower() == "none relevant":
                    return "No past session matches relevant to this query."
                if text:
                    header = (
                        f"[session_search: condensed by {result.model_label} "
                        f"in {result.elapsed_ms}ms from {result.excerpt_count} raw excerpts]"
                    )
                    return f"{header}\n\n{text}"
                # Empty content is a model/provider quirk (e.g. reasoning budget
                # exhausted), not a relevance judgment — fall back to raw.
                logging.warning(
                    "session_search summarizer [%s] returned empty content — falling back to raw excerpts",
                    result.model_label,
                )
        blocks = []
        for i, hit in enumerate(hits, 1):
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(hit.timestamp))
            blocks.append(
                f"[{i}] session={hit.session_id[:8]} turn={hit.turn_seq} role={hit.role} at={stamp}\n"
                f"{hit.snippet}"
            )
        return "\n\n".join(blocks)

    return session_search
