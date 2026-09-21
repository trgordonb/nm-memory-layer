"""SQLite + FTS5 session store (Hermes-style episodic memory).

One portable file-backed database replaces the OpenViking recorder:
every turn's messages are persisted after the agent loop completes,
and the agent can deliberately search past sessions via FTS5 instead
of loading whole transcripts into context.

WAL mode allows concurrent readers with a single writer.
"""

import json
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
                content = self._message_text(message).strip()
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
                    if not content:
                        content = "\n".join(
                            f"[tool call: {call['name']}({json.dumps(call['args'], default=str)})]"
                            for call in calls
                        )
                if isinstance(message, ToolMessage):
                    tool_name = message.name
                    tool_call_id = message.tool_call_id
                if not content:
                    continue
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

    def list_sessions(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT session_id, created_at, last_turn FROM session_meta ORDER BY created_at DESC"
        ).fetchall()
        return [
            {"session_id": row[0], "created_at": row[1], "turns": row[2]}
            for row in rows
        ]


def create_session_search_tool(store: SessionStore):
    """Factory returning an agent-callable session_search tool bound to the store."""

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
        hits = store.search(query, limit=max(1, min(limit, 20)))
        if not hits:
            return "No past session matches found."
        blocks = []
        for i, hit in enumerate(hits, 1):
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(hit.timestamp))
            blocks.append(
                f"[{i}] session={hit.session_id[:8]} turn={hit.turn_seq} role={hit.role} at={stamp}\n"
                f"{hit.snippet}"
            )
        return "\n\n".join(blocks)

    return session_search
