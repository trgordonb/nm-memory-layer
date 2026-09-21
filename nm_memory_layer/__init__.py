"""nm-memory-layer: standalone agent memory layer (SQLite + FTS5).

Currently provides the episodic session store (Hermes-style):
every agent turn is persisted locally, and the agent deliberately
searches past sessions instead of loading whole transcripts.
"""

from .store import SessionSearchHit, SessionStore, create_session_search_tool

__all__ = ["SessionStore", "SessionSearchHit", "create_session_search_tool"]
