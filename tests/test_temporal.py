"""Tests for temporal recall: NL time-phrase parsing + time-bounded search."""

import datetime as dt
import sqlite3
import tempfile

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from nm_memory_layer import SessionStore, parse_time_range


class TestParseBoundaries:
    def test_today_is_bounded(self):
        start, end = parse_time_range("today")
        assert start < end

    def test_last_week_starts_on_monday(self):
        today = dt.date(2026, 9, 24)  # a Thursday
        start, _ = parse_time_range("last week", now=dt.datetime.combine(today, dt.time(12)))
        assert dt.datetime.fromtimestamp(start).weekday() == 0  # Monday
        assert dt.date.fromtimestamp(start) == dt.date(2026, 9, 14)

    def test_past_n_days_rolls_window(self):
        now = dt.datetime(2026, 9, 29, 15, 0)
        start, end = parse_time_range("past 10 days", now=now)
        assert (end - start) >= 10 * 24 * 3600 - 2  # at least a 10-day window
        assert end >= now.timestamp() - 1

    def test_since_bounds_from_date_to_now(self):
        now = dt.datetime(2026, 9, 29, 15, 0)
        start, end = parse_time_range("since 2026-09-01", now=now)
        assert dt.datetime.fromtimestamp(start).date() == dt.date(2026, 9, 1)
        assert end >= now.timestamp() - 1

    def test_between_dates(self):
        start, end = parse_time_range("between 2026-09-01 and 2026-09-20")
        assert dt.datetime.fromtimestamp(start).date() == dt.date(2026, 9, 1)
        assert dt.datetime.fromtimestamp(end).date() == dt.date(2026, 9, 20)

    def test_calendar_month(self):
        start, end = parse_time_range("in September 2026", now=dt.datetime(2026, 9, 29))
        assert dt.datetime.fromtimestamp(start) == dt.datetime(2026, 9, 1)
        assert dt.datetime.fromtimestamp(end).date() == dt.date(2026, 9, 30)

    def test_until_bounds_to_end_of_day(self):
        _, end = parse_time_range("until 2026-09-15")
        assert dt.datetime.fromtimestamp(end).date() == dt.date(2026, 9, 15)

    def test_unparseable_returns_none(self):
        assert parse_time_range("quantum chromodynamics") is None
        assert parse_time_range("") is None
        assert parse_time_range("last 0 days") is None


class TestTimeBoundedSearch:
    @pytest.fixture()
    def store_with_backdated_rows(self):
        p = tempfile.mktemp(suffix=".db")
        store = SessionStore(db_path=p)
        sid = store.new_session_id()
        store.record_turn(sid, [
            HumanMessage(content="quarterly EURUSD backfill report"),
            AIMessage(content="done quickly"),
        ])
        target = dt.datetime(2026, 9, 15, 12, 0).timestamp()
        conn = sqlite3.connect(store.db_path)
        conn.execute("UPDATE sessions SET timestamp = ? WHERE session_id = ?", (target, sid))
        conn.commit()
        conn.close()
        yield store
        store.close()

    def test_in_range_hits_both_channels(self, store_with_backdated_rows):
        bounds = parse_time_range("between 2026-09-14 and 2026-09-16")
        hits = store_with_backdated_rows.search("euro backfill", limit=5, since=bounds[0], until=bounds[1])
        assert len(hits) == 1  # single turn, RRF-deduped to one row
        assert all(h.retrievers is not None for h in hits)

    def test_out_of_range_is_empty(self, store_with_backdated_rows):
        bounds = parse_time_range("between 2026-10-01 and 2026-12-31")
        misses = store_with_backdated_rows.search("euro backfill", limit=5, since=bounds[0], until=bounds[1])
        assert misses == []

    def test_no_bounds_still_applies(self, store_with_backdated_rows):
        hits = store_with_backdated_rows.search("euro backfill", limit=5)
        assert len(hits) == 1  # single turn, RRF-deduped to one row

    def test_semantic_channel_respects_bounds(self, store_with_backdated_rows):
        """Embedding-only rows must not leak outside a time range."""
        # fastembed is required for this store to have embeddings — skip if absent
        pytest.importorskip("fastembed")
        from nm_memory_layer.wiki import _load_hybrid_backend
        assert _load_hybrid_backend() is not None
        bounds = parse_time_range("between 2026-10-01 and 2026-12-31")
        misses = store_with_backdated_rows.search(
            "vector database embedding store", limit=5, since=bounds[0], until=bounds[1]
        )
        assert misses == []


class TestToolTimeRange:
    def test_invalid_phrase_is_nonfatal_rejection(self, monkeypatch):
        import tempfile
        from nm_memory_layer import SessionStore, create_session_search_tool

        p = tempfile.mktemp(suffix=".db")
        store = SessionStore(db_path=p)
        store.record_turn(store.new_session_id(), [HumanMessage(content="anything at all here")])
        tool = create_session_search_tool(store)
        result = tool.invoke({"query": "anything", "time_range": "whenever-ish"})
        assert result.startswith("Rejected:")
        store.close()

    def test_valid_phrase_narrows_results(self):
        import tempfile
        from langchain_core.messages import HumanMessage
        from nm_memory_layer import SessionStore, create_session_search_tool

        p = tempfile.mktemp(suffix=".db")
        store = SessionStore(db_path=p)
        sid = store.new_session_id()
        store.record_turn(sid, [HumanMessage(content="quarterly EURUSD backfill report")])
        target = dt.datetime(2026, 9, 15, 12, 0).timestamp()
        conn = sqlite3.connect(store.db_path)
        conn.execute("UPDATE sessions SET timestamp = ? WHERE session_id = ?", (target, sid))
        conn.commit(); conn.close()

        tool = create_session_search_tool(store)
        in_range = tool.invoke({"query": "euro backfill", "time_range": "between 2026-09-14 and 2026-09-16"})
        assert "quarterly EURUSD" in in_range
        out_range = tool.invoke({"query": "euro backfill", "time_range": "between 2026-10-01 and 2026-12-31"})
        assert out_range.startswith("No past session matches in this time range") or "euro" not in out_range
