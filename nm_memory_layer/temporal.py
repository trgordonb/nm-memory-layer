"""Temporal recall helpers: natural-language time phrases → epoch bounds.

Provides :func:`parse_time_range` — maps human time phrases ("last week",
"since 2026-09-01", "in August", "past 3 days", "between A and B") into
``(start_epoch, end_epoch)`` bounds for the session archive's ``timestamp``
column. Pure stdlib (datetime); naive local time consistent with
``time.time()`` used by the store.

Unparseable text returns ``None`` — callers decide the fallback (the
``session_search`` tool treats None as a soft rejection, never a hard turn
end). "Now" is injectable for tests.
"""

import datetime as _dt
import time as _time

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"], 1
    )
}

_DAY_START = _dt.time.min  # 00:00
_DAY_END = _dt.time.max  # 23:59:59.999999


def _start_of_day(d: _dt.datetime) -> _dt.datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of_day(d: _dt.datetime) -> _dt.datetime:
    return d.replace(hour=23, minute=59, second=59, microsecond=999999)


def _monday_of(week_start_day: _dt.datetime) -> _dt.datetime:
    d = _start_of_day(week_start_day)
    return d - _dt.timedelta(days=d.weekday())


def _iso_date(text: str) -> _dt.datetime | None:
    """Parse YYYY-MM-DD (or YYYY/MM/DD) → naive midnight datetime."""
    for sep in ("-", "/"):
        parts = text.strip().split(sep)
        if len(parts) != 3 or len(parts[0]) != 4:
            continue
        try:
            return _dt.datetime(int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
    try:
        return _dt.datetime.fromisoformat(text.strip())
    except ValueError:
        return None


def _month_range(text: str) -> tuple[_dt.datetime, _dt.datetime] | None:
    """Parse 'in August', 'August 2026', 'in Sep 2025' → calendar month bounds."""
    words: list[str] = text.lower().split()
    for month_name, month_idx in _MONTHS.items():
        short = month_name[:3]
        if month_name in words or short in words:
            idx = words.index(month_name if month_name in words else short)
            year = None
            for tail in words[idx + 1: idx + 3]:
                digits = tail.strip(",")
                if digits.isdigit() and len(digits) == 4:
                    year = int(digits)
                    break
            year = year if year is not None else _time.localtime().tm_year
            start = _dt.datetime(year, month_idx, 1, 0, 0)
            end_month = month_idx + 1
            end_year = year + 1 if end_month == 13 else year
            end = _dt.datetime(end_year, end_month % 12 or 12, 1, 0, 0) - _dt.timedelta(days=1)
            return start, _end_of_day(end)
    return None


def parse_time_range(text: str, now: _dt.datetime | None = None) -> tuple[float, float] | None:
    """Map a human time phrase to ``(start_epoch, end_epoch)``.

    Supported: today · yesterday · this week · last week · this month ·
    last month · last/past N (day|week|month) · since YYYY-MM-DD ·
    until YYYY-MM-DD · between A and B · in <Month> [YYYY] · bare ISO day.

    Returns None when the phrase is unparseable.
    """
    if not text or not text.strip():
        return None
    now = now or _dt.datetime.now()
    phrase = text.strip().lower()
    phrase = phrase.strip("\"'`")

    if phrase.startswith("since "):
        target = _iso_date(phrase[len("since "):])
        if target:
            return _start_of_day(target).timestamp(), now.timestamp()
    if phrase.startswith("until ") or phrase.startswith("before "):
        target = _iso_date(phrase.split(" ", 1)[1])
        if target:
            return 0.0, _end_of_day(target).timestamp()
    between: list[str] = phrase.split("between ", 1)[-1].split(" and ") \
        if " between " in phrase or phrase.startswith("between ") else []
    if len(between) == 2:
        a, b = _iso_date(between[0].strip()), _iso_date(between[1].strip())
        if a and b:
            return _start_of_day(a).timestamp(), _end_of_day(b).timestamp()

    daily = {"today": now, "yesterday": now - _dt.timedelta(days=1)}
    for key, day in daily.items():
        if phrase == key:
            return _start_of_day(day).timestamp(), _end_of_day(day).timestamp()

    week_bounds = {
        "this week": _monday_of(now),
        "last week": _monday_of(now) - _dt.timedelta(days=7),
    }
    for key, week_start in week_bounds.items():
        if phrase == key:
            week_end = week_start + _dt.timedelta(days=7)
            return week_start.timestamp(), _end_of_day(week_end - _dt.timedelta(days=1)).timestamp()

    month_bounds = {
        "this month": _start_of_day(now.replace(day=1)),
        "last month": _start_of_day(now.replace(day=1)) - _dt.timedelta(days=1),
    }
    for key, anchor in month_bounds.items():
        if phrase == key:
            start = _start_of_day(anchor.replace(day=1))
            end_anchor = anchor - _dt.timedelta(days=1)
            end = _end_of_day(end_anchor.replace(day=1))
            return start.timestamp(), end.timestamp()

    import re
    strip_words = ("past", "previous", "the")
    tokens = phrase
    for w in strip_words:
        tokens = re.sub(rf"\b{w}\b", "", tokens).strip()
    tokens = re.sub(r"\s+", " ", tokens)
    matched = re.match(r"^(\d+)\s+(day|week|month)s?$", tokens)
    if matched:
        count, unit = int(matched.group(1)), matched.group(2)
        if count <= 0:
            return None
        delta = {"day": dict(days=count), "week": dict(weeks=count), "month": dict(days=30 * count)}[unit]
        start = now - _dt.timedelta(**delta)
        return _start_of_day(start).timestamp(), now.timestamp()

    month_pair = _month_range(phrase)
    if month_pair:
        start, end = month_pair
        return start.timestamp(), end.timestamp()

    iso = _iso_date(phrase)
    if iso:
        return _start_of_day(iso).timestamp(), _end_of_day(iso).timestamp()

    return None
