"""Secondary-LLM summarization of FTS5 session-search results (Hermes-style).

Raw FTS5 search returns keyword matches — cheap but noisy. With a summarizer
attached, the session_search tool passes the excerpts through a SECONDARY LLM
(a different provider than the agent's main model) which condenses them to
only what is relevant to the current query before it enters the agent's
context. This mirrors Hermes: "retrieved results go through LLM
summarization before being injected".

Toggle + provider are env-driven so consumers can A/B compare:

- ``SEARCH_SUMMARIZER_ENABLED``  — "true"/"1"/"yes"/"on" to enable (default off)
- ``OPENROUTER_API_KEY``         — secondary provider key
- ``OPENROUTER_BASE_URL``        — default https://openrouter.ai/api/v1
- ``OPENROUTER_MODEL``           — e.g. an affordable fast model id

Failure semantics: if the summarizer is missing/misconfigured, or the LLM
call raises or times out, session_search silently falls back to the raw
excerpt list — search must never break because the summarizer did.
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Callable

from .store import SessionSearchHit

logger = logging.getLogger(__name__)

# Some weak models (esp. tiny "free" ones) leak their chat template's special
# tokens into completions and hallucinate tool-call syntax instead of answering.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_PSEUDO_TOOL_CALL_RE = re.compile(r"\[\s*(summarize|answer|condense|search)\s*\(", re.IGNORECASE)


def _content_to_text(content) -> str:
    """Normalize LangChain message content (str or block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content or "")

_SUMMARIZER_SYSTEM_PROMPT = (
    "You condense episodic session-search excerpts for an AI agent. You are "
    "given the agent's query and numbered excerpts retrieved from past session "
    "archives. Return ONLY the content relevant to the query, reorganized and "
    "tersely rewritten. Keep the [n] excerpt references when a statement comes "
    "from an excerpt. Drop irrelevant excerpts entirely. Never invent content "
    "that is not in the excerpts. If nothing is relevant to the query, reply "
    "exactly: none relevant."
)


@dataclass
class SummarizedSearch:
    """Result of a summarizer pass over FTS5 hits."""
    text: str
    model_label: str
    elapsed_ms: int
    excerpt_count: int


class SearchSummarizer:
    """Wraps any sync LangChain chat model as a session-search condenser."""

    def __init__(self, model, label: str | None = None):
        self.model = model
        self.label = label or getattr(model, "model_name", None) or "llm"

    def summarize(self, query: str, hits: list[SessionSearchHit]) -> SummarizedSearch:
        excerpts = "\n\n".join(
            f"[{i}] session={hit.session_id[:8]} turn={hit.turn_seq} role={hit.role} "
            f"at={time.strftime('%Y-%m-%d %H:%M', time.localtime(hit.timestamp))}\n{hit.snippet}"
            for i, hit in enumerate(hits, 1)
        )
        messages = [
            ("system", _SUMMARIZER_SYSTEM_PROMPT),
            ("human", f"QUERY: {query}\n\nEXCERPTS:\n{excerpts}\n\nCondense now."),
        ]
        start = time.monotonic()
        response = self.model.invoke(messages)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        text = _content_to_text(response.content)
        text = _SPECIAL_TOKEN_RE.sub("", text).strip()
        if _PSEUDO_TOOL_CALL_RE.search(text):
            logger.warning(
                "session_search summarizer [%s] emitted a pseudo tool call instead of "
                "condensed text — treating as failure (tool falls back to raw excerpts)",
                self.label,
            )
            text = ""
        logger.info(
            "session_search summarizer [%s]: %dms, %d excerpts, %d -> %d chars",
            self.label, elapsed_ms, len(hits), len(excerpts), len(text),
        )
        return SummarizedSearch(text=text, model_label=self.label, elapsed_ms=elapsed_ms, excerpt_count=len(hits))


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def create_openrouter_summarizer() -> SearchSummarizer | None:
    """Build the OpenRouter-backed summarizer if enabled and configured, else None.

    Never raises: any misconfiguration logs a warning and disables the
    summarizer so session_search keeps working in raw mode.
    """
    if not _env_bool("SEARCH_SUMMARIZER_ENABLED"):
        return None
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("OPENROUTER_MODEL", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key or not model:
        logger.warning(
            "SEARCH_SUMMARIZER_ENABLED is on but OPENROUTER_API_KEY/OPENROUTER_MODEL "
            "are not set — session_search summarizer disabled (raw excerpts mode)."
        )
        return None
    try:
        from langchain_openai import ChatOpenAI  # lazy: only needed when enabled

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            # max_tokens, NOT max_completion_tokens: OpenRouter reasoning models
            # count reasoning tokens against max_completion_tokens and can return
            # empty content with finish_reason=length when the budget is exhausted.
            max_tokens=1024,
            timeout=20,
            max_retries=1,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to construct summarizer model (%s) — disabled.", exc)
        return None
    logger.info("session_search summarizer enabled: openrouter/%s", model)
    return SearchSummarizer(llm, label=f"openrouter:{model}")
