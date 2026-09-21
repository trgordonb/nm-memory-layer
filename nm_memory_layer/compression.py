"""Context compression with lineage (Hermes-style Phase 5).

Before the hard context limit is hit, a pre-flight token estimate flags long
conversations. A secondary LLM (the OpenRouter model) then summarizes the
MIDDLE turns of the conversation — never dropping them — while the first
turn (the original task) and the most recent turns stay verbatim. The
summary is injected as a SystemMessage at the compression point, and a
lineage record (summarized turn range + summary text) is persisted to the
session store so earlier context remains traceable and the verbatim
transcript stays searchable via session_search.

Env-toggled (all optional except the toggle):

- ``COMPRESSION_ENABLED``            — "true"/"1"/"yes"/"on" (default off)
- ``COMPRESSION_TOKEN_THRESHOLD``    — pre-flight trigger, default 24000 (~chars/4)
- ``COMPRESSION_KEEP_RECENT_TURNS``  — verbatim recent turns, default 2
- ``OPENROUTER_API_KEY`` / ``OPENROUTER_MODEL`` / ``OPENROUTER_BASE_URL`` — the compressor model

Failure semantics: model errors, empty summaries, or edge cases (too few
turns) abort compression and the original message list is returned
untouched — compression must never break the agent. Note compression is
in-memory per run: after a restart/resume the (still fully archived)
history is re-evaluated and may be re-compressed.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .nudge import flatten_transcript
from .summarizer import _content_to_text, _env_bool

logger = logging.getLogger(__name__)

DEFAULT_COMPRESSION_TOKEN_THRESHOLD = 24_000
DEFAULT_COMPRESSION_KEEP_RECENT = 2
DEFAULT_MAX_SUMMARY_CHARS = 1500

_SUMMARY_SYSTEM_PROMPT = (
    "You compress a slice of an AI agent's conversation history so future turns "
    "fit the context window. Produce a terse, information-dense summary of the "
    "transcript below. Preserve: the user's goals and decisions, procedures and "
    "tool commands used (with exact file paths), key results and numbers, and "
    "any errors with their fixes. Plain text only — never emit tool-call syntax. "
    "Do not add outside knowledge."
)


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Cheap ~4-chars-per-token estimate, no tokenizer dependency.

    Only used as a fallback: the preferred usage measure is the provider-
    reported prompt_tokens of the last LLM response (see
    ``last_request_tokens``), which includes the system prompt and tool
    schemas — matching what the provider actually processed.
    """
    total = 0
    for message in messages:
        content = message.content if isinstance(message.content, str) else _content_to_text(message.content)
        total += len(content) + 8
    return total // 4


def last_request_tokens(messages: list[BaseMessage]) -> int | None:
    """Provider-reported prompt_tokens of the most recent LLM response.

    The history's last AIMessage carries response_metadata.token_usage from
    the request that generated it — the truest measure of how large the
    conversation currently is from the provider's perspective.
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            usage = (getattr(message, "response_metadata", None) or {}).get("token_usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            if prompt_tokens:
                return int(prompt_tokens)
    return None


def split_into_turns(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """Group messages into turns; each turn starts at a HumanMessage.

    Non-human messages before the first HumanMessage (e.g. an injected
    SystemMessage) stay attached to the first turn.
    """
    segments: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage) and current:
            segments.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        segments.append(current)
    if len(segments) >= 2 and not any(isinstance(m, HumanMessage) for m in segments[0]):
        segments[1] = segments[0] + segments[1]
        segments = segments[1:]
    return segments


@dataclass
class CompressionResult:
    compressed: bool = False
    reason: str = ""
    compressed_messages: list[BaseMessage] = field(default_factory=list)
    summary: str = ""
    summarized_first_turn: int = 0
    summarized_last_turn: int = 0
    original_count: int = 0
    compressed_count: int = 0
    elapsed_ms: int = 0
    model_label: str = ""


def _noop(reason: str, messages: list[BaseMessage]) -> CompressionResult:
    return CompressionResult(compressed=False, reason=reason, compressed_messages=list(messages))


class ConversationCompressor:
    """Pre-flight compressor over the live message list (sync LLM call)."""

    def __init__(
        self,
        model,
        label: str | None = None,
        token_threshold: int = DEFAULT_COMPRESSION_TOKEN_THRESHOLD,
        keep_recent_turns: int = DEFAULT_COMPRESSION_KEEP_RECENT,
        max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
    ):
        self.model = model
        self.label = label or getattr(model, "model_name", None) or "llm"
        self.token_threshold = max(1, token_threshold)
        self.keep_recent_turns = max(1, keep_recent_turns)
        self.max_summary_chars = max_summary_chars

    def should_compress(self, messages: list[BaseMessage]) -> bool:
        segments = split_into_turns(messages)
        if len(segments) < self.keep_recent_turns + 2:
            # need: first turn + >=1 middle turn + recent turns
            return False
        return self._current_usage(messages) >= self.token_threshold

    @staticmethod
    def _current_usage(messages: list[BaseMessage]) -> int:
        """Provider-reported prompt_tokens when available, else the estimate."""
        reported = last_request_tokens(messages)
        estimated = estimate_tokens(messages)
        return max(reported or 0, estimated)

    def compress(self, messages: list[BaseMessage]) -> CompressionResult:
        if not messages:
            return _noop("empty", messages)
        segments = split_into_turns(messages)
        if len(segments) < self.keep_recent_turns + 2:
            return _noop("no middle turns", messages)
        if self._current_usage(messages) < self.token_threshold:
            return _noop("below threshold", messages)

        segments = split_into_turns(messages)
        kept_first, middle, kept_recent = segments[0], segments[1:-self.keep_recent_turns], segments[-self.keep_recent_turns:]
        if not middle:
            return _noop("no middle turns", messages)

        transcript = flatten_transcript([m for segment in middle for m in segment])
        messages_payload = [
            ("system", _SUMMARY_SYSTEM_PROMPT),
            ("human", f"TRANSCRIPT SLICE ({len(middle)} turns):\n\n{transcript}\n\nSummarize now."),
        ]
        start = time.monotonic()
        try:
            response = self.model.invoke(messages_payload)
        except Exception as exc:
            logger.warning("context compression failed (%s: %s) — keeping original history", type(exc).__name__, str(exc)[:150])
            return _noop("model error", messages)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        summary = _content_to_text(response.content)
        summary = re.sub(r"<\|[^|>]*\|>", "", summary).strip()
        if not summary or re.search(r"\[\s*(summarize|answer|condense|search)\s*\(", summary, re.IGNORECASE):
            logger.warning("context compression got empty/pseudo-tool output — keeping original history")
            return _noop("empty or malformed model output", messages)
        if len(summary) > self.max_summary_chars:
            summary = summary[: max(self.max_summary_chars - 4, 0)].rstrip() + " […]"

        first_turn = 2  # segment 0 == turn 1 (kept verbatim); middle starts at turn 2
        last_turn = 1 + len(middle)
        summary_msg = SystemMessage(
            content=(
                f"<conversation_summary turns=\"{first_turn}-{last_turn}\">\n{summary}\n"
                f"</conversation_summary>\n"
                f"(Turns {first_turn}-{last_turn} were compressed into this summary. "
                f"The verbatim transcript remains in the session archive — search it with session_search.)"
            )
        )
        compressed = [*kept_first, summary_msg, *(m for segment in kept_recent for m in segment)]
        logger.info(
            "context compression [%s]: %dms, turns %d-%d summarized, %d -> %d messages, summary %d chars",
            self.label, elapsed_ms, first_turn, last_turn, len(messages), len(compressed), len(summary),
        )
        return CompressionResult(
            compressed=True,
            reason="compressed",
            compressed_messages=compressed,
            summary=summary,
            summarized_first_turn=first_turn,
            summarized_last_turn=last_turn,
            original_count=len(messages),
            compressed_count=len(compressed),
            elapsed_ms=elapsed_ms,
            model_label=self.label,
        )


def create_openrouter_compressor() -> ConversationCompressor | None:
    """Build the OpenRouter-backed compressor if enabled and configured, else None."""
    if not _env_bool("COMPRESSION_ENABLED"):
        return None
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("OPENROUTER_MODEL", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key or not model:
        logger.warning(
            "COMPRESSION_ENABLED is on but OPENROUTER_API_KEY/OPENROUTER_MODEL are not set — compressor disabled."
        )
        return None
    try:
        from langchain_openai import ChatOpenAI  # lazy: only needed when enabled

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            max_tokens=2048,
            timeout=60,
            max_retries=1,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to construct compressor model (%s) — disabled.", exc)
        return None
    threshold = int(os.getenv("COMPRESSION_TOKEN_THRESHOLD", str(DEFAULT_COMPRESSION_TOKEN_THRESHOLD)))
    keep = int(os.getenv("COMPRESSION_KEEP_RECENT_TURNS", str(DEFAULT_COMPRESSION_KEEP_RECENT)))
    logger.info("context compressor enabled: openrouter/%s (threshold=%d tokens, keep_recent=%d)", model, threshold, keep)
    return ConversationCompressor(llm, label=f"openrouter:{model}", token_threshold=threshold, keep_recent_turns=keep)
