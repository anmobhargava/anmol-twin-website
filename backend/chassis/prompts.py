"""
Prompt registry + optional Langfuse prompt management.

The repo is the source of truth: every prompt the live pipeline uses is
defined below, in git, reviewed like code. Langfuse is an optional layer on
top that adds (a) a version history per prompt, (b) the ability to change
which version is live by moving a label, no redeploy, and (c) per-trace
attribution -- every generation in Langfuse links to the exact prompt
version that produced it, so quality/cost/latency can be compared by version.

PROMPT_SOURCE controls behaviour:
  local    (default) use the text below. Nothing touches the network.
  langfuse fetch the version carrying label PROMPT_LABEL (default
           "production") from Langfuse, cached for PROMPT_CACHE_SECONDS.

Langfuse is never allowed to break a chat request (same fail-open rule as
tracing.py): any fetch error, a slow Langfuse, or a version that leaves a
{{placeholder}} unfilled -> use the local text instead. After a failed
fetch, Langfuse is skipped for 60s so an outage costs at most one short
timeout, not one per request.

Templates here use Python {var} placeholders. Langfuse uses {{var}}; the
conversion lives in to_langfuse() so the two stay in sync.
Offline-only prompts (chunk context, RAPTOR summaries) are not managed here:
they run at index-build time, not per request.
"""

import logging
import os
import re
import time

logger = logging.getLogger(__name__)

PROMPT_PREFIX = "twin-"
FETCH_TIMEOUT_S = 2
BREAKER_SECONDS = 60

PROMPTS: dict[str, str] = {
    "condense": """Given this conversation history and a follow-up question, \
rewrite the follow-up as a standalone question that makes sense without the \
history -- resolve any pronouns or references (e.g. "that", "it", "those") to \
what they actually refer to. If the follow-up is already standalone, return it \
unchanged. Return ONLY the rewritten question, nothing else.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:""",
    "hyde": """You are helping retrieve information from a person's professional \
background corpus (resume, work history, skills). Given a question a recruiter \
might ask, write a brief, plausible-sounding hypothetical answer — 2-3 sentences, \
resume/bio style — as if it were a real excerpt from that person's background. \
It's fine if the specific facts you invent are wrong; this is used only to \
improve semantic search, not shown to anyone. Do not add disclaimers or caveats.

Question: {question}

Hypothetical answer excerpt:""",
    "grade": """Question: {question}

Retrieved passage:
{passage}

Does this passage contain information that helps answer the question? \
Answer with exactly one word: YES or NO.""",
    "answer": """You are Anmol Bhargava's AI twin, speaking to a recruiter or \
interviewer visiting his portfolio site. Answer the question using ONLY the \
context provided below. Speak in first person, as Anmol. Keep the tone \
professional but approachable, per Anmol's own communication style. If the \
context doesn't contain enough to answer, say so honestly rather than \
guessing or inventing details.

{history_block}
Context:
{context}

Question: {question}

Answer (as Anmol, first person):""",
    "no_context": """You are Anmol Bhargava's AI twin, speaking to a recruiter or \
interviewer visiting his portfolio site. No specific background content matched \
this message. Handle it appropriately:

- If this is casual conversation (a greeting like "hi"/"hello", small talk, thanks, \
  a farewell), respond warmly and naturally as Anmol would -- e.g. greet them back \
  and invite them to ask about his background, WPP Media / marketing analytics work, \
  or the AI/ML projects he's been building. Do NOT say "I don't have information" \
  to a simple greeting -- that reads as broken, not honest.
- If this is a genuine, specific question about Anmol's background/experience that \
  you have no grounded information for, say so honestly -- don't guess or invent \
  details -- and suggest what topics you CAN help with.
- If this is a general-knowledge or off-topic question that has nothing to do with \
  Anmol (trivia, geography, math, news, coding help), do NOT answer it. Say politely, \
  in one or two sentences, that you're here to talk about Anmol's background and \
  projects, and suggest a few topics you can help with.

{history_block}
Message: {question}

Response (as Anmol, first person):""",
}

_VAR = re.compile(r"\{(\w+)\}")
_UNFILLED = re.compile(r"\{\{\w+\}\}")
_down_until = 0.0


def to_langfuse(template: str) -> str:
    """{var} -> {{var}} (Langfuse's placeholder syntax)."""
    return _VAR.sub(r"{{\1}}", template)


def langfuse_name(name: str) -> str:
    return PROMPT_PREFIX + name.replace("_", "-")


class ResolvedPrompt:
    """What a call site holds: format() to fill it in, `lf` for trace linking."""

    def __init__(self, name: str, lf=None):
        self.name = name
        self.lf = lf  # a Langfuse prompt object, or None when using local text
        self.version = getattr(lf, "version", None) if lf is not None else None

    def format(self, **kwargs) -> str:
        local = PROMPTS[self.name].format(**kwargs)
        if self.lf is None:
            return local
        try:
            out = self.lf.compile(**kwargs)
            if isinstance(out, str) and out.strip() and not _UNFILLED.search(out):
                return out
            logger.warning("prompt %s v%s left placeholders unfilled; using local text", self.name, self.version)
        except Exception as exc:
            logger.warning("prompt %s compile failed (%s); using local text", self.name, type(exc).__name__)
        self.lf = None  # don't link a version we did not actually send
        return local


def get_prompt(name: str) -> ResolvedPrompt:
    """Never raises for a known name: worst case is the local text."""
    global _down_until
    if name not in PROMPTS:
        raise KeyError(f"unknown prompt {name!r}")
    if os.environ.get("PROMPT_SOURCE", "local").lower() != "langfuse":
        return ResolvedPrompt(name)
    if time.monotonic() < _down_until:
        return ResolvedPrompt(name)
    try:
        from langfuse import get_client
        lf = get_client().get_prompt(
            langfuse_name(name),
            label=os.environ.get("PROMPT_LABEL", "production"),
            cache_ttl_seconds=int(os.environ.get("PROMPT_CACHE_SECONDS", 300)),
            fallback=to_langfuse(PROMPTS[name]),
            max_retries=0,
            fetch_timeout_seconds=FETCH_TIMEOUT_S,
        )
        if getattr(lf, "is_fallback", False):
            _down_until = time.monotonic() + BREAKER_SECONDS
            return ResolvedPrompt(name)
        return ResolvedPrompt(name, lf)
    except Exception as exc:
        logger.warning("prompt fetch failed for %s (%s); using local text", name, type(exc).__name__)
        _down_until = time.monotonic() + BREAKER_SECONDS
        return ResolvedPrompt(name)
