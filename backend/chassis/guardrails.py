"""
Deterministic guardrails for a public-facing chat endpoint.

Deliberately NO LLM calls here: a guardrail that adds a model round-trip
adds seconds of latency and another failure point to every request (this
project already learned what hidden per-request latency costs -- see
chassis/tracing.py's capture_input note). Everything below is regex/logic,
runs in well under a millisecond, and is unit-testable offline.

Three layers:
  1. validate_session_id -- session_id becomes part of an S3 object key
     (sessions/{session_id}.json). Unvalidated, a caller could point reads
     and writes at arbitrary keys in the bucket or send a megabyte-long id.
  2. check_input  -- length cap, prompt-injection patterns, and sensitive
     data (card numbers, SSNs) a visitor should not be typing into a
     chatbot whose conversations are logged to S3.
  3. check_output -- last line of defence: never return anything that looks
     like a credential or leaked prompt, whatever the model produced.

Blocked requests never reach the LLM, so they also cost nothing.

These are heuristics, not a complete defence. Regexes catch the common
injection phrasings, not every paraphrase; the real protection against a
successful injection is that the answer prompt only ever contains the
visitor's own question plus public corpus text, and the process holds no
privileges worth stealing. Treat this as one layer, not the whole wall.
"""

import re
from dataclasses import dataclass

MAX_INPUT_CHARS = 500
MAX_OUTPUT_CHARS = 4000

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None   # machine-readable, for logs/metrics
    message: str | None = None  # friendly, first-person text shown to the visitor


# --- 1. session id ---------------------------------------------------------

def validate_session_id(session_id) -> bool:
    return isinstance(session_id, str) and bool(_SESSION_ID_RE.match(session_id))


# --- 2. input --------------------------------------------------------------

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier|preceding)\s+(instructions?|prompts?|rules?|messages?|context)",
    r"disregard\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier|preceding)?\s*(instructions?|prompts?|rules?|guidelines?)",
    r"forget\s+(all\s+|everything\s+|your\s+)?(previous|prior|above|earlier)?\s*(instructions?|rules?|prompts?)",
    r"(reveal|show|print|repeat|output|display|leak|tell\s+me)\s+(me\s+)?(your|the)\s+(system|hidden|initial|original|secret)\s+(prompt|instructions?|message)",
    r"what\s+(is|are|were)\s+your\s+(system|initial|original)\s+(prompt|instructions?)",
    r"\bjailbreak(ed|ing)?\b",
    r"\bdan\s+mode\b",
    r"\bdeveloper\s+mode\b",
    r"you\s+are\s+(now\s+)?(dan|jailbroken|unrestricted|no\s+longer\s+bound)",
    r"<\s*/?\s*(system|assistant)\s*>",
    r"(?im)^\s*(system|assistant)\s*:",
    r"\b(api[_\s-]?key|secret[_\s-]?key|access[_\s-]?token)s?\b.*\b(show|print|give|reveal|send|what)",
    r"\b(show|print|give|reveal|send|what).*\b(api[_\s-]?key|secret[_\s-]?key|access[_\s-]?token)s?\b",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        n = int(ch)
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0


def _contains_card_number(text: str) -> bool:
    for match in _CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return True
    return False


def check_input(question: str) -> GuardrailResult:
    if len(question) > MAX_INPUT_CHARS:
        return GuardrailResult(
            False, "too_long",
            f"That message is a bit long for me to handle well. Could you shorten it to under {MAX_INPUT_CHARS} characters?",
        )

    if _CONTROL_CHARS_RE.search(question):
        return GuardrailResult(
            False, "control_characters",
            "I couldn't read part of that message. Could you retype it in plain text?",
        )

    if _SSN_RE.search(question) or _contains_card_number(question):
        return GuardrailResult(
            False, "sensitive_data",
            "Please don't share personal or financial details like card or ID numbers here -- "
            "this chat is logged. Ask me about my background and projects instead!",
        )

    for pattern in _INJECTION_RE:
        if pattern.search(question):
            return GuardrailResult(
                False, "prompt_injection",
                "I can't help with that one, but I'm happy to talk about my background, "
                "my work at WPP Media, or the AI/ML projects I've been building.",
            )

    return GuardrailResult(True)


# --- 3. output -------------------------------------------------------------

_SECRET_PATTERNS = [
    r"sk-ant-[A-Za-z0-9_\-]{10,}",
    r"sk-lf-[A-Za-z0-9\-]{10,}",
    r"pk-lf-[A-Za-z0-9\-]{10,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"ANTHROPIC_API_KEY|LANGFUSE_SECRET_KEY|LANGFUSE_PUBLIC_KEY",
]
_PROMPT_LEAK_MARKERS = [
    "You are Anmol Bhargava's AI twin",
    "Answer (as Anmol, first person)",
    "Response (as Anmol, first person)",
    "Standalone question:",
]
_SECRET_RE = [re.compile(p) for p in _SECRET_PATTERNS]

OUTPUT_FALLBACK = (
    "Sorry, I couldn't put together a good answer to that one. "
    "Could you ask it a different way, or ask about my background and projects?"
)


@dataclass
class OutputResult:
    answer: str
    reason: str | None = None   # None means the answer was returned unchanged


def check_output(answer: str) -> OutputResult:
    for pattern in _SECRET_RE:
        if pattern.search(answer):
            return OutputResult(OUTPUT_FALLBACK, "secret_in_output")
    for marker in _PROMPT_LEAK_MARKERS:
        if marker in answer:
            return OutputResult(OUTPUT_FALLBACK, "prompt_leak")
    if len(answer) > MAX_OUTPUT_CHARS:
        return OutputResult(answer[:MAX_OUTPUT_CHARS].rstrip() + "...", "truncated")
    return OutputResult(answer)