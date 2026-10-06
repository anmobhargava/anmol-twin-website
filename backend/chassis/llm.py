"""
Resilient LLM client: per-attempt timeout, bounded retries, model fallback,
and a whole-request time budget.

Why this exists: one chat request chains up to 4 sequential Anthropic calls,
and API Gateway cuts the connection at ~29s whatever the Lambda is doing.
Without limits, one slow or overloaded call turns into a 503 for the visitor
(we already saw this once). This wrapper makes the failure modes explicit:

  1. Every attempt has a timeout (default 10s) -- a hung call can't eat the
     whole budget.
  2. The SDK retries transient errors (429/5xx/connection) at most
     `max_retries` times with its own backoff -- bounded, never infinite.
  3. If the primary model still fails with a *transient* error, the call is
     repeated once on the next model in the fallback chain (Sonnet -> Haiku):
     a slightly less polished answer beats an error message.
  4. A request-wide deadline (set via `start_request_budget`) stops us from
     starting a fallback attempt that cannot finish before the gateway cuts
     us off.

Non-transient errors (bad request, auth, permission) are NOT retried or
fallen back: they would fail identically on another model and the fix is in
our code or config, so they should surface loudly.

Drop-in: exposes `.messages.create(**kwargs)` exactly like the Anthropic
client, so call sites don't change. Each fallback is logged as a structured
JSON line (`llm_fallback`) so it can become a CloudWatch metric filter.
"""

import contextvars
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_SECONDS", 10))
DEFAULT_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", 1))
DEFAULT_REQUEST_BUDGET_S = float(os.environ.get("LLM_REQUEST_BUDGET_SECONDS", 24))
MIN_SECONDS_FOR_ATTEMPT = 3.0

# model -> ordered list of models to try if it fails transiently.
DEFAULT_FALLBACKS = {
    "claude-sonnet-4-6": ["claude-haiku-4-5-20251001"],
}

_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("llm_deadline", default=None)


def start_request_budget(seconds: float = DEFAULT_REQUEST_BUDGET_S) -> None:
    """Call once at the start of handling a request. Worker threads that run
    inside `ctx.copy().run(...)` inherit it."""
    _deadline.set(time.monotonic() + seconds)


def _remaining() -> float | None:
    deadline = _deadline.get()
    return None if deadline is None else deadline - time.monotonic()


def is_transient(exc: Exception) -> bool:
    """True for errors where trying again / another model can plausibly help."""
    try:
        import anthropic
    except Exception:  # pragma: no cover - SDK always present in prod
        return False
    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return False


class _Messages:
    def __init__(self, owner: "ResilientClient"):
        self._owner = owner

    def create(self, **kwargs):
        return self._owner._create(**kwargs)


class ResilientClient:
    def __init__(self, client, fallbacks: dict[str, list[str]] | None = None,
                 timeout: float = DEFAULT_TIMEOUT_S):
        self._client = client
        self._fallbacks = DEFAULT_FALLBACKS if fallbacks is None else fallbacks
        self._timeout = timeout
        self.messages = _Messages(self)

    def _attempt(self, kwargs: dict):
        timeout = self._timeout
        remaining = _remaining()
        if remaining is not None:
            timeout = max(1.0, min(timeout, remaining))
        return self._client.messages.create(timeout=timeout, **kwargs)

    def _create(self, **kwargs):
        primary = kwargs.get("model")
        try:
            return self._attempt(kwargs)
        except Exception as exc:
            if not is_transient(exc):
                raise
            last_exc = exc

        for fallback in self._fallbacks.get(primary, []):
            remaining = _remaining()
            if remaining is not None and remaining < MIN_SECONDS_FOR_ATTEMPT:
                break
            logger.warning(json.dumps({
                "event": "llm_fallback", "from": primary, "to": fallback,
                "error": type(last_exc).__name__,
            }))
            try:
                return self._attempt({**kwargs, "model": fallback})
            except Exception as exc:
                if not is_transient(exc):
                    raise
                last_exc = exc
        raise last_exc


def build_client(api_key: str):
    """The one place the production Anthropic client is constructed."""
    from anthropic import Anthropic
    return ResilientClient(Anthropic(api_key=api_key, max_retries=DEFAULT_MAX_RETRIES))
