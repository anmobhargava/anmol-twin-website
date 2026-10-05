"""
Per-call LLM tracing, via Langfuse.

Answers a different question than tracking.py: not "how did the overall
run go" but "what happened inside THIS SPECIFIC LLM call" -- the exact
prompt sent, the exact response, tokens in/out, latency, cost, and
critically, WHICH KIND of call it was (call_type). That last part is what
makes the twin-website-style question -- "is extraction or grading eating
most of my latency/cost?" -- answerable from real data instead of
intuition, since Langfuse's dashboard can break down every one of these
dimensions BY call_type.

This module also answers a SECOND question, added once the portfolio grew
past one agent: "how does agent-25 compare to agent-13 -- tokens, cost,
latency?" That comparison is impossible unless every trace carries WHICH
AGENT produced it, not just which call_type within that agent. agent_run()
(below) is what stamps that dimension onto every trace.

API surface verified directly against the installed langfuse==4.15.6, not
assumed from memory -- the SDK's public interface has moved substantially
across versions (v2 used `langfuse.decorators.observe` + `langfuse_context`;
v4 uses a top-level `observe` decorator plus a client object from
`get_client()`). Two things were confirmed by direct introspection while
building this version specifically:
  1. There is NO `Langfuse.update_current_trace` method in this SDK version
     -- that was an incorrect assumption going in. `update_current_span`
     and `update_current_generation` exist, but both update the CURRENT
     OBSERVATION (span/generation), not trace-wide fields.
  2. The real, first-class way to set TRACE-level attributes (the ones
     that every child observation should inherit, and that Langfuse's
     aggregation/dashboard queries group by) is the top-level
     `propagate_attributes()` context manager -- not a client method.
     Re-verify against the installed version's actual API if this chassis
     is revisited later, rather than trusting these import paths blindly.

FAIL-OPEN DESIGN -- added after a real production incident: every entry
point below (agent_run, traced_llm_call, traced_tool_call) wraps its
Langfuse/OTel instrumentation in try/except and falls back to running the
real, untraced code on failure. This was NOT the original design -- it was
added after a Lambda deploy where a corrupted/stuck `contextvars.Context`
(most likely from Lambda freezing an OTel background export thread
mid-operation between invocations on a warm container) caused
`RuntimeError: cannot enter context: <Context ...> is already entered` to
propagate out of the tracing layer and take down every actual chat
request -- observability took down the product it was meant to observe.
Tracing must never be able to do that again: every call here either
succeeds and traces, or fails and silently falls back to the untraced
call. A broken trace is an acceptable loss; a broken chat response is not.
"""

import contextvars
from contextlib import contextmanager, ExitStack

from langfuse import get_client, observe, propagate_attributes

# Carries the real Anthropic response's model/token-usage from inside a
# @traced_llm_call-decorated function out to the decorator itself, without
# changing what those functions return (every call site currently returns
# just the extracted answer text, e.g. `response.content[0].text.strip()`
# -- not the raw response object). A contextvar, not a plain module global,
# because concurrent grading calls (corrective_rag.py's ThreadPoolExecutor)
# must each see their OWN captured usage, not clobber a shared variable.
_last_usage = contextvars.ContextVar("_last_usage", default=None)


def record_usage(response):
    """Call this immediately after any `client.messages.create(...)` call
    inside a function decorated with @traced_llm_call, e.g.:

        response = self.client.messages.create(...)
        record_usage(response)
        return response.content[0].text.strip()

    Without this, every GENERATION in Langfuse has empty modelId/
    usageDetails/costDetails -- update_current_generation was only ever
    being called with metadata={"call_type": ...}, never model= or
    usage_details=, so there was nothing for Langfuse's cost/token
    dashboards to show, confirmed by inspecting update_current_generation's
    real signature on the installed SDK (it accepts model, usage_details,
    cost_details directly -- Langfuse computes cost itself from
    usage_details + model, so cost_details doesn't need setting by hand).

    Fail-open, same reasoning as every other function in this module: a
    malformed/missing .usage on the response must never break the actual
    answer being returned.
    """
    try:
        _last_usage.set({
            "model": getattr(response, "model", None),
            "usage_details": {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        })
    except Exception:
        pass


def traced_llm_call(call_type: str):
    """Decorator for any function that makes an LLM call. call_type is the
    dimension every future dashboard/query slices by -- e.g. "extraction",
    "hyde", "grading", "raptor_summarize", "final_generation". Usage:

        @traced_llm_call("extraction")
        def extract_entities(text: str, client: Anthropic) -> list[Entity]:
            response = client.messages.create(...)
            ...
            return entities

    Every call this wraps shows up in Langfuse tagged with call_type, so
    "total cost by call_type this week" or "p95 latency for grading calls
    specifically" become real, answerable queries -- not something you'd
    have to compute by hand from raw logs.

    as_type="generation" (not the bare default "span") is what unlocks
    Langfuse's LLM-specific fields on this observation -- token counts,
    cost breakdown, model name -- fields a plain span doesn't have.

    Call this from WITHIN an `agent_run(...)` block so the resulting
    generation also inherits the agent_name tag/metadata that block sets
    up -- without that, the call is still traced, just not attributable
    to any one agent in cross-agent comparisons.

    Fail-open: if the Langfuse/OTel instrumentation itself throws (see this
    module's docstring), falls back to calling func() directly, untraced,
    rather than losing the actual LLM response.
    """
    def decorator(func):
        @observe(name=call_type, as_type="generation")
        def traced(*args, **kwargs):
            _last_usage.set(None)  # reset -- don't leak a stale value from a previous call on this context
            result = func(*args, **kwargs)
            try:
                update_kwargs = {"metadata": {"call_type": call_type}}
                captured = _last_usage.get()
                if captured:
                    update_kwargs["model"] = captured["model"]
                    update_kwargs["usage_details"] = captured["usage_details"]
                get_client().update_current_generation(**update_kwargs)
            except Exception:
                pass
            return result

        def wrapper(*args, **kwargs):
            try:
                return traced(*args, **kwargs)
            except Exception:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def traced_agent_step(step_name: str):
    """Decorator for a CrewAI agent's own reasoning/tool-selection step,
    as distinct from the raw LLM call underneath it -- as_type="agent"
    is a real, distinct observation type in Langfuse's schema (separate
    from "generation"), meant specifically for this: an agent DECIDING
    what to do, not the LLM call that decision might trigger underneath.
    Tracing these separately is what lets you later distinguish "the agent
    made a bad decision" from "the agent decided correctly but the tool it
    called failed" -- collapsing both into one generic trace would lose
    exactly that distinction.

    Fail-open, same reasoning as traced_llm_call above.
    """
    def decorator(func):
        traced = observe(name=step_name, as_type="agent")(func)

        def wrapper(*args, **kwargs):
            try:
                return traced(*args, **kwargs)
            except Exception:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def traced_tool_call(tool_name: str):
    """Decorator for a CrewAI tool invocation (e.g. the Cypher graph-query
    tool, the vector-search tool) -- as_type="tool" is its own distinct
    Langfuse observation type, same reasoning as traced_agent_step above:
    keeping "the agent decided to use this tool" and "the tool itself ran
    and returned X" as separate, identifiable spans in the trace.

    Fail-open, same reasoning as traced_llm_call above.
    """
    def decorator(func):
        traced = observe(name=tool_name, as_type="tool")(func)

        def wrapper(*args, **kwargs):
            try:
                return traced(*args, **kwargs)
            except Exception:
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def agent_run(agent_name: str, session_id: str | None = None, user_id: str | None = None):
    """Wraps ONE END-TO-END agent invocation (one request into agent-25's
    graph pipeline, one full run of agent-13's handbook Q&A) so every trace
    and observation created inside it -- every traced_llm_call,
    traced_agent_step, traced_tool_call -- is stamped with WHICH AGENT
    produced it. This is what turns "compare tokens/cost/latency across
    agents" from an unanswerable question into a real Langfuse dashboard
    filter/group-by.

    Concretely, this stamps two different kinds of thing, on purpose:
      - a TAG (agent_name itself, e.g. "agent-25-graphrag") -- tags are
        what Langfuse's UI lets you filter traces by directly, so "show me
        only agent-25's traces" is one click.
      - METADATA ({"agent_name": agent_name}) -- metadata is what
        Langfuse's aggregation queries (e.g. "total cost grouped by
        agent_name") key off, which is the actual cross-agent COMPARISON
        view, not just a filter.
    Both are set via propagate_attributes, the real trace-level API (see
    this module's docstring for why update_current_trace was wrong) --
    everything created inside this `with` block inherits both
    automatically, without every traced_* decorator needing an agent_name
    parameter threaded through it by hand.

    session_id/user_id are optional and separate from agent_name: they let
    you ALSO group by "this visitor's whole conversation" (ties into the
    twin website's existing session_id) or "this specific end user" --
    independent dimensions from "which agent handled it."

    Fail-open: if ENTERING the Langfuse/OTel context managers throws (see
    this module's docstring for why), yields None instead of propagating --
    the caller's code inside the `with` block still runs normally, just
    untraced for this request.

    This used to be a single `try: ... yield root_span \\n except Exception:
    yield None` wrapping BOTH the setup AND the caller's own code (the
    `yield` is where the caller's `with agent_run(...):` body actually
    runs). That looked right but was broken for exactly the case that
    matters most: an exception raised by the CALLER's own code (e.g. the
    contextvars collision corrective_rag.py's grading used to hit) gets
    thrown back into this generator at the `yield` line -- and a
    @contextmanager generator that catches an exception thrown into it and
    then yields AGAIN (as `except Exception: yield None` does) violates the
    one-yield contract: contextlib raises its own
    `RuntimeError: generator didn't stop after throw()` from `__exit__`,
    chained over the real exception, instead of either suppressing it or
    letting it through as itself. Verified directly: reproducing that shape
    confirms the original exception gets buried in `__context__` of a
    different, confusing RuntimeError -- real application errors (not just
    tracing failures) were coming out of this block looking like a tracing
    bug. Fail-open must only cover the TRACING setup itself, never the
    caller's business logic, which should propagate as whatever it really
    is.

    Fixed by splitting the two concerns with an ExitStack: the try/except
    now wraps ONLY entering the two context managers (the actual
    Langfuse/OTel setup that production incident was about). Once both
    are entered, `yield` happens OUTSIDE any try/except of this function's
    own, so an exception from the caller's code propagates untouched, and
    the ExitStack still closes both context managers correctly on the way
    out (each gets its normal chance to record the error on the span).

    Usage (wraps the whole pipeline.answer() call in the twin website, or
    the whole graph-query flow in agent-25):

        with agent_run("agent-25-graphrag", session_id=request_session_id):
            result = graph_pipeline.answer(question)
    """
    stack = ExitStack()
    try:
        client = get_client()
        root_span = stack.enter_context(client.start_as_current_observation(name=f"{agent_name}-run"))
        stack.enter_context(propagate_attributes(
            tags=[agent_name],
            metadata={"agent_name": agent_name},
            session_id=session_id,
            user_id=user_id,
        ))
    except Exception:
        stack.close()
        yield None
        return

    with stack:
        yield root_span


def current_trace_id() -> str | None:
    """Returns the trace id of whatever agent_run(...)/traced_*(...) block
    is currently active. Call this WHILE STILL INSIDE that block -- it
    returns None once you've exited it, since there's no active span left
    to read a trace id from.

    This is the piece that connects tracing.py to evaluation.py: without
    capturing the trace_id at request time, there's no way to later tell
    RAGAS's log_scores_to_langfuse() (or evaluate_batch_and_track()) which
    trace a given quality score belongs to.

        with agent_run("agent-25-graphrag", session_id=request_session_id):
            trace_id = current_trace_id()   # capture it HERE, inside the block
            result = graph_pipeline.answer(question)
        # trace_id is still usable after exiting -- it's just a plain
        # string now, no longer tied to the (now-closed) span

    Fail-open: returns None rather than raising if the client/context is
    in a broken state (see this module's docstring).
    """
    try:
        return get_client().get_current_trace_id()
    except Exception:
        return None


def flush():
    """Langfuse batches and sends traces asynchronously -- in a
    short-lived process (a script, a Lambda invocation) that exits right
    after its work is done, un-flushed traces can be lost entirely if the
    process ends before the background sender gets to them. Call this at
    the end of any such process, same principle as flushing a file buffer
    before closing it.

    Fail-open: this is called unconditionally from a `finally:` block in
    lambda_handler.py. If flush() itself raised, that exception would
    override and discard whatever response was about to be returned to
    the client (Python's finally-after-return semantics) -- so a tracing
    failure here must never propagate.
    """
    try:
        get_client().flush()
    except Exception:
        pass