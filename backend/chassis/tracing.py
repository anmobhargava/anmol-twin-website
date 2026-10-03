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
"""

from contextlib import contextmanager

from langfuse import get_client, observe, propagate_attributes


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
    """
    def decorator(func):
        @observe(name=call_type, as_type="generation")
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            get_client().update_current_generation(metadata={"call_type": call_type})
            return result
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
    """
    def decorator(func):
        return observe(name=step_name, as_type="agent")(func)
    return decorator


def traced_tool_call(tool_name: str):
    """Decorator for a CrewAI tool invocation (e.g. the Cypher graph-query
    tool, the vector-search tool) -- as_type="tool" is its own distinct
    Langfuse observation type, same reasoning as traced_agent_step above:
    keeping "the agent decided to use this tool" and "the tool itself ran
    and returned X" as separate, identifiable spans in the trace.
    """
    def decorator(func):
        return observe(name=tool_name, as_type="tool")(func)
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

    Usage (wraps the whole pipeline.answer() call in the twin website, or
    the whole graph-query flow in agent-25):

        with agent_run("agent-25-graphrag", session_id=request_session_id):
            result = graph_pipeline.answer(question)
    """
    client = get_client()
    with client.start_as_current_observation(name=f"{agent_name}-run") as root_span:
        with propagate_attributes(
            tags=[agent_name],
            metadata={"agent_name": agent_name},
            session_id=session_id,
            user_id=user_id,
        ):
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
    """
    try:
        return get_client().get_current_trace_id()
    except Exception:
        return None



def flush():
    try:
        get_client().flush()
    except Exception:
        pass