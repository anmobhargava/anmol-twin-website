"""
Streaming chat endpoint (POST /chat/stream), served through a Lambda Function
URL in response-streaming mode via the AWS Lambda Web Adapter (see
Dockerfile.stream and infra/website/stream.tf).

This is a SECOND entry point next to lambda_handler.py, which stays exactly as
it was: /chat (API Gateway, whole answer at once) keeps working and is the
frontend's fallback if this endpoint is unreachable.

The reply is sent as NDJSON -- one JSON object per line:
    {"type": "delta",   "text": "..."}   a piece of the answer, in order
    {"type": "replace", "text": "..."}   output guardrail tripped AFTER the
                                         text streamed: swap the whole message
                                         for this safe text
    {"type": "done",    "from_cache": bool}
    {"type": "error",   "message": "..."}  failed mid-stream
A request that is rejected up front (bad input, rate limit) is a normal JSON
response with a 4xx status, exactly like /chat. A guardrail-blocked INPUT is a
200 stream holding one "replace" + "done", so the page handles it the same way.

Design choices worth knowing:
  * The whole pipeline run -- including the Langfuse agent_run context -- lives
    in ONE worker thread; the HTTP generator only drains a queue. Starlette
    iterates a sync generator on different threadpool threads, and entering a
    tracing context in one thread and leaving it in another is exactly the
    contextvars failure chassis/tracing.py was hardened against.
  * Output guardrail = replace-after-check (chosen deliberately): the answer
    streams, then check_output runs on the finished text. A bad answer can be
    visible briefly before the "replace" event arrives. Buffering everything
    first would defeat the point of streaming.
  * Rate limiting, input guardrail, session history, session saving and
    logging reuse lambda_handler's helpers, so both paths behave the same.
"""

import contextvars
import json
import os
import queue
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

import lambda_handler as core
from chassis.guardrails import OUTPUT_FALLBACK, check_input, check_output, validate_session_id
from chassis.llm import start_request_budget
from chassis.tracing import agent_run, current_trace_id, flush as langfuse_flush

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

_DONE = object()


def _line(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def _source_ip(request: Request) -> str:
    """The Lambda Web Adapter forwards the original Lambda event's request
    context in the x-amzn-request-context header; its sourceIp is the real
    caller. Falls back to the socket peer for local runs."""
    try:
        ctx = json.loads(request.headers.get("x-amzn-request-context", "{}"))
        ip = ctx.get("http", {}).get("sourceIp") or ctx.get("identity", {}).get("sourceIp")
        if ip:
            return ip
    except Exception:
        pass
    return request.client.host if request.client else ""


def _json_error(status: int, body: dict, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(body, status_code=status, headers=headers)


@app.get("/health")
def health():
    return {"status": "healthy", "version": os.environ.get("GIT_SHA", "unknown"), "mode": "stream"}


def _run_chat(question: str, session_id: str, emit) -> None:
    """The whole request, in ONE thread. emit(event_dict) pushes to the client."""
    request_start = time.time()
    trace_id = None
    try:
        with agent_run(core.AGENT_NAME, session_id=session_id):
            trace_id = current_trace_id()
            history = core._load_session(session_id)

            start_request_budget()
            pipeline = core._get_pipeline()
            result = pipeline.answer_stream(
                question, lambda text: emit({"type": "delta", "text": text}), history=history,
            )

            safe = check_output(result.answer)
            if safe.reason:
                core._log_event("output_guardrail", session_id=session_id, reason=safe.reason,
                                trace_id=trace_id, streamed=True)
                # The visitor may already have read the original; replace it.
                emit({"type": "replace", "text": safe.answer})
            final_answer = safe.answer

            try:
                core._save_session(session_id, history, question, final_answer)
            except Exception as save_err:
                core._log_event("session_save_failed", error=str(save_err))

            core._log_event(
                "chat_request",
                session_id=session_id,
                cache_hit=result.from_cache,
                sources_used=len(result.sources),
                latency_ms=round((time.time() - request_start) * 1000),
                trace_id=trace_id,
                streamed=True,
            )
            emit({"type": "done", "from_cache": result.from_cache})
    except Exception as e:
        core._log_event("chat_request_error", session_id=session_id, error=str(e),
                        trace_id=trace_id, streamed=True)
        emit({"type": "error", "message": "Something went wrong processing your question. Please try again."})
    finally:
        # Lambda freezes the environment once the response ends; flush traces first.
        langfuse_flush()


def _event_stream(question: str, session_id: str):
    q: queue.Queue = queue.Queue()

    def worker():
        try:
            _run_chat(question, session_id, q.put)
        finally:
            q.put(_DONE)

    ctx = contextvars.copy_context()
    threading.Thread(target=ctx.run, args=(worker,), daemon=True).start()

    while True:
        item = q.get()
        if item is _DONE:
            return
        yield _line(item)


@app.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError
    except Exception:
        return _json_error(400, {"error": "Invalid JSON body"})

    question = body.get("message", "")
    question = question.strip() if isinstance(question, str) else ""
    session_id = body.get("session_id", "")
    if not question:
        return _json_error(400, {"error": "Missing 'message' field"})
    if not validate_session_id(session_id):
        return _json_error(400, {"error": "Missing or invalid 'session_id'"})

    # Same order as /chat: rate limit first (a rejected request costs one
    # DynamoDB write and no model work), then the input guardrail.
    if core._rate_limiter:
        limit = core._rate_limiter.check(_source_ip(request))
        if not limit.allowed:
            core._log_event("rate_limited", session_id=session_id, scope=limit.scope,
                            retry_after=limit.retry_after, streamed=True)
            if limit.scope == "daily_global":
                message = "I've hit my daily limit for chats -- please check back tomorrow, or reach me through the links on this page."
            else:
                message = "You're sending messages quickly -- give me a moment and try again."
            return _json_error(429, {"error": message, "retry_after": limit.retry_after},
                               {"Retry-After": str(limit.retry_after)})

    verdict = check_input(question)
    if not verdict.allowed:
        core._log_event("guardrail_blocked", session_id=session_id, reason=verdict.reason, streamed=True)

        def blocked():
            yield _line({"type": "replace", "text": verdict.message})
            yield _line({"type": "done", "from_cache": False, "blocked": True})

        return StreamingResponse(blocked(), media_type="application/x-ndjson")

    return StreamingResponse(
        _event_stream(question, session_id),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
