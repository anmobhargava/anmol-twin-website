"""
Lambda entrypoint. API Gateway invokes this function for every request to
/chat, /history, and /health. This file's only job is translating between
API Gateway's event/response format and pipeline.py's actual RAG logic --
it contains no retrieval/generation logic itself.

Cold start handling: pipeline.connect() runs ONCE per Lambda container, at
first use -- not on every request. AWS reuses "warm" containers across
requests when traffic is frequent enough, so subsequent invocations on the
same warm container skip connect() entirely and reuse the same in-memory
pipeline object. connect() itself is now lightweight (no embedding calls,
no RAPTOR -- see pipeline.py's own docstring), so even a genuinely cold
start should be fast, unlike an earlier version of this architecture.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
# Required because of how this file gets deployed to Lambda (see
# infra/website/lambda.tf): the deployment image preserves the same
# backend/ + corpus/ sibling layout this repo has locally, so
# lambda_handler.py ends up imported as a nested module
# (backend.lambda_handler) rather than a top-level script. That changes how
# Python resolves the bare `from pipeline import ...` import below --
# without this line, lambda_handler.py's own `from pipeline import
# TwinRAGPipeline` fails with ModuleNotFoundError, since backend/ isn't
# automatically added to sys.path just by importing it as a package. This
# is a packaging/import mechanics fix only -- it changes nothing about
# what the code does.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag"))

from pipeline import TwinRAGPipeline

# boto3 is pre-installed in Lambda's Python runtime -- no new dependency,
# no growth to the deployment package's size.
import boto3

VECTOR_BUCKET = os.environ.get("VECTOR_BUCKET")
CONVERSATION_LOG_BUCKET = os.environ.get("CONVERSATION_LOG_BUCKET")
_s3_client = boto3.client("s3") if CONVERSATION_LOG_BUCKET else None

# Module-level (not inside the handler) so this only runs once per container,
# not once per request.
_pipeline = None


def _log_event(event_type: str, **fields):
    """Structured JSON logging -- CloudWatch Logs Insights can query these
    fields directly (e.g. "average latency_ms where cache_hit=false"),
    unlike loose prose print statements. Still just a print() under the
    hood -- Lambda ships stdout to CloudWatch automatically, no separate
    logging call needed."""
    print(json.dumps({"event": event_type, **fields}))


def _get_pipeline() -> TwinRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TwinRAGPipeline(vector_bucket=VECTOR_BUCKET)
        _pipeline.connect()
        _log_event("pipeline_connected")
    return _pipeline


def _response(status_code: int, body: dict) -> dict:
    # CORS is intentionally NOT set here. It's configured at the API Gateway
    # level instead (infra/website/api_gateway.tf) -- the standard approach
    # for HTTP APIs: API Gateway handles the OPTIONS preflight itself,
    # without even invoking this Lambda, and restricts the allowed origin
    # to the actual deployed domain rather than "*". Setting CORS headers
    # here too would be redundant with, and could conflict with, that config.
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _load_session(session_id: str) -> list:
    """Reads existing conversation history for this session from S3. Returns
    an empty list if no session file exists yet (a new visitor) or if S3
    logging is disabled entirely. Best-effort: a read failure here means
    treating this as a fresh conversation, not erroring the whole request."""
    if not _s3_client:
        return []

    key = f"sessions/{session_id}.json"
    try:
        response = _s3_client.get_object(Bucket=CONVERSATION_LOG_BUCKET, Key=key)
        data = json.loads(response["Body"].read())
        return data.get("transcript", [])
    except _s3_client.exceptions.NoSuchKey:
        return []  # genuinely new session, not an error
    except Exception as e:
        _log_event("session_load_failed", error=str(e))
        return []


def _save_session(session_id: str, history: list, question: str, answer: str):
    """Best-effort write of the full transcript (so far) back to S3 -- same
    object _load_session read from, now updated with this turn appended.
    One object per session (sessions/{session_id}.json), not per-day --
    this makes it BOTH the live state _load_session reads back on the next
    request AND the record Anmol can browse later. A write failure must
    NEVER break the actual chat response -- handled by the caller wrapping
    this in its own try/except, separate from the main request handling."""
    if not _s3_client:
        return

    full_transcript = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    key = f"sessions/{session_id}.json"

    _s3_client.put_object(
        Bucket=CONVERSATION_LOG_BUCKET,
        Key=key,
        Body=json.dumps({"session_id": session_id, "transcript": full_transcript}, indent=2),
        ContentType="application/json",
    )


def handler(event: dict, context) -> dict:
    """API Gateway (HTTP API, payload format 2.0) event structure assumed.
    Adjust event parsing here if using REST API (payload format 1.0) instead
    -- the request body location differs slightly between the two."""

    method = event.get("requestContext", {}).get("http", {}).get("method", "")
    path = event.get("rawPath", "")

    # No OPTIONS handling here -- with API Gateway's native CORS config,
    # preflight OPTIONS requests are intercepted and answered by API Gateway
    # itself, and never reach this Lambda at all.

    if path.endswith("/health"):
        return _response(200, {"status": "healthy"})

    # Called once on page load so a returning visitor (recognized via a
    # session_id persisted in the browser's localStorage) sees their prior
    # conversation instead of a blank chat -- server is the source of
    # truth for history now, not the browser.
    if path.endswith("/history") and method == "GET":
        session_id = (event.get("queryStringParameters") or {}).get("session_id", "")
        if not session_id:
            return _response(400, {"error": "Missing 'session_id' query parameter"})
        history = _load_session(session_id)
        return _response(200, {"history": history})

    if path.endswith("/chat") and method == "POST":
        request_start = time.time()

        try:
            body = json.loads(event.get("body") or "{}")
        except json.JSONDecodeError:
            return _response(400, {"error": "Invalid JSON body"})

        question = body.get("message", "").strip()
        session_id = body.get("session_id", "unknown-session")
        if not question:
            return _response(400, {"error": "Missing 'message' field"})

        try:
            # History comes from S3, not the client -- the client only
            # needs to remember its session_id (in localStorage), not
            # replay the whole conversation on every request.
            history = _load_session(session_id)

            pipeline = _get_pipeline()
            result = pipeline.answer(question, history=history)

            # Save failures are isolated from the actual response -- a
            # recruiter should never see an error just because S3 hiccupped.
            try:
                _save_session(session_id, history, question, result.answer)
            except Exception as save_err:
                _log_event("session_save_failed", error=str(save_err))

            _log_event(
                "chat_request",
                session_id=session_id,
                cache_hit=result.from_cache,
                sources_used=len(result.sources),
                latency_ms=round((time.time() - request_start) * 1000),
            )

            return _response(200, {
                "reply": result.answer,
                "from_cache": result.from_cache,
            })
        except Exception as e:
            # Never leak internal exception details (stack traces, file
            # paths, API key hints) to a public-facing endpoint -- log the
            # real error server-side, return a generic message to the client.
            _log_event("chat_request_error", session_id=session_id, error=str(e))
            return _response(500, {"error": "Something went wrong processing your question. Please try again."})

    return _response(404, {"error": "Not found"})


if __name__ == "__main__":
    # Local smoke test, simulating API Gateway events. Requires
    # ANTHROPIC_API_KEY and VECTOR_BUCKET (pointing at a bucket
    # build_index.py has already populated) set as environment variables.
    fake_event = {
        "requestContext": {"http": {"method": "POST"}},
        "rawPath": "/chat",
        "body": json.dumps({"message": "What's your A/B testing experience?", "session_id": "test-session"}),
    }
    response = handler(fake_event, None)
    print(f"Status: {response['statusCode']}")
    print(f"Body: {response['body']}")

    fake_health_event = {
        "requestContext": {"http": {"method": "GET"}},
        "rawPath": "/health",
    }
    health_response = handler(fake_health_event, None)
    print(f"\nHealth check status: {health_response['statusCode']}")
    print(f"Body: {health_response['body']}")