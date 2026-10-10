"""Smoke-test a specific Lambda VERSION before it receives live traffic.

Invokes the function directly (qualifier = version number) with synthetic
API Gateway events, so the candidate is exercised without touching the
`live` alias. Checks:
  1. GET /health  -> 200, and its reported version equals the commit SHA
                     we just deployed (proves we're testing the right code)
  2. POST /chat   -> 200 with a non-empty reply (proves the whole pipeline:
                     index, Bedrock, Anthropic, guardrails all work)

Usage:
  python3 scripts/smoke_test.py <function-name> <version> <expected-sha>
Exit code 0 = safe to promote, 1 = do NOT promote.
"""
import json
import sys
import uuid


def health_event() -> dict:
    return {"requestContext": {"http": {"method": "GET", "sourceIp": "198.51.100.10"}}, "rawPath": "/health"}


def chat_event(question: str = "Where do you work and what is your role?") -> dict:
    return {
        "requestContext": {"http": {"method": "POST", "sourceIp": "198.51.100.10"}},
        "rawPath": "/chat",
        "body": json.dumps({"message": question, "session_id": str(uuid.uuid4())}),
    }


def check_health(result: dict, expected_sha: str) -> list[str]:
    problems = []
    if result.get("statusCode") != 200:
        return [f"/health returned {result.get('statusCode')}"]
    body = json.loads(result.get("body") or "{}")
    if body.get("version") != expected_sha:
        problems.append(f"/health reports version {body.get('version')!r}, expected {expected_sha!r}")
    return problems


def check_chat(result: dict) -> list[str]:
    if result.get("statusCode") != 200:
        return [f"/chat returned {result.get('statusCode')}: {str(result.get('body'))[:200]}"]
    body = json.loads(result.get("body") or "{}")
    if not str(body.get("reply", "")).strip():
        return ["/chat returned an empty reply"]
    return []


def invoke(client, function: str, version: str, event: dict) -> dict:
    response = client.invoke(FunctionName=function, Qualifier=version, Payload=json.dumps(event).encode())
    if response.get("FunctionError"):
        return {"statusCode": 500, "body": response["Payload"].read().decode()[:300]}
    return json.loads(response["Payload"].read())


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    function, version, expected_sha = sys.argv[1:4]

    import boto3
    from botocore.config import Config
    client = boto3.client("lambda", config=Config(read_timeout=90, retries={"max_attempts": 1}))

    problems = []
    problems += check_health(invoke(client, function, version, health_event()), expected_sha)
    problems += check_chat(invoke(client, function, version, chat_event()))

    if problems:
        print("SMOKE TEST FAILED:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print(f"SMOKE TEST PASSED for {function} version {version} ({expected_sha[:7]})")


if __name__ == "__main__":
    main()
