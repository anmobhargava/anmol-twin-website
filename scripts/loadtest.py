"""Small load test for the twin's /chat endpoint (stdlib only).

Usage:
  python3 scripts/loadtest.py https://api.example.com --requests 30 --concurrency 5

Expect MOST requests to come back 200 or 429: the per-IP limit (6/min,
40/hour) and API Gateway's throttle are DESIGNED to reject a single-IP
burst. The point is to verify (a) the limits trip, (b) nothing returns 5xx,
(c) latency of accepted requests stays well under ~29s.

Each request costs real Anthropic money (a few cents at most) -- keep
--requests modest. Exit code 1 if any 5xx or transport error was seen.
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

QUESTIONS = [
    "What is your experience with A/B testing?",
    "Tell me about your work at WPP Media.",
    "What AI/ML projects have you built?",
    "How do you approach marketing mix modeling?",
    "What is your background in attribution?",
]


def one(base_url: str, i: int):
    body = json.dumps({"message": QUESTIONS[i % len(QUESTIONS)], "session_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{base_url}/chat", data=body, headers={"Content-Type": "application/json"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            code = resp.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception:
        code = "error"
    return code, time.monotonic() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--requests", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=5)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda i: one(base, i), range(args.requests)))

    codes = Counter(c for c, _ in results)
    ok = sorted(t for c, t in results if c == 200)
    print("status codes:", dict(codes))
    if ok:
        p95 = ok[min(len(ok) - 1, int(len(ok) * 0.95))]
        print(f"200 latency: median {statistics.median(ok):.1f}s  p95 {p95:.1f}s  max {ok[-1]:.1f}s")
    bad = [c for c in codes if c == "error" or (isinstance(c, int) and c >= 500)]
    print("FAIL: saw 5xx/transport errors" if bad else "PASS: no 5xx or transport errors")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
