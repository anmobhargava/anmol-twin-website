"""Offline tests for chassis/llm.py -- no network, no real Anthropic calls.
Run from backend/:  python3 -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic
import httpx

from chassis import llm
from chassis.llm import ResilientClient

SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5-20251001"


def _status_error(code):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(code, request=req)
    return anthropic.APIStatusError("boom", response=resp, body=None)


def _timeout_error():
    return anthropic.APITimeoutError(request=httpx.Request("POST", "https://x"))


class FakeMessages:
    def __init__(self, behaviors):
        self.behaviors = behaviors  # model -> exception | "ok"
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        b = self.behaviors[kwargs["model"]]
        if isinstance(b, Exception):
            raise b
        return {"model": kwargs["model"]}


class FakeClient:
    def __init__(self, behaviors):
        self.messages = FakeMessages(behaviors)


class ResilientClientTests(unittest.TestCase):
    def setUp(self):
        llm._deadline.set(None)

    def test_success_passes_through_with_timeout(self):
        fake = FakeClient({SONNET: "ok"})
        out = ResilientClient(fake, timeout=7).messages.create(model=SONNET, max_tokens=5)
        self.assertEqual(out["model"], SONNET)
        self.assertEqual(fake.messages.calls[0]["timeout"], 7)

    def test_transient_error_falls_back_to_haiku(self):
        fake = FakeClient({SONNET: _status_error(529), HAIKU: "ok"})
        out = ResilientClient(fake).messages.create(model=SONNET, max_tokens=5)
        self.assertEqual(out["model"], HAIKU)
        self.assertEqual([c["model"] for c in fake.messages.calls], [SONNET, HAIKU])

    def test_timeout_falls_back(self):
        fake = FakeClient({SONNET: _timeout_error(), HAIKU: "ok"})
        out = ResilientClient(fake).messages.create(model=SONNET, max_tokens=5)
        self.assertEqual(out["model"], HAIKU)

    def test_non_transient_error_is_not_retried(self):
        fake = FakeClient({SONNET: _status_error(400), HAIKU: "ok"})
        with self.assertRaises(anthropic.APIStatusError):
            ResilientClient(fake).messages.create(model=SONNET, max_tokens=5)
        self.assertEqual(len(fake.messages.calls), 1)

    def test_auth_error_is_not_retried(self):
        fake = FakeClient({SONNET: _status_error(401), HAIKU: "ok"})
        with self.assertRaises(anthropic.APIStatusError):
            ResilientClient(fake).messages.create(model=SONNET, max_tokens=5)
        self.assertEqual(len(fake.messages.calls), 1)

    def test_model_without_fallback_raises(self):
        fake = FakeClient({HAIKU: _status_error(500)})
        with self.assertRaises(anthropic.APIStatusError):
            ResilientClient(fake).messages.create(model=HAIKU, max_tokens=5)
        self.assertEqual(len(fake.messages.calls), 1)

    def test_both_models_fail_raises_last_error(self):
        fake = FakeClient({SONNET: _status_error(500), HAIKU: _status_error(503)})
        with self.assertRaises(anthropic.APIStatusError) as ctx:
            ResilientClient(fake).messages.create(model=SONNET, max_tokens=5)
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(len(fake.messages.calls), 2)

    def test_no_fallback_when_budget_exhausted(self):
        fake = FakeClient({SONNET: _status_error(500), HAIKU: "ok"})
        llm.start_request_budget(1)  # less than MIN_SECONDS_FOR_ATTEMPT
        with self.assertRaises(anthropic.APIStatusError):
            ResilientClient(fake).messages.create(model=SONNET, max_tokens=5)
        self.assertEqual(len(fake.messages.calls), 1)

    def test_timeout_is_capped_by_remaining_budget(self):
        fake = FakeClient({SONNET: "ok"})
        llm.start_request_budget(5)
        ResilientClient(fake, timeout=10).messages.create(model=SONNET, max_tokens=5)
        self.assertLessEqual(fake.messages.calls[0]["timeout"], 5)


if __name__ == "__main__":
    unittest.main()
