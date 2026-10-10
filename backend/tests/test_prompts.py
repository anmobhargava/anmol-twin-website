"""Offline tests for chassis/prompts.py and scripts/sync_prompts.plan."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from chassis import prompts


class FakeLF:
    def __init__(self, text, version=3, is_fallback=False):
        self.prompt, self.version, self.is_fallback = text, version, is_fallback

    def compile(self, **kw):
        out = self.prompt
        for k, v in kw.items():
            out = out.replace("{{%s}}" % k, str(v))
        return out


class FakeClient:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def get_prompt(self, name, **kw):
        self.calls.append((name, kw))
        if self.error:
            raise self.error
        return self.result


def _env(**kw):
    return mock.patch.dict(os.environ, kw, clear=False)


class PromptTests(unittest.TestCase):
    def setUp(self):
        prompts._down_until = 0.0
        os.environ.pop("PROMPT_SOURCE", None)

    def test_to_langfuse_converts_placeholders(self):
        self.assertEqual(prompts.to_langfuse("a {x} b {y_1}"), "a {{x}} b {{y_1}}")

    def test_default_is_local_and_never_touches_langfuse(self):
        with mock.patch("langfuse.get_client", side_effect=AssertionError("network")):
            p = prompts.get_prompt("grade")
        self.assertIsNone(p.lf)
        out = p.format(question="Q?", passage="P")
        self.assertIn("Q?", out)
        self.assertNotIn("{", out)

    def test_every_prompt_formats_with_its_variables(self):
        args = {"condense": dict(history="h", question="q"), "hyde": dict(question="q"),
                "grade": dict(question="q", passage="p"),
                "answer": dict(history_block="", context="c", question="q"),
                "no_context": dict(history_block="", question="q")}
        self.assertEqual(set(args), set(prompts.PROMPTS))
        for name, kw in args.items():
            self.assertTrue(prompts.get_prompt(name).format(**kw))

    def test_langfuse_version_used_and_linked(self):
        fake = FakeLF("REMOTE {{question}}")
        with _env(PROMPT_SOURCE="langfuse"), mock.patch("langfuse.get_client", return_value=FakeClient(fake)):
            p = prompts.get_prompt("hyde")
        self.assertIs(p.lf, fake)
        self.assertEqual(p.version, 3)
        self.assertEqual(p.format(question="hi"), "REMOTE hi")

    def test_fetch_passes_label_prefix_and_fallback(self):
        client = FakeClient(FakeLF("x {{question}}"))
        with _env(PROMPT_SOURCE="langfuse", PROMPT_LABEL="staging"), mock.patch("langfuse.get_client", return_value=client):
            prompts.get_prompt("no_context")
        name, kw = client.calls[0]
        self.assertEqual(name, "twin-no-context")
        self.assertEqual(kw["label"], "staging")
        self.assertIn("{{question}}", kw["fallback"])

    def test_fetch_error_falls_back_and_trips_breaker(self):
        client = FakeClient(error=RuntimeError("down"))
        with _env(PROMPT_SOURCE="langfuse"), mock.patch("langfuse.get_client", return_value=client):
            p1 = prompts.get_prompt("hyde")
            p2 = prompts.get_prompt("grade")
        self.assertIsNone(p1.lf)
        self.assertIsNone(p2.lf)
        self.assertEqual(len(client.calls), 1)  # breaker skipped the second fetch

    def test_sdk_fallback_object_is_treated_as_local(self):
        client = FakeClient(FakeLF("x", is_fallback=True))
        with _env(PROMPT_SOURCE="langfuse"), mock.patch("langfuse.get_client", return_value=client):
            self.assertIsNone(prompts.get_prompt("hyde").lf)

    def test_unfilled_placeholder_falls_back_to_local(self):
        fake = FakeLF("REMOTE {{question}} {{typo}}")
        with _env(PROMPT_SOURCE="langfuse"), mock.patch("langfuse.get_client", return_value=FakeClient(fake)):
            p = prompts.get_prompt("hyde")
            out = p.format(question="hi")
        self.assertNotIn("REMOTE", out)
        self.assertIsNone(p.lf)  # no version link for text we didn't send

    def test_unknown_prompt_raises(self):
        with self.assertRaises(KeyError):
            prompts.get_prompt("nope")


class SyncPlanTests(unittest.TestCase):
    def test_plan(self):
        import sync_prompts
        existing = {n: None for n in prompts.PROMPTS}
        existing["hyde"] = prompts.to_langfuse(prompts.PROMPTS["hyde"])
        existing["grade"] = "old text"
        plan = sync_prompts.plan(existing)
        self.assertEqual(plan["hyde"], "unchanged")
        self.assertEqual(plan["grade"], "create")
        self.assertEqual(plan["answer"], "create")


if __name__ == "__main__":
    unittest.main()
