"""Regression: each pipeline LLM call must send its OWN prompt (a scripted edit once
left `prompt` undefined / pointing at the wrong template). Offline, fake client."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))

from chassis.prompts import PROMPTS
from rag.corrective_rag import CorrectiveGrader
from rag.hyde import HyDEGenerator
from pipeline import TwinRAGPipeline


class _Resp:
    class usage:
        input_tokens, output_tokens = 1, 1
    model = "m"
    content = [type("B", (), {"text": "YES"})()]


class _Client:
    def __init__(self):
        self.sent = []
        self.messages = self

    def create(self, **kw):
        self.sent.append(kw["messages"][0]["content"])
        return _Resp()


def _head(key):  # first words of a template, before any placeholder
    return PROMPTS[key].split("{")[0][:40]


class PipelinePromptTests(unittest.TestCase):
    def setUp(self):
        self.c = _Client()
        self.p = TwinRAGPipeline(vector_bucket="b", client=self.c)

    def test_condense(self):
        self.p._condense_question("and that?", [{"role": "user", "content": "hi"}])
        self.assertTrue(self.c.sent[-1].startswith(_head("condense")))

    def test_no_context(self):
        self.p._generate_no_context_response("", "hello")
        self.assertTrue(self.c.sent[-1].startswith(_head("no_context")))

    def test_final_answer(self):
        self.p._generate_final_answer("", "ctx", "q")
        self.assertTrue(self.c.sent[-1].startswith(_head("answer")))

    def test_hyde_and_grade(self):
        HyDEGenerator(client=self.c).generate("q")
        self.assertTrue(self.c.sent[-1].startswith(_head("hyde")))
        chunk = type("C", (), {"text": "passage"})()
        CorrectiveGrader(client=self.c).grade("q", chunk)
        self.assertTrue(self.c.sent[-1].startswith("Question: q"))


if __name__ == "__main__":
    unittest.main()
