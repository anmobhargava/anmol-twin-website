"""Offline tests for the eval gate's scoring logic (no network)."""
import importlib.util
import os
import unittest

spec = importlib.util.spec_from_file_location(
    "run_eval", os.path.join(os.path.dirname(__file__), "..", "..", "eval", "run_eval.py"))
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)

T = {"pass_rate": 0.85, "faithfulness": 0.85, "p95_latency_s": 20}


class CheckCaseTests(unittest.TestCase):
    def test_contains_any_is_case_insensitive(self):
        self.assertEqual(run_eval.check_case({"must_contain_any": ["WPP"]}, "I work at wpp media"), [])

    def test_missing_required_fact_fails(self):
        self.assertTrue(run_eval.check_case({"must_contain_any": ["WPP"]}, "I work somewhere"))

    def test_forbidden_text_fails(self):
        self.assertTrue(run_eval.check_case({"must_not_contain": ["Paris"]}, "It is Paris."))

    def test_forbidden_text_absent_passes(self):
        self.assertEqual(run_eval.check_case({"must_not_contain": ["Paris"]}, "I can't help with that."), [])


class SummaryTests(unittest.TestCase):
    def _r(self, failures=None, faithful=True, latency=3.0):
        return {"failures": failures or [], "faithful": faithful, "latency_s": latency}

    def test_all_good_passes_gate(self):
        s = run_eval.summarize([self._r() for _ in range(10)], T)
        self.assertEqual(s["gate_failures"], [])

    def test_low_pass_rate_fails_gate(self):
        rs = [self._r() for _ in range(7)] + [self._r(["x"]) for _ in range(3)]
        self.assertTrue(any("pass_rate" in g for g in run_eval.summarize(rs, T)["gate_failures"]))

    def test_low_faithfulness_fails_gate(self):
        rs = [self._r() for _ in range(7)] + [self._r(faithful=False) for _ in range(3)]
        self.assertTrue(any("faithfulness" in g for g in run_eval.summarize(rs, T)["gate_failures"]))

    def test_slow_p95_fails_gate(self):
        rs = [self._r(latency=25.0) for _ in range(10)]
        self.assertTrue(any("latency" in g for g in run_eval.summarize(rs, T)["gate_failures"]))

    def test_skipped_faithfulness_not_counted(self):
        rs = [self._r(faithful=None) for _ in range(10)]
        self.assertEqual(run_eval.summarize(rs, T)["faithfulness"], 1.0)


if __name__ == "__main__":
    unittest.main()
