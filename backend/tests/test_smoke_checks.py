"""Offline tests for scripts/smoke_test.py's decision logic."""
import importlib.util
import json
import os
import unittest

spec = importlib.util.spec_from_file_location(
    "smoke_test", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "smoke_test.py"))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def resp(code, body):
    return {"statusCode": code, "body": json.dumps(body)}


class HealthTests(unittest.TestCase):
    def test_matching_sha_passes(self):
        self.assertEqual(smoke.check_health(resp(200, {"status": "healthy", "version": "abc"}), "abc"), [])

    def test_wrong_sha_fails(self):
        self.assertTrue(smoke.check_health(resp(200, {"status": "healthy", "version": "old"}), "abc"))

    def test_non_200_fails(self):
        self.assertTrue(smoke.check_health(resp(500, {}), "abc"))


class ChatTests(unittest.TestCase):
    def test_good_reply_passes(self):
        self.assertEqual(smoke.check_chat(resp(200, {"reply": "I work at WPP."})), [])

    def test_empty_reply_fails(self):
        self.assertTrue(smoke.check_chat(resp(200, {"reply": "  "})))

    def test_rate_limited_fails(self):
        self.assertTrue(smoke.check_chat(resp(429, {"error": "slow down"})))

    def test_server_error_fails(self):
        self.assertTrue(smoke.check_chat(resp(500, {"error": "boom"})))


if __name__ == "__main__":
    unittest.main()
