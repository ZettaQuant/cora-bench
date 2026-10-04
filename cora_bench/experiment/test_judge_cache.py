"""Exact-request reuse, concurrent aliases, and billed-response crash recovery."""

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from .api import Ledger, digest, save
from .judge_cache import JudgeCache
from .score_answers import interpret, score_one


class JudgeCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.ledger = Ledger(self.out / "budget.sqlite")
        self.cache = JudgeCache(self.out, self.ledger)
        self.body = {
            "model": "gpt-4o-2024-08-06",
            "messages": [{"role": "user", "content": "question"}],
            "temperature": 0,
            "max_tokens": 10,
        }
        self.sha = digest(self.body)
        self.response = {
            "usage": {"prompt_tokens": 100, "completion_tokens": 1},
            "choices": [{"message": {"content": "Yes"}}],
        }
        self.receipt = {
            "key": "prior",
            "request_sha256": self.sha,
            "judge_model": self.body["model"],
            "response": self.response,
            "cost_usd": 0.00026,
            "latency_seconds": 0.1,
            **interpret("longmemeval", "Yes"),
        }

    def tearDown(self):
        self.ledger.db.close()
        self.temp.cleanup()

    def evaluate(self, key="new"):
        return self.cache.evaluate(
            "longmemeval", key, self.body, self.out / "judges/main" / (key + ".json"), interpret
        )

    def reserve(self):
        self.ledger.reserve("judge:" + self.sha, "judges", 0.01)

    @patch("cora_bench.experiment.judge_cache.requests.post")
    def test_reuses_canary_judgment_without_api_or_cost(self, post):
        prior = self.out / "judges/vm_canary/prior.json"
        save(prior, self.receipt)
        self.reserve()
        self.ledger.finish(
            "judge:" + self.sha, 0.00026, {"receipt": str(prior.relative_to(self.out))}
        )
        result = self.evaluate()
        post.assert_not_called()
        self.assertTrue(result["correct"])
        self.assertEqual(result["key"], "new")
        self.assertTrue(result["cache_reused"])
        self.assertEqual(result["incremental_cost_usd"], 0)
        self.assertAlmostEqual(self.ledger.totals()["judges"], 0.00026)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"})
    @patch("cora_bench.experiment.judge_cache.requests.post")
    def test_concurrent_aliases_make_only_one_request(self, post):
        post.return_value = Mock(status_code=200, json=lambda: self.response)
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(self.evaluate, [str(i) for i in range(16)]))
        self.assertEqual(post.call_count, 1)
        self.assertEqual(sum(not r["cache_reused"] for r in results), 1)
        self.assertEqual(len(list((self.out / "judges/main").glob("*.json"))), 16)
        self.assertAlmostEqual(sum(r["incremental_cost_usd"] for r in results), 0.00026)

    @patch("cora_bench.experiment.judge_cache.requests.post")
    def test_recovers_raw_receipt_saved_before_ledger_finish(self, post):
        self.reserve()
        save(self.out / "judges/api_receipts" / (self.sha + ".json"), self.receipt)
        self.assertTrue(self.evaluate()["correct"])
        post.assert_not_called()
        self.assertEqual(
            self.ledger.db.execute("SELECT state FROM spend").fetchone()[0], "complete"
        )

    @patch("cora_bench.experiment.judge_cache.requests.post")
    def test_recovers_legacy_result_saved_before_ledger_finish(self, post):
        self.reserve()
        save(self.out / "judges/vm_canary/prior.json", self.receipt)
        self.assertTrue(self.evaluate()["correct"])
        post.assert_not_called()

    @patch("cora_bench.experiment.judge_cache.requests.post")
    def test_ambiguous_reserved_request_is_not_resubmitted(self, post):
        self.reserve()
        with self.assertRaisesRegex(RuntimeError, "Unresolved judge reservation"):
            self.evaluate()
        post.assert_not_called()
        self.assertAlmostEqual(self.ledger.totals()["judges"], 0.01)

    @patch("cora_bench.experiment.judge_cache.requests.post")
    def test_mismatched_receipt_is_rejected(self, post):
        self.reserve()
        save(
            self.out / "judges/api_receipts" / (self.sha + ".json"),
            {**self.receipt, "request_sha256": "wrong"},
        )
        with self.assertRaisesRegex(RuntimeError, "cache request mismatch"):
            self.evaluate()
        post.assert_not_called()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"})
    @patch("cora_bench.experiment.judge_cache.requests.post")
    def test_malformed_paid_response_is_saved_and_not_rebilled(self, post):
        self.response["choices"][0]["message"]["content"] = "unclear"
        post.return_value = Mock(status_code=200, json=lambda: self.response)
        for _ in range(2):
            with self.assertRaisesRegex(AssertionError, "Malformed"):
                self.evaluate()
        self.assertEqual(post.call_count, 1)
        self.assertTrue((self.out / "judges/api_receipts" / (self.sha + ".json")).exists())
        self.assertAlmostEqual(self.ledger.totals()["judges"], 0.00026)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"})
    @patch("cora_bench.experiment.judge_cache.requests.post")
    def test_unparseable_judge_is_visible_without_blocking_other_results(self, post):
        malformed = {**self.response, "choices": [{"message": {"content": "unclear"}}]}
        post.side_effect = [
            Mock(status_code=200, json=lambda: malformed),
            Mock(status_code=200, json=lambda: self.response),
        ]
        bad_path = self.out / "judges/main/bad.json"
        self.assertIsNone(score_one(self.cache, "longmemeval", "bad", self.body, bad_path))
        self.assertFalse(bad_path.exists())
        unresolved = json.loads((bad_path.parent / "unresolved/bad.json").read_text())
        self.assertEqual(unresolved["status"], "unresolved")
        other_body = {**self.body, "messages": [{"role": "user", "content": "another question"}]}
        good = score_one(
            self.cache, "longmemeval", "good", other_body, bad_path.with_name("good.json")
        )
        self.assertTrue(good["correct"])
        # Re-running the failed logical item reuses its paid response; it cannot bill again.
        self.assertIsNone(score_one(self.cache, "longmemeval", "bad", self.body, bad_path))
        self.assertEqual(post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
