import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .api import usage_cost
from .stream import IncompleteStream, merge_events


class StreamTests(unittest.TestCase):
    def events(self):
        return [
            {"candidates": [{"content": {"parts": [{"text": "An"}]}}]},
            {
                "candidates": [{"content": {"parts": [{"text": "swer"}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 102,
                },
            },
            {"candidates": [{"index": 0}], "usageMetadata": {"trafficType": "ON_DEMAND"}},
        ]

    def test_partial_terminal_events_preserve_usage_and_finish(self):
        response, recovered = merge_events(self.events())
        self.assertEqual(response["candidates"][0]["finishReason"], "STOP")
        self.assertEqual(response["candidates"][0]["content"]["parts"][0]["text"], "Answer")
        self.assertEqual(response["usageMetadata"]["totalTokenCount"], 102)
        self.assertTrue(recovered)
        self.assertAlmostEqual(usage_cost(response), (100 * 0.1 + 2 * 0.4) / 1e6)

    def test_valid_stop_and_output_cap_both_preserved(self):
        for reason in ["STOP", "MAX_TOKENS"]:
            events = self.events()[:2]
            events[-1]["candidates"][0]["finishReason"] = reason
            response, recovered = merge_events(events)
            self.assertFalse(recovered)
            self.assertEqual(response["candidates"][0]["finishReason"], reason)

    def test_nonempty_without_terminal_metadata_is_not_accepted_or_free(self):
        with self.assertRaises(IncompleteStream):
            merge_events([self.events()[0], self.events()[-1]])
        events = self.events()[:2]
        del events[-1]["candidates"][0]["finishReason"]
        with self.assertRaises(IncompleteStream):
            merge_events(events)

    def test_native_usage_updates_and_thoughts(self):
        events = self.events()[:2]
        events[0]["candidates"][0]["content"]["parts"].append({"text": "hidden", "thought": True})
        events.append({"usageMetadata": {"thoughtsTokenCount": 5, "totalTokenCount": 107}})
        response, _ = merge_events(events)
        self.assertEqual(response["candidates"][0]["content"]["parts"][0]["text"], "Answer")
        self.assertAlmostEqual(usage_cost(response), (100 * 0.1 + 7 * 0.4) / 1e6)

    def test_incomplete_response_retry_keeps_raw_events_and_reservation(self):
        from . import generate_answers as reader
        from .retry import SharedBackoff

        class FakeAPI:
            def __init__(self):
                self.calls = []

            def count(self, payload):
                return 100

            def stream(self, payload):
                self.calls.append(payload)
                if len(self.calls) == 1:
                    raise IncompleteStream(
                        "Missing finish metadata", {"usageMetadata": {}}, [{"partial": "saved"}]
                    )
                response = {
                    "candidates": [
                        {"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 100,
                        "candidatesTokenCount": 2,
                        "totalTokenCount": 102,
                    },
                }
                return response, {
                    "reader_latency_seconds": 0.1,
                    "time_to_first_token_seconds": 0.01,
                }

        with tempfile.TemporaryDirectory() as tmp, patch.object(reader, "OUT", Path(tmp)):
            run = reader.ReaderRun("stream_retry_test")
            run.api = FakeAPI()
            now = [0.0]
            run.backoff = SharedBackoff(
                clock=lambda: now[0], sleep=lambda t: now.__setitem__(0, now[0] + t)
            )
            run.run_one(
                "loong",
                {"query": "q", "example_id": "case", "source_tokens": 1},
                "full_context",
                None,
                "text",
                5768,
            )
            self.assertEqual(run.api.calls[0], run.api.calls[1])
            self.assertEqual(
                dict(run.ledger.db.execute("SELECT state,COUNT(*) FROM spend GROUP BY state")),
                {"reserved": 1, "complete": 1},
            )
            attempts = json.loads(next(run.root.glob("receipts/*/*.attempts.json")).read_text())
            self.assertEqual(attempts[0]["events"], [{"partial": "saved"}])
            run.db.close()
            run.ledger.db.close()


if __name__ == "__main__":
    unittest.main()
