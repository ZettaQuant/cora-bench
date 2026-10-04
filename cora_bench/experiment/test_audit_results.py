import unittest

from cora_bench.data.babilong import TASK_LABELS, _official_compare

from .audit_results import reference_functions, summarize


class InterimAuditTests(unittest.TestCase):
    def test_incomplete_seed_set_does_not_bias_question_mean(self):
        rows = [{"example_id": "a"}, {"example_id": "b"}]
        scored = {
            ("a", 1): {"correct": True},
            ("a", 2): {"correct": False},
            ("a", 3): {"correct": False},
            ("b", 1): {"correct": True},
        }
        complete, values = summarize(rows, scored, [1, 2, 3])
        self.assertEqual(set(complete), {"a"})
        self.assertEqual(values, {"a": 1 / 3})

    def test_babilong_reference_and_vendored_metric_agree(self):
        ref = reference_functions()
        examples = [
            ("kitchen", "kitchen", "Where is Mary?", "qa1"),
            ("kitchen", "not kitchen", "Where is Mary?", "qa1"),
            ("kitchen", "kitchen or bedroom", "Where is Mary?", "qa1"),
            ("Mary", "Mary gave the milk to Bill.", "Who gave milk to Bill?", "qa5"),
        ]
        for answer, output, query, task in examples:
            self.assertEqual(
                ref["compare_answers"](answer, output, query, ref["TASK_LABELS"][task]),
                _official_compare(answer, output, query, TASK_LABELS[task]),
            )


if __name__ == "__main__":
    unittest.main()
