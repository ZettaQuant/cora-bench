import unittest

from .summarize_results import interval


class StatisticsTests(unittest.TestCase):
    def test_replicates_not_independent(self):
        rows = [{"example_id": str(i), "needle_id": i // 3} for i in range(12)]
        x = interval("nolima", rows, [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        self.assertEqual(x["clusters"], 4)
        self.assertEqual(x["estimate"], 0.5)

    def test_identical_paired_delta_zero(self):
        rows = [{"example_id": str(i), "task": "qa" + str(i % 5 + 1)} for i in range(20)]
        self.assertEqual(interval("babilong", rows, [0] * 20)["ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
