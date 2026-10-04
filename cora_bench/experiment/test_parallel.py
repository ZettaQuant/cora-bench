import unittest

from .select_context import controls, shard_rows


class ParallelOwnershipTest(unittest.TestCase):
    def test_disjoint_exhaustive_and_order_preserving(self):
        rows = [{"example_id": f"question-{i}", "source_tokens": 250000} for i in range(1945)]
        groups = [shard_rows(rows, i, 3) for i in range(3)]
        ids = [[r["example_id"] for r in group] for group in groups]
        self.assertEqual(sum(map(len, ids)), len(rows))
        self.assertEqual(set().union(*map(set, ids)), {r["example_id"] for r in rows})
        for i in range(3):
            for j in range(i):
                self.assertFalse(set(ids[i]) & set(ids[j]))
            self.assertEqual(groups[i], [r for r in rows if r["example_id"] in set(ids[i])])

    def test_full_cohort_controls_preserved_across_shards(self):
        rows = [
            {"example_id": f"q-{i}", "source_tokens": 250000, "task": str(i % 10)}
            for i in range(500)
        ]
        original = controls(rows, "babilong")
        owned = set()
        for i in range(3):
            owned.update(original & {r["example_id"] for r in shard_rows(rows, i, 3)})
        self.assertEqual(original, owned)
        self.assertEqual(len(owned), 30)

    def test_invalid_shards_rejected(self):
        for i, n in [(0, 0), (-1, 3), (3, 3)]:
            with self.assertRaises(AssertionError):
                shard_rows([], i, n)


if __name__ == "__main__":
    unittest.main()
