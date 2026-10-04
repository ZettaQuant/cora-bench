import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cora_bench.budgets.token_budget import Tokenizer
from cora_bench.chunking.token_chunker import TokenChunker

from .api import Ledger, digest
from .packing import SourceUnionEnforcer, validate_source_union
from .prompts import request


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()

    def test_union_no_duplicates_and_more_chunks_fit(self):
        text = " ".join("word" + str(i) for i in range(600))
        chunks = TokenChunker(32, 8, self.tok).chunk(text)
        p = SourceUnionEnforcer(self.tok)
        p.prepare(text)
        selected, out, _ = p.pack_chunks(chunks, 56)
        self.assertEqual(len(selected), 2)
        self.assertEqual(out, self.tok.decode(self.tok.encode(text)[:56]))
        self.assertEqual(self.tok.count(out), 56)
        validate_source_union(text, out, p.last_spans, self.tok)

    def test_gaps_have_one_separator_and_source_order(self):
        text = " ".join("x" + str(i) for i in range(200))
        chunks = TokenChunker(32, 8, self.tok).chunk(text)
        p = SourceUnionEnforcer(self.tok)
        p.prepare(text)
        _, out, _ = p.pack_chunks([chunks[4], chunks[0]], 100)
        self.assertEqual(out, chunks[0].text + "\n" + chunks[4].text)
        validate_source_union(text, out, p.last_spans, self.tok)

    def test_unicode_tables_literal_tokens_and_roundtrip(self):
        text = "公司😀 café\r\nMetric | 2024\nEPS | ($3.91)\n<|endoftext|>\n" * 100
        p = SourceUnionEnforcer(self.tok)
        p.prepare(text)
        chunks = TokenChunker(31, 7, self.tok).chunk(text)
        for budget in [0, 1, 31, 63, 99, 300]:
            _, out, _ = p.pack_chunks(chunks[::2], budget)
            self.assertLessEqual(self.tok.count(out), budget)
            self.assertNotIn("\ufffd", out)
            validate_source_union(text, out, p.last_spans, self.tok)

    def test_reader_never_sees_gold_or_method(self):
        row = {
            "query": "What is EPS?",
            "answer": "SECRET_GOLD",
            "method": "SECRET_METHOD",
            "instruction": "",
        }
        a = request("loong", row, "Example EPS $4", 5768)
        b = request("loong", row, "Example EPS $4", 78516)
        self.assertNotIn("SECRET", str(a))
        self.assertNotEqual(digest(a), digest(b))

    def test_budget_atomic_reservations(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = Ledger(Path(d) / "budget.db")

            def reserve(i):
                try:
                    ledger.reserve(str(i), "online_smoke_latency", 9)
                    return True
                except RuntimeError:
                    return False

            with ThreadPoolExecutor(max_workers=8) as pool:
                ok = list(pool.map(reserve, range(8)))
            self.assertEqual(sum(ok), 4)


if __name__ == "__main__":
    unittest.main()
