"""CPU-only regression tests for evidence scoring, caching, and report integrity."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cora_bench.analysis.dataset_report import index_selections
from cora_bench.budgets.token_budget import Tokenizer
from cora_bench.chunking.token_chunker import TokenChunker
from cora_bench.data.babilong import BabiLongAdapter
from cora_bench.data.base import Example, GoldEvidence
from cora_bench.data.longmemeval import LongMemEvalAdapter
from cora_bench.evaluation.loong_evidence import candidate_line_retention
from cora_bench.evaluation.provenance import char_token_span
from cora_bench.evaluation.selection import score_selection
from cora_bench.instrumentation.selection_cost import selection_cost
from cora_bench.methods.dense_rerank import DenseRerankSelector


class EvaluationTests(unittest.TestCase):
    def test_loong_false_positives(self):
        self.assertEqual(
            candidate_line_retention("$101", "Revenue $101\nOther $100", "Other $100")[
                "fraction_retained"
            ],
            0,
        )
        self.assertEqual(
            candidate_line_retention("Apple Inc.", "Apple Inc.", "pineapple orchard")[
                "fraction_retained"
            ],
            0,
        )
        self.assertEqual(
            candidate_line_retention("$101", "Revenue $101", "Revenue $101")["fraction_retained"], 1
        )
        self.assertFalse(candidate_line_retention("$101", "No relevant row", "")["evaluable"])

    def test_babilong_complete_occurrence(self):
        tok = Tokenizer()
        fact = "Daniel moved to the office."
        for n in range(1, 64):
            ctx = "Filler. " * n + fact + " Padding." * 120 + " " + fact
            cs = ctx.index(fact)
            chunks = TokenChunker(16, 0, tok).chunk(ctx, "probe")
            start, end = char_token_span(tok.char_offsets(ctx), cs, cs + len(fact))
            first = next(c for c in chunks if c.token_start <= start < c.token_end)
            duplicates = [c for c in chunks if fact in c.text and c.token_start > first.token_end]
            if end <= first.token_end or not duplicates:
                continue
            ex = Example(
                "probe",
                "Where is Daniel?",
                ctx,
                "office",
                gold_evidence=[
                    GoldEvidence(fact, metadata={"char_start": cs, "char_end": cs + len(fact)})
                ],
            )
            chosen = [first, duplicates[0]]
            result = score_selection(
                BabiLongAdapter(),
                ex,
                "\n".join(c.text for c in chosen),
                [c.chunk_id for c in chosen],
                method="bm25",
                chunk_size=16,
                overlap=0,
            )
            self.assertEqual(result["fraction_retained"], 0)
            complete = [c for c in chunks if c.token_start < end and start < c.token_end]
            result = score_selection(
                BabiLongAdapter(),
                ex,
                "\n".join(c.text for c in complete),
                [c.chunk_id for c in complete],
                method="bm25",
                chunk_size=16,
                overlap=0,
            )
            self.assertEqual(result["fraction_retained"], 1)
            return
        self.fail("No boundary fixture constructed")

    def test_provence_does_not_inherit_parent_turn(self):
        text = "Mary lives in London. Her dog is Max."
        ex = Example(
            "e",
            "Where?",
            text,
            "London",
            gold_evidence=[
                GoldEvidence(
                    text,
                    metadata={
                        "token_start": 0,
                        "token_end": Tokenizer().count(text),
                        "session_neutral": 1,
                    },
                )
            ],
        )
        result = score_selection(
            LongMemEvalAdapter(), ex, "Mary lives in London.", ["e::0"], method="provence"
        )
        self.assertEqual(result["fraction_retained"], 0)
        self.assertEqual(result["provenance_mode"], "verbatim")

    def test_forged_provenance_rejected(self):
        ex = Example("e", "?", "Mary lives in London.", "London")
        with self.assertRaises(ValueError):
            score_selection(LongMemEvalAdapter(), ex, "removed", ["e::0"], method="bm25")

    def test_partial_selection_rejected(self):
        with self.assertRaises(ValueError):
            score_selection(
                BabiLongAdapter(),
                Example("e", "?", "", "x"),
                "",
                method="bm25",
                metadata={"n_unscored": 1},
            )

    def test_budget_order_invariant(self):
        class FakeEmbed:
            def encode(self, texts, **kw):
                return (
                    np.array([[1.0]])
                    if len(texts) == 1
                    else np.arange(len(texts), 0, -1, dtype=float).reshape(-1, 1)
                )

        selector = DenseRerankSelector(chunk_size=16, overlap=0, candidate_count=1)
        selector._emb_model = lambda: FakeEmbed()
        selector._rerank_scores = lambda q, docs: list(range(len(docs)))
        ctx = " ".join("item" + str(i) for i in range(300))
        a = selector.select("query", ctx, 16, {"example_id": "probe"})
        selector.select("query", ctx, 64, {"example_id": "probe"})
        b = selector.select("query", ctx, 16, {"example_id": "probe"})
        self.assertEqual(a.text, b.text)
        c = selector.select("query", ctx, 16, {"example_id": "other"})
        self.assertTrue(all(cid.startswith("other::") for cid in c.selected_chunk_ids))

    def test_cross_file_dedup_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl"
            row = {
                "method": "bm25",
                "budget": 32,
                "example_id": "e",
                "filtered_text": "x",
                "selected_chunk_ids": ["e::0"],
            }
            a.write_text(json.dumps(row) + "\n")
            b.write_text(json.dumps(row) + "\n")
            index, _ = index_selections([a, b])
            self.assertEqual(len(index), 1)
            row["filtered_text"] = "different"
            b.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                index_selections([a, b])

    def test_unknown_cost_not_zero(self):
        self.assertIsNone(
            selection_cost("external_selector", {"cost_status": "unknown_api_price"}, None)
        )
        self.assertIsNone(selection_cost("bm25", {}, None))

    def test_llmlingua_literal_special_tokens(self):
        import tiktoken

        from cora_bench.methods.llmlingua2 import LiteralTextEncoding

        enc = LiteralTextEncoding(tiktoken.get_encoding("cl100k_base"))
        text = "A chat quote: <|endoftext|> remains ordinary text."
        self.assertEqual(enc.decode(enc.encode(text)), text)

    def test_cluster_bootstrap_preserves_pairs(self):
        from cora_bench.analysis.paired_statistics import paired_interval

        rows = {
            str(i): {
                "dataset": "nolima",
                "example_id": str(i),
                "strata": {"needle_id": i // 3},
                "evidence": {"fraction_retained": i % 2},
                "error": None,
            }
            for i in range(12)
        }
        result = paired_interval(rows, rows, repetitions=100)
        self.assertEqual(result["ci95"], [0.0, 0.0])
        self.assertEqual(result["n_clusters"], 4)

    def test_reader_and_report_agree_for_provence(self):
        from cora_bench.config.schema import RunConfig
        from cora_bench.instrumentation.cost import Pricing
        from cora_bench.readers.base import ReaderResult
        from cora_bench.runners.run import read_one

        class FakeReader:
            name = "fake"
            model = "fake"

            def generate(self, *args, **kw):
                return ReaderResult("London", "London")

        text = "Mary lives in London. Her dog is Max."
        ex = Example(
            "e",
            "Where?",
            text,
            "London",
            gold_evidence=[
                GoldEvidence(
                    text,
                    metadata={
                        "token_start": 0,
                        "token_end": Tokenizer().count(text),
                        "session_neutral": 1,
                    },
                )
            ],
        )
        selected = {
            "method": "provence",
            "budget": 32,
            "source_tokens": Tokenizer().count(text),
            "filtered_text": "Mary lives in London.",
            "selected_chunk_ids": ["e::0"],
            "metadata": {"prunes_within_chunk": True},
        }
        row = read_one(
            LongMemEvalAdapter(),
            FakeReader(),
            ex,
            selected,
            RunConfig("test", "longmemeval", "provence", "fake"),
            Pricing("test", {}, {}),
        )
        self.assertIsNone(row["error_type"])
        self.assertEqual(row["evidence_fraction_retained"], 0)
        self.assertIsNone(row["total_cost_usd"])

    def test_failure_repair_requires_explicit_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "original.jsonl", Path(tmp) / "repair.jsonl"
            original = {"method": "llmlingua2", "budget": 32, "example_id": "e", "error": "failed"}
            a.write_text(json.dumps(original) + "\n")
            corrected = {
                "method": "llmlingua2",
                "budget": 32,
                "example_id": "e",
                "filtered_text": "x",
                "source_sha256": "hash",
                "config_fingerprint": "version",
                "repair_of": str(a),
            }
            b.write_text(json.dumps(corrected) + "\n")
            index, _ = index_selections([a], [b])
            self.assertEqual(index[("llmlingua2", 32, "e")][0], b)
            original.pop("error")
            original["filtered_text"] = "old"
            a.write_text(json.dumps(original) + "\n")
            with self.assertRaises(ValueError):
                index_selections([a], [b])

    def test_input_identity_tracks_contents(self):
        from cora_bench.scripts.select_evidence import input_identity

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ctx").mkdir()
            rows = [{"example_id": "e"}]
            (root / "input.jsonl").write_text(json.dumps(rows[0]) + "\n")
            source = root / "ctx" / "e.txt"
            source.write_text("original")
            first = input_identity(root, rows)
            source.write_text("changed")
            second = input_identity(root, rows)
            self.assertNotEqual(first["context_sha256"], second["context_sha256"])

    def test_missing_annotations_are_not_selector_failures(self):
        from cora_bench.analysis.paired_statistics import paired_interval

        rows = {
            str(i): {
                "dataset": "loong",
                "example_id": str(i),
                "evidence_evaluable": i != 0,
                "evidence": {"fraction_retained": 1},
                "error": None,
            }
            for i in range(4)
        }
        result = paired_interval(rows, rows, repetitions=100)
        self.assertEqual(result["n_paired"], 3)
        self.assertEqual(result["n_missing_annotations"], 1)
        self.assertEqual(result["ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
