"""Budget enforcement and chunk packing invariants."""

from __future__ import annotations

from cora_bench.budgets.token_budget import BudgetEnforcer, Tokenizer
from cora_bench.chunking.token_chunker import TokenChunker

TOK = Tokenizer()
BE = BudgetEnforcer(TOK)


def _chunks(text: str, size: int = 64, overlap: int = 8):
    return TokenChunker(size, overlap, TOK).chunk(text, doc_id="d")


def test_pack_never_exceeds_budget_incl_separators():
    chunks = _chunks(("alpha beta gamma delta. " * 400).strip())
    for B in (50, 100, 250, 500, 1000):
        sel, text, _ = BE.pack_chunks(chunks, B)
        assert TOK.count(text) <= B, (TOK.count(text), B)


def test_pack_provenance_matches_transmitted_text():
    # the returned chunks, joined with newlines, must equal the packed text
    chunks = _chunks(("one two three four five. " * 300).strip())
    for B in (60, 175, 400):
        sel, text, _ = BE.pack_chunks(chunks, B)
        assert "\n".join(c.text for c in sel) == text
        assert TOK.count(text) <= B


def test_clip_text_guarantees_budget():
    big = " ".join(f"line{i} value {i * 3}" for i in range(5000))
    for B in (10, 128, 999):
        clipped, was = BE.clip_text(big, B)
        assert TOK.count(clipped) <= B
        assert was is True


def test_clip_text_noop_when_fits():
    s = "short text"
    out, was = BE.clip_text(s, 1000)
    assert out == s and was is False


def test_chunker_deterministic():
    text = ("repeatable content here. " * 200).strip()
    a = _chunks(text)
    b = _chunks(text)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


def test_pack_empty_and_tiny_budget():
    chunks = _chunks("hello world foo bar baz")
    sel, text, clipped = BE.pack_chunks(chunks, 1)
    assert TOK.count(text) <= 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
