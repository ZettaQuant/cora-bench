"""Source interval utilities. Intervals are half-open in the shared tokenizer."""

from bisect import bisect_left, bisect_right
from functools import lru_cache

from cora_bench.budgets.token_budget import Tokenizer
from cora_bench.chunking.token_chunker import TokenChunker


@lru_cache(maxsize=2)
def source_layout(context, example_id, chunk_size=512, overlap=64):
    tok = Tokenizer()
    chunks = TokenChunker(chunk_size, overlap, tok).chunk(context, example_id)
    return chunks, tok.char_offsets(context)


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if start < 0 or end < start:
            raise ValueError("Invalid source interval")
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def covers(intervals, start, end):
    return end > start and any(s <= start and end <= e for s, e in intervals)


def char_token_span(offsets, start, end):
    if not offsets or not (0 <= start < end):
        raise ValueError("Empty or invalid evidence span")
    left = bisect_left(offsets, start)
    if left == len(offsets) or offsets[left] != start:
        left = max(0, bisect_right(offsets, start) - 1)
    return left, bisect_left(offsets, end)


def chunk_intervals(example, chunk_ids, chunk_size=512, overlap=64):
    chunks, _ = source_layout(example.context, example.example_id, chunk_size, overlap)
    by_id = {c.chunk_id: c for c in chunks}
    unknown = set(chunk_ids) - by_id.keys()
    if unknown:
        raise ValueError(f"Chunk IDs do not belong to source: {sorted(unknown)[:3]}")
    return merge_intervals((by_id[c].token_start, by_id[c].token_end) for c in chunk_ids)
