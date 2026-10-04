"""Strict-prefix packing that emits the union of the selected chunks' source-token spans.

Overlapping chunks are merged instead of repeated, and gaps are joined with a newline.
"""

from cora_bench.budgets.token_budget import BudgetEnforcer
from cora_bench.evaluation.provenance import merge_intervals


class SourceUnionEnforcer(BudgetEnforcer):
    def prepare(self, context):
        self.context = context
        self.ids = self.tok.encode(context)
        pieces = [self.tok._e.decode_single_token_bytes(i) for i in self.ids]
        self.raw = context.encode("utf-8")
        self.offsets = [0]
        for part in pieces:
            self.offsets.append(self.offsets[-1] + len(part))
        assert b"".join(pieces) == self.raw
        self.last_spans = []
        self.last_char_spans = []

    def intervals(self, chunks):
        spans = merge_intervals((c.token_start, c.token_end) for c in chunks)
        out = []
        for s, e in spans:
            assert 0 <= s <= e <= len(self.ids)
            # Drop edge tokens that split a UTF-8 character, so the recorded span decodes exactly.
            while (
                s < e
                and self.offsets[s] < len(self.raw)
                and self.raw[self.offsets[s]] & 0xC0 == 0x80
            ):
                s += 1
            while (
                e > s
                and self.offsets[e] < len(self.raw)
                and self.raw[self.offsets[e]] & 0xC0 == 0x80
            ):
                e -= 1
            if e > s:
                out.append((s, e))
        return out

    def render(self, spans):
        return "\n".join(
            self.raw[self.offsets[s] : self.offsets[e]].decode("utf-8", errors="strict")
            for s, e in spans
        )

    def pack_chunks(self, ranked, budget_tokens, order="source_order"):
        if order != "source_order":
            raise ValueError("Source union requires source order")
        kept = []
        clipped = False
        for ch in ranked:
            candidate = kept + [ch]
            spans = self.intervals(candidate)
            conservative = sum(e - s for s, e in spans) + max(0, len(spans) - 1)
            if conservative <= budget_tokens or self.tok.count(self.render(spans)) <= budget_tokens:
                kept.append(ch)
            else:
                clipped = True
                break
        spans = self.intervals(kept)
        text = self.render(spans)
        while kept and self.tok.count(text) > budget_tokens:
            kept.pop()
            clipped = True
            spans = self.intervals(kept)
            text = self.render(spans)
        self.last_spans = spans
        return sorted(kept, key=lambda c: c.original_order), text, clipped


def validate_source_union(context, text, spans, tokenizer):
    enc = SourceUnionEnforcer(tokenizer)
    enc.prepare(context)
    if enc.render(spans) != text:
        raise ValueError("Emitted text disagrees with its source spans")
    if merge_intervals(spans) != [tuple(x) for x in spans]:
        raise ValueError("Spans are not merged/source-ordered")
    return True
