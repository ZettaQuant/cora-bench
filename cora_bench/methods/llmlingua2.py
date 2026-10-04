"""LLMLingua-2 compression (query-agnostic token classifier), capped to the budget.

torch and llmlingua are imported lazily.
"""

from __future__ import annotations

import time

from cora_bench.budgets.token_budget import BudgetEnforcer, Tokenizer
from cora_bench.methods.base import ContextSelector, SelectionResult

DEFAULT_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


class LiteralTextEncoding:
    """Wraps LLMLingua's tiktoken counter so literal "<|endoftext|>" in text doesn't raise."""

    def __init__(self, encoding):
        self.encoding = encoding

    def encode(self, text, **kwargs):
        kwargs.update(allowed_special=set(), disallowed_special=())
        return self.encoding.encode(text, **kwargs)

    def __getattr__(self, name):
        return getattr(self.encoding, name)


class LLMLingua2Selector(ContextSelector):
    name = "llmlingua2"

    def __init__(self, model: str = DEFAULT_MODEL, tokenizer: Tokenizer | None = None):
        self.model = model
        self.tok = tokenizer or Tokenizer()
        self.enforcer = BudgetEnforcer(self.tok)
        self._pc = None
        self._device = None

    def _compressor(self):
        if self._pc is None:
            import torch
            from llmlingua import PromptCompressor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pc = PromptCompressor(
                model_name=self.model, use_llmlingua2=True, device_map=self._device
            )
            self._pc.oai_tokenizer = LiteralTextEncoding(self._pc.oai_tokenizer)
        return self._pc

    def select(
        self, query: str, context: str, budget_tokens: int | None, metadata: dict | None = None
    ) -> SelectionResult:
        src = self.tok.count(context)
        if budget_tokens is None:
            return SelectionResult(
                text=context,
                selected_token_count=src,
                source_token_count=src,
                metadata={"note": "llmlingua2 has no natural arm; returned full"},
            )
        pc = self._compressor()
        t0 = time.perf_counter()
        # target_token fills the budget much more closely than rate=B/source, which underfills.
        out = pc.compress_prompt(context, target_token=budget_tokens)
        gpu_s = time.perf_counter() - t0
        raw = out.get("compressed_prompt", "") if isinstance(out, dict) else str(out)
        raw_tok = self.tok.count(raw)
        text, clipped = self.enforcer.clip_text(raw, budget_tokens)  # if target_token overshoots
        return SelectionResult(
            text=text,
            selected_token_count=self.tok.count(text),
            source_token_count=src,
            latency_seconds=round(gpu_s, 4),
            gpu_seconds=round(gpu_s, 4),
            clipped=clipped,
            metadata={
                "target_token": budget_tokens,
                "device": self._device,
                "model": self.model,
                "raw_compressed_tokens": raw_tok,
                "literal_special_tokens": True,
                "utilization": round(self.tok.count(text) / budget_tokens, 3)
                if budget_tokens
                else None,
            },
        )
