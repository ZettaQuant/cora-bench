"""Context selector registry."""

from __future__ import annotations

from cora_bench.budgets.token_budget import Tokenizer
from cora_bench.config.schema import RunConfig
from cora_bench.methods.base import ContextSelector


def get_selector(name: str, cfg: RunConfig, tokenizer: Tokenizer | None = None) -> ContextSelector:
    tok = tokenizer or Tokenizer(cfg.tokenizer_encoding)
    if name == "full_context":
        from cora_bench.methods.full_context import FullContext

        return FullContext(tok)
    if name in ("head", "tail", "head_tail"):
        from cora_bench.methods.truncation import Truncation

        return Truncation(name, tok)
    if name == "bm25":
        from cora_bench.methods.bm25 import BM25Selector

        return BM25Selector(cfg.chunk_size_tokens, cfg.chunk_overlap_tokens, cfg.chunk_order, tok)
    if name == "llmlingua2":
        from cora_bench.methods.llmlingua2 import LLMLingua2Selector

        return LLMLingua2Selector(
            cfg.compressor_model
            or "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            tok,
        )
    if name == "dense":
        from cora_bench.methods.dense import DenseSelector

        return DenseSelector(
            cfg.embedding_model or "Qwen/Qwen3-Embedding-0.6B",
            cfg.chunk_size_tokens,
            cfg.chunk_overlap_tokens,
            cfg.chunk_order,
            tokenizer=tok,
        )
    if name == "provence":
        # Provence uses its own checkpoint, independent of the dense reranker.
        from cora_bench.methods.provence import ProvenceSelector

        return ProvenceSelector(
            cfg.extra.get("provence_model", "naver/provence-reranker-debertav3-v1")
            if cfg.extra
            else "naver/provence-reranker-debertav3-v1",
            cfg.chunk_size_tokens,
            cfg.chunk_overlap_tokens,
            tokenizer=tok,
        )
    if name == "dense_rerank":
        from cora_bench.methods.dense_rerank import DenseRerankSelector

        return DenseRerankSelector(
            cfg.embedding_model or "Qwen/Qwen3-Embedding-0.6B",
            cfg.reranker_model or "Qwen/Qwen3-Reranker-0.6B",
            cfg.chunk_size_tokens,
            cfg.chunk_overlap_tokens,
            cfg.chunk_order,
            cfg.retrieval_candidate_count,
            tokenizer=tok,
        )
    raise KeyError(f"unknown method {name!r}")
