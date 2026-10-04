"""Dense retrieval plus reranking: Qwen3-Embedding picks candidates, Qwen3-Reranker reorders them.

The reranker scores each candidate with the model card's yes/no prompt. Heavy imports are lazy.
"""

from __future__ import annotations

import time

from cora_bench.budgets.token_budget import BudgetEnforcer, Tokenizer
from cora_bench.chunking.token_chunker import TokenChunker
from cora_bench.methods.base import ContextSelector, SelectionResult

EMB_MODEL = "Qwen/Qwen3-Embedding-4B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-4B"
_INSTRUCT = "Given a question, retrieve passages that help answer it"
_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class DenseRerankSelector(ContextSelector):
    name = "dense_rerank"

    def __init__(
        self,
        emb_model: str = EMB_MODEL,
        rerank_model: str = RERANK_MODEL,
        chunk_size: int = 512,
        overlap: int = 64,
        chunk_order: str = "source_order",
        candidate_count: int = 100,
        rerank_batch: int = 8,
        tokenizer: Tokenizer | None = None,
    ):
        self.emb_model, self.rerank_model = emb_model, rerank_model
        self.tok = tokenizer or Tokenizer()
        self.chunker = TokenChunker(chunk_size, overlap, self.tok)
        self.enforcer = BudgetEnforcer(self.tok)
        self.chunk_order = chunk_order
        self.candidate_count = candidate_count
        self.rerank_batch = rerank_batch
        self._emb = None
        self._rk = None

    def _emb_model(self):
        if self._emb is None:
            import torch
            from sentence_transformers import SentenceTransformer

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._emb = SentenceTransformer(
                self.emb_model, device=dev, model_kwargs={"torch_dtype": "float16"}
            )
        return self._emb

    def _reranker(self):
        if self._rk is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tk = AutoTokenizer.from_pretrained(self.rerank_model, padding_side="left")
            mdl = AutoModelForCausalLM.from_pretrained(self.rerank_model, torch_dtype=torch.float16)
            mdl = (mdl.cuda() if torch.cuda.is_available() else mdl).eval()
            self._rk = (
                tk,
                mdl,
                tk.convert_tokens_to_ids("no"),
                tk.convert_tokens_to_ids("yes"),
                tk.encode(_PREFIX, add_special_tokens=False),
                tk.encode(_SUFFIX, add_special_tokens=False),
            )
        return self._rk

    def _rerank_scores(self, query: str, docs: list[str]) -> list[float]:
        import torch

        tk, mdl, no_id, yes_id, pre, suf = self._reranker()
        maxlen = 8192
        pairs = [f"<Instruct>: {_INSTRUCT}\n<Query>: {query}\n<Document>: {d}" for d in docs]
        out: list[float] = []
        for i in range(0, len(pairs), self.rerank_batch):
            grp = pairs[i : i + self.rerank_batch]
            enc = tk(
                grp,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=maxlen - len(pre) - len(suf),
            )
            enc["input_ids"] = [pre + e + suf for e in enc["input_ids"]]
            enc = tk.pad(enc, padding=True, return_tensors="pt", max_length=maxlen)
            enc = {k: v.to(mdl.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = mdl(**enc).logits[:, -1, :]
                pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
                probs = torch.nn.functional.log_softmax(pair, dim=1)[:, 1].exp().tolist()
            out.extend(probs)
        return out

    def select(
        self, query: str, context: str, budget_tokens: int | None, metadata: dict | None = None
    ) -> SelectionResult:
        import hashlib
        import math

        # Rerank enough candidates to fill the budget, not just candidate_count.
        need = (
            self.candidate_count
            if budget_tokens is None
            else max(self.candidate_count, math.ceil(budget_tokens / self.chunker.chunk_size) + 8)
        )
        # Chunk embeddings are query-independent; the ranking depends on the query, so its cache
        # key includes the query and candidate depth. Both keys also include the example id.
        ckey = hashlib.sha1(context.encode("utf-8")).hexdigest()
        eid = (metadata or {}).get("example_id")
        rkey = (hashlib.sha1((context + "\x00" + query).encode("utf-8")).hexdigest(), need, eid)
        ckey = (ckey, eid)
        doc_cache = getattr(self, "_doc_cache", None) or {}
        self._doc_cache = doc_cache
        rank_cache = getattr(self, "_rank_cache", None) or {}
        self._rank_cache = rank_cache
        t0 = time.perf_counter()
        rentry = rank_cache.get(rkey)
        if rentry and rentry["depth"] == need:
            chunks, src, ranked = rentry["chunks"], rentry["src"], rentry["ranked"]
            gpu_s = 0.0
        else:
            m = self._emb_model()
            dentry = doc_cache.get(ckey)
            if dentry:
                chunks, src, doc_emb = dentry["chunks"], dentry["src"], dentry["doc_emb"]
            else:
                chunks = self.chunker.chunk(context, doc_id=(metadata or {}).get("example_id"))
                src = self.tok.count(context)
                doc_emb = m.encode(
                    [c.text for c in chunks],
                    batch_size=32,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                doc_cache[ckey] = {"chunks": chunks, "src": src, "doc_emb": doc_emb}
            try:
                q_emb = m.encode(
                    [query], prompt_name="query", normalize_embeddings=True, convert_to_numpy=True
                )[0]
            except Exception:
                q_emb = m.encode(
                    [f"Instruct: {_INSTRUCT}\nQuery: {query}"],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )[0]
            sim = doc_emb @ q_emb
            cand_idx = sorted(range(len(chunks)), key=lambda i: (-float(sim[i]), i))[:need]
            rerank = self._rerank_scores(query, [chunks[i].text for i in cand_idx])
            ranked = [chunks[i] for i, _ in sorted(zip(cand_idx, rerank), key=lambda z: -z[1])]
            rank_cache[rkey] = {"chunks": chunks, "src": src, "ranked": ranked, "depth": need}
            gpu_s = time.perf_counter() - t0
        if budget_tokens is None:
            selected = sorted(chunks, key=lambda c: c.original_order)
            text, clipped = "\n".join(c.text for c in selected), False
        else:
            selected, text, clipped = self.enforcer.pack_chunks(
                ranked, budget_tokens, self.chunk_order
            )
        return SelectionResult(
            text=text,
            selected_token_count=self.tok.count(text),
            source_token_count=src,
            selected_chunk_ids=[c.chunk_id for c in selected],
            latency_seconds=round(time.perf_counter() - t0, 4),
            gpu_seconds=round(gpu_s, 4),
            clipped=clipped,
            metadata={
                "n_chunks": len(chunks),
                "candidate_count": min(need, len(chunks)),
                "n_selected": len(selected),
                "emb_model": self.emb_model,
                "rerank_model": self.rerank_model,
                "candidate_policy": "budget_depth_v2",
                "cache_hit": bool(rentry),
                "chunk_size_tokens": self.chunker.chunk_size,
                "chunk_overlap_tokens": self.chunker.overlap,
            },
        )
