# Experimental protocol

Six selectors share the query and source text: head/tail truncation, BM25, Qwen3-Embedding-0.6B, Qwen3 dense retrieval followed by Qwen3-Reranker-0.6B, LLMLingua-2, and Provence. Full and empty contexts are controls. Gold answers, support annotations, and selector identities are excluded from reader requests.

The current primary protocol corrects duplicated overlap in BM25/dense/dense-rerank packing. Ranked chunks are accepted as a strict prefix; their source spans are unioned and restored to source order. Final emitted text must fit the shared token cap. Transformed outputs retain their own text. Tables receive no special preservation or whitespace normalization. The concatenation-based evidence experiment uses different selected texts and must be reported separately.

Reader: `gemini-2.5-flash-lite`, temperature 0, thinking budget 0, maximum 1,024 output tokens, online concurrency 8. Seeds are 5768, 78516, and 944601 on identical questions. A provider seed does not guarantee bitwise determinism. Identical requests may share a receipt within one seed, never across seeds. Identical judge requests may reuse a judgment; this is not an estimate of judge variance.

BABILong uses the pinned official task-label scorer. NoLiMa uses the released case-sensitive contains metric on the custom frozen diagnostic, without official base-length normalization. LongMemEval uses its question-type and abstention rubric with `gpt-4o-2024-08-06`. LOONG uses its original rubric with adapted judge `gpt-5.5-2026-04-23`, reasoning disabled, 512 output tokens; Perfect Rate requires exactly 100. Preserve numeric parser outputs and flag rubric-range violations rather than silently clipping them.

At 32K/64K/128K, LOONG has 75/70/51 eligible questions. Cross-budget curves must use the common 51-question cohort. LongMemEval evidence recall excludes 30 abstention questions; answer accuracy includes all 500. LOONG automatically matched candidate lines are not human-reviewed gold evidence.

Confidence intervals use 2,000 paired bootstrap draws after averaging reader repetitions within each question. BABILong is task-stratified; NoLiMa clusters related questions, depths, and backgrounds by needle family. Three seed scores are reported separately. These intervals reflect uncertainty over the evaluated cohort, not judge calibration or generalization to new datasets.

Standalone selection timings include chunking, query encoding, inference, and packing, with CUDA synchronization and fresh data caches. Cold model loading is separate. The separate reader latency pass uses 30 matched questions per cell at seed 5768 and concurrency 8. Serving cost uses native API usage plus the whole-instance selector time; judging and experiment setup costs are reported separately. Cached incremental selection times are not standalone latency measurements.
