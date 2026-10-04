# Code map

| File or directory | Responsibility |
| --- | --- |
| `cora_bench/cli.py` | Public commands and stage-specific help |
| `scripts/prepare_babilong.py` | Seeded bAbI/PG19 generation with support spans |
| `scripts/prepare_loong.py` | Financial-document assembly in the pinned order |
| `scripts/prepare_nolima.py` | Frozen needle/background placements |
| `scripts/prepare_longmemeval.py` | Conversation assembly and answer-turn sidecars |
| `scripts/verify_data.py` | Exact file-hash and cohort verification |
| `methods/` | Context selector implementations |
| `chunking/`, `budgets/` | Shared token accounting and concatenation-based packing |
| `experiment/packing.py` | Overlap-merged packing and source-span validation |
| `experiment/select_context.py` | Selection matrix and standalone timing controls |
| `experiment/generate_answers.py` | Saved-context reader requests, retries, and receipts |
| `experiment/score_answers.py` | Native dataset scorers and judge rubrics |
| `experiment/judge_cache.py` | Exact-request judgment reuse and durable billing records |
| `experiment/audit_results.py` | Identity, metric, and sampled-context audit |
| `experiment/summarize_results.py` | Paired intervals, comparisons, and Pareto plots |
| `experiment/run_pipeline.py` | Local coordination of reader, scoring, and reporting |
| `experiment/upstream/`, `audit_reference/` | Pinned scoring/prompt sources and attribution |
| `scripts/select_evidence.py`, `analysis/dataset_report.py` | Evidence-only selection and scoring |
| `analysis/paper_tables.py` | LaTeX tables from an audited aggregate snapshot |

Paths after the first row are relative to `cora_bench/`. Comments explain protocol decisions and invariants; upstream attribution is retained. The public selector set excludes experimental methods outside the paper.
