# Reproducing CoRA-Bench

## Environment

Run from the checkout root. The recorded selector workers used Python **3.10.12**. The CPU and GPU environments differed, including their `tiktoken` package versions; both used the same `cl100k_base` encoding and saved source-token identities.

For the CPU selector and reader environment:

```bash
python -m pip install -c requirements/constraints-cpu.txt -e '.[dev]'
```

For CUDA selectors, use a separate environment. The recorded PyTorch build was `2.9.1+cu129` on an NVIDIA L4:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -c requirements/constraints-gpu.txt -e '.[gpu]'
python -m nltk.downloader punkt punkt_tab
```

The constraints pin recorded benchmark packages, not every operating-system library or transitive dependency. They are not a cross-platform lockfile. [recorded_environments.json](../requirements/recorded_environments.json) preserves the observed versions; the dependency resolver and wheel availability still depend on your platform. Data preparation requires `.[data]`. The five generation-package versions captured for BABILong are in `requirements/constraints-data.txt`; its full transitive environment and the PG19 revision were not captured. Check generated data against the frozen hashes instead of assuming a matching seed is sufficient.

Model identifiers are frozen in the protocol. Immutable weight revisions were not recorded for every selector, and hosted API implementations can change. Exact bitwise reproduction is therefore not guaranteed. Save your generated manifests and provider model versions with results. Tokenizer and provider-native token counts are checked separately.

## Data and validation

Follow [DATA.md](DATA.md), then:

```bash
cora-bench verify-data
# Or validate one prepared dataset at a time:
cora-bench verify-data --datasets loong
```

The verifier checks the exact bytes of every frozen `input.jsonl`, source context, and evidence sidecar. It also checks cohort sizes and question-ID uniqueness. Missing or modified files produce a nonzero exit status. Expected counts are 500 BABILong, 75 LOONG, 870 NoLiMa, and 500 LongMemEval questions.

## Full reader experiment

The primary entry points are:

```bash
cora-bench select cpu
cora-bench select gpu
cora-bench answer --phase main
cora-bench score --phase main
cora-bench audit
cora-bench answer --phase latency
cora-bench report
```

CPU methods are head/tail and BM25. GPU methods are dense retrieval, dense retrieval with reranking, LLMLingua-2, and Provence. Selection writes source-order contexts and provenance under `selection/`. Reader requests use those saved texts without rechunking. Full and empty contexts are controls.

The matrix contains 68 dataset/method/budget cells including controls, and 91,518 logical reader answers across three seeds. A logical answer may reuse an identical request within a seed; it never borrows an answer from another seed. LOONG eligibility changes with the budget (75/70/51 questions), so cross-budget comparisons require a common cohort.

The reader uses Vertex AI application-default credentials plus `GOOGLE_CLOUD_PROJECT`. Judges use `OPENAI_API_KEY`. Set credentials in your process environment; `.env.example` is documentation, not a file automatically loaded by the primary workflow. Selection does not require reader/judge credentials. Some upstream data/model downloads may require `HF_TOKEN`.

The protocol contains per-category spending ceilings. These are authorization limits for a local run, not estimates of its final cost. Paid stages are explicit commands. Before changing datasets, models, prompts, seeds, or budgets, create a new protocol and output directory:

```bash
export CORA_PROTOCOL=/absolute/path/to/my_protocol.json
# Set local_outputs to a new directory inside that JSON.
cora-bench select cpu
```

Never edit a protocol and resume into an old output directory. Request hashes, model/configuration identity, saved code hashes, and ledger reservations protect resumption. An uncertain API attempt must be inspected; deleting its reservation can lead to duplicate billing. The audit stores separate results under `interim_audits/` and reports incomplete cells without assigning them partial-cohort headline accuracy.

## Outputs

All paths below are relative to `local_outputs` in the protocol:

| Path | Meaning |
| --- | --- |
| `selection/{cpu,gpu}/` | Selected text, source spans, evidence diagnostics, timing, and manifests |
| `reader/main/links/` | One identity record per dataset/question/method/budget/seed |
| `reader/main/receipts/` | Provider outputs, token usage, model version, and timings |
| `judges/main/` | Dataset-native scores and judgment provenance |
| `judges/api_receipts/` | Reusable exact-request judge responses |
| `interim_audits/` | Independent scoring and sampled-context checks |
| `reader/latency/` | Fresh matched latency requests at fixed concurrency |
| `report/` | Aggregate statistics, confidence intervals, tables, and Pareto figures |

Latency measurements are separate from bulk generation. Reported serving latency includes selection, query encoding, and reader latency; startup and experiment/judging costs are distinct. The standalone CPU timing stage reruns selections and verifies byte-identical output.

Cloud export is optional. Set `CORA_GCS_PREFIX` to your own bucket and pass `--cloud` where supported. VM provisioning, shutdown monitors, recovery logs, and private infrastructure settings are not part of this repository.

## Evidence-only experiment

The paper's evidence-only tables use the concatenation-based packer. That workflow remains available so the two protocols are not conflated:

```bash
cora-bench select-evidence --in cora_bench/outputs/materialized/loong \
  --out outputs/evidence_example --methods head_tail,bm25 \
  --budgets 32000,64000,128000
cora-bench evidence --dataset loong --sel-dirs outputs/evidence_example \
  --out outputs/evidence_example/summary.csv
```

This is an example invocation, not a claim to reconstruct every submitted-paper table. Match the original table's cohort, budgets, and configuration when reproducing it. Use the overlap-merged primary workflow for the new reader results. Evidence recall, verbatim survival, and final-answer correctness are different measurements.

## Validation and provenance

```bash
python -m pytest -q
cora-bench smoke
python -m build --wheel
```

Tests cover packing, Unicode/table text, provenance, gold-label isolation, native scorers, retries, cost reservations, data verification, and incomplete-cell reporting. No paid API calls or GPU inference run in the offline suite. The BABILong integration test is skipped unless its separate materialized data exists.

Bundled upstream sources and licenses are byte-identical to their pinned commits. [source_map.json](source_map.json) maps each release path to the file that produced the paper's results. Hashes differ because of renames and comment cleanup; [validation.json](validation.json) lists the scientific files whose executable code is unchanged.
