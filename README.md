# CoRA-Bench

CoRA-Bench evaluates evidence preservation and answer quality under fixed context budgets. It compares head/tail truncation, BM25, dense retrieval, dense retrieval with reranking, LLMLingua-2, and Provence on BABILong, LOONG, a custom NoLiMa diagnostic, and LongMemEval.

## Quick start

From the repository root, using Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
cora-bench smoke
```

The smoke example selects relevant text with BM25, checks the token budget, and verifies that the answer sentence survives. It needs no dataset or API credentials. `python -m cora_bench` is equivalent to `cora-bench`.

## Reproduce the experiment

1. Follow [data preparation](docs/DATA.md), then run `cora-bench verify-data`.
2. Choose the recorded environment and review the settings in [the reproduction guide](docs/REPRODUCING.md).
3. Run the stages below. Selection uses local models; answer generation and some scorers make paid API calls.

```bash
# Selection: CPU methods, then CUDA methods (requires the gpu extra).
cora-bench select cpu
cora-bench select gpu

# Reader and native dataset scoring: configure credentials first.
export GOOGLE_CLOUD_PROJECT=your-project-id
gcloud auth application-default login
# Set OPENAI_API_KEY securely in your environment.
cora-bench answer --phase main
cora-bench score --phase main
cora-bench audit

# Separate matched latency pass, followed by tables and Pareto plots.
cora-bench answer --phase latency
cora-bench report
```

`cora-bench run` coordinates the reader, scoring, latency, and reporting stages after selection. It resumes saved work and stops on unresolved errors. Output directories and spending caps are in [protocol.json](cora_bench/experiment/protocol.json). The example environment file lists the required variable names; commands do not source it automatically.

## Protocol

- Shared accounting: `cl100k_base`, 512-token chunks, 64-token overlap. Retrieval packing merges overlapping source spans and restores source order.
- Reader: `gemini-2.5-flash-lite`, temperature 0, thinking disabled, 1,024 output tokens, concurrency 8.
- The same questions are evaluated at seeds **5768, 78516, and 944601**. Selectors have one full deterministic pass and three-seed controls on a matched subset.
- Confidence intervals use 2,000 paired bootstrap draws over question-level seed averages, with task stratification for BABILong and needle-family clustering for NoLiMa.
- LOONG uses Perfect Rate (judge score exactly 100), with its original rubric and an adapted GPT-5.5 judge. LongMemEval uses its native question-type rubric and GPT-4o judge. Full details are in [PROTOCOL.md](docs/PROTOCOL.md).

The paper's concatenation-based evidence experiment has a separate `cora-bench evidence` entry point. Its selections differ from overlap-merged reader selections; see [reproduction](docs/REPRODUCING.md#evidence-only-experiment). The NoLiMa cohort is a custom fixed-context diagnostic, not the official benchmark protocol.

## Repository guide

| Location | Contents |
| --- | --- |
| `cora_bench/experiment/` | Selection, reader, native scoring, auditing, and reporting |
| `cora_bench/methods/` | The paper's context selectors |
| `cora_bench/scripts/prepare_*.py` | Dataset preparation |
| `cora_bench/config/data_files.json` | Frozen question, context, and evidence hashes |
| `cora_bench/tests/` | Token, evidence, data, and packaging tests |
| `requirements/` | Recorded dependency versions |
| `docs/` | Data, protocol, and reproduction instructions |

Raw datasets, model weights, credentials, deployment scripts, and intermediate experiment outputs are excluded. The reader experiment is still being finalized; this checkout does not bundle incomplete results as final tables.

See [CODE_MAP.md](docs/CODE_MAP.md) and [THIRD_PARTY.md](THIRD_PARTY.md).

## License

CoRA-Bench's original code is released under the [Apache License 2.0](LICENSE). Code adapted from upstream benchmarks (`cora_bench/experiment/upstream/`, `cora_bench/experiment/audit_reference/`, and the BABILong generator and prompts in `cora_bench/data/`) keeps its original license. The NoLiMa files are for non-commercial research only. See [THIRD_PARTY.md](THIRD_PARTY.md).
