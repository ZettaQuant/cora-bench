# Data preparation and provenance

Raw source contexts are intentionally absent. Obtain datasets from their owners under their licenses; the source commits and file checksums are recorded in the bundled upstream manifests. The frozen reader experiment expects materialized data under `cora_bench/outputs/materialized/<dataset>/`.

Each dataset directory contains `input.jsonl` (one row per question), `ctx/<example_id>.txt` (exact UTF-8 source text), and dataset-specific `evidence/<example_id>.json` sidecars. Rows contain `example_id`, `query`, `answer`, and `source_tokens` counted with `cl100k_base`. Gold fields are used by scorers, never forwarded as reader context. Preserve source bytes and IDs; `audit_reference/sources.json` records the exact input-file hashes used in the experiment.

## BABILong

The frozen cohort contains qa1–qa5, 100 questions per task, generated at 256K GPT-2 tokens with generation seed **42** and **100** PG19 background books. This generation seed differs from the three reader seeds. Obtain the bAbI en-10k **training** task files (`qa1_*_train.txt` through `qa5_*_train.txt`). The generator streams the first 100 books from the `emozilla/pg19` test split; source-file checksums and recorded generation-package versions are in the data manifest. The attributed generator preserves injected support occurrences:

```bash
python -m cora_bench.scripts.prepare_babilong \
  --task-dir /path/to/babi/en-10k --tasks qa1,qa2,qa3,qa4,qa5 \
  --n 100 --length 256k --seed 42 --bg-books 100 \
  --out cora_bench/outputs/materialized/babilong
```

Validate generated files against `docs/data_manifests/babilong.json`. Regeneration requires matching upstream data/tokenizer versions; a new upstream revision is not an exact reproduction merely because it uses the same seed.

## LOONG

Obtain the pinned upstream LOONG question file (`loong.jsonl`) and financial documents under `doc/financial/`:

```bash
git clone https://github.com/MozerWang/Loong.git data/upstream/loong
git -C data/upstream/loong checkout 6d2115b8b48a3d19412ccb52a0c9c4ee37869af4
```

Use `data/upstream/loong` as `--raw-root`. The materializer selects English level 1 and keeps the listed document order and title delimiters:

```bash
python -m cora_bench.scripts.prepare_loong --raw-root /path/to/loong \
  --document-manifest docs/data_manifests/loong_documents.json
```

The reader protocol has 75 questions, with budget-eligible cohorts of 75/70/51. The pinned filename/hash manifest resolves duplicate filename matches; missing or modified files are errors. Without a manifest, ambiguous matches are rejected. The evidence adapter uses automatic candidate-line retention unless reviewed source spans are explicitly supplied. Automatic candidate lines are not gold chunk labels. The original reference answers are retained even where units are ambiguous.

## NoLiMa diagnostic

Use the bundled needle template and five ordered upstream `haystack/rand_shuffle_long/long_*.txt` files:

```bash
git clone https://github.com/adobe-research/NoLiMa.git data/upstream/nolima
git -C data/upstream/nolima checkout cb14780b249fecf2851127b2101a062c1b2c6430
```

Pass `long_1.txt` through `long_5.txt` in that order. This is a custom diagnostic, not the official NoLiMa evaluation:

```bash
python -m cora_bench.scripts.prepare_nolima \
  --needles cora_bench/experiment/upstream/nolima/needle_set.json \
  --haystacks /path/long_1.txt,/path/long_2.txt,/path/long_3.txt,/path/long_4.txt,/path/long_5.txt \
  --haystack-tokens 130000 --depths 0.25,0.5,0.75 \
  --out cora_bench/outputs/materialized/nolima
```

The expected 870 observations repeat 58 question/needle variants over three insertion depths and five backgrounds. Bootstrap groups are ten needle families. Compare hashes with `docs/data_manifests/nolima.json`; respect the non-commercial research license.

## LongMemEval

Obtain `longmemeval_s_cleaned.json` from `xiaowu0162/longmemeval-cleaned`:

```bash
python -m cora_bench.scripts.prepare_longmemeval \
  --in /path/to/longmemeval_s_cleaned.json \
  --out cora_bench/outputs/materialized/longmemeval
```

The 500-question cohort includes 30 abstentions. The materializer replaces source session identifiers with neutral sequential IDs so `answer_*` names do not expose evidence to the reader. Evidence sidecars retain answer-bearing turn spans. Compare the generated manifest and input hash with the frozen provenance before reporting a reproduction.
