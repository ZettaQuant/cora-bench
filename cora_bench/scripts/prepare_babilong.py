"""Generate the frozen BABILong cohort with supporting-fact spans and data hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BABILONG_COMMIT = "7a6efee29f5cac03c3c410e6799c80fd2ffe3610"
TASK_FILES = {
    "qa1": "qa1_single-supporting-fact",
    "qa2": "qa2_two-supporting-facts",
    "qa3": "qa3_three-supporting-facts",
    "qa4": "qa4_two-arg-relations",
    "qa5": "qa5_three-arg-relations",
}
# Upstream message lengths in gpt2 tokens; 300 are reserved for the prompt, as in create_tasks.py.
LENGTHS = {
    "8k": 8000,
    "16k": 16000,
    "32k": 32000,
    "64k": 64000,
    "128k": 128000,
    "256k": 256000,
    "512k": 512000,
}


def _char_spans(tokenizer, tokens, spans_tok):
    """Map fact token spans to (char_start, char_end, stripped_text) without string search."""
    order = sorted(range(len(spans_tok)), key=lambda i: spans_tok[i][0])
    out: list = [None] * len(spans_tok)
    cur_tok, cur_char = 0, 0
    for i in order:
        s, e = spans_tok[i]
        cur_char += len(tokenizer.decode(tokens[cur_tok:s]))
        frag = tokenizer.decode(tokens[s:e])
        lead = len(frag) - len(frag.lstrip())
        cs = cur_char + lead
        text = frag.strip()
        cur_char += len(frag)
        cur_tok = e
        out[i] = (cs, cs + len(text), text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--tasks", default="qa1,qa2,qa3,qa4,qa5")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--length", default="256k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)  # cap samples per task
    ap.add_argument(
        "--bg-books", type=int, default=300
    )  # PG19 books streamed for background text
    args = ap.parse_args()

    import tiktoken
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from cora_bench.data._babilong_gen import (
        InstrumentedNoiseInjectionDataset,
        SentenceSampler,
        TaskDataset,
    )

    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    cl100k = tiktoken.get_encoding("cl100k_base")
    msg_len = LENGTHS[args.length] - 300
    n = args.limit or args.n

    print(f"loading PG19 background ({args.bg_books} books, streaming)...", flush=True)
    stream = load_dataset("emozilla/pg19", split="test", streaming=True)
    pg19 = []  # SentenceSampler needs an indexable list of {"text": ...}
    for i, ex in enumerate(stream):
        if i >= args.bg_books:
            break
        pg19.append({"text": ex["text"]})
    print(f"  loaded {len(pg19)} background books", flush=True)

    out = Path(args.out)
    (out / "ctx").mkdir(parents=True, exist_ok=True)
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    input_fh = (out / "input.jsonl").open("w")
    import importlib.metadata as md

    def _ver(p):
        try:
            return md.version(p)
        except Exception:
            return None

    manifest = {
        "babilong_commit": BABILONG_COMMIT,
        "background": "emozilla/pg19:test",
        "bg_books": len(pg19),
        "tokenizer_gen": "gpt2",
        "tokenizer_budget": "cl100k_base",
        "length_config": args.length,
        "gen_message_tokens_gpt2": msg_len,
        "n_per_task": n,
        "seed": args.seed,
        "tasks": {},
        "identity_ok": True,
        "babi_file_sha256": {},
        "min_source_tokens_cl100k": 128001,
        "packages": {p: _ver(p) for p in ["datasets", "transformers", "tiktoken", "torch", "nltk"]},
        "pg19_pin_note": "emozilla/pg19 test split, first bg_books in stream order (revision not hash-pinned)",
    }

    for task in args.tasks.split(","):
        tp = Path(args.task_dir) / f"{TASK_FILES[task]}_train.txt"
        # deterministic, distinct seed per task
        base = args.seed + sum(ord(c) for c in task)
        sampler = SentenceSampler(pg19, tokenizer=gpt2, shuffle=True, random_seed=base)
        task_ds = TaskDataset(str(tp), max_n_facts=msg_len // 8)
        inj = InstrumentedNoiseInjectionDataset(
            task_ds, sampler, gpt2, sample_size=msg_len, random_seed=base + 1
        )
        manifest["babi_file_sha256"][task] = hashlib.sha256(tp.read_bytes()).hexdigest()
        cand = np.random.default_rng(base + 2).permutation(len(inj)).tolist()  # candidate order

        src_cl100k, kept, used_inds = [], 0, []
        for ind in cand:
            if kept >= n:
                break
            sample = inj[int(ind)]
            tokens = sample["input_tokens"]
            full_text = gpt2.decode(tokens)
            ck = len(cl100k.encode(full_text))
            if (
                ck <= 128000
            ):  # every kept example must exceed 128K cl100k tokens
                continue
            ex_id = f"{task}_{args.length}_{kept:03d}"
            support = set(sample["support_idx"])
            phrase_nums = sample["phrase_nums"]
            spans = _char_spans(gpt2, tokens, sample["fact_token_spans"])
            ev_facts, all_found = [], True
            for fi, (cs, ce, frag) in enumerate(spans):
                if fi not in support:
                    continue
                found = 0 <= cs and full_text[cs:ce] == frag
                all_found = all_found and found
                ev_facts.append(
                    {
                        "fact_id": int(phrase_nums[fi]),
                        "local_idx": int(fi),
                        "text": frag,
                        "char_start": int(cs),
                        "char_end": int(ce),
                        "token_start": int(sample["fact_token_spans"][fi][0]),
                        "token_end": int(sample["fact_token_spans"][fi][1]),
                    }
                )
            if not all_found:  # decoded span does not match the fact text; skip
                manifest["identity_ok"] = False
                print(f"  WARN skip {task} babi_idx={ind}: support span mismatch", flush=True)
                continue
            (out / "ctx" / f"{ex_id}.txt").write_text(full_text)
            (out / "evidence" / f"{ex_id}.json").write_text(
                json.dumps(
                    {
                        "example_id": ex_id,
                        "task": task,
                        "babi_sample_index": int(ind),
                        "reference_nums": sample["reference_nums"],
                        "support_facts": ev_facts,
                        "n_support": len(ev_facts),
                        "fact_positions": sample["fact_positions"],
                    }
                )
            )
            src_cl100k.append(ck)
            used_inds.append(int(ind))
            input_fh.write(
                json.dumps(
                    {
                        "example_id": ex_id,
                        "task": task,
                        "query": str(sample["question"]),
                        "answer": str(sample["answer"]),
                        "source_tokens": ck,
                        "source_tokens_gpt2": len(tokens),
                        "n_support": len(ev_facts),
                        "babi_sample_index": int(ind),
                    }
                )
                + "\n"
            )
            kept += 1
            if kept % 20 == 0:
                print(f"  {task}: {kept}/{n}", flush=True)
        if kept < n:
            print(f"  WARN {task}: only {kept}/{n} exceeded 128K (increase candidates)", flush=True)
        manifest["tasks"][task] = {
            "kept": kept,
            "selected_babi_indices": used_inds,
            "cl100k_min": min(src_cl100k) if src_cl100k else 0,
            "cl100k_max": max(src_cl100k) if src_cl100k else 0,
            "cl100k_mean": int(np.mean(src_cl100k)) if src_cl100k else 0,
            "below_128k": int(sum(t <= 128000 for t in src_cl100k)),
        }
        print(
            f"{task}: kept {kept}, cl100k [{manifest['tasks'][task]['cl100k_min']}.."
            f"{manifest['tasks'][task]['cl100k_max']}], <=128k: "
            f"{manifest['tasks'][task]['below_128k']}",
            flush=True,
        )

    input_fh.close()
    manifest["input_jsonl_sha256"] = hashlib.sha256((out / "input.jsonl").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ndone -> {out}  identity_ok={manifest['identity_ok']}", flush=True)


if __name__ == "__main__":
    main()
