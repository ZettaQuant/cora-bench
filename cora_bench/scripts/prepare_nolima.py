"""Prepare the fixed-background NoLiMa diagnostic with occurrence-level evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def _fill(template: str, char: str, args: list[str]) -> str:
    s = template.replace("{CHAR}", char)
    for i, a in enumerate(args):  # {1},{2},{3} are 1-indexed in the needle set
        s = s.replace(f"{{{i + 1}}}", a)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--needles", required=True)
    ap.add_argument(
        "--haystacks", required=True, help="comma-separated haystack files (avg over background)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--haystack-tokens", type=int, default=130000)
    ap.add_argument("--depths", default="0.25,0.5,0.75")
    args = ap.parse_args()

    needles = json.load(open(args.needles))
    depths = [float(x) for x in args.depths.split(",")]
    hpaths = args.haystacks.split(",")
    # truncate each haystack to N cl100k tokens and split into sentences for insertion points
    haystacks, hhashes = [], {}
    for hp in hpaths:
        raw = ENC.decode(ENC.encode(open(hp).read(), disallowed_special=())[: args.haystack_tokens])
        haystacks.append(re.split(r"(?<=[.!?])\s+", raw))
        hhashes[Path(hp).name] = hashlib.sha256(raw.encode()).hexdigest()

    out = Path(args.out)
    (out / "ctx").mkdir(parents=True, exist_ok=True)
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    fh = (out / "input.jsonl").open("w")
    n = 0
    for hi, sents in enumerate(haystacks):
        for ni, nd in enumerate(needles):
            chars = nd["character_set"]
            for ti, (tid, t) in enumerate(nd["tests"].items()):
                argv = t["input_args"]
                char = chars[(ni + ti) % len(chars)]  # deterministic character assignment
                needle_txt = _fill(nd["needle"], char, argv)
                for hop, qtpl in nd["questions"].items():  # onehop, twohop
                    question = _fill(qtpl, char, argv)
                    for d in depths:
                        pos = max(1, int(len(sents) * d))
                        before = " ".join(sents[:pos])
                        ctx = before + " " + needle_txt + " " + " ".join(sents[pos:])
                        cs = len(before) + 1
                        assert ctx[cs : cs + len(needle_txt)] == needle_txt
                        eid = f"{nd['id']}_{tid}_{hop}_d{int(d * 100)}_h{hi + 1}"
                        (out / "ctx" / f"{eid}.txt").write_text(ctx)
                        (out / "evidence" / f"{eid}.json").write_text(
                            json.dumps(
                                {
                                    "example_id": eid,
                                    "needle": needle_txt,
                                    "char_start": cs,
                                    "char_end": cs + len(needle_txt),
                                    "depth": d,
                                    "answer_char": char,
                                }
                            )
                        )
                        fh.write(
                            json.dumps(
                                {
                                    "example_id": eid,
                                    "query": question,
                                    "answer": char,
                                    "reasoning_type": nd["reasoning_type"],
                                    "hop": hop,
                                    "depth": d,
                                    "needle_id": nd["id"],
                                    "test_id": tid,
                                    "haystack": hi + 1,
                                    "source_tokens": len(ENC.encode(ctx, disallowed_special=())),
                                }
                            )
                            + "\n"
                        )
                        n += 1
    fh.close()
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "source": "amodaresi/NoLiMa needle_set.json + haystack/rand_shuffle_long",
                "license": "Adobe Research (non-commercial research)",
                "note": "CUSTOM fixed-source selection diagnostic; NOT an official NoLiMa run "
                "(official: 26 placements, token-depth, base-length normalization). "
                "Primary unit = semantic test id (depths + haystacks are repeated measures).",
                "haystack_tokens": args.haystack_tokens,
                "depths": depths,
                "n_haystacks": len(haystacks),
                "n": n,
                "needleset_sha256": hashlib.sha256(Path(args.needles).read_bytes()).hexdigest(),
                "haystack_sha256": hhashes,
                "input_sha256": hashlib.sha256((out / "input.jsonl").read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )
    print(f"built {n} NoLiMa instances across {len(haystacks)} haystacks -> {out}")


if __name__ == "__main__":
    main()
