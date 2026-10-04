"""Prepare LongMemEval histories with neutral session IDs and evidence sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def _toks(s: str) -> int:
    return len(ENC.encode(s, disallowed_special=()))


def build(ex: dict):
    """Return (context_with_neutral_ids, answer_turns_with_spans, answer_session_neutral_idxs)."""
    ans_sids = set(ex.get("answer_session_ids") or [])
    buf, off = [], 0
    answer_turns, answer_sessions = [], []
    for si, (sid, date, sess) in enumerate(
        zip(
            ex["haystack_session_ids"],
            ex.get("haystack_dates") or [""] * len(ex["haystack_session_ids"]),
            ex["haystack_sessions"],
        )
    ):
        neutral = si + 1
        if sid in ans_sids:
            answer_sessions.append(neutral)
        header = f"[Session {neutral} | {date}]"  # raw session ids can reveal answer sessions
        buf.append(header)
        off += len(header) + 1
        for t in sess:
            line = f"{t['role']}: {t['content']}"
            cstart = off + len(t["role"]) + 2  # after "role: "
            buf.append(line)
            off += len(line) + 1
            if t.get("has_answer"):
                answer_turns.append(
                    {
                        "session_neutral": neutral,
                        "role": t["role"],
                        "text": t["content"],
                        "char_start": cstart,
                        "char_end": cstart + len(t["content"]),
                    }
                )
    import bisect

    ctx = "\n".join(buf)
    ids = ENC.encode(ctx, disallowed_special=())
    _, offs = ENC.decode_with_offsets(ids)  # char offsets of each token in one full encoding
    for (
        at
    ) in answer_turns:  # verify text and add token spans
        assert ctx[at["char_start"] : at["char_end"]] == at["text"], ex["question_id"]
        at["token_start"] = max(0, bisect.bisect_right(offs, at["char_start"]) - 1)
        at["token_end"] = (
            max(at["token_start"], bisect.bisect_right(offs, at["char_end"] - 1) - 1) + 1
        )
    return ctx, answer_turns, sorted(set(answer_sessions))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data = json.load(open(args.inp))
    if args.limit:
        data = data[: args.limit]
    out = Path(args.out)
    (out / "ctx").mkdir(parents=True, exist_ok=True)
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    fh = (out / "input.jsonl").open("w")
    n = 0
    for ex in data:
        eid = ex["question_id"]
        abstention = eid.endswith("_abs")  # official abstention marker
        ctx, turns, ans_sess = build(ex)
        (out / "ctx" / f"{eid}.txt").write_text(ctx)
        (out / "evidence" / f"{eid}.json").write_text(
            json.dumps(
                {
                    "example_id": eid,
                    "question_type": ex["question_type"],
                    "abstention": abstention,
                    "answer_session_neutral": ans_sess,
                    "answer_turns": turns,
                    "n_turns": len(turns),
                }
            )
        )
        fh.write(
            json.dumps(
                {
                    "example_id": eid,
                    "query": ex["question"],
                    "answer": str(ex["answer"]),
                    "question_type": ex["question_type"],
                    "question_date": ex.get("question_date"),
                    "abstention": abstention,
                    "n_evidence_turns": len(turns),
                    "n_answer_sessions": len(ans_sess),
                    "source_tokens": _toks(ctx),
                }
            )
            + "\n"
        )
        n += 1
    fh.close()
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "source": "xiaowu0162/longmemeval-cleaned:longmemeval_s_cleaned.json",
                "n": n,
                "tokenizer": "cl100k_base",
                "visible_session_ids": "neutral sequential (no answer_* leak)",
                "abstention": "official _abs question-id suffix",
                "input_sha256": hashlib.sha256((out / "input.jsonl").read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )
    print(
        f"materialized {n} LongMemEval-S (neutral ids, {sum(1 for e in data if e['question_id'].endswith('_abs'))} abstention) -> {out}"
    )


if __name__ == "__main__":
    main()
