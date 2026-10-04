"""Reader prompts built from each benchmark's upstream templates.

Gold answers and method names never enter a prompt.
"""

import json
import re
from pathlib import Path

from cora_bench.data.babilong_prompts import BABILONG_PROMPTS

HERE = Path(__file__).parent
LME_TEMPLATE = "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):"


def request(dataset, row, context, seed):
    system = None
    if dataset == "babilong":
        p = BABILONG_PROMPTS[row["task"]]
        system = p["instruction"] + "\n\n" + p["examples"]
        text = f"<context>\n{context}\n</context>\n\n{row['query']}\n\n{p['post_prompt']}"
    elif dataset == "loong":
        # All 75 LOONG rows share this template (see loong_templates.json).
        text = (
            "{docs}\n\n{instruction}\n\n{question}".replace("{question}", row["query"])
            .replace("{instruction}", row.get("instruction", ""))
            .replace("{docs}", context)
        )
    elif dataset == "longmemeval":
        text = LME_TEMPLATE.format(context, row["question_date"], row["query"])
    elif dataset == "nolima":
        templates = json.loads((HERE / "upstream/nolima/needle_set.json").read_text())
        nd = next(x for x in templates if x["id"] == row["needle_id"])
        text = (
            nd["task_template"].replace("{haystack}", context).replace("{question}", row["query"])
        )
        system = "You are a helpful assistant"  # NoLiMa's default system prompt
    else:
        raise ValueError(dataset)
    body = {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 1024,
            "seed": seed,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


def parse(response):
    cs = response.get("candidates") or []
    text = "".join(
        p.get("text", "")
        for p in (cs[0].get("content", {}).get("parts", []) if cs else [])
        if not p.get("thought")
    )
    return text, text.strip(), cs[0].get("finishReason") if cs else None
