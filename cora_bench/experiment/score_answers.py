"""Score reader answers with each dataset's native metric or upstream judge rubric; judge prompts
never include the selection method."""

import argparse
import ast
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cora_bench.data.babilong import TASK_LABELS, _official_compare

from .api import OUT, PROTOCOL, ROOT, Ledger, digest, save
from .judge_cache import JudgeCache

HERE = Path(__file__).parent


def function_only(path, name):
    tree = ast.parse(path.read_text())
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    ns = {"re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
    return ns[name]


LME_PROMPT = function_only(
    HERE / "upstream/longmemeval/src__evaluation__evaluate_qa.py", "get_anscheck_prompt"
)
LOONG_EXTRACT_NUMBER = function_only(
    HERE / "upstream/loong/src__utils__metric.py", "extract_number"
)
_loong = ast.parse((HERE / "upstream/loong/src__utils__prompt.py").read_text())
LOONG_PROMPT = ast.literal_eval(
    next(
        n.value
        for f in _loong.body
        if isinstance(f, ast.FunctionDef) and f.name == "get_evaluate_prompts"
        for n in f.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "prompt" for t in n.targets)
    )
)
LME_MODEL = "gpt-4o-2024-08-06"
LOONG_MODEL = "gpt-5.5-2026-04-23"


class MalformedJudgment(AssertionError):
    """Unparseable judge output; recorded as unresolved instead of aborting the run."""


def direct_score(ds, row, prediction):
    if ds == "babilong":
        ok = _official_compare(
            str(row["answer"]), prediction, row["query"], TASK_LABELS[row["task"]]
        )
        return {"correct": bool(ok), "score": float(ok), "metric": "babilong_official_accuracy"}
    if ds == "nolima":
        ok = str(row["answer"]) in prediction
        return {
            "correct": ok,
            "score": float(ok),
            "metric": "nolima_official_contains_on_frozen_diagnostic",
        }
    raise ValueError(ds)


def judge_request(ds, row, prediction):
    if ds == "longmemeval":
        text = LME_PROMPT(
            row["question_type"],
            row["query"],
            str(row["answer"]),
            prediction,
            abstention=bool(row["abstention"]),
        )
        return {
            "model": LME_MODEL,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "max_tokens": 10,
        }
    if ds == "loong":
        q = "\n\n" + row.get("instruction", "") + "\n\n" + row["query"]
        return {
            "model": LOONG_MODEL,
            "messages": [
                {"role": "user", "content": LOONG_PROMPT.format(q, row["answer"], prediction)}
            ],
            "reasoning_effort": "none",
            "max_completion_tokens": 512,
        }
    raise ValueError(ds)


def interpret(ds, text):
    if ds == "longmemeval":
        if not isinstance(text, str) or not re.fullmatch(r"(yes|no)[.!]?", text.strip(), re.I):  # noqa: FURB167
            raise MalformedJudgment("Malformed LongMemEval judgment")
        ok = "yes" in text.lower()
        return {
            "correct": ok,
            "score": float(ok),
            "metric": "longmemeval_official_gpt4o_question_type_accuracy",
        }
    # The upstream parser accepts any bracketed number, including 0. Keep its value and flag
    # scores outside the rubric's 1-100 range instead of clipping or re-judging.
    n = LOONG_EXTRACT_NUMBER(text) if isinstance(text, str) else None
    if n is None:
        raise MalformedJudgment("Malformed LOONG judgment")
    return {
        "correct": n == 100,
        "score": n,
        "metric": "loong_native_rubric_adapted_gpt55",
        "perfect": n == 100,
        "judge_out_of_range": not 1 <= n <= 100,
    }


def score_one(cache, ds, key, body, dst):
    unresolved = dst.parent / "unresolved" / (key + ".json")
    try:
        result = cache.evaluate(ds, key, body, dst, interpret)
    except MalformedJudgment as error:
        save(
            unresolved,
            {
                "key": key,
                "dataset": ds,
                "request_sha256": digest(body),
                "judge_model": body["model"],
                "error": str(error),
                "status": "unresolved",
                "updated_unix": time.time(),
            },
        )
        return None
    unresolved.unlink(missing_ok=True)
    return result


def run(phase="main"):
    dest = OUT / "judges" / phase
    dest.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(OUT / "budget.sqlite")
    cache = JudgeCache(OUT, ledger)
    rows = {
        ds: {
            r["example_id"]: r
            for l in (ROOT / "cora_bench/outputs/materialized" / ds / "input.jsonl").open()
            if (r := json.loads(l))
        }
        for ds in PROTOCOL["datasets"]
    }
    pending = {}
    for p in (OUT / "reader" / phase / "links").glob("*/*/*.json"):
        link = json.loads(p.read_text())
        ds = link["dataset"]
        row = rows[ds][link["example_id"]]
        receipt = json.loads((OUT / link["receipt"]).read_text())
        key = digest([ds, row["example_id"], receipt["raw_output"]])
        dst = dest / (key + ".json")
        if dst.exists():
            continue
        if not receipt["raw_output"] or receipt["finish_reason"] not in ["STOP", "MAX_TOKENS"]:
            save(
                dst,
                {
                    "correct": False,
                    "score": 0,
                    "metric": "reader_failure",
                    "reader_finish": receipt["finish_reason"],
                    "key": key,
                },
            )
            continue
        if ds in ["babilong", "nolima"]:
            save(dst, {**direct_score(ds, row, receipt["raw_output"]), "key": key})
            continue
        pending[key] = (ds, row, receipt["raw_output"], dst)

    def work(item):
        key, (ds, row, pred, dst) = item
        return score_one(cache, ds, key, judge_request(ds, row, pred), dst)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(work, pending.items()):
            pass
    save(
        dest / "status.json",
        {
            "pending_submitted": len(pending),
            "unresolved_unique_judgments": sum(1 for _ in (dest / "unresolved").glob("*.json")),
            "reserved_or_spent_usd": ledger.totals(),
            "updated_unix": time.time(),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="main")
    a = p.parse_args()
    run(a.phase)
