"""Generate reader answers from saved selection text at each seed, under a spend cap.

Requests interrupted by a crash keep their reservation and are reported rather than resubmitted.
"""

import argparse
import gzip
import json
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cora_bench.budgets.token_budget import Tokenizer

from .api import OUT, PROTOCOL, ROOT, Ledger, Vertex, digest, save, usage_cost
from .prompts import parse, request
from .retry import MAX_ATTEMPTS, SharedBackoff, transient
from .stream import AuditedVertex, IncompleteStream


class ReaderRun:
    def __init__(self, phase="main"):
        self.phase = phase
        self.cloud = False
        self.api = AuditedVertex()
        self.ledger = Ledger(OUT / "budget.sqlite")
        self.tok = Tokenizer()
        self.backoff = SharedBackoff()
        self.root = OUT / "reader" / phase
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "jobs.sqlite", check_same_thread=False)
        self.lock = threading.RLock()
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS jobs (key TEXT PRIMARY KEY, state TEXT, receipt TEXT, error TEXT)"
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS counts (key TEXT PRIMARY KEY, tokens INTEGER)")
        self.db.commit()
        self.manifest = {
            "protocol_sha256": digest(PROTOCOL),
            "phase": phase,
            "model": PROTOCOL["models"]["reader"],
            "code": {
                p.name: digest(p.read_bytes())
                for p in [
                    Path(__file__),
                    Path(__file__).with_name("prompts.py"),
                    Path(__file__).with_name("api.py"),
                ]
            },
            "upstream": {
                str(p.relative_to(Path(__file__).parent)): digest(p.read_bytes())
                for p in (Path(__file__).parent / "upstream").rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            },
        }
        self.manifest["code"]["retry.py"] = digest(
            Path(__file__).with_name("retry.py").read_bytes()
        )
        self.manifest["code"]["stream.py"] = digest(
            Path(__file__).with_name("stream.py").read_bytes()
        )
        mp = self.root / "manifest.json"
        if mp.exists():
            assert json.loads(mp.read_text()) == self.manifest, (
                "Reader code/protocol changed; use a new run directory"
            )
        else:
            save(mp, self.manifest)

    def done(self, key):
        with self.lock:
            return self.db.execute("SELECT state FROM jobs WHERE key=?", (key,)).fetchone()

    def run_one(self, ds, row, method, budget, context, seed, selection_ref=None):
        payload = request(ds, row, context, seed)
        key = digest(
            {"phase": self.phase, "model": PROTOCOL["models"]["reader"], "request": payload}
        )
        refkey = digest([ds, row["example_id"], method, budget, seed])
        link = self.root / "links" / ds / method / (refkey + ".json")
        if link.exists():
            return "existing"
        receipt = self.root / "receipts" / key[:2] / (key + ".json")
        # Identical requests (e.g. two methods emitting the same context) are paid for once.
        with self.lock:
            status = self.done(key)
            if status and status[0] != "complete":
                return "pending_or_error"
            if not status:
                self.db.execute(
                    "INSERT INTO jobs VALUES (?,?,?,NULL)",
                    (key, "inflight", str(receipt.relative_to(OUT))),
                )
                self.db.commit()
        if not status:
            try:
                countkey = digest({k: v for k, v in payload.items() if k != "generationConfig"})
                with self.lock:
                    ct = self.db.execute(
                        "SELECT tokens FROM counts WHERE key=?", (countkey,)
                    ).fetchone()
                count_attempts = []
                if ct:
                    native = ct[0]
                else:
                    for attempt in range(MAX_ATTEMPTS):
                        self.backoff.wait()
                        try:
                            native = self.api.count(payload)
                            break
                        except Exception as exc:
                            count_attempts.append({"attempt": attempt, "error": str(exc)[:500]})
                            save(receipt.with_suffix(".count_attempts.json"), count_attempts)
                            if attempt + 1 == MAX_ATTEMPTS or not transient(exc):
                                raise
                            self.backoff.defer(attempt)
                assert native + payload["generationConfig"]["maxOutputTokens"] <= 1048576, (
                    "Native context window exceeded"
                )
                with self.lock:
                    self.db.execute("INSERT OR IGNORE INTO counts VALUES (?,?)", (countkey, native))
                    self.db.commit()
                reservation = (native * 0.1 + 1024 * 0.4) / 1e6 * 1.01
                category = "reader_online" if self.phase == "main" else "online_smoke_latency"
                attempts = []
                request_started = time.perf_counter()
                for attempt in range(MAX_ATTEMPTS):
                    self.backoff.wait()
                    ledger_key = "reader:" + key + ":" + str(attempt)
                    self.ledger.reserve(
                        ledger_key,
                        category,
                        reservation,
                        {"native_input_estimate": native, "dataset": ds, "seed": seed},
                    )
                    try:
                        before_success = time.perf_counter() - request_started
                        response, timing = self.api.stream(payload)
                        cost = usage_cost(response)
                        timing["successful_attempt_reader_seconds"] = timing[
                            "reader_latency_seconds"
                        ]
                        timing["successful_attempt_ttft_seconds"] = timing[
                            "time_to_first_token_seconds"
                        ]
                        timing["reader_latency_seconds"] = time.perf_counter() - request_started
                        if timing["time_to_first_token_seconds"] is not None:
                            timing["time_to_first_token_seconds"] += before_success
                        break
                    except Exception as exc:
                        failure = {"attempt": attempt, "error": str(exc)[:500]}
                        if isinstance(exc, IncompleteStream):
                            failure.update(response=exc.response, events=exc.events)
                        attempts.append(failure)
                        save(receipt.with_suffix(".attempts.json"), attempts)
                        # A failed attempt's reservation stays counted; retry transient faults only.
                        if attempt + 1 == MAX_ATTEMPTS or not (
                            transient(exc) or isinstance(exc, IncompleteStream)
                        ):
                            raise
                        self.backoff.defer(attempt)

                raw, answer, finish = parse(response)
                out = {
                    "key": key,
                    "request_sha256": digest(payload),
                    "model": PROTOCOL["models"]["reader"],
                    "model_version": response.get("modelVersion"),
                    "seed": seed,
                    "native_count_estimate": native,
                    "raw_output": raw,
                    "answer": answer,
                    "finish_reason": finish,
                    "response": response,
                    "cost_usd": cost,
                    **timing,
                    "finished_unix": time.time(),
                    "previous_attempt_errors": attempts,
                }
                save(receipt, out)
                self.ledger.finish(ledger_key, cost, {"receipt": str(receipt.relative_to(OUT))})
                with self.lock:
                    self.db.execute("UPDATE jobs SET state=? WHERE key=?", ("complete", key))
                    self.db.commit()
                assert cost <= reservation, "Native billing exceeded reservation; stop and inspect"
            except Exception as e:
                with self.lock:
                    self.db.execute(
                        "UPDATE jobs SET state=?,error=? WHERE key=?", ("error", str(e)[:500], key)
                    )
                    self.db.commit()
                raise
        save(
            link,
            {
                "dataset": ds,
                "example_id": row["example_id"],
                "method": method,
                "budget": budget,
                "seed": seed,
                "receipt": str(receipt.relative_to(OUT)),
                "context_sha256": digest(context.encode()),
                "selected_tokens_cl100k": self.tok.count(context),
                "source_tokens_cl100k": row["source_tokens"],
                "selection_ref": selection_ref,
            },
        )
        return "complete"

    def tasks(self, limit=None):
        tasks = []
        from .select_context import controls

        for ds in PROTOCOL["datasets"]:
            rows = [
                json.loads(l)
                for l in (ROOT / "cora_bench/outputs/materialized" / ds / "input.jsonl").open()
            ]
            if self.phase == "latency":
                rows = [r for r in rows if r["example_id"] in controls(rows, ds)]
            seeds = PROTOCOL["seeds"] if self.phase != "latency" else [PROTOCOL["seeds"][0]]
            for row in rows[:limit]:
                for method in PROTOCOL["controls"]:
                    context = (
                        (
                            ROOT
                            / "cora_bench/outputs/materialized"
                            / ds
                            / "ctx"
                            / (row["example_id"] + ".txt")
                        )
                        .read_bytes()
                        .decode("utf8")
                        if method == "full_context"
                        else ""
                    )
                    for seed in seeds:
                        rk = digest([ds, row["example_id"], method, None, seed])
                        if not (self.root / "links" / ds / method / (rk + ".json")).exists():
                            tasks.append((ds, row, method, None, context, seed, None))
            for kind in ["cpu", "gpu"]:
                for p in sorted((OUT / "selection" / kind / ds).glob("*/*.json.gz")):
                    with gzip.open(p, "rt") as f:
                        r = json.load(f)
                    if (limit or self.phase == "latency") and r["example_id"] not in {
                        x["example_id"] for x in rows[:limit]
                    }:
                        continue
                    for s in r["selections"]:
                        assert digest(s["filtered_text"].encode()) == s["text_sha256"]
                        assert s["selected_tokens"] <= s["budget"]
                        for seed in seeds:
                            rk = digest([ds, r["example_id"], r["method"], s["budget"], seed])
                            if not (
                                self.root / "links" / ds / r["method"] / (rk + ".json")
                            ).exists():
                                tasks.append(
                                    (
                                        ds,
                                        r["row"],
                                        r["method"],
                                        s["budget"],
                                        s["filtered_text"],
                                        seed,
                                        str(p.relative_to(OUT)),
                                    )
                                )
        # Hash order spreads spend across datasets and seeds in case the cap is reached.
        tasks.sort(key=lambda x: digest([x[0], x[1]["example_id"], x[2], x[3], x[5]]))
        return tasks

    def batch(self, limit=None):
        tasks = self.tasks(limit)
        counts = {
            "planned": len(tasks),
            "complete": 0,
            "errors": 0,
            "existing": 0,
            "pending_or_error": 0,
        }
        # Submit in bounded waves so a cap or provider failure stops later requests.
        with ThreadPoolExecutor(max_workers=8) as pool:
            for offset in range(0, len(tasks), 64):
                futures = [pool.submit(self.run_one, *t) for t in tasks[offset : offset + 64]]
                errors = []
                for f in as_completed(futures):
                    try:
                        counts[f.result()] += 1
                    except Exception as e:
                        counts["errors"] += 1
                        errors.append(str(e))
                save(
                    self.root / "status.json",
                    {
                        **counts,
                        "ledger_reserved_or_actual_usd": self.ledger.totals(),
                        "updated_unix": time.time(),
                        "last_errors": errors[:3],
                    },
                )
                print(json.dumps(counts), flush=True)
                if self.cloud:
                    subprocess.run(
                        [
                            "gcloud",
                            "storage",
                            "rsync",
                            str(self.root),
                            PROTOCOL["gcs_prefix"] + "/reader/" + self.phase,
                            "--recursive",
                            "--exclude",
                            ".*sqlite.*",
                            "--quiet",
                        ],
                        check=True,
                    )
                if errors:
                    raise RuntimeError(
                        "Reader wave failed; receipts/reservations retained: " + errors[0]
                    )
        return counts


def sync_cloud():
    check = subprocess.run(
        ["gcloud", "storage", "ls", PROTOCOL["gcs_prefix"] + "/selection/gpu/manifest.json"],
        capture_output=True,
        text=True,
    )
    if check.returncode:
        if "matched no objects" in check.stderr.lower() or "not found" in check.stderr.lower():
            return
        raise RuntimeError(check.stderr[:400])
    subprocess.run(
        [
            "gcloud",
            "storage",
            "rsync",
            PROTOCOL["gcs_prefix"] + "/selection/gpu",
            str(OUT / "selection/gpu"),
            "--recursive",
            "--quiet",
        ],
        check=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="main")
    p.add_argument("--limit", type=int)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--cloud", action="store_true")
    a = p.parse_args()
    run = ReaderRun(a.phase)
    run.cloud = a.cloud
    while True:
        if a.cloud:
            sync_cloud()
        run.batch(a.limit)
        if a.cloud:
            subprocess.run(
                [
                    "gcloud",
                    "storage",
                    "rsync",
                    str(run.root),
                    PROTOCOL["gcs_prefix"] + "/reader/" + a.phase,
                    "--recursive",
                    "--exclude",
                    ".*sqlite.*",
                    "--quiet",
                ],
                check=True,
            )
        if not a.watch:
            break
        complete = all((OUT / "selection" / k / "complete.json").exists() for k in ["cpu", "gpu"])
        if complete and not run.tasks(a.limit):
            save(run.root / "complete.json", {"complete": True, "finished_unix": time.time()})
            break
        time.sleep(45)


if __name__ == "__main__":
    main()
