"""Reuse exact judge requests across logical answers and phases, without double billing."""

import json
import os
import threading
import time
from pathlib import Path

import requests

from .api import digest, save


class JudgeCache:
    def __init__(self, out, ledger):
        self.out = Path(out)
        self.ledger = ledger
        # Different logical keys can have the same judge body. Serialize those calls.
        self.locks = [threading.Lock() for _ in range(256)]

    def _read(self, path, request_hash, body):
        receipt = json.loads(path.read_text())
        if (
            receipt.get("request_sha256") != request_hash
            or receipt.get("judge_model") != body["model"]
        ):
            raise RuntimeError(f"Judge cache request mismatch: {path}")
        if not receipt.get("response", {}).get("usage"):
            raise RuntimeError(f"Judge cache missing usage: {path}")
        return receipt

    def evaluate(self, ds, key, body, dst, interpret):
        request_hash = digest(body)
        with self.locks[int(request_hash[:2], 16)]:
            return self._evaluate(ds, key, body, Path(dst), interpret, request_hash)

    def _evaluate(self, ds, key, body, dst, interpret, request_hash):
        ledger_key = "judge:" + request_hash
        raw_path = self.out / "judges" / "api_receipts" / (request_hash + ".json")
        with self.ledger.lock:
            record = self.ledger.db.execute(
                "SELECT state,cost,detail FROM spend WHERE key=?", (ledger_key,)
            ).fetchone()
        reused = True
        source = None
        if record and record[0] == "complete":
            detail = json.loads(record[2])
            source = self.out / detail["receipt"]
            receipt = self._read(source, request_hash, body)
            if abs(receipt["cost_usd"] - record[1]) > 1e-9:
                raise RuntimeError(f"Judge cache cost mismatch: {ledger_key}")
        elif raw_path.exists():
            # A previous run stopped after saving the response, before finishing the ledger entry.
            if not record:
                raise RuntimeError(f"Judge receipt has no reservation: {ledger_key}")
            source = raw_path
            receipt = self._read(source, request_hash, body)
            self.ledger.finish(
                ledger_key, receipt["cost_usd"], {"receipt": str(source.relative_to(self.out))}
            )
        elif record:
            # Older runs saved only the per-answer result before finish(); recover from that.
            for candidate in (self.out / "judges").glob("*/*.json"):
                value = json.loads(candidate.read_text())
                if value.get("request_sha256") == request_hash and value.get("response"):
                    source = candidate
                    receipt = self._read(source, request_hash, body)
                    break
            if source is None:
                raise RuntimeError(
                    f"Unresolved judge reservation; inspect before resubmitting: {ledger_key}"
                )
            self.ledger.finish(
                ledger_key, receipt["cost_usd"], {"receipt": str(source.relative_to(self.out))}
            )
        else:
            inp_rate, out_rate = (2.5, 10) if ds == "longmemeval" else (5, 30)
            reserve = (
                len(json.dumps(body, ensure_ascii=False).encode()) + 1024
            ) * inp_rate / 1e6 + body.get(
                "max_tokens", body.get("max_completion_tokens")
            ) * out_rate / 1e6
            self.ledger.reserve(
                ledger_key, "judges", reserve, {"dataset": ds, "request_sha256": request_hash}
            )
            start = time.perf_counter()
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
                json=body,
                timeout=180,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Judge HTTP {response.status_code}: {response.text[:500]}")
            response = response.json()
            usage = response["usage"]
            cost = (usage["prompt_tokens"] * inp_rate + usage["completion_tokens"] * out_rate) / 1e6
            receipt = {
                "request_sha256": request_hash,
                "judge_model": body["model"],
                "response": response,
                "cost_usd": cost,
                "latency_seconds": time.perf_counter() - start,
            }
            # Save the billed response and settle the ledger before parsing, so a malformed
            # judgment is never billed twice.
            save(raw_path, receipt)
            self.ledger.finish(ledger_key, cost, {"receipt": str(raw_path.relative_to(self.out))})
            source = raw_path
            reused = False
        text = receipt["response"]["choices"][0]["message"]["content"]
        result = {
            **receipt,
            **interpret(ds, text),
            "key": key,
            "api_receipt": str(source.relative_to(self.out)),
            "cache_reused": reused,
            "incremental_cost_usd": 0 if reused else receipt["cost_usd"],
        }
        save(dst, result)
        return result
