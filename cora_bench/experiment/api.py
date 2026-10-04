"""Vertex AI REST client, request hashing, and a SQLite ledger of spend reservations."""

import hashlib
import json
import os
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import requests

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
API = "https://aiplatform.googleapis.com/v1"
MODEL = "gemini-2.5-flash-lite"
ROOT = Path(os.environ.get("CORA_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL = json.loads(
    Path(os.environ.get("CORA_PROTOCOL", Path(__file__).with_name("protocol.json"))).read_text()
)
OUT = ROOT / PROTOCOL["local_outputs"]
PROTOCOL["gcs_prefix"] = os.environ.get("CORA_GCS_PREFIX", PROTOCOL.get("gcs_prefix", ""))


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


class Vertex:
    def __init__(self):
        self.local = threading.local()

    def session(self):
        if not PROJECT:
            raise ValueError("Set GOOGLE_CLOUD_PROJECT before using the reader API.")
        if not getattr(self.local, "session", None):
            import google.auth
            from google.auth.transport.requests import AuthorizedSession

            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self.local.session = AuthorizedSession(creds)
        return self.local.session

    def call(self, path, body=None, method="POST", timeout=180):
        r = self.session().request(method, API + "/" + path, json=body, timeout=timeout)
        if r.status_code >= 400:
            # Report the provider's error body only, never request headers or credentials.
            raise RuntimeError(f"Vertex HTTP {r.status_code}: {r.text[:1200]}")
        return r.json()

    def model(self, action, body):
        return self.call(
            f"projects/{PROJECT}/locations/global/publishers/google/models/{MODEL}:{action}", body
        )

    def count(self, request):
        body = {k: v for k, v in request.items() if k in ("contents", "systemInstruction")}
        return int(self.model("countTokens", body)["totalTokens"])


class Ledger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60, check_same_thread=False)
        self.lock = threading.RLock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS spend (key TEXT PRIMARY KEY, category TEXT, reserved REAL, cost REAL, state TEXT, detail TEXT)"
        )
        self.db.commit()

    def reserve(self, key, category, amount, detail=None):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                row = self.db.execute("SELECT state FROM spend WHERE key=?", (key,)).fetchone()
                if row:
                    raise RuntimeError(
                        f"Existing {row[0]} reservation: {key}; inspect before resubmitting"
                    )
                total = self.db.execute(
                    "SELECT COALESCE(SUM(COALESCE(cost,reserved)),0) FROM spend WHERE category=?",
                    (category,),
                ).fetchone()[0]
                if amount < 0 or total + amount > PROTOCOL["budget_usd"][category]:
                    raise RuntimeError(f"{category} budget exceeded")
                self.db.execute(
                    "INSERT INTO spend VALUES (?,?,?,NULL,?,?)",
                    (key, category, amount, "reserved", json.dumps(detail or {})),
                )
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def finish(self, key, cost, detail):
        with self.lock:
            self.db.execute(
                "UPDATE spend SET cost=?,state=?,detail=? WHERE key=?",
                (cost, "complete", json.dumps(detail), key),
            )
            self.db.commit()

    def totals(self):
        with self.lock:
            return dict(
                self.db.execute(
                    "SELECT category,SUM(COALESCE(cost,reserved)) FROM spend GROUP BY category"
                )
            )


def usage_cost(response, batch=False):
    u = response.get("usageMetadata", {})
    factor = 0.5 if batch else 1
    return (
        (
            int(u.get("promptTokenCount", 0)) * 0.1
            + (int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0))) * 0.4
        )
        / 1e6
        * factor
    )
