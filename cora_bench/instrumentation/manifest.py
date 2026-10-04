"""Run manifest: everything needed to reproduce a run."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from cora_bench.instrumentation.hardware import host_info


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def package_versions(names: list[str]) -> dict:
    import importlib.metadata as md

    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except Exception:
            out[n] = None
    return out


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
        if not {"outputs", "tmp", "__pycache__"} & set(p.parts)
    }


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def build_manifest(
    config: dict, prompt_hashes: dict | None = None, packages: list[str] | None = None
) -> dict:
    return {
        "run_id": uuid.uuid4().hex,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha(),
        "source_hashes": source_hashes(),
        "config_fingerprint": fingerprint(config),
        "config": config,
        "python": sys.version.split()[0],
        "host": host_info(),
        "packages": package_versions(
            packages
            or ["tiktoken", "torch", "transformers", "llmlingua", "google-genai", "litellm"]
        ),
        "prompt_hashes": prompt_hashes or {},
    }
