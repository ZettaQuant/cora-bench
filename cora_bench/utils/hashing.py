"""Stable hashing for cache keys and example references."""

from __future__ import annotations

import hashlib
import json


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cache_key(**parts) -> str:
    """Deterministic key over all params that can change a result."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()
