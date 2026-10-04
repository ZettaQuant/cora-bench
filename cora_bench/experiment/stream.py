"""Merge partial Vertex stream updates without losing final usage or finish metadata."""

import json
import time

from .api import API, MODEL, PROJECT, Vertex


class IncompleteStream(RuntimeError):
    def __init__(self, message, response, events):
        super().__init__(message)
        self.response = response
        self.events = events


def merge_events(events):
    response, candidate, parts = {}, {}, []
    latest_usage, latest_candidate = {}, {}
    for event in events:
        for key, value in event.items():
            if key == "candidates":
                continue
            if key in ("usageMetadata", "promptFeedback"):
                response.setdefault(key, {}).update(value)
                if key == "usageMetadata":
                    latest_usage = value
            else:
                response[key] = value
        if event.get("candidates"):
            if len(event["candidates"]) != 1:
                raise IncompleteStream("Unexpected multiple stream candidates", response, events)
            latest_candidate = event["candidates"][0]
            candidate.update({k: v for k, v in latest_candidate.items() if k != "content"})
            parts.extend(
                p["text"]
                for p in latest_candidate.get("content", {}).get("parts", [])
                if p.get("text") and not p.get("thought")
            )
    candidate["content"] = {"role": "model", "parts": [{"text": "".join(parts)}]}
    response["candidates"] = [candidate]
    usage = response.get("usageMetadata", {})
    if not isinstance(usage.get("promptTokenCount"), int) or not isinstance(
        usage.get("totalTokenCount"), int
    ):
        raise IncompleteStream("Stream ended without complete native token usage", response, events)
    if parts and not isinstance(usage.get("candidatesTokenCount"), int):
        raise IncompleteStream("Nonempty stream ended without output token usage", response, events)
    if not candidate.get("finishReason"):
        raise IncompleteStream("Stream ended without a final finish reason", response, events)
    recovered = (
        "promptTokenCount" not in latest_usage
        or "totalTokenCount" not in latest_usage
        or "finishReason" not in latest_candidate
    )
    return response, recovered


class AuditedVertex(Vertex):
    def stream(self, body):
        path = f"projects/{PROJECT}/locations/global/publishers/google/models/{MODEL}:streamGenerateContent?alt=sse"
        start = time.perf_counter()
        first = None
        events = []
        with self.session().post(
            API + "/" + path, json=body, stream=True, timeout=(30, 300)
        ) as response:
            if response.status_code >= 400:
                raise RuntimeError(f"Vertex HTTP {response.status_code}: {response.text[:1200]}")
            for raw in response.iter_lines(chunk_size=1):
                if not raw.startswith(b"data:"):
                    continue
                raw = raw[5:].lstrip()
                if raw == b"[DONE]":
                    continue
                event = json.loads(raw)
                events.append(event)
                if first is None and any(
                    p.get("text") and not p.get("thought")
                    for c in event.get("candidates", [])
                    for p in c.get("content", {}).get("parts", [])
                ):
                    first = time.perf_counter() - start
        result, recovered = merge_events(events)
        return result, {
            "time_to_first_token_seconds": first,
            "reader_latency_seconds": time.perf_counter() - start,
            "stream_events": len(events),
            "stream_terminal_complete": True,
            "partial_terminal_metadata_merged": recovered,
        }
