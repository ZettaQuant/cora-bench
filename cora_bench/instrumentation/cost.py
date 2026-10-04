"""Cost accounting. API cost from versioned per-token pricing; GPU cost from GPU-seconds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PRICING_DIR = Path(__file__).resolve().parent.parent / "config" / "pricing"


@dataclass
class Pricing:
    version: str
    models: dict
    gpu_hourly: dict

    @classmethod
    def load(cls, version: str) -> "Pricing":
        data = yaml.safe_load((PRICING_DIR / f"{version}.yaml").read_text())
        return cls(
            version=data["version"],
            models=data.get("models", {}),
            gpu_hourly=data.get("gpu_hourly", {}),
        )

    def api_cost(
        self, model: str, input_tokens: int, output_tokens: int, thought_tokens: int = 0
    ) -> float:
        m = self.models.get(model)
        if m is None:
            raise KeyError(f"no pricing for model {model!r} in {self.version}")
        thr = m.get("long_context_threshold")
        long_req = thr is not None and input_tokens > thr
        in_rate = m["input_per_1m_long"] if long_req else m["input_per_1m"]
        out_rate = m["output_per_1m_long"] if long_req else m["output_per_1m"]
        thinking_rate = m.get("thinking_per_1m", out_rate)
        return (
            input_tokens * in_rate + output_tokens * out_rate + thought_tokens * thinking_rate
        ) / 1e6

    def gpu_cost(self, gpu_type: str, gpu_seconds: float) -> float:
        rate = self.gpu_hourly.get(gpu_type)
        if rate is None:
            return 0.0
        return rate * gpu_seconds / 3600.0
