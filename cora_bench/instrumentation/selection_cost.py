"""Selector cost estimates. Anything without a known price returns None, not 0."""


def selection_cost(
    method,
    metadata,
    pricing,
    input_tokens=0,
    output_tokens=0,
    gpu_seconds=0,
    gpu_type=None,
    reported_cost=None,
):
    if reported_cost is not None:
        return reported_cost
    md = metadata or {}
    if md.get("device") == "cpu":
        return None
    if md.get("cost_status") == "unknown_api_price":
        return None
    if method in {"head", "tail", "head_tail", "full_context", "bm25"}:
        return None  # CPU compute has not been priced.
    model = md.get("filter_model")
    try:
        if model:
            # Tiered pricing is per call, so it can't be applied to aggregated token counts.
            if pricing.models.get(model, {}).get("long_context_threshold"):
                return None
            return pricing.api_cost(model, input_tokens, output_tokens)
        if gpu_type and gpu_type in pricing.gpu_hourly:
            return pricing.gpu_cost(gpu_type, gpu_seconds)
    except KeyError:
        pass
    return None


def round_cost(value):
    return round(value, 8) if value is not None else None


def sum_cost(a, b):
    return a + b if a is not None and b is not None else None
